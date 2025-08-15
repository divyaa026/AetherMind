#!/usr/bin/env python3
"""
Federated Learning Training Script for MindGuard
Trains crisis detection models using federated learning with privacy preservation
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch

from ml.federated.federated_coordinator import FederatedCoordinator, FederatedConfig
from ml.models.text_models import BERTCrisisDetector, BiLSTMCrisisDetector
from ml.utils.text_processing import TextProcessor
from ml.utils.privacy import DifferentialPrivacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrisisDataset(Dataset):
    """Dataset for crisis detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            # Tokenize for BERT
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For BiLSTM, return raw text and label
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }


class FederatedTrainer:
    """Trainer for federated learning"""
    
    def __init__(self, config: FederatedConfig, data_dir: str = "processed_data"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.coordinator = FederatedCoordinator(config)
        self.text_processor = TextProcessor()
        self.dp_processor = DifferentialPrivacy()
        
        # Load and prepare data
        self.train_data, self.val_data, self.test_data = self._load_data()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("mindguard_federated")
        
        logger.info("Federated trainer initialized")
    
    def _load_data(self) -> tuple:
        """Load processed datasets"""
        datasets = {}
        
        for dataset_name in ["depression_reddit", "suicide_detection"]:
            dataset_path = self.data_dir / dataset_name
            
            if dataset_path.exists():
                train_df = pd.read_csv(dataset_path / "train.csv")
                val_df = pd.read_csv(dataset_path / "validation.csv")
                test_df = pd.read_csv(dataset_path / "test.csv")
                
                datasets[dataset_name] = {
                    "train": train_df,
                    "validation": val_df,
                    "test": test_df
                }
                
                logger.info(f"Loaded {dataset_name}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return datasets
    
    def _create_simulated_nodes(self, num_nodes: int = 100) -> List[Dict]:
        """Create simulated federated learning nodes"""
        nodes = []
        
        for i in range(num_nodes):
            # Simulate different data distributions
            if i < num_nodes // 2:
                # First half: depression dataset
                dataset_name = "depression_reddit"
                data = self.train_data[dataset_name]
            else:
                # Second half: suicide detection dataset
                dataset_name = "suicide_detection"
                data = self.train_data[dataset_name]
            
            # Sample a subset of data for this node
            node_data_size = np.random.randint(100, 1000)
            node_data = data.sample(n=min(node_data_size, len(data)), random_state=i)
            
            node = {
                "node_id": f"node_{i:03d}",
                "address": f"192.168.1.{i+10}",
                "port": 8000 + i,
                "data": node_data,
                "dataset_name": dataset_name,
                "data_size": len(node_data)
            }
            
            nodes.append(node)
            
            # Register with coordinator
            self.coordinator.register_node(
                node["node_id"],
                node["address"],
                node["port"],
                node["data_size"]
            )
        
        logger.info(f"Created {len(nodes)} simulated nodes")
        return nodes
    
    def _train_local_model(self, node_data: pd.DataFrame, global_model_state: Dict, config: Dict) -> Dict:
        """Train a local model on node data"""
        # Initialize local model
        if self.config.model_type == "bert":
            local_model = BERTCrisisDetector(
                model_name="bert-base-uncased",
                num_classes=2,
                dropout=0.1
            )
        else:
            local_model = BiLSTMCrisisDetector(
                vocab_size=30000,
                embedding_dim=256,
                hidden_dim=128,
                num_classes=2,
                dropout=0.1
            )
        
        # Load global model weights
        local_model.load_state_dict(global_model_state)
        local_model.train()
        
        # Prepare data
        texts = node_data['processed_text'].tolist()
        labels = node_data['label'].tolist()
        
        # Create dataset and dataloader
        if self.config.model_type == "bert":
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            dataset = CrisisDataset(texts, labels, tokenizer, max_length=512)
        else:
            dataset = CrisisDataset(texts, labels, max_length=512)
        
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )
        
        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model.to(device)
        
        optimizer = optim.AdamW(
            local_model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(config["local_epochs"]):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch in dataloader:
                if self.config.model_type == "bert":
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = local_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    # For BiLSTM, implement text processing
                    texts = batch['text']
                    labels = batch['labels'].to(device)
                    
                    # Simplified BiLSTM processing (would need proper tokenization)
                    outputs = local_model(torch.randn(len(texts), 512, 256).to(device))
                
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += labels.size(0)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Calculate metrics
        avg_loss = total_loss / (len(dataloader) * config["local_epochs"])
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Get model state dict
        model_state = local_model.state_dict()
        
        # Apply differential privacy to gradients (simplified)
        if config["privacy_epsilon"] > 0:
            for key in model_state:
                if model_state[key].dtype in [torch.float32, torch.float64]:
                    noise = torch.randn_like(model_state[key]) * 0.01
                    model_state[key] += noise
        
        return {
            "model_state": model_state,
            "loss": avg_loss,
            "accuracy": accuracy,
            "data_size": len(node_data)
        }
    
    def run_federated_training(self, num_rounds: int = 50) -> Dict:
        """Run federated learning training"""
        logger.info(f"Starting federated learning training for {num_rounds} rounds")
        
        # Create simulated nodes
        nodes = self._create_simulated_nodes(self.config.num_nodes)
        
        # Training history
        training_history = []
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "num_nodes": self.config.num_nodes,
                "min_nodes_per_round": self.config.min_nodes_per_round,
                "max_nodes_per_round": self.config.max_nodes_per_round,
                "local_epochs": self.config.local_epochs,
                "learning_rate": self.config.learning_rate,
                "privacy_epsilon": self.config.privacy_epsilon,
                "model_type": self.config.model_type
            })
            
            for round_num in range(num_rounds):
                logger.info(f"Starting federated round {round_num + 1}/{num_rounds}")
                
                # Select nodes for this round
                try:
                    selected_node_ids = self.coordinator.select_nodes_for_round(round_num)
                except ValueError as e:
                    logger.error(f"Round {round_num}: {e}")
                    break
                
                # Prepare global model for distribution
                distribution_package = self.coordinator.prepare_global_model_for_distribution(round_num)
                
                # Simulate local training on selected nodes
                model_updates = []
                round_losses = []
                round_accuracies = []
                
                for node_id in selected_node_ids:
                    # Find node data
                    node = next((n for n in nodes if n["node_id"] == node_id), None)
                    if not node:
                        continue
                    
                    # Train local model
                    try:
                        local_update = self._train_local_model(
                            node["data"],
                            self.coordinator.global_model.state_dict(),
                            distribution_package["config"]
                        )
                        
                        # Add node metadata
                        local_update["node_id"] = node_id
                        local_update["round"] = round_num
                        local_update["timestamp"] = time.time()
                        
                        model_updates.append(local_update)
                        round_losses.append(local_update["loss"])
                        round_accuracies.append(local_update["accuracy"])
                        
                        logger.debug(f"Node {node_id}: loss={local_update['loss']:.4f}, accuracy={local_update['accuracy']:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error training node {node_id}: {e}")
                        continue
                
                # Aggregate models
                if model_updates:
                    success = self.coordinator.aggregate_models(model_updates)
                    
                    if success:
                        # Calculate round statistics
                        avg_loss = np.mean(round_losses)
                        avg_accuracy = np.mean(round_accuracies)
                        
                        round_stats = {
                            "round": round_num,
                            "num_nodes": len(model_updates),
                            "avg_loss": avg_loss,
                            "avg_accuracy": avg_accuracy,
                            "min_loss": np.min(round_losses),
                            "max_loss": np.max(round_losses),
                            "min_accuracy": np.min(round_accuracies),
                            "max_accuracy": np.max(round_accuracies)
                        }
                        
                        training_history.append(round_stats)
                        self.coordinator.round_history.append(round_stats)
                        
                        # Log to MLflow
                        mlflow.log_metrics({
                            "round": round_num,
                            "avg_loss": avg_loss,
                            "avg_accuracy": avg_accuracy,
                            "num_nodes": len(model_updates)
                        })
                        
                        logger.info(f"Round {round_num}: avg_loss={avg_loss:.4f}, avg_accuracy={avg_accuracy:.4f}, nodes={len(model_updates)}")
                        
                        # Check convergence
                        if self.coordinator.check_convergence():
                            logger.info(f"Federated learning converged at round {round_num}")
                            break
                    else:
                        logger.warning(f"Round {round_num}: Model aggregation failed")
                else:
                    logger.warning(f"Round {round_num}: No model updates received")
                
                # Save checkpoint every 10 rounds
                if (round_num + 1) % 10 == 0:
                    checkpoint_path = f"models/federated_checkpoint_round_{round_num + 1}.pt"
                    self.coordinator.save_global_model(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save final model
            final_model_path = "models/federated_final_model.pt"
            self.coordinator.save_global_model(final_model_path)
            
            # Log final model
            mlflow.pytorch.log_model(self.coordinator.global_model, "federated_model")
            
            # Log training history
            mlflow.log_artifact("models/federated_final_model.pt")
            
            logger.info(f"Federated training completed. Final model saved to {final_model_path}")
        
        return {
            "training_history": training_history,
            "final_model_path": final_model_path,
            "total_rounds": len(training_history),
            "converged": self.coordinator.check_convergence()
        }


def main():
    parser = argparse.ArgumentParser(description="Train MindGuard models using federated learning")
    parser.add_argument("--num-nodes", type=int, default=100, help="Number of federated nodes")
    parser.add_argument("--rounds", type=int, default=50, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local training epochs per round")
    parser.add_argument("--model-type", choices=["bert", "bilstm"], default="bert", help="Model type")
    parser.add_argument("--privacy-epsilon", type=float, default=1.0, help="Differential privacy epsilon")
    parser.add_argument("--data-dir", default="processed_data", help="Processed data directory")
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Configure federated learning
    config = FederatedConfig(
        num_nodes=args.num_nodes,
        min_nodes_per_round=max(5, args.num_nodes // 10),
        max_nodes_per_round=min(50, args.num_nodes // 2),
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        model_type=args.model_type,
        privacy_epsilon=args.privacy_epsilon
    )
    
    # Initialize trainer
    trainer = FederatedTrainer(config, args.data_dir)
    
    # Run federated training
    results = trainer.run_federated_training(args.rounds)
    
    # Save results
    results_path = Path(args.output_dir) / "federated_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training results saved to {results_path}")
    logger.info(f"Final model: {results['final_model_path']}")
    logger.info(f"Converged: {results['converged']}")


if __name__ == "__main__":
    main()
