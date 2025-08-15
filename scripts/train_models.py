#!/usr/bin/env python3
"""
MindGuard Model Training Script

This script trains all crisis detection models from scratch using the processed datasets.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import mlflow
import mlflow.pytorch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.models.text_models import BERTCrisisDetector, BiLSTMCrisisDetector, ModelConfig
from ml.utils.text_processing import TextProcessor
from ml.utils.privacy import PrivacyProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrisisDataset(Dataset):
    """Dataset for crisis detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
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


class ModelTrainer:
    """Trainer for crisis detection models"""
    
    def __init__(self, config: ModelConfig, output_dir: str = "trained_models"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.privacy_processor = PrivacyProcessor()
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("mindguard-crisis-detection")
    
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load processed data"""
        logger.info(f"Loading data from {data_path}")
        
        data = {}
        for split in ['train', 'validation', 'test']:
            split_path = Path(data_path) / f"{split}.csv"
            if split_path.exists():
                data[split] = pd.read_csv(split_path)
                logger.info(f"Loaded {len(data[split])} samples for {split} split")
            else:
                logger.warning(f"Split file not found: {split_path}")
        
        return data
    
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        texts = data['processed_text'].tolist()
        labels = data['label'].tolist()
        
        # Clean and preprocess texts
        cleaned_texts = []
        for text in texts:
            if pd.isna(text):
                cleaned_texts.append("")
            else:
                cleaned_text = self.text_processor.preprocess_text(str(text))
                cleaned_texts.append(cleaned_text)
        
        return cleaned_texts, labels
    
    def create_data_loaders(self, train_texts: List[str], train_labels: List[int],
                           val_texts: List[str], val_labels: List[int],
                           tokenizer) -> tuple:
        """Create data loaders for training"""
        
        # Create datasets
        train_dataset = CrisisDataset(train_texts, train_labels, tokenizer, self.config.max_length)
        val_dataset = CrisisDataset(val_texts, val_labels, tokenizer, self.config.max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_bert_model(self, train_loader: DataLoader, val_loader: DataLoader,
                        tokenizer) -> BERTCrisisDetector:
        """Train BERT model"""
        logger.info("Training BERT model...")
        
        # Initialize model
        model = BERTCrisisDetector(self.config)
        model.to(self.config.device)
        
        # Setup training
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        
        # Training loop
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.config.device)
                    attention_mask = batch['attention_mask'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model)
        
        return model
    
    def train_bilstm_model(self, train_loader: DataLoader, val_loader: DataLoader,
                          vocab_size: int) -> BiLSTMCrisisDetector:
        """Train BiLSTM model"""
        logger.info("Training BiLSTM model...")
        
        # Initialize model
        model = BiLSTMCrisisDetector(self.config, vocab_size=vocab_size)
        model.to(self.config.device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        
        # Training loop
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.config.device)
                    attention_mask = batch['attention_mask'].to(self.config.device)
                    labels = batch['labels'].to(self.config.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model)
        
        return model
    
    def evaluate_model(self, model, test_loader: DataLoader, model_name: str) -> Dict[str, Any]:
        """Evaluate model on test set"""
        logger.info(f"Evaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(all_labels, all_predictions, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Log results
        logger.info(f"{model_name} Results:")
        logger.info(f"Accuracy: {report['accuracy']:.4f}")
        logger.info(f"Precision: {report['weighted avg']['precision']:.4f}")
        logger.info(f"Recall: {report['weighted avg']['recall']:.4f}")
        logger.info(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return {
            "accuracy": report['accuracy'],
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score'],
            "confusion_matrix": conf_matrix.tolist(),
            "detailed_report": report
        }
    
    def save_model(self, model, model_name: str, metrics: Dict[str, Any]):
        """Save trained model and metrics"""
        # Create model directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        if isinstance(model, BERTCrisisDetector):
            torch.save(model.state_dict(), model_dir / "model.pth")
        else:
            torch.save(model.state_dict(), model_dir / "model.pth")
        
        # Save metrics
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model config
        with open(model_dir / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    def log_to_mlflow(self, model, model_name: str, metrics: Dict[str, Any]):
        """Log model and metrics to MLflow"""
        with mlflow.start_run(run_name=f"{model_name}-training"):
            # Log parameters
            mlflow.log_params(self.config.__dict__)
            
            # Log metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            
            # Log model
            mlflow.pytorch.log_model(model, model_name)
            
            # Log confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
            plt.title(f"{model_name} Confusion Matrix")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), f"{model_name}_confusion_matrix.png")
            plt.close()
    
    def train_all_models(self, data_path: str):
        """Train all models"""
        logger.info("Starting model training...")
        
        # Load data
        data = self.load_data(data_path)
        
        if not data:
            logger.error("No data found for training")
            return
        
        # Prepare data
        train_texts, train_labels = self.prepare_data(data['train'])
        val_texts, val_labels = self.prepare_data(data['validation'])
        test_texts, test_labels = self.prepare_data(data['test'])
        
        # Initialize tokenizer
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels, tokenizer
        )
        test_loader, _ = self.create_data_loaders(
            test_texts, test_labels, val_texts, val_labels, tokenizer
        )
        
        # Train BERT model
        bert_model = self.train_bert_model(train_loader, val_loader, tokenizer)
        bert_metrics = self.evaluate_model(bert_model, test_loader, "BERT")
        self.save_model(bert_model, "bert_crisis_detector", bert_metrics)
        self.log_to_mlflow(bert_model, "bert_crisis_detector", bert_metrics)
        
        # Train BiLSTM model
        bilstm_model = self.train_bilstm_model(train_loader, val_loader, vocab_size=30000)
        bilstm_metrics = self.evaluate_model(bilstm_model, test_loader, "BiLSTM")
        self.save_model(bilstm_model, "bilstm_crisis_detector", bilstm_metrics)
        self.log_to_mlflow(bilstm_model, "bilstm_crisis_detector", bilstm_metrics)
        
        # Generate training report
        self.generate_training_report([bert_metrics, bilstm_metrics])
        
        logger.info("Model training completed successfully!")
    
    def generate_training_report(self, all_metrics: List[Dict[str, Any]]):
        """Generate comprehensive training report"""
        report = {
            "training_info": {
                "timestamp": datetime.now().isoformat(),
                "models_trained": len(all_metrics),
                "config": self.config.__dict__
            },
            "model_performance": {}
        }
        
        model_names = ["BERT", "BiLSTM"]
        for i, metrics in enumerate(all_metrics):
            report["model_performance"][model_names[i]] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MindGuard crisis detection models")
    parser.add_argument("--data-path", type=str, default="processed_data",
                       help="Path to processed data directory")
    parser.add_argument("--output-dir", type=str, default="trained_models",
                       help="Output directory for trained models")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create model config
    config = ModelConfig(
        model_name="crisis_detection",
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config, args.output_dir)
    
    # Train models
    trainer.train_all_models(args.data_path)


if __name__ == "__main__":
    main()
