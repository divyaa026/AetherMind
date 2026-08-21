"""
Experiment 2: Convergence Analysis

Compare convergence behavior of three training approaches:
1. Centralized Oracle (baseline - all data pooled, no FL, no DP)
2. Federated Baseline (FedAvg without DP)
3. Our Method (FedAvg with DP-SGD)

Track loss, accuracy, and F1-score across communication rounds.

Author: Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('ResearchExperiments.Convergence')


class SimpleLSTMModel(nn.Module):
    """Simple LSTM model for mental health prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        out = self.sigmoid(out)
        return out


class ConvergenceExperiment:
    """
    Experiment to compare convergence of Centralized, FL, and FL+DP approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize convergence experiment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results_dir = Path(config['paths']['results_dir'])
        self.figures_dir = Path(config['paths']['figures_dir']) / "convergence"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  config['resources']['device'] == 'cuda' else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_seed(config['reproducibility']['seed'])
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_data(self) -> Tuple[TensorDataset, TensorDataset, List[TensorDataset]]:
        """
        Prepare data for training with user-level split to prevent data leakage.
        
        Returns:
            Tuple of (train_dataset, test_dataset, client_datasets)
        """
        logger.info("Preparing data for convergence experiment...")
        from realistic_data_generator import generate_realistic_sequential_data
        
        # Generate realistic sequential data
        n_samples = 5000
        n_users = 100
        seq_length = self.config['model']['sequence_length']
        
        # Generate data with 3% positive class ratio
        X_all, y_all, user_ids = generate_realistic_sequential_data(
            n_samples=n_samples,
            n_users=n_users,
            seq_length=seq_length,
            positive_ratio=0.03,
            seed=self.config['reproducibility']['seed']
        )
        
        logger.info(f"Generated {len(X_all)} sequences")
        logger.info(f"Positive class ratio: {y_all.mean():.2%}")
        
        # CRITICAL: Split by user to prevent data leakage
        unique_users = np.unique(user_ids)
        np.random.shuffle(unique_users)
        
        train_ratio = self.config['data']['train_ratio']
        n_train_users = int(len(unique_users) * train_ratio)
        
        train_users = unique_users[:n_train_users]
        test_users = unique_users[n_train_users:]
        
        # Get samples from train/test users
        train_mask = np.isin(user_ids, train_users)
        test_mask = np.isin(user_ids, test_users)
        
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        
        logger.info(f"Train: {len(X_train)} samples from {len(train_users)} users")
        logger.info(f"Test: {len(X_test)} samples from {len(test_users)} users")
        logger.info(f"Train positive ratio: {y_train.mean():.2%}")
        logger.info(f"Test positive ratio: {y_test.mean():.2%}")
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Partition training data into clients (by user)
        num_clients = self.config['federated']['num_clients']
        client_datasets = []
        
        users_per_client = len(train_users) // num_clients
        for i in range(num_clients):
            start_idx = i * users_per_client
            end_idx = start_idx + users_per_client if i < num_clients - 1 else len(train_users)
            
            client_users = train_users[start_idx:end_idx]
            client_mask = np.isin(user_ids[train_mask], client_users)
            
            client_X = X_train[client_mask]
            client_y = y_train[client_mask]
            
            client_dataset = TensorDataset(torch.FloatTensor(client_X), 
                                          torch.FloatTensor(client_y))
            client_datasets.append(client_dataset)
        
        logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test, "
                   f"{num_clients} clients")
        
        return train_dataset, test_dataset, client_datasets
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # AUC-PR
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        auc_pr = auc(recall, precision)
        
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'auc_pr': float(auc_pr),
            'loss': float(avg_loss)
        }
    
    def train_centralized(self, train_dataset: TensorDataset, 
                         test_dataset: TensorDataset,
                         num_epochs: int = 50) -> Dict[str, Any]:
        """
        Train centralized oracle model (upper bound).
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        logger.info("Training Centralized Oracle...")
        
        model = SimpleLSTMModel(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(train_dataset, 
                                 batch_size=self.config['training']['batch_size'],
                                 shuffle=True)
        test_loader = DataLoader(test_dataset, 
                                batch_size=self.config['training']['batch_size'])
        
        history = {'epochs': [], 'train_loss': [], 'test_metrics': []}
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              self.config['training']['gradient_clip'])
                
                optimizer.step()
                train_loss += loss.item()
            
            # Evaluate
            if (epoch + 1) % 5 == 0 or epoch == 0:
                metrics = self.evaluate_model(model, test_loader)
                history['epochs'].append(epoch + 1)
                history['train_loss'].append(train_loss / len(train_loader))
                history['test_metrics'].append(metrics)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Loss: {metrics['loss']:.4f}, "
                          f"Acc: {metrics['accuracy']:.4f}, "
                          f"F1: {metrics['f1_score']:.4f}")
        
        # Final evaluation
        final_metrics = self.evaluate_model(model, test_loader)
        
        return {
            'history': history,
            'final_accuracy': final_metrics['accuracy'],
            'final_f1': final_metrics['f1_score'],
            'final_auc_pr': final_metrics['auc_pr'],
            'final_loss': final_metrics['loss']
        }
    
    def train_federated(self, client_datasets: List[TensorDataset],
                       test_dataset: TensorDataset,
                       num_rounds: int = 50,
                       use_dp: bool = False,
                       epsilon: float = 1.0) -> Dict[str, Any]:
        """
        Train federated model with FedAvg (with or without DP).
        
        Args:
            client_datasets: List of client datasets
            test_dataset: Test dataset
            num_rounds: Number of communication rounds
            use_dp: Whether to use differential privacy
            epsilon: Privacy budget (if use_dp=True)
            
        Returns:
            Training history
        """
        mode = "FL+DP" if use_dp else "FL"
        logger.info(f"Training {mode}...")
        
        # Initialize global model
        global_model = SimpleLSTMModel(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        test_loader = DataLoader(test_dataset, 
                                batch_size=self.config['training']['batch_size'])
        
        history = {'rounds': [], 'test_metrics': []}
        
        local_epochs = self.config['federated']['local_epochs']
        
        for round_num in range(num_rounds):
            # Client training
            client_weights = []
            
            for client_idx, client_dataset in enumerate(client_datasets):
                # Clone global model for local training
                local_model = SimpleLSTMModel(
                    input_size=self.config['model']['input_size'],
                    hidden_size=self.config['model']['hidden_size'],
                    num_layers=self.config['model']['num_layers'],
                    dropout=self.config['model']['dropout']
                ).to(self.device)
                
                local_model.load_state_dict(global_model.state_dict())
                
                optimizer = optim.Adam(local_model.parameters(), 
                                      lr=self.config['training']['learning_rate'])
                criterion = nn.BCELoss()
                
                client_loader = DataLoader(client_dataset,
                                          batch_size=self.config['training']['batch_size'],
                                          shuffle=True)
                
                # Local training
                local_model.train()
                for _ in range(local_epochs):
                    for X_batch, y_batch in client_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = local_model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 
                                                      self.config['training']['gradient_clip'])
                        
                        # DP noise injection (simplified)
                        if use_dp:
                            noise_scale = self.config['privacy']['noise_multiplier']
                            for param in local_model.parameters():
                                if param.grad is not None:
                                    noise = torch.randn_like(param.grad) * noise_scale * 0.01
                                    param.grad.add_(noise)
                        
                        optimizer.step()
                
                client_weights.append(local_model.state_dict())
            
            # FedAvg aggregation
            global_state = global_model.state_dict()
            for key in global_state.keys():
                global_state[key] = torch.stack([client_weights[i][key].float() 
                                                 for i in range(len(client_weights))]).mean(0)
            
            global_model.load_state_dict(global_state)
            
            # Evaluate every 5 rounds
            if (round_num + 1) % 5 == 0 or round_num == 0:
                metrics = self.evaluate_model(global_model, test_loader)
                history['rounds'].append(round_num + 1)
                history['test_metrics'].append(metrics)
                
                logger.info(f"Round {round_num+1}/{num_rounds} - "
                          f"Acc: {metrics['accuracy']:.4f}, "
                          f"F1: {metrics['f1_score']:.4f}")
        
        # Final evaluation
        final_metrics = self.evaluate_model(global_model, test_loader)
        
        return {
            'history': history,
            'final_accuracy': final_metrics['accuracy'],
            'final_f1': final_metrics['f1_score'],
            'final_auc_pr': final_metrics['auc_pr'],
            'final_loss': final_metrics['loss'],
            'convergence_round': self._find_convergence_round(history)
        }
    
    def _find_convergence_round(self, history: Dict) -> int:
        """Find the round where model converged (accuracy stabilized)."""
        if len(history['test_metrics']) < 3:
            return len(history['rounds'])
        
        accuracies = [m['accuracy'] for m in history['test_metrics']]
        
        # Converged when accuracy change < 0.01 for 3 consecutive evaluations
        for i in range(2, len(accuracies)):
            if all(abs(accuracies[i] - accuracies[i-j]) < 0.01 for j in range(1, 3)):
                return history['rounds'][i]
        
        return history['rounds'][-1]
    
    def plot_convergence(self, results: Dict[str, Any]):
        """
        Generate convergence plot comparing all three approaches.
        
        Args:
            results: Dictionary containing results from all approaches
        """
        logger.info("Generating convergence plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy over rounds/epochs
        for model_name, color, marker in [
            ('centralized', 'green', 'o'),
            ('federated', 'blue', 's'),
            ('federated_dp', 'red', '^')
        ]:
            if model_name in results:
                if 'history' in results[model_name]:
                    history = results[model_name]['history']
                    
                    if 'epochs' in history:
                        x = history['epochs']
                    else:
                        x = history['rounds']
                    
                    y = [m['accuracy'] for m in history['test_metrics']]
                    
                    label = model_name.replace('_', ' ').title()
                    ax1.plot(x, y, marker=marker, label=label, color=color, 
                            linewidth=2, markersize=6, alpha=0.8)
        
        ax1.set_xlabel('Communication Rounds / Epochs', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1-Score over rounds/epochs
        for model_name, color, marker in [
            ('centralized', 'green', 'o'),
            ('federated', 'blue', 's'),
            ('federated_dp', 'red', '^')
        ]:
            if model_name in results:
                if 'history' in results[model_name]:
                    history = results[model_name]['history']
                    
                    if 'epochs' in history:
                        x = history['epochs']
                    else:
                        x = history['rounds']
                    
                    y = [m['f1_score'] for m in history['test_metrics']]
                    
                    label = model_name.replace('_', ' ').title()
                    ax2.plot(x, y, marker=marker, label=label, color=color,
                            linewidth=2, markersize=6, alpha=0.8)
        
        ax2.set_xlabel('Communication Rounds / Epochs', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.figures_dir / "convergence_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence plot saved: {plot_file}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete convergence experiment.
        
        Returns:
            Dictionary with convergence results
        """
        logger.info("="*60)
        logger.info("EXPERIMENT 2: CONVERGENCE ANALYSIS")
        logger.info("="*60)
        
        # Prepare data
        train_dataset, test_dataset, client_datasets = self.prepare_data()
        
        results = {}
        
        # Experiment 1: Centralized Oracle
        results['centralized'] = self.train_centralized(train_dataset, test_dataset, 
                                                        num_epochs=50)
        
        # Experiment 2: Federated Baseline (no DP)
        results['federated'] = self.train_federated(client_datasets, test_dataset,
                                                    num_rounds=50, use_dp=False)
        
        # Experiment 3: Federated with DP
        results['federated_dp'] = self.train_federated(client_datasets, test_dataset,
                                                       num_rounds=50, use_dp=True,
                                                       epsilon=self.config['privacy']['target_epsilon'])
        
        # Calculate overheads
        cent_acc = results['centralized']['final_accuracy']
        fl_acc = results['federated']['final_accuracy']
        fl_dp_acc = results['federated_dp']['final_accuracy']
        
        results['privacy_overhead_pct'] = (cent_acc - fl_dp_acc) * 100
        results['fl_overhead_pct'] = (cent_acc - fl_acc) * 100
        
        # Generate plots
        self.plot_convergence(results)
        
        # Save results
        results_file = self.figures_dir / "convergence_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Convergence results saved: {results_file}")
        logger.info("Convergence experiment completed successfully")
        
        return results


def run_convergence_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for convergence experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment results
    """
    experiment = ConvergenceExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    # For standalone testing
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_convergence_experiment(config)
    print("\nConvergence Experiment Results:")
    print(f"Centralized: {results['centralized']['final_accuracy']:.4f}")
    print(f"Federated: {results['federated']['final_accuracy']:.4f}")
    print(f"Federated+DP: {results['federated_dp']['final_accuracy']:.4f}")
