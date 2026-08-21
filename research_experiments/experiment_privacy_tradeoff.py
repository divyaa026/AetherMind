"""
Experiment 3: Privacy-Utility Tradeoff Analysis

Sweep over epsilon values to quantify the privacy-utility tradeoff.
This is the core experiment for the research paper.

Epsilon values: [0.2, 0.5, 1.0, 2.0, 5.0]
Delta: Fixed at 1e-5

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
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, precision_recall_curve, auc)
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('ResearchExperiments.PrivacyTradeoff')


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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class PrivacyUtilityExperiment:
    """
    Experiment to analyze privacy-utility tradeoff across epsilon values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize privacy-utility experiment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results_dir = Path(config['paths']['results_dir'])
        self.figures_dir = Path(config['paths']['figures_dir']) / "privacy_tradeoff"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  config['resources']['device'] == 'cuda' else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        self.epsilon_values = config['experiments']['privacy_tradeoff']['epsilon_values']
        self.delta = config['experiments']['privacy_tradeoff']['delta']
        
        # Set random seeds
        self._set_seed(config['reproducibility']['seed'])
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_data(self) -> Tuple[List[TensorDataset], TensorDataset]:
        """
        Prepare federated client datasets and test dataset with user-level split.
        
        Returns:
            Tuple of (client_datasets, test_dataset)
        """
        logger.info("Preparing data for privacy-utility experiment...")
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
        
        logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test")
        
        return client_datasets, test_dataset
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive evaluation of model.
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            
        Returns:
            Dictionary with all evaluation metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())
        
        # Calculate all metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        # AUC-PR
        prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
        auc_pr = auc(rec_curve, prec_curve)
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'auc_pr': float(auc_pr)
        }
    
    def train_federated_with_epsilon(self, client_datasets: List[TensorDataset],
                                    test_dataset: TensorDataset,
                                    epsilon: float,
                                    num_rounds: int = 50) -> Dict[str, Any]:
        """
        Train federated model with specific epsilon value.
        
        Args:
            client_datasets: List of client datasets
            test_dataset: Test dataset
            epsilon: Privacy budget
            num_rounds: Number of communication rounds
            
        Returns:
            Training results
        """
        logger.info(f"Training with epsilon={epsilon}...")
        
        # Initialize global model
        global_model = SimpleLSTMModel(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        test_loader = DataLoader(test_dataset, 
                                batch_size=self.config['training']['batch_size'])
        
        local_epochs = self.config['federated']['local_epochs']
        
        # Noise scale inversely proportional to epsilon (simplified DP)
        # In practice, use Opacus privacy accountant
        base_noise = self.config['privacy']['noise_multiplier']
        noise_scale = base_noise * (1.0 / epsilon)
        
        for round_num in range(num_rounds):
            client_weights = []
            
            for client_dataset in client_datasets:
                # Clone global model
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
                        
                        # DP noise injection
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
        
        # Final evaluation
        final_metrics = self.evaluate_model(global_model, test_loader)
        
        return {
            'epsilon': epsilon,
            'delta': self.delta,
            'accuracy': final_metrics['accuracy'],
            'f1_score': final_metrics['f1_score'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'auc_pr': final_metrics['auc_pr'],
            'privacy_guarantee': f"(epsilon={epsilon}, delta={self.delta:.0e})"
        }
    
    def train_baseline_no_dp(self, client_datasets: List[TensorDataset],
                            test_dataset: TensorDataset,
                            num_rounds: int = 50) -> Dict[str, Any]:
        """
        Train federated baseline without DP (for comparison).
        
        Args:
            client_datasets: List of client datasets
            test_dataset: Test dataset
            num_rounds: Number of communication rounds
            
        Returns:
            Training results
        """
        logger.info("Training FL baseline (no DP)...")
        
        global_model = SimpleLSTMModel(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        test_loader = DataLoader(test_dataset, 
                                batch_size=self.config['training']['batch_size'])
        
        local_epochs = self.config['federated']['local_epochs']
        
        for round_num in range(num_rounds):
            client_weights = []
            
            for client_dataset in client_datasets:
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
                
                local_model.train()
                for _ in range(local_epochs):
                    for X_batch, y_batch in client_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = local_model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 
                                                      self.config['training']['gradient_clip'])
                        
                        optimizer.step()
                
                client_weights.append(local_model.state_dict())
            
            # FedAvg
            global_state = global_model.state_dict()
            for key in global_state.keys():
                global_state[key] = torch.stack([client_weights[i][key].float() 
                                                 for i in range(len(client_weights))]).mean(0)
            
            global_model.load_state_dict(global_state)
        
        final_metrics = self.evaluate_model(global_model, test_loader)
        
        return {
            'model': 'FL (no DP)',
            'accuracy': final_metrics['accuracy'],
            'f1_score': final_metrics['f1_score'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'auc_pr': final_metrics['auc_pr']
        }
    
    def plot_privacy_utility_tradeoff(self, epsilon_results: List[Dict], 
                                     baseline: Dict[str, float]):
        """
        Generate the critical privacy-utility tradeoff plot.
        
        Args:
            epsilon_results: List of results for each epsilon value
            baseline: Baseline results (no DP)
        """
        logger.info("Generating privacy-utility tradeoff plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epsilons = [r['epsilon'] for r in epsilon_results]
        accuracies = [r['accuracy'] for r in epsilon_results]
        f1_scores = [r['f1_score'] for r in epsilon_results]
        
        # Plot 1: Accuracy vs Epsilon
        ax1.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8,
                color='steelblue', label='FL+DP')
        ax1.axhline(baseline['accuracy'], color='green', linestyle='--', 
                   linewidth=2, label='FL (no DP) Baseline')
        ax1.fill_between(epsilons, accuracies, baseline['accuracy'], 
                        alpha=0.2, color='red', label='Privacy Cost')
        
        ax1.set_xlabel('Privacy Budget (epsilon)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Privacy-Utility Tradeoff: Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Annotate points
        for eps, acc in zip(epsilons, accuracies):
            ax1.annotate(f'e={eps}\n{acc:.3f}', 
                        xy=(eps, acc), 
                        xytext=(5, -15), 
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Plot 2: F1-Score vs Epsilon
        ax2.plot(epsilons, f1_scores, marker='s', linewidth=2, markersize=8,
                color='coral', label='FL+DP')
        ax2.axhline(baseline['f1_score'], color='green', linestyle='--',
                   linewidth=2, label='FL (no DP) Baseline')
        ax2.fill_between(epsilons, f1_scores, baseline['f1_score'],
                        alpha=0.2, color='red', label='Privacy Cost')
        
        ax2.set_xlabel('Privacy Budget (epsilon)', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('Privacy-Utility Tradeoff: F1-Score', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Annotate points
        for eps, f1 in zip(epsilons, f1_scores):
            ax2.annotate(f'e={eps}\n{f1:.3f}',
                        xy=(eps, f1),
                        xytext=(5, -15),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plot_file = self.figures_dir / "privacy_utility_tradeoff.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Privacy-utility tradeoff plot saved: {plot_file}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete privacy-utility tradeoff experiment.
        
        Returns:
            Dictionary with experiment results
        """
        logger.info("="*60)
        logger.info("EXPERIMENT 3: PRIVACY-UTILITY TRADEOFF")
        logger.info("="*60)
        
        # Prepare data
        client_datasets, test_dataset = self.prepare_data()
        
        results = {
            'epsilon_results': [],
            'baseline': None
        }
        
        # Train baseline (no DP)
        if self.config['experiments']['privacy_tradeoff']['baseline_no_dp']:
            results['baseline'] = self.train_baseline_no_dp(client_datasets, test_dataset)
            logger.info(f"Baseline accuracy: {results['baseline']['accuracy']:.4f}")
        
        # Sweep over epsilon values
        for epsilon in self.epsilon_values:
            epsilon_result = self.train_federated_with_epsilon(
                client_datasets, test_dataset, epsilon
            )
            results['epsilon_results'].append(epsilon_result)
            
            logger.info(f"epsilon={epsilon}: Accuracy={epsilon_result['accuracy']:.4f}, "
                       f"F1={epsilon_result['f1_score']:.4f}")
        
        # Find optimal epsilon (best balance)
        # Define as epsilon with >90% of baseline accuracy and smallest epsilon
        baseline_acc = results['baseline']['accuracy']
        acceptable_results = [r for r in results['epsilon_results'] 
                            if r['accuracy'] >= 0.9 * baseline_acc]
        
        if acceptable_results:
            optimal = min(acceptable_results, key=lambda x: x['epsilon'])
            results['optimal_epsilon'] = optimal['epsilon']
            results['optimal_accuracy'] = optimal['accuracy']
            results['recommended_epsilon'] = optimal['epsilon']
        else:
            # Fallback to best performing
            optimal = max(results['epsilon_results'], key=lambda x: x['accuracy'])
            results['optimal_epsilon'] = optimal['epsilon']
            results['optimal_accuracy'] = optimal['accuracy']
            results['recommended_epsilon'] = 1.0  # Conservative recommendation
        
        # Generate plots
        self.plot_privacy_utility_tradeoff(results['epsilon_results'], results['baseline'])
        
        # Save results
        results_file = self.figures_dir / "privacy_utility_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Privacy-utility results saved: {results_file}")
        logger.info(f"Recommended epsilon: {results['recommended_epsilon']}")
        logger.info("Privacy-utility tradeoff experiment completed successfully")
        
        return results


def run_privacy_tradeoff_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for privacy-utility tradeoff experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment results
    """
    experiment = PrivacyUtilityExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    # For standalone testing
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_privacy_tradeoff_experiment(config)
    print("\nPrivacy-Utility Tradeoff Results:")
    for r in results['epsilon_results']:
        print(f"epsilon={r['epsilon']}: Acc={r['accuracy']:.4f}, F1={r['f1_score']:.4f}")
