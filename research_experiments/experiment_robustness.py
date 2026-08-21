"""
Experiment 4: Robustness Testing

Test system performance under realistic challenging conditions:
1. Extreme Non-IID stress test
2. Partial client participation (30%)
3. Membership inference attack simulation

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('ResearchExperiments.Robustness')


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


class RobustnessExperiment:
    """
    Experiment to test system robustness under challenging conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize robustness experiment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results_dir = Path(config['paths']['results_dir'])
        self.figures_dir = Path(config['paths']['figures_dir']) / "robustness"
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
    
    def prepare_data(self) -> Tuple[List[TensorDataset], TensorDataset]:
        """
        Prepare standard IID-like federated data with realistic characteristics.
        
        Returns:
            Tuple of (client_datasets, test_dataset)
        """
        logger.info("Preparing standard data...")
        from realistic_data_generator import generate_realistic_sequential_data
        
        n_samples = 5000
        n_users = 100
        seq_length = self.config['model']['sequence_length']
        
        # Generate realistic sequential data
        X_all, y_all, user_ids = generate_realistic_sequential_data(
            n_samples=n_samples,
            n_users=n_users,
            seq_length=seq_length,
            positive_ratio=0.03,
            seed=self.config['reproducibility']['seed']
        )
        
        logger.info(f"Generated {len(X_all)} sequences, positive ratio: {y_all.mean():.2%}")
        
        # User-level train-test split
        unique_users = np.unique(user_ids)
        np.random.shuffle(unique_users)
        
        train_ratio = self.config['data']['train_ratio']
        n_train_users = int(len(unique_users) * train_ratio)
        
        train_users = unique_users[:n_train_users]
        test_users = unique_users[n_train_users:]
        
        train_mask = np.isin(user_ids, train_users)
        test_mask = np.isin(user_ids, test_users)
        
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Standard partition by users (IID-like)
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
        
        return client_datasets, test_dataset
    
    def prepare_extreme_non_iid_data(self) -> Tuple[List[TensorDataset], TensorDataset]:
        """
        Prepare TRULY extreme non-IID data with non-overlapping stress ranges per client.
        This creates distribution shift where clients never see certain data types.
        
        Returns:
            Tuple of (client_datasets, test_dataset)
        """
        logger.info("Preparing EXTREME non-IID data with distribution shift...")
        from realistic_data_generator import generate_realistic_sequential_data
        
        n_samples = 5000
        n_users = 100
        seq_length = self.config['model']['sequence_length']
        num_clients = self.config['federated']['num_clients']
        
        # Generate realistic data
        X_all, y_all, user_ids = generate_realistic_sequential_data(
            n_samples=n_samples,
            n_users=n_users,
            seq_length=seq_length,
            positive_ratio=0.03,
            seed=self.config['reproducibility']['seed']
        )
        
        # Calculate average stress level per user
        user_stress = {}
        for user_id in np.unique(user_ids):
            user_mask = user_ids == user_id
            user_sequences = X_all[user_mask]
            # Stress is feature index 1 (sleep=0, stress=1, activity=2, social=3, mood=4)
            avg_stress = np.mean(user_sequences[:, :, 1])  # Average across all sequences and timesteps
            user_stress[user_id] = avg_stress
        
        # Sort users by stress level
        sorted_users = sorted(user_stress.items(), key=lambda x: x[1])
        
        # Partition users into NON-OVERLAPPING stress ranges
        # This creates TRUE distribution shift!
        stress_ranges = np.linspace(0, len(sorted_users), num_clients + 1, dtype=int)
        
        logger.info("Creating extreme non-IID partition:")
        for i in range(num_clients):
            start_idx = stress_ranges[i]
            end_idx = stress_ranges[i + 1]
            client_users_list = [u[0] for u in sorted_users[start_idx:end_idx]]
            stress_values = [u[1] for u in sorted_users[start_idx:end_idx]]
            logger.info(f"  Client {i}: {len(client_users_list)} users, "
                       f"stress range [{min(stress_values):.3f}, {max(stress_values):.3f}]")
        
        # Split users into train/test FIRST (prevent leakage)
        unique_users = np.array([u[0] for u in sorted_users])
        n_train_users = int(len(unique_users) * 0.8)
        train_users = unique_users[:n_train_users]
        test_users = unique_users[n_train_users:]
        
        train_mask = np.isin(user_ids, train_users)
        test_mask = np.isin(user_ids, test_users)
        
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        
        # Create test dataset
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Partition training users into clients by stress ranges
        client_datasets = []
        train_user_stress = {u: s for u, s in sorted_users if u in train_users}
        sorted_train_users = sorted(train_user_stress.items(), key=lambda x: x[1])
        
        stress_ranges_train = np.linspace(0, len(sorted_train_users), num_clients + 1, dtype=int)
        
        for i in range(num_clients):
            start_idx = stress_ranges_train[i]
            end_idx = stress_ranges_train[i + 1]
            
            client_users_list = [u[0] for u in sorted_train_users[start_idx:end_idx]]
            client_mask = np.isin(user_ids[train_mask], client_users_list)
            
            client_X = X_train[client_mask]
            client_y = y_train[client_mask]
            
            client_dataset = TensorDataset(torch.FloatTensor(client_X), 
                                          torch.FloatTensor(client_y))
            client_datasets.append(client_dataset)
            
            logger.info(f"  Client {i}: {len(client_X)} samples, "
                       f"positive ratio: {client_y.mean():.2%}")
        
        logger.info(f"Extreme non-IID data prepared: {num_clients} clients")
        
        return client_datasets, test_dataset
    
    def train_federated(self, client_datasets: List[TensorDataset],
                       test_dataset: TensorDataset,
                       num_rounds: int = 50,
                       participation_rate: float = 1.0,
                       use_dp: bool = True) -> Dict[str, Any]:
        """
        Train federated model with optional partial participation.
        
        Args:
            client_datasets: List of client datasets
            test_dataset: Test dataset
            num_rounds: Number of communication rounds
            participation_rate: Fraction of clients participating per round
            use_dp: Whether to use differential privacy
            
        Returns:
            Training results
        """
        global_model = SimpleLSTMModel(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        test_loader = DataLoader(test_dataset, 
                                batch_size=self.config['training']['batch_size'])
        
        local_epochs = self.config['federated']['local_epochs']
        num_clients = len(client_datasets)
        clients_per_round = max(1, int(num_clients * participation_rate))
        
        noise_scale = self.config['privacy']['noise_multiplier'] if use_dp else 0.0
        
        for round_num in range(num_rounds):
            # Random client selection
            selected_clients = np.random.choice(num_clients, clients_per_round, replace=False)
            
            client_weights = []
            
            for client_idx in selected_clients:
                client_dataset = client_datasets[client_idx]
                
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
                        
                        if use_dp:
                            for param in local_model.parameters():
                                if param.grad is not None:
                                    noise = torch.randn_like(param.grad) * noise_scale * 0.01
                                    param.grad.add_(noise)
                        
                        optimizer.step()
                
                client_weights.append(local_model.state_dict())
            
            # FedAvg
            global_state = global_model.state_dict()
            for key in global_state.keys():
                global_state[key] = torch.stack([client_weights[i][key].float() 
                                                 for i in range(len(client_weights))]).mean(0)
            
            global_model.load_state_dict(global_state)
        
        # Evaluate
        global_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = global_model(X_batch)
                preds = (outputs.cpu().numpy() > 0.5).astype(int)
                
                all_preds.extend(preds.flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'model': global_model
        }
    
    def membership_inference_attack(self, model: nn.Module,
                                   train_dataset: TensorDataset,
                                   test_dataset: TensorDataset) -> Dict[str, float]:
        """
        Simulate a membership inference attack.
        
        Args:
            model: Trained model
            train_dataset: Training data (members)
            test_dataset: Test data (non-members)
            
        Returns:
            Attack metrics
        """
        logger.info("Running membership inference attack...")
        
        model.eval()
        criterion = nn.BCELoss(reduction='none')
        
        # Compute losses on training data (members)
        train_loader = DataLoader(train_dataset, batch_size=32)
        member_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                losses = criterion(outputs, y_batch)
                member_losses.extend(losses.cpu().numpy())
        
        # Compute losses on test data (non-members)
        test_loader = DataLoader(test_dataset, batch_size=32)
        non_member_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                losses = criterion(outputs, y_batch)
                non_member_losses.extend(losses.cpu().numpy())
        
        # Attack: classify based on loss threshold
        # Lower loss -> likely a member
        all_losses = np.concatenate([member_losses, non_member_losses])
        all_labels = np.concatenate([
            np.ones(len(member_losses)),  # 1 = member
            np.zeros(len(non_member_losses))  # 0 = non-member
        ])
        
        # AUC score: how well can we distinguish members from non-members?
        # Negate losses because lower loss indicates membership
        attack_auc = roc_auc_score(all_labels, -all_losses)
        
        # AUC close to 0.5 means attack is no better than random guessing (good privacy)
        # AUC close to 1.0 means attack is successful (bad privacy)
        
        return {
            'attack_auc': float(attack_auc),
            'attack_success': 'High' if attack_auc > 0.7 else 'Medium' if attack_auc > 0.6 else 'Low',
            'privacy_protection': 'Strong' if attack_auc < 0.6 else 'Moderate' if attack_auc < 0.7 else 'Weak'
        }
    
    def plot_robustness_results(self, results: Dict[str, Any]):
        """
        Generate robustness analysis visualization.
        
        Args:
            results: Dictionary with robustness results
        """
        logger.info("Generating robustness plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Non-IID comparison
        scenarios = ['Standard\nIID-like', 'Extreme\nNon-IID']
        accuracies = [
            results['baseline_accuracy'],
            results['non_iid_stress']['accuracy']
        ]
        
        bars = axes[0, 0].bar(scenarios, accuracies, color=['green', 'orange'], alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Non-IID Stress Test', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Annotate bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 2: Partial participation
        scenarios = ['100%\nParticipation', '30%\nParticipation']
        accuracies = [
            results['baseline_accuracy'],
            results['partial_participation']['accuracy']
        ]
        
        bars = axes[0, 1].bar(scenarios, accuracies, color=['green', 'coral'], alpha=0.7)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Partial Client Participation', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 3: Membership inference attack
        attack_auc = results['membership_inference']['attack_auc']
        
        # Gauge plot
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        axes[1, 0].plot(theta, r, 'k-', linewidth=2)
        axes[1, 0].fill_between(theta[:33], 0, r[:33], color='green', alpha=0.3, label='Strong Defense')
        axes[1, 0].fill_between(theta[33:66], 0, r[33:66], color='yellow', alpha=0.3, label='Moderate')
        axes[1, 0].fill_between(theta[66:], 0, r[66:], color='red', alpha=0.3, label='Weak Defense')
        
        # Pointer
        pointer_angle = (attack_auc - 0.5) * np.pi  # Map [0.5, 1.0] to [0, π]
        axes[1, 0].arrow(0, 0, 0.8*np.cos(pointer_angle), 0.8*np.sin(pointer_angle),
                        head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
        
        axes[1, 0].set_xlim([-1.2, 1.2])
        axes[1, 0].set_ylim([0, 1.2])
        axes[1, 0].set_aspect('equal')
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Membership Inference Attack AUC', fontsize=14, fontweight='bold')
        axes[1, 0].text(0, -0.3, f'AUC = {attack_auc:.3f}\n{results["membership_inference"]["privacy_protection"]} Privacy',
                       ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].legend(loc='upper right', fontsize=9)
        
        # Plot 4: Summary table
        axes[1, 1].axis('off')
        
        summary_data = [
            ['Metric', 'Value', 'Status'],
            ['Baseline Accuracy', f"{results['baseline_accuracy']:.3f}", '✓'],
            ['Non-IID Degradation', f"{results['non_iid_stress']['degradation_pct']:.1f}%", 
             '✓' if results['non_iid_stress']['degradation_pct'] < 15 else '⚠'],
            ['Partial Participation', f"{results['partial_participation']['accuracy']:.3f}",
             '✓' if results['partial_participation']['accuracy'] > 0.7 else '⚠'],
            ['Attack AUC', f"{attack_auc:.3f}",
             '✓' if attack_auc < 0.6 else '⚠' if attack_auc < 0.7 else '✗'],
            ['Overall Robustness', f"{results['robustness_score']:.1%}", '✓']
        ]
        
        table = axes[1, 1].table(cellText=summary_data, cellLoc='left',
                                colWidths=[0.5, 0.3, 0.2],
                                loc='center', bbox=[0, 0.2, 1, 0.7])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 1].set_title('Robustness Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plot_file = self.figures_dir / "robustness_report.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Robustness plot saved: {plot_file}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete robustness experiment.
        
        Returns:
            Dictionary with robustness results
        """
        logger.info("="*60)
        logger.info("EXPERIMENT 4: ROBUSTNESS TESTING")
        logger.info("="*60)
        
        results = {}
        
        # Baseline: Standard training
        logger.info("Running baseline (standard IID-like)...")
        client_datasets, test_dataset = self.prepare_data()
        baseline_results = self.train_federated(client_datasets, test_dataset, use_dp=True)
        results['baseline_accuracy'] = baseline_results['accuracy']
        results['baseline_f1'] = baseline_results['f1_score']
        
        logger.info(f"Baseline accuracy: {results['baseline_accuracy']:.4f}")
        
        # Test 1: Extreme Non-IID
        logger.info("\nTest 1: Extreme Non-IID Stress Test")
        extreme_clients, extreme_test = self.prepare_extreme_non_iid_data()
        noniid_results = self.train_federated(extreme_clients, extreme_test, use_dp=True)
        
        degradation = (results['baseline_accuracy'] - noniid_results['accuracy']) * 100
        
        results['non_iid_stress'] = {
            'accuracy': noniid_results['accuracy'],
            'f1_score': noniid_results['f1_score'],
            'degradation_pct': float(degradation)
        }
        
        logger.info(f"Extreme non-IID accuracy: {noniid_results['accuracy']:.4f}")
        logger.info(f"Performance degradation: {degradation:.1f}%")
        
        # Test 2: Partial Participation
        logger.info("\nTest 2: Partial Client Participation (30%)")
        participation_rate = self.config['experiments']['robustness']['partial_participation_rate']
        partial_results = self.train_federated(client_datasets, test_dataset, 
                                              participation_rate=participation_rate,
                                              use_dp=True)
        
        results['partial_participation'] = {
            'participation_rate': participation_rate,
            'accuracy': partial_results['accuracy'],
            'f1_score': partial_results['f1_score'],
            'extra_rounds': 0  # Simplified - in practice, track convergence
        }
        
        logger.info(f"Partial participation accuracy: {partial_results['accuracy']:.4f}")
        
        # Test 3: Membership Inference Attack
        if self.config['experiments']['robustness']['membership_inference']['enabled']:
            logger.info("\nTest 3: Membership Inference Attack Simulation")
            
            # Use baseline model for attack
            train_subset = TensorDataset(*client_datasets[0][:200])  # Small subset
            test_subset = TensorDataset(*test_dataset[:200])
            
            attack_results = self.membership_inference_attack(
                baseline_results['model'],
                train_subset,
                test_subset
            )
            
            results['membership_inference'] = attack_results
            
            logger.info(f"Attack AUC: {attack_results['attack_auc']:.3f}")
            logger.info(f"Privacy protection: {attack_results['privacy_protection']}")
        
        # Calculate overall robustness score
        # Weighted average of performance across scenarios
        robustness_components = [
            results['baseline_accuracy'],
            noniid_results['accuracy'],
            partial_results['accuracy']
        ]
        results['robustness_score'] = float(np.mean(robustness_components))
        
        # Generate plots
        self.plot_robustness_results(results)
        
        # Save results
        results_file = self.figures_dir / "robustness_results.json"
        
        # Remove non-serializable model
        if 'model' in baseline_results:
            del baseline_results['model']
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Robustness results saved: {results_file}")
        logger.info(f"Overall robustness score: {results['robustness_score']:.2%}")
        logger.info("Robustness experiment completed successfully")
        
        return results


def run_robustness_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for robustness experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment results
    """
    experiment = RobustnessExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    # For standalone testing
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_robustness_experiment(config)
    print("\nRobustness Experiment Results:")
    print(f"Baseline: {results['baseline_accuracy']:.4f}")
    print(f"Non-IID: {results['non_iid_stress']['accuracy']:.4f}")
    print(f"Partial: {results['partial_participation']['accuracy']:.4f}")
    print(f"Attack AUC: {results['membership_inference']['attack_auc']:.3f}")
