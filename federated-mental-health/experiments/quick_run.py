#!/usr/bin/env python3
"""
Quick experiment runner script.
Runs a simplified federated learning experiment to generate baseline results.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

import json
import numpy as np
import torch
from datetime import datetime

from data.synthetic_generator import SyntheticMentalHealthData, SyntheticConfig
from data.preprocess import MentalHealthPreprocessor, PreprocessingConfig
from data.federated_partition import FederatedPartitioner, PartitionConfig
from models.architecture import create_model
from models.train_local import LocalTrainer, TrainingConfig
from federated.server import FederatedServer, ServerConfig
from federated.client import FederatedClient, ClientConfig
from federated.aggregation import ModelAggregator, AggregationConfig, AggregationMethod
from evaluation.metrics import MetricsCalculator


def run_quick_experiment(n_clients=5, n_rounds=10, n_users=500, output_dir='./experiments/quick_test'):
    """Run a quick federated learning experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment_name': 'quick_test',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_clients': n_clients,
            'n_rounds': n_rounds,
            'n_users': n_users,
        },
        'phases': {},
        'final_metrics': {}
    }
    
    print("="*60)
    print("Phase 1: Generating Synthetic Data")
    print("="*60)
    
    config = SyntheticConfig(n_users=n_users, n_days=14)
    generator = SyntheticMentalHealthData(config=config)
    time_series_df, profiles_df = generator.generate()
    
    results['phases']['data_generation'] = {
        'n_samples': len(time_series_df),
        'n_users': n_users,
        'high_risk_rate': float(time_series_df['high_risk_day'].mean())
    }
    
    print("\n" + "="*60)
    print("Phase 2: Preprocessing Data")
    print("="*60)
    
    preprocess_config = PreprocessingConfig(sequence_length=7)
    preprocessor = MentalHealthPreprocessor(preprocess_config)
    
    X, y, user_ids = preprocessor.fit_transform(time_series_df)
    
    # Split into train/test (80/20) while keeping user_ids aligned
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
        X, y, user_ids, test_size=0.2, random_state=42
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    results['phases']['preprocessing'] = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'input_dim': int(X_train.shape[2]) if len(X_train.shape) > 2 else X_train.shape[1],
        'sequence_length': int(X_train.shape[1]) if len(X_train.shape) > 1 else 1
    }
    
    print("\n" + "="*60)
    print("Phase 3: Partitioning for Federated Learning")
    print("="*60)
    
    partition_config = PartitionConfig(n_clients=n_clients, partition_strategy='iid')
    partitioner = FederatedPartitioner(partition_config)
    client_data_dict = partitioner.partition(X_train, y_train, user_ids_train)
    
    # Convert dict to list for easier iteration
    client_data = [client_data_dict[i] for i in range(n_clients)]
    
    print(f"Created {n_clients} client partitions")
    for i, data in enumerate(client_data):
        print(f"  Client {i}: {len(data['y'])} samples")
    
    print("\n" + "="*60)
    print("Phase 4: Federated Training")
    print("="*60)
    
    input_dim = X_train.shape[2]
    
    # Create server
    server_config = ServerConfig(
        n_rounds=n_rounds,
        min_clients_per_round=min(3, n_clients),
        aggregation_strategy='fedavg',
        add_dp_noise=False  # Disable for quick test
    )
    server = FederatedServer(input_dim=input_dim, config=server_config)
    
    # Create clients
    clients = []
    for i, data in enumerate(client_data):
        client_config = ClientConfig(
            client_id=i,
            local_epochs=2,
            batch_size=32,
            use_dp=False  # Disable DP for quick test
        )
        client = FederatedClient(
            client_id=i,
            X_train=data['X'],
            y_train=data['y'],
            config=client_config
        )
        clients.append(client)
    
    # Run federated training
    round_metrics = []
    for round_num in range(n_rounds):
        print(f"\nRound {round_num + 1}/{n_rounds}")
        
        # Get global model state
        global_state = server.get_global_model_state()
        
        # Train each client
        client_updates = []
        for client in clients:
            client.receive_global_model(global_state)
            model_state, metrics = client.train_local()
            client_updates.append((client.client_id, model_state, client.n_samples))
        
        # Aggregate updates
        aggregated_state = server.aggregate_updates(client_updates)
        server.global_state = aggregated_state
        server.current_round += 1
        
        # Evaluate on test set (every few rounds)
        if (round_num + 1) % 2 == 0 or round_num == n_rounds - 1:
            server._set_model_state(server.global_model, server.global_state)
            server.global_model.eval()
            
            X_test_tensor = torch.FloatTensor(X_test)
            with torch.no_grad():
                logits, _ = server.global_model(X_test_tensor)
                probs = torch.sigmoid(logits).numpy().flatten()
            
            calc = MetricsCalculator()
            metrics = calc.compute_classification_metrics(y_test, probs)
            
            print(f"  Test Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")
            
            round_metrics.append({
                'round': round_num + 1,
                'accuracy': float(metrics.accuracy),
                'auc_roc': float(metrics.auc_roc),
                'f1': float(metrics.f1)
            })
    
    results['phases']['training'] = {
        'n_rounds': n_rounds,
        'round_metrics': round_metrics
    }
    
    print("\n" + "="*60)
    print("Phase 5: Final Evaluation")
    print("="*60)
    
    # Final evaluation
    server._set_model_state(server.global_model, server.global_state)
    server.global_model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        logits, attention = server.global_model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
    
    calc = MetricsCalculator()
    final_metrics = calc.compute_classification_metrics(y_test, probs)
    
    results['final_metrics'] = {
        'accuracy': float(final_metrics.accuracy),
        'balanced_accuracy': float(final_metrics.balanced_accuracy),
        'precision': float(final_metrics.precision),
        'recall': float(final_metrics.recall),
        'f1': float(final_metrics.f1),
        'auc_roc': float(final_metrics.auc_roc),
        'auc_pr': float(final_metrics.auc_pr),
        'mcc': float(final_metrics.mcc)
    }
    
    print(f"\nFinal Results:")
    print(f"  Accuracy: {final_metrics.accuracy:.4f}")
    print(f"  Balanced Accuracy: {final_metrics.balanced_accuracy:.4f}")
    print(f"  Precision: {final_metrics.precision:.4f}")
    print(f"  Recall: {final_metrics.recall:.4f}")
    print(f"  F1 Score: {final_metrics.f1:.4f}")
    print(f"  AUC-ROC: {final_metrics.auc_roc:.4f}")
    print(f"  AUC-PR: {final_metrics.auc_pr:.4f}")
    print(f"  MCC: {final_metrics.mcc:.4f}")
    
    # Save model checkpoint
    print("\n" + "="*60)
    print("Saving Results and Checkpoints")
    print("="*60)
    
    checkpoint_path = output_path / 'model_checkpoint.pt'
    torch.save({
        'model_state_dict': server.global_model.state_dict(),
        'round': n_rounds,
        'final_metrics': results['final_metrics'],
        'config': results['config']
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")
    
    # Save results JSON
    results_path = output_path / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save round metrics for plotting
    metrics_path = output_path / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(round_metrics, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run quick federated learning experiment')
    parser.add_argument('--n-clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--n-rounds', type=int, default=10, help='Number of rounds')
    parser.add_argument('--n-users', type=int, default=500, help='Number of synthetic users')
    parser.add_argument('--output-dir', type=str, default='./experiments/quick_test', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_quick_experiment(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        n_users=args.n_users,
        output_dir=args.output_dir
    )
