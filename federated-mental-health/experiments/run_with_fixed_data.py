#!/usr/bin/env python3
"""
Experiment runner using the fixed synthetic dataset.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split

from models.architecture import create_model
from models.train_local import LocalTrainer, TrainingConfig
from federated.server import FederatedServer, ServerConfig
from federated.client import FederatedClient, ClientConfig
from federated.aggregation import ModelAggregator, AggregationConfig, AggregationMethod
from evaluation.metrics import MetricsCalculator


def load_fixed_dataset(data_dir='./data/synthetic_fixed'):
    """Load the fixed synthetic dataset from parquet files."""
    print("Loading fixed dataset from parquet files...")
    
    # Load client partitions
    client_data = []
    clients_dir = Path(data_dir) / 'clients'
    
    for i in range(10):  # 10 clients
        client_file = clients_dir / f'client_{i}.parquet'
        if client_file.exists():
            df = pd.read_parquet(client_file)
            client_data.append(df)
            print(f"  Loaded client {i}: {len(df)} samples, {df['high_risk'].mean():.2%} positive")
        else:
            print(f"  Warning: {client_file} not found")
    
    return client_data


def prepare_sequences(df, sequence_length=7):
    """Convert dataframe to sequences for LSTM input."""
    # Feature columns (excluding high_risk and user_id and metadata)
    feature_cols = [col for col in df.columns if col not in ['high_risk', 'user_id', 'day', 'date', 'archetype', 'client_id']]
    
    # Group by user and create sequences
    sequences = []
    labels = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_index()
        
        # Create sequences
        for i in range(len(user_data) - sequence_length + 1):
            seq = user_data.iloc[i:i+sequence_length][feature_cols].values
            label = user_data.iloc[i+sequence_length-1]['high_risk']
            
            sequences.append(seq)
            labels.append(label)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    return X, y


def run_experiment(n_clients=10, n_rounds=20, sequence_length=7, output_dir='./experiments/fixed_dataset_test'):
    """Run federated learning experiment with fixed dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment_name': 'fixed_dataset_test',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_clients': n_clients,
            'n_rounds': n_rounds,
            'sequence_length': sequence_length,
        },
        'phases': {},
        'final_metrics': {}
    }
    
    print("="*60)
    print("Phase 1: Loading Fixed Synthetic Data")
    print("="*60)
    
    client_datasets = load_fixed_dataset()
    
    if len(client_datasets) == 0:
        raise ValueError("No client data loaded! Check if synthetic_fixed/ directory exists.")
    
    # Use only the requested number of clients
    n_clients = min(n_clients, len(client_datasets))
    client_datasets = client_datasets[:n_clients]
    
    print(f"\nUsing {n_clients} clients")
    
    results['phases']['data_loading'] = {
        'n_clients': n_clients,
        'total_samples': sum(len(df) for df in client_datasets)
    }
    
    print("\n" + "="*60)
    print("Phase 2: Preparing Sequences for Each Client")
    print("="*60)
    
    client_data = []
    all_X_test = []
    all_y_test = []
    
    for i, df in enumerate(client_datasets):
        # Prepare sequences
        X, y = prepare_sequences(df, sequence_length)
        
        # Split into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Client {i}: Train={len(X_train)}, Test={len(X_test)}, "
              f"Pos rate={y_train.mean():.2%}")
        
        client_data.append({
            'X': X_train,
            'y': y_train
        })
        
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    
    # Combine all test sets for global evaluation
    X_test = np.concatenate(all_X_test, axis=0)
    y_test = np.concatenate(all_y_test, axis=0)
    
    input_dim = client_data[0]['X'].shape[2]
    
    print(f"\nGlobal test set: {len(X_test)} samples, {y_test.mean():.2%} positive")
    
    results['phases']['preprocessing'] = {
        'sequence_length': sequence_length,
        'input_dim': int(input_dim),
        'global_test_size': len(X_test),
        'global_test_positive_rate': float(y_test.mean())
    }
    
    print("\n" + "="*60)
    print("Phase 3: Federated Training")
    print("="*60)
    
    # Create server
    server_config = ServerConfig(
        n_rounds=n_rounds,
        min_clients_per_round=min(5, n_clients),
        aggregation_strategy='fedavg',
        add_dp_noise=False  # Disable for initial test
    )
    server = FederatedServer(input_dim=input_dim, config=server_config)
    
    # Create clients
    clients = []
    for i, data in enumerate(client_data):
        client_config = ClientConfig(
            client_id=i,
            local_epochs=3,
            batch_size=32,
            use_dp=False  # Disable DP for initial test
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
    best_f1 = 0.0
    best_model_state = None
    
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
        
        # Evaluate on test set every 2 rounds
        if (round_num + 1) % 2 == 0 or round_num == n_rounds - 1:
            server._set_model_state(server.global_model, server.global_state)
            server.global_model.eval()
            
            X_test_tensor = torch.FloatTensor(X_test)
            with torch.no_grad():
                logits, _ = server.global_model(X_test_tensor)
                probs = torch.sigmoid(logits).numpy().flatten()
            
            calc = MetricsCalculator()
            metrics = calc.compute_classification_metrics(y_test, probs)
            
            print(f"  Accuracy: {metrics.accuracy:.4f}, "
                  f"F1: {metrics.f1:.4f}, "
                  f"AUC-ROC: {metrics.auc_roc:.4f}, "
                  f"AUC-PR: {metrics.auc_pr:.4f}")
            
            round_metrics.append({
                'round': round_num + 1,
                'accuracy': float(metrics.accuracy),
                'balanced_accuracy': float(metrics.balanced_accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1': float(metrics.f1),
                'auc_roc': float(metrics.auc_roc),
                'auc_pr': float(metrics.auc_pr),
                'mcc': float(metrics.mcc)
            })
            
            # Track best model
            if metrics.f1 > best_f1:
                best_f1 = metrics.f1
                best_model_state = server.global_model.state_dict().copy()
    
    results['phases']['training'] = {
        'n_rounds': n_rounds,
        'round_metrics': round_metrics
    }
    
    print("\n" + "="*60)
    print("Phase 4: Final Evaluation")
    print("="*60)
    
    # Final evaluation with best model
    if best_model_state is not None:
        server.global_model.load_state_dict(best_model_state)
    
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
    
    print(f"\nFinal Results (Best Model):")
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
        'model_state_dict': best_model_state if best_model_state else server.global_model.state_dict(),
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
    print(f"\nKey Improvements vs Previous Run:")
    print(f"  F1 Score: {final_metrics.f1:.4f} (previous: 0.0000)")
    print(f"  AUC-PR: {final_metrics.auc_pr:.4f} (previous: 0.0115)")
    print(f"  Balanced Acc: {final_metrics.balanced_accuracy:.4f} (vs raw accuracy)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run federated learning with fixed dataset')
    parser.add_argument('--n-clients', type=int, default=10, help='Number of clients (max 10)')
    parser.add_argument('--n-rounds', type=int, default=20, help='Number of rounds')
    parser.add_argument('--sequence-length', type=int, default=7, help='Sequence length')
    parser.add_argument('--output-dir', type=str, default='./experiments/fixed_dataset_test', 
                        help='Output directory')
    
    args = parser.parse_args()
    
    results = run_experiment(
        n_clients=args.n_clients,
        n_rounds=args.n_rounds,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir
    )
