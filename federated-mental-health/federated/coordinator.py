"""
Federated Training Coordinator
Orchestrates the complete federated learning pipeline.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import copy

from .server import FederatedServer, ServerConfig
from .client import ClientManager, ClientConfig, FederatedClient

import sys
sys.path.append('..')


@dataclass
class FederatedConfig:
    """Complete configuration for federated training."""
    # Data
    data_dir: str = './data/processed'
    partition_dir: str = './data/partitions'
    
    # Server
    n_rounds: int = 100
    client_fraction: float = 0.3
    min_clients_per_round: int = 3
    aggregation_strategy: str = 'fedavg'
    server_momentum: float = 0.9
    server_dp_noise: float = 0.1
    
    # Client
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Differential Privacy
    use_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    
    # Training
    early_stopping_rounds: int = 15
    target_accuracy: Optional[float] = None
    checkpoint_every: int = 10
    
    # Output
    output_dir: str = './experiments'
    experiment_name: str = 'federated_run'
    
    # Model
    model_config: Dict = field(default_factory=lambda: {
        'hidden_dim': 128,
        'n_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    })


class FederatedCoordinator:
    """
    Main coordinator for federated learning experiments.
    
    Orchestrates:
    - Server and client initialization
    - Training round execution
    - Metrics collection and logging
    - Checkpointing and experiment management
    """
    
    def __init__(self, config: FederatedConfig):
        """
        Initialize federated coordinator.
        
        Args:
            config: Federated training configuration
        """
        self.config = config
        
        # Initialize paths
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Components (initialized later)
        self.server: Optional[FederatedServer] = None
        self.client_manager: Optional[ClientManager] = None
        
        # Training state
        self.current_round = 0
        self.is_initialized = False
        
        # Metrics
        self.training_metrics: List[Dict] = []
        self.client_metrics: List[Dict] = []
        
        # Validation data
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        
        print(f"Federated Coordinator initialized")
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Output: {self.experiment_dir}")
    
    def setup_from_partitions(self, 
                             partition_dir: Optional[str] = None,
                             val_data_path: Optional[str] = None) -> None:
        """
        Setup server and clients from pre-partitioned data.
        
        Args:
            partition_dir: Path to partitioned data
            val_data_path: Path to validation data directory
        """
        partition_path = Path(partition_dir or self.config.partition_dir)
        
        # Create client config template
        client_config = ClientConfig(
            client_id=0,
            local_epochs=self.config.local_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            use_dp=self.config.use_dp,
            dp_epsilon=self.config.dp_epsilon,
            dp_delta=self.config.dp_delta,
            dp_max_grad_norm=self.config.dp_max_grad_norm,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize client manager and load clients
        self.client_manager = ClientManager(client_config_template=client_config)
        self.client_manager.load_from_partitions(str(partition_path))
        
        if len(self.client_manager.clients) == 0:
            raise RuntimeError("No clients loaded from partitions")
        
        # Get input dimension from first client
        first_client = list(self.client_manager.clients.values())[0]
        input_dim = first_client.input_dim
        
        # Create server config
        server_config = ServerConfig(
            n_rounds=self.config.n_rounds,
            min_clients_per_round=self.config.min_clients_per_round,
            client_fraction=self.config.client_fraction,
            aggregation_strategy=self.config.aggregation_strategy,
            momentum=self.config.server_momentum,
            add_dp_noise=self.config.use_dp,
            dp_noise_multiplier=self.config.server_dp_noise,
            early_stopping_rounds=self.config.early_stopping_rounds,
            target_accuracy=self.config.target_accuracy,
            checkpoint_every=self.config.checkpoint_every,
            model_config=self.config.model_config
        )
        
        # Initialize server
        self.server = FederatedServer(input_dim, server_config)
        
        # Register clients with server
        for client_id, client in self.client_manager.clients.items():
            self.server.register_client(client_id, client.n_samples)
        
        # Load validation data if provided
        if val_data_path:
            val_path = Path(val_data_path)
            if (val_path / 'X_val.npy').exists():
                self.X_val = np.load(val_path / 'X_val.npy')
                self.y_val = np.load(val_path / 'y_val.npy')
                print(f"Loaded validation data: {len(self.X_val)} samples")
        
        self.is_initialized = True
        
        print(f"\nSetup complete:")
        print(f"  Clients: {len(self.client_manager.clients)}")
        print(f"  Total samples: {self.client_manager.get_total_samples()}")
        print(f"  Input dimension: {input_dim}")
    
    def setup_with_data(self,
                        client_data: List[Dict[str, np.ndarray]],
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None) -> None:
        """
        Setup server and clients with provided data.
        
        Args:
            client_data: List of dicts with 'X_train', 'y_train' for each client
            X_val: Validation features
            y_val: Validation labels
        """
        # Create client config template
        client_config = ClientConfig(
            client_id=0,
            local_epochs=self.config.local_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            use_dp=self.config.use_dp,
            dp_epsilon=self.config.dp_epsilon,
            dp_delta=self.config.dp_delta,
            dp_max_grad_norm=self.config.dp_max_grad_norm,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize client manager
        self.client_manager = ClientManager(client_config_template=client_config)
        
        # Add clients
        for i, data in enumerate(client_data):
            self.client_manager.add_client(
                client_id=i,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_val=data.get('X_val'),
                y_val=data.get('y_val')
            )
        
        # Get input dimension
        input_dim = client_data[0]['X_train'].shape[2]
        
        # Create server config
        server_config = ServerConfig(
            n_rounds=self.config.n_rounds,
            min_clients_per_round=self.config.min_clients_per_round,
            client_fraction=self.config.client_fraction,
            aggregation_strategy=self.config.aggregation_strategy,
            momentum=self.config.server_momentum,
            add_dp_noise=self.config.use_dp,
            dp_noise_multiplier=self.config.server_dp_noise,
            early_stopping_rounds=self.config.early_stopping_rounds,
            target_accuracy=self.config.target_accuracy,
            checkpoint_every=self.config.checkpoint_every,
            model_config=self.config.model_config
        )
        
        # Initialize server
        self.server = FederatedServer(input_dim, server_config)
        
        # Register clients
        for client_id, client in self.client_manager.clients.items():
            self.server.register_client(client_id, client.n_samples)
        
        # Store validation data
        self.X_val = X_val
        self.y_val = y_val
        
        self.is_initialized = True
        
        print(f"\nSetup complete:")
        print(f"  Clients: {len(self.client_manager.clients)}")
        print(f"  Total samples: {self.client_manager.get_total_samples()}")
    
    def run_round(self) -> Dict[str, Any]:
        """
        Execute one federated training round.
        
        Returns:
            Round metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Must call setup_* before training")
        
        self.current_round += 1
        print(f"\n{'='*50}")
        print(f"Round {self.current_round}/{self.config.n_rounds}")
        print(f"{'='*50}")
        
        # 1. Select clients
        available_clients = self.client_manager.get_all_client_ids()
        selected_clients = self.server.select_clients(available_clients)
        print(f"Selected {len(selected_clients)} clients: {selected_clients}")
        
        # 2. Distribute global model to selected clients
        global_state = self.server.get_global_model_state()
        self.client_manager.distribute_global_model(global_state, selected_clients)
        
        # 3. Train selected clients
        print("Training clients...")
        client_updates = self.client_manager.train_selected_clients(selected_clients)
        
        # 4. Aggregate updates on server
        round_metrics = self.server.run_round(
            client_updates,
            self.X_val,
            self.y_val
        )
        
        # Log metrics
        print(f"\nRound {self.current_round} results:")
        if 'accuracy' in round_metrics:
            print(f"  Validation Accuracy: {round_metrics['accuracy']:.4f}")
            print(f"  Validation Loss: {round_metrics['loss']:.4f}")
            print(f"  Best Accuracy: {round_metrics.get('best_accuracy', 0):.4f}")
        
        self.training_metrics.append(round_metrics)
        
        # Collect client metrics
        client_round_metrics = {
            'round': self.current_round,
            'clients': [
                self.client_manager.clients[cid].get_client_summary()
                for cid in selected_clients
            ]
        }
        self.client_metrics.append(client_round_metrics)
        
        # Checkpointing
        if round_metrics.get('checkpoint'):
            self._save_checkpoint()
        
        return round_metrics
    
    def train(self, n_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete federated training.
        
        Args:
            n_rounds: Number of rounds (overrides config)
            
        Returns:
            Final training summary
        """
        if not self.is_initialized:
            raise RuntimeError("Must call setup_* before training")
        
        rounds = n_rounds or self.config.n_rounds
        
        print(f"\n{'#'*60}")
        print(f"Starting Federated Training: {rounds} rounds")
        print(f"{'#'*60}")
        
        start_time = datetime.now()
        
        for _ in range(rounds):
            metrics = self.run_round()
            
            # Early stopping
            if metrics.get('early_stop'):
                print(f"\nEarly stopping at round {self.current_round}")
                break
            
            # Target reached
            if metrics.get('target_reached'):
                print(f"\nTarget accuracy reached at round {self.current_round}")
                break
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save final model and results
        self._save_final_results(training_time)
        
        return self.get_training_summary()
    
    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.experiment_dir / f'checkpoint_round_{self.current_round}.pt'
        self.server.save_checkpoint(str(checkpoint_path))
    
    def _save_final_results(self, training_time: float) -> None:
        """Save final training results."""
        # Save final model
        final_model_path = self.experiment_dir / 'final_model.pt'
        self.server.save_checkpoint(str(final_model_path))
        
        # Save metrics
        metrics_path = self.experiment_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'config': {
                    'n_rounds': self.config.n_rounds,
                    'n_clients': len(self.client_manager.clients),
                    'aggregation': self.config.aggregation_strategy,
                    'dp_epsilon': self.config.dp_epsilon,
                    'use_dp': self.config.use_dp
                },
                'training_time': training_time,
                'rounds_completed': self.current_round,
                'round_metrics': self.training_metrics,
                'client_metrics': self.client_metrics
            }, f, indent=2, default=str)
        
        # Save summary
        summary = self.get_training_summary()
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to: {self.experiment_dir}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        server_summary = self.server.get_training_summary() if self.server else {}
        
        return {
            'experiment_name': self.config.experiment_name,
            'rounds_completed': self.current_round,
            'server': server_summary,
            'clients': {
                'n_clients': len(self.client_manager.clients) if self.client_manager else 0,
                'total_samples': self.client_manager.get_total_samples() if self.client_manager else 0
            },
            'privacy': {
                'dp_enabled': self.config.use_dp,
                'epsilon': self.config.dp_epsilon,
                'delta': self.config.dp_delta
            }
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training from checkpoint."""
        if not self.is_initialized:
            raise RuntimeError("Must call setup_* before loading checkpoint")
        
        self.server.load_checkpoint(checkpoint_path)
        self.current_round = self.server.current_round


def run_federated_experiment(
    partition_dir: str,
    val_data_path: str,
    experiment_name: str = 'experiment',
    n_rounds: int = 50,
    use_dp: bool = True,
    dp_epsilon: float = 1.0,
    aggregation: str = 'fedavg',
    output_dir: str = './experiments'
) -> Dict[str, Any]:
    """
    Run a complete federated learning experiment.
    
    Args:
        partition_dir: Path to partitioned data
        val_data_path: Path to validation data
        experiment_name: Name for this experiment
        n_rounds: Number of training rounds
        use_dp: Enable differential privacy
        dp_epsilon: Privacy budget
        aggregation: Aggregation strategy
        output_dir: Output directory
        
    Returns:
        Training summary
    """
    config = FederatedConfig(
        partition_dir=partition_dir,
        n_rounds=n_rounds,
        use_dp=use_dp,
        dp_epsilon=dp_epsilon,
        aggregation_strategy=aggregation,
        output_dir=output_dir,
        experiment_name=experiment_name
    )
    
    coordinator = FederatedCoordinator(config)
    coordinator.setup_from_partitions(partition_dir, val_data_path)
    
    return coordinator.train()


if __name__ == "__main__":
    # Example usage
    import tempfile
    import os
    
    # Create temporary data for testing
    np.random.seed(42)
    
    input_dim = 42
    seq_len = 7
    n_clients = 5
    
    # Create synthetic client data
    client_data = []
    for i in range(n_clients):
        n_samples = np.random.randint(50, 150)
        client_data.append({
            'X_train': np.random.randn(n_samples, seq_len, input_dim).astype(np.float32),
            'y_train': np.random.randint(0, 2, n_samples).astype(np.float32)
        })
    
    # Validation data
    X_val = np.random.randn(100, seq_len, input_dim).astype(np.float32)
    y_val = np.random.randint(0, 2, 100).astype(np.float32)
    
    # Create config
    config = FederatedConfig(
        n_rounds=5,
        local_epochs=2,
        use_dp=True,
        dp_epsilon=1.0,
        aggregation_strategy='fedavg',
        experiment_name='test_experiment',
        output_dir='./test_experiments'
    )
    
    # Create and run coordinator
    coordinator = FederatedCoordinator(config)
    coordinator.setup_with_data(client_data, X_val, y_val)
    
    summary = coordinator.train(n_rounds=5)
    
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
