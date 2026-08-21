"""
Federated Learning Client for Mental Health Prediction
Handles local training with differential privacy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from pathlib import Path
import copy

import sys
sys.path.append('..')
from models.architecture import create_model, MentalHealthPredictor
from models.train_local import LocalTrainer, TrainingConfig


@dataclass
class ClientConfig:
    """Configuration for federated client."""
    client_id: int
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    use_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    weight_decay: float = 1e-5
    early_stopping: bool = True
    patience: int = 3


class FederatedClient:
    """
    Federated Learning Client.
    
    Responsible for:
    - Receiving global model from server
    - Training on local data with differential privacy
    - Computing and sending model updates
    - Privacy budget management
    """
    
    def __init__(self,
                 client_id: int,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 config: Optional[ClientConfig] = None):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            X_train: Training features [n_samples, seq_len, features]
            y_train: Training labels [n_samples]
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            config: Client configuration
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.config = config or ClientConfig(client_id=client_id)
        
        # Infer model input dimension
        self.input_dim = X_train.shape[2]
        self.n_samples = len(X_train)
        
        # Local model (initialized when receiving global model)
        self.local_model: Optional[MentalHealthPredictor] = None
        
        # Training state
        self.rounds_participated = 0
        self.total_privacy_spent = 0.0
        self.training_history: List[Dict] = []
        
        print(f"Client {client_id} initialized: {self.n_samples} samples")
    
    def receive_global_model(self, global_state: Dict[str, torch.Tensor]) -> None:
        """
        Receive and set global model state from server.
        
        Args:
            global_state: Model state dictionary from server
        """
        # Create model if not exists
        if self.local_model is None:
            self.local_model = create_model(self.input_dim)
        
        # Set model state
        for name, param in self.local_model.named_parameters():
            if name in global_state:
                param.data = global_state[name].clone()
    
    def train_local(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Perform local training on client data.
        
        Returns:
            Tuple of (model_state, training_metrics)
        """
        if self.local_model is None:
            raise RuntimeError("Must receive global model before training")
        
        # Create training config
        train_config = TrainingConfig(
            epochs=self.config.local_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_dp=self.config.use_dp,
            dp_epsilon=self.config.dp_epsilon,
            dp_delta=self.config.dp_delta,
            max_grad_norm=self.config.dp_max_grad_norm,
            early_stopping_patience=self.config.patience
        )
        
        # Create trainer
        trainer = LocalTrainer(
            model=self.local_model,
            config=train_config
        )
        
        # Train
        metrics = trainer.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val
        )
        
        # Update privacy budget
        if hasattr(trainer, 'privacy_spent'):
            self.total_privacy_spent += trainer.privacy_spent
        
        # Get updated model state
        model_state = trainer.get_model_state()
        
        # Update training history
        self.rounds_participated += 1
        self.training_history.append({
            'round': self.rounds_participated,
            'final_loss': metrics.get('final_loss', 0),
            'final_accuracy': metrics.get('final_accuracy', 0),
            'epochs_trained': metrics.get('epochs_trained', self.config.local_epochs)
        })
        
        training_metrics = {
            'client_id': self.client_id,
            'n_samples': self.n_samples,
            'final_loss': metrics.get('final_loss', 0),
            'final_accuracy': metrics.get('final_accuracy', 0),
            'privacy_spent': self.total_privacy_spent
        }
        
        return model_state, training_metrics
    
    def compute_model_update(self,
                            old_state: Dict[str, torch.Tensor],
                            new_state: Dict[str, torch.Tensor]
                            ) -> Dict[str, torch.Tensor]:
        """
        Compute model update (delta) for sending to server.
        
        Args:
            old_state: Model state before training
            new_state: Model state after training
            
        Returns:
            Model update (new - old)
        """
        update = {}
        for name in new_state.keys():
            update[name] = new_state[name] - old_state[name]
        return update
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state."""
        if self.local_model is None:
            raise RuntimeError("Model not initialized")
        
        return {
            name: param.data.clone()
            for name, param in self.local_model.named_parameters()
        }
    
    def evaluate_local(self) -> Dict[str, float]:
        """
        Evaluate model on local data.
        
        Returns:
            Evaluation metrics
        """
        if self.local_model is None:
            raise RuntimeError("Model not initialized")
        
        self.local_model.eval()
        
        # Evaluate on validation data if available, else training data
        X = self.X_val if self.X_val is not None else self.X_train
        y = self.y_val if self.y_val is not None else self.y_train
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        with torch.no_grad():
            logits, _ = self.local_model(X_tensor)
            probs = torch.sigmoid(logits).squeeze()
            
            # Loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(), y_tensor
            ).item()
            
            # Accuracy
            preds = (probs >= 0.5).float()
            accuracy = (preds == y_tensor).float().mean().item()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'n_samples': len(X)
        }
    
    def get_client_summary(self) -> Dict[str, Any]:
        """Get summary of client status."""
        return {
            'client_id': self.client_id,
            'n_samples': self.n_samples,
            'rounds_participated': self.rounds_participated,
            'total_privacy_spent': self.total_privacy_spent,
            'has_validation': self.X_val is not None,
            'training_history': self.training_history
        }


class ClientManager:
    """
    Manages multiple federated clients.
    
    Handles client creation, selection, and coordination.
    """
    
    def __init__(self, client_config_template: Optional[ClientConfig] = None):
        """
        Initialize client manager.
        
        Args:
            client_config_template: Template configuration for new clients
        """
        self.config_template = client_config_template
        self.clients: Dict[int, FederatedClient] = {}
    
    def add_client(self,
                   client_id: int,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> FederatedClient:
        """
        Add a new client.
        
        Args:
            client_id: Unique client ID
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Created client
        """
        config = None
        if self.config_template:
            config = ClientConfig(
                client_id=client_id,
                local_epochs=self.config_template.local_epochs,
                batch_size=self.config_template.batch_size,
                learning_rate=self.config_template.learning_rate,
                use_dp=self.config_template.use_dp,
                dp_epsilon=self.config_template.dp_epsilon,
                dp_delta=self.config_template.dp_delta,
                dp_max_grad_norm=self.config_template.dp_max_grad_norm,
                weight_decay=self.config_template.weight_decay,
                early_stopping=self.config_template.early_stopping,
                patience=self.config_template.patience
            )
        
        client = FederatedClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config
        )
        
        self.clients[client_id] = client
        return client
    
    def load_from_partitions(self, partition_dir: str) -> None:
        """
        Load clients from partitioned data directory.
        
        Args:
            partition_dir: Path to partition directory
        """
        import json
        
        partition_path = Path(partition_dir)
        
        # Load partition stats
        stats_path = partition_path / 'partition_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                n_clients = stats.get('n_clients', 0)
        else:
            # Count client directories
            n_clients = len(list(partition_path.glob('client_*')))
        
        print(f"Loading {n_clients} clients from {partition_dir}")
        
        for client_id in range(n_clients):
            client_dir = partition_path / f'client_{client_id}'
            
            if not client_dir.exists():
                print(f"Warning: Client directory not found: {client_dir}")
                continue
            
            # Load data
            X_train = np.load(client_dir / 'X.npy')
            y_train = np.load(client_dir / 'y.npy')
            
            # Load validation if exists
            X_val = None
            y_val = None
            if (client_dir / 'X_val.npy').exists():
                X_val = np.load(client_dir / 'X_val.npy')
                y_val = np.load(client_dir / 'y_val.npy')
            
            self.add_client(client_id, X_train, y_train, X_val, y_val)
        
        print(f"Loaded {len(self.clients)} clients")
    
    def distribute_global_model(self, 
                                global_state: Dict[str, torch.Tensor],
                                client_ids: Optional[List[int]] = None) -> None:
        """
        Distribute global model to clients.
        
        Args:
            global_state: Global model state
            client_ids: Specific clients to update (None = all)
        """
        targets = client_ids if client_ids else list(self.clients.keys())
        
        for client_id in targets:
            if client_id in self.clients:
                self.clients[client_id].receive_global_model(global_state)
    
    def train_selected_clients(self,
                              client_ids: List[int]
                              ) -> List[Tuple[int, Dict[str, torch.Tensor], int]]:
        """
        Train selected clients and collect updates.
        
        Args:
            client_ids: IDs of clients to train
            
        Returns:
            List of (client_id, model_state, n_samples)
        """
        updates = []
        
        for client_id in client_ids:
            if client_id not in self.clients:
                print(f"Warning: Client {client_id} not found")
                continue
            
            client = self.clients[client_id]
            model_state, metrics = client.train_local()
            
            updates.append((client_id, model_state, client.n_samples))
            
            print(f"  Client {client_id}: loss={metrics['final_loss']:.4f}, "
                  f"acc={metrics['final_accuracy']:.4f}")
        
        return updates
    
    def get_all_client_ids(self) -> List[int]:
        """Get all registered client IDs."""
        return list(self.clients.keys())
    
    def get_total_samples(self) -> int:
        """Get total samples across all clients."""
        return sum(client.n_samples for client in self.clients.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all clients."""
        return {
            'n_clients': len(self.clients),
            'total_samples': self.get_total_samples(),
            'clients': {
                client_id: client.get_client_summary()
                for client_id, client in self.clients.items()
            }
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    input_dim = 42
    seq_len = 7
    
    # Create sample clients with synthetic data
    manager = ClientManager(
        client_config_template=ClientConfig(
            client_id=0,
            local_epochs=3,
            batch_size=16,
            use_dp=True,
            dp_epsilon=1.0
        )
    )
    
    # Add clients
    for i in range(3):
        n_samples = np.random.randint(50, 200)
        X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        
        manager.add_client(i, X, y)
    
    print(f"\nTotal clients: {len(manager.clients)}")
    print(f"Total samples: {manager.get_total_samples()}")
    
    # Simulate one round
    from models.architecture import create_model
    
    global_model = create_model(input_dim)
    global_state = {
        name: param.data.clone()
        for name, param in global_model.named_parameters()
    }
    
    # Distribute global model
    manager.distribute_global_model(global_state)
    
    # Train all clients
    print("\nTraining clients...")
    updates = manager.train_selected_clients([0, 1, 2])
    
    print(f"\nReceived {len(updates)} updates")
