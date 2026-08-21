"""
Federated Learning Server with FedAvg and Differential Privacy
Coordinates training across distributed clients.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import copy
from datetime import datetime

import sys
sys.path.append('..')
from models.architecture import create_model, MentalHealthPredictor


@dataclass
class ServerConfig:
    """Configuration for federated server."""
    n_rounds: int = 100
    min_clients_per_round: int = 3
    client_fraction: float = 0.3  # Fraction of clients per round
    aggregation_strategy: str = 'fedavg'  # 'fedavg', 'fedavg_momentum', 'median', 'trimmed_mean'
    momentum: float = 0.9  # For FedAvg with momentum
    add_dp_noise: bool = True
    dp_noise_multiplier: float = 0.1  # Noise added to aggregated model
    dp_sensitivity: float = 1.0  # Sensitivity for DP noise
    checkpoint_every: int = 10
    early_stopping_rounds: int = 10
    target_accuracy: Optional[float] = None
    model_config: Dict = field(default_factory=dict)


class FederatedServer:
    """
    Federated Learning Server implementing FedAvg with extensions.
    
    Features:
    - FedAvg with optional momentum
    - Byzantine-robust aggregation (median, trimmed mean)
    - Differential privacy noise addition
    - Client selection strategies
    - Model versioning and checkpointing
    - Early stopping based on validation metrics
    """
    
    def __init__(self, 
                 input_dim: int,
                 config: Optional[ServerConfig] = None):
        """
        Initialize federated server.
        
        Args:
            input_dim: Model input dimension
            config: Server configuration
        """
        self.config = config or ServerConfig()
        self.input_dim = input_dim
        
        # Initialize global model
        self.global_model = create_model(input_dim, self.config.model_config)
        self.global_state = self._get_model_state(self.global_model)
        
        # Momentum buffer for FedAvg with momentum
        self.momentum_buffer: Optional[Dict[str, torch.Tensor]] = None
        
        # Training state
        self.current_round = 0
        self.best_accuracy = 0.0
        self.rounds_without_improvement = 0
        
        # History
        self.history = {
            'rounds': [],
            'val_accuracy': [],
            'val_loss': [],
            'n_clients': [],
            'aggregation_time': []
        }
        
        # Client registry
        self.registered_clients: Dict[int, Dict] = {}
        
        print(f"Federated Server initialized")
        print(f"  Aggregation: {self.config.aggregation_strategy}")
        print(f"  DP noise: {self.config.add_dp_noise} (sigma={self.config.dp_noise_multiplier})")
    
    def _get_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract model state dictionary."""
        return {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
    
    def _set_model_state(self, model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
        """Set model state from dictionary."""
        for name, param in model.named_parameters():
            if name in state:
                param.data = state[name].clone()
    
    def register_client(self, client_id: int, n_samples: int) -> None:
        """Register a client with the server."""
        self.registered_clients[client_id] = {
            'n_samples': n_samples,
            'rounds_participated': 0,
            'last_participation': None
        }
    
    def select_clients(self, available_clients: List[int]) -> List[int]:
        """
        Select clients for current round.
        
        Args:
            available_clients: List of available client IDs
            
        Returns:
            Selected client IDs
        """
        n_select = max(
            self.config.min_clients_per_round,
            int(len(available_clients) * self.config.client_fraction)
        )
        n_select = min(n_select, len(available_clients))
        
        # Random selection
        selected = np.random.choice(
            available_clients, 
            size=n_select, 
            replace=False
        ).tolist()
        
        return selected
    
    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state for distribution to clients."""
        return {
            name: param.clone()
            for name, param in self.global_state.items()
        }
    
    def aggregate_updates(self, 
                         client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                         ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: List of (client_id, model_state, n_samples)
            
        Returns:
            Aggregated model state
        """
        strategy = self.config.aggregation_strategy
        
        if strategy == 'fedavg':
            aggregated = self._fedavg(client_updates)
        elif strategy == 'fedavg_momentum':
            aggregated = self._fedavg_momentum(client_updates)
        elif strategy == 'median':
            aggregated = self._median_aggregation(client_updates)
        elif strategy == 'trimmed_mean':
            aggregated = self._trimmed_mean(client_updates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
        
        # Add DP noise if configured
        if self.config.add_dp_noise:
            aggregated = self._add_dp_noise(aggregated)
        
        return aggregated
    
    def _fedavg(self, 
                client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                ) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (weighted by sample count).
        """
        total_samples = sum(n_samples for _, _, n_samples in client_updates)
        
        aggregated = {}
        for name in client_updates[0][1].keys():
            weighted_sum = torch.zeros_like(client_updates[0][1][name])
            
            for _, state, n_samples in client_updates:
                weight = n_samples / total_samples
                weighted_sum += weight * state[name]
            
            aggregated[name] = weighted_sum
        
        return aggregated
    
    def _fedavg_momentum(self,
                         client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                         ) -> Dict[str, torch.Tensor]:
        """
        FedAvg with server-side momentum.
        """
        # Get vanilla FedAvg result
        fedavg_result = self._fedavg(client_updates)
        
        if self.momentum_buffer is None:
            # Initialize momentum buffer
            self.momentum_buffer = {
                name: torch.zeros_like(param)
                for name, param in fedavg_result.items()
            }
        
        # Calculate update delta and apply momentum
        aggregated = {}
        for name in fedavg_result.keys():
            delta = fedavg_result[name] - self.global_state[name]
            
            # Update momentum
            self.momentum_buffer[name] = (
                self.config.momentum * self.momentum_buffer[name] + delta
            )
            
            # Apply momentum to get new state
            aggregated[name] = self.global_state[name] + self.momentum_buffer[name]
        
        return aggregated
    
    def _median_aggregation(self,
                            client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                            ) -> Dict[str, torch.Tensor]:
        """
        Byzantine-robust aggregation using coordinate-wise median.
        """
        aggregated = {}
        
        for name in client_updates[0][1].keys():
            # Stack all client parameters
            stacked = torch.stack([
                state[name] for _, state, _ in client_updates
            ], dim=0)
            
            # Coordinate-wise median
            aggregated[name] = torch.median(stacked, dim=0).values
        
        return aggregated
    
    def _trimmed_mean(self,
                      client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
                      trim_ratio: float = 0.1
                      ) -> Dict[str, torch.Tensor]:
        """
        Byzantine-robust aggregation using trimmed mean.
        """
        n_clients = len(client_updates)
        n_trim = max(1, int(n_clients * trim_ratio))
        
        aggregated = {}
        
        for name in client_updates[0][1].keys():
            # Stack all client parameters
            stacked = torch.stack([
                state[name] for _, state, _ in client_updates
            ], dim=0)
            
            # Sort and trim
            sorted_stacked, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_stacked[n_trim:-n_trim] if n_trim > 0 else sorted_stacked
            
            # Mean of trimmed values
            aggregated[name] = trimmed.mean(dim=0)
        
        return aggregated
    
    def _add_dp_noise(self, 
                      model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise for differential privacy.
        """
        noisy_state = {}
        
        for name, param in model_state.items():
            noise = torch.normal(
                mean=0,
                std=self.config.dp_noise_multiplier * self.config.dp_sensitivity,
                size=param.shape
            )
            noisy_state[name] = param + noise
        
        return noisy_state
    
    def update_global_model(self, 
                            client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                            ) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates and update global model.
        
        Args:
            client_updates: List of (client_id, model_state, n_samples)
            
        Returns:
            New global model state
        """
        start_time = datetime.now()
        
        # Aggregate
        aggregated = self.aggregate_updates(client_updates)
        
        # Update global state
        self.global_state = aggregated
        self._set_model_state(self.global_model, aggregated)
        
        # Update client participation stats
        for client_id, _, _ in client_updates:
            if client_id in self.registered_clients:
                self.registered_clients[client_id]['rounds_participated'] += 1
                self.registered_clients[client_id]['last_participation'] = self.current_round
        
        aggregation_time = (datetime.now() - start_time).total_seconds()
        
        return aggregated
    
    def evaluate(self, 
                 X_val: np.ndarray, 
                 y_val: np.ndarray) -> Dict[str, float]:
        """
        Evaluate global model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        
        X_tensor = torch.FloatTensor(X_val)
        y_tensor = torch.FloatTensor(y_val)
        
        with torch.no_grad():
            logits, _ = self.global_model(X_tensor)
            probs = torch.sigmoid(logits).squeeze()
            
            # Loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(), y_tensor
            ).item()
            
            # Predictions
            preds = (probs >= 0.5).float()
            
            # Accuracy
            accuracy = (preds == y_tensor).float().mean().item()
            
            # AUC (approximation)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_val, probs.numpy())
            except:
                auc = accuracy
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc
        }
        
        return metrics
    
    def run_round(self,
                  client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute one round of federated training.
        
        Args:
            client_updates: Updates from participating clients
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Round metrics
        """
        self.current_round += 1
        
        # Aggregate updates
        self.update_global_model(client_updates)
        
        # Evaluate if validation data provided
        metrics = {'round': self.current_round, 'n_clients': len(client_updates)}
        
        if X_val is not None and y_val is not None:
            eval_metrics = self.evaluate(X_val, y_val)
            metrics.update(eval_metrics)
            
            # Update history
            self.history['rounds'].append(self.current_round)
            self.history['val_accuracy'].append(eval_metrics['accuracy'])
            self.history['val_loss'].append(eval_metrics['loss'])
            self.history['n_clients'].append(len(client_updates))
            
            # Early stopping check
            if eval_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = eval_metrics['accuracy']
                self.rounds_without_improvement = 0
            else:
                self.rounds_without_improvement += 1
            
            metrics['best_accuracy'] = self.best_accuracy
            metrics['early_stop'] = self.rounds_without_improvement >= self.config.early_stopping_rounds
            
            # Target accuracy check
            if self.config.target_accuracy and eval_metrics['accuracy'] >= self.config.target_accuracy:
                metrics['target_reached'] = True
        
        # Checkpointing
        if self.current_round % self.config.checkpoint_every == 0:
            metrics['checkpoint'] = True
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save server state checkpoint."""
        checkpoint = {
            'global_state': self.global_state,
            'current_round': self.current_round,
            'best_accuracy': self.best_accuracy,
            'history': self.history,
            'config': self.config,
            'registered_clients': self.registered_clients
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load server state from checkpoint."""
        checkpoint = torch.load(path)
        
        self.global_state = checkpoint['global_state']
        self._set_model_state(self.global_model, self.global_state)
        self.current_round = checkpoint['current_round']
        self.best_accuracy = checkpoint['best_accuracy']
        self.history = checkpoint['history']
        self.registered_clients = checkpoint.get('registered_clients', {})
        
        print(f"Checkpoint loaded: round {self.current_round}, best_acc {self.best_accuracy:.4f}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of federated training."""
        return {
            'total_rounds': self.current_round,
            'best_accuracy': self.best_accuracy,
            'final_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else 0,
            'total_clients': len(self.registered_clients),
            'avg_clients_per_round': np.mean(self.history['n_clients']) if self.history['n_clients'] else 0,
            'aggregation_strategy': self.config.aggregation_strategy,
            'dp_enabled': self.config.add_dp_noise
        }


if __name__ == "__main__":
    # Example usage
    input_dim = 42
    
    # Create server
    config = ServerConfig(
        n_rounds=10,
        aggregation_strategy='fedavg_momentum',
        add_dp_noise=True,
        dp_noise_multiplier=0.1
    )
    
    server = FederatedServer(input_dim, config)
    
    # Simulate client updates
    n_clients = 5
    client_updates = []
    
    for client_id in range(n_clients):
        # Create dummy model state
        model = create_model(input_dim)
        state = {
            name: param.data + torch.randn_like(param.data) * 0.1
            for name, param in model.named_parameters()
        }
        n_samples = np.random.randint(100, 500)
        client_updates.append((client_id, state, n_samples))
        server.register_client(client_id, n_samples)
    
    # Run a round
    X_val = np.random.randn(100, 7, input_dim).astype(np.float32)
    y_val = np.random.randint(0, 2, 100).astype(np.float32)
    
    metrics = server.run_round(client_updates, X_val, y_val)
    
    print("\nRound metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
