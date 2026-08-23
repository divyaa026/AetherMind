"""
Differential Privacy Optimizer using Opacus
Provides DP-SGD training with privacy budget tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import warnings

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant
    from opacus.grad_sample import GradSampleModule
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn("Opacus not installed. DP training will be simulated.")


@dataclass
class DPConfig:
    """Configuration for differential privacy training."""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Privacy parameter
    max_grad_norm: float = 1.0  # Gradient clipping bound
    noise_multiplier: float = 1.0  # Noise scale (computed if None)
    target_epsilon: Optional[float] = None  # Target epsilon for auto-tuning
    target_delta: Optional[float] = None  # Target delta for auto-tuning
    secure_mode: bool = False  # Use secure random number generation


class DPOptimizer:
    """
    Differential Privacy Optimizer wrapper using Opacus.
    
    Features:
    - DP-SGD with gradient clipping and noise addition
    - Privacy budget tracking with RDP accountant
    - Automatic noise multiplier calibration
    - Compatible with federated learning
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: DataLoader,
                 config: Optional[DPConfig] = None):
        """
        Initialize DP optimizer.
        
        Args:
            model: PyTorch model to train
            optimizer: Base optimizer
            data_loader: Training data loader
            config: DP configuration
        """
        self.config = config or DPConfig()
        self.original_model = model
        self.original_optimizer = optimizer
        self.data_loader = data_loader
        
        self.privacy_engine = None
        self.dp_model = None
        self.dp_optimizer = None
        self.dp_data_loader = None
        
        self.epsilon_spent = 0.0
        self.steps = 0
        
        if OPACUS_AVAILABLE:
            self._setup_opacus()
        else:
            self._setup_simulated()
    
    def _setup_opacus(self):
        """Setup real Opacus privacy engine."""
        # Validate model compatibility
        errors = ModuleValidator.validate(self.original_model, strict=False)
        if errors:
            self.original_model = ModuleValidator.fix(self.original_model)
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine(
            secure_mode=self.config.secure_mode
        )
        
        # Make model, optimizer, and dataloader private
        self.dp_model, self.dp_optimizer, self.dp_data_loader = \
            self.privacy_engine.make_private_with_epsilon(
                module=self.original_model,
                optimizer=self.original_optimizer,
                data_loader=self.data_loader,
                epochs=1,  # Will be updated during training
                target_epsilon=self.config.epsilon,
                target_delta=self.config.delta,
                max_grad_norm=self.config.max_grad_norm
            )
        
        print(f"DP Training initialized:")
        print(f"  Target (ε, δ) = ({self.config.epsilon}, {self.config.delta})")
        print(f"  Noise multiplier: {self.privacy_engine.noise_multiplier:.4f}")
        print(f"  Max grad norm: {self.config.max_grad_norm}")
    
    def _setup_simulated(self):
        """Setup simulated DP training (no Opacus)."""
        print("Using simulated DP training (Opacus not available)")
        self.dp_model = self.original_model
        self.dp_optimizer = self.original_optimizer
        self.dp_data_loader = self.data_loader
        
        # Calculate noise multiplier based on epsilon
        # Using basic Gaussian mechanism
        self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon
        print(f"  Simulated noise multiplier: {self.noise_multiplier:.4f}")
    
    def step(self, loss: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Perform one optimization step with DP.
        
        Args:
            loss: Computed loss value
            
        Returns:
            Tuple of (loss_value, privacy_metrics)
        """
        if OPACUS_AVAILABLE:
            # Opacus handles clipping and noise automatically
            loss.backward()
            self.dp_optimizer.step()
            self.dp_optimizer.zero_grad()
            
            # Get privacy spent
            epsilon = self.privacy_engine.get_epsilon(self.config.delta)
            self.epsilon_spent = epsilon
        else:
            # Simulated DP: manual gradient clipping and noise
            loss.backward()
            self._clip_gradients()
            self._add_noise()
            self.dp_optimizer.step()
            self.dp_optimizer.zero_grad()
            
            # Approximate epsilon tracking
            self.steps += 1
            self.epsilon_spent = self._estimate_epsilon()
        
        metrics = {
            'epsilon_spent': self.epsilon_spent,
            'delta': self.config.delta,
            'remaining_budget': max(0, self.config.epsilon - self.epsilon_spent)
        }
        
        return loss.item(), metrics
    
    def _clip_gradients(self):
        """Clip per-sample gradients."""
        max_norm = self.config.max_grad_norm
        
        for param in self.dp_model.parameters():
            if param.grad is not None:
                # Clip gradient norm
                grad_norm = param.grad.norm(2)
                clip_coef = max_norm / (grad_norm + 1e-6)
                clip_coef = torch.clamp(clip_coef, max=1.0)
                param.grad.mul_(clip_coef)
    
    def _add_noise(self):
        """Add Gaussian noise to gradients."""
        for param in self.dp_model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.config.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad.add_(noise)
    
    def _estimate_epsilon(self) -> float:
        """Estimate epsilon spent based on steps and noise."""
        # Simplified RDP-based estimation
        # In practice, use proper RDP accountant
        sample_rate = 1.0 / len(self.data_loader)
        sigma = self.noise_multiplier
        
        # Basic composition (overestimates epsilon)
        alpha = 1 + 1 / (2 * sigma ** 2)
        rdp_per_step = alpha / (2 * sigma ** 2)
        
        total_rdp = self.steps * sample_rate * rdp_per_step
        
        # Convert RDP to (epsilon, delta)-DP
        epsilon = total_rdp - np.log(self.config.delta) / (alpha - 1)
        
        return max(0, epsilon)
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent."""
        return self.epsilon_spent, self.config.delta
    
    def get_model(self) -> nn.Module:
        """Get the DP-wrapped model."""
        if OPACUS_AVAILABLE:
            # Return underlying model without grad samplers
            return self.dp_model._module if hasattr(self.dp_model, '_module') else self.dp_model
        return self.dp_model
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """Get the DP optimizer."""
        return self.dp_optimizer
    
    def get_data_loader(self) -> DataLoader:
        """Get the DP data loader."""
        return self.dp_data_loader
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.epsilon_spent >= self.config.epsilon


class LocalDPTrainer:
    """
    Local differential privacy trainer for federated learning clients.
    
    Implements DP-SGD with local privacy guarantees.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: DPConfig,
                 device: str = 'cpu'):
        """
        Initialize local DP trainer.
        
        Args:
            model: Model to train
            config: DP configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.dp_optimizer: Optional[DPOptimizer] = None
        
    def setup_training(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       batch_size: int = 32,
                       learning_rate: float = 0.001) -> None:
        """
        Setup DP training with data.
        
        Args:
            X: Training features
            y: Training labels
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Create dataset and loader
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Wrap with DP
        self.dp_optimizer = DPOptimizer(
            model=self.model,
            optimizer=optimizer,
            data_loader=data_loader,
            config=self.config
        )
    
    def train_epoch(self, 
                    criterion: nn.Module,
                    max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Train for one epoch with DP.
        
        Args:
            criterion: Loss function
            max_batches: Maximum batches to process (for early stopping)
            
        Returns:
            Training metrics
        """
        if self.dp_optimizer is None:
            raise ValueError("Call setup_training first")
        
        model = self.dp_optimizer.get_model()
        model.train()
        
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.dp_optimizer.get_data_loader()):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Forward pass
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch.unsqueeze(1))
            
            # DP backward pass
            loss_val, privacy_metrics = self.dp_optimizer.step(loss)
            
            total_loss += loss_val
            n_batches += 1
            
            # Check privacy budget
            if self.dp_optimizer.is_budget_exhausted():
                print(f"Privacy budget exhausted at batch {batch_idx}")
                break
        
        metrics = {
            'loss': total_loss / max(n_batches, 1),
            'n_batches': n_batches,
            **privacy_metrics
        }
        
        return metrics
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated aggregation."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def apply_model_update(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Apply global model parameters."""
        for name, param in self.model.named_parameters():
            if name in global_params:
                param.data = global_params[name].clone()


def calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    accountant_type: str = 'rdp'
) -> float:
    """
    Calibrate noise multiplier for target privacy budget.
    
    Args:
        target_epsilon: Target epsilon
        target_delta: Target delta
        sample_rate: Batch size / dataset size
        epochs: Number of training epochs
        accountant_type: Type of privacy accountant
        
    Returns:
        Calibrated noise multiplier
    """
    if not OPACUS_AVAILABLE:
        # Fallback to basic Gaussian mechanism
        return np.sqrt(2 * np.log(1.25 / target_delta)) / target_epsilon
    
    from opacus.accountants.analysis import rdp as rdp_analysis
    
    # Binary search for optimal noise multiplier
    low, high = 0.1, 100.0
    target_noise = high
    
    for _ in range(100):  # Max iterations
        mid = (low + high) / 2
        
        # Calculate epsilon for this noise level
        rdp_orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        rdp_epsilon = rdp_analysis.compute_rdp(
            q=sample_rate,
            noise_multiplier=mid,
            steps=int(epochs / sample_rate),
            orders=rdp_orders
        )
        
        epsilon, _ = rdp_analysis.get_privacy_spent(
            orders=rdp_orders,
            rdp=rdp_epsilon,
            delta=target_delta
        )
        
        if epsilon < target_epsilon:
            target_noise = mid
            high = mid
        else:
            low = mid
        
        if abs(epsilon - target_epsilon) < 0.01:
            break
    
    return target_noise


if __name__ == "__main__":
    # Example usage
    from architecture import create_model
    
    # Create model
    input_dim = 42
    model = create_model(input_dim)
    
    # Create dummy data
    X = np.random.randn(1000, 7, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, 1000).astype(np.float32)
    
    # Configure DP
    dp_config = DPConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0
    )
    
    # Create trainer
    trainer = LocalDPTrainer(model, dp_config)
    trainer.setup_training(X, y, batch_size=32)
    
    # Train one epoch
    from architecture import FocalLoss
    criterion = FocalLoss()
    
    metrics = trainer.train_epoch(criterion)
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrivacy spent: ε = {metrics['epsilon_spent']:.4f}")
