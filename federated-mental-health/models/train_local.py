"""
Local Training Module for Federated Learning Clients
Handles training loop, evaluation, and model updates.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

from .architecture import MentalHealthPredictor, FocalLoss, WeightedBCELoss, create_model
from .dp_optimizer import DPConfig, LocalDPTrainer


@dataclass
class TrainingConfig:
    """Configuration for local training."""
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_dp: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    max_grad_norm: float = 1.0
    loss_type: str = 'focal'  # 'focal', 'weighted_bce', 'bce'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    pos_weight: float = 3.0  # For weighted BCE
    early_stopping_patience: int = 3
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'none'
    gradient_accumulation_steps: int = 1
    device: str = 'auto'


class LocalTrainer:
    """
    Local training for federated learning clients.
    
    Features:
    - Differential privacy support
    - Multiple loss functions for imbalanced data
    - Learning rate scheduling
    - Gradient accumulation
    - Early stopping
    - Model evaluation and metrics
    """
    
    def __init__(self, 
                 model: MentalHealthPredictor,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize local trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Setup device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        self.model = model.to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.dp_trainer: Optional[LocalDPTrainer] = None
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self._setup_criterion()
    
    def _setup_criterion(self):
        """Setup loss function based on config."""
        if self.config.loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma
            )
        elif self.config.loss_type == 'weighted_bce':
            self.criterion = WeightedBCELoss(pos_weight=self.config.pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.config.epochs // 3),
                gamma=0.5
            )
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray, 
                            shuffle: bool = True) -> DataLoader:
        """Create PyTorch data loader from numpy arrays."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle
        )
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train model on local data.
        
        Args:
            X_train: Training features (n_samples, seq_len, n_features)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history and metrics
        """
        self._setup_optimizer()
        
        if self.config.use_dp:
            return self._fit_with_dp(X_train, y_train, X_val, y_val)
        else:
            return self._fit_standard(X_train, y_train, X_val, y_val)
    
    def _fit_standard(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: Optional[np.ndarray],
                      y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """Standard training without DP."""
        train_loader = self._create_data_loader(X_train, y_train)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False) if X_val is not None else None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                val_loss = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                self.val_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Progress
            if (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                val_str = f", val_loss: {val_loss:.4f}" if val_loader else ""
                print(f"Epoch {epoch + 1}/{self.config.epochs} - train_loss: {train_loss:.4f}{val_str}")
        
        return history
    
    def _fit_with_dp(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: Optional[np.ndarray],
                     y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """Training with differential privacy."""
        dp_config = DPConfig(
            epsilon=self.config.dp_epsilon,
            delta=self.config.dp_delta,
            max_grad_norm=self.config.max_grad_norm
        )
        
        self.dp_trainer = LocalDPTrainer(self.model, dp_config, str(self.device))
        self.dp_trainer.setup_training(
            X_train, y_train,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate
        )
        
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False) if X_val is not None else None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'epsilon_spent': [],
            'privacy_budget_remaining': []
        }
        
        for epoch in range(self.config.epochs):
            # DP Training
            metrics = self.dp_trainer.train_epoch(self.criterion)
            
            history['train_loss'].append(metrics['loss'])
            history['epsilon_spent'].append(metrics['epsilon_spent'])
            history['privacy_budget_remaining'].append(metrics['remaining_budget'])
            self.train_losses.append(metrics['loss'])
            
            # Validation
            if val_loader:
                val_loss = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                self.val_losses.append(val_loss)
            
            # Progress
            if (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                val_str = f", val_loss: {val_loss:.4f}" if val_loader else ""
                print(f"Epoch {epoch + 1}/{self.config.epochs} - loss: {metrics['loss']:.4f}{val_str}, "
                      f"ε: {metrics['epsilon_spent']:.4f}")
            
            # Check privacy budget
            if metrics['remaining_budget'] <= 0:
                print("Privacy budget exhausted")
                break
        
        # Update model reference
        self.model = self.dp_trainer.model
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        accumulation_steps = self.config.gradient_accumulation_steps
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Forward pass
            logits, _ = self.model(X_batch)
            loss = self.criterion(logits, y_batch.unsqueeze(1))
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            n_batches += 1
        
        return total_loss / n_batches
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on data."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch.unsqueeze(1))
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for data."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return probs.squeeze()
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get class predictions."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get model state dict for aggregation."""
        return {
            name: param.data.clone().cpu()
            for name, param in self.model.named_parameters()
        }
    
    def set_model_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set model state from aggregated parameters."""
        for name, param in self.model.named_parameters():
            if name in state:
                param.data = state[name].to(self.device)
    
    def get_sample_count(self) -> int:
        """Get number of training samples (for weighted aggregation)."""
        return len(self.train_losses) * self.config.batch_size if self.train_losses else 0
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))


def train_local_model(
    client_data_dir: str,
    model_config: Optional[Dict] = None,
    training_config: Optional[TrainingConfig] = None,
    output_dir: Optional[str] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Train model on local client data.
    
    Args:
        client_data_dir: Directory with client data (X.npy, y.npy)
        model_config: Model configuration
        training_config: Training configuration
        output_dir: Optional output directory for checkpoints
        
    Returns:
        Tuple of (model_state, training_metrics)
    """
    data_path = Path(client_data_dir)
    
    # Load data
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")
    
    print(f"Loaded client data: {X.shape}")
    
    # Create model
    input_dim = X.shape[-1]
    model = create_model(input_dim, model_config)
    
    # Create trainer
    trainer = LocalTrainer(model, training_config)
    
    # Split into train/val
    n_val = max(1, int(len(X) * 0.1))
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    
    # Train
    history = trainer.fit(X_train, y_train, X_val, y_val)
    
    # Get model state
    model_state = trainer.get_model_state()
    
    # Compute metrics
    metrics = {
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else 0,
        'n_samples': len(X_train),
        'n_epochs_trained': len(history['train_loss'])
    }
    
    if training_config and training_config.use_dp:
        metrics['epsilon_spent'] = history['epsilon_spent'][-1] if history.get('epsilon_spent') else 0
    
    # Save checkpoint if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(output_path / "checkpoint.pt"))
        
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return model_state, metrics


if __name__ == "__main__":
    # Example usage
    from architecture import create_model
    
    # Create dummy data
    n_samples = 500
    seq_len = 7
    input_dim = 42
    
    X_train = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    y_train = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    X_val = np.random.randn(100, seq_len, input_dim).astype(np.float32)
    y_val = np.random.randint(0, 2, 100).astype(np.float32)
    
    # Create model and trainer
    model = create_model(input_dim)
    config = TrainingConfig(
        epochs=5,
        use_dp=False,
        loss_type='focal'
    )
    trainer = LocalTrainer(model, config)
    
    # Train
    history = trainer.fit(X_train, y_train, X_val, y_val)
    
    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Test predictions
    preds = trainer.predict(X_val)
    print(f"Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
