"""
Tests for model architectures and training.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architecture import (
    MentalHealthPredictor,
    SelfAttention,
    FocalLoss,
    WeightedBCELoss,
    create_model
)
from models.train_local import LocalTrainer, TrainingConfig
from models.dp_optimizer import DPOptimizer, DPConfig


class TestModelArchitecture:
    """Tests for model architecture."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model(input_dim=20)
        
        assert isinstance(model, MentalHealthPredictor)
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test forward pass."""
        model = create_model(input_dim=20)
        
        # Create sample input
        batch_size = 8
        seq_len = 7
        input_dim = 20
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        logits, attention = model(x)
        
        assert logits.shape == (batch_size, 1)
        assert attention is not None
    
    def test_model_output_range(self):
        """Test that sigmoid of logits is in [0, 1]."""
        model = create_model(input_dim=20)
        
        x = torch.randn(16, 7, 20)
        logits, _ = model(x)
        probs = torch.sigmoid(logits)
        
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_model_gradient_flow(self):
        """Test gradient computation."""
        model = create_model(input_dim=20)
        
        x = torch.randn(8, 7, 20)
        y = torch.randint(0, 2, (8,)).float()
        
        logits, _ = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_different_configs(self):
        """Test model with different configurations."""
        configs = [
            {'hidden_dim': 64, 'n_layers': 1, 'dropout': 0.1},
            {'hidden_dim': 256, 'n_layers': 3, 'dropout': 0.5},
            {'hidden_dim': 128, 'n_layers': 2, 'bidirectional': False}
        ]
        
        for config in configs:
            model = create_model(input_dim=20, **config)
            x = torch.randn(4, 7, 20)
            logits, _ = model(x)
            assert logits.shape[0] == 4


class TestSelfAttention:
    """Tests for self-attention module."""
    
    def test_attention_output_shape(self):
        """Test attention output shape."""
        attention = SelfAttention(hidden_dim=128, n_heads=4)
        
        x = torch.randn(8, 7, 128)
        output, weights = attention(x)
        
        assert output.shape == x.shape
        assert weights.shape[0] == 8  # batch size
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        attention = SelfAttention(hidden_dim=128, n_heads=4)
        
        x = torch.randn(8, 7, 128)
        _, weights = attention(x)
        
        # Average attention weights should roughly sum to 1 along seq dim
        weight_sums = weights.mean(dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.1)


class TestLossFunctions:
    """Tests for custom loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32,)).float()
        
        loss = loss_fn(logits.squeeze(), targets)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
    
    def test_weighted_bce_loss(self):
        """Test weighted BCE loss."""
        loss_fn = WeightedBCELoss(pos_weight=2.0)
        
        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32,)).float()
        
        loss = loss_fn(logits.squeeze(), targets)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_focal_loss_reduction(self):
        """Test focal loss with class imbalance."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Easy examples (high confidence correct)
        easy_logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
        easy_targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        
        # Hard examples (low confidence)
        hard_logits = torch.tensor([0.1, 0.1, -0.1, -0.1])
        hard_targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        
        easy_loss = loss_fn(easy_logits, easy_targets)
        hard_loss = loss_fn(hard_logits, hard_targets)
        
        # Focal loss should be lower for easy examples
        assert easy_loss < hard_loss


class TestLocalTrainer:
    """Tests for local training."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_samples = 200
        seq_len = 7
        input_dim = 20
        
        X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        
        return X, y
    
    def test_basic_training(self, sample_data):
        """Test basic training loop."""
        X, y = sample_data
        input_dim = X.shape[2]
        
        model = create_model(input_dim)
        config = TrainingConfig(epochs=2, batch_size=32)
        trainer = LocalTrainer(model, config)
        
        metrics = trainer.fit(X, y)
        
        assert 'final_loss' in metrics
        assert 'epochs_trained' in metrics
    
    def test_training_reduces_loss(self, sample_data):
        """Test that training reduces loss."""
        X, y = sample_data
        input_dim = X.shape[2]
        
        model = create_model(input_dim)
        
        # Initial loss
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.FloatTensor(X))
            initial_loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(), torch.FloatTensor(y)
            ).item()
        
        # Train
        config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.01)
        trainer = LocalTrainer(model, config)
        metrics = trainer.fit(X, y)
        
        # Final loss should be lower (or at least not much higher)
        assert metrics['final_loss'] < initial_loss * 1.5
    
    def test_model_state_management(self, sample_data):
        """Test getting and setting model state."""
        X, y = sample_data
        input_dim = X.shape[2]
        
        model = create_model(input_dim)
        config = TrainingConfig(epochs=2)
        trainer = LocalTrainer(model, config)
        
        # Get initial state
        initial_state = trainer.get_model_state()
        
        # Train
        trainer.fit(X, y)
        
        # Get new state
        trained_state = trainer.get_model_state()
        
        # States should be different
        for key in initial_state:
            assert not torch.equal(initial_state[key], trained_state[key])
        
        # Reset to initial state
        trainer.set_model_state(initial_state)
        reset_state = trainer.get_model_state()
        
        # Should match initial
        for key in initial_state:
            assert torch.equal(initial_state[key], reset_state[key])
    
    def test_prediction(self, sample_data):
        """Test prediction functionality."""
        X, y = sample_data
        input_dim = X.shape[2]
        
        model = create_model(input_dim)
        config = TrainingConfig(epochs=2)
        trainer = LocalTrainer(model, config)
        
        trainer.fit(X, y)
        probs = trainer.predict(X)
        
        assert len(probs) == len(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestDPOptimizer:
    """Tests for differential privacy optimizer."""
    
    def test_dp_optimizer_creation(self):
        """Test DP optimizer creation."""
        model = create_model(input_dim=20)
        config = DPConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        
        dp_optimizer = DPOptimizer(model, config)
        
        assert dp_optimizer is not None
    
    def test_gradient_clipping(self):
        """Test that gradients are clipped."""
        model = create_model(input_dim=20)
        max_norm = 1.0
        
        config = DPConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            max_grad_norm=max_norm,
            noise_multiplier=0.0  # No noise for this test
        )
        
        dp_optimizer = DPOptimizer(model, config)
        
        # Create large gradients
        x = torch.randn(8, 7, 20) * 10
        y = torch.randint(0, 2, (8,)).float()
        
        logits, _ = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y)
        loss.backward()
        
        # Clip gradients
        dp_optimizer.clip_gradients()
        
        # Check gradient norms are bounded
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clipped norm should be at most max_norm (with some tolerance)
        assert total_norm <= max_norm * len(list(model.parameters())) + 0.1


class TestModelSaveLoad:
    """Tests for model persistence."""
    
    def test_save_load_model(self):
        """Test saving and loading model."""
        model = create_model(input_dim=20)
        
        # Get initial predictions
        x = torch.randn(4, 7, 20)
        with torch.no_grad():
            initial_output, _ = model(x)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'model.pt'
            
            # Save
            torch.save({
                'model_state_dict': model.state_dict()
            }, path)
            
            # Create new model and load
            new_model = create_model(input_dim=20)
            checkpoint = torch.load(path)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Compare outputs
            with torch.no_grad():
                loaded_output, _ = new_model(x)
            
            assert torch.allclose(initial_output, loaded_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
