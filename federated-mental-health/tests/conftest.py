"""
Shared pytest fixtures and configuration.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


@pytest.fixture
def sample_sequences():
    """Generate sample sequence data."""
    np.random.seed(42)
    
    n_samples = 200
    seq_len = 7
    input_dim = 20
    
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    return X, y


@pytest.fixture
def sample_partitions():
    """Generate sample federated partitions."""
    np.random.seed(42)
    
    partitions = {}
    for i in range(5):
        n_samples = 50 + i * 10
        X = np.random.randn(n_samples, 7, 20).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        partitions[f'client_{i}'] = {'X': X, 'y': y}
    
    return partitions


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def model_config():
    """Default model configuration."""
    return {
        'input_dim': 20,
        'hidden_dim': 128,
        'n_layers': 2,
        'n_heads': 4,
        'dropout': 0.3
    }


@pytest.fixture
def training_config():
    """Default training configuration."""
    return {
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }


@pytest.fixture
def privacy_config():
    """Default privacy configuration."""
    return {
        'target_epsilon': 1.0,
        'target_delta': 1e-5,
        'max_grad_norm': 1.0,
        'noise_multiplier': 1.0
    }


@pytest.fixture
def federated_config():
    """Default federated configuration."""
    return {
        'n_rounds': 3,
        'clients_per_round': 3,
        'local_epochs': 2,
        'batch_size': 32
    }


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(20, 64)
            self.fc2 = nn.Linear(64, 1)
        
        def forward(self, x):
            # Average pool over sequence
            x = x.mean(dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x), None
    
    return MockModel()


@pytest.fixture
def predictions_binary():
    """Generate binary classification predictions."""
    np.random.seed(42)
    n = 500
    
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true * 0.6 + np.random.uniform(0, 0.4, n)
    y_pred = np.clip(y_pred, 0, 1)
    
    return y_true, y_pred


@pytest.fixture
def model_states():
    """Generate sample model states for aggregation testing."""
    torch.manual_seed(42)
    
    base_state = {
        'layer1.weight': torch.randn(64, 20),
        'layer1.bias': torch.randn(64),
        'layer2.weight': torch.randn(1, 64),
        'layer2.bias': torch.randn(1)
    }
    
    client_states = {}
    for i in range(5):
        state = {k: v + torch.randn_like(v) * 0.1 for k, v in base_state.items()}
        client_states[f'client_{i}'] = state
    
    return base_state, client_states


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip GPU tests if no GPU available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
