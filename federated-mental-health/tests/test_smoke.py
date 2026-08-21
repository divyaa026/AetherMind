"""
Smoke tests for federated learning pipeline.
These tests verify basic functionality works end-to-end.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSmokeTests:
    """Smoke tests for core components."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation works."""
        from data.synthetic_generator import SyntheticMentalHealthData, SyntheticConfig
        
        config = SyntheticConfig(n_users=50, n_days=7)
        generator = SyntheticMentalHealthData(config=config)
        result = generator.generate()
        
        # generate() returns (time_series_df, profiles_df)
        df = result[0] if isinstance(result, tuple) else result
        
        assert len(df) > 0
        assert 'user_id' in df.columns
        print(f"[OK] Generated {len(df)} samples for {config.n_users} users")
    
    def test_model_creation(self):
        """Test model architecture creation."""
        from models.architecture import create_model
        
        model = create_model(input_dim=10)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(4, 7, 10)  # batch=4, seq_len=7, features=10
        with torch.no_grad():
            logits, attention = model(x)
        
        assert logits.shape == (4, 1)
        print(f"[OK] Model forward pass works: input {x.shape} -> output {logits.shape}")
    
    def test_federated_server_creation(self):
        """Test federated server can be created."""
        from federated.server import FederatedServer, ServerConfig
        
        config = ServerConfig(
            n_rounds=5,
            min_clients_per_round=2,
            aggregation_strategy='fedavg'
        )
        server = FederatedServer(input_dim=10, config=config)
        
        assert server is not None
        assert server.global_model is not None
        print("[OK] Federated server created successfully")
    
    def test_federated_client_creation(self):
        """Test federated client can be created."""
        from federated.client import FederatedClient, ClientConfig
        
        # Create dummy data
        X = np.random.randn(100, 7, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        
        config = ClientConfig(client_id=1, local_epochs=1, batch_size=16)
        client = FederatedClient(
            client_id=1,
            X_train=X,
            y_train=y,
            config=config
        )
        
        assert client is not None
        print(f"[OK] Federated client created with {len(X)} samples")
    
    def test_aggregation_methods(self):
        """Test aggregation methods work."""
        from federated.aggregation import ModelAggregator, AggregationConfig, AggregationMethod
        
        config = AggregationConfig(method=AggregationMethod.FEDAVG)
        aggregator = ModelAggregator(config)
        
        # Create mock client updates
        client_updates = []
        for i in range(3):
            state = {
                'layer1.weight': torch.randn(10, 10),
                'layer1.bias': torch.randn(10),
            }
            client_updates.append((i, state, 100 + i * 10))
        
        aggregated = aggregator.aggregate(client_updates)
        
        assert 'layer1.weight' in aggregated
        assert 'layer1.bias' in aggregated
        print("[OK] FedAvg aggregation works")
    
    def test_dp_accountant(self):
        """Test RDP accountant works."""
        from privacy.dp_accounting import RDPAccountant
        
        accountant = RDPAccountant(target_delta=1e-5)
        
        # Compute RDP for a single step
        rdp = accountant.compute_rdp_single_step(
            noise_multiplier=1.0,
            sampling_probability=0.01
        )
        
        assert rdp is not None
        assert len(rdp) > 0
        print(f"[OK] RDP accountant works, computed {len(rdp)} RDP values")
    
    def test_metrics_calculator(self):
        """Test metrics calculation works."""
        from evaluation.metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4])
        
        metrics = calculator.compute_classification_metrics(y_true, y_prob)
        
        assert metrics.accuracy >= 0
        assert metrics.auc_roc >= 0
        print(f"[OK] Metrics: accuracy={metrics.accuracy:.3f}, AUC={metrics.auc_roc:.3f}")
    
    def test_local_training_step(self):
        """Test local training works."""
        from models.architecture import create_model
        from models.train_local import LocalTrainer, TrainingConfig
        
        model = create_model(input_dim=10)
        config = TrainingConfig(epochs=1, batch_size=16, use_dp=False)
        trainer = LocalTrainer(model, config)
        
        X = np.random.randn(50, 7, 10).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        
        history = trainer.fit(X, y)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
        print(f"[OK] Local training works, final loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
