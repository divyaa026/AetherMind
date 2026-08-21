"""
Tests for federated learning components.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated.server import FederatedServer, ServerConfig
from federated.client import FederatedClient, ClientConfig, ClientManager
from federated.coordinator import FederatedCoordinator, FederatedConfig
from federated.aggregation import ModelAggregator, AggregationConfig, AggregationMethod


class TestFederatedServer:
    """Tests for federated server."""
    
    @pytest.fixture
    def server(self):
        """Create a federated server."""
        config = ServerConfig(
            min_clients_per_round=2,
            n_rounds=3,
            aggregation_strategy='fedavg'
        )
        return FederatedServer(input_dim=20, config=config)
    
    def test_server_creation(self, server):
        """Test server initialization."""
        assert server is not None
        assert server.global_model is not None
        assert server.current_round == 0
    
    def test_get_global_model(self, server):
        """Test getting global model state."""
        state = server.get_global_model_state()
        
        assert isinstance(state, dict)
        assert len(state) > 0
    
    def test_select_clients(self, server):
        """Test client selection."""
        available = [f'client_{i}' for i in range(10)]
        
        selected = server.select_clients(available, n_clients=5)
        
        assert len(selected) == 5
        assert all(c in available for c in selected)
    
    def test_aggregate_updates(self, server):
        """Test model aggregation."""
        # Get model state shape
        global_state = server.get_global_model_state()
        
        # Create mock client updates
        client_updates = {}
        for i in range(3):
            update = {}
            for key, value in global_state.items():
                update[key] = value + torch.randn_like(value) * 0.1
            client_updates[f'client_{i}'] = {
                'model_state': update,
                'n_samples': 100 + i * 10,
                'metrics': {'loss': 0.5 - i * 0.1}
            }
        
        # Aggregate
        server.aggregate_updates(client_updates)
        
        # Model should have been updated
        assert server.current_round == 1
    
    def test_checkpoint_save_load(self, server):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            
            # Save checkpoint
            server.save_checkpoint(checkpoint_path)
            
            # Create new server and load
            config = ServerConfig(min_clients_per_round=2, n_rounds=3)
            new_server = FederatedServer(input_dim=20, config=config)
            new_server.load_checkpoint(checkpoint_path)
            
            # States should match
            for key in server.get_global_model_state():
                assert torch.equal(
                    server.get_global_model_state()[key],
                    new_server.get_global_model_state()[key]
                )


class TestFederatedClient:
    """Tests for federated client."""
    
    @pytest.fixture
    def client_data(self):
        """Generate client data."""
        np.random.seed(42)
        n_samples = 100
        seq_len = 7
        input_dim = 20
        
        X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)
        
        return X, y
    
    @pytest.fixture
    def client(self, client_data):
        """Create a federated client."""
        X, y = client_data
        config = ClientConfig(
            client_id='test_client',
            local_epochs=2,
            batch_size=32
        )
        return FederatedClient(X, y, input_dim=20, config=config)
    
    def test_client_creation(self, client):
        """Test client initialization."""
        assert client is not None
        assert client.config.client_id == 'test_client'
    
    def test_local_training(self, client):
        """Test local training."""
        # Get initial model state
        global_state = client.get_model_state()
        
        # Train
        update = client.train(global_state)
        
        assert 'model_state' in update
        assert 'n_samples' in update
        assert 'metrics' in update
    
    def test_multiple_rounds(self, client):
        """Test multiple training rounds."""
        state = client.get_model_state()
        
        losses = []
        for round_num in range(3):
            update = client.train(state)
            losses.append(update['metrics']['loss'])
            state = update['model_state']
        
        # Training should generally reduce loss
        assert len(losses) == 3


class TestClientManager:
    """Tests for client manager."""
    
    @pytest.fixture
    def partitions(self):
        """Create sample data partitions."""
        np.random.seed(42)
        partitions = {}
        
        for i in range(5):
            n_samples = 50 + i * 10
            X = np.random.randn(n_samples, 7, 20).astype(np.float32)
            y = np.random.randint(0, 2, n_samples).astype(np.float32)
            partitions[f'client_{i}'] = {'X': X, 'y': y}
        
        return partitions
    
    def test_client_manager_creation(self, partitions):
        """Test client manager initialization."""
        manager = ClientManager(partitions, input_dim=20)
        
        assert len(manager.clients) == 5
    
    def test_get_client_ids(self, partitions):
        """Test getting client IDs."""
        manager = ClientManager(partitions, input_dim=20)
        ids = manager.get_client_ids()
        
        assert len(ids) == 5
        assert all(f'client_{i}' in ids for i in range(5))
    
    def test_train_clients(self, partitions):
        """Test training multiple clients."""
        manager = ClientManager(partitions, input_dim=20)
        
        client_ids = ['client_0', 'client_1', 'client_2']
        global_state = manager.clients['client_0'].get_model_state()
        
        updates = manager.train_clients(client_ids, global_state)
        
        assert len(updates) == 3
        for client_id in client_ids:
            assert client_id in updates


class TestModelAggregator:
    """Tests for model aggregation strategies."""
    
    @pytest.fixture
    def model_updates(self):
        """Create sample model updates."""
        torch.manual_seed(42)
        
        # Create base model state
        base_state = {
            'layer1.weight': torch.randn(64, 20),
            'layer1.bias': torch.randn(64),
            'layer2.weight': torch.randn(1, 64),
            'layer2.bias': torch.randn(1)
        }
        
        # Create client updates
        updates = {}
        for i in range(5):
            client_state = {}
            for key, value in base_state.items():
                client_state[key] = value + torch.randn_like(value) * 0.1
            updates[f'client_{i}'] = {
                'model_state': client_state,
                'n_samples': 100 + i * 20
            }
        
        return updates, base_state
    
    def test_fedavg_aggregation(self, model_updates):
        """Test FedAvg aggregation."""
        updates, global_state = model_updates
        
        config = AggregationConfig(method=AggregationMethod.FEDAVG)
        aggregator = ModelAggregator(config)
        
        result = aggregator.aggregate(updates, global_state)
        
        assert 'model_state' in result
        for key in global_state:
            assert key in result['model_state']
    
    def test_fedavg_momentum(self, model_updates):
        """Test FedAvg with momentum."""
        updates, global_state = model_updates
        
        config = AggregationConfig(
            method=AggregationMethod.FEDAVG_MOMENTUM,
            momentum=0.9
        )
        aggregator = ModelAggregator(config)
        
        # First aggregation
        result1 = aggregator.aggregate(updates, global_state)
        
        # Second aggregation should use momentum
        result2 = aggregator.aggregate(updates, result1['model_state'])
        
        assert 'model_state' in result2
    
    def test_median_aggregation(self, model_updates):
        """Test median aggregation (Byzantine-robust)."""
        updates, global_state = model_updates
        
        config = AggregationConfig(method=AggregationMethod.MEDIAN)
        aggregator = ModelAggregator(config)
        
        result = aggregator.aggregate(updates, global_state)
        
        assert 'model_state' in result
    
    def test_trimmed_mean_aggregation(self, model_updates):
        """Test trimmed mean aggregation."""
        updates, global_state = model_updates
        
        config = AggregationConfig(
            method=AggregationMethod.TRIMMED_MEAN,
            trim_ratio=0.2
        )
        aggregator = ModelAggregator(config)
        
        result = aggregator.aggregate(updates, global_state)
        
        assert 'model_state' in result
    
    def test_krum_aggregation(self, model_updates):
        """Test Krum aggregation."""
        updates, global_state = model_updates
        
        config = AggregationConfig(
            method=AggregationMethod.KRUM,
            n_byzantine=1
        )
        aggregator = ModelAggregator(config)
        
        result = aggregator.aggregate(updates, global_state)
        
        assert 'model_state' in result
    
    def test_byzantine_resilience(self, model_updates):
        """Test Byzantine-robust aggregation with malicious updates."""
        updates, global_state = model_updates
        
        # Add malicious client with very large updates
        malicious_state = {}
        for key, value in global_state.items():
            malicious_state[key] = value * 100  # Very large
        updates['malicious'] = {
            'model_state': malicious_state,
            'n_samples': 100
        }
        
        # Median should be robust
        config = AggregationConfig(method=AggregationMethod.MEDIAN)
        aggregator = ModelAggregator(config)
        
        result = aggregator.aggregate(updates, global_state)
        
        # Result should not be dominated by malicious update
        for key in global_state:
            assert not torch.allclose(
                result['model_state'][key], 
                malicious_state[key],
                atol=1.0
            )


class TestFederatedCoordinator:
    """Tests for federated coordinator."""
    
    @pytest.fixture
    def partitions(self):
        """Create data partitions."""
        np.random.seed(42)
        partitions = {}
        
        for i in range(3):
            n_samples = 50
            X = np.random.randn(n_samples, 7, 20).astype(np.float32)
            y = np.random.randint(0, 2, n_samples).astype(np.float32)
            partitions[f'client_{i}'] = {'X': X, 'y': y}
        
        return partitions
    
    @pytest.fixture
    def test_data(self):
        """Create test data."""
        np.random.seed(123)
        X = np.random.randn(30, 7, 20).astype(np.float32)
        y = np.random.randint(0, 2, 30).astype(np.float32)
        return X, y
    
    def test_coordinator_creation(self, partitions, test_data):
        """Test coordinator initialization."""
        X_test, y_test = test_data
        
        config = FederatedConfig(
            n_rounds=2,
            clients_per_round=2
        )
        
        coordinator = FederatedCoordinator(
            partitions=partitions,
            test_data=(X_test, y_test),
            input_dim=20,
            config=config
        )
        
        assert coordinator is not None
    
    def test_run_federated_training(self, partitions, test_data):
        """Test full federated training."""
        X_test, y_test = test_data
        
        config = FederatedConfig(
            n_rounds=2,
            clients_per_round=2,
            local_epochs=1
        )
        
        coordinator = FederatedCoordinator(
            partitions=partitions,
            test_data=(X_test, y_test),
            input_dim=20,
            config=config
        )
        
        results = coordinator.train()
        
        assert 'history' in results
        assert 'final_metrics' in results
        assert len(results['history']) == 2  # 2 rounds


class TestAggregationWeighting:
    """Tests for weighted aggregation."""
    
    def test_sample_weighted_average(self):
        """Test that sample weighting works correctly."""
        torch.manual_seed(42)
        
        # Create updates with known weights
        updates = {
            'client_1': {
                'model_state': {'w': torch.ones(10)},
                'n_samples': 100
            },
            'client_2': {
                'model_state': {'w': torch.zeros(10)},
                'n_samples': 100
            }
        }
        
        config = AggregationConfig(method=AggregationMethod.FEDAVG)
        aggregator = ModelAggregator(config)
        
        global_state = {'w': torch.zeros(10)}
        result = aggregator.aggregate(updates, global_state)
        
        # With equal samples, average should be 0.5
        expected = torch.ones(10) * 0.5
        assert torch.allclose(result['model_state']['w'], expected, atol=0.01)
    
    def test_unequal_weights(self):
        """Test aggregation with unequal sample sizes."""
        torch.manual_seed(42)
        
        updates = {
            'client_1': {
                'model_state': {'w': torch.ones(10)},
                'n_samples': 300  # 3x the samples
            },
            'client_2': {
                'model_state': {'w': torch.zeros(10)},
                'n_samples': 100
            }
        }
        
        config = AggregationConfig(method=AggregationMethod.FEDAVG)
        aggregator = ModelAggregator(config)
        
        global_state = {'w': torch.zeros(10)}
        result = aggregator.aggregate(updates, global_state)
        
        # Weighted average: (300*1 + 100*0) / 400 = 0.75
        expected = torch.ones(10) * 0.75
        assert torch.allclose(result['model_state']['w'], expected, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
