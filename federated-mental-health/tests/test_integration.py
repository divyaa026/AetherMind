"""
End-to-end integration tests for the federated learning pipeline.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    def test_synthetic_data_to_training(self, temp_dir):
        """Test from synthetic data generation to model training."""
        from data.synthetic_generator import SyntheticDataGenerator
        from data.preprocessing import Preprocessor
        from data.federated_partitioner import FederatedPartitioner
        from models.architecture import create_model
        
        # Phase 1: Generate synthetic data
        generator = SyntheticDataGenerator(
            n_users=100,
            days_per_user=14,
            seed=42
        )
        data = generator.generate()
        
        assert 'user_id' in data.columns
        assert len(data) == 100 * 14  # n_users * days
        
        # Phase 2: Preprocess
        preprocessor = Preprocessor(sequence_length=7)
        X, y = preprocessor.fit_transform(data)
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 7  # sequence_length
        
        # Phase 3: Partition for federated learning
        partitioner = FederatedPartitioner(n_clients=5)
        partitions = partitioner.partition(X, y)
        
        assert len(partitions) == 5
        
        # Phase 4: Create model
        input_dim = X.shape[2]
        model = create_model(input_dim=input_dim)
        
        # Verify model works
        sample_x = torch.FloatTensor(partitions['client_0']['X'][:8])
        with torch.no_grad():
            logits, attention = model(sample_x)
        
        assert logits.shape[0] == 8
    
    def test_federated_training_round(self, sample_partitions, temp_dir):
        """Test a complete federated training round."""
        from federated.server import FederatedServer, ServerConfig
        from federated.client import ClientManager
        
        # Create server
        config = ServerConfig(
            min_clients_per_round=2,
            n_rounds=3,
            aggregation_strategy='fedavg'
        )
        server = FederatedServer(input_dim=20, config=config)
        
        # Create clients
        client_manager = ClientManager(sample_partitions, input_dim=20)
        
        # Select clients for round
        all_clients = client_manager.get_client_ids()
        selected = server.select_clients(all_clients, n_clients=3)
        
        # Get global model
        global_state = server.get_global_model_state()
        
        # Train clients
        updates = client_manager.train_clients(selected, global_state)
        
        # Aggregate
        server.aggregate_updates(updates)
        
        # Verify round completed
        assert server.current_round == 1
    
    def test_privacy_integration(self, sample_sequences, temp_dir):
        """Test DP integration with training."""
        from models.architecture import create_model
        from models.dp_optimizer import DPOptimizer, DPConfig
        from privacy.dp_accounting import RDPAccountant
        
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        # Create model
        model = create_model(input_dim=input_dim)
        
        # Create DP optimizer
        dp_config = DPConfig(
            target_epsilon=2.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        dp_optimizer = DPOptimizer(model, dp_config)
        
        # Create accountant
        accountant = RDPAccountant(
            target_epsilon=2.0,
            target_delta=1e-5
        )
        
        # Simulate training steps
        for _ in range(10):
            accountant.accumulate(
                noise_multiplier=1.0,
                sample_rate=0.1
            )
        
        epsilon = accountant.get_epsilon()
        
        assert epsilon > 0
        assert not accountant.is_budget_exceeded()
    
    def test_full_evaluation(self, predictions_binary):
        """Test full evaluation pipeline."""
        from evaluation.metrics import MetricsCalculator
        from evaluation.privacy_eval import PrivacyEvaluator
        
        y_true, y_pred = predictions_binary
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all(y_true, y_pred)
        
        assert 'classification' in metrics
        assert metrics['classification']['auc_roc'] > 0.5
        
        # Privacy evaluation
        evaluator = PrivacyEvaluator()
        privacy = evaluator.evaluate({
            'epsilon': 1.0,
            'delta': 1e-5
        })
        
        assert 'privacy_level' in privacy
    
    def test_checkpoint_save_load(self, sample_partitions, temp_dir):
        """Test checkpoint saving and loading."""
        from federated.coordinator import FederatedCoordinator, FederatedConfig
        
        # Create test data
        X_test = np.random.randn(30, 7, 20).astype(np.float32)
        y_test = np.random.randint(0, 2, 30).astype(np.float32)
        
        config = FederatedConfig(
            n_rounds=2,
            clients_per_round=2,
            checkpoint_dir=str(temp_dir)
        )
        
        coordinator = FederatedCoordinator(
            partitions=sample_partitions,
            test_data=(X_test, y_test),
            input_dim=20,
            config=config
        )
        
        # Train
        results = coordinator.train()
        
        # Save checkpoint
        checkpoint_path = temp_dir / 'final_checkpoint.pt'
        coordinator.save_checkpoint(checkpoint_path)
        
        # Verify checkpoint exists
        assert checkpoint_path.exists()


@pytest.mark.integration
class TestModelPersistence:
    """Tests for model persistence."""
    
    def test_model_reproducibility(self, sample_sequences, temp_dir):
        """Test that training is reproducible with same seed."""
        from models.architecture import create_model
        from models.train_local import LocalTrainer, TrainingConfig
        
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        def train_model(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = create_model(input_dim=input_dim)
            config = TrainingConfig(epochs=3, batch_size=32)
            trainer = LocalTrainer(model, config)
            
            metrics = trainer.fit(X, y)
            return metrics['final_loss'], trainer.get_model_state()
        
        # Train twice with same seed
        loss1, state1 = train_model(42)
        loss2, state2 = train_model(42)
        
        # Should be identical
        assert abs(loss1 - loss2) < 1e-5
        for key in state1:
            assert torch.equal(state1[key], state2[key])


@pytest.mark.integration
class TestSecureAggregationIntegration:
    """Tests for secure aggregation integration."""
    
    def test_secure_federated_round(self, model_states):
        """Test federated round with secure aggregation."""
        from privacy.secure_aggregation import SecureAggregator
        from federated.aggregation import ModelAggregator, AggregationConfig, AggregationMethod
        
        global_state, client_states = model_states
        
        # Create secure aggregator
        client_ids = list(client_states.keys())
        secure_agg = SecureAggregator(client_ids=client_ids)
        
        # Secure aggregation
        masked_updates = secure_agg.mask_updates(client_states)
        aggregated = secure_agg.aggregate(masked_updates)
        
        # Verify result shape
        for key in global_state:
            assert key in aggregated
            assert aggregated[key].shape == global_state[key].shape


@pytest.mark.integration
class TestPrivacyAttackIntegration:
    """Tests for privacy attack integration."""
    
    def test_membership_inference_evaluation(self, sample_sequences):
        """Test MIA evaluation on trained model."""
        from models.architecture import create_model
        from models.train_local import LocalTrainer, TrainingConfig
        from privacy.attacks import MembershipInferenceAttack
        
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        # Split into member/non-member
        split_idx = len(X) // 2
        member_X, member_y = X[:split_idx], y[:split_idx]
        nonmember_X, nonmember_y = X[split_idx:], y[split_idx:]
        
        # Train model on member data
        model = create_model(input_dim=input_dim)
        config = TrainingConfig(epochs=5, batch_size=32)
        trainer = LocalTrainer(model, config)
        trainer.fit(member_X, member_y)
        
        # Create wrapper for attack
        class ModelWrapper:
            def __init__(self, trainer):
                self.trainer = trainer
            
            def predict_proba(self, X):
                return self.trainer.predict(X)
        
        # Run MIA
        attack = MembershipInferenceAttack()
        results = attack.run(
            target_model=ModelWrapper(trainer),
            member_data=(member_X, member_y),
            nonmember_data=(nonmember_X, nonmember_y)
        )
        
        assert 'attack_accuracy' in results


@pytest.mark.slow
@pytest.mark.integration
class TestFullPipelineWithPrivacy:
    """Full pipeline tests with privacy (slow)."""
    
    def test_complete_federated_training_with_dp(self, temp_dir):
        """Test complete training with differential privacy."""
        from data.synthetic_generator import SyntheticDataGenerator
        from data.preprocessing import Preprocessor
        from data.federated_partitioner import FederatedPartitioner
        from federated.coordinator import FederatedCoordinator, FederatedConfig
        
        # Generate data
        generator = SyntheticDataGenerator(n_users=50, days_per_user=14, seed=42)
        data = generator.generate()
        
        # Preprocess
        preprocessor = Preprocessor(sequence_length=7)
        X, y = preprocessor.fit_transform(data)
        
        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # Partition
        partitioner = FederatedPartitioner(n_clients=3)
        partitions = partitioner.partition(X_train, y_train)
        
        # Configure federated training
        config = FederatedConfig(
            n_rounds=3,
            clients_per_round=2,
            local_epochs=2,
            use_dp=True,
            target_epsilon=2.0
        )
        
        # Train
        input_dim = X.shape[2]
        coordinator = FederatedCoordinator(
            partitions=partitions,
            test_data=(X_test, y_test),
            input_dim=input_dim,
            config=config
        )
        
        results = coordinator.train()
        
        # Verify training completed
        assert len(results['history']) == 3
        assert 'final_metrics' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
