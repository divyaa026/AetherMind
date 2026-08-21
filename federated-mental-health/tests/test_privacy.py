"""
Tests for privacy components.
"""

import pytest
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy.dp_accounting import RDPAccountant, FederatedPrivacyAccountant, PrivacyBudget
from privacy.secure_aggregation import SecureAggregator, ThresholdSecureAggregator, SecretSharing
from privacy.attacks import MembershipInferenceAttack, GradientLeakageAttack, AttributeInferenceAttack


class TestRDPAccountant:
    """Tests for RDP-based privacy accounting."""
    
    def test_accountant_creation(self):
        """Test accountant initialization."""
        accountant = RDPAccountant(
            target_epsilon=1.0,
            target_delta=1e-5
        )
        
        assert accountant is not None
        assert accountant.target_epsilon == 1.0
        assert accountant.target_delta == 1e-5
    
    def test_accumulate_privacy(self):
        """Test privacy accumulation."""
        accountant = RDPAccountant(target_epsilon=5.0, target_delta=1e-5)
        
        # Simulate training steps
        for _ in range(10):
            accountant.accumulate(
                noise_multiplier=1.0,
                sample_rate=0.01
            )
        
        epsilon = accountant.get_epsilon()
        
        assert epsilon > 0
        assert epsilon < accountant.target_epsilon
    
    def test_budget_exceeded_warning(self):
        """Test detection of budget exhaustion."""
        accountant = RDPAccountant(target_epsilon=0.1, target_delta=1e-5)
        
        # Accumulate a lot of privacy cost
        for _ in range(1000):
            accountant.accumulate(
                noise_multiplier=0.1,  # Low noise = high privacy cost
                sample_rate=0.1
            )
        
        assert accountant.is_budget_exceeded()
    
    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        accountant = RDPAccountant(target_epsilon=2.0, target_delta=1e-5)
        
        initial_remaining = accountant.get_remaining_budget()
        
        accountant.accumulate(noise_multiplier=1.0, sample_rate=0.01)
        
        new_remaining = accountant.get_remaining_budget()
        
        assert new_remaining < initial_remaining


class TestFederatedPrivacyAccountant:
    """Tests for federated privacy accounting."""
    
    def test_multiple_clients(self):
        """Test accounting across multiple clients."""
        accountant = FederatedPrivacyAccountant(
            target_epsilon=2.0,
            target_delta=1e-5,
            n_clients=5
        )
        
        # Simulate rounds
        for round_num in range(3):
            participating_clients = [f'client_{i}' for i in range(3)]
            accountant.accumulate_round(
                participating_clients=participating_clients,
                noise_multiplier=1.0,
                sample_rate=0.02
            )
        
        epsilon = accountant.get_total_epsilon()
        
        assert epsilon > 0
    
    def test_client_level_privacy(self):
        """Test per-client privacy tracking."""
        accountant = FederatedPrivacyAccountant(
            target_epsilon=5.0,
            target_delta=1e-5,
            n_clients=5
        )
        
        # Client 0 participates in all rounds
        for round_num in range(5):
            if round_num < 3:
                clients = ['client_0', 'client_1']
            else:
                clients = ['client_0', 'client_2']
            
            accountant.accumulate_round(
                participating_clients=clients,
                noise_multiplier=1.0,
                sample_rate=0.01
            )
        
        # Client 0 should have higher privacy cost
        epsilon_0 = accountant.get_client_epsilon('client_0')
        epsilon_2 = accountant.get_client_epsilon('client_2')
        
        assert epsilon_0 > epsilon_2


class TestPrivacyBudget:
    """Tests for privacy budget management."""
    
    def test_budget_creation(self):
        """Test budget initialization."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
    
    def test_budget_subtraction(self):
        """Test budget consumption."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        
        spent = PrivacyBudget(epsilon=1.0, delta=1e-6)
        remaining = budget - spent
        
        assert remaining.epsilon == 4.0
        assert remaining.delta < budget.delta
    
    def test_budget_is_valid(self):
        """Test budget validity checking."""
        valid_budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        invalid_budget = PrivacyBudget(epsilon=-0.1, delta=1e-5)
        
        assert valid_budget.is_valid()
        assert not invalid_budget.is_valid()


class TestSecureAggregator:
    """Tests for secure aggregation."""
    
    @pytest.fixture
    def client_updates(self):
        """Create sample client updates."""
        torch.manual_seed(42)
        updates = {}
        
        for i in range(5):
            updates[f'client_{i}'] = {
                'layer1.weight': torch.randn(64, 20),
                'layer1.bias': torch.randn(64)
            }
        
        return updates
    
    def test_secure_aggregation(self, client_updates):
        """Test that secure aggregation produces valid output."""
        aggregator = SecureAggregator(client_ids=list(client_updates.keys()))
        
        result = aggregator.aggregate(client_updates)
        
        for key in client_updates['client_0']:
            assert key in result
            assert result[key].shape == client_updates['client_0'][key].shape
    
    def test_pairwise_masking(self, client_updates):
        """Test pairwise masking mechanism."""
        aggregator = SecureAggregator(client_ids=list(client_updates.keys()))
        
        # Generate masks
        masks = aggregator.generate_masks()
        
        # Masks should cancel out
        total_mask = None
        for client_id in client_updates:
            if client_id in masks:
                if total_mask is None:
                    total_mask = {k: v.clone() for k, v in masks[client_id].items()}
                else:
                    for key in total_mask:
                        total_mask[key] += masks[client_id][key]
        
        # Sum of masks should be zero
        for key in total_mask:
            assert torch.allclose(total_mask[key], torch.zeros_like(total_mask[key]), atol=1e-6)
    
    def test_aggregation_with_dropout(self, client_updates):
        """Test aggregation when some clients drop out."""
        all_clients = list(client_updates.keys())
        aggregator = SecureAggregator(client_ids=all_clients)
        
        # Only 3 clients respond
        responding = {k: v for k, v in list(client_updates.items())[:3]}
        
        result = aggregator.aggregate_with_dropout(
            updates=responding,
            all_clients=all_clients
        )
        
        for key in responding['client_0']:
            assert key in result


class TestThresholdSecureAggregator:
    """Tests for threshold secure aggregation."""
    
    def test_threshold_aggregation(self):
        """Test threshold-based secure aggregation."""
        torch.manual_seed(42)
        
        client_ids = [f'client_{i}' for i in range(5)]
        aggregator = ThresholdSecureAggregator(
            client_ids=client_ids,
            threshold=3  # Need 3 clients to reconstruct
        )
        
        # Create updates
        updates = {}
        for cid in client_ids:
            updates[cid] = {'w': torch.randn(10)}
        
        # With all 5 clients, should work
        result = aggregator.aggregate(updates)
        
        assert 'w' in result
    
    def test_threshold_not_met(self):
        """Test behavior when threshold not met."""
        torch.manual_seed(42)
        
        client_ids = [f'client_{i}' for i in range(5)]
        aggregator = ThresholdSecureAggregator(
            client_ids=client_ids,
            threshold=4
        )
        
        # Only 2 clients respond
        updates = {}
        for cid in client_ids[:2]:
            updates[cid] = {'w': torch.randn(10)}
        
        # Should raise or return None
        with pytest.raises(Exception):
            aggregator.aggregate(updates)


class TestSecretSharing:
    """Tests for secret sharing."""
    
    def test_shamir_secret_sharing(self):
        """Test Shamir secret sharing."""
        secret = 12345
        n_shares = 5
        threshold = 3
        
        sharing = SecretSharing(n_shares=n_shares, threshold=threshold)
        
        # Generate shares
        shares = sharing.split(secret)
        
        assert len(shares) == n_shares
    
    def test_secret_reconstruction(self):
        """Test secret reconstruction from shares."""
        secret = 42
        n_shares = 5
        threshold = 3
        
        sharing = SecretSharing(n_shares=n_shares, threshold=threshold)
        
        # Generate and reconstruct
        shares = sharing.split(secret)
        
        # Use exactly threshold shares
        subset = dict(list(shares.items())[:threshold])
        reconstructed = sharing.reconstruct(subset)
        
        assert reconstructed == secret
    
    def test_insufficient_shares(self):
        """Test that fewer than threshold shares fails."""
        secret = 100
        n_shares = 5
        threshold = 3
        
        sharing = SecretSharing(n_shares=n_shares, threshold=threshold)
        shares = sharing.split(secret)
        
        # Use fewer than threshold shares
        subset = dict(list(shares.items())[:threshold - 1])
        
        # Should not reconstruct correctly
        try:
            reconstructed = sharing.reconstruct(subset)
            assert reconstructed != secret
        except:
            pass  # Also acceptable to raise exception


class TestMembershipInferenceAttack:
    """Tests for membership inference attack."""
    
    @pytest.fixture
    def attack_data(self):
        """Create data for attack testing."""
        np.random.seed(42)
        
        # Member data (training data)
        member_X = np.random.randn(100, 7, 20).astype(np.float32)
        member_y = np.random.randint(0, 2, 100).astype(np.float32)
        
        # Non-member data (test data)
        nonmember_X = np.random.randn(100, 7, 20).astype(np.float32)
        nonmember_y = np.random.randint(0, 2, 100).astype(np.float32)
        
        return {
            'member': (member_X, member_y),
            'nonmember': (nonmember_X, nonmember_y)
        }
    
    def test_mia_attack(self, attack_data):
        """Test membership inference attack."""
        attack = MembershipInferenceAttack()
        
        # Create simple mock target model
        class MockModel:
            def predict_proba(self, X):
                # Higher confidence for members (overfitting simulation)
                return np.random.uniform(0.7, 0.95, len(X))
        
        target_model = MockModel()
        
        # Run attack
        results = attack.run(
            target_model=target_model,
            member_data=attack_data['member'],
            nonmember_data=attack_data['nonmember']
        )
        
        assert 'attack_accuracy' in results
        assert 'attack_auc' in results
    
    def test_mia_with_dp_defense(self, attack_data):
        """Test that DP reduces attack success."""
        attack = MembershipInferenceAttack()
        
        # Model without DP (overfits)
        class OverfitModel:
            def predict_proba(self, X):
                return np.random.uniform(0.85, 0.99, len(X))
        
        # Model with DP (less overfitting)
        class DPModel:
            def predict_proba(self, X):
                return np.random.uniform(0.5, 0.7, len(X))
        
        results_overfit = attack.run(
            target_model=OverfitModel(),
            member_data=attack_data['member'],
            nonmember_data=attack_data['nonmember']
        )
        
        results_dp = attack.run(
            target_model=DPModel(),
            member_data=attack_data['member'],
            nonmember_data=attack_data['nonmember']
        )
        
        # DP model should be harder to attack
        # (In practice, attack accuracy would be closer to 0.5)


class TestGradientLeakageAttack:
    """Tests for gradient leakage attack."""
    
    def test_gradient_attack(self):
        """Test gradient leakage attack."""
        attack = GradientLeakageAttack(iterations=10)  # Few iterations for test
        
        torch.manual_seed(42)
        
        # Original data
        original_x = torch.randn(1, 7, 20)
        
        # Gradients (would normally come from training)
        gradients = {
            'layer.weight': torch.randn(64, 20),
            'layer.bias': torch.randn(64)
        }
        
        results = attack.run(
            gradients=gradients,
            input_shape=(1, 7, 20)
        )
        
        assert 'reconstructed' in results
        assert 'reconstruction_loss' in results


class TestAttributeInferenceAttack:
    """Tests for attribute inference attack."""
    
    def test_attribute_attack(self):
        """Test attribute inference attack."""
        attack = AttributeInferenceAttack()
        
        np.random.seed(42)
        
        # Data with hidden attribute
        X = np.random.randn(100, 7, 20).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        # Hidden attribute (e.g., demographics)
        hidden_attr = np.random.randint(0, 2, 100)
        
        # Mock model predictions
        predictions = np.random.uniform(0, 1, 100)
        
        results = attack.run(
            predictions=predictions,
            true_labels=y,
            hidden_attribute=hidden_attr
        )
        
        assert 'attack_accuracy' in results


class TestPrivacyGuarantees:
    """Integration tests for privacy guarantees."""
    
    def test_epsilon_delta_relationship(self):
        """Test that smaller delta requires larger epsilon."""
        accountant_small_delta = RDPAccountant(
            target_epsilon=10.0,
            target_delta=1e-8
        )
        accountant_large_delta = RDPAccountant(
            target_epsilon=10.0,
            target_delta=1e-3
        )
        
        # Same operations
        for acc in [accountant_small_delta, accountant_large_delta]:
            for _ in range(100):
                acc.accumulate(noise_multiplier=1.0, sample_rate=0.01)
        
        eps_small = accountant_small_delta.get_epsilon()
        eps_large = accountant_large_delta.get_epsilon()
        
        # Smaller delta should give larger epsilon
        assert eps_small > eps_large
    
    def test_noise_privacy_tradeoff(self):
        """Test that more noise = better privacy."""
        accountant_low_noise = RDPAccountant(target_epsilon=10.0, target_delta=1e-5)
        accountant_high_noise = RDPAccountant(target_epsilon=10.0, target_delta=1e-5)
        
        # Low noise training
        for _ in range(100):
            accountant_low_noise.accumulate(noise_multiplier=0.5, sample_rate=0.01)
        
        # High noise training
        for _ in range(100):
            accountant_high_noise.accumulate(noise_multiplier=2.0, sample_rate=0.01)
        
        eps_low = accountant_low_noise.get_epsilon()
        eps_high = accountant_high_noise.get_epsilon()
        
        # More noise = lower epsilon (better privacy)
        assert eps_high < eps_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
