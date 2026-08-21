"""
Differential Privacy Accounting for Federated Learning.
Tracks privacy budget consumption using Renyi Differential Privacy (RDP).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


class AccountingMethod(Enum):
    """Privacy accounting methods."""
    RDP = "rdp"  # Renyi Differential Privacy
    GDP = "gdp"  # Gaussian Differential Privacy
    ZCDP = "zcdp"  # Zero-Concentrated DP
    MOMENTS = "moments"  # Moments Accountant


@dataclass
class PrivacyBudget:
    """Privacy budget specification."""
    epsilon: float  # Privacy parameter (smaller = more private)
    delta: float  # Failure probability
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not 0 < self.delta < 1:
            raise ValueError("Delta must be in (0, 1)")


@dataclass
class MechanismParams:
    """Parameters for a DP mechanism."""
    noise_multiplier: float  # σ / sensitivity ratio
    sampling_probability: float  # Subsampling rate
    n_steps: int  # Number of gradient steps
    n_samples: int  # Dataset size
    batch_size: int  # Batch size
    max_grad_norm: float = 1.0  # Gradient clipping bound


class RDPAccountant:
    """
    Renyi Differential Privacy Accountant.
    
    Tracks privacy loss using RDP and converts to (ε, δ)-DP.
    Based on Mironov (2017) and Abadi et al. (2016).
    """
    
    # RDP orders to track
    DEFAULT_ORDERS = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    
    def __init__(self, 
                 orders: Optional[List[float]] = None,
                 target_delta: float = 1e-5):
        """
        Initialize RDP accountant.
        
        Args:
            orders: RDP orders to track
            target_delta: Target delta for (ε, δ)-DP conversion
        """
        self.orders = orders or self.DEFAULT_ORDERS
        self.target_delta = target_delta
        
        # Accumulated RDP values for each order
        self.rdp_budget = np.zeros(len(self.orders))
        
        # History of mechanisms
        self.history: List[Dict] = []
    
    def compute_rdp_gaussian(self,
                            q: float,  # Sampling probability
                            sigma: float,  # Noise multiplier
                            order: float  # RDP order
                            ) -> float:
        """
        Compute RDP of the Sampled Gaussian Mechanism.
        
        For subsampled Gaussian with sampling probability q and noise σ.
        Uses the analytical formula from Mironov (2017).
        """
        if q == 0:
            return 0
        
        if order == 1:
            # First order is just mutual information
            return q * q / (2 * sigma * sigma)
        
        if q == 1:
            # No subsampling - standard Gaussian mechanism
            return order / (2 * sigma * sigma)
        
        # Subsampled Gaussian mechanism (approximation)
        # Using the formula from "Rényi Differential Privacy of the Sampled Gaussian Mechanism"
        
        # For small q, use the log-sum-exp trick
        log_a = float('-inf')
        
        for j in range(int(order) + 1):
            log_coeff = (
                math.lgamma(order + 1) - 
                math.lgamma(j + 1) - 
                math.lgamma(order - j + 1)
            )
            log_coeff += j * math.log(q) + (order - j) * math.log(1 - q)
            log_coeff += j * (j - 1) / (2 * sigma * sigma)
            
            log_a = self._log_add(log_a, log_coeff)
        
        return log_a / (order - 1)
    
    def _log_add(self, log_a: float, log_b: float) -> float:
        """Compute log(a + b) from log(a) and log(b)."""
        if log_a < log_b:
            log_a, log_b = log_b, log_a
        
        if log_a == float('-inf'):
            return log_b
        if log_b == float('-inf'):
            return log_a
        
        return log_a + math.log1p(math.exp(log_b - log_a))
    
    def compute_rdp_single_step(self,
                                 noise_multiplier: float,
                                 sampling_probability: float
                                 ) -> np.ndarray:
        """
        Compute RDP for a single step of DP-SGD.
        
        Args:
            noise_multiplier: Noise multiplier (σ / max_grad_norm)
            sampling_probability: Batch sampling probability
            
        Returns:
            RDP values for all tracked orders
        """
        rdp = np.zeros(len(self.orders))
        
        for i, order in enumerate(self.orders):
            rdp[i] = self.compute_rdp_gaussian(
                q=sampling_probability,
                sigma=noise_multiplier,
                order=order
            )
        
        return rdp
    
    def add_mechanism(self,
                      noise_multiplier: float,
                      sampling_probability: float,
                      n_steps: int = 1,
                      description: str = "") -> None:
        """
        Account for a DP mechanism.
        
        Args:
            noise_multiplier: Noise multiplier
            sampling_probability: Sampling probability
            n_steps: Number of steps
            description: Description of mechanism
        """
        # RDP composes linearly
        rdp_per_step = self.compute_rdp_single_step(
            noise_multiplier, sampling_probability
        )
        
        self.rdp_budget += n_steps * rdp_per_step
        
        # Record in history
        self.history.append({
            'noise_multiplier': noise_multiplier,
            'sampling_probability': sampling_probability,
            'n_steps': n_steps,
            'description': description,
            'rdp_added': n_steps * rdp_per_step.sum()
        })
    
    def add_training_round(self, params: MechanismParams, description: str = "") -> None:
        """
        Account for a training round.
        
        Args:
            params: Mechanism parameters
            description: Description
        """
        sampling_prob = params.batch_size / params.n_samples
        
        self.add_mechanism(
            noise_multiplier=params.noise_multiplier,
            sampling_probability=sampling_prob,
            n_steps=params.n_steps,
            description=description or "training_round"
        )
    
    def rdp_to_dp(self, 
                  rdp: np.ndarray,
                  orders: List[float],
                  delta: float
                  ) -> Tuple[float, float]:
        """
        Convert RDP to (ε, δ)-DP.
        
        Uses the formula: ε = rdp - log(δ) / (α - 1)
        
        Args:
            rdp: RDP values for each order
            orders: RDP orders
            delta: Target delta
            
        Returns:
            (epsilon, optimal_order)
        """
        eps_candidates = []
        
        for i, (alpha, rdp_alpha) in enumerate(zip(orders, rdp)):
            if alpha <= 1:
                continue
            
            eps = rdp_alpha + math.log(1 / delta) / (alpha - 1)
            eps_candidates.append((eps, alpha))
        
        if not eps_candidates:
            return float('inf'), 0
        
        best_eps, best_order = min(eps_candidates, key=lambda x: x[0])
        return best_eps, best_order
    
    def get_privacy_spent(self, delta: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Get total privacy spent.
        
        Args:
            delta: Target delta (uses default if not specified)
            
        Returns:
            (epsilon, delta, optimal_order)
        """
        delta = delta or self.target_delta
        epsilon, optimal_order = self.rdp_to_dp(self.rdp_budget, self.orders, delta)
        return epsilon, delta, optimal_order
    
    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Get epsilon for given delta."""
        epsilon, _, _ = self.get_privacy_spent(delta)
        return epsilon
    
    def compute_noise_for_budget(self,
                                  target_epsilon: float,
                                  target_delta: float,
                                  sampling_probability: float,
                                  n_steps: int
                                  ) -> float:
        """
        Compute noise multiplier needed to achieve target budget.
        
        Uses binary search to find optimal noise level.
        
        Args:
            target_epsilon: Target epsilon
            target_delta: Target delta
            sampling_probability: Sampling probability
            n_steps: Number of steps
            
        Returns:
            Required noise multiplier
        """
        low, high = 0.1, 100.0
        
        while high - low > 0.01:
            mid = (low + high) / 2
            
            # Compute RDP for this noise level
            rdp = np.zeros(len(self.orders))
            rdp_per_step = self.compute_rdp_single_step(mid, sampling_probability)
            rdp = n_steps * rdp_per_step
            
            # Convert to epsilon
            epsilon, _ = self.rdp_to_dp(rdp, self.orders, target_delta)
            
            if epsilon < target_epsilon:
                high = mid
            else:
                low = mid
        
        return high
    
    def remaining_budget(self, target: PrivacyBudget) -> float:
        """
        Compute remaining epsilon budget.
        
        Args:
            target: Target privacy budget
            
        Returns:
            Remaining epsilon
        """
        current_eps = self.get_epsilon(target.delta)
        return max(0, target.epsilon - current_eps)
    
    def is_budget_exhausted(self, target: PrivacyBudget) -> bool:
        """Check if privacy budget is exhausted."""
        return self.remaining_budget(target) <= 0
    
    def reset(self) -> None:
        """Reset accountant."""
        self.rdp_budget = np.zeros(len(self.orders))
        self.history = []
    
    def get_summary(self) -> Dict:
        """Get accounting summary."""
        epsilon, delta, order = self.get_privacy_spent()
        
        return {
            'epsilon': epsilon,
            'delta': delta,
            'optimal_order': order,
            'n_mechanisms': len(self.history),
            'total_rdp': float(self.rdp_budget.sum()),
            'history': self.history
        }


class CompositionAccountant:
    """
    Handles composition of multiple DP mechanisms.
    
    Supports various composition theorems.
    """
    
    def __init__(self):
        """Initialize composition accountant."""
        self.mechanisms: List[Tuple[float, float]] = []  # (epsilon, delta) pairs
    
    def add_mechanism(self, epsilon: float, delta: float) -> None:
        """Add a mechanism to the composition."""
        self.mechanisms.append((epsilon, delta))
    
    def basic_composition(self) -> Tuple[float, float]:
        """
        Basic composition theorem.
        
        ε_total = Σ ε_i
        δ_total = Σ δ_i
        """
        total_eps = sum(eps for eps, _ in self.mechanisms)
        total_delta = sum(delta for _, delta in self.mechanisms)
        return total_eps, total_delta
    
    def advanced_composition(self, target_delta: float) -> Tuple[float, float]:
        """
        Advanced composition theorem (Kairouz et al., 2015).
        
        For k mechanisms each with (ε, δ):
        Total is (ε_total, δ_total) where:
        ε_total = √(2k ln(1/δ')) · ε + k·ε(e^ε - 1)
        δ_total = k·δ + δ'
        """
        if not self.mechanisms:
            return 0, 0
        
        k = len(self.mechanisms)
        eps_avg = sum(eps for eps, _ in self.mechanisms) / k
        delta_sum = sum(delta for _, delta in self.mechanisms)
        
        # Slack delta
        delta_prime = target_delta - delta_sum
        if delta_prime <= 0:
            # Fallback to basic
            return self.basic_composition()
        
        # Advanced composition
        term1 = math.sqrt(2 * k * math.log(1 / delta_prime)) * eps_avg
        term2 = k * eps_avg * (math.exp(eps_avg) - 1)
        
        total_eps = term1 + term2
        total_delta = delta_sum + delta_prime
        
        return total_eps, total_delta
    
    def get_optimal_composition(self, target_delta: float) -> Tuple[float, float]:
        """Get best composition bound."""
        basic = self.basic_composition()
        advanced = self.advanced_composition(target_delta)
        
        # Return the better bound
        if advanced[0] < basic[0] and advanced[1] <= target_delta:
            return advanced
        return basic
    
    def reset(self) -> None:
        """Reset accountant."""
        self.mechanisms = []


class FederatedPrivacyAccountant:
    """
    Privacy accountant for federated learning.
    
    Tracks privacy across:
    - Local training on each client
    - Aggregation with DP noise
    - Multiple training rounds
    """
    
    def __init__(self, 
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5,
                 n_clients: int = 10):
        """
        Initialize federated privacy accountant.
        
        Args:
            target_epsilon: Total privacy budget
            target_delta: Target delta
            n_clients: Number of clients
        """
        self.target = PrivacyBudget(target_epsilon, target_delta)
        self.n_clients = n_clients
        
        # Per-client accountants
        self.client_accountants: Dict[int, RDPAccountant] = {}
        
        # Server aggregation accountant
        self.server_accountant = RDPAccountant(target_delta=target_delta)
        
        # Overall accountant
        self.global_accountant = RDPAccountant(target_delta=target_delta)
    
    def add_local_training(self,
                           client_id: int,
                           noise_multiplier: float,
                           sampling_probability: float,
                           n_steps: int) -> None:
        """
        Account for local training on a client.
        
        Args:
            client_id: Client identifier
            noise_multiplier: Noise multiplier
            sampling_probability: Sampling probability
            n_steps: Number of gradient steps
        """
        if client_id not in self.client_accountants:
            self.client_accountants[client_id] = RDPAccountant(
                target_delta=self.target.delta
            )
        
        self.client_accountants[client_id].add_mechanism(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            n_steps=n_steps,
            description=f"local_training_client_{client_id}"
        )
    
    def add_aggregation(self,
                        noise_multiplier: float,
                        n_participating_clients: int) -> None:
        """
        Account for server-side aggregation with DP.
        
        Args:
            noise_multiplier: Noise added during aggregation
            n_participating_clients: Number of clients in round
        """
        # Privacy amplification by subsampling clients
        sampling_prob = n_participating_clients / self.n_clients
        
        self.server_accountant.add_mechanism(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_prob,
            n_steps=1,
            description="server_aggregation"
        )
    
    def add_round(self,
                  client_noise_multiplier: float,
                  server_noise_multiplier: float,
                  n_local_steps: int,
                  batch_size: int,
                  client_dataset_sizes: Dict[int, int],
                  participating_clients: List[int]) -> None:
        """
        Account for a complete training round.
        
        Args:
            client_noise_multiplier: Noise for local training
            server_noise_multiplier: Noise for aggregation
            n_local_steps: Local training steps per client
            batch_size: Batch size
            client_dataset_sizes: Dataset size per client
            participating_clients: Clients in this round
        """
        # Account for each client's local training
        for client_id in participating_clients:
            dataset_size = client_dataset_sizes.get(client_id, 1000)
            sampling_prob = batch_size / dataset_size
            
            self.add_local_training(
                client_id=client_id,
                noise_multiplier=client_noise_multiplier,
                sampling_probability=sampling_prob,
                n_steps=n_local_steps
            )
        
        # Account for aggregation
        self.add_aggregation(
            noise_multiplier=server_noise_multiplier,
            n_participating_clients=len(participating_clients)
        )
    
    def get_client_privacy(self, client_id: int) -> Tuple[float, float]:
        """Get privacy spent for a specific client."""
        if client_id not in self.client_accountants:
            return 0, self.target.delta
        
        return self.client_accountants[client_id].get_privacy_spent()[:2]
    
    def get_server_privacy(self) -> Tuple[float, float]:
        """Get privacy spent by server aggregation."""
        return self.server_accountant.get_privacy_spent()[:2]
    
    def get_total_privacy(self) -> Tuple[float, float]:
        """
        Get total privacy spent.
        
        Uses composition of client and server privacy.
        """
        # Get max client privacy (worst case)
        max_client_eps = 0
        total_client_delta = 0
        
        for client_id, accountant in self.client_accountants.items():
            eps, delta, _ = accountant.get_privacy_spent()
            max_client_eps = max(max_client_eps, eps)
            total_client_delta += delta
        
        # Server privacy
        server_eps, server_delta, _ = self.server_accountant.get_privacy_spent()
        
        # Compose client and server (conservative)
        total_eps = max_client_eps + server_eps
        total_delta = total_client_delta + server_delta
        
        return total_eps, min(total_delta, 1.0)
    
    def is_budget_available(self) -> bool:
        """Check if privacy budget is still available."""
        eps, delta = self.get_total_privacy()
        return eps < self.target.epsilon and delta < self.target.delta
    
    def get_summary(self) -> Dict:
        """Get complete privacy summary."""
        total_eps, total_delta = self.get_total_privacy()
        
        client_summaries = {}
        for client_id, accountant in self.client_accountants.items():
            eps, delta, order = accountant.get_privacy_spent()
            client_summaries[client_id] = {
                'epsilon': eps,
                'delta': delta,
                'optimal_order': order
            }
        
        return {
            'target': {'epsilon': self.target.epsilon, 'delta': self.target.delta},
            'total': {'epsilon': total_eps, 'delta': total_delta},
            'remaining_epsilon': max(0, self.target.epsilon - total_eps),
            'budget_exhausted': not self.is_budget_available(),
            'server': self.server_accountant.get_summary(),
            'clients': client_summaries,
            'n_clients_tracked': len(self.client_accountants)
        }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Testing RDP Accountant")
    print("=" * 60)
    
    # Create accountant
    accountant = RDPAccountant(target_delta=1e-5)
    
    # Simulate training
    n_epochs = 10
    batch_size = 32
    dataset_size = 10000
    noise_multiplier = 1.0
    
    sampling_prob = batch_size / dataset_size
    steps_per_epoch = dataset_size // batch_size
    
    for epoch in range(n_epochs):
        accountant.add_mechanism(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_prob,
            n_steps=steps_per_epoch,
            description=f"epoch_{epoch}"
        )
        
        eps, delta, order = accountant.get_privacy_spent()
        print(f"After epoch {epoch + 1}: ε = {eps:.4f}, δ = {delta:.2e}, order = {order:.1f}")
    
    print("\n" + "=" * 60)
    print("Testing Federated Privacy Accountant")
    print("=" * 60)
    
    # Create federated accountant
    fed_accountant = FederatedPrivacyAccountant(
        target_epsilon=1.0,
        target_delta=1e-5,
        n_clients=10
    )
    
    # Simulate 5 rounds
    for round_num in range(5):
        participating = list(range(0, 10, 2))  # Every other client
        dataset_sizes = {i: np.random.randint(500, 1500) for i in range(10)}
        
        fed_accountant.add_round(
            client_noise_multiplier=1.0,
            server_noise_multiplier=0.1,
            n_local_steps=100,
            batch_size=32,
            client_dataset_sizes=dataset_sizes,
            participating_clients=participating
        )
        
        eps, delta = fed_accountant.get_total_privacy()
        print(f"After round {round_num + 1}: ε = {eps:.4f}, δ = {delta:.2e}")
    
    print("\nSummary:")
    summary = fed_accountant.get_summary()
    print(f"  Budget exhausted: {summary['budget_exhausted']}")
    print(f"  Remaining ε: {summary['remaining_epsilon']:.4f}")
