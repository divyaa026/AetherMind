"""
Federated aggregation strategies.
Implements various methods for combining client model updates.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import copy


class AggregationMethod(Enum):
    """Available aggregation methods."""
    FEDAVG = "fedavg"
    FEDAVG_MOMENTUM = "fedavg_momentum"
    FEDPROX = "fedprox"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    GEOMETRIC_MEDIAN = "geometric_median"


@dataclass
class AggregationConfig:
    """Configuration for aggregation."""
    method: AggregationMethod = AggregationMethod.FEDAVG
    momentum: float = 0.9  # For FedAvg with momentum
    trim_ratio: float = 0.1  # For trimmed mean
    krum_f: int = 1  # Number of Byzantine clients for Krum
    n_selected: int = 3  # For Multi-Krum
    geo_med_iterations: int = 10  # For geometric median
    fedprox_mu: float = 0.01  # Proximal term for FedProx


class ModelAggregator:
    """
    Aggregates model updates from multiple clients.
    
    Supports various aggregation strategies including:
    - FedAvg: Weighted average by sample count
    - FedAvg with Momentum: Server-side momentum
    - FedProx: Proximal regularization
    - Median: Coordinate-wise median (Byzantine-robust)
    - Trimmed Mean: Trim outliers before averaging (Byzantine-robust)
    - Krum: Select update closest to others (Byzantine-robust)
    - Geometric Median: Minimize sum of distances (Byzantine-robust)
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        """
        Initialize aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        
        # Momentum buffer for FedAvg with momentum
        self.momentum_buffer: Optional[Dict[str, torch.Tensor]] = None
        
        # Previous global state for proximal methods
        self.prev_global_state: Optional[Dict[str, torch.Tensor]] = None
    
    def aggregate(self,
                  client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
                  global_state: Optional[Dict[str, torch.Tensor]] = None
                  ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: List of (client_id, model_state, n_samples)
            global_state: Current global model state (needed for some methods)
            
        Returns:
            Aggregated model state
        """
        method = self.config.method
        
        if method == AggregationMethod.FEDAVG:
            return self._fedavg(client_updates)
        
        elif method == AggregationMethod.FEDAVG_MOMENTUM:
            if global_state is None:
                raise ValueError("FedAvg with momentum requires global_state")
            return self._fedavg_momentum(client_updates, global_state)
        
        elif method == AggregationMethod.FEDPROX:
            if global_state is None:
                raise ValueError("FedProx requires global_state")
            return self._fedprox(client_updates, global_state)
        
        elif method == AggregationMethod.MEDIAN:
            return self._median(client_updates)
        
        elif method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(client_updates)
        
        elif method == AggregationMethod.KRUM:
            return self._krum(client_updates)
        
        elif method == AggregationMethod.MULTI_KRUM:
            return self._multi_krum(client_updates)
        
        elif method == AggregationMethod.GEOMETRIC_MEDIAN:
            return self._geometric_median(client_updates)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _fedavg(self,
                client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                ) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging: Weighted average by sample count.
        
        McMahan et al., "Communication-Efficient Learning of Deep Networks 
        from Decentralized Data" (2017)
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
                         client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
                         global_state: Dict[str, torch.Tensor]
                         ) -> Dict[str, torch.Tensor]:
        """
        FedAvg with server-side momentum.
        
        Hsu et al., "Measuring the Effects of Non-Identical Data Distribution 
        for Federated Visual Classification" (2019)
        """
        # Get vanilla FedAvg result
        fedavg_result = self._fedavg(client_updates)
        
        if self.momentum_buffer is None:
            self.momentum_buffer = {
                name: torch.zeros_like(param)
                for name, param in fedavg_result.items()
            }
        
        # Calculate update and apply momentum
        aggregated = {}
        for name in fedavg_result.keys():
            delta = fedavg_result[name] - global_state[name]
            
            self.momentum_buffer[name] = (
                self.config.momentum * self.momentum_buffer[name] + delta
            )
            
            aggregated[name] = global_state[name] + self.momentum_buffer[name]
        
        return aggregated
    
    def _fedprox(self,
                 client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
                 global_state: Dict[str, torch.Tensor]
                 ) -> Dict[str, torch.Tensor]:
        """
        FedProx: FedAvg with proximal term toward global model.
        
        Li et al., "Federated Optimization in Heterogeneous Networks" (2020)
        
        Note: The proximal term is applied during client training.
        Here we just do standard FedAvg aggregation.
        """
        return self._fedavg(client_updates)
    
    def _median(self,
                client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise median: Byzantine-robust aggregation.
        
        Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal 
        Statistical Rates" (2018)
        """
        aggregated = {}
        
        for name in client_updates[0][1].keys():
            stacked = torch.stack([
                state[name] for _, state, _ in client_updates
            ], dim=0)
            
            aggregated[name] = torch.median(stacked, dim=0).values
        
        return aggregated
    
    def _trimmed_mean(self,
                      client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                      ) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean: Remove extreme values before averaging.
        
        Yin et al., "Byzantine-Robust Distributed Learning" (2018)
        """
        n_clients = len(client_updates)
        n_trim = max(1, int(n_clients * self.config.trim_ratio))
        
        if n_clients - 2 * n_trim < 1:
            # Not enough clients to trim, fallback to median
            return self._median(client_updates)
        
        aggregated = {}
        
        for name in client_updates[0][1].keys():
            stacked = torch.stack([
                state[name] for _, state, _ in client_updates
            ], dim=0)
            
            # Sort and trim along client dimension
            sorted_stacked, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_stacked[n_trim:-n_trim] if n_trim > 0 else sorted_stacked
            
            aggregated[name] = trimmed.mean(dim=0)
        
        return aggregated
    
    def _compute_distances(self,
                           client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                           ) -> np.ndarray:
        """Compute pairwise distances between client updates."""
        n_clients = len(client_updates)
        
        # Flatten each client's update
        flattened = []
        for _, state, _ in client_updates:
            flat = torch.cat([p.flatten() for p in state.values()])
            flattened.append(flat)
        
        # Compute pairwise distances
        distances = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = torch.norm(flattened[i] - flattened[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _krum(self,
              client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
              ) -> Dict[str, torch.Tensor]:
        """
        Krum: Select the update closest to n-f-2 others.
        
        Blanchard et al., "Machine Learning with Adversaries: Byzantine 
        Tolerant Gradient Descent" (2017)
        """
        n_clients = len(client_updates)
        f = min(self.config.krum_f, n_clients - 2)
        n_closest = n_clients - f - 2
        
        if n_closest < 1:
            # Not enough clients, fallback to FedAvg
            return self._fedavg(client_updates)
        
        # Compute distances
        distances = self._compute_distances(client_updates)
        
        # For each client, sum distances to n_closest nearest neighbors
        scores = []
        for i in range(n_clients):
            sorted_distances = np.sort(distances[i])
            # Sum of distances to n_closest nearest (excluding self)
            score = np.sum(sorted_distances[1:n_closest + 1])
            scores.append(score)
        
        # Select client with minimum score
        selected_idx = np.argmin(scores)
        
        return client_updates[selected_idx][1]
    
    def _multi_krum(self,
                    client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                    ) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum: Average of m Krum-selected updates.
        
        Blanchard et al. (2017)
        """
        n_clients = len(client_updates)
        m = min(self.config.n_selected, n_clients)
        f = min(self.config.krum_f, n_clients - 2)
        n_closest = n_clients - f - 2
        
        if n_closest < 1 or m < 1:
            return self._fedavg(client_updates)
        
        # Compute distances
        distances = self._compute_distances(client_updates)
        
        # Compute Krum scores
        scores = []
        for i in range(n_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[1:n_closest + 1])
            scores.append(score)
        
        # Select top m clients with lowest scores
        selected_indices = np.argsort(scores)[:m]
        selected_updates = [client_updates[i] for i in selected_indices]
        
        # Average selected updates
        return self._fedavg(selected_updates)
    
    def _geometric_median(self,
                          client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]]
                          ) -> Dict[str, torch.Tensor]:
        """
        Geometric median: Point minimizing sum of distances to all updates.
        
        Computed using Weiszfeld's algorithm.
        
        Pillutla et al., "Robust Aggregation for Federated Learning" (2019)
        """
        n_clients = len(client_updates)
        n_iterations = self.config.geo_med_iterations
        
        # Flatten all updates
        flattened = []
        for _, state, _ in client_updates:
            flat = torch.cat([p.flatten() for p in state.values()])
            flattened.append(flat)
        
        flattened = torch.stack(flattened)
        
        # Initialize with FedAvg
        median = flattened.mean(dim=0)
        
        # Weiszfeld's algorithm
        eps = 1e-8
        for _ in range(n_iterations):
            distances = torch.norm(flattened - median.unsqueeze(0), dim=1)
            weights = 1.0 / (distances + eps)
            weights = weights / weights.sum()
            median = (weights.unsqueeze(1) * flattened).sum(dim=0)
        
        # Unflatten back to state dict
        aggregated = {}
        idx = 0
        template = client_updates[0][1]
        for name in template.keys():
            shape = template[name].shape
            numel = template[name].numel()
            aggregated[name] = median[idx:idx + numel].reshape(shape)
            idx += numel
        
        return aggregated
    
    def reset(self) -> None:
        """Reset aggregator state (momentum buffer, etc.)."""
        self.momentum_buffer = None
        self.prev_global_state = None


def compare_aggregation_methods(
    client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],
    global_state: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compare all aggregation methods on the same updates.
    
    Args:
        client_updates: Client model updates
        global_state: Current global state
        
    Returns:
        Dict mapping method name to aggregated state
    """
    results = {}
    
    for method in AggregationMethod:
        try:
            config = AggregationConfig(method=method)
            aggregator = ModelAggregator(config)
            
            result = aggregator.aggregate(client_updates, global_state)
            results[method.value] = result
        except Exception as e:
            print(f"Error with {method.value}: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create sample updates
    n_clients = 10
    param_size = 100
    
    client_updates = []
    base_params = {'weight': torch.randn(param_size), 'bias': torch.randn(10)}
    
    for i in range(n_clients):
        # Add noise to base params
        state = {
            'weight': base_params['weight'] + torch.randn(param_size) * 0.1,
            'bias': base_params['bias'] + torch.randn(10) * 0.1
        }
        n_samples = np.random.randint(50, 200)
        client_updates.append((i, state, n_samples))
    
    # Add one Byzantine client with very different update
    byzantine_state = {
        'weight': torch.randn(param_size) * 10,
        'bias': torch.randn(10) * 10
    }
    client_updates.append((n_clients, byzantine_state, 100))
    
    # Test all methods
    print("Testing aggregation methods with 1 Byzantine client:")
    print("=" * 60)
    
    global_state = base_params
    
    for method in AggregationMethod:
        try:
            config = AggregationConfig(method=method)
            aggregator = ModelAggregator(config)
            
            result = aggregator.aggregate(client_updates, global_state)
            
            # Compute distance from true mean (without Byzantine)
            true_mean = {}
            for name in base_params.keys():
                values = torch.stack([
                    state[name] for _, state, _ in client_updates[:-1]
                ])
                true_mean[name] = values.mean(dim=0)
            
            distance = sum(
                torch.norm(result[name] - true_mean[name]).item()
                for name in result.keys()
            )
            
            print(f"{method.value:20s}: distance from true mean = {distance:.4f}")
        
        except Exception as e:
            print(f"{method.value:20s}: Error - {e}")
