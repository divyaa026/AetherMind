"""
Federated learning package.
"""

from .server import FederatedServer, ServerConfig
from .client import FederatedClient, ClientConfig, ClientManager
from .coordinator import FederatedCoordinator, FederatedConfig, run_federated_experiment
from .aggregation import (
    ModelAggregator,
    AggregationConfig, 
    AggregationMethod,
    compare_aggregation_methods
)

__all__ = [
    # Server
    'FederatedServer',
    'ServerConfig',
    
    # Client
    'FederatedClient',
    'ClientConfig',
    'ClientManager',
    
    # Coordinator
    'FederatedCoordinator',
    'FederatedConfig',
    'run_federated_experiment',
    
    # Aggregation
    'ModelAggregator',
    'AggregationConfig',
    'AggregationMethod',
    'compare_aggregation_methods'
]
