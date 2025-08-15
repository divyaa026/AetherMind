#!/usr/bin/env python3
"""
Federated Learning Coordinator for MindGuard
Manages federated learning across multiple nodes with privacy-preserving techniques
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
import hashlib
import pickle

from ..utils.privacy import DifferentialPrivacy, HomomorphicEncryption
from ..models.text_models import BERTCrisisDetector, BiLSTMCrisisDetector

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Configuration for a federated learning node"""
    node_id: str
    address: str
    port: int
    is_active: bool = True
    last_seen: Optional[float] = None
    model_version: int = 0
    data_size: int = 0
    privacy_level: str = "standard"  # minimal, standard, strict


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    num_nodes: int = 100
    min_nodes_per_round: int = 10
    max_nodes_per_round: int = 50
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    aggregation_method: str = "fedavg"  # fedavg, fedprox, fednova
    model_type: str = "bert"  # bert, bilstm
    convergence_threshold: float = 0.05
    max_wait_time: int = 300  # seconds


class FederatedCoordinator:
    """Main coordinator for federated learning"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.nodes: Dict[str, NodeConfig] = {}
        self.global_model = None
        self.model_history: List[Dict] = []
        self.round_history: List[Dict] = []
        self.dp_processor = DifferentialPrivacy()
        self.he_processor = HomomorphicEncryption()
        
        # Initialize global model
        self._initialize_global_model()
        
        # Generate encryption key for secure communication
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        logger.info(f"Federated Coordinator initialized with {config.num_nodes} nodes")
    
    def _initialize_global_model(self):
        """Initialize the global model based on configuration"""
        if self.config.model_type == "bert":
            self.global_model = BERTCrisisDetector(
                model_name="bert-base-uncased",
                num_classes=2,
                dropout=0.1
            )
        elif self.config.model_type == "bilstm":
            self.global_model = BiLSTMCrisisDetector(
                vocab_size=30000,
                embedding_dim=256,
                hidden_dim=128,
                num_classes=2,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        logger.info(f"Global model initialized: {self.config.model_type}")
    
    def register_node(self, node_id: str, address: str, port: int, data_size: int = 0) -> str:
        """Register a new federated learning node"""
        if len(self.nodes) >= self.config.num_nodes:
            raise ValueError("Maximum number of nodes reached")
        
        node_config = NodeConfig(
            node_id=node_id,
            address=address,
            port=port,
            data_size=data_size,
            last_seen=time.time()
        )
        
        self.nodes[node_id] = node_config
        logger.info(f"Node {node_id} registered at {address}:{port}")
        
        return node_id
    
    def deregister_node(self, node_id: str):
        """Deregister a federated learning node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node {node_id} deregistered")
    
    def update_node_status(self, node_id: str, is_active: bool = True, data_size: int = None):
        """Update node status and metadata"""
        if node_id in self.nodes:
            self.nodes[node_id].is_active = is_active
            self.nodes[node_id].last_seen = time.time()
            if data_size is not None:
                self.nodes[node_id].data_size = data_size
    
    def get_active_nodes(self) -> List[str]:
        """Get list of active nodes"""
        current_time = time.time()
        active_nodes = []
        
        for node_id, node in self.nodes.items():
            # Check if node is marked as active and seen recently
            if (node.is_active and 
                (node.last_seen is None or 
                 current_time - node.last_seen < self.config.max_wait_time)):
                active_nodes.append(node_id)
        
        return active_nodes
    
    def select_nodes_for_round(self, round_num: int) -> List[str]:
        """Select nodes to participate in the current round"""
        active_nodes = self.get_active_nodes()
        
        if len(active_nodes) < self.config.min_nodes_per_round:
            raise ValueError(f"Insufficient active nodes: {len(active_nodes)} < {self.config.min_nodes_per_round}")
        
        # Select random subset of active nodes
        num_selected = min(
            self.config.max_nodes_per_round,
            len(active_nodes)
        )
        
        selected_nodes = np.random.choice(
            active_nodes, 
            size=num_selected, 
            replace=False
        ).tolist()
        
        logger.info(f"Round {round_num}: Selected {len(selected_nodes)} nodes from {len(active_nodes)} active nodes")
        return selected_nodes
    
    def prepare_global_model_for_distribution(self, round_num: int) -> Dict[str, Any]:
        """Prepare global model for distribution to nodes"""
        # Get model state dict
        model_state = self.global_model.state_dict()
        
        # Apply differential privacy to model weights
        if self.config.privacy_epsilon > 0:
            model_state = self._apply_dp_to_model(model_state)
        
        # Encrypt model for secure transmission
        encrypted_state = self._encrypt_model_state(model_state)
        
        distribution_package = {
            "round": round_num,
            "model_type": self.config.model_type,
            "model_state": encrypted_state,
            "config": {
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "privacy_epsilon": self.config.privacy_epsilon,
                "privacy_delta": self.config.privacy_delta
            },
            "timestamp": time.time()
        }
        
        return distribution_package
    
    def _apply_dp_to_model(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to model weights"""
        dp_state = {}
        
        for key, tensor in model_state.items():
            if tensor.dtype in [torch.float32, torch.float64]:
                # Add Laplace noise for differential privacy
                noise = self.dp_processor.add_laplace_noise(
                    tensor.numpy(),
                    epsilon=self.config.privacy_epsilon,
                    delta=self.config.privacy_delta
                )
                dp_state[key] = torch.from_numpy(noise)
            else:
                dp_state[key] = tensor
        
        return dp_state
    
    def _encrypt_model_state(self, model_state: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt model state for secure transmission"""
        # Serialize model state
        serialized_state = pickle.dumps(model_state)
        
        # Encrypt using Fernet
        encrypted_state = self.cipher.encrypt(serialized_state)
        
        return encrypted_state
    
    def _decrypt_model_state(self, encrypted_state: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model state"""
        # Decrypt using Fernet
        decrypted_state = self.cipher.decrypt(encrypted_state)
        
        # Deserialize model state
        model_state = pickle.loads(decrypted_state)
        
        return model_state
    
    def aggregate_models(self, model_updates: List[Dict[str, Any]], weights: List[float] = None) -> bool:
        """Aggregate model updates from participating nodes"""
        if not model_updates:
            logger.warning("No model updates to aggregate")
            return False
        
        # Normalize weights if not provided
        if weights is None:
            weights = [1.0 / len(model_updates)] * len(model_updates)
        
        # Ensure weights sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Initialize aggregated state
        aggregated_state = {}
        
        # Aggregate model states
        for i, update in enumerate(model_updates):
            # Decrypt model state
            model_state = self._decrypt_model_state(update["model_state"])
            weight = weights[i]
            
            for key, tensor in model_state.items():
                if key not in aggregated_state:
                    aggregated_state[key] = torch.zeros_like(tensor)
                
                aggregated_state[key] += weight * tensor
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        # Log aggregation statistics
        avg_loss = np.mean([update.get("loss", 0.0) for update in model_updates])
        avg_accuracy = np.mean([update.get("accuracy", 0.0) for update in model_updates])
        
        logger.info(f"Model aggregation complete: avg_loss={avg_loss:.4f}, avg_accuracy={avg_accuracy:.4f}")
        
        return True
    
    def check_convergence(self, recent_rounds: int = 5) -> bool:
        """Check if federated learning has converged"""
        if len(self.round_history) < recent_rounds:
            return False
        
        recent_losses = [round_data["avg_loss"] for round_data in self.round_history[-recent_rounds:]]
        
        # Calculate variance in recent losses
        loss_variance = np.var(recent_losses)
        
        # Check if variance is below threshold
        converged = loss_variance < self.config.convergence_threshold
        
        logger.info(f"Convergence check: variance={loss_variance:.6f}, threshold={self.config.convergence_threshold}, converged={converged}")
        
        return converged
    
    def save_global_model(self, path: str):
        """Save the global model"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.global_model.state_dict(),
            "config": asdict(self.config),
            "round_history": self.round_history,
            "model_history": self.model_history
        }, model_path)
        
        logger.info(f"Global model saved to {model_path}")
    
    def load_global_model(self, path: str):
        """Load the global model"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.round_history = checkpoint.get("round_history", [])
        self.model_history = checkpoint.get("model_history", [])
        
        logger.info(f"Global model loaded from {path}")
    
    def get_round_statistics(self) -> Dict[str, Any]:
        """Get statistics about the federated learning process"""
        if not self.round_history:
            return {}
        
        recent_rounds = self.round_history[-10:]  # Last 10 rounds
        
        stats = {
            "total_rounds": len(self.round_history),
            "active_nodes": len(self.get_active_nodes()),
            "total_nodes": len(self.nodes),
            "recent_avg_loss": np.mean([r["avg_loss"] for r in recent_rounds]),
            "recent_avg_accuracy": np.mean([r["avg_accuracy"] for r in recent_rounds]),
            "convergence_status": self.check_convergence(),
            "last_round": self.round_history[-1] if self.round_history else None
        }
        
        return stats


class FederatedNode:
    """Represents a federated learning node"""
    
    def __init__(self, node_id: str, coordinator_address: str, coordinator_port: int):
        self.node_id = node_id
        self.coordinator_address = coordinator_address
        self.coordinator_port = coordinator_port
        self.local_model = None
        self.local_data = None
        self.current_round = 0
        
        logger.info(f"Federated node {node_id} initialized")
    
    def set_local_data(self, data):
        """Set local training data"""
        self.local_data = data
        logger.info(f"Node {self.node_id}: Local data set with {len(data)} samples")
    
    def train_local_model(self, global_model_state: bytes, config: Dict) -> Dict[str, Any]:
        """Train local model on local data"""
        # Decrypt and load global model state
        # This would be implemented with the coordinator's decryption key
        
        # Train local model (simplified implementation)
        local_loss = 0.1  # Placeholder
        local_accuracy = 0.85  # Placeholder
        
        # Encrypt local model updates
        # This would be implemented with the coordinator's encryption key
        
        return {
            "node_id": self.node_id,
            "round": self.current_round,
            "loss": local_loss,
            "accuracy": local_accuracy,
            "model_state": b"encrypted_model_state",  # Placeholder
            "timestamp": time.time()
        }


if __name__ == "__main__":
    # Example usage
    config = FederatedConfig(
        num_nodes=100,
        min_nodes_per_round=10,
        max_nodes_per_round=50,
        rounds=100,
        local_epochs=5,
        model_type="bert"
    )
    
    coordinator = FederatedCoordinator(config)
    
    # Register some nodes
    for i in range(10):
        coordinator.register_node(
            f"node_{i}",
            f"192.168.1.{i+10}",
            8000 + i,
            data_size=1000 + i * 100
        )
    
    print("Federated Learning Coordinator initialized successfully!")
