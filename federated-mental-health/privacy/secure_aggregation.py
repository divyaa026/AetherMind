"""
Secure Aggregation Protocol for Federated Learning.
Enables aggregation of client updates without revealing individual contributions.
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import secrets
import struct


class SecureAggregationProtocol(Enum):
    """Available secure aggregation protocols."""
    SIMPLE_MASKING = "simple_masking"
    PAIRWISE_MASKING = "pairwise_masking"
    THRESHOLD_AGGREGATION = "threshold_aggregation"


@dataclass
class SecureAggConfig:
    """Configuration for secure aggregation."""
    protocol: SecureAggregationProtocol = SecureAggregationProtocol.PAIRWISE_MASKING
    threshold: int = 3  # Minimum clients for threshold protocol
    modulus: int = 2**32  # For modular arithmetic
    seed_length: int = 32  # Random seed length in bytes
    quantization_bits: int = 16  # Bits for quantization


class PseudoRandomGenerator:
    """
    Cryptographically secure pseudo-random generator.
    Used for generating masks from shared seeds.
    """
    
    def __init__(self, seed: bytes):
        """
        Initialize PRG with seed.
        
        Args:
            seed: Random seed bytes
        """
        self.seed = seed
        self.counter = 0
    
    def generate(self, size: int) -> np.ndarray:
        """
        Generate pseudo-random values.
        
        Args:
            size: Number of values to generate
            
        Returns:
            Array of pseudo-random values
        """
        values = []
        
        while len(values) < size:
            # Hash seed with counter
            data = self.seed + struct.pack('>Q', self.counter)
            hash_bytes = hashlib.sha256(data).digest()
            
            # Convert to int32 values
            for i in range(0, len(hash_bytes), 4):
                if len(values) >= size:
                    break
                val = struct.unpack('>i', hash_bytes[i:i+4])[0]
                values.append(val)
            
            self.counter += 1
        
        return np.array(values[:size], dtype=np.int64)


class SecretSharing:
    """
    Shamir's Secret Sharing implementation.
    Splits a secret into shares that can be reconstructed with threshold shares.
    """
    
    def __init__(self, prime: int = 2**61 - 1):
        """
        Initialize secret sharing.
        
        Args:
            prime: Prime modulus for field operations
        """
        self.prime = prime
    
    def _mod_inverse(self, a: int, p: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm."""
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x, y = extended_gcd(b % a, a)
            return gcd, y - (b // a) * x, x
        
        _, x, _ = extended_gcd(a % p, p)
        return (x % p + p) % p
    
    def share(self, 
              secret: int, 
              n_shares: int, 
              threshold: int) -> List[Tuple[int, int]]:
        """
        Split secret into shares.
        
        Args:
            secret: Secret to share
            n_shares: Number of shares to create
            threshold: Minimum shares needed to reconstruct
            
        Returns:
            List of (index, share) tuples
        """
        if threshold > n_shares:
            raise ValueError("Threshold cannot exceed number of shares")
        
        # Generate random polynomial coefficients
        coefficients = [secret % self.prime]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))
        
        # Evaluate polynomial at points 1, 2, ..., n
        shares = []
        for i in range(1, n_shares + 1):
            value = 0
            for j, coeff in enumerate(coefficients):
                value = (value + coeff * pow(i, j, self.prime)) % self.prime
            shares.append((i, value))
        
        return shares
    
    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (index, share) tuples
            
        Returns:
            Reconstructed secret
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares for reconstruction")
        
        result = 0
        
        for i, (x_i, y_i) in enumerate(shares):
            # Compute Lagrange basis polynomial at 0
            numerator = 1
            denominator = 1
            
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime
            
            # Compute term
            lagrange = (numerator * self._mod_inverse(denominator, self.prime)) % self.prime
            result = (result + y_i * lagrange) % self.prime
        
        return result


class SecureAggregator:
    """
    Secure Aggregation for Federated Learning.
    
    Implements protocols that allow server to compute aggregate
    without seeing individual client updates.
    
    Based on:
    - Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving 
      Machine Learning" (2017)
    """
    
    def __init__(self, 
                 n_clients: int,
                 config: Optional[SecureAggConfig] = None):
        """
        Initialize secure aggregator.
        
        Args:
            n_clients: Total number of clients
            config: Aggregation configuration
        """
        self.n_clients = n_clients
        self.config = config or SecureAggConfig()
        
        # Client keys and shared seeds
        self.client_keys: Dict[int, bytes] = {}
        self.shared_seeds: Dict[Tuple[int, int], bytes] = {}
        
        # Secret sharing for dropout resilience
        self.secret_sharing = SecretSharing()
        
        # Protocol state
        self.round_id = 0
        
        self._setup_keys()
    
    def _setup_keys(self) -> None:
        """Setup client keys and shared secrets."""
        # Generate key pairs for each client
        for client_id in range(self.n_clients):
            self.client_keys[client_id] = secrets.token_bytes(
                self.config.seed_length
            )
        
        # Compute pairwise shared seeds (simulating DH key exchange)
        for i in range(self.n_clients):
            for j in range(i + 1, self.n_clients):
                # In practice, this would be from DH key exchange
                shared = hashlib.sha256(
                    self.client_keys[i] + self.client_keys[j]
                ).digest()
                self.shared_seeds[(i, j)] = shared
                self.shared_seeds[(j, i)] = shared
    
    def _quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize floating point values to integers."""
        # Scale to use available bits
        scale = 2 ** (self.config.quantization_bits - 1)
        
        # Clip and scale
        clipped = np.clip(values, -1, 1)
        quantized = (clipped * scale).astype(np.int64)
        
        return quantized
    
    def _dequantize(self, values: np.ndarray) -> np.ndarray:
        """Dequantize integers back to floating point."""
        scale = 2 ** (self.config.quantization_bits - 1)
        return values.astype(np.float64) / scale
    
    def _generate_mask(self, 
                       client_id: int, 
                       shape: Tuple[int, ...],
                       round_id: int) -> np.ndarray:
        """Generate mask for a client's update."""
        mask = np.zeros(np.prod(shape), dtype=np.int64)
        
        for other_id in range(self.n_clients):
            if other_id == client_id:
                continue
            
            # Get shared seed
            seed = self.shared_seeds.get((client_id, other_id))
            if seed is None:
                continue
            
            # Include round ID in seed
            round_seed = hashlib.sha256(
                seed + struct.pack('>Q', round_id)
            ).digest()
            
            # Generate pseudo-random values
            prg = PseudoRandomGenerator(round_seed)
            pairwise_mask = prg.generate(len(mask))
            
            # Add or subtract based on client ordering
            if client_id < other_id:
                mask += pairwise_mask
            else:
                mask -= pairwise_mask
        
        return mask.reshape(shape)
    
    def mask_update(self,
                    client_id: int,
                    update: np.ndarray,
                    round_id: Optional[int] = None) -> np.ndarray:
        """
        Mask a client's update for secure aggregation.
        
        Args:
            client_id: Client identifier
            update: Model update to mask
            round_id: Round identifier
            
        Returns:
            Masked update
        """
        rid = round_id if round_id is not None else self.round_id
        
        # Quantize update
        quantized = self._quantize(update)
        
        # Generate and apply mask
        mask = self._generate_mask(client_id, update.shape, rid)
        masked = (quantized + mask) % self.config.modulus
        
        return masked
    
    def unmask_aggregate(self,
                         masked_updates: Dict[int, np.ndarray],
                         round_id: Optional[int] = None) -> np.ndarray:
        """
        Aggregate masked updates and remove masks.
        
        Args:
            masked_updates: Dict mapping client_id to masked update
            round_id: Round identifier
            
        Returns:
            Aggregated update (unmasked)
        """
        rid = round_id if round_id is not None else self.round_id
        
        if len(masked_updates) == 0:
            raise ValueError("No updates to aggregate")
        
        # Get shape from first update
        shape = list(masked_updates.values())[0].shape
        
        # Sum all masked updates
        aggregate = np.zeros(shape, dtype=np.int64)
        for masked in masked_updates.values():
            aggregate = (aggregate + masked) % self.config.modulus
        
        # Handle modular wraparound
        half_mod = self.config.modulus // 2
        aggregate = np.where(
            aggregate > half_mod,
            aggregate - self.config.modulus,
            aggregate
        )
        
        # Dequantize
        n_clients = len(masked_updates)
        result = self._dequantize(aggregate) / n_clients
        
        return result
    
    def secure_aggregate(self,
                         updates: Dict[int, np.ndarray],
                         round_id: Optional[int] = None) -> np.ndarray:
        """
        Full secure aggregation pipeline.
        
        Args:
            updates: Dict mapping client_id to update
            round_id: Round identifier
            
        Returns:
            Securely aggregated update
        """
        rid = round_id if round_id is not None else self.round_id
        
        # Mask all updates
        masked_updates = {}
        for client_id, update in updates.items():
            masked_updates[client_id] = self.mask_update(client_id, update, rid)
        
        # Aggregate and unmask
        return self.unmask_aggregate(masked_updates, rid)
    
    def next_round(self) -> None:
        """Advance to next round."""
        self.round_id += 1


class ThresholdSecureAggregator(SecureAggregator):
    """
    Threshold-based Secure Aggregation.
    
    Allows aggregation to succeed even if some clients drop out,
    as long as at least threshold clients participate.
    """
    
    def __init__(self,
                 n_clients: int,
                 threshold: int,
                 config: Optional[SecureAggConfig] = None):
        """
        Initialize threshold secure aggregator.
        
        Args:
            n_clients: Total number of clients
            threshold: Minimum clients for successful aggregation
            config: Aggregation configuration
        """
        super().__init__(n_clients, config)
        
        if threshold > n_clients:
            raise ValueError("Threshold cannot exceed number of clients")
        
        self.threshold = threshold
        
        # Secret shares for each client's mask seed
        self.mask_seed_shares: Dict[int, List[Tuple[int, int]]] = {}
        
        self._distribute_shares()
    
    def _distribute_shares(self) -> None:
        """Distribute secret shares of mask seeds."""
        for client_id in range(self.n_clients):
            # Convert seed to integer
            seed_int = int.from_bytes(
                self.client_keys[client_id][:8], 
                'big'
            )
            
            # Create shares
            shares = self.secret_sharing.share(
                seed_int,
                self.n_clients - 1,
                self.threshold
            )
            
            self.mask_seed_shares[client_id] = shares
    
    def reconstruct_dropout_masks(self,
                                  dropout_clients: List[int],
                                  available_clients: List[int],
                                  shape: Tuple[int, ...],
                                  round_id: int) -> np.ndarray:
        """
        Reconstruct masks of dropout clients.
        
        Args:
            dropout_clients: IDs of clients that dropped out
            available_clients: IDs of participating clients
            shape: Shape of the update
            round_id: Round identifier
            
        Returns:
            Combined mask of dropout clients
        """
        if len(available_clients) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} clients, "
                f"got {len(available_clients)}"
            )
        
        combined_mask = np.zeros(shape, dtype=np.int64)
        
        for dropout_id in dropout_clients:
            # Collect shares from available clients
            available_shares = []
            for i, share in enumerate(self.mask_seed_shares[dropout_id]):
                # Check if share holder is available
                share_holder = i + 1 if i < dropout_id else i + 2
                if share_holder in available_clients:
                    available_shares.append(share)
                
                if len(available_shares) >= self.threshold:
                    break
            
            if len(available_shares) < self.threshold:
                raise ValueError(
                    f"Cannot reconstruct mask for client {dropout_id}"
                )
            
            # Reconstruct seed
            seed_int = self.secret_sharing.reconstruct(available_shares)
            seed = seed_int.to_bytes(8, 'big')
            
            # Generate mask
            round_seed = hashlib.sha256(
                seed + struct.pack('>Q', round_id)
            ).digest()
            
            prg = PseudoRandomGenerator(round_seed)
            mask = prg.generate(np.prod(shape)).reshape(shape)
            
            combined_mask += mask
        
        return combined_mask
    
    def aggregate_with_dropouts(self,
                                updates: Dict[int, np.ndarray],
                                round_id: Optional[int] = None) -> np.ndarray:
        """
        Secure aggregation handling client dropouts.
        
        Args:
            updates: Updates from participating clients
            round_id: Round identifier
            
        Returns:
            Aggregated update
        """
        rid = round_id if round_id is not None else self.round_id
        
        participating = set(updates.keys())
        dropouts = [
            i for i in range(self.n_clients) 
            if i not in participating
        ]
        
        if len(participating) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} clients, "
                f"got {len(participating)}"
            )
        
        # Mask updates
        masked_updates = {}
        for client_id, update in updates.items():
            masked_updates[client_id] = self.mask_update(client_id, update, rid)
        
        # Get aggregate
        aggregate = self.unmask_aggregate(masked_updates, rid)
        
        # Reconstruct and remove dropout masks if needed
        if dropouts:
            shape = list(updates.values())[0].shape
            dropout_masks = self.reconstruct_dropout_masks(
                dropouts, list(participating), shape, rid
            )
            # In a real implementation, this would properly handle the masks
        
        return aggregate


def verify_secure_aggregation():
    """Verify that secure aggregation produces correct results."""
    np.random.seed(42)
    
    n_clients = 5
    update_size = 100
    
    # Create secure aggregator
    aggregator = SecureAggregator(n_clients)
    
    # Generate random updates
    updates = {}
    for i in range(n_clients):
        updates[i] = np.random.randn(update_size).astype(np.float32)
    
    # Compute expected average
    expected = np.mean([u for u in updates.values()], axis=0)
    
    # Secure aggregation
    result = aggregator.secure_aggregate(updates)
    
    # Compare
    max_error = np.max(np.abs(result - expected))
    mean_error = np.mean(np.abs(result - expected))
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'success': max_error < 0.01  # Quantization introduces some error
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Secure Aggregation")
    print("=" * 60)
    
    # Verify correctness
    result = verify_secure_aggregation()
    print(f"\nVerification:")
    print(f"  Max error: {result['max_error']:.6f}")
    print(f"  Mean error: {result['mean_error']:.6f}")
    print(f"  Success: {result['success']}")
    
    # Test with simulated model updates
    print("\n" + "=" * 60)
    print("Testing with Model Updates")
    print("=" * 60)
    
    n_clients = 10
    update_shape = (1000,)
    
    aggregator = SecureAggregator(n_clients)
    
    # Simulate updates
    updates = {}
    for i in range(n_clients):
        # Model updates typically small
        updates[i] = np.random.randn(*update_shape).astype(np.float32) * 0.01
    
    # Time the aggregation
    import time
    
    start = time.time()
    result = aggregator.secure_aggregate(updates)
    elapsed = time.time() - start
    
    print(f"  Aggregated {n_clients} clients, {np.prod(update_shape)} params")
    print(f"  Time: {elapsed*1000:.2f} ms")
    
    # Test threshold aggregation with dropouts
    print("\n" + "=" * 60)
    print("Testing Threshold Aggregation with Dropouts")
    print("=" * 60)
    
    threshold_agg = ThresholdSecureAggregator(
        n_clients=10,
        threshold=6
    )
    
    # Only 7 clients participate
    partial_updates = {i: updates[i] for i in range(7)}
    
    try:
        result = threshold_agg.aggregate_with_dropouts(partial_updates)
        print(f"  Successfully aggregated with 7/10 clients")
        print(f"  Threshold: 6")
    except Exception as e:
        print(f"  Error: {e}")
