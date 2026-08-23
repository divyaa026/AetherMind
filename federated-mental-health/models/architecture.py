"""
Neural Network Architecture for Mental Health Risk Prediction
LSTM with Self-Attention for temporal pattern recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for temporal sequences.
    Allows the model to focus on the most relevant time steps.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize self-attention layer.
        
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head self-attention.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Residual connection
        residual = x
        
        # Linear projections
        Q = self.query(x)  # (batch, seq_len, hidden_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.output_proj(context)
        output = self.output_dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        
        return output, avg_attention


class MentalHealthPredictor(nn.Module):
    """
    LSTM with Self-Attention for mental health risk prediction.
    
    Architecture:
    - Input embedding layer
    - Bidirectional LSTM for temporal encoding
    - Multi-head self-attention for focusing on key time steps
    - Classification head with focal loss support
    
    Features:
    - Attention visualization for interpretability
    - Dropout and batch normalization for regularization
    - Designed for differential privacy compatibility
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 output_dim: int = 1):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features per time step
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            output_dim: Output dimension (1 for binary classification)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention dimension adjustment for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Project LSTM output to attention dimension
        self.lstm_proj = nn.Linear(lstm_output_dim, hidden_dim)
        
        # Self-attention
        self.attention = SelfAttention(hidden_dim, num_heads, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights or None)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2 if bidirectional)
        
        # Project to attention dimension
        lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_dim)
        
        # Self-attention
        attended, attention_weights = self.attention(lstm_out)  # (batch, seq_len, hidden_dim)
        
        # Global average pooling over sequence
        pooled = attended.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (batch, output_dim)
        
        if return_attention:
            return logits, attention_weights
        return logits, None
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights (batch, seq_len, seq_len)
        """
        _, attention = self.forward(x, return_attention=True)
        return attention


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified examples to focus on hard cases.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Predicted logits (before sigmoid)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        # Ensure proper shapes
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal term
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final loss
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy for imbalanced data."""
    
    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize weighted BCE.
        
        Args:
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted BCE loss."""
        pos_weight = torch.tensor([self.pos_weight], device=inputs.device)
        return F.binary_cross_entropy_with_logits(
            inputs.view(-1), 
            targets.view(-1).float(),
            pos_weight=pos_weight
        )


def create_model(input_dim: int, 
                 config: Optional[dict] = None) -> MentalHealthPredictor:
    """
    Factory function to create model with configuration.
    
    Args:
        input_dim: Number of input features
        config: Optional configuration dictionary
        
    Returns:
        Initialized model
    """
    default_config = {
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.3,
        'bidirectional': True,
        'output_dim': 1
    }
    
    if config:
        default_config.update(config)
    
    model = MentalHealthPredictor(
        input_dim=input_dim,
        **default_config
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: MentalHealthPredictor, input_shape: Tuple[int, ...]):
    """Print model summary."""
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(f"\nInput shape: {input_shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("\nLayer summary:")
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    
    # Test forward pass
    x = torch.randn(2, *input_shape)
    with torch.no_grad():
        logits, attention = model(x, return_attention=True)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"Attention shape: {attention.shape}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    input_dim = 42  # Example feature dimension
    seq_len = 7
    batch_size = 32
    
    # Create model
    model = create_model(input_dim)
    
    # Print summary
    model_summary(model, (seq_len, input_dim))
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, 2, (batch_size, 1)).float()
    
    logits, attention = model(x, return_attention=True)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Test losses
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    weighted_bce = WeightedBCELoss(pos_weight=3.0)
    
    fl = focal_loss(logits, y)
    wbce = weighted_bce(logits, y)
    
    print(f"\nFocal Loss: {fl.item():.4f}")
    print(f"Weighted BCE: {wbce.item():.4f}")
