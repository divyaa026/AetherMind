"""
Deliberately minimal text classifier for Feature 3.

Two Linear layers, ReLU, Dropout - no LSTM, no attention, no BatchNorm.
BatchNorm is specifically excluded because it normalizes using per-batch
statistics mixed across samples, which is incompatible with DP-SGD's
per-sample gradient requirement (Opacus rejects BatchNorm layers outright).
Dropout and LayerNorm are both fine for DP-SGD if needed later; this
version doesn't even need LayerNorm since there's no sequence/recurrent
component whose activations drift across depth.

Takes TF-IDF vectors directly (batch, n_features) - no fake sequence
dimension, unlike the LSTM-based models used earlier in Phase 2.
"""
import torch.nn as nn


class SimpleTextClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, return_attention: bool = False):
        logits = self.net(x)
        return logits, None


def create_text_model(input_dim: int, config: dict = None) -> SimpleTextClassifier:
    cfg = {"hidden_dim": 64, "dropout": 0.2}
    if config:
        cfg.update(config)
    return SimpleTextClassifier(input_dim=input_dim, **cfg)
