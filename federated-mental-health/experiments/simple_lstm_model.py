"""
Simple LSTM model for Phase 2 Steps 2 (rerun)/3/4.

The BatchNorm-in-MentalHealthPredictor diagnostic (swapping the classifier
head's BatchNorm1d for LayerNorm) did not resolve the near-random AUC seen
on research_experiments/realistic_data_generator.py data - see the phase 2
report. Per direction, MentalHealthPredictor debugging stops here and this
simpler architecture is used for Steps 3/4 instead.

Ported from research_experiments/experiment_convergence.py's
SimpleLSTMModel - the one architecture in this repo with a real prior
result on this exact data generator
(research_experiments/results/figures/convergence/convergence_results.json).
Only change from the original: forward() returns raw logits (and None,
for interface parity) instead of sigmoid-applied probabilities, so it is a
drop-in replacement for MentalHealthPredictor within LocalTrainer /
FederatedClient / FederatedServer, all of which unpack (logits, attention)
and apply sigmoid / BCEWithLogits themselves.
"""
import torch.nn as nn


class SimpleLSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention: bool = False):
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits, None


def create_simple_model(input_dim: int, config: dict = None) -> SimpleLSTMPredictor:
    cfg = {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}
    if config:
        cfg.update(config)
    return SimpleLSTMPredictor(input_dim=input_dim, **cfg)
