# Federated Mental Health Prediction with Differential Privacy

A comprehensive implementation of federated learning for mental health risk prediction with differential privacy guarantees.

## Overview

This project implements a privacy-preserving machine learning system for predicting mental health risk using federated learning. Key features include:

- **Federated Learning**: Train models across distributed clients without sharing raw data
- **Differential Privacy**: Strong mathematical privacy guarantees using DP-SGD
- **Synthetic Data Generation**: Realistic mental health time series data
- **Comprehensive Evaluation**: Classification, calibration, fairness, and privacy metrics
- **Byzantine Resilience**: Robust aggregation strategies

## Project Structure

```
federated-mental-health/
├── data/                      # Data generation and preprocessing
│   ├── synthetic_generator.py # Generate synthetic mental health data
│   ├── preprocess.py          # Feature engineering and normalization
│   └── federated_partition.py # Partition data across clients
├── models/                    # Model architectures
│   ├── architecture.py        # LSTM + Self-Attention model
│   ├── dp_optimizer.py        # Differential privacy optimizer
│   └── train_local.py         # Local training utilities
├── federated/                 # Federated learning components
│   ├── server.py              # Federated server with FedAvg
│   ├── client.py              # Federated client
│   ├── coordinator.py         # Training orchestration
│   └── aggregation.py         # Aggregation strategies
├── privacy/                   # Privacy components
│   ├── dp_accounting.py       # Renyi DP accounting
│   ├── secure_aggregation.py  # Secure aggregation protocol
│   └── attacks.py             # Privacy attack implementations
├── evaluation/                # Evaluation utilities
│   ├── metrics.py             # Classification and clinical metrics
│   ├── privacy_eval.py        # Privacy evaluation
│   └── comparative_analysis.py # Experiment comparison
├── experiments/               # Experiment runners
│   └── run_experiment.py      # Main experiment script
├── tests/                     # Unit tests
├── config.yaml                # Default configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd federated-mental-health

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run with default configuration
python experiments/run_experiment.py

# Run with custom parameters
python experiments/run_experiment.py \
    --n-clients 20 \
    --n-rounds 100 \
    --epsilon 0.5 \
    --experiment-name my_experiment

# Run with config file
python experiments/run_experiment.py --config config.yaml
```

## Key Components

### 1. Synthetic Data Generation

Generates realistic mental health time series data with:
- 10,000+ users, 30 days of daily records
- Correlated features (sleep → stress, exercise → mood)
- Weekend patterns and missing data
- High-risk label based on clinical criteria

```python
from data.synthetic_generator import SyntheticMentalHealthData, SyntheticConfig

config = SyntheticConfig(n_users=10000, n_days=30)
generator = SyntheticMentalHealthData(config)
df = generator.generate()
generator.save('./data/synthetic')
```

### 2. Model Architecture

LSTM with Self-Attention for temporal mental health patterns:

```python
from models.architecture import create_model

model = create_model(
    input_dim=42,
    hidden_dim=128,
    n_layers=2,
    dropout=0.3,
    bidirectional=True
)
```

### 3. Federated Training

```python
from federated.coordinator import FederatedCoordinator, FederatedConfig

config = FederatedConfig(
    n_rounds=50,
    n_clients=10,
    use_dp=True,
    dp_epsilon=1.0
)

coordinator = FederatedCoordinator(config)
coordinator.setup_from_partitions('./data/partitions')
summary = coordinator.train()
```

### 4. Differential Privacy

```python
from privacy.dp_accounting import RDPAccountant

accountant = RDPAccountant(target_delta=1e-5)
accountant.add_mechanism(
    noise_multiplier=1.0,
    sampling_probability=0.01,
    n_steps=1000
)

epsilon, delta, order = accountant.get_privacy_spent()
print(f"Privacy spent: ε={epsilon:.4f}, δ={delta:.2e}")
```

### 5. Evaluation

```python
from evaluation.metrics import MetricsCalculator
from evaluation.privacy_eval import run_comprehensive_privacy_evaluation

# Classification metrics
calculator = MetricsCalculator()
metrics = calculator.compute_all_metrics(y_true, y_prob)

# Privacy evaluation
privacy_results = run_comprehensive_privacy_evaluation(
    model=model,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    dp_params={'epsilon': 1.0, 'delta': 1e-5}
)
```

## Configuration

See `config.yaml` for all configuration options. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clients` | 10 | Number of federated clients |
| `n_rounds` | 50 | Federated training rounds |
| `local_epochs` | 5 | Local training epochs per round |
| `dp_epsilon` | 1.0 | Privacy budget epsilon |
| `dp_delta` | 1e-5 | Privacy budget delta |
| `aggregation` | fedavg | Aggregation strategy |

## Privacy Guarantees

This implementation provides (ε, δ)-differential privacy through:

1. **DP-SGD**: Per-sample gradient clipping and Gaussian noise
2. **RDP Accounting**: Tight privacy budget tracking via Renyi DP
3. **Secure Aggregation**: Aggregation without revealing individual updates
4. **Privacy Audit**: Membership inference attacks to validate guarantees

Target privacy: ε ≤ 1.0, δ = 10⁻⁵

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Matthews Correlation Coefficient

### Calibration
- Expected Calibration Error (ECE)
- Brier Score
- Reliability Diagrams

### Fairness
- Demographic Parity
- Equalized Odds
- Predictive Parity

### Privacy
- Theoretical ε from RDP accounting
- Empirical MIA success rate
- Privacy-utility tradeoff analysis

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code, please cite:

```bibtex
@software{federated_mental_health,
  title={Federated Mental Health Prediction with Differential Privacy},
  author={AetherMind Team},
  year={2024},
  url={https://github.com/aethermind/federated-mental-health}
}
```

## References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
2. Abadi et al., "Deep Learning with Differential Privacy" (2016)
3. Mironov, "Renyi Differential Privacy" (2017)
4. Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (2017)
