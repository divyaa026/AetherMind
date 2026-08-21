# Research Experiment Suite
## Federated Learning with Differential Privacy for Mental Health Prediction

A comprehensive, automated experiment suite for conducting research on federated learning with differential privacy applied to mental health prediction. This suite runs all validation tests, generates publication-ready results, and creates a final summary report.

---

## 📋 Overview

This research project implements a federated learning (FL) system using FedAvg with Differential Privacy (DP-SGD via Opacus) for mental health prediction. The system trains models on synthetic mental health data partitioned across simulated clients.

### Research Questions

1. **Data Quality**: Is the synthetic dataset representative and suitable for FL research?
2. **Convergence**: Does federated learning converge, and what is the impact of DP?
3. **Privacy-Utility Tradeoff**: How does privacy budget (ε) affect model performance?
4. **Robustness**: How does the system perform under challenging real-world conditions?

---

## 🚀 Quick Start

### 1. Installation

```bash
cd research_experiments
pip install -r requirements.txt
```

### 2. Run All Experiments

```bash
python run_research_experiments.py
```

### 3. Run Specific Experiments

```bash
# Run only data quality and convergence
python run_research_experiments.py --experiments data_quality convergence

# Run with custom config
python run_research_experiments.py --config custom_config.yaml
```

---

## 📁 Directory Structure

```
research_experiments/
├── run_research_experiments.py      # Master experiment runner
├── config.yaml                       # Configuration file
├── requirements.txt                  # Python dependencies
├── experiment_data_quality.py        # Experiment 1: Data validation
├── experiment_convergence.py         # Experiment 2: Convergence analysis
├── experiment_privacy_tradeoff.py    # Experiment 3: Privacy-utility tradeoff
├── experiment_robustness.py          # Experiment 4: Robustness testing
├── README.md                         # This file
└── results/                          # Auto-created results directory
    ├── figures/                      # Generated plots and visualizations
    │   ├── data_quality/
    │   ├── convergence/
    │   ├── privacy_tradeoff/
    │   └── robustness/
    ├── research_summary.md           # Final summary report
    ├── consolidated_results_*.json   # All experiment results
    └── experiment_*.log              # Execution logs
```

---

## 🔬 Experiments

### Experiment 1: Data Quality Validation

**Purpose**: Validate the synthetic dataset for federated learning research.

**Tests**:
- Class Distribution: Global and per-client label ratios
- Feature-Label Correlation: Point-Biserial correlations
- Temporal Check: Auto-correlation in sequential features
- Non-IID Validation: Distribution divergence across clients

**Outputs**:
- `figures/data_quality/data_quality_diagnostics.png`
- `figures/data_quality/data_diagnostic_report.json`

---

### Experiment 2: Convergence Analysis

**Purpose**: Compare convergence behavior of three training approaches.

**Protocol**:
1. **Centralized Oracle**: Train on pooled data (no FL, no DP) - upper bound
2. **Federated Baseline**: Standard FedAvg without DP
3. **Our Method (FL+DP)**: FedAvg with DP-SGD (ε=1.0, δ=1e-5)

Train for 50 communication rounds and track test accuracy/F1-score.

**Outputs**:
- `figures/convergence/convergence_plot.png` - Three-curve comparison
- Final performance gap between FL+DP and Centralized Oracle

---

### Experiment 3: Privacy-Utility Tradeoff

**Purpose**: Quantify the privacy-utility tradeoff (core experiment).

**Protocol**:
- Sweep over 5 epsilon values: [0.2, 0.5, 1.0, 2.0, 5.0]
- Keep δ fixed at 1e-5
- Record: Accuracy, F1-Score, AUC-PR, and formal privacy guarantee

**Outputs**:
- `figures/privacy_tradeoff/privacy_utility_tradeoff.png` - **CRITICAL FIGURE**
  - Line plot: ε (x-axis) vs. Accuracy/F1 (y-axis)
  - Horizontal line for FL baseline (no DP)
- Table comparing all metrics across ε values

---

### Experiment 4: Robustness Testing

**Purpose**: Test performance under realistic, challenging conditions.

**Protocol**:

1. **Non-IID Stress Test**: 
   - Create extreme non-IID split (sort users by stress level)
   - Compare accuracy to IID baseline

2. **Partial Participation**: 
   - Simulate 30% client participation per round
   - Compare convergence and final accuracy

3. **Membership Inference Attack**: 
   - Calculate loss on training vs. held-out data
   - AUC metric: closer to 0.5 = better defense

**Outputs**:
- `figures/robustness/robustness_report.png` - Multi-panel analysis
- Attack AUC score and privacy protection rating

---

## ⚙️ Configuration

Edit `config.yaml` to customize experiments:

```yaml
# Key Configuration Options

federated:
  num_clients: 10              # Number of simulated clients
  communication_rounds: 50      # FL training rounds
  local_epochs: 5              # Local training epochs per round

privacy:
  target_epsilon: 1.0          # Default privacy budget
  target_delta: 1.0e-5         # Privacy parameter
  epsilon_sweep: [0.2, 0.5, 1.0, 2.0, 5.0]  # For tradeoff experiment

model:
  type: "lstm"
  hidden_size: 64
  num_layers: 2
  sequence_length: 7           # Past 7 days

training:
  batch_size: 32
  learning_rate: 0.001
  
experiments:
  run_data_quality: true
  run_convergence: true
  run_privacy_tradeoff: true
  run_robustness: true
```

---

## 📊 Output Files

### Generated Figures

1. **Data Quality Diagnostics** (`data_quality_diagnostics.png`)
   - Class distribution across clients
   - Feature-label correlations
   - Feature distributions by risk level
   - Client divergence heatmap

2. **Convergence Plot** (`convergence_plot.png`)
   - Accuracy and F1-score over communication rounds
   - Three curves: Centralized, FL, FL+DP

3. **Privacy-Utility Tradeoff** (`privacy_utility_tradeoff.png`) ⭐ **KEY FIGURE**
   - Accuracy and F1-score vs. epsilon
   - Shows privacy cost and optimal operating point

4. **Robustness Report** (`robustness_report.png`)
   - Non-IID stress test comparison
   - Partial participation results
   - Membership inference attack gauge
   - Summary table

### Summary Report

`results/research_summary.md` contains:
- Executive Summary
- Detailed results for each experiment
- Publication-ready tables
- Research question answers
- Conclusions and recommendations

---

## 🔧 Advanced Usage

### Run Individual Experiments

```python
# Example: Run only convergence experiment
import yaml
from experiment_convergence import run_convergence_experiment

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

results = run_convergence_experiment(config)
print(f"Final accuracy: {results['federated_dp']['final_accuracy']:.4f}")
```

### Customize Epsilon Values

Edit `config.yaml`:
```yaml
experiments:
  privacy_tradeoff:
    epsilon_values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
```

### Use Real Data

Replace synthetic data generation in each experiment module with your data loader:

```python
def load_data(self):
    # Load your federated datasets
    return global_df, client_dfs
```

---

## 📈 Expected Results

Based on typical FL+DP experiments:

- **Baseline (Centralized)**: ~85-90% accuracy
- **Federated (no DP)**: ~82-88% accuracy (2-5% overhead)
- **Federated + DP (ε=1.0)**: ~78-85% accuracy (5-10% overhead)
- **Privacy Protection**: MIA AUC < 0.6 (strong defense)
- **Non-IID Degradation**: < 15% accuracy drop
- **Partial Participation**: Graceful degradation

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# In config.yaml
resources:
  device: "cpu"  # Use CPU instead of CUDA
```

### Slow Execution
```yaml
# Reduce data/rounds
federated:
  num_clients: 5
  communication_rounds: 20
```

### Import Errors
```bash
pip install --upgrade torch torchvision opacus scikit-learn
```

---

## 📝 Citation

If you use this experiment suite in your research, please cite:

```bibtex
@article{your_paper_2026,
  title={Federated Learning with Differential Privacy for Mental Health Prediction},
  author={Your Name},
  journal={Conference/Journal},
  year={2026}
}
```

---

## 📄 License

This research code is provided for academic and research purposes.

---

## 🤝 Contributing

For questions or contributions, please contact the research team.

---

## ✅ Validation Checklist

Before running experiments:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `config.yaml` reviewed and customized
- [ ] Sufficient disk space for results (~500MB)
- [ ] (Optional) CUDA-capable GPU for faster training

After running experiments:
- [ ] Check `results/experiment_*.log` for errors
- [ ] Verify all figures generated in `results/figures/`
- [ ] Review `results/research_summary.md`
- [ ] Validate metrics are within expected ranges

---

## 🎯 Research Milestones

- [x] Implement data quality validation
- [x] Implement convergence analysis
- [x] Implement privacy-utility tradeoff
- [x] Implement robustness testing
- [x] Generate publication-ready figures
- [x] Create automated summary report
- [ ] Run on real mental health datasets
- [ ] Integrate with production FL frameworks (Flower, PySyft)
- [ ] Add more sophisticated privacy attacks
- [ ] Extend to multimodal data

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production-Ready ✅
