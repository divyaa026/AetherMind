#!/usr/bin/env python3
"""
Feature 3, single-client validation (do this before touching federation or
DP at all - the synthetic-data path skipped straight to a federated/DP
sweep and burned a lot of time discovering the base classifier never
converged; this script exists specifically so that mistake isn't repeated).

Plain train/val split, real text data, unweighted BCE (both datasets are
~50/50 balanced, so none of Phase 2's imbalance machinery is needed), the
real LocalTrainer/TrainingConfig unchanged.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)

FEDERATED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEDERATED_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_data_pipeline import prepare_tfidf_split  # noqa: E402
from simple_text_model import create_text_model  # noqa: E402
from models.train_local import LocalTrainer, TrainingConfig  # noqa: E402
from evaluation.metrics import MetricsCalculator  # noqa: E402

SEED = 42
N_SAMPLES = 20000
MAX_FEATURES = 3000
EPOCHS = 15


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading and vectorizing data...", flush=True)
    t0 = time.time()
    X_train, y_train, X_val, y_val, vectorizer = prepare_tfidf_split(
        dataset="suicide", n_samples=N_SAMPLES, max_features=MAX_FEATURES, val_frac=0.2, seed=SEED,
    )
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    print(f"  X_train {X_train.shape}  pos_ratio={y_train.mean():.3f}", flush=True)
    print(f"  X_val   {X_val.shape}  pos_ratio={y_val.mean():.3f}", flush=True)
    print(f"  vocab size {len(vectorizer.vocabulary_)}", flush=True)

    model = create_text_model(input_dim=X_train.shape[1])
    config = TrainingConfig(
        epochs=EPOCHS, batch_size=64, learning_rate=0.001, use_dp=False, loss_type="bce",
    )
    trainer = LocalTrainer(model, config)

    print(f"\nTraining for {EPOCHS} epochs...", flush=True)
    t0 = time.time()
    trainer.fit(X_train, y_train)
    print(f"  trained in {time.time() - t0:.1f}s", flush=True)

    probs = trainer.predict(X_val)
    calc = MetricsCalculator()
    print("\nValidation metrics at several thresholds:", flush=True)
    for thr in [0.3, 0.5, 0.7]:
        m = calc.compute_classification_metrics(y_val, probs, threshold=thr)
        print(f"  thr={thr}: f1={m.f1:.3f} precision={m.precision:.3f} recall={m.recall:.3f} "
              f"accuracy={m.accuracy:.3f} auc_roc={m.auc_roc:.3f} auc_pr={m.auc_pr:.3f}", flush=True)

    m5 = calc.compute_classification_metrics(y_val, probs, threshold=0.5)
    print(f"\n{'CONVERGED' if m5.f1 > 0.7 and m5.auc_roc > 0.85 else 'DID NOT CLEARLY CONVERGE'}: "
          f"F1@0.5={m5.f1:.3f}  AUC-ROC={m5.auc_roc:.3f}", flush=True)


if __name__ == "__main__":
    main()
