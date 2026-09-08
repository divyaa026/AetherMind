#!/usr/bin/env python3
"""
Phase 2 Step 2 (rerun): systematic sweep over loss config x decision
threshold for the class-imbalance fix, on a single (non-federated) client.

Uses the project's validated synthetic generator
(research_experiments/realistic_data_generator.py) instead of ad hoc
np.random data, so results are grounded in the same data source behind the
convergence results in research_experiments/results/figures/convergence/.

Model selection uses the validation split only. The test split is touched
exactly once, at the end, with the winning (loss config, threshold) chosen
from validation.

This is the SECOND run of this script. The first (results saved to
phase2_imbalance_sweep_results.json, not overwritten - kept as evidence)
used models.architecture.MentalHealthPredictor and found near-random AUC
(0.42-0.68) regardless of loss config, dataset scale, or epoch budget, and
a BatchNorm-to-LayerNorm diagnostic on that architecture didn't resolve it
either. This run uses simple_lstm_model.SimpleLSTMPredictor instead - the
one architecture in this repo with a real prior result on this exact data
generator - per direction to stop debugging MentalHealthPredictor and
switch models rather than keep investigating.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# This model is small (hidden_dim=128) and batches are small (32); PyTorch's
# default CPU intra-op parallelism (one thread per core) spends more time on
# thread dispatch/sync than on the actual matmuls, making runs far slower
# than a single-threaded run of the same workload. Cap it.
torch.set_num_threads(2)

FEDERATED_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FEDERATED_ROOT.parent
sys.path.insert(0, str(FEDERATED_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research_experiments"))

from realistic_data_generator import generate_realistic_sequential_data  # noqa: E402
from simple_lstm_model import create_simple_model as create_model  # noqa: E402
from models.train_local import LocalTrainer, TrainingConfig  # noqa: E402
from evaluation.metrics import MetricsCalculator  # noqa: E402

SEEDS = [0, 1, 2]
THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.901, 0.05)]
EPOCHS = 15
N_USERS = 200
N_SAMPLES = 1200  # passed to generate_realistic_sequential_data; final sequence
                   # count ends up larger after windowing (see module docstring)
SEQ_LEN = 7
POSITIVE_RATIO = 0.03

LOSS_CONFIGS = [
    {"name": "bce_unweighted", "loss_type": "bce"},
    {"name": "weighted_bce_pw1", "loss_type": "weighted_bce", "pos_weight": 1.0},
    {"name": "weighted_bce_pw3_default", "loss_type": "weighted_bce", "pos_weight": 3.0},
    {"name": "weighted_bce_pw5", "loss_type": "weighted_bce", "pos_weight": 5.0},
    {"name": "weighted_bce_pw10", "loss_type": "weighted_bce", "pos_weight": 10.0},
    {"name": "weighted_bce_pw20", "loss_type": "weighted_bce", "pos_weight": 20.0},
    {"name": "weighted_bce_pw_empirical", "loss_type": "weighted_bce", "pos_weight": "empirical"},
    {"name": "focal_a025_g2_default", "loss_type": "focal", "focal_alpha": 0.25, "focal_gamma": 2.0},
    {"name": "focal_a075_g2", "loss_type": "focal", "focal_alpha": 0.75, "focal_gamma": 2.0},
    {"name": "focal_a09_g2", "loss_type": "focal", "focal_alpha": 0.9, "focal_gamma": 2.0},
]


def split_by_user(X, y, user_ids, seed, train_frac=0.7, val_frac=0.15):
    """Split at the user level so overlapping sliding-window sequences from
    the same user never straddle train/val/test (avoids leakage)."""
    rng = np.random.RandomState(seed)
    unique_users = np.unique(user_ids)
    rng.shuffle(unique_users)

    n_users = len(unique_users)
    n_train = int(n_users * train_frac)
    n_val = int(n_users * val_frac)

    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train:n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])

    def mask_for(users):
        return np.array([uid in users for uid in user_ids])

    train_mask = mask_for(train_users)
    val_mask = mask_for(val_users)
    test_mask = mask_for(test_users)

    return (
        (X[train_mask], y[train_mask]),
        (X[val_mask], y[val_mask]),
        (X[test_mask], y[test_mask]),
    )


def build_training_config(loss_config, y_train):
    kwargs = dict(loss_config)
    name = kwargs.pop("name")

    if kwargs.get("pos_weight") == "empirical":
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        kwargs["pos_weight"] = (n_neg / n_pos) if n_pos > 0 else 1.0

    config = TrainingConfig(
        epochs=EPOCHS,
        batch_size=32,
        learning_rate=0.003,
        use_dp=False,
        **kwargs,
    )
    return name, config


def metrics_at_threshold(calc, y_true, probs, threshold):
    m = calc.compute_classification_metrics(y_true, probs, threshold=threshold)
    return {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "auc_roc": m.auc_roc,
        "auc_pr": m.auc_pr,
    }


def main():
    calc = MetricsCalculator()
    all_entries = []  # every (seed, config, threshold) validation row
    test_cache = {}   # (seed, config_name) -> (y_test, test_probs)

    for seed in SEEDS:
        print(f"\n{'=' * 70}\nSeed {seed}: generating data\n{'=' * 70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        X, y, user_ids = generate_realistic_sequential_data(
            n_samples=N_SAMPLES, n_users=N_USERS, seq_length=SEQ_LEN,
            positive_ratio=POSITIVE_RATIO, seed=seed,
        )
        y = y.ravel()  # generator returns (n, 1); LocalTrainer expects 1-D labels

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_user(
            X, y, user_ids, seed=seed
        )
        print(f"  train={len(y_train)} (pos={int(y_train.sum())}), "
              f"val={len(y_val)} (pos={int(y_val.sum())}), "
              f"test={len(y_test)} (pos={int(y_test.sum())})")

        input_dim = X.shape[-1]

        for loss_config in LOSS_CONFIGS:
            config_name, train_config = build_training_config(loss_config, y_train)
            torch.manual_seed(seed)
            model = create_model(input_dim=input_dim)
            trainer = LocalTrainer(model, train_config)
            trainer.fit(X_train, y_train)

            val_probs = trainer.predict(X_val)
            test_probs = trainer.predict(X_test)
            test_cache[(seed, config_name)] = (y_test, test_probs)

            for threshold in THRESHOLDS:
                m = metrics_at_threshold(calc, y_val, val_probs, threshold)
                all_entries.append({
                    "seed": seed,
                    "config_name": config_name,
                    "loss_config": {k: v for k, v in loss_config.items() if k != "name"},
                    "resolved_pos_weight": train_config.pos_weight,
                    "threshold": threshold,
                    "split": "val",
                    **m,
                })

            best_row = max(
                (e for e in all_entries if e["seed"] == seed and e["config_name"] == config_name),
                key=lambda e: e["f1"],
            )
            default_row = next(
                e for e in all_entries
                if e["seed"] == seed and e["config_name"] == config_name and e["threshold"] == 0.5
            )
            print(f"  {config_name:28s} val F1@0.5={default_row['f1']:.3f}  "
                  f"best val F1={best_row['f1']:.3f}@thr={best_row['threshold']:.2f}  "
                  f"AUC={best_row['auc_roc']:.3f}")

    # --- Aggregate validation results across seeds to pick a winner ---
    from collections import defaultdict
    agg = defaultdict(list)
    for e in all_entries:
        agg[(e["config_name"], e["threshold"])].append(e["f1"])

    mean_f1 = {k: float(np.mean(v)) for k, v in agg.items()}
    std_f1 = {k: float(np.std(v)) for k, v in agg.items()}
    (winner_config, winner_threshold) = max(mean_f1, key=mean_f1.get)
    winner_mean_f1 = mean_f1[(winner_config, winner_threshold)]

    # Also report the best default-threshold (0.5) config, and how close the
    # runner-up is to the winner, to be honest about whether this is a clear
    # winner or a close call.
    ranked = sorted(mean_f1.items(), key=lambda kv: -kv[1])
    top5 = ranked[:5]

    print(f"\n{'=' * 70}\nValidation-selected winner: {winner_config} @ threshold={winner_threshold:.2f} "
          f"(mean val F1={winner_mean_f1:.3f} +/- {std_f1[(winner_config, winner_threshold)]:.3f})\n{'=' * 70}")
    print("Top 5 (config, threshold) by mean validation F1:")
    for (cfg, thr), f1 in top5:
        print(f"  {cfg:28s} thr={thr:.2f}  mean_f1={f1:.3f} +/- {std_f1[(cfg, thr)]:.3f}")

    # --- Final, one-time test evaluation of the winner ---
    test_results = []
    for seed in SEEDS:
        y_test, test_probs = test_cache[(seed, winner_config)]
        m = metrics_at_threshold(calc, y_test, test_probs, winner_threshold)
        test_results.append({"seed": seed, **m})
        print(f"  seed={seed} TEST: f1={m['f1']:.3f} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} auc_roc={m['auc_roc']:.3f} auc_pr={m['auc_pr']:.3f}")

    test_f1s = [r["f1"] for r in test_results]
    print(f"\nFinal test F1: mean={np.mean(test_f1s):.3f} +/- {np.std(test_f1s):.3f} "
          f"(single touch, winner config+threshold chosen from validation only)")

    output = {
        "experiment_name": "phase2_imbalance_sweep_simplelstm",
        "model": "SimpleLSTMPredictor",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": SEEDS,
            "thresholds": THRESHOLDS,
            "epochs": EPOCHS,
            "n_users": N_USERS,
            "n_samples_param": N_SAMPLES,
            "seq_len": SEQ_LEN,
            "positive_ratio_target": POSITIVE_RATIO,
            "loss_configs": LOSS_CONFIGS,
        },
        "validation_entries": all_entries,
        "validation_summary": {
            f"{cfg}|thr={thr:.2f}": {"mean_f1": f1, "std_f1": std_f1[(cfg, thr)]}
            for (cfg, thr), f1 in mean_f1.items()
        },
        "winner": {
            "config_name": winner_config,
            "threshold": winner_threshold,
            "mean_val_f1": winner_mean_f1,
            "std_val_f1": std_f1[(winner_config, winner_threshold)],
            "top5_by_mean_val_f1": [
                {"config_name": cfg, "threshold": thr, "mean_f1": f1, "std_f1": std_f1[(cfg, thr)]}
                for (cfg, thr), f1 in top5
            ],
        },
        "test_results": test_results,
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
    }

    out_path = FEDERATED_ROOT / "experiments" / "phase2_imbalance_sweep_simplelstm_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
