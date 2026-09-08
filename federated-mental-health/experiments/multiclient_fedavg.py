#!/usr/bin/env python3
"""
Phase 2 Step 3: multi-client FedAvg (no DP yet), using the real
federated/server.py + federated/client.py - not reimplemented here.

Loads the winning (loss config, threshold) from Step 2's
phase2_imbalance_sweep_results.json and checks whether it survives being
split across multiple simulated clients and averaged, which was explicitly
left untested by Phase 1.

Mirrors the output shape of the original broken
experiments/quick_test/experiment_results.json (same top-level keys) so the
two are directly comparable, but with the full precision/recall/f1/auc_roc/
auc_pr per round instead of just accuracy/auc_roc/f1.

Does NOT modify federated/client.py or federated/server.py. client.py's
FederatedClient.train_local() hardcodes TrainingConfig's loss_type/
pos_weight/focal_alpha/focal_gamma at their defaults (ClientConfig has no
fields for them at all), so a small local subclass
(ConfigurableFederatedClient) is used to actually forward Step 2's winning
loss config - this is additive (a new subclass in this file), not a rewrite
of the shared client code.
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

torch.set_num_threads(2)

FEDERATED_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FEDERATED_ROOT.parent
sys.path.insert(0, str(FEDERATED_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research_experiments"))

from realistic_data_generator import generate_realistic_sequential_data  # noqa: E402
from models.train_local import LocalTrainer, TrainingConfig  # noqa: E402
from data.federated_partition import FederatedPartitioner, PartitionConfig  # noqa: E402
from evaluation.metrics import MetricsCalculator  # noqa: E402
from federated.server import FederatedServer, ServerConfig  # noqa: E402
from federated.client import FederatedClient, ClientConfig  # noqa: E402

SEQ_LEN = 7
POSITIVE_RATIO = 0.03
N_USERS = 250
N_SAMPLES = 1500
DATA_SEED = 0


class ConfigurableFederatedClient(FederatedClient):
    """FederatedClient that actually forwards a loss configuration.

    See module docstring: the real train_local() ignores loss_type /
    pos_weight / focal_alpha / focal_gamma entirely.
    """

    def __init__(self, *args, loss_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config or {}

    def train_local(self):
        if self.local_model is None:
            raise RuntimeError("Must receive global model before training")

        train_config = TrainingConfig(
            epochs=self.config.local_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_dp=self.config.use_dp,
            dp_epsilon=self.config.dp_epsilon,
            dp_delta=self.config.dp_delta,
            max_grad_norm=self.config.dp_max_grad_norm,
            early_stopping_patience=self.config.patience,
            **self.loss_config,
        )

        trainer = LocalTrainer(model=self.local_model, config=train_config)
        history = trainer.fit(
            X_train=self.X_train, y_train=self.y_train,
            X_val=self.X_val, y_val=self.y_val,
        )

        model_state = trainer.get_model_state()
        self.rounds_participated += 1
        train_loss = float(history["train_loss"][-1]) if history.get("train_loss") else 0.0
        self.training_history.append({"round": self.rounds_participated, "final_loss": train_loss})

        training_metrics = {
            "client_id": self.client_id,
            "n_samples": self.n_samples,
            "final_loss": train_loss,
        }
        return model_state, training_metrics


def split_by_user(X, y, user_ids, seed, train_frac=0.7, val_frac=0.15):
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
        (X[train_mask], y[train_mask], user_ids[train_mask]),
        (X[val_mask], y[val_mask], user_ids[val_mask]),
        (X[test_mask], y[test_mask], user_ids[test_mask]),
    )


def stratified_partition(X, y, user_ids, n_clients, seed):
    """Guarantees every client gets an (approximately) equal share of
    positive examples - used only as a fallback if IID partitioning
    produces a zero-positive client."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0].copy()
    neg_idx = np.where(y == 0)[0].copy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_splits = np.array_split(pos_idx, n_clients)
    neg_splits = np.array_split(neg_idx, n_clients)

    partitions = {}
    for cid in range(n_clients):
        idx = np.concatenate([pos_splits[cid], neg_splits[cid]])
        rng.shuffle(idx)
        partitions[cid] = {"X": X[idx], "y": y[idx], "user_ids": user_ids[idx]}
    return partitions


def load_winner():
    sweep_path = FEDERATED_ROOT / "experiments" / "phase2_imbalance_sweep_results.json"
    with open(sweep_path) as f:
        sweep = json.load(f)
    winner = sweep["winner"]
    loss_config = None
    for lc in sweep["config"]["loss_configs"]:
        if lc["name"] == winner["config_name"]:
            loss_config = {k: v for k, v in lc.items() if k != "name"}
            break
    if loss_config is None:
        raise ValueError(f"Could not find loss config for winner {winner['config_name']}")

    # Resolve "empirical" pos_weight the same way the sweep did, but here we
    # don't know the federated training split's exact ratio ahead of time,
    # so this run computes it fresh from the actual training data at
    # partition-time if needed (see main()).
    return winner["config_name"], loss_config, winner["threshold"]


def run_experiment(n_clients, n_rounds, winner_name, loss_config, threshold, local_epochs=2):
    calc = MetricsCalculator()
    results = {
        "experiment_name": f"phase2_multiclient_fedavg_{n_clients}clients",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_clients": n_clients,
            "n_rounds": n_rounds,
            "local_epochs": local_epochs,
            "n_users": N_USERS,
            "winner_loss_config_name": winner_name,
            "winner_loss_config": loss_config,
            "winner_threshold": threshold,
        },
        "phases": {},
        "final_metrics": {},
    }

    np.random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)

    X, y, user_ids = generate_realistic_sequential_data(
        n_samples=N_SAMPLES, n_users=N_USERS, seq_length=SEQ_LEN,
        positive_ratio=POSITIVE_RATIO, seed=DATA_SEED,
    )
    y = y.ravel()
    results["phases"]["data_generation"] = {
        "n_samples": int(len(X)), "n_users": N_USERS, "high_risk_rate": float(y.mean()),
    }

    (X_train, y_train, user_ids_train), (X_val, y_val, _), (X_test, y_test, _) = split_by_user(
        X, y, user_ids, seed=DATA_SEED
    )
    results["phases"]["preprocessing"] = {
        "train_size": int(len(X_train)), "val_size": int(len(X_val)), "test_size": int(len(X_test)),
        "input_dim": int(X.shape[-1]), "sequence_length": SEQ_LEN,
        "train_positive": int(y_train.sum()), "val_positive": int(y_val.sum()), "test_positive": int(y_test.sum()),
    }
    print(f"train={len(y_train)} (pos={int(y_train.sum())})  val={len(y_val)} (pos={int(y_val.sum())})  "
          f"test={len(y_test)} (pos={int(y_test.sum())})", flush=True)

    if loss_config.get("pos_weight") == "empirical":
        n_pos, n_neg = float(y_train.sum()), float((y_train == 0).sum())
        loss_config = dict(loss_config, pos_weight=(n_neg / n_pos) if n_pos > 0 else 1.0)

    # --- Partition training data across clients; fall back to stratified
    # partitioning if IID leaves any client with zero positives. ---
    partitioner = FederatedPartitioner(PartitionConfig(n_clients=n_clients, partition_strategy="iid", random_state=DATA_SEED))
    partitions = partitioner.partition(X_train, y_train, user_ids_train)
    pos_counts = {cid: int(p["y"].sum()) for cid, p in partitions.items()}
    print(f"IID partition positive counts per client: {pos_counts}", flush=True)

    stratification_used = False
    if min(pos_counts.values()) == 0:
        stratification_used = True
        print("Zero-positive client found under IID partitioning -> switching to stratified partitioning", flush=True)
        partitions = stratified_partition(X_train, y_train, user_ids_train, n_clients, seed=DATA_SEED)
        pos_counts = {cid: int(p["y"].sum()) for cid, p in partitions.items()}
        print(f"Stratified partition positive counts per client: {pos_counts}", flush=True)

    results["phases"]["partitioning"] = {
        "strategy": "stratified" if stratification_used else "iid",
        "stratification_used": stratification_used,
        "positive_counts_per_client": pos_counts,
        "samples_per_client": {cid: int(len(p["y"])) for cid, p in partitions.items()},
    }

    # --- Set up server + clients ---
    input_dim = X.shape[-1]
    server = FederatedServer(input_dim=input_dim, config=ServerConfig(
        n_rounds=n_rounds, min_clients_per_round=n_clients, client_fraction=1.0,
        aggregation_strategy="fedavg", add_dp_noise=False,
    ))

    clients = []
    for cid in range(n_clients):
        data = partitions[cid]
        client_config = ClientConfig(client_id=cid, local_epochs=local_epochs, batch_size=32, use_dp=False)
        client = ConfigurableFederatedClient(
            client_id=cid, X_train=data["X"], y_train=data["y"], config=client_config, loss_config=loss_config,
        )
        server.register_client(cid, len(data["y"]))
        clients.append(client)

    # --- Round loop ---
    round_metrics = []
    t_start = time.time()
    for round_num in range(n_rounds):
        global_state = server.get_global_model_state()
        client_updates = []
        for client in clients:
            client.receive_global_model(global_state)
            model_state, _ = client.train_local()
            client_updates.append((client.client_id, model_state, client.n_samples))

        server.update_global_model(client_updates)
        server.current_round += 1

        server._set_model_state(server.global_model, server.global_state)
        server.global_model.eval()
        with torch.no_grad():
            logits, _ = server.global_model(torch.FloatTensor(X_val))
            val_probs = torch.sigmoid(logits).numpy().flatten()
        m = calc.compute_classification_metrics(y_val, val_probs, threshold=threshold)
        elapsed = time.time() - t_start
        entry = {
            "round": round_num + 1, "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "auc_roc": m.auc_roc, "auc_pr": m.auc_pr, "accuracy": m.accuracy, "elapsed_sec": round(elapsed, 1),
        }
        round_metrics.append(entry)
        print(f"  round {round_num + 1}/{n_rounds}  f1={m.f1:.3f} auc_roc={m.auc_roc:.3f} "
              f"auc_pr={m.auc_pr:.3f} precision={m.precision:.3f} recall={m.recall:.3f}  "
              f"({elapsed:.0f}s elapsed)", flush=True)

    results["phases"]["training"] = {"n_rounds": n_rounds, "round_metrics": round_metrics}

    # --- Final, one-time test evaluation ---
    server.global_model.eval()
    with torch.no_grad():
        logits, _ = server.global_model(torch.FloatTensor(X_test))
        test_probs = torch.sigmoid(logits).numpy().flatten()
    final = calc.compute_classification_metrics(y_test, test_probs, threshold=threshold)
    results["final_metrics"] = {
        "accuracy": final.accuracy, "balanced_accuracy": final.balanced_accuracy,
        "precision": final.precision, "recall": final.recall, "f1": final.f1,
        "auc_roc": final.auc_roc, "auc_pr": final.auc_pr, "mcc": final.mcc,
    }
    print(f"\nFINAL TEST ({n_clients} clients): f1={final.f1:.3f} precision={final.precision:.3f} "
          f"recall={final.recall:.3f} auc_roc={final.auc_roc:.3f} auc_pr={final.auc_pr:.3f}", flush=True)

    out_path = FEDERATED_ROOT / "experiments" / f"phase2_multiclient_{n_clients}clients_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()

    winner_name, loss_config, threshold = load_winner()
    print(f"Using Step 2 winning config: {winner_name} @ threshold={threshold}  loss_config={loss_config}", flush=True)

    for n_clients in args.n_clients:
        print(f"\n{'=' * 70}\n{n_clients} clients, {args.n_rounds} rounds\n{'=' * 70}", flush=True)
        run_experiment(n_clients, args.n_rounds, winner_name, loss_config, threshold, args.local_epochs)
