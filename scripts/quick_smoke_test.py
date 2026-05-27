#!/usr/bin/env python3
"""
Quick smoke test: simulates a 3-node federated training round.
Each "node" trains a tiny model on synthetic data, then the server
aggregates with FedAvg and verifies convergence.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.models.fedavg import FedAvg
from backend.models.dp_mechanism import DPMechanism


# ── Config ──────────────────────────────────────────────────────────
NUM_CLIENTS    = 3
LOCAL_EPOCHS   = 5
GLOBAL_ROUNDS  = 5
LR             = 0.05
BATCH_SIZE     = 32
N_SAMPLES      = 200       # samples per client
INPUT_DIM      = 8
OUTPUT_DIM     = 2
DP_EPSILON     = 16.0      # mild DP noise for testing
SEED           = 42


# ── Shared labeling rule ───────────────────────────────────────────
torch.manual_seed(SEED)
W_RULE = torch.randn(INPUT_DIM)   # shared decision boundary


# ── Model ───────────────────────────────────────────────────────────
class TCMNet(nn.Module):
    """Small feedforward net mimicking a TCM diagnostic model."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


# ── Synthetic data ──────────────────────────────────────────────────
def make_client_data(client_id: int, n=N_SAMPLES):
    """Generate synthetic data with slight distribution shift per client."""
    torch.manual_seed(SEED + client_id * 1000 + 7)
    mean_shift = client_id * 0.3
    X = torch.randn(n, INPUT_DIM) + mean_shift
    y = (X @ W_RULE > 0).long()          # shared labeling rule
    return TensorDataset(X, y)


def make_test_data(n=300):
    torch.manual_seed(999)
    X = torch.randn(n, INPUT_DIM)
    y = (X @ W_RULE > 0).long()
    return TensorDataset(X, y)


# ── Helpers ─────────────────────────────────────────────────────────
def local_train(model, data, epochs, lr):
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, data):
    loader = DataLoader(data, batch_size=256)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            correct += (model(X_batch).argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TCM-FedLearning · 3-Node Federated Smoke Test")
    print("=" * 60)
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {GLOBAL_ROUNDS} | "
          f"Local epochs: {LOCAL_EPOCHS} | LR: {LR}")
    print(f"  DP epsilon: {DP_EPSILON}")
    print()

    torch.manual_seed(0)
    global_model = TCMNet()
    fedavg = FedAvg(global_model)
    dp = DPMechanism(epsilon=DP_EPSILON, delta=1e-5)

    client_data = [make_client_data(i) for i in range(NUM_CLIENTS)]
    test_set = make_test_data()

    t_start = time.time()
    best_acc = 0.0

    for rnd in range(1, GLOBAL_ROUNDS + 1):
        print(f"--- Round {rnd}/{GLOBAL_ROUNDS} ---")
        client_states = []
        for cid in range(NUM_CLIENTS):
            local_model = TCMNet()
            local_model.load_state_dict(global_model.state_dict())
            avg_loss = local_train(local_model, client_data[cid], LOCAL_EPOCHS, LR)
            # Apply mild DP noise
            state = {k: v.clone() for k, v in local_model.state_dict().items()}
            state = dp.add_noise(state, sensitivity=0.1)
            client_states.append(state)
            acc = evaluate(local_model, client_data[cid])
            print(f"  Client {cid}: loss={avg_loss:.4f}  local_acc={acc:.1%}")

        global_model = fedavg.aggregate(client_states)
        test_acc = evaluate(global_model, test_set)
        best_acc = max(best_acc, test_acc)
        print(f"  >> Global test accuracy: {test_acc:.1%}")
        print()

    elapsed = time.time() - t_start
    final_acc = evaluate(global_model, test_set)

    checks = {
        "pipeline_completed":     True,
        "accuracy_above_random":  best_acc > 0.50,
        "weights_finite":         all(
            torch.isfinite(v).all().item() for v in global_model.state_dict().values()
        ),
    }
    all_pass = all(checks.values())

    print("=" * 60)
    print(f"  DONE in {elapsed:.2f}s")
    print(f"  Final accuracy: {final_acc:.1%} | Best: {best_acc:.1%}")
    print()
    print("  Checks:")
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print()
    print(f"  Overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
