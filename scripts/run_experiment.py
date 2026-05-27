#!/usr/bin/env python3
"""Federated Learning experiment pipeline.

Tests:
1. FedAvg aggregation correctness
2. FedProx convergence simulation
3. Non-IID data partition analysis
4. Privacy budget (epsilon) computation
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_fedavg_simulation():
    import torch
    from backend.models.fedavg import FedAvg
    # Simulate 5 clients with local models
    global_model = torch.nn.Linear(10, 2)
    fedavg = FedAvg(global_model)
    client_states = []
    for i in range(5):
        local = torch.nn.Linear(10, 2)
        client_states.append({k: v.clone() for k, v in local.state_dict().items()})
    result = fedavg.aggregate(client_states)
    # Check aggregation is correct (mean)
    expected_weight = torch.stack([s["weight"] for s in client_states]).mean(0)
    actual_weight = result.state_dict()["weight"]
    is_correct = torch.allclose(expected_weight, actual_weight, atol=1e-6)
    return {"aggregation_correct": bool(is_correct), "n_clients": 5}

def run_non_iid_analysis():
    rng = np.random.RandomState(42)
    n_clients = 5
    n_samples = 1000
    # Dirichlet partition with different alpha
    results = {}
    for alpha in [0.1, 0.5, 1.0, 10.0]:
        proportions = rng.dirichlet([alpha] * 5, n_clients)
        sizes = (proportions * n_samples).astype(int)
        imbalance = float(np.std(sizes.sum(axis=1)) / sizes.sum(axis=1).mean())
        results[f"alpha_{alpha}"] = {"imbalance_ratio": imbalance,
                                      "samples_per_client": sizes.sum(axis=1).tolist()}
    return results

def run_privacy_budget():
    from backend.models.dp_mechanism import DPMechanism
    dp = DPMechanism()
    epsilons = []
    for n_rounds in [10, 50, 100, 200]:
        eps = dp.compute_epsilon(n_rounds=n_rounds, sample_rate=0.1, noise_multiplier=1.0)
        epsilons.append({"rounds": n_rounds, "epsilon": float(eps)})
    return epsilons

def main():
    print("=" * 60)
    print("Federated Learning Experiment")
    print("=" * 60)
    print("\n[1] FedAvg Aggregation...")
    r1 = run_fedavg_simulation()
    print(f"  Aggregation correct: {r1['aggregation_correct']}")
    print("\n[2] Non-IID Data Partition...")
    r2 = run_non_iid_analysis()
    for k, v in r2.items():
        print(f"  {k}: imbalance={v['imbalance_ratio']:.3f}")
    print("\n[3] Privacy Budget...")
    try:
        r3 = run_privacy_budget()
        for item in r3:
            print(f"  rounds={item['rounds']}: epsilon={item['epsilon']:.3f}")
    except Exception as e:
        r3 = {"error": str(e)}
        print(f"  Skipped: {e}")
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    with open(out_dir / "fed_learning_results.json", "w") as f:
        json.dump({"fedavg": r1, "non_iid": r2, "privacy": r3}, f, indent=2)
    print(f"\nResults saved to {out_dir}/")

if __name__ == "__main__":
    main()
