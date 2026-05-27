"""Tests for FedAvg aggregation logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import pytest
from backend.models.fedavg import FedAvg


# ---------- helpers ----------
class TinyModel(nn.Module):
    """Minimal 2-layer model for deterministic testing."""
    def __init__(self, in_f=4, h=8, out_f=2):
        super().__init__()
        self.fc1 = nn.Linear(in_f, h)
        self.fc2 = nn.Linear(h, out_f)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _clone_state(sd):
    return {k: v.clone() for k, v in sd.items()}


# ---------- tests ----------
class TestFedAvg:
    """Unit tests for backend.models.fedavg.FedAvg."""

    def test_aggregate_two_clients_identical_weights(self):
        """Two clients with identical weights should produce same weights."""
        m = TinyModel()
        sd = _clone_state(m.state_dict())
        fed = FedAvg(m)
        result = fed.aggregate([sd, sd])
        for k in result.state_dict():
            assert torch.allclose(result.state_dict()[k], sd[k]), (
                f"Key {k} mismatch after averaging identical states"
            )

    def test_aggregate_two_clients_mean(self):
        """Average of two different client states should be their mean."""
        m1, m2 = TinyModel(), TinyModel()
        torch.manual_seed(42)
        for p in m1.parameters():
            nn.init.normal_(p, mean=0.0, std=1.0)
        torch.manual_seed(99)
        for p in m2.parameters():
            nn.init.normal_(p, mean=0.0, std=1.0)

        global_m = TinyModel()
        fed = FedAvg(global_m)
        result = fed.aggregate([m1.state_dict(), m2.state_dict()])

        for k in result.state_dict():
            expected = (m1.state_dict()[k].float() + m2.state_dict()[k].float()) / 2.0
            assert torch.allclose(result.state_dict()[k], expected, atol=1e-6), (
                f"Key {k}: expected mean, got {result.state_dict()[k]}"
            )

    def test_aggregate_three_clients_weighted(self):
        """Three clients: manual mean check."""
        models = []
        for seed in [1, 2, 3]:
            m = TinyModel()
            torch.manual_seed(seed)
            for p in m.parameters():
                nn.init.uniform_(p, -1.0, 1.0)
            models.append(m)

        global_m = TinyModel()
        fed = FedAvg(global_m)
        result = fed.aggregate([m.state_dict() for m in models])

        for k in result.state_dict():
            expected = torch.stack([m.state_dict()[k].float() for m in models]).mean(dim=0)
            assert torch.allclose(result.state_dict()[k], expected, atol=1e-6), (
                f"Key {k}: 3-client mean mismatch"
            )

    def test_global_model_mutated_in_place(self):
        """After aggregate, the same global_model object should be updated."""
        m = TinyModel()
        original_keys = set(m.state_dict().keys())
        fed = FedAvg(m)
        client_sd = _clone_state(m.state_dict())
        returned = fed.aggregate([client_sd])
        assert returned is m, "aggregate should return the same global_model object"
        assert set(m.state_dict().keys()) == original_keys, "state_dict keys changed"

    def test_single_client_passthrough(self):
        """A single client's weights should pass through unchanged."""
        m = TinyModel()
        torch.manual_seed(7)
        for p in m.parameters():
            nn.init.normal_(p)
        sd = _clone_state(m.state_dict())

        global_m = TinyModel()
        fed = FedAvg(global_m)
        result = fed.aggregate([sd])
        for k in result.state_dict():
            assert torch.allclose(result.state_dict()[k], sd[k], atol=1e-7)

    def test_aggregate_preserves_shape(self):
        """All parameter shapes must be preserved after aggregation."""
        m = TinyModel()
        shapes_before = {k: v.shape for k, v in m.state_dict().items()}
        fed = FedAvg(m)
        states = [_clone_state(m.state_dict()) for _ in range(5)]
        result = fed.aggregate(states)
        for k, shape in shapes_before.items():
            assert result.state_dict()[k].shape == shape, (
                f"Key {k}: shape changed from {shape} to {result.state_dict()[k].shape}"
            )

    def test_aggregate_converges_with_repeated_rounds(self):
        """Multiple aggregation rounds should not blow up or collapse weights."""
        m = TinyModel()
        torch.manual_seed(0)
        for p in m.parameters():
            nn.init.normal_(p, std=0.1)
        fed = FedAvg(m)

        for _ in range(10):
            states = []
            for s in range(NUM_CLIENTS := 3):
                clone = _clone_state(m.state_dict())
                # Add small noise to simulate local training divergence
                clone = {k: v + torch.randn_like(v) * 0.01 for k, v in clone.items()}
                states.append(clone)
            m = fed.aggregate(states)

        # Weights should still be finite (no NaN/Inf)
        for k, v in m.state_dict().items():
            assert torch.isfinite(v).all(), f"Key {k} has NaN/Inf after 10 rounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
