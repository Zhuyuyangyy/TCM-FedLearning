"""Training API endpoints for TCM-FedLearning.

POST /api/train/start   — Kick off a federated training job
GET  /api/train/status   — Query current / latest training status
"""
import asyncio
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/train", tags=["training"])


# ── Pydantic schemas ────────────────────────────────────────────────

class TrainConfig(BaseModel):
    """Configuration for a federated training run."""
    num_clients: int = Field(3, ge=1, le=100, description="Number of federated clients")
    global_rounds: int = Field(5, ge=1, le=1000, description="Number of global rounds")
    local_epochs: int = Field(5, ge=1, le=100, description="Local epochs per client per round")
    lr: float = Field(0.01, gt=0, description="Client learning rate")
    batch_size: int = Field(32, ge=1, description="Client mini-batch size")
    algorithm: str = Field("fedavg", description="Aggregation algorithm: fedavg | fedprox | scaffold")
    mu: float = Field(0.01, ge=0, description="Proximal coefficient (FedProx only)")
    dp_epsilon: float = Field(16.0, gt=0, description="DP epsilon (0 = no DP)")
    dp_delta: float = Field(1e-5, gt=0, description="DP delta")
    server_lr: float = Field(1.0, gt=0, description="Server learning rate (SCAFFOLD only)")
    seed: int = Field(42, description="Random seed")


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainStatus(BaseModel):
    """Response model for training status."""
    job_id: str
    state: JobState
    current_round: int = 0
    total_rounds: int = 0
    best_accuracy: float = 0.0
    elapsed_seconds: float = 0.0
    message: str = ""


# ── In-memory job store (single-job model for v0.2) ─────────────────

_jobs: Dict[str, dict] = {}
_current_job_id: Optional[str] = None


# ── Tiny model & helpers (same as smoke test) ───────────────────────

INPUT_DIM = 8
OUTPUT_DIM = 2


class TCMNet(nn.Module):
    """Small feed-forward net for TCM diagnostic demo."""
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


def _make_synthetic_data(client_id: int, n: int = 200, seed: int = 42):
    """Generate synthetic client data with distribution shift."""
    from torch.utils.data import TensorDataset
    torch.manual_seed(seed + client_id * 1000 + 7)
    W = torch.randn(INPUT_DIM)
    X = torch.randn(n, INPUT_DIM) + client_id * 0.3
    y = (X @ W > 0).long()
    return TensorDataset(X, y)


def _make_test_data(n: int = 300, seed: int = 42):
    from torch.utils.data import TensorDataset
    torch.manual_seed(seed + 999)
    W = torch.randn(INPUT_DIM)
    X = torch.randn(n, INPUT_DIM)
    y = (X @ W > 0).long()
    return TensorDataset(X, y)


# ── Background training loop ────────────────────────────────────────

def _run_training(job_id: str, cfg: TrainConfig):
    """Execute the federated training loop in a background thread."""
    from torch.utils.data import DataLoader

    job = _jobs[job_id]
    job["state"] = JobState.RUNNING
    t0 = time.time()

    try:
        # Lazy imports to keep startup fast
        from backend.models.fedavg import FedAvg
        from backend.models.fedprox import FedProx
        from backend.models.scaffold import SCAFFOLD as Scaffold
        from backend.models.dp_mechanism import DPMechanism

        torch.manual_seed(cfg.seed)
        global_model = TCMNet()

        # Choose algorithm
        if cfg.algorithm == "fedprox":
            aggregator = FedProx(global_model, mu=cfg.mu)
        elif cfg.algorithm == "scaffold":
            aggregator = Scaffold(global_model, server_lr=cfg.server_lr)
            client_controls = [aggregator.init_client_control() for _ in range(cfg.num_clients)]
        else:
            aggregator = FedAvg(global_model)

        dp = DPMechanism(epsilon=cfg.dp_epsilon, delta=cfg.dp_delta) if cfg.dp_epsilon > 0 else None
        client_data = [_make_synthetic_data(i, seed=cfg.seed) for i in range(cfg.num_clients)]
        test_set = _make_test_data(seed=cfg.seed)
        best_acc = 0.0

        for rnd in range(1, cfg.global_rounds + 1):
            client_states = []
            global_sd = {k: v.clone() for k, v in global_model.state_dict().items()}

            for cid in range(cfg.num_clients):
                local = TCMNet()
                local.load_state_dict(global_sd)
                loader = DataLoader(client_data[cid], batch_size=cfg.batch_size, shuffle=True)

                if cfg.algorithm == "fedprox":
                    state = FedProx.local_train(
                        local, global_sd, loader,
                        epochs=cfg.local_epochs, lr=cfg.lr, mu=cfg.mu,
                    )
                elif cfg.algorithm == "scaffold":
                    state, client_controls[cid] = Scaffold.local_train(
                        local, global_sd,
                        aggregator.get_server_control(), client_controls[cid],
                        loader, epochs=cfg.local_epochs, lr=cfg.lr,
                    )
                else:
                    # FedAvg local training
                    criterion = nn.CrossEntropyLoss()
                    opt = torch.optim.SGD(local.parameters(), lr=cfg.lr)
                    local.train()
                    for _ in range(cfg.local_epochs):
                        for X_b, y_b in loader:
                            opt.zero_grad()
                            criterion(local(X_b), y_b).backward()
                            opt.step()
                    state = {k: v.clone() for k, v in local.state_dict().items()}

                # DP noise
                if dp:
                    state = dp.add_noise(state, sensitivity=0.1)
                client_states.append(state)

            # Aggregate
            if cfg.algorithm == "scaffold":
                global_model = aggregator.aggregate(client_states, client_controls, cfg.num_clients)
            else:
                global_model = aggregator.aggregate(client_states)

            # Evaluate
            global_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_b, y_b in DataLoader(test_set, batch_size=256):
                    correct += (global_model(X_b).argmax(1) == y_b).sum().item()
                    total += y_b.size(0)
            test_acc = correct / total
            best_acc = max(best_acc, test_acc)

            # Update job status
            job["current_round"] = rnd
            job["best_accuracy"] = round(best_acc, 4)
            job["elapsed_seconds"] = round(time.time() - t0, 2)
            job["message"] = f"Round {rnd}/{cfg.global_rounds} — test_acc={test_acc:.1%}"

        job["state"] = JobState.COMPLETED
        job["message"] = f"Done. Best accuracy: {best_acc:.1%}"

    except Exception as exc:
        job["state"] = JobState.FAILED
        job["message"] = f"Error: {exc}"


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/start", response_model=TrainStatus)
async def start_training(config: TrainConfig, bg: BackgroundTasks):
    """Start a federated training job (runs in the background)."""
    global _current_job_id

    # Only one running job at a time
    if _current_job_id and _jobs.get(_current_job_id, {}).get("state") == JobState.RUNNING:
        raise HTTPException(409, "A training job is already running. Wait for it to finish or check status.")

    job_id = uuid.uuid4().hex[:12]
    _current_job_id = job_id
    _jobs[job_id] = {
        "job_id": job_id,
        "state": JobState.PENDING,
        "current_round": 0,
        "total_rounds": config.global_rounds,
        "best_accuracy": 0.0,
        "elapsed_seconds": 0.0,
        "message": "Queued",
    }

    bg.add_task(_run_training, job_id, config)

    return TrainStatus(**_jobs[job_id])


@router.get("/status", response_model=TrainStatus)
async def get_status(job_id: Optional[str] = None):
    """Get training status. Defaults to the most recent job."""
    target = job_id or _current_job_id
    if not target or target not in _jobs:
        raise HTTPException(404, "No training job found.")
    return TrainStatus(**_jobs[target])
