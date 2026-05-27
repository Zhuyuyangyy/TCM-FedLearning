"""FedProx: Proximal term for heterogeneous (non-IID) federated data.

Reference: "Federated Optimization in Heterogeneous Networks" (Li et al., 2020).

FedProx adds a proximal term  (mu/2)||w - w_global||^2  to the local loss,
which penalizes client drift from the global model. This stabilises convergence
under data heterogeneity where FedAvg diverges.
"""
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class FedProx:
    """FedProx aggregation with configurable proximal coefficient.

    Parameters
    ----------
    global_model : nn.Module
        The shared global model.
    mu : float
        Proximal term coefficient. mu=0 recovers FedAvg; larger mu = tighter
        coupling to the global model.  Typical range [0.001, 0.1].
    """

    def __init__(self, global_model: nn.Module, mu: float = 0.01):
        self.global_model = global_model
        self.mu = mu

    # ── aggregation (same as FedAvg, included for API completeness) ──

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]]) -> nn.Module:
        """Simple averaging of client state dicts (identical to FedAvg)."""
        avg: Dict[str, torch.Tensor] = {}
        for key in client_states[0]:
            avg[key] = torch.stack([s[key].float() for s in client_states]).mean(dim=0)
        self.global_model.load_state_dict(avg)
        return self.global_model

    # ── local training with proximal term ──

    @staticmethod
    def local_train(
        local_model: nn.Module,
        global_state: Dict[str, torch.Tensor],
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.01,
        mu: float = 0.01,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Train a client model with the FedProx proximal penalty.

        The extra loss term is::

            (mu / 2) * ||w_local - w_global||^2

        Parameters
        ----------
        local_model : nn.Module
            Model pre-loaded with the current global weights.
        global_state : dict
            The *frozen* global state_dict used for the proximal penalty.
        data_loader : DataLoader
            Client-local training data.
        epochs : int
            Number of local passes.
        lr : float
            Learning rate for SGD.
        mu : float
            Proximal coefficient.
        device : str
            Torch device.

        Returns
        -------
        dict
            Updated local state_dict.
        """
        local_model.to(device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        # Cache global parameters (detached, no grad)
        global_params: Dict[str, torch.Tensor] = {
            k: v.clone().detach().to(device) for k, v in global_state.items()
        }

        for _ in range(epochs):
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                output = local_model(X_batch)
                loss = criterion(output, y_batch)

                # ── Proximal term ──
                if mu > 0:
                    prox = torch.tensor(0.0, device=device)
                    for name, param in local_model.named_parameters():
                        if name in global_params:
                            diff = param - global_params[name]
                            prox = prox + diff.pow(2).sum()
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                optimizer.step()

        return {k: v.clone().cpu() for k, v in local_model.state_dict().items()}


# ── convenience alias ──
def create_fedprox(global_model: nn.Module, mu: float = 0.01) -> FedProx:
    return FedProx(global_model, mu=mu)
