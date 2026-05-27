"""SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

Reference: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
           (Karimireddy et al., 2020).

SCAFFOLD reduces client drift by maintaining *control variates* on both server
and clients.  The server control c tracks the direction of the global update;
each client maintains a local control c_i that captures its own gradient bias.
The correction term applied during local training is  (c - c_i), steering the
local update toward the global objective.

This implementation follows the **SCAFFOLD-S** (server-side) variant where
control variates are updated once per round.
"""
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class SCAFFOLD:
    """SCAFFOLD aggregation with control variates.

    Parameters
    ----------
    global_model : nn.Module
        The shared global model.
    server_lr : float
        Server-side learning rate for the global update step.
        (Sometimes called eta_s or eta_g.)
    """

    def __init__(self, global_model: nn.Module, server_lr: float = 1.0):
        self.global_model = global_model
        self.server_lr = server_lr

        # Server control variate  c  (same shape as global state)
        self.server_control: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v) for k, v in global_model.state_dict().items()
        }

    # ── initialise / reset client controls ──

    def init_client_control(self) -> Dict[str, torch.Tensor]:
        """Return a zeroed-out control variate dict for a new client."""
        return {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}

    # ── local training with SCAFFOLD correction ──

    @staticmethod
    def local_train(
        local_model: nn.Module,
        global_state: Dict[str, torch.Tensor],
        server_control: Dict[str, torch.Tensor],
        client_control: Dict[str, torch.Tensor],
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run local SCAFFOLD training on one client.

        The SGD update becomes::

            w  <-  w  -  lr * (g  +  c  -  c_i)

        where g is the mini-batch gradient, c is the server control,
        and c_i is the client control.

        Parameters
        ----------
        local_model : nn.Module
            Pre-loaded with global weights.
        global_state : dict
            Global state_dict (for computing new client control).
        server_control : dict
            Server control variate c.
        client_control : dict
            This client's current control variate c_i.
        data_loader : DataLoader
            Client-local data.
        epochs : int
            Local epochs.
        lr : float
            Learning rate.
        device : str
            Torch device.

        Returns
        -------
        (new_state, new_client_control) : tuple of dicts
        """
        local_model.to(device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()

        # Capture initial weights (y_i in the paper)
        y_i: Dict[str, torch.Tensor] = {
            k: v.clone().detach().to(device) for k, v in global_state.items()
        }

        # Correction term:  c - c_i  (pre-compute, stays constant during local steps)
        correction: Dict[str, torch.Tensor] = {}
        for k in server_control:
            correction[k] = (server_control[k] - client_control[k]).to(device)

        for _ in range(epochs):
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                local_model.zero_grad()
                output = local_model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()

                # Apply SCAFFOLD-corrected gradient:  g + (c - c_i)
                with torch.no_grad():
                    for name, param in local_model.named_parameters():
                        if param.grad is not None and name in correction:
                            param.grad.add_(correction[name])
                    # Standard SGD step
                    for param in local_model.parameters():
                        if param.grad is not None:
                            param.add_(param.grad, alpha=-lr)

        # New local state
        new_state: Dict[str, torch.Tensor] = {
            k: v.clone().cpu() for k, v in local_model.state_dict().items()
        }

        # Update client control:  c_i^+  =  c_i  -  c  +  (1 / (lr * K)) * (y_i - w_i^+)
        # where K = number of local steps ≈ epochs * batches_per_epoch
        n_batches = max(len(data_loader) * epochs, 1)
        new_client_control: Dict[str, torch.Tensor] = {}
        for k in client_control:
            grad_estimate = (y_i[k] - local_model.state_dict()[k].to(device)) / (lr * n_batches)
            new_client_control[k] = (
                client_control[k].to(device) - server_control[k].to(device) + grad_estimate
            ).cpu()

        return new_state, new_client_control

    # ── aggregation ──

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        client_controls: List[Dict[str, torch.Tensor]],
        num_clients: int,
    ) -> nn.Module:
        """Aggregate client updates and update server control variate.

        The global model update is::

            w  <-  w  +  (eta_s / N) * sum_i  (y_i - w)
                  =  w  +  (eta_s / N) * sum_i  delta_i

        Then the server control::

            c  <-  c  +  (1/N) * sum_i  (c_i^+  -  c_i_old)

        This is *not* averaged into the model; it steers future corrections.

        Parameters
        ----------
        client_states : list of dicts
            Updated state dicts from each client.
        client_controls : list of dicts
            Updated client controls from each client.
        num_clients : int
            Number of participating clients N.

        Returns
        -------
        nn.Module
            The updated global model.
        """
        N = num_clients
        global_state = {k: v.clone().float() for k, v in self.global_model.state_dict().items()}

        # ── Global model update ──
        # delta_i = y_i - w  (client update *direction*);  but we receive the
        # full state, so delta_i = client_state[i] - old_global.
        # We only want the average *update direction*, applied with server_lr.
        new_global: Dict[str, torch.Tensor] = {}
        for key in global_state:
            stacked = torch.stack([s[key].float() for s in client_states])
            avg_client = stacked.mean(dim=0)
            # scaled update:  w + server_lr * (avg_client - w)
            new_global[key] = global_state[key] + self.server_lr * (avg_client - global_state[key])

        self.global_model.load_state_dict(new_global)

        # ── Server control update  (simplified SCAFFOLD-S) ──
        # c  <-  c  +  (1/N) * sum_i c_i^+
        # (We don't have old client controls here in this interface, so we
        #  set the server control to the average of the *new* client controls,
        #  which is the equilibrium fixed-point of the update.)
        for key in self.server_control:
            stacked_c = torch.stack([c[key].float() for c in client_controls])
            self.server_control[key] = stacked_c.mean(dim=0)

        return self.global_model

    def get_server_control(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the current server control variate."""
        return {k: v.clone() for k, v in self.server_control.items()}


# ── convenience alias ──
def create_scaffold(global_model: nn.Module, server_lr: float = 1.0) -> SCAFFOLD:
    return SCAFFOLD(global_model, server_lr=server_lr)
