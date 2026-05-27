"""Federated Averaging."""
import torch

class FedAvg:
    def __init__(self, global_model): self.global_model = global_model
    def aggregate(self, client_states):
        avg = {}
        for key in client_states[0]:
            avg[key] = torch.stack([s[key].float() for s in client_states]).mean(dim=0)
        self.global_model.load_state_dict(avg)
        return self.global_model
