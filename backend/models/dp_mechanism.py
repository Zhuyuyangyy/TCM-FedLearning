"""Differential privacy."""
import torch

class DPMechanism:
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon; self.delta = delta
    def add_noise(self, gradients, sensitivity=1.0):
        sigma = sensitivity * (2*torch.log(torch.tensor(1.25/self.delta))).sqrt() / self.epsilon
        return {k: v + torch.randn_like(v)*sigma for k, v in gradients.items()}
