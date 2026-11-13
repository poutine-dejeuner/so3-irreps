"""optimizers.py — Per-neuron scaled SGD for one-hidden-layer networks.

Time-rescaled gradient flow of Marchetti et al. (2026) / Kunin et al. (2025):
η_i = lr · ‖θ_i‖^{1−degree}, where θ_i = (row i of the first layer, its bias,
column i of the readout) gathers all parameters of hidden unit i and `degree` is the
activation degree (2 for z²). Near-zero (dormant) units get a large step, which makes
the vanishing-initialisation regime trainable in reasonable time. The output bias is
updated with plain SGD.
"""

from __future__ import annotations

import torch


class PerNeuronScaledSGD(torch.optim.Optimizer):
    def __init__(self, lin1: torch.nn.Linear, lin2: torch.nn.Linear, lr: float,
                 degree: int = 2, eps: float = 1e-12):
        params = [p for p in (lin1.weight, lin1.bias, lin2.weight, lin2.bias) if p is not None]
        super().__init__(params, dict(lr=lr))
        self.lin1, self.lin2, self.degree, self.eps = lin1, lin2, degree, eps

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        lr = self.param_groups[0]["lr"]
        W1, b1, W2, b2 = self.lin1.weight, self.lin1.bias, self.lin2.weight, self.lin2.bias
        sq = (W1**2).sum(1) + (W2**2).sum(0) + (b1**2 if b1 is not None else 0.0)
        scale = torch.sqrt(sq + self.eps) ** (1 - self.degree)
        W1.add_(W1.grad * scale[:, None], alpha=-lr)
        W2.add_(W2.grad * scale[None, :], alpha=-lr)
        if b1 is not None:  # bias-free (homogeneous) networks have no biases
            b1.add_(b1.grad * scale, alpha=-lr)
        if b2 is not None:
            b2.add_(b2.grad, alpha=-lr)
        return loss
