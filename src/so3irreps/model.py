"""so3_comp_net.py — Network for SO(3) group composition.

Architecture per PROJECT.md:
    - Input: concatenation of G and H encodings in the tensor-power rep V
    - Hidden layers: LayerNorm → SiLU (SiLU, not ReLU per PROJECT.md)
    - Width: 2048
    - 1 hidden layer (d=1)
    - Output: configurable for either a 9D matrix or a tensor-orbit vector

The SVD projection is handled by the learner (so that tests can test the raw
network output separately).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Square(nn.Module):
    """Elementwise z² activation (the quadratic network of Marchetti, Kunin et al.)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * z


_ACTIVATIONS: dict[str, type[nn.Module]] = {"silu": nn.SiLU, "square": Square}


class SO3CompositionNet(nn.Module):
    """MLP for SO(3) group composition: (G, H) → GH.

    Args:
        input_dim:    Total input dimension (= 2 * 3^k for tensor power k).
        output_dim:   Output dimension (9 for GH, 3^k for an orbit target).
        hidden_size:  Width of hidden layers (default: 2048 per PROJECT.md).
        num_layers:   Number of extra hidden H→H layers (default: 1 per PROJECT.md).
                      num_layers=0 gives a single hidden layer.
        activation:   "silu" (default) or "square" (z²).
        layernorm:    Apply LayerNorm before each activation (default: False).
        init_scale:   Multiplier on the default init of all Linear layers.
        bias:         Use biases (False: homogeneous network, as in the AGF setting).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 9,
        hidden_size: int = 2048,
        num_layers: int = 1,
        activation: str = "silu",
        layernorm: bool = False,
        init_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}; got {activation}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.layernorm = layernorm

        def norm_act() -> list[nn.Module]:
            norm = [nn.LayerNorm(hidden_size)] if layernorm else []
            return norm + [_ACTIVATIONS[activation]()]

        layers: list[nn.Module] = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_size, bias=bias))

        # Hidden layers: (LayerNorm →) activation → Linear
        for _ in range(num_layers):
            layers.extend(norm_act())
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))

        # Output: (final LayerNorm →) activation → requested representation
        layers.extend(norm_act())
        layers.append(nn.Linear(hidden_size, output_dim, bias=bias))

        self.net = nn.Sequential(*layers)

        # Small initialisation (AGF regime): scale the default Linear init
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        p.mul_(init_scale)

    def first_preactivation(self, G_vec: torch.Tensor, H_vec: torch.Tensor) -> torch.Tensor:
        """Output of the input projection, before any LayerNorm or activation."""
        return self.net[0](torch.cat([G_vec, H_vec], dim=-1))

    def forward(self, G_vec: torch.Tensor, H_vec: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            G_vec: (batch, input_dim/2) — encoded G.
            H_vec: (batch, input_dim/2) — encoded H.

        Returns:
            (batch, output_dim) — raw output.
        """
        x = torch.cat([G_vec, H_vec], dim=-1)
        return self.net(x)
