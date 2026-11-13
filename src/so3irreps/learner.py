"""so3_comp_learner.py — LightningModule for SO(3) group composition.

Trains a network to predict GH from G, H inputs. The 9D output is SVD-projected
to SO(3). Per PROJECT.md:

    - AdamW optimizer
    - Exponential LR decay (×0.85 every 120 epochs)
    - Batch size 128
    - 1001 epochs
    - Grid: lr ∈ {3e-5, 1e-4, 3e-4}, λ ∈ {0, 1e-4, 1e-3}
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from lightning import LightningModule

from so3irreps.model import SO3CompositionNet
from so3irreps.data.so3_comp import SO3CompositionDataModule, _svd_project_to_so3

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class SO3CompositionLearner(LightningModule):
    """Train an SO3CompositionNet to predict GH from G, H.

    Args:
        datamodule:   The SO3CompositionDataModule.
        hidden_size:  Hidden layer width (default: 2048).
        num_layers:   Number of hidden layers (default: 1).
        lr:           Learning rate for AdamW.
        weight_decay: Weight decay (AdamW decoupled).
        lr_gamma:     Exponential LR decay factor (default: 0.85).
        lr_step:      Number of epochs per decay step (default: 120).
        activation:   Hidden activation, "silu" or "square".
        layernorm:    Whether to apply LayerNorm before each activation.
        optimizer:    "adamw" (default), "sgd" (plain, no momentum) or "pnsgd" (per-neuron
                      scaled SGD, η_i ∝ ‖θ_i‖^{-1}; single hidden layer; no weight decay).
        init_scale:   Multiplier on the default Linear init (small init = AGF regime).
        train_metrics: Log geodesic/per-block errors during training too (slow: ~3x).
    """

    def __init__(
        self,
        datamodule: SO3CompositionDataModule,
        hidden_size: int = 2048,
        num_layers: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lr_gamma: float = 0.85,
        lr_step: int = 120,
        activation: str = "silu",
        layernorm: bool = False,
        optimizer: str = "adamw",
        init_scale: float = 1.0,
        train_metrics: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr_gamma = lr_gamma
        self.lr_step = lr_step
        self.optimizer = optimizer
        self.train_metrics = train_metrics
        self.save_hyperparameters(ignore=["datamodule"])

        self.network = SO3CompositionNet(
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            layernorm=layernorm,
            init_scale=init_scale,
            bias=bias,
        )

    def forward(self, G_vec: torch.Tensor, H_vec: torch.Tensor) -> torch.Tensor:
        """Raw output in the configured target representation."""
        return self.network(G_vec, H_vec)

    def predict_so3(self, G_vec: torch.Tensor, H_vec: torch.Tensor) -> torch.Tensor:
        """Predict and project to SO(3).

        Returns (batch, 3, 3) SO(3) matrices.
        """
        if self.datamodule.target_type != "matrix":
            raise RuntimeError("predict_so3 is only available for matrix-target models")
        raw = self.forward(G_vec, H_vec)  # (batch, 9)
        return _svd_project_to_so3(raw)

    # ------------------------------------------------------------------ #
    # Lightning steps
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Batch, batch_index: int) -> dict:
        return self.shared_step(batch, batch_index, "train")

    def validation_step(self, batch: Batch, batch_index: int) -> dict:
        return self.shared_step(batch, batch_index, "val")

    def test_step(self, batch: Batch, batch_index: int) -> dict:
        return self.shared_step(batch, batch_index, "test")

    def shared_step(
        self,
        batch: Batch,
        batch_index: int,
        phase: Literal["train", "val", "test"],
    ) -> dict:
        G_vec, H_vec, GH_mat, target = batch
        raw_out = self.forward(G_vec, H_vec)

        target_mse = F.mse_loss(raw_out, target)
        if self.datamodule.target_type == "orbit":
            # Orbit targets have unit squared norm, while rotation matrices
            # have squared Frobenius norm 3. Match their per-sample loss scale.
            loss = target_mse * (target.shape[-1] / 3)
        else:
            loss = target_mse * getattr(self.datamodule, "loss_scale", 1.0)

        self.log(f"{phase}/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        if phase == "train" and not self.train_metrics:
            return {"loss": loss}
        self.log(
            f"{phase}/target_mse",
            target_mse,
            on_epoch=True,
            on_step=False,
        )
        if self.datamodule.target_type == "onehot":
            # Per-irrep relative error; geodesic error read off the D^1 block,
            # which is conjugate to the 3×3 matrix by a fixed orthogonal map.
            # (position-basis encodings are first mapped back to δ^L coordinates)
            raw_out, target = self.datamodule.to_fourier(raw_out), self.datamodule.to_fourier(target)
            for l, sl in self.datamodule.blocks.items():
                rel = ((raw_out[:, sl] - target[:, sl]) ** 2).sum() / (target[:, sl] ** 2).sum()
                self.log(f"{phase}/rel_mse_l{l}", rel, on_epoch=True, on_step=False)
            d1 = self.datamodule.blocks[1]
            pred_so3 = _svd_project_to_so3(raw_out[:, d1] / 3**0.5)
            true_so3 = (target[:, d1] / 3**0.5).reshape(-1, 3, 3)
            geodesic_error = _so3_geodesic_distance(pred_so3, true_so3)
            self.log(
                f"{phase}/geodesic_error",
                geodesic_error.mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
            )
        elif self.datamodule.target_type == "matrix":
            pred_so3 = _svd_project_to_so3(raw_out)
            geodesic_error = _so3_geodesic_distance(pred_so3, GH_mat)
            self.log(
                f"{phase}/geodesic_error",
                geodesic_error.mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
            )
        else:
            orbit_error = _vector_angular_distance(raw_out, target)
            self.log(
                f"{phase}/orbit_error",
                orbit_error.mean(),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
            )

        return {"loss": loss}

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        if self.optimizer == "pnsgd":
            # per-neuron scaled SGD (vanishing-init regime); single hidden layer, z²
            from so3irreps.optimizers import PerNeuronScaledSGD

            lin1, lin2 = self.network.net[0], self.network.net[-1]
            optimizer = PerNeuronScaledSGD(lin1, lin2, lr=self.lr, degree=2)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step,
            gamma=self.lr_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _so3_geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Geodesic (angular) distance between two batches of SO(3) matrices.

    d(R1, R2) = arccos((trace(R1^T @ R2) - 1) / 2).

    Args:
        R1: (batch, 3, 3)
        R2: (batch, 3, 3).

    Returns:
        (batch,) tensor of distances in radians.
    """
    R_diff = torch.bmm(R1.transpose(1, 2), R2)
    trace = R_diff.diagonal(dim1=1, dim2=2).sum(-1)  # (batch,)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-9, 1.0 - 1e-9)
    return torch.acos(cos_angle)


def _vector_angular_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Angular distance between corresponding nonzero vectors."""
    cosine = F.cosine_similarity(x, y, dim=-1).clamp(-1.0 + 1e-9, 1.0 - 1e-9)
    return torch.acos(cosine)
