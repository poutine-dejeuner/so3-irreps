"""so3_onehot.py — SO(3) composition with the band-limited one-hot encoding.

By Peter–Weyl, the one-hot (delta) of G is the stack (√d_l D^l(G))_{l≥0} of
all irrep matrices, d_l = 2l+1. We truncate at l ≤ L and drop the constant
l=0 block:

    δ^L(G) = ⊕_{l=1}^{L} √d_l vec D^l(G)        (row-major vec, real Wigner D)

Input (δ^L(G), δ^L(H)), target δ^L(GH); D^l(GH) = D^l(G) D^l(H). Every entry
has unit Haar variance, and a first-layer weight block W^(l) gives a
preactivation whose Haar variance is exactly ‖W^(l)‖²_F.

Position basis (basis="position"): G is instead encoded by the truncated delta as a
*function*, the inverse Fourier transform K_L(G, R) = Σ_{l=0}^{L} d_l χ_l(GᵀR)
(l=0 included), sampled at N Haar-random anchors R_i with Riemann-sum weights:
x_G[i] = K_L(G, R_i)/√N, so that ⟨x_G, x_H⟩ ≈ ∫ K_L(G,·)K_L(H,·) = K_L(G, H). Since
K_L(G, R) = ⟨δ̃^L(G), δ̃^L(R)⟩ (δ̃ includes the l=0 block), x_G = Φ δ̃^L(G) with
Φ[i] = δ̃^L(R_i)/√N; `to_fourier` maps back with the pseudo-inverse of Φ.
"""

from __future__ import annotations

import lightning
import numpy as np
import torch
from torch.utils.data import TensorDataset

from so3irreps.wigner import so3_random, wigner_D_real
from so3irreps.data.so3_comp import SO3CompositionDataModule


def block_slices(L: int) -> dict[int, slice]:
    """Index range of each l-block in δ^L, for l = 1..L."""
    slices, start = {}, 0
    for l in range(1, L + 1):
        size = (2 * l + 1) ** 2
        slices[l] = slice(start, start + size)
        start += size
    return slices


def bandlimited_onehot(R: np.ndarray, L: int, include_l0: bool = False) -> np.ndarray:
    """Band-limited one-hot δ^L(R) of a batch of rotations R (n, 3, 3) → (n, Σ_{l≤L} d_l²).

    With include_l0, a leading constant column (the l=0 block, D^0 = 1) is prepended.
    """
    n = len(R)
    blocks = [np.ones((n, 1))] if include_l0 else []
    blocks += [np.sqrt(2 * l + 1) * wigner_D_real(l, R).reshape(n, -1) for l in range(1, L + 1)]
    return np.concatenate(blocks, axis=1)


def anchor_matrix(L: int, n_anchors: int, seed: int = 0) -> np.ndarray:
    """Φ (n_anchors, 1 + Σ d_l²) with rows δ̃^L(R_i)/√N for Haar-random anchors R_i."""
    anchors, *_ = so3_random(n_anchors, seed=seed)
    return bandlimited_onehot(anchors, L, include_l0=True) / np.sqrt(n_anchors)


def position_encoding(R: np.ndarray, L: int, phi: np.ndarray) -> np.ndarray:
    """x_R[i] = K_L(R, R_i)/√N, the truncated delta at R sampled at the anchors of Φ."""
    return bandlimited_onehot(R, L, include_l0=True) @ phi.T


class EncodedDataset(torch.utils.data.Dataset):
    """Stores δ̃^L coordinates (compact) and returns position encodings x = δ̃ Φᵀ on access.

    Avoids materialising N-dimensional encodings for the whole dataset (2.5 GB for
    N = 4096); indexing with a list of indices encodes a whole batch in one matmul.
    """

    def __init__(self, dG, dH, GH, dGH, phi: torch.Tensor):
        self.base = (dG, dH, GH, dGH)
        self.phi_t = phi.T.contiguous()

    def __len__(self) -> int:
        return len(self.base[0])

    def __getitem__(self, idx):
        dG, dH, GH, dGH = (t[idx] for t in self.base)
        return dG @ self.phi_t, dH @ self.phi_t, GH, dGH @ self.phi_t

    @property
    def tensors(self):
        return self[list(range(len(self)))]


class SO3OneHotDataModule(SO3CompositionDataModule):
    """Datamodule for SO(3) composition with band-limited one-hot input and target.

    Samples are (δ^L(G), δ^L(H), GH, δ^L(GH)) with G, H Haar-distributed. Reuses the split and
    dataloaders of SO3CompositionDataModule.
    """

    target_type = "onehot"

    def __init__(
        self,
        L: int = 3,
        n: int = 50_000,
        batch_size: int = 128,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
        basis: str = "fourier",
        n_anchors: int = 1024,
        anchor_seed: int = 0,
        device: str = "cpu",
    ):
        lightning.LightningDataModule.__init__(self)
        self.save_hyperparameters()
        if L < 1:
            raise ValueError(f"L must be ≥ 1; got {L}")
        if basis not in ("fourier", "position"):
            raise ValueError(f"basis must be 'fourier' or 'position'; got {basis}")
        self.L = L
        self.n = n
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_workers = num_workers
        self.seed = seed
        self.blocks = block_slices(L)
        self.basis = basis
        self.device = device  # where position-basis batches are encoded
        n_fourier = sum((2 * l + 1) ** 2 for l in range(1, L + 1))
        if basis == "position":
            if n_anchors < n_fourier + 1:
                raise ValueError(f"need n_anchors ≥ {n_fourier + 1} for an invertible Φ")
            self.phi = anchor_matrix(L, n_anchors, anchor_seed)
            self._pinv = torch.from_numpy(np.linalg.pinv(self.phi)).float()
            self.single_input_dim = n_anchors
        else:
            self.single_input_dim = n_fourier
        # MSE averages over output coordinates: rescale so that the loss is the squared
        # error per δ^L coordinate in both bases (else position-basis gradients are
        # N/N_L times smaller).
        self.loss_scale = self.single_input_dim / n_fourier
        self.input_dim = 2 * self.single_input_dim
        self.output_dim = self.single_input_dim
        self._full = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def encode(self, R: np.ndarray) -> np.ndarray:
        """Encoding of a batch of rotations in this datamodule's basis."""
        if self.basis == "position":
            return position_encoding(R, self.L, self.phi)
        return bandlimited_onehot(R, self.L)

    def to_fourier(self, y: torch.Tensor) -> torch.Tensor:
        """Map encodings/outputs to δ^L coordinates (l ≥ 1 blocks); identity for fourier."""
        if self.basis == "fourier":
            return y
        return (y @ self._pinv.to(y.device).T)[:, 1:]

    def build_full_dataset(self) -> TensorDataset:
        G, *_ = so3_random(self.n, seed=self.seed)
        H, *_ = so3_random(self.n, seed=self.seed + 10_000)
        GH = G @ H

        def t(a):
            return torch.from_numpy(a).float()

        if self.basis == "position":
            dev = self.device
            phi = t(self.phi).to(dev)
            return EncodedDataset(*(t(bandlimited_onehot(R, self.L, include_l0=True)).to(dev)
                                    for R in (G, H)), t(GH).to(dev),
                                  t(bandlimited_onehot(GH, self.L, include_l0=True)).to(dev), phi)
        return TensorDataset(t(self.encode(G)), t(self.encode(H)), t(GH), t(self.encode(GH)))
