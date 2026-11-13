"""Tests for the position-basis (inverse-Fourier, sampled truncated delta) SO(3) encoding."""

from __future__ import annotations

import numpy as np
import torch

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from so3irreps.wigner import so3_random
from so3irreps.data.so3_onehot import (
    SO3OneHotDataModule,
    anchor_matrix,
    bandlimited_onehot,
    position_encoding,
)


def _rotation_angle(R: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def _kernel_closed_form(theta: np.ndarray, L: int) -> np.ndarray:
    # K_L(θ) = Σ_{l=0}^{L} (2l+1) sin((2l+1)θ/2) / sin(θ/2)
    s = np.sin(theta / 2)
    return sum((2 * l + 1) * np.sin((2 * l + 1) * theta / 2) / s for l in range(L + 1))


def test_onehot_with_l0_prepends_constant_block():
    R, *_ = so3_random(10, seed=0)
    with0 = bandlimited_onehot(R, 2, include_l0=True)
    assert np.allclose(with0[:, 0], 1.0)
    assert np.allclose(with0[:, 1:], bandlimited_onehot(R, 2))


def test_position_encoding_samples_the_truncated_delta():
    L, N = 3, 50
    phi = anchor_matrix(L, N, seed=1)
    anchors, *_ = so3_random(N, seed=1)
    G, *_ = so3_random(5, seed=2)
    x = position_encoding(G, L, phi)
    assert x.shape == (5, N)
    theta = _rotation_angle(np.einsum("nji,mjk->nmik", G, anchors))  # angle of Gᵀ R_i
    assert np.allclose(x * np.sqrt(N), _kernel_closed_form(theta, L), atol=1e-6)


def test_riemann_gram_converges_to_kernel():
    L = 2
    G, *_ = so3_random(20, seed=3)
    H, *_ = so3_random(20, seed=4)
    exact = (bandlimited_onehot(G, L, True) * bandlimited_onehot(H, L, True)).sum(1)
    errs = []
    for N in (256, 4096):
        phi = anchor_matrix(L, N, seed=5)
        gram = (position_encoding(G, L, phi) * position_encoding(H, L, phi)).sum(1)
        errs.append(np.abs(gram - exact).mean())
    assert errs[1] < errs[0]


def test_position_datamodule_roundtrip_to_fourier():
    dm = SO3OneHotDataModule(L=2, n=40, batch_size=8, seed=0, basis="position", n_anchors=64)
    G_enc, H_enc, GH, target = dm.build_full_dataset().tensors
    assert dm.input_dim == 128 and dm.output_dim == 64
    back = dm.to_fourier(target)
    assert torch.allclose(back, torch.from_numpy(bandlimited_onehot(GH.numpy(), 2)).float(),
                          atol=1e-4)
    # fourier basis: to_fourier is the identity
    dmf = SO3OneHotDataModule(L=2, n=40, batch_size=8, seed=0)
    t = dmf.build_full_dataset().tensors[3]
    assert torch.equal(dmf.to_fourier(t), t)


def test_learner_metrics_in_position_basis_are_zero_for_perfect_prediction():
    from so3irreps.learner import SO3CompositionLearner

    dm = SO3OneHotDataModule(L=2, n=40, batch_size=8, seed=0, basis="position", n_anchors=64)
    learner = SO3CompositionLearner(datamodule=dm, hidden_size=8, num_layers=0)
    G_enc, H_enc, GH, target = dm.build_full_dataset().tensors
    learner.forward = lambda g, h: target
    logged = {}
    learner.log = lambda name, value, **kw: logged.__setitem__(name, value)
    learner.shared_step((G_enc, H_enc, GH, target), 0, "val")
    assert logged["val/geodesic_error"] < 1e-2
    assert logged["val/rel_mse_l2"] < 1e-8


def test_position_dataset_encodes_lazily_and_matches_eager_encoding():
    dm = SO3OneHotDataModule(L=2, n=40, batch_size=8, seed=0, basis="position", n_anchors=64)
    ds = dm.build_full_dataset()
    # stored compactly (Fourier coordinates incl. l=0), encoded on access
    assert ds.base[0].shape[1] == 1 + 9 + 25
    G_enc, H_enc, GH, target = ds[list(range(5))]
    assert G_enc.shape == (5, 64)
    eager = torch.from_numpy(dm.encode(GH[:5].numpy())).float()
    assert torch.allclose(target, eager, atol=1e-5)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch[0].shape == (8, 64) and batch[3].shape == (8, 64)


def test_position_loss_matches_fourier_loss_for_same_fourier_error():
    # MSE averages over output coordinates (N vs N_L); the learner rescales so that the
    # same error in δ^L coordinates gives (approximately, Riemann sum) the same loss.
    from so3irreps.learner import SO3CompositionLearner

    torch.manual_seed(0)
    losses = {}
    for basis in ("fourier", "position"):
        dm = SO3OneHotDataModule(L=2, n=200, batch_size=8, seed=0, basis=basis, n_anchors=4096)
        learner = SO3CompositionLearner(datamodule=dm, hidden_size=8, num_layers=0)
        G_enc, H_enc, GH, target = dm.build_full_dataset().tensors
        torch.manual_seed(1)
        eps = 0.1 * torch.randn(len(GH), 35)  # error in δ̃ coordinates (incl. l=0)
        eps[:, 0] = 0
        noisy = eps[:, 1:] if basis == "fourier" else eps @ torch.from_numpy(dm.phi).float().T
        learner.forward = lambda g, h, t=target, n=noisy: t + n
        learner.log = lambda *a, **k: None
        losses[basis] = float(learner.shared_step((G_enc, H_enc, GH, target), 0, "train")["loss"])
    assert abs(losses["position"] / losses["fourier"] - 1) < 0.1
