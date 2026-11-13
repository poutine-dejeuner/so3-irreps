"""Tests for the band-limited one-hot encoding of SO(3) and its datamodule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from so3irreps.wigner import so3_random, wigner_D_real
from so3irreps.data.so3_onehot import (
    SO3OneHotDataModule,
    bandlimited_onehot,
    block_slices,
)


def test_block_slices_layout():
    sl = block_slices(3)
    assert [l for l in sl] == [1, 2, 3]
    assert [s.stop - s.start for s in sl.values()] == [9, 25, 49]
    assert sl[1].start == 0 and sl[3].stop == 83


def test_onehot_blocks_are_scaled_wigner_matrices():
    R, *_ = so3_random(50, seed=0)
    enc = bandlimited_onehot(R, L=3)
    assert enc.shape == (50, 83)
    for l, s in block_slices(3).items():
        expected = np.sqrt(2 * l + 1) * wigner_D_real(l, R).reshape(50, -1)
        assert np.allclose(enc[:, s], expected, atol=1e-10)
    # every entry has unit Haar variance, so ‖δ^L(G)‖² = Σ_l d_l²
    assert np.allclose((enc**2).sum(1), 9 + 25 + 49)


def test_onehot_kernel_peaks_at_identity():
    # <δ^L(G), δ^L(H)> = Σ_l d_l χ_l(GᵀH): maximal (= Σ d_l²) iff G = H
    G, *_ = so3_random(200, seed=1)
    H, *_ = so3_random(200, seed=2)
    k_same = (bandlimited_onehot(G, 3) ** 2).sum(1)
    k_diff = (bandlimited_onehot(G, 3) * bandlimited_onehot(H, 3)).sum(1)
    assert np.all(k_diff < k_same - 1e-6)


def test_datamodule_target_is_onehot_of_product():
    dm = SO3OneHotDataModule(L=3, n=64, batch_size=8, seed=0)
    G_enc, H_enc, GH, target = dm.build_full_dataset().tensors
    assert dm.input_dim == 2 * 83 and dm.output_dim == 83
    assert dm.target_type == "onehot"
    assert torch.allclose(target, torch.from_numpy(bandlimited_onehot(GH.numpy(), 3)).float(), atol=1e-5)
    # D^l(GH) = D^l(G) D^l(H), blockwise, with the √d_l weights
    for l, s in block_slices(3).items():
        d = 2 * l + 1
        DG = G_enc[:, s].reshape(-1, d, d) / np.sqrt(d)
        DH = H_enc[:, s].reshape(-1, d, d) / np.sqrt(d)
        assert torch.allclose(target[:, s].reshape(-1, d, d), np.sqrt(d) * DG @ DH, atol=1e-4)


def test_datamodule_rejects_bad_L():
    with pytest.raises(ValueError):
        SO3OneHotDataModule(L=0)


def test_train_dataloader_matches_default_batching():
    # Bulk indexing must give exactly the batches of DataLoader(shuffle=True).
    from torch.utils.data import DataLoader

    dm = SO3OneHotDataModule(L=1, n=100, batch_size=16, seed=0)
    dm.setup("fit")
    torch.manual_seed(3)
    ref = list(DataLoader(dm.train_dataset, batch_size=16, shuffle=True))
    torch.manual_seed(3)
    new = list(dm.train_dataloader())
    assert len(ref) == len(new)
    assert all(torch.equal(x, y) for a, b in zip(ref, new) for x, y in zip(a, b))
