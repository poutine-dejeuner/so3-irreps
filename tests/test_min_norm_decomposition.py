"""Tests for the minimum-norm rank-one decomposition of the matmul tensor."""

from __future__ import annotations

from scripts.min_norm_decomposition import matmul_tensor_torch, min_norm_decomposition


def test_matmul_tensor_shape_and_entries():
    M = matmul_tensor_torch(2)
    assert M.shape == (4, 4, 4) and int(M.sum()) == 8


def test_d3_selects_27_equal_units():
    out = min_norm_decomposition(d=3, n_units=60, lam=3e-3, iters=8000, seed=0, device="cpu")
    assert out["units_alive"] == 27
    assert out["rel_err"] < 1e-3
    assert out["contribution_spread"] < 0.01
