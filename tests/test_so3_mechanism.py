"""Tests for the SO(3) composition mechanism analysis."""

from __future__ import annotations

import numpy as np
import torch

from so3irreps.analysis.mechanism import (
    bilinear_signal,
    composability,
    fit_l1_maps,
    irrep_shares,
    isotypic_projectors,
    readout_alignment,
)
from so3irreps.wigner import so3_random
from so3irreps.data.so3_comp import _rotate_tensor_power


def _rho(R: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    Rt = torch.from_numpy(R).float()
    yt = torch.from_numpy(y).float().expand(len(R), -1)
    return _rotate_tensor_power(Rt, yt, k).double().numpy()


def test_isotypic_projectors_ranks_and_completeness():
    # (R^3)^{⊗2} = 0 ⊕ 1 ⊕ 2 ; (R^3)^{⊗3} = 0 ⊕ 3·1 ⊕ 2·2 ⊕ 3
    expected = {1: {1: 3}, 2: {0: 1, 1: 3, 2: 5}, 3: {0: 1, 1: 9, 2: 10, 3: 7}}
    for k, ranks in expected.items():
        P = isotypic_projectors(k)
        assert {l: int(round(np.trace(p))) for l, p in P.items()} == ranks
        assert np.allclose(sum(P.values()), np.eye(3**k), atol=1e-8)
        for p in P.values():
            assert np.allclose(p @ p, p, atol=1e-8)


def test_isotypic_projectors_commute_with_rotation():
    R, *_ = so3_random(5, seed=0)
    rng = np.random.default_rng(0)
    for k in (2, 3):
        P = isotypic_projectors(k)
        y = rng.standard_normal(3**k)
        for p in P.values():
            assert np.allclose(_rho(R, p @ y, k), _rho(R, y, k) @ p.T, atol=1e-5)


def test_irrep_shares_of_pure_l1_input_is_all_l1():
    R, *_ = so3_random(4000, seed=1)
    P = isotypic_projectors(3)
    rng = np.random.default_rng(1)
    x = P[1] @ rng.standard_normal(27)
    W = rng.standard_normal((32, 27))
    shares = irrep_shares(W, x, 3, R)
    assert shares[1] > 0.999 and shares[2] < 1e-3 and shares[3] < 1e-3


def test_bilinear_signal_matches_haar_expectation():
    # E[ GH · <A,G> · <B,H> ] = A @ B / 9 under Haar
    rng = np.random.default_rng(2)
    A, B = rng.standard_normal((1, 3, 3)), rng.standard_normal((1, 3, 3))
    G, *_ = so3_random(400_000, seed=3)
    H, *_ = so3_random(400_000, seed=4)
    zG = np.einsum("nik,ik->n", G, A[0])
    zH = np.einsum("nik,ik->n", H, B[0])
    mc = np.einsum("nij,n->ij", G @ H, zG * zH) / len(G)
    assert np.allclose(mc, bilinear_signal(A, B)[0] / 9, atol=0.02)


def test_fit_l1_maps_recovers_linear_maps():
    rng = np.random.default_rng(5)
    A, B = rng.standard_normal((8, 3, 3)), rng.standard_normal((8, 3, 3))
    G, *_ = so3_random(500, seed=6)
    H, *_ = so3_random(500, seed=7)
    z = np.einsum("nik,jik->nj", G, A) + np.einsum("nik,jik->nj", H, B) + 0.3
    A_fit, B_fit, r2 = fit_l1_maps(z, G, H)
    assert np.allclose(A_fit, A, atol=1e-6) and np.allclose(B_fit, B, atol=1e-6)
    assert np.all(r2 > 0.999)


def test_composability_and_alignment():
    A = np.eye(3)[None].repeat(2, 0)
    B = np.stack([np.eye(3), np.diag([1.0, -1.0, 0.0])])
    c = composability(A, B)
    assert np.isclose(c[0], 1 / np.sqrt(3))
    W_out = np.stack([np.eye(3).ravel(), -np.diag([1.0, -1.0, 0.0]).ravel()], axis=1)
    assert np.allclose(readout_alignment(W_out, bilinear_signal(A, B)), [1.0, -1.0])
