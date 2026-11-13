"""Tests for the per-neuron irrep spectrum analysis (band-limited one-hot nets)."""

from __future__ import annotations

import numpy as np

from so3irreps.analysis.spectrum import (
    active_neurons,
    agreement,
    block_energies,
    block_tensor_alignment,
    matmul_tensor,
    neuron_report,
)
from so3irreps.data.so3_onehot import block_slices


def test_block_energies_and_spectra():
    sl = block_slices(2)  # l=1: 0..9, l=2: 9..34
    W = np.zeros((3, 34))
    W[0, 0:9] = 1.0          # pure l=1, energy 9
    W[1, 9:34] = 2.0         # pure l=2, energy 100
    W[2, :] = 1.0            # mixed: 9 and 25
    E = block_energies(W, sl)
    assert np.allclose(E, [[9, 0], [0, 100], [9, 25]])


def test_active_neurons_covers_contribution():
    c = np.array([10.0, 5.0, 0.01, 0.0])
    assert set(active_neurons(c, frac=0.95)) == {0, 1}


def test_agreement_confusion():
    a = np.array([1, 1, 2, 3])
    b = np.array([1, 2, 2, 3])
    M = agreement(a, b, L=3)
    assert M.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_matmul_tensor_reproduces_product():
    rng = np.random.default_rng(0)
    for d in (3, 5):
        A, B = rng.standard_normal((d, d)), rng.standard_normal((d, d))
        M = matmul_tensor(d)
        assert np.allclose(np.einsum("opq,p,q->o", M, A.ravel(), B.ravel()), (A @ B).ravel())


def test_block_tensor_alignment_of_exact_construction():
    # 27 product units (E_ik, E_kj, E_ij) implement 3×3 matmul exactly
    d = 3
    A, B, R = [], [], []
    for i in range(d):
        for j in range(d):
            for k in range(d):
                a, b, r = np.zeros((d, d)), np.zeros((d, d)), np.zeros((d, d))
                a[i, k], b[k, j], r[i, j] = 1, 1, 1
                A.append(a), B.append(b), R.append(r)
    out = block_tensor_alignment(np.array(A), np.array(B), np.array(R))
    assert np.isclose(out["tensor_cos_with_matmul"], 1.0)
    assert np.isclose(out["weighted_readout_cos"], 1.0)


def test_neuron_report_on_pure_irrep_network():
    # 2 neurons per irrep, each pure and consistent across G, H, readout
    L, sl = 2, block_slices(2)
    D = 34
    W_G, W_H, W_out = np.zeros((4, D)), np.zeros((4, D)), np.zeros((D, 4))
    for j, l in enumerate([1, 1, 2, 2]):
        W_G[j, sl[l]] = 1.0
        W_H[j, sl[l]] = 1.0
        W_out[sl[l], j] = 1.0
    rep = neuron_report(W_G, W_H, W_out, L)
    assert rep["n_active"] == 4
    assert rep["frac_purity_gt_0.9"] == {"G": 1.0, "H": 1.0, "R": 1.0}
    assert rep["frac_all_three_agree"] == 1.0
    assert rep["census"] == {"1": 2, "2": 2}


def test_product_unit_structure_of_rotated_textbook_construction():
    from so3irreps.analysis.spectrum import product_unit_structure

    rng = np.random.default_rng(1)
    d = 3
    U, V, W = (np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(3))
    A, B, R = [], [], []
    for i in range(d):
        for j in range(d):
            for k in range(d):
                A.append(np.outer(U[:, i], V[:, k]))
                B.append(np.outer(V[:, k], W[:, j]))
                R.append(np.outer(U[:, i], W[:, j]))
    s = product_unit_structure(np.array(A), np.array(B), np.array(R))
    assert s["median_rank1_ratio"]["A"] < 1e-8 and s["median_rank1_ratio"]["R"] < 1e-8
    assert s["median_shared_inner_cos"] > 0.999
    assert s["median_readout_outer_cos"] > 0.999
    assert np.isclose(s["norm_spread"], 0.0)
