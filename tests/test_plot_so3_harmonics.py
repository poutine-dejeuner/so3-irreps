"""Tests for decoding spherical harmonics from irrep-labelled neurons."""

from __future__ import annotations

import numpy as np

from so3irreps.wigner import real_sh
from so3irreps.data.so3_onehot import block_slices
from scripts.plot_so3_harmonics import decode_harmonics, sphere_grid


def test_sphere_grid_rotations_send_north_pole_to_grid_points():
    theta, phi, R = sphere_grid(5, 8)
    pts = R[:, :, 2]  # R ẑ
    expected = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                         np.cos(theta)], -1).reshape(-1, 3)
    assert np.allclose(pts, expected)


def test_decode_harmonics_from_neurons_spanning_the_block():
    # Neurons with random G-weights inside the l=2 block (L=3) span vec D^2:
    # the decoded zonal column must equal the real harmonics Y_2^m.
    L, l = 3, 2
    s = block_slices(L)[l]
    rng = np.random.default_rng(0)
    W = np.zeros((40, block_slices(L)[L].stop))
    W[:, s] = rng.standard_normal((40, s.stop - s.start))
    theta, phi, Y_hat, r2 = decode_harmonics(W, L, l, n_fit=3000, n_theta=6, n_phi=9)
    Y = np.stack([real_sh(l, m, theta.ravel(), phi.ravel()) for m in range(-l, l + 1)], 1)
    assert r2 > 0.999
    assert np.allclose(Y_hat.reshape(-1, 2 * l + 1), Y, atol=1e-6)
