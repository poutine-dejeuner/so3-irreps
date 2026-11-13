"""Tests for the truncated SO(3) delta kernel used in the paper figure."""

from __future__ import annotations

import numpy as np

from scripts.plot_delta_kernel import haar_angle_density, kernel, kernel_closed_form


def test_kernel_matches_closed_form_and_peak_height():
    theta = np.linspace(0.01, np.pi, 200)
    for L in (1, 4, 16):
        assert np.allclose(kernel(theta, L), kernel_closed_form(theta, L), atol=1e-8)
        assert np.isclose(kernel(np.array([1e-9]), L)[0], (L + 1) * (2 * L + 1) * (2 * L + 3) / 3)


def test_kernel_has_unit_mass_under_haar():
    theta = np.linspace(0, np.pi, 200_001)
    for L in (1, 4, 16):
        mass = np.trapezoid(kernel(theta, L) * haar_angle_density(theta), theta)
        assert abs(mass - 1) < 1e-6


def test_heat_weighted_kernel_is_positive_with_unit_mass():
    from scripts.plot_delta_kernel import heat_kernel

    theta = np.linspace(0, np.pi, 200_001)
    for L in (4, 16):
        K = heat_kernel(theta, L)
        assert K.min() > -1e-3 * K.max()
        assert abs(np.trapezoid(K * haar_angle_density(theta), theta) - 1) < 1e-6
