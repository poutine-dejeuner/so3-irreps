"""Tests for the numerical check of the availability (capacity) bound."""

from __future__ import annotations

from scripts.capacity_check import best_bilinear_r2


def test_one_probe_k3_matches_rank_over_dimension():
    # k=3, one probe x: ranks (m_1, m_2, m_3) = (3, 2, 1), bound r_l/(2l+1) = (1, .4, .143)
    r2 = best_bilinear_r2(k=3, n_probes=1, n=40_000, seed=0)  # held-out shrinkage ~ features/n
    assert r2[1] > 0.99
    assert abs(r2[2] - 0.4) < 0.05
    assert abs(r2[3] - 1 / 7) < 0.05
