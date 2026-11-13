"""Tests for the degree filtration: D^l is a polynomial of degree exactly l in G."""

from __future__ import annotations

from scripts.degree_filtration import degree_fit_r2


def test_d2_needs_degree_two():
    r2 = degree_fit_r2(l=2, max_degree=2, n=3000)
    assert r2[1] < 0.05 and r2[2] > 0.999
