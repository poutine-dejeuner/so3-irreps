"""Tests for the irrep-share computation behind the purity figure."""

from __future__ import annotations

import numpy as np

from so3irreps.data.so3_onehot import block_slices
from scripts.plot_so3_purity import irrep_shares_sorted


def test_shares_rows_sum_to_one_and_sort_by_label():
    sl = block_slices(2)
    W = np.zeros((3, 34))
    W[0, sl[2]] = 1.0
    W[1, sl[1]] = 1.0
    W[2, :] = 1.0
    shares, labels = irrep_shares_sorted(W, L=2, idx=np.arange(3))
    assert np.allclose(shares.sum(1), 1)
    assert labels.tolist() == sorted(labels.tolist())
    assert np.allclose(shares[0], [1, 0])
