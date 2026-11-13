"""Tests for the experiment summary used to produce the paper's numbers."""

from __future__ import annotations

import json

from scripts.summarize import summarize_run


def test_summarize_run_reads_analysis(tmp_path):
    rep = {
        "config": {"L": 2, "optimizer": "sgd", "lr": 0.2, "wd": 1e-3, "init_scale": 1.0,
                   "seed": 42, "basis": "fourier"},
        "test_rel_mse": {"1": 0.01, "2": 0.02}, "test_geodesic_deg": 0.5,
        "neurons": {"final": {
            "n_active": 152, "frac_purity_gt_0.9": {"G": 1.0, "H": 1.0, "R": 0.99},
            "median_purity": {"G": 1.0, "H": 1.0, "R": 1.0}, "frac_all_three_agree": 1.0,
            "census": {"1": 27, "2": 125},
            "within_irrep": {"1": {"tensor_cos_with_matmul": 1.0},
                             "2": {"tensor_cos_with_matmul": 0.99}}}},
    }
    d = tmp_path / "run"
    d.mkdir()
    (d / "onehot_analysis.json").write_text(json.dumps(rep))
    row = summarize_run(d / "onehot_analysis.json")
    assert row["census"] == "27/125" and row["pure"] == 0.99 and row["all_blocks_fitted"]
