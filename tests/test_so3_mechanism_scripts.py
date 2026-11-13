"""End-to-end tests for the SO(3) mechanism training and analysis scripts."""

from __future__ import annotations

import json

import pytest

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from scripts.analyze_so3_mechanism import analyze_run
from scripts.train_so3_mechanism import parse_args, train


@pytest.mark.parametrize("k", [2, 3])
def test_train_then_analyze_tiny_square_run(tmp_path, k):
    args = parse_args(
        [
            "--k", str(k), "--activation", "square", "--no-layernorm",
            "--num-layers", "0", "--hidden-size", "16", "--n", "64",
            "--epochs", "1", "--output-dir", str(tmp_path),
        ]
    )
    run_dir = train(args)
    for name in ("init_model.pt", "model.pt", "config.json", "results.json"):
        assert (run_dir / name).exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert len(config["x"]) == 3**k and config["activation"] == "square"

    report = analyze_run(run_dir, n_samples=500)
    assert set(report["irrep_shares"]["final"]) == {str(l) for l in range(1, k + 1)}
    expected = {"none", "l1", "l2"} | ({"l3", "l2+l3"} if k == 3 else set())
    assert set(report["ablation_geodesic_deg"]) == expected
    assert set(report["ablation_geodesic_deg_recentered"]) == expected
    sec = report["output_sectors"]
    assert abs(sec["frac_pure_G"] + sec["frac_pure_H"] + sec["frac_bilinear"] - 1) < 1e-6
    assert set(sec["bilinear_only_geodesic_deg"]) == {"full", "ablate_l>=2"}
    assert set(sec["bilinear_rel_mse"]) == {"full", "ablate_l>=2"}
    assert "readout_alignment" in report  # single hidden layer → readout test runs
    assert (run_dir / "mechanism.json").exists()
