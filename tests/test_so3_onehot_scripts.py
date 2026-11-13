"""End-to-end tests for the band-limited one-hot SO(3) training and analysis scripts."""

from __future__ import annotations

import json

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from scripts.analyze_so3_onehot import analyze_run
from scripts.train_so3_onehot import parse_args, train


def test_train_then_analyze_tiny_run(tmp_path):
    args = parse_args(
        [
            "--L", "2", "--hidden-size", "16", "--n", "80", "--epochs", "3",
            "--optimizer", "sgd", "--lr", "0.01", "--init-scale", "0.5",
            "--output-dir", str(tmp_path),
        ]
    )
    run_dir = train(args)
    for name in ("init_model.pt", "model.pt", "config.json", "results.json"):
        assert (run_dir / name).exists()
    assert len(list((run_dir / "snapshots").glob("epoch_*.pt"))) == 3
    config = json.loads((run_dir / "config.json").read_text())
    assert config["activation"] == "square" and not config["layernorm"]

    rep = analyze_run(run_dir, n_samples=300)
    assert set(rep["test_rel_mse"]) == {"1", "2"}
    assert {"final", "init", "shuffled"} <= set(rep["neurons"])
    assert set(rep["ablate_by_label"]) <= {"1", "2"}
    assert [s["epoch"] for s in rep["dynamics"]] == [0, 1, 2]
    assert (run_dir / "onehot_analysis.json").exists()


def test_continue_training_from_previous_run(tmp_path):
    import torch

    first = train(parse_args([
        "--L", "1", "--hidden-size", "8", "--n", "40", "--epochs", "1",
        "--output-dir", str(tmp_path / "a"),
    ]))
    second = train(parse_args([
        "--L", "1", "--hidden-size", "8", "--n", "40", "--epochs", "4",
        "--init-from", str(first / "model.pt"), "--snapshot-every", "2",
        "--output-dir", str(tmp_path / "b"),
    ]))
    start = torch.load(second / "init_model.pt", weights_only=True)
    trained = torch.load(first / "model.pt", weights_only=True)
    assert all(torch.equal(start[k], trained[k]) for k in trained)
    config = json.loads((second / "config.json").read_text())
    assert config["init_from"] == str(first / "model.pt")
    # log-spaced snapshots for 4 epochs are 0..3; every-2 adds nothing new but must not break
    names = sorted(p.name for p in (second / "snapshots").glob("epoch_*.pt"))
    assert names == ["epoch_0000.pt", "epoch_0001.pt", "epoch_0002.pt", "epoch_0003.pt"]


def test_snapshot_every_adds_regular_epochs():
    from scripts.train_so3_onehot import snapshot_epochs

    e = snapshot_epochs(10_000, every=2_000)
    assert {0, 2_000, 4_000, 6_000, 8_000, 9_999} <= set(e)


def test_train_then_analyze_tiny_position_basis_run(tmp_path):
    args = parse_args([
        "--L", "2", "--hidden-size", "16", "--n", "80", "--epochs", "2",
        "--basis", "position", "--n-anchors", "64", "--anchor-seed", "3",
        "--output-dir", str(tmp_path),
    ])
    run_dir = train(args)
    assert "posN64" in run_dir.name
    config = json.loads((run_dir / "config.json").read_text())
    assert config["basis"] == "position" and config["input_dim"] == 128
    rep = analyze_run(run_dir, n_samples=300)
    assert set(rep["test_rel_mse"]) == {"1", "2"}
    assert 0.0 <= rep["frac_first_layer_norm_in_range_phi"] <= 1.0
    assert {"final", "init", "shuffled"} <= set(rep["neurons"])


def test_no_snapshots_option(tmp_path):
    run_dir = train(parse_args([
        "--L", "1", "--hidden-size", "8", "--n", "40", "--epochs", "2", "--no-snapshots",
        "--output-dir", str(tmp_path),
    ]))
    assert not (run_dir / "snapshots").exists() or not any((run_dir / "snapshots").iterdir())


def test_grad_clip_option_is_recorded(tmp_path):
    run_dir = train(parse_args([
        "--L", "1", "--hidden-size", "8", "--n", "40", "--epochs", "1", "--grad-clip", "0.1",
        "--no-snapshots", "--output-dir", str(tmp_path),
    ]))
    assert json.loads((run_dir / "config.json").read_text())["grad_clip"] == 0.1
