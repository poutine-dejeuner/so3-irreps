"""train_so3_onehot.py — SO(3) composition with band-limited one-hot input and target.

Input (δ^L(G), δ^L(H)), target δ^L(GH), δ^L(G) = ⊕_{l≤L} √(2l+1) vec D^l(G).
One hidden layer, z² activation, no normalisation, MSE (the SO(3) counterpart of
one-hot finite-group composition). Saves the initial and final weights and
log-spaced weight snapshots for the dynamics analysis.

Usage:
    python scripts/train_so3_onehot.py --L 3 --optimizer sgd --lr 0.05 \
        --init-scale 0.1 --wd 1e-4 --output-dir results/<date>_so3_onehot/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import so3irreps.patch_torchvision  # noqa: F401  # must precede Lightning

# isort: split
import lightning
import torch

from so3irreps.snapshots import WeightSnapshots, log_spaced_epochs
from so3irreps.learner import SO3CompositionLearner
from so3irreps.data.so3_onehot import SO3OneHotDataModule


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--activation", choices=["silu", "square"], default="square")
    p.add_argument("--layernorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--optimizer", choices=["sgd", "adamw", "pnsgd"], default="sgd")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr-gamma", type=float, default=1.0, help="StepLR factor; 1 = constant lr")
    p.add_argument("--lr-step", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", default=None,
                   help="start from these network weights (exact continuation for plain SGD)")
    p.add_argument("--no-bias", action="store_true",
                   help="bias-free (homogeneous) network, as in Marchetti et al.")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="clip the gradient norm (0 = off), as in Marchetti et al.")
    p.add_argument("--no-snapshots", action="store_true",
                   help="save no weight snapshots (very wide networks: ~1 GB each)")
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="also snapshot every N epochs (in addition to log-spaced ones)")
    p.add_argument("--basis", choices=["fourier", "position"], default="fourier",
                   help="position: sampled truncated delta K_L(G, R_i)/√N at Haar anchors")
    p.add_argument("--n-anchors", type=int, default=1024)
    p.add_argument("--anchor-seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def run_name(args: argparse.Namespace) -> str:
    name = (f"L{args.L}_{args.activation}_{args.optimizer}_lr{args.lr}_wd{args.wd}"
            f"_init{args.init_scale}_h{args.hidden_size}_s{args.seed}")
    if args.basis == "position":
        name += f"_posN{args.n_anchors}"
    return f"{name}_cont{args.epochs}" if args.init_from else name


def snapshot_epochs(max_epochs: int, every: int = 0) -> list[int]:
    """Log-spaced epochs, plus every `every` epochs if every > 0."""
    epochs = set(log_spaced_epochs(max_epochs))
    if every > 0:
        epochs |= set(range(0, max_epochs, every))
    return sorted(epochs)


def train(args: argparse.Namespace) -> Path:
    lightning.seed_everything(args.seed)
    run_dir = Path(args.output_dir) / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = SO3OneHotDataModule(L=args.L, n=args.n, batch_size=args.batch_size, seed=args.seed,
                             basis=args.basis, n_anchors=args.n_anchors,
                             anchor_seed=args.anchor_seed,
                             device="cuda" if torch.cuda.is_available() else "cpu")
    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=args.hidden_size, num_layers=0,
        lr=args.lr, weight_decay=args.wd, lr_gamma=args.lr_gamma, lr_step=args.lr_step,
        activation=args.activation, layernorm=args.layernorm,
        optimizer=args.optimizer, init_scale=args.init_scale, train_metrics=False,
        bias=not args.no_bias,
    )
    if args.init_from:
        learner.network.load_state_dict(torch.load(args.init_from, weights_only=True))
    torch.save(learner.network.state_dict(), run_dir / "init_model.pt")
    config = {**vars(args), "num_layers": 0, "input_dim": dm.input_dim,
              "output_dim": dm.output_dim}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    trainer = lightning.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1, log_every_n_steps=50,
        default_root_dir=str(run_dir), enable_progress_bar=False,
        enable_model_summary=False, enable_checkpointing=False,
        gradient_clip_val=args.grad_clip or None,
        callbacks=[] if args.no_snapshots else [
            WeightSnapshots(run_dir / "snapshots", snapshot_epochs(args.epochs, args.snapshot_every))
        ],
    )
    trainer.fit(learner, datamodule=dm)
    test = trainer.test(learner, datamodule=dm)
    torch.save(learner.network.state_dict(), run_dir / "model.pt")

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    results = {"metrics": metrics, "test": test[0] if test else {}}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"run": run_name(args), **results}, indent=2))
    return run_dir


if __name__ == "__main__":
    train(parse_args())
