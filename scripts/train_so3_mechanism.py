"""train_so3_mechanism.py — Train one SO(3) composition model for mechanism analysis.

Vector encoding G^{⊗k}·x, matrix target GH, MSE loss, AdamW (as in the earlier
SO(3) runs, for comparability). Saves the initial and final weights and the
fixed vector x so `analyze_so3_mechanism.py` can compare against init.

Usage:
    python scripts/train_so3_mechanism.py --k 3 --activation square --no-layernorm \
        --num-layers 0 --output-dir results/<date>_so3_mechanism/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import so3irreps.patch_torchvision  # noqa: F401  # must precede Lightning

# isort: split
import lightning
import torch

from so3irreps.learner import SO3CompositionLearner
from so3irreps.data.so3_comp import SO3CompositionDataModule


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--activation", choices=["silu", "square"], default="silu")
    p.add_argument("--layernorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--num-layers", type=int, default=0,
                   help="extra H→H layers; 0 = a single hidden layer")
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1001)
    p.add_argument("--lr-gamma", type=float, default=0.95)
    p.add_argument("--lr-step", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def run_name(args: argparse.Namespace) -> str:
    ln = "ln" if args.layernorm else "noln"
    return (f"k{args.k}_{args.activation}_{ln}_d{args.num_layers}_h{args.hidden_size}"
            f"_lr{args.lr}_wd{args.wd}_s{args.seed}")


def train(args: argparse.Namespace) -> Path:
    lightning.seed_everything(args.seed)
    run_dir = Path(args.output_dir) / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = SO3CompositionDataModule(
        k=args.k, n=args.n, batch_size=args.batch_size,
        train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=args.seed,
    )
    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=args.hidden_size, num_layers=args.num_layers,
        lr=args.lr, weight_decay=args.wd, lr_gamma=args.lr_gamma, lr_step=args.lr_step,
        activation=args.activation, layernorm=args.layernorm,
    )
    torch.save(learner.network.state_dict(), run_dir / "init_model.pt")

    # Same construction as SO3CompositionDataModule.build_full_dataset
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(3**args.k, generator=gen)
    x = x / x.norm()
    config = {**vars(args), "x": x.tolist(), "input_dim": dm.input_dim}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    trainer = lightning.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1, log_every_n_steps=50,
        default_root_dir=str(run_dir), enable_progress_bar=False,
        enable_model_summary=False,
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
