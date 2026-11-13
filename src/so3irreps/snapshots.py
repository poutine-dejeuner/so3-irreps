"""snapshots.py — Save the network's weights at chosen epochs, for training-dynamics analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lightning import Callback


def log_spaced_epochs(max_epochs: int, per_decade: int = 8) -> list[int]:
    """0, then ~per_decade log-spaced epochs per decade, up to max_epochs - 1."""
    grid = np.logspace(0, np.log10(max(max_epochs - 1, 1)), per_decade * 4 + 1)
    epochs = {0, max_epochs - 1, *np.round(grid).astype(int).tolist()}
    for p in range(int(np.log10(max(max_epochs, 1))) + 1):
        epochs.add(10**p)
    return sorted(e for e in epochs if 0 <= e < max_epochs)


class WeightSnapshots(Callback):
    """Save `pl_module.network.state_dict()` at the end of the listed training epochs."""

    def __init__(self, out_dir: str | Path, epochs: list[int]):
        self.out_dir = Path(out_dir)
        self.epochs = set(epochs)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if epoch in self.epochs:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            state = {k: v.detach().cpu() for k, v in pl_module.network.state_dict().items()}
            torch.save(state, self.out_dir / f"epoch_{epoch:04d}.pt")
