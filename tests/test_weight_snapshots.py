"""Tests for the WeightSnapshots callback."""

from __future__ import annotations

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports

# isort: split
import lightning
import torch

from so3irreps.snapshots import WeightSnapshots, log_spaced_epochs
from so3irreps.learner import SO3CompositionLearner
from so3irreps.data.so3_onehot import SO3OneHotDataModule


def test_log_spaced_epochs():
    e = log_spaced_epochs(1000, per_decade=4)
    assert e[0] == 0 and e[-1] == 999 and e == sorted(set(e))
    assert {1, 10, 100} <= set(e)


def test_snapshots_saved_at_requested_epochs(tmp_path):
    dm = SO3OneHotDataModule(L=1, n=40, batch_size=8)
    learner = SO3CompositionLearner(datamodule=dm, hidden_size=8, num_layers=0)
    cb = WeightSnapshots(tmp_path, epochs=[0, 2])
    trainer = lightning.Trainer(
        max_epochs=3, callbacks=[cb], logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(learner, datamodule=dm)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["epoch_0000.pt", "epoch_0002.pt"]
    state = torch.load(tmp_path / "epoch_0002.pt", weights_only=True)
    assert set(state) == set(learner.network.state_dict())
