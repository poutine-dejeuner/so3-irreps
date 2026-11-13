"""Tests for training options used by the band-limited one-hot SO(3) experiments."""

from __future__ import annotations

import torch

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from so3irreps.model import SO3CompositionNet
from so3irreps.learner import SO3CompositionLearner
from so3irreps.data.so3_onehot import SO3OneHotDataModule


def test_init_scale_scales_linear_parameters_only():
    torch.manual_seed(0)
    ref = SO3CompositionNet(input_dim=6, hidden_size=16, num_layers=0)
    torch.manual_seed(0)
    small = SO3CompositionNet(input_dim=6, hidden_size=16, num_layers=0, init_scale=0.1)
    for m_ref, m_small in zip(ref.net, small.net):
        factor = 0.1 if isinstance(m_ref, torch.nn.Linear) else 1.0
        for p_ref, p_small in zip(m_ref.parameters(), m_small.parameters()):
            assert torch.allclose(p_small, factor * p_ref)


def test_learner_optimizer_choice():
    dm = SO3OneHotDataModule(L=2, n=32, batch_size=8)
    sgd = SO3CompositionLearner(datamodule=dm, hidden_size=16, num_layers=0, optimizer="sgd")
    adamw = SO3CompositionLearner(datamodule=dm, hidden_size=16, num_layers=0)
    assert isinstance(sgd.configure_optimizers()["optimizer"], torch.optim.SGD)
    assert isinstance(adamw.configure_optimizers()["optimizer"], torch.optim.AdamW)


def test_onehot_step_logs_per_irrep_errors_and_geodesic():
    dm = SO3OneHotDataModule(L=2, n=32, batch_size=8)
    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=16, num_layers=0, activation="square", layernorm=False
    )
    logged = {}
    learner.log = lambda name, value, **kw: logged.__setitem__(name, value)
    batch = dm.build_full_dataset().tensors
    out = learner.shared_step(batch, 0, "val")
    assert torch.isfinite(out["loss"])
    assert {"val/rel_mse_l1", "val/rel_mse_l2", "val/geodesic_error"} <= set(logged)


def test_train_step_logs_only_the_loss_when_train_metrics_off():
    # Metrics (SVD, per-block errors) cost ~3x in training speed; they can be eval-only.
    dm = SO3OneHotDataModule(L=2, n=32, batch_size=8)
    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=16, num_layers=0, train_metrics=False
    )
    logged = {}
    learner.log = lambda name, value, **kw: logged.__setitem__(name, value)
    learner.shared_step(dm.build_full_dataset().tensors, 0, "train")
    assert set(logged) == {"train/loss"}


def test_onehot_geodesic_is_zero_for_perfect_prediction():
    dm = SO3OneHotDataModule(L=2, n=32, batch_size=8)
    learner = SO3CompositionLearner(datamodule=dm, hidden_size=16, num_layers=0)
    G_enc, H_enc, GH, target = dm.build_full_dataset().tensors
    learner.forward = lambda g, h: target
    logged = {}
    learner.log = lambda name, value, **kw: logged.__setitem__(name, value)
    learner.shared_step((G_enc, H_enc, GH, target), 0, "val")
    assert logged["val/geodesic_error"] < 1e-3
    assert logged["val/rel_mse_l2"] < 1e-10


def test_learner_per_neuron_sgd_option():
    from so3irreps.optimizers import PerNeuronScaledSGD

    dm = SO3OneHotDataModule(L=1, n=32, batch_size=8)
    learner = SO3CompositionLearner(datamodule=dm, hidden_size=8, num_layers=0,
                                    activation="square", layernorm=False, optimizer="pnsgd")
    assert isinstance(learner.configure_optimizers()["optimizer"], PerNeuronScaledSGD)


def test_bias_free_network_is_homogeneous_of_degree_three():
    net = SO3CompositionNet(input_dim=6, hidden_size=16, num_layers=0, activation="square",
                            layernorm=False, bias=False)
    assert all(m.bias is None for m in net.modules() if isinstance(m, torch.nn.Linear))
    g, h = torch.randn(4, 3), torch.randn(4, 3)
    out = net(g, h)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(2.0)
    assert torch.allclose(net(g, h), 8 * out, atol=1e-5)  # θ → 2θ gives output × 2³


def test_per_neuron_sgd_without_biases():
    from so3irreps.optimizers import PerNeuronScaledSGD

    dm = SO3OneHotDataModule(L=1, n=32, batch_size=8)
    learner = SO3CompositionLearner(datamodule=dm, hidden_size=8, num_layers=0, activation="square",
                                    layernorm=False, optimizer="pnsgd", bias=False)
    opt = learner.configure_optimizers()["optimizer"]
    assert isinstance(opt, PerNeuronScaledSGD)
    learner.log = lambda *a, **k: None
    loss = learner.shared_step(dm.build_full_dataset().tensors, 0, "train")["loss"]
    loss.backward()
    opt.step()
