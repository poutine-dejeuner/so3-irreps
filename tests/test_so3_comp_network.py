"""Tests for the SO(3) composition network and learner."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from so3irreps.model import SO3CompositionNet
from so3irreps.learner import (
    SO3CompositionLearner,
    _so3_geodesic_distance,
)
from so3irreps.data.so3_comp import SO3CompositionDataModule


def test_network_forward_shape():
    net = SO3CompositionNet(input_dim=18, hidden_size=64, num_layers=1)
    G_vec = torch.randn(8, 9)
    H_vec = torch.randn(8, 9)
    out = net(G_vec, H_vec)
    assert out.shape == (8, 9)


def test_network_forward_shape_k2():
    net = SO3CompositionNet(input_dim=162, hidden_size=64, num_layers=1)
    G_vec = torch.randn(8, 81)
    H_vec = torch.randn(8, 81)
    out = net(G_vec, H_vec)
    assert out.shape == (8, 9)


def test_network_forward_shape_k3():
    net = SO3CompositionNet(input_dim=2 * 3**3, hidden_size=64, num_layers=1)
    G_vec = torch.randn(8, 27)
    H_vec = torch.randn(8, 27)
    out = net(G_vec, H_vec)
    assert out.shape == (8, 9)


def test_network_supports_tensor_orbit_output_dimension():
    net = SO3CompositionNet(
        input_dim=2 * 3**3,
        output_dim=3**3,
        hidden_size=64,
        num_layers=1,
    )
    G_vec = torch.randn(8, 27)
    H_vec = torch.randn(8, 27)

    assert net(G_vec, H_vec).shape == (8, 27)


def test_network_num_layers():
    net = SO3CompositionNet(input_dim=18, hidden_size=32, num_layers=3)
    G_vec = torch.randn(4, 9)
    H_vec = torch.randn(4, 9)
    out = net(G_vec, H_vec)
    assert out.shape == (4, 9)
    # Should be a Sequential with the right number of Linear layers
    linear_count = sum(1 for m in net.net if isinstance(m, torch.nn.Linear))
    # input proj + num_layers * hidden-to-hidden + output proj
    assert linear_count == 1 + 3 + 1  # 3 hidden layers, each has a Linear


def test_geodesic_distance():
    # Identity should have zero distance to itself
    R1 = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
    R2 = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
    d = _so3_geodesic_distance(R1, R2)
    assert torch.allclose(d, torch.zeros(10), atol=1e-5)

    # A 90° rotation around z should have distance π/2
    theta = torch.tensor([torch.pi / 2])
    R = torch.zeros(1, 3, 3)
    R[0, 0, 0] = torch.cos(theta)
    R[0, 0, 1] = -torch.sin(theta)
    R[0, 1, 0] = torch.sin(theta)
    R[0, 1, 1] = torch.cos(theta)
    R[0, 2, 2] = 1.0
    R_eye = torch.eye(3).unsqueeze(0)
    d2 = _so3_geodesic_distance(R, R_eye)
    assert torch.allclose(d2, theta, atol=1e-4)


def test_learner_predict_so3():
    dm = SO3CompositionDataModule(k=1, n=32, batch_size=8, seed=0)
    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=64, num_layers=1, lr=1e-4, weight_decay=0.0
    )

    G_vec = torch.randn(8, 3)
    H_vec = torch.randn(8, 3)
    pred = learner.predict_so3(G_vec, H_vec)
    assert pred.shape == (8, 3, 3)

    # Check it's in SO(3)
    identity = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
    RtR = torch.bmm(pred.transpose(1, 2), pred)
    assert torch.allclose(RtR, identity, atol=1e-5)
    det = torch.det(pred)
    assert torch.allclose(det, torch.ones(8), atol=1e-5)


def test_learner_training_step():
    dm = SO3CompositionDataModule(k=1, n=32, batch_size=8, seed=0)
    dm.prepare_data()
    dm.setup("fit")

    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=64, num_layers=1, lr=1e-4, weight_decay=0.0
    )

    batch = next(iter(dm.train_dataloader()))
    result = learner.training_step(batch, 0)
    assert "loss" in result
    assert result["loss"].item() > 0


def test_learner_trains_against_tensor_orbit_target():
    dm = SO3CompositionDataModule(
        k=3,
        n=32,
        batch_size=8,
        seed=0,
        target_type="orbit",
    )
    dm.setup("fit")
    learner = SO3CompositionLearner(
        datamodule=dm,
        hidden_size=64,
        num_layers=1,
        lr=1e-4,
        weight_decay=0.0,
    )
    batch = next(iter(dm.train_dataloader()))
    raw_output = learner(*batch[:2])
    expected_loss = F.mse_loss(raw_output, batch[3]) * (dm.output_dim / 3)
    result = learner.training_step(batch, 0)

    assert learner(*batch[:2]).shape == (8, 27)
    assert torch.allclose(result["loss"], expected_loss)
    with pytest.raises(RuntimeError, match="matrix-target"):
        learner.predict_so3(*batch[:2])


def test_network_square_activation_without_layernorm_is_quadratic_readout():
    # num_layers=0, no LayerNorm, z² activation: out = W2 (W1 x + b1)² + b2.
    net = SO3CompositionNet(
        input_dim=6, hidden_size=16, num_layers=0, activation="square", layernorm=False
    )
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in net.modules())
    lin1, lin2 = (m for m in net.modules() if isinstance(m, torch.nn.Linear))
    G_vec, H_vec = torch.randn(5, 3), torch.randn(5, 3)
    z = lin1(torch.cat([G_vec, H_vec], dim=-1))
    assert torch.allclose(net(G_vec, H_vec), lin2(z**2), atol=1e-5)


def test_network_first_preactivation_is_first_linear_output():
    net = SO3CompositionNet(input_dim=6, hidden_size=16, num_layers=0)
    G_vec, H_vec = torch.randn(5, 3), torch.randn(5, 3)
    lin1 = net.net[0]
    expected = lin1(torch.cat([G_vec, H_vec], dim=-1))
    assert torch.allclose(net.first_preactivation(G_vec, H_vec), expected)


def test_network_rejects_unknown_activation():
    with pytest.raises(ValueError):
        SO3CompositionNet(input_dim=6, hidden_size=16, activation="tanh")


def test_learner_passes_activation_and_layernorm():
    dm = SO3CompositionDataModule(k=1, n=32, batch_size=8)
    learner = SO3CompositionLearner(
        datamodule=dm, hidden_size=16, num_layers=0, activation="square", layernorm=False
    )
    assert learner.network.activation == "square"
    assert not learner.network.layernorm
