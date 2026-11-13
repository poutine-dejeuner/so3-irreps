"""Tests for per-neuron scaled SGD (time-rescaled gradient flow of Marchetti et al.)."""

from __future__ import annotations

import torch

from so3irreps.optimizers import PerNeuronScaledSGD


def _layers(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(4, 3), torch.nn.Linear(3, 2)


def test_step_scales_each_neuron_gradient_by_norm_power():
    lin1, lin2 = _layers()
    for p in (*lin1.parameters(), *lin2.parameters()):
        p.grad = torch.ones_like(p)
    before = [p.detach().clone() for p in (lin1.weight, lin1.bias, lin2.weight, lin2.bias)]
    norms = torch.sqrt((lin1.weight**2).sum(1) + lin1.bias**2 + (lin2.weight**2).sum(0))
    PerNeuronScaledSGD(lin1, lin2, lr=0.1, degree=2).step()
    scale = norms ** (1 - 2)
    assert torch.allclose(lin1.weight, before[0] - 0.1 * scale[:, None])
    assert torch.allclose(lin1.bias, before[1] - 0.1 * scale)
    assert torch.allclose(lin2.weight, before[2] - 0.1 * scale[None, :])
    assert torch.allclose(lin2.bias, before[3] - 0.1)  # output bias: plain SGD


def test_degree_one_is_plain_sgd():
    lin1, lin2 = _layers(1)
    ref1, ref2 = _layers(1)
    for layers in ((lin1, lin2), (ref1, ref2)):
        for p in (*layers[0].parameters(), *layers[1].parameters()):
            p.grad = torch.full_like(p, 0.5)
    PerNeuronScaledSGD(lin1, lin2, lr=0.2, degree=1).step()
    torch.optim.SGD([*ref1.parameters(), *ref2.parameters()], lr=0.2).step()
    for a, b in zip((*lin1.parameters(), *lin2.parameters()), (*ref1.parameters(), *ref2.parameters())):
        assert torch.allclose(a, b)
