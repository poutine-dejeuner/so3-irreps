"""Tests for the SO(3) composition datamodule."""

from __future__ import annotations

import torch

import so3irreps.patch_torchvision  # noqa: F401  # must precede lightning imports
from so3irreps.data.so3_comp import (
    SO3CompositionDataModule,
    _random_so3,
    _rotate_tensor_power,
    _svd_project_to_so3,
    _tensor_power,
)


def test_random_so3_shape_and_orthogonality():
    n = 100
    R = _random_so3(n)
    assert R.shape == (n, 3, 3)

    # Each matrix should be orthogonal: R @ R^T = I
    identity = torch.eye(3).unsqueeze(0).expand(n, -1, -1)
    RtR = torch.bmm(R.transpose(1, 2), R)
    assert torch.allclose(RtR, identity, atol=1e-5)

    # Determinant should be +1
    det = torch.det(R)
    assert torch.allclose(det, torch.ones(n), atol=1e-5)


def test_random_so3_is_haar():
    # Character orthogonality under Haar: E[χ_1] = 0, E[χ_1²] = 1, χ_1 = tr G.
    # Angle θ has density (1 - cos θ)/π, so E[cos θ] = -1/2.
    torch.manual_seed(0)
    R = _random_so3(200_000)
    tr = R.diagonal(dim1=1, dim2=2).sum(-1)
    assert abs(tr.mean().item()) < 0.02
    assert abs((tr**2).mean().item() - 1.0) < 0.03
    cos_theta = (tr - 1) / 2
    assert abs(cos_theta.mean().item() + 0.5) < 0.01

    # Left invariance: G0 @ R has the same character moments
    G0 = _random_so3(1)[0]
    tr_l = (G0 @ R).diagonal(dim1=1, dim2=2).sum(-1)
    assert abs(tr_l.mean().item()) < 0.02


def test_svd_project_to_so3():
    # Test with known SO(3) matrices — should be identity
    R = _random_so3(50)
    R_flat = R.reshape(50, 9)
    R_proj = _svd_project_to_so3(R_flat)
    assert R_proj.shape == (50, 3, 3)
    assert torch.allclose(R_proj, R, atol=1e-5)

    # Test with slightly perturbed matrices
    noise = 0.01 * torch.randn(50, 3, 3)
    R_noisy = R + noise
    R_proj2 = _svd_project_to_so3(R_noisy)
    # Projected should be in SO(3)
    identity = torch.eye(3).unsqueeze(0).expand(50, -1, -1)
    RtR2 = torch.bmm(R_proj2.transpose(1, 2), R_proj2)
    assert torch.allclose(RtR2, identity, atol=1e-5)
    det2 = torch.det(R_proj2)
    assert torch.allclose(det2, torch.ones(50), atol=1e-5)


def test_tensor_power_k1():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = _tensor_power(x, k=1)
    assert result.shape == (2, 3)
    assert torch.allclose(result, x)


def test_tensor_power_k2():
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = _tensor_power(x, k=2)
    # x ⊗ x should give (3,3) tensor flattened to 9
    assert result.shape == (2, 9)
    # For [1,0,0]: outer product has only [0,0]=1
    assert result[0, 0] == 1.0
    assert result[0, 1:].sum() < 1e-6
    # For [0,1,0]: outer product has only [4]=1
    assert result[1, 4] == 1.0


def test_tensor_power_k3():
    x = torch.tensor([[1.0, 0.0, 0.0]])
    result = _tensor_power(x, k=3)
    assert result.shape == (1, 27)
    assert result[0, 0] == 1.0
    assert result[0, 1:].sum() < 1e-6


def test_datamodule_full_dataset_k1():
    dm = SO3CompositionDataModule(k=1, n=100, batch_size=16, seed=0)
    full = dm.build_full_dataset()
    assert len(full) == 100

    G_vec, H_vec, GH_mat, GH_flat = full[0]
    assert G_vec.shape == (3,)  # G·x where x is uniform on S²
    assert H_vec.shape == (3,)  # H·y
    assert GH_mat.shape == (3, 3)
    assert GH_flat.shape == (9,)

    # G_vec = G @ x, so ||G_vec|| = ||x|| = 1 (G is orthogonal)
    assert torch.allclose(G_vec.norm(), torch.tensor(1.0), atol=1e-5)


def test_datamodule_full_dataset_k2():
    dm = SO3CompositionDataModule(k=2, n=50, batch_size=8, seed=1)
    full = dm.build_full_dataset()
    assert len(full) == 50

    G_vec, H_vec, GH_mat, GH_flat = full[0]
    assert G_vec.shape == (9,)  # 3^2 = 9
    assert H_vec.shape == (9,)


def test_datamodule_splits():
    dm = SO3CompositionDataModule(k=1, n=200, batch_size=16, seed=0)
    dm.prepare_data()
    dm.setup("fit")

    assert len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset) == 200

    # Splits should be disjoint
    train_idx = set(dm.train_dataset.indices)
    val_idx = set(dm.val_dataset.indices)
    test_idx = set(dm.test_dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_dataloader_batch():
    dm = SO3CompositionDataModule(k=1, n=200, batch_size=10, seed=0)
    dm.prepare_data()
    dm.setup("fit")

    G_vec, H_vec, GH_mat, GH_flat = next(iter(dm.train_dataloader()))
    assert G_vec.shape[0] <= 10
    assert G_vec.shape[1] == dm.single_input_dim
    assert H_vec.shape[1] == dm.single_input_dim
    assert GH_mat.shape == (G_vec.shape[0], 3, 3)
    assert GH_flat.shape == (G_vec.shape[0], 9)


def test_datamodule_full_dataset_k3():
    dm = SO3CompositionDataModule(k=3, n=50, batch_size=8, seed=2)
    full = dm.build_full_dataset()
    assert len(full) == 50

    G_vec, H_vec, GH_mat, GH_flat = full[0]
    assert G_vec.shape == (27,)  # 3^3 = 27
    assert H_vec.shape == (27,)


def test_datamodule_orbit_target_uses_input_representation_and_fixed_vector():
    dm = SO3CompositionDataModule(
        k=3,
        n=20,
        batch_size=4,
        seed=7,
        target_type="orbit",
    )
    full = dm.build_full_dataset()

    _, _, GH_mat, target = full[0]
    generator = torch.Generator().manual_seed(7)
    x_fixed = torch.randn(3**3, generator=generator)
    x_fixed = x_fixed / x_fixed.norm()
    expected = _rotate_tensor_power(
        GH_mat.unsqueeze(0),
        x_fixed.unsqueeze(0),
        k=3,
    ).squeeze(0)

    assert dm.output_dim == 27
    assert target.shape == (27,)
    assert torch.allclose(target, expected)
    assert torch.allclose(target.norm(), torch.tensor(1.0), atol=1e-5)


def test_encoding_uses_one_fixed_x_for_all_samples():
    # Regression: x must be shared by every sample and by G and H; a fresh x
    # per sample makes GH unrecoverable from the inputs.
    for k in (1, 2, 3):
        dm = SO3CompositionDataModule(k=k, n=64, seed=42)
        G_vec, H_vec, GH, _ = dm.build_full_dataset().tensors
        # G, H are not stored; rebuild them from the datamodule's RNG stream
        torch.manual_seed(42)
        G = _random_so3(64)
        H = _random_so3(64)
        assert torch.allclose(G @ H, GH, atol=1e-5)
        # Undo the rotation: (G^T)^{⊗k}·G_vec should be the same x everywhere
        x_from_G = _rotate_tensor_power(G.transpose(1, 2), G_vec, k)
        x_from_H = _rotate_tensor_power(H.transpose(1, 2), H_vec, k)
        assert torch.allclose(x_from_G, x_from_G[:1].expand_as(x_from_G), atol=1e-5)
        assert torch.allclose(x_from_H, x_from_G, atol=1e-5)
