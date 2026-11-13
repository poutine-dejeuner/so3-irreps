"""so3_comp.py — Datamodule for SO(3) group composition.

The task: given two SO(3) elements G, H encoded in the tensor-product
representation V = (R^3)^{⊗k} for k=1,2,3, predict the product GH.

Input: G and H are each observed by applying their k-fold tensor-power
representation to one fixed vector x. The resulting orbit vectors have
dimensions 3, 9, and 27 for k=1,2,3.

The target is either GH as a flattened 3×3 matrix or its orbit vector
(GH)^{⊗k}·x in the same representation used for the inputs.

Dataset: N triples (G, H, GH) with G, H sampled independently from the Haar
measure on SO(3).
"""

from __future__ import annotations

from typing import Literal

import lightning
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    random_split,
)


def _random_so3(n: int) -> torch.Tensor:
    """Sample n Haar-distributed SO(3) matrices.

    A uniform unit quaternion on S³ maps to a Haar-distributed rotation
    (S³ = SU(2) double-covers SO(3)). The rotation angle then has density
    (1 - cos θ)/π on [0, π]; note that cos θ ~ Uniform(-1, 1) is NOT Haar.

    Returns (n, 3, 3) tensor.
    """
    q = torch.randn(n, 4)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(-1)

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(n, 3, 3)

    return R


def _tensor_power(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return x^{⊗k} = outer product of x with itself k times, flattened.

    Args:
        x: (batch, 3) vectors.
        k: tensor power (1, 2, or 3).

    Returns:
        (batch, 3^k) tensor.
    """
    if k == 1:
        return x
    batch = x.shape[0]
    result = x  # (batch, 3)
    for _ in range(k - 1):
        # result: (batch, d), x: (batch, 3)
        result = result.unsqueeze(-1) * x.unsqueeze(1)  # (batch, d, 3)
        result = result.reshape(batch, -1)  # (batch, d*3)
    return result


def _rotate_tensor_power(G: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply G^{⊗k} to x ∈ (R^3)^{⊗k}, i.e. apply G on each tensor axis.

    For k=2: (G⊗G)·vec(X) = vec(G @ X @ G^T) where X ∈ R^{3×3}.
    For k=3: (G⊗G⊗G)·vec(X) = flatten of G applied on each index.

    Args:
        G: (batch, 3, 3) rotation matrices.
        x: (batch, 3^k) vectors in the tensor-power space.
        k: tensor power (1, 2, or 3).

    Returns:
        (batch, 3^k) rotated vectors. Preserves norm (G is orthogonal).
    """
    batch = G.shape[0]
    if k == 1:
        return torch.bmm(G, x.unsqueeze(-1)).squeeze(-1)

    # Reshape x to (batch, 3, ..., 3) with k spatial axes
    shape = (batch,) + (3,) * k
    result = x.reshape(shape)

    # Apply G on each spatial axis using matrix multiplication
    # For each axis i, contract: result_...,a_i,... = sum_j G_...,i,j * result_...,j,...
    # This is equivalent to: result = G ⊗ G ⊗ ... ⊗ G @ vec(X) but done per-axis

    if k == 2:
        # result (batch, 3, 3): apply G to rows and G^T to columns
        result = G @ result @ G.transpose(1, 2)
    elif k == 3:
        # result (batch, 3, 3, 3): apply G on each axis
        result = torch.einsum('bij,bjkl->bikl', G, result)   # axis 0
        result = torch.einsum('bij,bkjl->bkil', G, result)   # axis 1
        result = torch.einsum('bij,bklj->bkli', G, result)   # axis 2
    else:
        raise ValueError(f'k={k} not supported')

    return result.reshape(batch, -1)


def _so3_product(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication: R1 @ R2 for batched 3×3.

    Returns (batch, 3, 3).
    """
    return torch.bmm(R1, R2)


def _svd_project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """Project a batch of 3×3 matrices to SO(3) via SVD.

    M: (batch, 3, 3) or (batch, 9).
    Returns: (batch, 3, 3) in SO(3).

    Per Geist et al. (2024): R = U @ diag(1, 1, det(U @ V^T)) @ V^T
    This ensures det(R) = 1.
    """
    if M.dim() == 2 and M.shape[1] == 9:
        M = M.reshape(-1, 3, 3)

    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    # Vt is (batch, 3, 3) as returned by SVD
    Vt.transpose(-2, -1)

    det_UVt = torch.det(U @ Vt)  # (batch,)
    # Build D = diag(1, 1, det) on the same device as M
    D = torch.eye(3, device=M.device, dtype=M.dtype).unsqueeze(0).expand(M.shape[0], -1, -1).clone()
    D[:, 2, 2] = det_UVt

    R = U @ D @ Vt
    return R


class SO3CompositionDataModule(lightning.LightningDataModule):
    """Datamodule for SO(3) group composition: predict GH given G, H.

    Args:
        k:              Tensor power of the input representation (1, 2, or 3).
        n:              Number of triples in the full dataset.
        batch_size:     Mini-batch size.
        train_frac:     Fraction for training.
        val_frac:       Fraction for validation.
        test_frac:      Fraction for testing.
        num_workers:    Dataloader workers.
        seed:           Seed for reproducibility.
        target_type:    "matrix" for flattened GH or "orbit" for
                        (GH)^{⊗k}·x.
    """

    def __init__(
        self,
        k: int = 1,
        n: int = 10_000,
        batch_size: int = 128,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
        target_type: Literal["matrix", "orbit"] = "matrix",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])

        if k not in (1, 2, 3):
            raise ValueError(f"k must be 1, 2, or 3; got {k}")
        if target_type not in ("matrix", "orbit"):
            raise ValueError(f"target_type must be 'matrix' or 'orbit'; got {target_type!r}")

        self.k = k
        self.n = n
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_workers = num_workers
        self.seed = seed
        self.target_type = target_type

        # Input dimension: ρ_G(x) = G^{⊗k}·x where x is uniform on sphere in V^{⊗k}.
        # k=1: 3, k=2: 9, k=3: 27 per element; concat of G and H doubles it.
        self.single_input_dim: int = 3**k
        self.input_dim: int = 2 * self.single_input_dim
        self.output_dim: int = 9 if target_type == "matrix" else self.single_input_dim

        self._full: TensorDataset | None = None
        self.train_dataset: torch.utils.data.Dataset | None = None
        self.val_dataset: torch.utils.data.Dataset | None = None
        self.test_dataset: torch.utils.data.Dataset | None = None

    def build_full_dataset(self) -> TensorDataset:
        """Generate N triples (G, H, GH) and return a TensorDataset.

        Each sample: (G_vec, H_vec, GH_mat, target) where:
            G_vec: (3^k,) — ρ_G(x) = G^{⊗k}·x, action of G on a single
                    vector x drawn once (from ``seed``) uniformly on the
                    sphere in V^{⊗k} and shared by every sample.
            H_vec: (3^k,) — ρ_H(x), using the same fixed x.
            GH_mat: (3, 3) — product matrix, retained for evaluation.
            target: (9,) flattened GH for target_type="matrix", otherwise
                    (3^k,) ρ_{GH}(x).
        """
        torch.manual_seed(self.seed)

        G = _random_so3(self.n)  # (n, 3, 3)
        H = _random_so3(self.n)  # (n, 3, 3)
        GH = _so3_product(G, H)  # (n, 3, 3)

        # Sample one fixed vector x in V^{⊗k} = R^{3^k} for the whole dataset.
        # x is uniform on the unit sphere; shared for both G and H encodings.
        gen = torch.Generator().manual_seed(self.seed)
        x_fixed = torch.randn(3**self.k, generator=gen)
        x_fixed = x_fixed / x_fixed.norm()

        # Broadcast to batch
        x = x_fixed.unsqueeze(0).expand(self.n, -1)  # (n, 3^k)

        # Encode: ρ_G(x) = G^{⊗k}·x, same x for both
        G_vec = _rotate_tensor_power(G, x, self.k)  # (n, 3^k)
        H_vec = _rotate_tensor_power(H, x, self.k)  # (n, 3^k)

        if self.target_type == "matrix":
            target = GH.reshape(self.n, -1)
        else:
            target = _rotate_tensor_power(GH, x, self.k)

        return TensorDataset(G_vec, H_vec, GH, target)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        if self._full is None:
            self._full = self.build_full_dataset()

        n_total = len(self._full)
        n_val = int(round(self.val_frac * n_total))
        n_test = int(round(self.test_frac * n_total))
        n_train = n_total - n_val - n_test
        if n_train < 0:
            raise ValueError(f"val_frac + test_frac = {self.val_frac + self.test_frac} > 1")

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self._full, [n_train, n_val, n_test], generator=generator
        )

    def _dataloader(self, dataset, shuffle: bool) -> DataLoader:
        assert dataset is not None, "call setup() before requesting a dataloader"
        # Fetch each batch with one indexing call instead of per-sample + collate
        # (~5x faster); BatchSampler over the same sampler gives identical batches.
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=None,
            sampler=BatchSampler(sampler, self.batch_size, drop_last=False),
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None
        return self._dataloader(self.test_dataset, shuffle=False)
