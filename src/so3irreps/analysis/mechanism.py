"""so3_mechanism.py — Mechanistic analysis of SO(3) composition networks.

Hypothesis tested: first-layer preactivations are z_j ≈ <A_j, G> + <B_j, H> + b_j (l=1 maps), the
activation's quadratic term forms <A_j, G><B_j, H>, and the readout sums these into GH. Under Haar,
the part of neuron j's cross term that correlates with GH is A_j @ B_j / 9 (see `bilinear_signal`).
"""

from __future__ import annotations

import numpy as np
import torch

from so3irreps.data.so3_comp import _rotate_tensor_power

# so(3) generators: (L_a)_{bc} = -ε_{abc}
_EPS = np.zeros((3, 3, 3))
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[2, 1, 0] = _EPS[1, 0, 2] = -1.0
_GENERATORS = -_EPS


def isotypic_projectors(k: int) -> dict[int, np.ndarray]:
    """Orthogonal projectors of (R^3)^{⊗k} onto its l-isotypic components.

    Built from the Casimir C = -Σ_a (L_a^{(k)})², whose eigenvalue on the
    l-component is l(l+1). The index order matches `_rotate_tensor_power`.
    """
    dim = 3**k
    casimir = np.zeros((dim, dim))
    for L in _GENERATORS:
        Lk = np.zeros((dim, dim))
        for axis in range(k):
            factors = [np.eye(3)] * k
            factors[axis] = L
            term = factors[0]
            for f in factors[1:]:
                term = np.kron(term, f)
            Lk += term
        casimir -= Lk @ Lk
    evals, evecs = np.linalg.eigh(casimir)
    ls = np.round((-1 + np.sqrt(1 + 4 * np.clip(evals, 0, None))) / 2).astype(int)
    return {
        int(l): evecs[:, ls == l] @ evecs[:, ls == l].T for l in np.unique(ls)
    }


def encode(R: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """ρ(R)·x = R^{⊗k} x for a batch of rotations R (n, 3, 3)."""
    Rt = torch.from_numpy(np.asarray(R)).float()
    xt = torch.from_numpy(np.asarray(x)).float().expand(len(R), -1)
    return _rotate_tensor_power(Rt, xt, k).double().numpy()


def irrep_shares(W: np.ndarray, x: np.ndarray, k: int, R: np.ndarray) -> dict[int, float]:
    """Share of the variance over Haar G of z = W ρ(G) x carried by each l ≥ 1.

    Exact decomposition (no Wigner fit): z_l = W ρ(G) P_l x. W is (H, 3^k)
    (the G-block of the first layer), R are Haar samples.
    """
    P = isotypic_projectors(k)
    var = {}
    for l, p in P.items():
        if l == 0:
            continue  # constant in G
        z = encode(R, p @ x, k) @ W.T
        var[l] = float(((z - z.mean(0)) ** 2).sum())
    total = sum(var.values())
    return {l: v / total for l, v in var.items()}


def fit_l1_maps(
    z: np.ndarray, G: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit z_j ≈ <A_j, G> + <B_j, H> + c_j by least squares.

    Returns A, B of shape (n_neurons, 3, 3) and the per-neuron R².
    """
    n = len(G)
    X = np.concatenate([G.reshape(n, 9), H.reshape(n, 9), np.ones((n, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ coef
    r2 = 1 - (resid**2).sum(0) / ((z - z.mean(0)) ** 2).sum(0)
    A = coef[:9].T.reshape(-1, 3, 3)
    B = coef[9:18].T.reshape(-1, 3, 3)
    return A, B, r2


def bilinear_signal(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """9 · E[GH <A,G><B,H>] under Haar = A @ B, per neuron (n, 3, 3)."""
    return A @ B


def composability(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """‖A B‖_F / (‖A‖_F ‖B‖_F) per neuron; 0 means the cross term is useless."""
    num = np.linalg.norm(A @ B, axis=(1, 2))
    return num / (np.linalg.norm(A, axis=(1, 2)) * np.linalg.norm(B, axis=(1, 2)))


def readout_alignment(W_out: np.ndarray, S: np.ndarray) -> np.ndarray:
    """cos(W_out[:, j], vec S_j) per neuron.

    W_out is (9, n), S is (n, 3, 3).
    """
    s = S.reshape(len(S), 9)
    num = (W_out.T * s).sum(1)
    return num / (np.linalg.norm(W_out, axis=0) * np.linalg.norm(s, axis=1) + 1e-12)
