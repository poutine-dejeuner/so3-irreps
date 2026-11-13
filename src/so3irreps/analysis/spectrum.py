"""so3_irrep_spectrum.py — Fit-free per-neuron irrep spectra for one-hidden-layer
networks on the band-limited one-hot encoding of SO(3).

With input δ^L(G) = ⊕_l √d_l vec D^l(G), a first-layer weight block W_j^(l) gives a
preactivation whose Haar variance is exactly ‖W_j^(l)‖²_F (Schur orthogonality),
so a neuron's irrep spectrum is read off its weights: no regression, no sampling.
The same holds for readout blocks, whose l-block writes into √d_l vec D^l(GH).

Within a block, with A = W_G^(l), B = W_H^(l), r = W_out^(l) as d×d matrices, the
z² cross term contributes r · d⟨A, D(G)⟩⟨B, D(H)⟩, and the target block is
√d D(G)D(H); so Σ_j r_j ⊗ A_j ⊗ B_j should be ∝ the d×d matmul tensor and
each r_j ∝ A_j B_j (Lemma 7 of the draft).
"""

from __future__ import annotations

import numpy as np


def block_energies(W: np.ndarray, blocks: dict[int, slice]) -> np.ndarray:
    """(n_neurons, L) squared norm of each neuron's weights in each l-block.

    W is (n_neurons, D): rows are first-layer G- or H-weights, or the transposed readout.
    """
    return np.stack([(W[:, s] ** 2).sum(1) for s in blocks.values()], axis=1)


def active_neurons(contribution: np.ndarray, frac: float = 0.95) -> np.ndarray:
    """Indices of the smallest set of neurons covering `frac` of the total contribution."""
    order = np.argsort(contribution)[::-1]
    cum = np.cumsum(contribution[order])
    n = int(np.searchsorted(cum, frac * cum[-1]) + 1)
    return order[:n]


def agreement(a: np.ndarray, b: np.ndarray, L: int) -> np.ndarray:
    """L×L confusion matrix of labels a (rows) vs b (columns), labels in 1..L."""
    M = np.zeros((L, L), dtype=int)
    np.add.at(M, (a - 1, b - 1), 1)
    return M


def matmul_tensor(d: int) -> np.ndarray:
    """M[(i,j), (i,k), (k,j)] = 1, so that vec(AB) = M · vec A · vec B (row-major)."""
    M = np.zeros((d * d, d * d, d * d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                M[d * i + j, d * i + k, d * k + j] = 1.0
    return M


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def block_tensor_alignment(A: np.ndarray, B: np.ndarray, R: np.ndarray) -> dict:
    """A, B, R: (n, d, d) G-, H- and readout blocks of the neurons.

    Returns cos(Σ_j r_j⊗A_j⊗B_j, matmul tensor) and the per-neuron readout cosine cos(r_j, A_j
    B_j), weighted by ‖r_j‖‖A_j B_j‖.
    """
    n, d, _ = A.shape
    T = np.einsum("jo,jp,jq->opq", R.reshape(n, -1), A.reshape(n, -1), B.reshape(n, -1))
    S = (A @ B).reshape(n, -1)
    r = R.reshape(n, -1)
    norms = np.linalg.norm(r, axis=1) * np.linalg.norm(S, axis=1)
    cos = (r * S).sum(1) / (norms + 1e-30)
    return {
        "tensor_cos_with_matmul": _cos(T, matmul_tensor(d)),
        "weighted_readout_cos": float((cos * norms).sum() / (norms.sum() + 1e-30)),
    }


def product_unit_structure(A: np.ndarray, B: np.ndarray, R: np.ndarray) -> dict:
    """Is each neuron an elementary product unit (aᵀ D(G) u)(uᵀ D(H) b) → a bᵀ?

    A, B, R: (n, d, d) G-, H- and readout blocks. Reports the median ratio of
    second to first singular value (0 = rank one), |cos| between A's right and
    B's left singular vectors (shared inner vector u), |cos| of R with the
    outer product of A's left and B's right singular vectors, and the relative
    spread (max - min)/median of ‖r‖‖A‖‖B‖ across neurons.
    """
    svd = [np.linalg.svd(M) for M in (A, B, R)]
    ratio = {k: float(np.median(s[1][:, 1] / s[1][:, 0])) for k, s in zip("ABR", svd)}
    (uA, _, vA), (uB, _, vB) = svd[0], svd[1]
    a, u_A, u_B, b = uA[:, :, 0], vA[:, 0, :], uB[:, :, 0], vB[:, 0, :]
    outer = np.einsum("ni,nj->nij", a, b).reshape(len(A), -1)
    r = R.reshape(len(R), -1)
    readout_cos = np.abs((outer * r).sum(1)) / np.linalg.norm(r, axis=1)
    size = np.prod([np.linalg.norm(M.reshape(len(M), -1), axis=1) for M in (A, B, R)], axis=0)
    return {
        "median_rank1_ratio": ratio,
        "median_shared_inner_cos": float(np.median(np.abs((u_A * u_B).sum(1)))),
        "median_readout_outer_cos": float(np.median(readout_cos)),
        "norm_spread": float((size.max() - size.min()) / np.median(size)),
    }


def neuron_report(
    W_G: np.ndarray, W_H: np.ndarray, W_out: np.ndarray, L: int, frac: float = 0.95
) -> dict:
    """Irrep spectra, purity, agreement and census of the active neurons.

    W_G, W_H: (n, D) first-layer weights on δ^L(G) and δ^L(H); W_out: (D, n).
    """
    from so3irreps.data.so3_onehot import block_slices

    blocks = block_slices(L)
    E = {"G": block_energies(W_G, blocks), "H": block_energies(W_H, blocks),
         "R": block_energies(W_out.T, blocks)}
    contribution = np.sqrt(E["G"].sum(1) * E["H"].sum(1) * E["R"].sum(1))
    act = active_neurons(contribution, frac)
    P = {k: e[act] / (e[act].sum(1, keepdims=True) + 1e-30) for k, e in E.items()}
    purity = {k: p.max(1) for k, p in P.items()}
    labels = {k: p.argmax(1) + 1 for k, p in P.items()}
    joint = (P["G"] + P["H"] + P["R"]).argmax(1) + 1
    agree = (labels["G"] == labels["H"]) & (labels["H"] == labels["R"])

    within = {}
    for l, s in blocks.items():
        d = 2 * l + 1
        idx = act[joint == l]
        if len(idx) == 0:
            continue
        blocks_l = (W_G[idx, s].reshape(-1, d, d), W_H[idx, s].reshape(-1, d, d),
                    W_out[s][:, idx].T.reshape(-1, d, d))
        within[str(l)] = block_tensor_alignment(*blocks_l)
        within[str(l)]["product_units"] = product_unit_structure(*blocks_l)
        within[str(l)]["tensor_cos_all_neurons"] = block_tensor_alignment(
            W_G[:, s].reshape(-1, d, d), W_H[:, s].reshape(-1, d, d),
            W_out[s].T.reshape(-1, d, d),
        )["tensor_cos_with_matmul"]

    return {
        "n_active": int(len(act)),
        "median_purity": {k: float(np.median(v)) for k, v in purity.items()},
        "frac_purity_gt_0.9": {k: float((v > 0.9).mean()) for k, v in purity.items()},
        "frac_all_three_agree": float(agree.mean()),
        "agreement_G_vs_R": agreement(labels["G"], labels["R"], L).tolist(),
        "agreement_G_vs_H": agreement(labels["G"], labels["H"], L).tolist(),
        "census": {str(l): int((joint == l).sum()) for l in blocks},
        "within_irrep": within,
        "active_index": act.tolist(),
        "joint_label": joint.tolist(),
    }
