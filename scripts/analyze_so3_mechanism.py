"""analyze_so3_mechanism.py — Test the quadratic-product mechanism on trained SO(3) nets.

For a run directory written by `train_so3_mechanism.py`, computes on fresh Haar
samples:
  - geodesic error, and after removing the l-components of x (irrep ablation)
  - exact irrep shares of the first-layer G-block preactivation, init vs final
  - fit z_j ≈ <A_j,G> + <B_j,H> + c_j (l=1 maps) and its R²
  - composability ‖A_j B_j‖ / (‖A_j‖‖B_j‖) vs a mismatched-neuron baseline
  - single-hidden-layer nets only: second-order (Taylor) readout test.
    Effective cross-term weight r_j = W_out[:, j] · σ''(c_j) · γ_j², per-neuron
    cos(r_j, vec A_j B_j), and cos between the implied bilinear map
    T = Σ_j r_j ⊗ A_j ⊗ B_j and the 3×3 matrix-multiplication tensor.
  - sign pairing: does neuron (A,B) have a partner (A,-B)?
Writes mechanism.json in the run directory.

Usage:
    python scripts/analyze_so3_mechanism.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import so3irreps.patch_torchvision  # noqa: F401  # must precede Lightning
from so3irreps.analysis.mechanism import (
    bilinear_signal,
    composability,
    encode,
    fit_l1_maps,
    irrep_shares,
    isotypic_projectors,
    readout_alignment,
)
from so3irreps.wigner import so3_random
from so3irreps.model import SO3CompositionNet
from so3irreps.data.so3_comp import _svd_project_to_so3

ABLATIONS = {"none": [], "l1": [1], "l2": [2], "l3": [3], "l2+l3": [2, 3]}


def _matmul_tensor() -> np.ndarray:
    """M[(i,j), (i,k), (k,j)] = 1, so that vec(GH) = M · vec G · vec H."""
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                M[3 * i + j, 3 * i + k, 3 * k + j] = 1.0
    return M


def _silu_dd(t: np.ndarray) -> np.ndarray:
    s = 1 / (1 + np.exp(-t))
    return 2 * s * (1 - s) + t * s * (1 - s) * (1 - 2 * s)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _load(run_dir: Path, name: str, config: dict) -> SO3CompositionNet:
    net = SO3CompositionNet(
        input_dim=config["input_dim"], output_dim=9, hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], activation=config["activation"],
        layernorm=config["layernorm"],
    )
    net.load_state_dict(torch.load(run_dir / name, map_location="cpu", weights_only=True))
    return net.eval().double()


def _output(net, G, H, x, k) -> torch.Tensor:
    with torch.no_grad():
        return net(torch.from_numpy(encode(G, x, k)), torch.from_numpy(encode(H, x, k)))


def _geodesic_deg_of(out: torch.Tensor, G, H) -> float:
    pred = _svd_project_to_so3(out).numpy()
    tr = np.einsum("nij,nij->n", pred, G @ H)
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).mean())


def _output_sectors(net, x: np.ndarray, k: int, P: dict, m: int = 150) -> dict:
    """Split the output on an m×m grid of (G, H) pairs into pure-G, pure-H and bilinear (double-
    centred) parts.

    GH is purely bilinear under Haar, so the bilinear part carries all the signal; the pure parts
    are junk to cancel. The bilinear part (rescaled to the target's) is compared with GH's own
    bilinear part, with and without the l ≥ 2 input components.
    """
    G, *_ = so3_random(m, seed=2001)
    H, *_ = so3_random(m, seed=2002)
    Gg, Hg = np.repeat(G, m, axis=0), np.tile(H, (m, 1, 1))
    x_l1 = x - sum((p @ x for l, p in P.items() if l >= 2), np.zeros_like(x))

    def parts(out):
        out = out.reshape(m, m, 9)
        mean = out.mean((0, 1))
        pure_G = out.mean(1) - mean
        pure_H = out.mean(0) - mean
        bil = out - pure_G[:, None] - pure_H[None] - mean
        return pure_G, pure_H, bil

    target = (Gg @ Hg).reshape(-1, 9)
    t_bil = parts(target)[2]
    t_rest = target.reshape(m, m, 9) - t_bil  # GH's own non-bilinear residue on the grid
    outs = {name: _output(net, Gg, Hg, xv, k).numpy() for name, xv in
            (("full", x), ("ablate_l>=2", x_l1))}

    pure_G, pure_H, bil = parts(outs["full"])
    v = {"G": m * (pure_G**2).sum(), "H": m * (pure_H**2).sum(), "B": (bil**2).sum()}
    tot = sum(v.values())
    err, rel = {}, {}
    for name, out in outs.items():
        b = parts(out)[2]
        # Rescale to the target's bilinear part (SVD projection is scale-free, MSE is not)
        b = b * (b * t_bil).sum() / (b**2).sum()
        rel[name] = float(((b - t_bil) ** 2).sum() / (t_bil**2).sum())
        err[name] = _geodesic_deg_of(torch.from_numpy((b + t_rest).reshape(-1, 9)), Gg, Hg)
    return {
        "frac_pure_G": float(v["G"] / tot), "frac_pure_H": float(v["H"] / tot),
        "frac_bilinear": float(v["B"] / tot), "bilinear_only_geodesic_deg": err,
        "bilinear_rel_mse": rel,
    }


def _pairing(A: np.ndarray, B: np.ndarray, W_out: np.ndarray, rng) -> dict:
    v = np.concatenate([A.reshape(len(A), 9), B.reshape(len(B), 9)], 1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    flipped = v.copy()
    flipped[:, 9:] *= -1
    shuffled = np.concatenate([v[:, :9], v[rng.permutation(len(v)), 9:]], 1)
    shuffled /= np.linalg.norm(shuffled, axis=1, keepdims=True)

    def best(q):
        sim = q @ v.T
        np.fill_diagonal(sim, -np.inf)
        return sim.max(1), sim.argmax(1)

    score, partner = best(flipped)
    base, _ = best(shuffled)
    paired = score > 0.9
    r_cos = [_cos(W_out[:, j], W_out[:, partner[j]]) for j in np.where(paired)[0]]
    return {
        "median_flip_match": float(np.median(score)),
        "median_shuffled_match": float(np.median(base)),
        "frac_paired_gt_0.9": float(paired.mean()),
        "frac_shuffled_gt_0.9": float((base > 0.9).mean()),
        "median_readout_cos_of_pairs": float(np.median(r_cos)) if r_cos else None,
    }


def analyze_run(run_dir: Path, n_samples: int = 20_000) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    k, x = config["k"], np.array(config["x"])
    net, net0 = _load(run_dir, "model.pt", config), _load(run_dir, "init_model.pt", config)
    G, *_ = so3_random(n_samples, seed=1001)
    H, *_ = so3_random(n_samples, seed=1002)
    P = isotypic_projectors(k)
    rng = np.random.default_rng(0)
    d = 3**k
    report: dict = {"run": run_dir.name, "config": {kk: v for kk, v in config.items() if kk != "x"}}

    # Irrep ablation: remove the l-components of x (equivalent to projecting W1).
    # Squares of the removed components have a nonzero mean that the output bias
    # absorbed, so also report the error after restoring the original output mean.
    out_full = _output(net, G, H, x, k)
    report["ablation_geodesic_deg"], report["ablation_geodesic_deg_recentered"] = {}, {}
    for name, ls in ABLATIONS.items():
        if not all(l in P for l in ls):
            continue
        out = _output(net, G, H, x - sum((P[l] @ x for l in ls), np.zeros(d)), k)
        recentered = out - out.mean(0) + out_full.mean(0)
        report["ablation_geodesic_deg"][name] = _geodesic_deg_of(out, G, H)
        report["ablation_geodesic_deg_recentered"][name] = _geodesic_deg_of(recentered, G, H)

    report["output_sectors"] = _output_sectors(net, x, k, P)

    # Exact irrep shares of the first-layer G-block, init vs final
    W1 = net.net[0].weight.detach().numpy()
    W1_0 = net0.net[0].weight.detach().numpy()
    R_sh = G[:5000]
    report["irrep_shares"] = {
        "init": {str(l): s for l, s in irrep_shares(W1_0[:, :d], x, k, R_sh).items()},
        "final": {str(l): s for l, s in irrep_shares(W1[:, :d], x, k, R_sh).items()},
        "final_H_block": {str(l): s for l, s in irrep_shares(W1[:, d:], x, k, R_sh).items()},
    }

    # l=1 maps of the first-layer preactivation, centred as LayerNorm would
    with torch.no_grad():
        z = net.first_preactivation(
            torch.from_numpy(encode(G, x, k)), torch.from_numpy(encode(H, x, k))
        ).numpy()
    if config["layernorm"]:
        z = z - z.mean(1, keepdims=True)
    A, B, r2 = fit_l1_maps(z, G, H)
    var = ((z - z.mean(0)) ** 2).sum(0)
    report["l1_fit"] = {
        "median_r2": float(np.median(r2)),
        "variance_weighted_r2": float((r2 * var).sum() / var.sum()),
    }

    comp = composability(A, B)
    comp_base = composability(A, B[rng.permutation(len(B))])
    report["composability"] = {
        "median": float(np.median(comp)), "median_mismatched": float(np.median(comp_base)),
    }

    W_out = net.net[-1].weight.detach().numpy()  # (9, hidden)
    report["pairing"] = _pairing(A, B, W_out, rng)

    if config["num_layers"] == 0:
        b1 = net.net[0].bias.detach().numpy()
        if config["layernorm"]:
            ln = net.net[1]
            gamma, beta = ln.weight.detach().numpy(), ln.bias.detach().numpy()
            gain, point = gamma**2, beta
        else:
            gain, point = np.ones_like(b1), z.mean(0)  # mean preactivation
        dd = np.full_like(b1, 2.0) if config["activation"] == "square" else _silu_dd(point)
        r = W_out * (dd * gain)[None, :]
        S = bilinear_signal(A, B)
        cos = readout_alignment(r, S)
        cos_base = readout_alignment(r, S[rng.permutation(len(S))])
        w = np.linalg.norm(r, axis=0) * np.linalg.norm(S.reshape(len(S), 9), axis=1)
        T = np.einsum("oj,jp,jq->opq", r, A.reshape(-1, 9), B.reshape(-1, 9))
        report["readout_alignment"] = {
            "weighted_mean_cos": float((cos * w).sum() / w.sum()),
            "weighted_mean_cos_shuffled": float((cos_base * w).sum() / w.sum()),
            "weight_frac_cos_gt_0.5": float(w[cos > 0.5].sum() / w.sum()),
            "weight_frac_cos_lt_-0.5": float(w[cos < -0.5].sum() / w.sum()),
            "bilinear_tensor_cos_with_matmul": _cos(T, _matmul_tensor()),
        }

    (run_dir / "mechanism.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--n-samples", type=int, default=20_000)
    args = p.parse_args()
    for rd in args.run_dirs:
        print(json.dumps(analyze_run(Path(rd), args.n_samples), indent=2))


if __name__ == "__main__":
    main()
