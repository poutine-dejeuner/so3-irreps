"""plot_so3_harmonics.py — Spherical harmonics reconstructed from irrep-labelled neurons.

For each irrep l, take the neurons labelled l (by the block norms of their weights,
`neuron_report`), and fit a linear decoder from their G-preactivations
z_j(G) = <w_j, δ^L(G)> to vec D^l(G) on Haar samples. Evaluated at rotations
R(φ, θ, 0), which send the north pole ẑ to the point (θ, φ), the zonal column of the
decoded matrix is the vector of real spherical harmonics:
    D^l(R)[:, m'=0] = √(4π/(2l+1)) · Y_l^m(R ẑ)        (checked numerically)
so each sphere shows one Y_l^m as read out from the network. The figure puts it
above the exact harmonic, with a shared colour scale per (l, m).

Usage:
    python scripts/plot_so3_harmonics.py RUN_DIR --out litnotes/papier/so3/figures/harmonics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from so3irreps.wigner import euler_to_matrix, so3_random, wigner_D_real
from so3irreps.data.so3_onehot import bandlimited_onehot, block_slices

BLUE, NEUTRAL, RED = "#2a78d6", "#f0efec", "#e34948"  # diverging pair, gray midpoint
INK, INK_MUTED = "#0b0b0b", "#52514e"


def sphere_grid(n_theta: int, n_phi: int):
    """(θ, φ) mesh and rotations R(φ, θ, 0) with R ẑ = point (θ, φ)."""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n_theta),
                             np.linspace(0, 2 * np.pi, n_phi), indexing="ij")
    R = euler_to_matrix(phi.ravel(), theta.ravel(), np.zeros(theta.size))
    return theta, phi, R


def decode_harmonics(W_G: np.ndarray, L: int, l: int, n_fit: int = 20_000,
                     n_theta: int = 60, n_phi: int = 120, seed: int = 0):
    """Decode vec D^l(G) from the G-preactivations of the neurons whose G-weights are
    the rows of W_G, and return the harmonics Y_l^m on a (θ, φ) grid.

    Returns theta, phi (n_theta, n_phi), Y_hat (n_theta, n_phi, 2l+1), held-out R².
    """
    d = 2 * l + 1
    R_fit, *_ = so3_random(n_fit, seed=seed)
    Z = bandlimited_onehot(R_fit, L) @ W_G.T
    X = np.concatenate([Z, np.ones((n_fit, 1))], 1)
    T = wigner_D_real(l, R_fit).reshape(n_fit, -1)
    n_tr = int(0.8 * n_fit)
    coef, *_ = np.linalg.lstsq(X[:n_tr], T[:n_tr], rcond=None)
    resid = T[n_tr:] - X[n_tr:] @ coef
    r2 = 1 - (resid**2).sum() / ((T[n_tr:] - T[n_tr:].mean(0)) ** 2).sum()

    theta, phi, R = sphere_grid(n_theta, n_phi)
    Zg = bandlimited_onehot(R, L) @ W_G.T
    D_hat = (np.concatenate([Zg, np.ones((len(R), 1))], 1) @ coef).reshape(-1, d, d)
    Y_hat = D_hat[:, :, l] / np.sqrt(4 * np.pi / d)
    return theta, phi, Y_hat.reshape(n_theta, n_phi, d), float(r2)


def plot(run_dir: Path, out: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    from so3irreps.analysis.spectrum import neuron_report
    from so3irreps.wigner import real_sh

    config = json.loads((run_dir / "config.json").read_text())
    L = config["L"]
    sd = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    W1 = sd["net.0.weight"].double().numpy()
    W_out = sd["net.2.weight"].double().numpy()
    half = W1.shape[1] // 2
    rep = neuron_report(W1[:, :half], W1[:, half:], W_out, L)
    act, lab = np.array(rep["active_index"]), np.array(rep["joint_label"])

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    ncol = 2 * L + 1
    # Explicit layout (inches): columns aligned by m; for each l a "network" row
    # directly above an "exact" row, and a larger gap between successive l.
    cell, gap_in, gap_out, left, top, head = 0.95, 0.02, 0.22, 0.9, 0.1, 0.25
    width = left + ncol * cell + 0.1
    height = top + head + L * (2 * cell + gap_in) + (L - 1) * gap_out + 0.1
    fig = plt.figure(figsize=(width, height))

    def axes_at(row_top: float, col: int):
        rect = [(left + col * cell) / width, 1 - (row_top + cell) / height,
                cell / width, cell / height]
        return fig.add_axes(rect, projection="3d")

    for m in range(-L, L + 1):
        fig.text((left + (L + m + 0.5) * cell) / width, 1 - (top + 0.12) / height,
                 f"$m={m}$", ha="center", va="center", fontsize=8, color=INK_MUTED)
    stats = {}
    for l in range(1, L + 1):
        idx = act[lab == l]
        theta, phi, Y_hat, r2 = decode_harmonics(W1[idx, :half], L, l)
        Y = np.stack([real_sh(l, m, theta, phi) for m in range(-l, l + 1)], -1)
        stats[str(l)] = {"n_neurons": int(len(idx)), "decoder_r2": r2,
                         "max_abs_err_vs_Y": float(np.abs(Y_hat - Y).max()),
                         "max_abs_Y": float(np.abs(Y).max())}
        x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        group_top = top + head + (l - 1) * (2 * cell + gap_in + gap_out)
        # one shared symmetric colour scale per (l, m), set by the exact harmonic
        for row, (name, V) in enumerate((("network", Y_hat), ("exact", Y))):
            row_top = group_top + row * (cell + gap_in)
            fig.text((left + (L - l) * cell - 0.05) / width, 1 - (row_top + cell / 2) / height,
                     f"$l={l}$, {name}" if row == 0 else name, ha="right", va="center",
                     fontsize=8, color=INK if row == 0 else INK_MUTED)
            for i, m in enumerate(range(-l, l + 1)):
                ax = axes_at(row_top, L + m)
                lim = np.abs(Y[:, :, i]).max()
                ax.plot_surface(x, y, z,
                                facecolors=cmap(TwoSlopeNorm(0, -lim, lim)(V[:, :, i])),
                                rstride=1, cstride=1, linewidth=0, antialiased=False,
                                shade=False)
                ax.set_box_aspect((1, 1, 1))
                ax.set_proj_type("ortho")
                ax.view_init(elev=20, azim=35)
                for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                    setlim(-0.75, 0.75)
                ax.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    out.with_suffix(".json").write_text(json.dumps({"run": run_dir.name, **stats}, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default="litnotes/papier/so3/figures/harmonics")
    args = p.parse_args()
    print(json.dumps(plot(Path(args.run_dir), Path(args.out)), indent=2))


if __name__ == "__main__":
    main()
