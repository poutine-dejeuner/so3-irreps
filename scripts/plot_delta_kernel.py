"""plot_delta_kernel.py — The truncated delta on SO(3) for increasing band-limit L.

K_L(θ) = Σ_{l=0}^{L} (2l+1) χ_l(θ), χ_l(θ) = sin((2l+1)θ/2)/sin(θ/2), θ the rotation
angle of GᵀH. Panel (a): K_L/K_L(0), the peak narrows (width ~ 1/L). Panel (b): the
cumulative Haar mass ∫_0^θ K_L dμ of the sharp truncation does NOT tend to a step:
K_L is not positive and its tail, weighted by the Haar density ∝ θ², keeps an O(1)
oscillation (the Pinsky phenomenon of Fourier series in dimension ≥ 3); K_L → δ only
against smooth test functions. Panel (c): with heat-kernel weights
e^{-t l(l+1)}, t = 8/L², the kernel is positive and its mass concentrates.

Usage:
    python scripts/plot_delta_kernel.py --out litnotes/papier/so3/figures/delta_kernel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LS = (1, 2, 4, 8, 16)
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")  # ordinal blue, light→dark
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def kernel(theta: np.ndarray, L: int) -> np.ndarray:
    """K_L(θ) by summing characters (θ = 0 handled as the limit Σ d_l²)."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)
    small = np.abs(theta) < 1e-6
    s = np.sin(theta[~small] / 2)
    for l in range(L + 1):
        d = 2 * l + 1
        out[~small] += d * np.sin(d * theta[~small] / 2) / s
        out[small] += d * d
    return out


def kernel_closed_form(theta: np.ndarray, L: int) -> np.ndarray:
    """K_L = −(1/sin x) d/dx[sin(2(L+1)x)/(2 sin x)], x = θ/2 (derivative written out)."""
    x = np.asarray(theta, dtype=float) / 2
    a = 2 * (L + 1)
    num = a * np.cos(a * x) * np.sin(x) - np.sin(a * x) * np.cos(x)
    return -num / (2 * np.sin(x) ** 3)


def heat_kernel(theta: np.ndarray, L: int, t: float | None = None) -> np.ndarray:
    """Σ_{l≤L} e^{-t l(l+1)} d_l χ_l(θ), default t = 8/L² (weight ~2e-4 at l = L, so the
    truncation leaves the kernel positive to ~1e-5)."""
    t = 8 / L**2 if t is None else t
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)
    small = np.abs(theta) < 1e-6
    s = np.sin(theta[~small] / 2)
    for l in range(L + 1):
        d, w = 2 * l + 1, np.exp(-t * l * (l + 1))
        out[~small] += w * d * np.sin(d * theta[~small] / 2) / s
        out[small] += w * d * d
    return out


def haar_angle_density(theta: np.ndarray) -> np.ndarray:
    """Density of the rotation angle under Haar measure on SO(3)."""
    return (1 - np.cos(theta)) / np.pi


def plot(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED})
    theta = np.linspace(1e-4, np.pi, 4000)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.5))
    for ax in axes:
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                      ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"rotation angle $\theta$ of $G^\top H$")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for L, c in zip(LS, RAMP):
        K = kernel(theta, L)
        axes[0].plot(theta, K / K[0], color=c, lw=1.5, label=f"$L={L}$")
        for ax, kern in ((axes[1], K), (axes[2], heat_kernel(theta, L))):
            dens = kern * haar_angle_density(theta)
            cum = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(theta))])
            ax.plot(theta, cum, color=c, lw=1.5)
    axes[0].set_ylabel(r"$K_L(\theta)\,/\,K_L(0)$")
    axes[0].set_title("(a) truncated delta, normalised height", fontsize=8, color=INK, loc="left")
    for ax in axes[1:]:
        ax.axhline(1, color=GRID, lw=0.8, zorder=0)
        ax.set_ylabel(r"mass within angle $\theta$")
    axes[1].set_title("(b) sharp truncation: no convergence", fontsize=8, color=INK, loc="left")
    axes[2].set_title(r"(c) heat weights $e^{-t\,l(l+1)}$, $t=8/L^2$", fontsize=8, color=INK,
                      loc="left")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="litnotes/papier/so3/figures/delta_kernel")
    plot(Path(p.parse_args().out))


if __name__ == "__main__":
    main()
