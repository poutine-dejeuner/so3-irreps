"""plot_so3_purity.py — Irrep selection: per-neuron irrep shares before and after training.

(a) Initialisation: share of each l in the energy of each neuron's G-weights (a random
sample of neurons). (b) After training: the active neurons, sorted by label; each row
is concentrated in one l. (c) Number of active neurons per irrep, against d_l³ and
d_l³ + d_l.

Usage:
    python scripts/plot_so3_purity.py INIT_RUN_DIR TRAINED_RUN_DIR --out litnotes/papier/so3/figures/purity
INIT_RUN_DIR provides init_model.pt, TRAINED_RUN_DIR provides model.pt and config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from so3irreps.analysis.spectrum import block_energies, neuron_report
from so3irreps.data.so3_onehot import block_slices

INK, INK_MUTED, BAR, MARK = "#0b0b0b", "#52514e", "#2a78d6", "#0b0b0b"


def irrep_shares_sorted(W_G: np.ndarray, L: int, idx: np.ndarray):
    """Energy share per l of the rows idx of W_G, sorted by dominant l."""
    E = block_energies(W_G[idx], block_slices(L))
    shares = E / E.sum(1, keepdims=True)
    labels = shares.argmax(1) + 1
    order = np.lexsort((-shares.max(1), labels))
    return shares[order], labels[order]


def plot(init_dir: Path, run_dir: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import LinearSegmentedColormap

    L = json.loads((run_dir / "config.json").read_text())["L"]

    def load(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        W1, Wo = sd["net.0.weight"].double().numpy(), sd["net.2.weight"].double().numpy()
        h = W1.shape[1] // 2
        return W1[:, :h], W1[:, h:], Wo

    G0, _, _ = load(init_dir / "init_model.pt")
    G1, H1, R1 = load(run_dir / "model.pt")
    rep = neuron_report(G1, H1, R1, L, frac=0.999)
    act = np.array(rep["active_index"])
    rng = np.random.default_rng(0)
    s0, _ = irrep_shares_sorted(G0, L, rng.choice(len(G0), size=len(act), replace=False))
    s1, lab1 = irrep_shares_sorted(G1, L, act)

    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_MUTED, "xtick.color": INK_MUTED,
                         "ytick.color": INK_MUTED, "axes.labelcolor": INK})
    fig = plt.figure(figsize=(7.4, 2.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.55, 1.3], wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    cax = fig.add_subplot(gs[0, 2])
    for ax, sh, title in ((axes[0], s0, "(a) initialisation"),
                          (axes[1], s1, "(b) trained, active neurons")):
        im = ax.imshow(sh, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(L), [f"$l={l}$" for l in range(1, L + 1)])
        ax.set_ylabel("neuron (sorted by dominant $l$)")
        ax.set_yticks([])
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    axes[1].set_ylabel("")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("energy share", fontsize=7)
    counts = [int((lab1 == l).sum()) for l in range(1, L + 1)]
    ls = np.arange(1, L + 1)
    d = 2 * ls + 1
    ax = axes[2]
    ax.bar(ls, counts, width=0.6, color=BAR, zorder=2)
    ax.scatter(ls, d**3, marker="_", s=300, color=MARK, zorder=3, label="$d_l^3$")
    ax.scatter(ls, d**3 + d, marker="_", s=300, color=INK_MUTED, zorder=3, label="$d_l^3+d_l$")
    for l, c in zip(ls, counts):
        ax.annotate(str(c), (l, c), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=7, color=INK)
    ax.set_ylim(0, 820)
    ax.set_xticks(ls, [f"$l={l}$" for l in ls])
    ax.set_ylabel("active neurons")
    ax.set_title("(c) neurons per irrep", fontsize=8, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(json.dumps({"counts": counts, "n_active": int(len(act))}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("init_run_dir")
    p.add_argument("run_dir")
    p.add_argument("--out", default="litnotes/papier/so3/figures/purity")
    a = p.parse_args()
    plot(Path(a.init_run_dir), Path(a.run_dir), Path(a.out))


if __name__ == "__main__":
    main()
