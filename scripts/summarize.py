"""summarize.py — One line per analysed run: the numbers reported in the paper.

Reads `onehot_analysis.json` (written by analyze_so3_onehot.py) under the given result
directories and prints, per run: per-block relative MSE, whether every block is fitted
(relative MSE < 0.1), minimum pure fraction over G/H/readout, label agreement, neuron
count per irrep, and the matrix-multiplication tensor cosine per irrep.

Usage:
    python scripts/summarize.py results/onehot_grid [results/onehot_seeds ...] [--markdown]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def summarize_run(path: Path) -> dict:
    a = json.loads(Path(path).read_text())
    c, n = a["config"], a["neurons"]["final"]
    L = c["L"]
    rel = [a["test_rel_mse"][str(l)] for l in range(1, L + 1)]
    tc = [n["within_irrep"].get(str(l), {}).get("tensor_cos_with_matmul", float("nan"))
          for l in range(1, L + 1)]
    return {
        "run": Path(path).parent.name, "L": L, "basis": c.get("basis", "fourier"),
        "optimizer": c["optimizer"], "lr": c["lr"], "wd": c["wd"], "init": c["init_scale"],
        "seed": c["seed"], "geodesic_deg": a["test_geodesic_deg"],
        "rel_mse": "/".join(f"{x:.0e}" for x in rel),
        "all_blocks_fitted": all(x < 0.1 for x in rel),
        "pure": min(n["frac_purity_gt_0.9"].values()),
        "agree": n["frac_all_three_agree"],
        "census": "/".join(str(n["census"].get(str(l), 0)) for l in range(1, L + 1)),
        "matmul_cos": "/".join(f"{x:.3f}" for x in tc),
    }


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("result_dirs", nargs="+")
    a.add_argument("--markdown", action="store_true")
    args = a.parse_args()
    rows = [summarize_run(Path(f)) for d in args.result_dirs
            for f in sorted(glob.glob(f"{d}/**/onehot_analysis.json", recursive=True))]
    cols = ["run", "geodesic_deg", "rel_mse", "all_blocks_fitted", "pure", "agree", "census",
            "matmul_cos"]
    if args.markdown:
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
    for r in rows:
        vals = [f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in cols]
        print(("| " + " | ".join(vals) + " |") if args.markdown else "  ".join(vals))


if __name__ == "__main__":
    main()
