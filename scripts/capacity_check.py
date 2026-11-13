"""capacity_check.py — Best achievable fit of an orbit target by a quadratic layer.

Input u(G) = ρ(G)x (ρ = (R³)^{⊗k}, n_probes fixed probe vectors concatenated),
target y = ρ(GH)x₁. A one-hidden-layer z² network can realise any bilinear form in
(u(G), u(H)) plus a constant; the l-component of y that such forms can capture is at
most rank(x̂_l)/(2l+1) (Proposition "Capacity of a quadratic layer"). This script fits
the best bilinear-plus-constant function by least squares and reports the held-out
R² per output irrep.

Usage:
    python scripts/capacity_check.py --k 3 --n-probes 1
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from so3irreps.analysis.mechanism import encode, isotypic_projectors
from so3irreps.wigner import so3_random


def best_bilinear_r2(k: int = 3, n_probes: int = 1, n: int = 40_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    P = isotypic_projectors(k)
    G, *_ = so3_random(n, seed=seed + 11)
    H, *_ = so3_random(n, seed=seed + 12)
    xs = [v / np.linalg.norm(v) for v in rng.standard_normal((n_probes, 3**k))]
    XG = np.concatenate([encode(G, x, k) for x in xs], 1)
    XH = np.concatenate([encode(H, x, k) for x in xs], 1)
    Y = encode(G @ H, xs[0], k)
    F = np.concatenate([np.einsum("np,nq->npq", XG, XH).reshape(n, -1), np.ones((n, 1))], 1)
    tr, te = slice(0, int(0.75 * n)), slice(int(0.75 * n), None)
    coef, *_ = np.linalg.lstsq(F[tr], Y[tr], rcond=None)
    R = Y[te] - F[te] @ coef
    Yc = Y[te] - Y[tr].mean(0)
    return {l: float(1 - ((R @ p) ** 2).sum() / ((Yc @ p) ** 2).sum())
            for l, p in P.items() if l > 0}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--k", type=int, default=3)
    a.add_argument("--n-probes", type=int, default=1)
    a.add_argument("--n", type=int, default=40_000)
    args = a.parse_args()
    print(json.dumps(best_bilinear_r2(args.k, args.n_probes, args.n), indent=2))


if __name__ == "__main__":
    main()
