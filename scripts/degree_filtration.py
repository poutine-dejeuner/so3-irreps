"""degree_filtration.py — Fit D^l(G) by polynomials of degree ≤ k in the entries of G.

Proposition (degree filtration): the entries of D^l(G) are polynomials of degree exactly l
in the entries of G; lower-degree polynomials are orthogonal to them. Prints the held-out
R² for l = 1..4 and k = 0..4.

Usage:
    python scripts/degree_filtration.py
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from so3irreps.wigner import so3_random, wigner_D_real


def _monomials(g: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones(len(g))]
    for d in range(1, degree + 1):
        for c in itertools.combinations_with_replacement(range(9), d):
            cols.append(np.prod(g[:, c], axis=1))
    return np.stack(cols, 1)


def degree_fit_r2(l: int, max_degree: int, n: int = 6000, seed: int = 5) -> dict[int, float]:
    G, *_ = so3_random(n, seed=seed)
    g = G.reshape(n, 9)
    D = wigner_D_real(l, G).reshape(n, -1)
    tr = int(2 * n / 3)
    out = {}
    for k in range(max_degree + 1):
        X = _monomials(g, k)
        c, *_ = np.linalg.lstsq(X[:tr], D[:tr], rcond=None)
        R = D[tr:] - X[tr:] @ c
        out[k] = float(1 - (R**2).sum() / ((D[tr:] - D[tr:].mean(0)) ** 2).sum())
    return out


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--max-l", type=int, default=4)
    args = a.parse_args()
    print("R² of D^l(G) by polynomials of degree ≤ k in the entries of G")
    print("      " + "  ".join(f"k={k}" for k in range(args.max_l + 1)))
    for l in range(1, args.max_l + 1):
        r2 = degree_fit_r2(l, args.max_l)
        print(f"l={l}: " + "  ".join(f"{r2[k]:5.3f}" for k in range(args.max_l + 1)))


if __name__ == "__main__":
    main()
