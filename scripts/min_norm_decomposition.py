"""min_norm_decomposition.py — Weight-decay-optimal rank-one decompositions of d×d matmul.

Minimises ‖Σ_j r_j ⊗ a_j ⊗ b_j − M_d‖² + λ Σ_j (‖r_j‖² + ‖a_j‖² + ‖b_j‖²) from many more
units than needed, with M_d the d×d matrix-multiplication tensor (the target block of
irrep l, d = 2l+1), and counts the units that survive. This is the weight-decay
objective of the bilinear part of a one-hidden-layer z² network, without the
non-bilinear (junk) terms a real network must also cancel.

Usage:
    python scripts/min_norm_decomposition.py --d 3 5 7 --lam 3e-3
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def matmul_tensor_torch(d: int) -> torch.Tensor:
    M = torch.zeros(d * d, d * d, d * d)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                M[d * i + j, d * i + k, d * k + j] = 1.0
    return M


def min_norm_decomposition(d: int, n_units: int, lam: float, iters: int = 20_000,
                           seed: int = 0, device: str = "cpu", thresh: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    D = d * d
    M = matmul_tensor_torch(d).to(device)
    R, A, B = (0.3 * torch.randn(n_units, D, device=device) for _ in range(3))
    params = [p.requires_grad_() for p in (R, A, B)]
    opt = torch.optim.Adam(params, lr=1e-2)
    for it in range(iters):
        T = torch.einsum("jo,jp,jq->opq", R, A, B)
        loss = ((T - M) ** 2).sum() + lam * sum((p**2).sum() for p in params)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it == int(0.75 * iters):
            for g in opt.param_groups:
                g["lr"] = 1e-3
    with torch.no_grad():
        T = torch.einsum("jo,jp,jq->opq", R, A, B)
        c = (R.norm(dim=1) * A.norm(dim=1) * B.norm(dim=1)).cpu().numpy()
        rel = float(((T - M) ** 2).sum() / (M**2).sum())
    c = c / c.max()
    alive = c > thresh
    return {"d": d, "lam": lam, "seed": seed, "n_units": n_units,
            "units_alive": int(alive.sum()), "rel_err": rel,
            "contribution_spread": float(c[alive].max() - c[alive].min()),
            "largest_dead": float(np.max(c[~alive], initial=0.0))}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--d", type=int, nargs="+", default=[3, 5, 7])
    a.add_argument("--lam", type=float, nargs="+", default=[3e-3])
    a.add_argument("--seeds", type=int, default=3)
    a.add_argument("--iters", type=int, default=20_000)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = a.parse_args()
    for d in args.d:
        for lam in args.lam:
            for s in range(args.seeds):
                out = min_norm_decomposition(d, n_units=3 * d**3, lam=lam, iters=args.iters,
                                             seed=s, device=args.device)
                print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
