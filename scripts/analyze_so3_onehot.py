"""analyze_so3_onehot.py — Irrep specialisation in band-limited one-hot SO(3) nets.

For a run directory written by `train_so3_onehot.py`, on fresh Haar samples:
  - test relative MSE per output irrep block, and geodesic error (D^1 block)
  - per-neuron irrep spectra of the G-weights, H-weights and readout (fit-free:
    block norms), purity, G/H/readout label agreement, census, within-irrep
    matmul-tensor alignment — for the trained net, at init, and for a null that
    re-pairs G-weights, H-weights and readouts of different neurons
  - causal test: remove all neurons with a given label, per-block error
  - dynamics: per-block error, per-block weight energy and census at each snapshot
Writes onehot_analysis.json in the run directory.

Usage:
    python scripts/analyze_so3_onehot.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import so3irreps.patch_torchvision  # noqa: F401  # must precede Lightning
from so3irreps.analysis.spectrum import block_energies, neuron_report
from so3irreps.wigner import so3_random
from so3irreps.model import SO3CompositionNet
from so3irreps.data.so3_comp import _svd_project_to_so3
from so3irreps.data.so3_onehot import (
    anchor_matrix,
    bandlimited_onehot,
    block_slices,
    position_encoding,
)


def _net(config: dict, state: dict) -> SO3CompositionNet:
    net = SO3CompositionNet(
        input_dim=config["input_dim"], output_dim=config["output_dim"],
        hidden_size=config["hidden_size"], num_layers=0,
        activation=config["activation"], layernorm=config["layernorm"],
        bias=not config.get("no_bias", False),
    )
    net.load_state_dict(state)
    return net.eval().double()


class _Basis:
    """Encoding of a run (fourier or position) and the maps back to δ^L coordinates."""

    def __init__(self, config: dict):
        self.L = config["L"]
        self.position = config.get("basis", "fourier") == "position"
        if self.position:
            self.phi = anchor_matrix(self.L, config["n_anchors"], config["anchor_seed"])
            self.pinv = np.linalg.pinv(self.phi)

    def encode(self, R: np.ndarray) -> np.ndarray:
        if self.position:
            return position_encoding(R, self.L, self.phi)
        return bandlimited_onehot(R, self.L)

    def out_to_fourier(self, out: np.ndarray) -> np.ndarray:
        return (out @ self.pinv.T)[:, 1:] if self.position else out

    def weights(self, net: SO3CompositionNet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-layer G/H weights and readout, in δ^L coordinates (l ≥ 1 blocks)."""
        W1 = net.net[0].weight.detach().numpy()
        half = W1.shape[1] // 2
        W_G, W_H, W_out = W1[:, :half], W1[:, half:], net.net[-1].weight.detach().numpy()
        if self.position:
            # x = Φ δ̃: first layer acts as W Φ; readout r writes Φ⁺ r in δ̃ coordinates
            W_G, W_H = (W_G @ self.phi)[:, 1:], (W_H @ self.phi)[:, 1:]
            W_out = (self.pinv @ W_out)[1:]
        return W_G, W_H, W_out

    def frac_in_range(self, net: SO3CompositionNet) -> float:
        """Share of ‖W1‖² acting on range(Φ), the only directions the data occupies."""
        if not self.position:
            return 1.0
        W1 = net.net[0].weight.detach().numpy()
        half = W1.shape[1] // 2
        P = self.phi @ self.pinv
        inside = sum(np.linalg.norm(W @ P) ** 2 for W in (W1[:, :half], W1[:, half:]))
        return float(inside / np.linalg.norm(W1) ** 2)


def _errors(net, data, L, basis: _Basis, keep: np.ndarray | None = None) -> dict:
    """Per-block relative MSE (and geodesic error) on (x(G), x(H), δ(GH), GH),
    with the network output mapped to δ^L coordinates."""
    xg, xh, y, _ = data
    with torch.no_grad():
        if keep is None:
            out = net(xg, xh).numpy()
        else:
            h = net.net[:-1](torch.cat([xg, xh], -1))
            W2, b2 = net.net[-1].weight, net.net[-1].bias
            out = h[:, keep] @ W2[:, keep].T
            out = (out + b2 if b2 is not None else out).numpy()
    out = basis.out_to_fourier(out)
    yn = y.numpy()
    rel = {str(l): float(((out[:, s] - yn[:, s]) ** 2).sum() / (yn[:, s] ** 2).sum())
           for l, s in block_slices(L).items()}
    d1 = block_slices(L)[1]
    pred = _svd_project_to_so3(torch.from_numpy(out[:, d1] / np.sqrt(3))).numpy()
    true = yn[:, d1].reshape(-1, 3, 3) / np.sqrt(3)
    tr = np.einsum("nij,nij->n", pred, true)
    geo = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).mean())
    return {"rel_mse": rel, "geodesic_deg": geo}


def _compact(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if k not in ("active_index", "joint_label")}


def analyze_run(run_dir: Path, n_samples: int = 5000) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    L = config["L"]
    G, *_ = so3_random(n_samples, seed=7001)
    H, *_ = so3_random(n_samples, seed=7002)

    def t(a):
        return torch.from_numpy(a)

    basis = _Basis(config)
    data = (t(basis.encode(G)), t(basis.encode(H)), t(bandlimited_onehot(G @ H, L)), G @ H)

    def load(name):
        return torch.load(name, map_location="cpu", weights_only=True)

    net = _net(config, load(run_dir / "model.pt"))
    net0 = _net(config, load(run_dir / "init_model.pt"))
    err = _errors(net, data, L, basis)
    report: dict = {"run": run_dir.name, "config": config,
                    "test_rel_mse": err["rel_mse"], "test_geodesic_deg": err["geodesic_deg"],
                    "frac_first_layer_norm_in_range_phi": basis.frac_in_range(net)}

    W_G, W_H, W_out = basis.weights(net)
    rng = np.random.default_rng(0)
    n = W_G.shape[0]
    final = neuron_report(W_G, W_H, W_out, L)
    report["neurons"] = {
        "final": _compact(final),
        "init": _compact(neuron_report(*basis.weights(net0), L)),
        "shuffled": _compact(neuron_report(
            W_G, W_H[rng.permutation(n)], W_out[:, rng.permutation(n)], L)),
    }

    # Causal test: drop the active neurons with joint label l
    act, lab = np.array(final["active_index"]), np.array(final["joint_label"])
    report["ablate_by_label"] = {}
    for l in sorted(set(lab.tolist())):
        keep = np.setdiff1d(np.arange(n), act[lab == l])
        report["ablate_by_label"][str(l)] = _errors(net, data, L, basis, keep)["rel_mse"]

    # Dynamics over snapshots (small eval set)
    small = tuple(x[:1000] for x in data[:3]) + (data[3][:1000],)
    blocks = block_slices(L)
    report["dynamics"] = []
    for path in sorted((run_dir / "snapshots").glob("epoch_*.pt")):
        snet = _net(config, load(path))
        sG, sH, sR = basis.weights(snet)
        rep = neuron_report(sG, sH, sR, L)
        report["dynamics"].append({
            "epoch": int(path.stem.split("_")[1]),
            "rel_mse": _errors(snet, small, L, basis)["rel_mse"],
            "energy_G": block_energies(sG, blocks).sum(0).tolist(),
            "energy_R": block_energies(sR.T, blocks).sum(0).tolist(),
            "census": rep["census"], "n_active": rep["n_active"],
            "frac_all_three_agree": rep["frac_all_three_agree"],
        })

    (run_dir / "onehot_analysis.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--n-samples", type=int, default=5000)
    args = p.parse_args()
    for rd in args.run_dirs:
        rep = analyze_run(Path(rd), args.n_samples)
        print(json.dumps({k: rep[k] for k in ("run", "test_rel_mse", "test_geodesic_deg")}))


if __name__ == "__main__":
    main()
