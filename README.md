# so3-irreps

Code to reproduce the experiments and figures of the draft
*The hidden representations and interpretation of SO(3) group operation in neural
networks* (`paper/draft.tex`).

One-hidden-layer networks with a quadratic activation learn SO(3) composition
(G, H) ↦ GH, with inputs and targets encoded by rotation matrices, orbits of a fixed
tensor, or the **truncated delta** δ^L(G) = ⊕_{l≤L} √(2l+1) vec D^l(G) (Fourier basis) and
its samples at anchor rotations (pixel basis). The analyses read each hidden unit's irrep
content off its weights.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
make test                                  # 83 tests
```

## Paper ↔ code

| paper | command | notes |
|---|---|---|
| Fig. 1 (truncated delta, Pinsky) | `make paper/figures/delta_kernel.pdf` | no training |
| Prop. degree filtration (table) | `make checks` (first line) | no training |
| Availability bound, numerics | `make checks` (second line) | least-squares fit, no training |
| R8e: minimum-norm count d³ | `make checks` (third line) | ~1 h on a GPU for d = 9 |
| Sec. 6 (target GH, Table 1) | `make results/gh_target` | 5 runs, `configs/gh_target.txt` |
| Sec. 7, Table 2, R6–R8 (1500 epochs) | `make results/onehot_grid` | 11 runs |
| R8: converged counts (21 500 epochs) | `make results/onehot_long` | 2 runs, long |
| R8c: seeds | `make results/onehot_seeds` | 15 runs × 5000 epochs |
| R8b: pixel basis | `make results/pixel` | 34 runs (N = 2⁸…2¹², L = 2–4) |
| R8d: larger L | `make results/large_L` | 12 runs, widths up to 32 768 |
| R9b: norm control (wd = 0, lazy) | `make results/norm_control` | 7 runs |
| R10: learning order | `make results/vanishing` | 3 runs, per-neuron scaled SGD |
| Fig. 2 (purity), Fig. 3 (harmonics) | `make figures` | need `onehot_grid`, `onehot_long` |
| numbers in the text | `python scripts/summarize.py results/<exp> --markdown` | per-run table |

Each line of a `configs/*.txt` file holds the exact arguments of one run reported in the
paper (generated from the original runs' `config.json`); runs that diverged or did not
leave the initial plateau are included, as in the text. Runs are deterministic given the
seed, up to GPU nondeterminism.

`make results/<exp> P=4` runs four configurations in parallel on one machine. On a SLURM
cluster:

```bash
n=$(wc -l < configs/onehot_seeds.txt)
sbatch --array=0-$((n-1)) --output=results/onehot_seeds/logs/%A_%a.out \
    run/run_slurm.sh scripts/train_so3_onehot.py configs/onehot_seeds.txt results/onehot_seeds
```

## Layout

```
src/so3irreps/   wigner.py (real Wigner matrices, Haar sampling), data/ (orbit and
                 truncated-delta datamodules, pixel basis), model.py, learner.py,
                 optimizers.py (per-neuron scaled SGD), snapshots.py,
                 analysis/spectrum.py (fit-free irrep spectra, purity, counts,
                 matmul alignment, product units), analysis/mechanism.py (Sec. 6)
scripts/         training, analysis, figures, checks
configs/         one file per experiment
run/             local and SLURM runners
paper/           draft and figures
```

## Notes

- Losses are reported per Fourier coordinate; in the pixel basis the initialisation and
  the loss are rescaled (see the draft, *Normalisation*).
- Lightning may need `so3irreps.patch_torchvision` imported first on some torch builds;
  the scripts do this (`# isort: split` keeps the order).
