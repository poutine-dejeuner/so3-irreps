# Reproduce the paper. PYTHON can point to any environment with the dependencies.
PYTHON ?= python
P ?= 2                      # parallel local runs
RUN = PYTHON=$(PYTHON) run/run_local.sh
PY = PYTHONPATH=src:. $(PYTHON)
L4_RUN = results/onehot_long/models/L4_square_sgd_lr0.5_wd0.0001_init0.1_h4096_s42

.PHONY: test checks figures paper
test:
	$(PY) -m pytest -q

# --- results that need no training (seconds to minutes) -------------------------------
checks:
	$(PY) scripts/degree_filtration.py                 # Proposition: degree filtration
	$(PY) scripts/capacity_check.py --k 3 --n-probes 1 # availability bound, numerics
	$(PY) scripts/min_norm_decomposition.py --d 3 5 7 9 --lam 3e-3   # R8e: d^3 units

paper/figures/delta_kernel.pdf:
	$(PY) scripts/plot_delta_kernel.py --out paper/figures/delta_kernel

# --- training (one target per config in configs/) -------------------------------------
results/gh_target:
	$(RUN) scripts/train_so3_mechanism.py configs/gh_target.txt $@ $(P)
	$(PY) scripts/analyze_so3_mechanism.py $@/models/*
results/%:
	$(RUN) scripts/train_so3_onehot.py configs/$*.txt $@ $(P)
	$(PY) scripts/analyze_so3_onehot.py $@/models/*

# --- figures from trained runs ------------------------------------------------------
paper/figures/purity.pdf: results/onehot_grid results/onehot_long
	$(PY) scripts/plot_so3_purity.py results/onehot_grid/models/$(notdir $(L4_RUN)) $(L4_RUN) --out paper/figures/purity
paper/figures/harmonics.pdf: results/onehot_long
	$(PY) scripts/plot_so3_harmonics.py $(L4_RUN) --out paper/figures/harmonics

figures: paper/figures/delta_kernel.pdf paper/figures/purity.pdf paper/figures/harmonics.pdf
paper: figures
	cd paper && latexmk -pdf draft.tex
