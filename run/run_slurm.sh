#!/usr/bin/env bash
#SBATCH --job-name=so3irreps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# One array task per line of CONFIG. Submit from the repo root:
#   n=$(wc -l < CONFIG); sbatch --array=0-$((n-1)) --output=OUTDIR/logs/%A_%a.out \
#       run/run_slurm.sh SCRIPT CONFIG OUTDIR
set -euo pipefail
SCRIPT=$1; CONFIG=$2; OUT=$3
PY=${PYTHON:-python}
mkdir -p "$OUT/models" "$OUT/logs"
mapfile -t CONFIGS < "$CONFIG"
# shellcheck disable=SC2086
PYTHONPATH=src:. $PY "$SCRIPT" ${CONFIGS[$SLURM_ARRAY_TASK_ID]} --output-dir "$OUT/models"
