#!/usr/bin/env bash
# Run every line of a config file with a training script, P runs in parallel.
#   run/run_local.sh SCRIPT CONFIG OUTDIR [P]
# e.g. run/run_local.sh scripts/train_so3_onehot.py configs/onehot_grid.txt results/onehot_grid 4
set -euo pipefail
SCRIPT=$1; CONFIG=$2; OUT=$3; P=${4:-2}
PY=${PYTHON:-python}
mkdir -p "$OUT/logs"
nl -ba -w1 -s' ' "$CONFIG" | xargs -P "$P" -L1 bash -c '
    i=$0
    PYTHONPATH=src:. '"$PY"' '"$SCRIPT"' "$@" --output-dir '"$OUT"'/models \
        > '"$OUT"'/logs/run_$i.out 2> '"$OUT"'/logs/run_$i.err
    echo "run $i exit $?"'
