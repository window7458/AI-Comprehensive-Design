#!/usr/bin/env bash
# Full run with the winner of the pilot: all train/val images of the same sequence split,
# evaluated on the held-out test sequences too.
#   EXPS="E0 B2" nohup bash scripts/run_full.sh > full.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data/processed}
SPLIT=${SPLIT:-$DATA/splits/full}
PROJECT=${PROJECT:-runs/full}
EXPS=${EXPS:?set EXPS, e.g. EXPS=\"E0 B2\" (baseline + best pilot run)}
DEVICE=${DEVICE:-0}
EXTRA=${EXTRA:-epochs=150 patience=40}

[ -f "$SPLIT/data.yaml" ] || python tools/make_subset.py --data "$DATA" --out "$SPLIT" \
    --train-images 0 --val-images 0 --test-images 0

python tools/run_experiments.py --data "$SPLIT/data.yaml" --project "$PROJECT" --device "$DEVICE" \
    --only $EXPS --eval-test --set $EXTRA
python tools/summarize.py --project "$PROJECT" --split test
