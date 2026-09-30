#!/usr/bin/env bash
# End-to-end CPU check with synthetic data and a random "dummy" teacher (no downloads, ~3 min).
#   bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
T=${T:-/tmp/kdseg_smoke}
rm -rf "$T"
python tests/make_synthetic.py --out "$T/raw" --groups 30 --frames 20
python tools/prepare_dataset.py --src "$T/raw" --out "$T/data"
python tools/make_subset.py --data "$T/data" --out "$T/data/splits/pilot" --train-images 200 --val-images 60 \
    --min-train-per-class 20 --min-eval-per-class 5
python tools/run_experiments.py --data "$T/data/splits/pilot/data.yaml" --project "$T/runs" \
    --config tests/experiments_smoke.yaml --device cpu --eval-test
test -f "$T/runs/B1/weights/student.pt" && test -f "$T/runs/summary_val.md" && echo "SMOKE TEST PASSED"
