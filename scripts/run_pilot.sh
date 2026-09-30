#!/usr/bin/env bash
# Pilot: balanced subset -> E0, B1..B5, A-best -> summary.  Restartable (finished runs are skipped).
#   nohup bash scripts/run_pilot.sh > pilot.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=${RAW:-data/raw}                      # downloaded Google Drive data
DATA=${DATA:-data/processed}              # YOLO-seg tree + index.csv
SPLIT=${SPLIT:-$DATA/splits/pilot}
PROJECT=${PROJECT:-runs/pilot}
TRAIN_IMAGES=${TRAIN_IMAGES:-6000}
VAL_IMAGES=${VAL_IMAGES:-1500}
DEVICE=${DEVICE:-0}
EXTRA=${EXTRA:-}                          # e.g. EXTRA="epochs=40 batch=32"
OPTIONAL=${OPTIONAL:-0}                   # 1 = also run B2b (DINOv2-B) and B5b (C-RADIOv4-SO400M)

[ -f "$DATA/index.csv" ] || python tools/prepare_dataset.py --src "$RAW" --out "$DATA"
[ -f "$SPLIT/data.yaml" ] || python tools/make_subset.py --data "$DATA" --out "$SPLIT" \
    --train-images "$TRAIN_IMAGES" --val-images "$VAL_IMAGES"

TEACHERS="siglip2_b dinov3_b radio_v2.5_b cradio_v3_b"
[ "$OPTIONAL" = 1 ] && TEACHERS="$TEACHERS dinov2_b cradio_v4_so400m"
python tools/check_teachers.py $TEACHERS

ARGS=(--data "$SPLIT/data.yaml" --project "$PROJECT" --device "$DEVICE")
[ "$OPTIONAL" = 1 ] && ARGS+=(--with-optional)
[ -n "$EXTRA" ] && ARGS+=(--set $EXTRA)
python tools/run_experiments.py "${ARGS[@]}"
