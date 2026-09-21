#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CONFIG=${CONFIG:-configs/default.yaml}
TRAIN=${TRAIN:-data/processed/train.jsonl}
VAL=${VAL:-data/processed/validation.jsonl}
OUT=${OUT:-artifacts/checkpoint}
python -m sparcs.training.train --config "$CONFIG" --train "$TRAIN" --validation "$VAL" --output "$OUT" --resume auto
[[ -f "$OUT/model.pt" ]] || { echo "No trained model at $OUT/model.pt"; exit 2; }
[[ -f artifacts/policy_centroid.pt ]] || python -m sparcs.training.centroid --checkpoint "$OUT/model.pt" --source "$TRAIN" --output artifacts/policy_centroid.pt
[[ -f artifacts/calibration.json && -f artifacts/calibrated.yaml ]] || python -m sparcs.training.calibrate --config "$CONFIG" --checkpoint "$OUT/model.pt" --centroid artifacts/policy_centroid.pt --validation "$VAL" --batch-size 32 --output artifacts/calibration.json
