#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG=${CONFIG:-configs/default.yaml}
CKPT=artifacts/checkpoint/model.pt
CENTROID=artifacts/policy_centroid.pt
CAL=artifacts/calibration.json
FP32=artifacts/onnx/spdmd_fp32.onnx
INT8=artifacts/onnx/spdmd_int8.onnx

if [[ ! -f "$CKPT" ]]; then
  python -m sparcs.training.train --config "$CONFIG" --resume auto
fi
[[ -f "$CKPT" ]] || { echo "Training has not produced $CKPT; rerun to resume from last.pt."; exit 2; }

if [[ ! -f "$CENTROID" ]]; then
  python -m sparcs.training.centroid --checkpoint "$CKPT" --source data/processed/train.jsonl --output "$CENTROID"
fi
if [[ ! -f "$CAL" || ! -f artifacts/calibrated.yaml ]]; then
  python -m sparcs.training.calibrate --config "$CONFIG" --checkpoint "$CKPT" --centroid "$CENTROID" --batch-size 32
fi
if [[ ! -f "$FP32" ]]; then
  python -m sparcs.export.onnx_export --checkpoint "$CKPT" --output "$FP32"
fi
python -m sparcs.export.verify --checkpoint "$CKPT" --onnx "$FP32"
if [[ ! -f "$INT8" ]]; then
  python -m sparcs.export.quantize "$FP32" "$INT8"
fi
python -m sparcs.export.benchmark_onnx --onnx "$INT8" --tokenizer artifacts/checkpoint
