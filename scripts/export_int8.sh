#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CKPT=${1:?Usage: $0 CHECKPOINT}
python -m sparcs.export.onnx_export --checkpoint "$CKPT" --output artifacts/onnx/spdmd_fp32.onnx
python -m sparcs.export.quantize artifacts/onnx/spdmd_fp32.onnx artifacts/onnx/spdmd_int8.onnx
