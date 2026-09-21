#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${SPARCS_CHECKPOINT:?Set SPARCS_CHECKPOINT to a trained model.pt}"
: "${SPARCS_CENTROID:?Set SPARCS_CENTROID to a training-derived centroid}"
export SPARCS_CONFIG=${SPARCS_CONFIG:-configs/calibrated.yaml}
uvicorn 'sparcs.api:create_app' --factory --host 0.0.0.0 --port 8000
