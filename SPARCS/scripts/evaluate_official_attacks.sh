#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${SPARCS_CHECKPOINT:?Set SPARCS_CHECKPOINT}"
: "${SPARCS_CENTROID:?Set SPARCS_CENTROID}"
CONFIG=${CONFIG:-configs/calibrated.yaml}
mkdir -p results/raw
python -m sparcs.benchmarks.evaluate --config "$CONFIG" --checkpoint "$SPARCS_CHECKPOINT" --centroid "$SPARCS_CENTROID" --dataset sparcs/benchmarks/datasets/harmbench_subset.json --assume-malicious --name harmbench --output results/raw/harmbench.json
python -m sparcs.benchmarks.evaluate --config "$CONFIG" --checkpoint "$SPARCS_CHECKPOINT" --centroid "$SPARCS_CENTROID" --dataset sparcs/benchmarks/datasets/GCG_subset.json --assume-malicious --name gcg_whitebox_vicuna13b --output results/raw/gcg.json
python -m sparcs.benchmarks.evaluate --config "$CONFIG" --checkpoint "$SPARCS_CHECKPOINT" --centroid "$SPARCS_CENTROID" --dataset sparcs/benchmarks/datasets/pair_subset.json --assume-malicious --name pair_blackbox_vicuna13b --output results/raw/pair.json
