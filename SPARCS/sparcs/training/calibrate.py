"""Calibrate the final risk gate on validation data only."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..guardrail import SPARCS
from .train import load_jsonl


def metrics(scores, labels, threshold):
    pred = scores >= threshold
    tp = int(np.sum((labels == 1) & pred)); tn = int(np.sum((labels == 0) & ~pred))
    fp = int(np.sum((labels == 0) & pred)); fn = int(np.sum((labels == 1) & ~pred))
    return {
        "threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fpr": float(fp / max(fp + tn, 1)), "fnr": float(fn / max(fn + tp, 1)),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "balanced_error": float(0.5 * (fp / max(fp + tn, 1) + fn / max(fn + tp, 1))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--centroid", required=True)
    ap.add_argument("--validation", default="data/processed/validation.jsonl")
    ap.add_argument("--output", default="artifacts/calibration.json")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    g = SPARCS(args.config, args.checkpoint, centroid=args.centroid)
    rows = load_jsonl(args.validation)
    scores, labels = [], []
    for i in range(0, len(rows), max(1, args.batch_size)):
        chunk = rows[i:i+max(1, args.batch_size)]
        outputs = g.inspect_batch([r["text"] for r in chunk], batch_size=args.batch_size)
        for r, z in zip(chunk, outputs):
            scores.append(z["risk"].score); labels.append(int(r["label"]))
    scores, labels = np.asarray(scores), np.asarray(labels)
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))
    best = min((metrics(scores, labels, t) for t in candidates), key=lambda x: x["balanced_error"])

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result = {**best, "n": int(len(labels)), "positive": int(labels.sum()), "negative": int((labels == 0).sum())}
    out.write_text(json.dumps(result, indent=2))
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["risk"]["threshold"] = best["threshold"]
    cfg["risk"]["centroid_path"] = str(args.centroid)
    out.with_name("calibrated.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
