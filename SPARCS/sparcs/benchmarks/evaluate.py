"""Benchmark runner for malicious/benign prompt suites.

The runner reports observed firewall decisions. It never substitutes manuscript
headline numbers and records source metadata when present.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..guardrail import SPARCS


def load_records(path: str | Path):
    obj = json.loads(Path(path).read_text())
    return obj.get("records", obj.get("jailbreaks", obj))


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(values, q, method="linear"))


def bootstrap_ci(values, statistic, seed=42, n_boot=2000):
    values = np.asarray(values)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = np.asarray([statistic(values[i]) for i in idx])
    return [float(np.quantile(stats, .025)), float(np.quantile(stats, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--centroid", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default="results/raw/eval.json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--assume-malicious", action="store_true", help="Treat every record as a malicious attack attempt; reports middleware pass-through rate, not model ASR")
    args = ap.parse_args()

    g = SPARCS(args.config, args.checkpoint, centroid=args.centroid)
    obj = json.loads(Path(args.dataset).read_text())
    rec = obj.get("records", obj.get("jailbreaks", obj))
    rows, times = [], []
    for i, r in enumerate(rec):
        text = r.get("prompt", r.get("behavior", r.get("input", "")))
        if not text:
            raise ValueError(f"record {i} has no supported prompt field")
        t0 = time.perf_counter()
        z = g.inspect(text)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        rv = z["risk"]
        # Explicit dataset label, when supplied, is interpreted as 1=harmful.
        label = r.get("label")
        if label is None and "harmful" in r:
            label = int(bool(r["harmful"]))
        rows.append({
            "id": r.get("id", r.get("index", i)),
            "label": None if label is None else int(label),
            "decision_block": bool(rv.decision),
            "score": float(rv.score),
            "l1": float(rv.l1), "l2": float(rv.l2), "l3": float(rv.l3), "l4": float(rv.l4),
            "entities": int(rv.entities), "token_count": int(rv.token_count),
            "forward_calls": int(z["forward_calls"]), "latency_ms": elapsed,
        })

    labels = np.asarray([r["label"] for r in rows if r["label"] is not None], dtype=int)
    scores = np.asarray([r["score"] for r in rows if r["label"] is not None], dtype=float)
    decisions = np.asarray([r["decision_block"] for r in rows if r["label"] is not None], dtype=bool)
    metrics = {}
    if len(labels) and len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        metrics["fpr"] = float(np.mean(decisions[labels == 0]))
        metrics["fnr"] = float(np.mean(~decisions[labels == 1]))
        metrics["accuracy"] = float(np.mean(decisions == labels.astype(bool)))
        metrics["tpr_block"] = float(np.mean(decisions[labels == 1]))
    elif len(labels):
        metrics["block_rate"] = float(np.mean(decisions))

    malicious = [r["decision_block"] for r in rows if r["label"] == 1]
    if args.assume_malicious and not malicious:
        malicious = [r["decision_block"] for r in rows]
        pass_through = 1 - np.asarray(malicious, dtype=float)
        metrics["middleware_pass_through_rate"] = float(np.mean(pass_through))
        metrics["middleware_pass_through_rate_ci95"] = bootstrap_ci(pass_through, np.mean)
        metrics["metric_note"] = "middleware pass-through rate; downstream model response was not evaluated, so this is not attack success rate"
    elif malicious:
        asr = float(1 - np.mean(malicious))
        metrics["attack_success_rate"] = asr
        metrics["attack_success_rate_ci95"] = bootstrap_ci(
            1 - np.asarray(malicious, dtype=float), np.mean
        )
    benign = [r["decision_block"] for r in rows if r["label"] == 0]
    if benign:
        metrics["benign_false_positive_rate"] = float(np.mean(benign))

    result = {
        "name": args.name or Path(args.dataset).stem,
        "dataset": str(args.dataset),
        "dataset_metadata": {k: v for k, v in obj.items() if k not in {"records", "jailbreaks"}},
        "count": len(rows),
        "unique_record_ids": len({str(r["id"]) for r in rows}),
        "metrics": metrics,
        "latency_ms": {"p50": percentile(times, 50), "p95": percentile(times, 95), "p99": percentile(times, 99)},
        "rows": rows,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
