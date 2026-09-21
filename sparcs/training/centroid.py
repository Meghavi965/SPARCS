import argparse, json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..models import SPDMD
from .train import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", default="data/processed/train.jsonl")
    ap.add_argument("--output", default="artifacts/policy_centroid.pt")
    ap.add_argument("--max-items", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    model_name = state.get("config", {}).get("model", {}).get("name", "microsoft/deberta-v3-base")
    max_length = int(state.get("config", {}).get("model", {}).get("max_length", 512))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = SPDMD(model_name, 2).to(device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(ck.parent))
    rows = [r for r in load_jsonl(args.source) if int(r["label"]) == 0]
    if args.max_items > 0:
        rows = rows[:args.max_items]
    if not rows:
        raise ValueError("No benign training examples available for centroid construction")

    chunks = []
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch_size):
            e = tok(
                [r["text"] for r in rows[i : i + args.batch_size]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            e = {k: v.to(device, non_blocking=True) for k, v in e.items()}
            chunks.append(model(**e).embedding.float().cpu())
    c = torch.cat(chunks).mean(0)
    c = c / c.norm().clamp(min=1e-8)
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(c.unsqueeze(0), p)
    p.with_suffix(".json").write_text(json.dumps({
        "n": len(rows),
        "dimension": int(c.numel()),
        "source": args.source,
        "checkpoint": str(ck),
        "model_name": model_name,
        "max_length": max_length,
    }, indent=2))


if __name__ == "__main__":
    main()
