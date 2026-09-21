"""Reproducible PyTorch middleware latency benchmark."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from ..guardrail import SPARCS


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--centroid',required=True); ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--repeats',type=int,default=1000); ap.add_argument('--text',default='Explain how SPARCS evaluates an input request.'); ap.add_argument('--output',default='results/raw/latency.json'); args=ap.parse_args()
    g=SPARCS(args.config,args.checkpoint,centroid=args.centroid)
    for _ in range(args.warmup): g.inspect(args.text)
    if g.device.type=='cuda': torch.cuda.synchronize()
    ts=[]
    for _ in range(args.repeats):
        t=time.perf_counter(); g.inspect(args.text)
        if g.device.type=='cuda': torch.cuda.synchronize()
        ts.append((time.perf_counter()-t)*1000)
    out={'device':str(g.device),'warmup':args.warmup,'repeats':args.repeats,'sequence_length':int(g.tokenizer(args.text,return_tensors='pt')['input_ids'].shape[1]),'p50_ms':float(np.percentile(ts,50)),'p95_ms':float(np.percentile(ts,95)),'p99_ms':float(np.percentile(ts,99)),'mean_ms':float(np.mean(ts))}
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
