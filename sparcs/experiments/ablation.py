"""Component ablations with weight renormalization.

L1-L3 ablations isolate the corresponding inbound signal by renormalizing the
remaining gate weights. L5 is stateful output protection and is evaluated by
experiments/leakage.py, not by an inbound jailbreak dataset.
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import yaml
from ..guardrail import SPARCS
from ..benchmarks.evaluate import load_records

BASE_WEIGHTS = [0.30, 0.35, 0.25, 0.10]
ABLATIONS = {
    "full": None,
    "minus_l1": [0.0, 0.35, 0.25, 0.10],
    "minus_l2": [0.30, 0.0, 0.25, 0.10],
    "minus_l3": [0.30, 0.35, 0.0, 0.10],
}


def renormalize(weights):
    total=sum(float(x) for x in weights)
    if total <= 0: raise ValueError("Ablation removed all risk weight")
    return [float(x)/total for x in weights]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/calibrated.yaml'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--centroid',required=True); ap.add_argument('--dataset',required=True); ap.add_argument('--output',default='results/raw/ablation.json'); args=ap.parse_args()
    base=yaml.safe_load(Path(args.config).read_text()); records=load_records(args.dataset); result={}
    with tempfile.TemporaryDirectory() as td:
        for name, weights in ABLATIONS.items():
            cfg=yaml.safe_load(yaml.safe_dump(base))
            if weights is not None: cfg['risk']['weights']=renormalize(weights)
            p=Path(td)/f'{name}.yaml'; p.write_text(yaml.safe_dump(cfg,sort_keys=False)); g=SPARCS(p,args.checkpoint,args.centroid)
            blocks=[]; labels=[]
            for r in records:
                text=r.get('prompt',r.get('behavior',r.get('input',''))); rv=g.inspect(text)['risk']; blocks.append(rv.decision)
                if 'label' in r: labels.append(int(r['label']))
            row={'count':len(blocks),'block_rate':sum(blocks)/len(blocks),'weights':cfg['risk']['weights']}
            if labels:
                import numpy as np
                y=np.asarray(labels); b=np.asarray(blocks); row.update({'fpr':float(np.mean(b[y==0])) if np.any(y==0) else None,'fnr':float(np.mean(~b[y==1])) if np.any(y==1) else None,'accuracy':float(np.mean(b==y.astype(bool)))})
            result[name]=row
    result['_note']='L5 is not included in inbound ablation; use sparcs.experiments.leakage for stateful outbound canary detection.'
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
