"""Empirical latent perturbation test for the scoped certification claim.

Perturbs the pooled latent representation directly. This is intentionally not
presented as an end-to-end input-space certificate.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from ..models import SPDMD

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--samples',type=int,default=1000); ap.add_argument('--eps',type=float,default=0.05); ap.add_argument('--output',default='results/raw/lipschitz.json'); args=ap.parse_args()
    st=torch.load(args.checkpoint,map_location='cpu',weights_only=False); cfg=st.get('config',{}); name=cfg.get('model',{}).get('name','microsoft/deberta-v3-base'); m=SPDMD(name,2).eval(); m.load_state_dict(st.get('model',st),strict=True)
    # Spectral norm of the constrained latent->logit map is <= 1 by construction;
    # measure empirical probability change under bounded latent perturbations.
    w=m.classifier.weight.detach().float(); spectral=float(torch.linalg.matrix_norm(w,2))
    rng=torch.Generator().manual_seed(42); x=torch.randn(args.samples,m.embedding_dim,generator=rng); x=x/x.norm(dim=1,keepdim=True)
    delta=torch.randn(x.shape,generator=rng); delta=delta/delta.norm(dim=1,keepdim=True)*args.eps; x2=x+delta
    with torch.inference_mode(): p=torch.softmax(m.classifier(x),-1)[:,1]; q=torch.softmax(m.classifier(x2),-1)[:,1]
    empirical=float((q-p).abs().max()); result={'samples':args.samples,'epsilon_latent_l2':args.eps,'classifier_spectral_norm':spectral,'max_probability_change':empirical,'scope':'continuous latent perturbations; invariant discrete structure'}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
