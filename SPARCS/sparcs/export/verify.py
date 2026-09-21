"""Check PyTorch/ONNX numerical agreement over multiple representative inputs."""
import argparse, json
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from transformers import AutoTokenizer
from ..models import SPDMD


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--onnx',required=True); ap.add_argument('--texts',nargs='+',default=['SPARCS numerical agreement test']); ap.add_argument('--output',default=None); args=ap.parse_args()
    ck=Path(args.checkpoint); st=torch.load(ck,map_location='cpu',weights_only=False); cfg=st.get('config',{}); name=cfg.get('model',{}).get('name','microsoft/deberta-v3-base'); max_len=int(cfg.get('model',{}).get('max_length',512));
    m=SPDMD(name,2).eval(); m.load_state_dict(st.get('model',st),strict=True); tok=AutoTokenizer.from_pretrained(str(ck.parent)); e=tok(args.texts,padding=True,truncation=True,max_length=max_len,return_tensors='np')
    with torch.inference_mode(): pt=m(torch.from_numpy(e['input_ids']),torch.from_numpy(e['attention_mask']))
    sess=ort.InferenceSession(args.onnx,providers=['CPUExecutionProvider']); oe=sess.run(None,{'input_ids':e['input_ids'],'attention_mask':e['attention_mask']})
    result={'texts':len(args.texts),'embedding_max_abs_error':float(np.max(np.abs(pt.embedding.numpy()-oe[0]))),'embedding_mean_abs_error':float(np.mean(np.abs(pt.embedding.numpy()-oe[0]))),'logits_max_abs_error':float(np.max(np.abs(pt.logits.numpy()-oe[1]))),'logits_mean_abs_error':float(np.mean(np.abs(pt.logits.numpy()-oe[1])))}
    p=Path(args.output or (str(args.onnx)+'.agreement.json')); p.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
