import argparse,time,json
from pathlib import Path
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--onnx',required=True); ap.add_argument('--tokenizer',required=True); ap.add_argument('--repeats',type=int,default=1000); ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--output',default='results/raw/onnx_latency.json'); args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.tokenizer); text='Explain how SPARCS evaluates an input request.'; x=tok([text],return_tensors='np'); s=ort.InferenceSession(args.onnx,providers=['CPUExecutionProvider']); feed={'input_ids':x['input_ids'],'attention_mask':x['attention_mask']}
    for _ in range(args.warmup): s.run(None,feed)
    ts=[]
    for _ in range(args.repeats):
        t=time.perf_counter(); s.run(None,feed); ts.append((time.perf_counter()-t)*1000)
    out={'provider':s.get_providers()[0],'warmup':args.warmup,'repeats':args.repeats,'sequence_length':int(x['input_ids'].shape[1]),'p50_ms':float(np.percentile(ts,50)),'p95_ms':float(np.percentile(ts,95)),'p99_ms':float(np.percentile(ts,99)),'mean_ms':float(np.mean(ts))}; p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
