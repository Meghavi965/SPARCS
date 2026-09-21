import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from ..models import SPDMD

class ONNXWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self,input_ids,attention_mask):
        o=self.model(input_ids,attention_mask)
        return o.embedding, o.logits

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",default="artifacts/onnx/spdmd_fp32.onnx"); ap.add_argument("--model-name",default="microsoft/deberta-v3-base"); args=ap.parse_args()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    model=SPDMD(args.model_name,2).eval(); state=torch.load(args.checkpoint,map_location="cpu"); model.load_state_dict(state.get("model",state))
    tok=AutoTokenizer.from_pretrained(str(Path(args.checkpoint).parent))
    x=tok(["SPARCS export verification"],return_tensors="pt")
    torch.onnx.export(ONNXWrapper(model),(x["input_ids"],x["attention_mask"]),str(out),input_names=["input_ids","attention_mask"],output_names=["embedding","logits"],dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"attention_mask":{0:"batch",1:"sequence"},"embedding":{0:"batch"},"logits":{0:"batch"}},opset_version=17, dynamo=False)
    json.dump({"checkpoint":args.checkpoint,"onnx":str(out),"opset":17},open(out.with_suffix(".json"),"w"),indent=2)
if __name__=="__main__": main()
