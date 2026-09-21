"""Deterministic corpus preparation. Expects persisted HF datasets at data/raw/."""
import argparse, hashlib, json, re, random
from pathlib import Path
from collections import defaultdict

def norm(s): return re.sub(r"\s+"," ",s.strip().lower())
def sha(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def load_json(path): return json.loads(Path(path).read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--wildguardmix",default="data/raw/wildguardmix"); ap.add_argument("--wildjailbreak",default="data/raw/wildjailbreak"); ap.add_argument("--out",default="data/processed"); ap.add_argument("--benchmark-dir",default="sparcs/benchmarks/datasets"); args=ap.parse_args()
    from datasets import load_from_disk
    wg=load_from_disk(args.wildguardmix)["train"]; wj=load_from_disk(args.wildjailbreak)["train"]
    rows=[]
    for r in wg:
        y=1 if r["prompt_harm_label"]=="harmful" else 0 if r["prompt_harm_label"]=="unharmful" else None
        if y is not None: rows.append({"text":r["prompt"],"label":y,"source":"wildguardmix","category":r.get("subcategory")})
    for r in wj:
        dt=r["data_type"]; y=1 if dt.endswith("harmful") else 0 if dt.endswith("benign") else None
        if y is not None: rows.append({"text":r["adversarial"] if dt.startswith("adversarial") else r["vanilla"],"label":y,"source":"wildjailbreak","category":dt,"lineage":norm(r["vanilla"])})
    keys=defaultdict(list)
    for r in rows: keys[norm(r["text"])].append(r)
    rows=[v[0] for v in keys.values() if len({x["label"] for x in v})==1]
    # Exclude exact benchmark-text contamination when the official benchmark artifacts are present.
    benchmark_keys=set()
    bdir=Path(args.benchmark_dir)
    for bp in sorted(bdir.glob("*.json")) if bdir.exists() else []:
        try:
            obj=json.loads(bp.read_text()); rec=obj.get("records",obj.get("jailbreaks",obj))
            for r in rec:
                t=r.get("prompt",r.get("behavior",r.get("input","")))
                if t: benchmark_keys.add(norm(t))
        except Exception:
            continue
    if benchmark_keys:
        rows=[r for r in rows if norm(r["text"]) not in benchmark_keys]
    # lineage-aware grouping for WJ; standalone WG rows are their own groups.
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r.get("lineage",f"row-{i}")].append(r)
    groups=list(groups.values()); random.Random(42).shuffle(groups)
    cut=int(len(groups)*0.8); train=[x for g in groups[:cut] for x in g]; val=[x for g in groups[cut:] for x in g]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for name,data in [("train",train),("validation",val)]:
        p=out/f"{name}.jsonl"; p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in data)); print(name,len(data),sha(p))
    source_path=out/"training_source.jsonl"; source_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    manifest={"seed":42,"rows":len(rows),"train_rows":len(train),"validation_rows":len(val),"benchmark_exact_keys_excluded":len(benchmark_keys),"sources":{},"artifacts":{}}
    for r in rows: manifest["sources"][r["source"]]=manifest["sources"].get(r["source"],0)+1
    for name in ("train.jsonl","validation.jsonl","training_source.jsonl"):
        manifest["artifacts"][name]=sha(out/name)
    (out.parent/"manifests").mkdir(parents=True,exist_ok=True)
    (out.parent/"manifests"/"train_corpus.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
