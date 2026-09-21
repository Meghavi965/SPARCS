import argparse, json, os, random
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from ..models import SPDMD


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows=[]
    with open(path,encoding="utf8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


class Collator:
    def __init__(self, tok, max_length): self.tok,self.max_length=tok,max_length
    def __call__(self,batch):
        texts=[x["text"] for x in batch]; y=torch.tensor([x["label"] for x in batch])
        e=self.tok(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
        e["labels"]=y; return e


def evaluate(model, loader, device):
    model.eval(); correct=n=0; losses=[]; ce=torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for b in loader:
            y=b.pop("labels").to(device); b={k:v.to(device, non_blocking=True) for k,v in b.items()}
            out=model(**b); loss=ce(out.logits,y); losses.append(float(loss)); correct += int((out.logits.argmax(-1)==y).sum()); n+=len(y)
    return {"loss":float(np.mean(losses)) if losses else float("nan"),"accuracy":correct/max(n,1)}


def rng_state():
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"]=torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    if not state: return
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state: torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, opt, sched, scaler, cfg, epoch, batch_in_epoch, step, best, tok, train_args):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "config":cfg,"epoch":epoch,"batch_in_epoch":batch_in_epoch,"global_step":step,
                "best_validation_accuracy":best,"rng_state":rng_state(),"train_args":train_args}, path)
    tok.save_pretrained(path.parent)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/default.yaml"); ap.add_argument("--train",default="data/processed/train.jsonl"); ap.add_argument("--validation",default="data/processed/validation.jsonl")
    ap.add_argument("--max_steps",type=int,default=-1); ap.add_argument("--eval_items",type=int,default=-1)
    ap.add_argument("--output",default=None); ap.add_argument("--resume",default=None,help="Checkpoint to resume; use auto to resume output/last.pt when present")
    ap.add_argument("--save_every_steps",type=int,default=None); ap.add_argument("--eval_every_epochs",type=int,default=None)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_all(cfg["training"]["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:",device)
    if device.type == "cuda": torch.set_float32_matmul_precision("high")
    out=Path(args.output or cfg["training"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    resume_path = (out/"last.pt") if args.resume == "auto" or (args.resume is None and (out/"last.pt").exists()) else (Path(args.resume) if args.resume else None)
    tok=AutoTokenizer.from_pretrained(cfg["model"]["name"]); model=SPDMD(cfg["model"]["name"],2).to(device)
    precision=str(cfg.get("runtime",{}).get("mixed_precision","none")).lower(); use_fp16=device.type=="cuda" and precision=="fp16"
    scaler=torch.cuda.amp.GradScaler(enabled=use_fp16) if device.type=="cuda" else None
    train=load_jsonl(args.train); val=load_jsonl(args.validation)
    if args.eval_items > 0: val=val[:args.eval_items]
    workers=int(cfg.get("runtime",{}).get("num_workers",2)); pin=device.type=="cuda"
    coll=Collator(tok,cfg["model"]["max_length"])
    # Use an explicit per-epoch generator so a mid-epoch checkpoint can recreate
    # the exact same shuffled batch order after restart.
    batch_size=int(cfg["training"]["batch_size"])
    eval_batch_size=int(cfg["training"]["eval_batch_size"])
    def make_train_loader(epoch):
        gen=torch.Generator()
        gen.manual_seed(int(cfg["training"]["seed"]) + int(epoch))
        return DataLoader(train,batch_size=batch_size,shuffle=True,generator=gen,collate_fn=coll,num_workers=workers,pin_memory=pin)
    va=DataLoader(val,batch_size=eval_batch_size,shuffle=False,collate_fn=coll,num_workers=workers,pin_memory=pin)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    steps_per_epoch=max(1,(len(train)+batch_size-1)//batch_size)
    steps_per_epoch=max(1,(steps_per_epoch+cfg["training"]["grad_accum"]-1)//cfg["training"]["grad_accum"])
    total=steps_per_epoch*cfg["training"]["epochs"]
    if args.max_steps>0: total=min(total,args.max_steps)
    sched=get_linear_schedule_with_warmup(opt,int(total*cfg["training"]["warmup_ratio"]),max(total,1))
    best=-1; step=0; start_epoch=0; start_batch=0
    if resume_path and resume_path.exists():
        print("resuming:",resume_path)
        state=torch.load(resume_path,map_location=device,weights_only=False); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"): scaler.load_state_dict(state["scaler"])
        best=float(state.get("best_validation_accuracy",-1)); step=int(state.get("global_step",0)); start_epoch=int(state.get("epoch",0)); start_batch=int(state.get("batch_in_epoch",0)); restore_rng(state.get("rng_state"))
        if step>=total: print("checkpoint already reached requested max steps"); return
    opt.zero_grad(set_to_none=True); ce=torch.nn.CrossEntropyLoss(); save_every=args.save_every_steps or int(cfg["training"].get("save_every_steps",250)); eval_every=args.eval_every_epochs or int(cfg["training"].get("eval_every_epochs",1))
    for epoch in range(start_epoch,cfg["training"]["epochs"]):
        model.train(); micro_since_update=0
        tr=make_train_loader(epoch)
        for i,b in enumerate(tr):
            if epoch==start_epoch and i<start_batch: continue
            y=b.pop("labels").to(device); b={k:v.to(device,non_blocking=True) for k,v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16): loss=ce(model(**b).logits,y)
            micro_since_update += 1
            (scaler.scale(loss) if use_fp16 else loss).div_(cfg["training"]["grad_accum"]).backward()
            boundary=(micro_since_update==cfg["training"]["grad_accum"])
            last_batch=(i+1==len(tr))
            if boundary or last_batch:
                if last_batch and not boundary:
                    for param in model.parameters():
                        if param.grad is not None: param.grad.mul_(cfg["training"]["grad_accum"]/micro_since_update)
                if use_fp16: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["max_grad_norm"])
                if use_fp16: scaler.step(opt); scaler.update()
                else: opt.step()
                sched.step(); opt.zero_grad(set_to_none=True); step+=1; micro_since_update=0
                if step%100==0: print("step",step,"loss",float(loss))
                if save_every>0 and step%save_every==0:
                    save_checkpoint(out/"last.pt",model,opt,sched,scaler,cfg,epoch,i+1,step,best,tok,vars(args)); print("checkpoint saved",step)
                if args.max_steps>0 and step>=args.max_steps: break
        start_batch=0
        if (epoch+1)%eval_every==0 or (args.max_steps>0 and step>=args.max_steps):
            metrics=evaluate(model,va,device); print("epoch",epoch+1,metrics)
            if metrics["accuracy"]>best:
                best=metrics["accuracy"]; torch.save({"model":model.state_dict(),"config":cfg,"validation":metrics,"global_step":step},out/"model.pt"); tok.save_pretrained(out); json.dump({"best_validation_accuracy":best,"steps":step},open(out/"training_manifest.json","w"),indent=2)
        save_checkpoint(out/"last.pt",model,opt,sched,scaler,cfg,epoch+1,0,step,best,tok,vars(args))
        if args.max_steps>0 and step>=args.max_steps: break
if __name__=="__main__": main()
