#!/usr/bin/env python3
"""Quality-focused trainer for the standalone HGAI v10 checkpoint.

Keeps runtime/checkpoint compatibility with hgai_core.py while improving
training stability: deterministic raw validation, curated curriculum sampling,
optimizer/RNG resume, and cached augmented_train.txt reuse.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
import torch
import hgai_core as h


def load_training_pairs(train_file: str):
    tf=Path(train_file)
    aug=tf.with_name("augmented_train.txt")
    if tf.name == "train.txt" and aug.exists():
        pairs=h.parse_pairs(aug.read_text(encoding="utf-8"))
        if pairs:
            return pairs
    return h.training_pairs(tf)


def make_batch(tok,cfg,pairs,device,rng,ctx_prob=.30,focus_pairs=None,focus_prob=0.0):
    xs=[]
    for _ in range(cfg["batch_size"]):
        src=focus_pairs if focus_pairs and rng.random() < focus_prob else pairs
        pair=src[rng.randrange(len(src))]
        xs.append(h.example(tok,cfg,*pair,rng.random() < ctx_prob))
    T=max(len(x[0]) for x in xs)
    xb=torch.full((len(xs),T),tok.pad,dtype=torch.long)
    yb=torch.full((len(xs),T),-100,dtype=torch.long)
    for i,(ids,labels) in enumerate(xs):
        xb[i,:len(ids)]=torch.tensor(ids)
        yb[i,:len(labels)]=torch.tensor(labels)
    return xb.to(device),yb.to(device)


@torch.no_grad()
def eval_raw_loss(model,tok,cfg,pairs,device,seed,batches=16):
    if not pairs:
        return 999.0
    rng=random.Random(seed)
    model.eval(); vals=[]
    for _ in range(batches):
        x,y=make_batch(tok,cfg,pairs,device,rng,ctx_prob=0.0)
        _,loss=model(x,y)
        vals.append(float(loss))
    model.train()
    return sum(vals)/len(vals)


def cpu_state_dict(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def move_optimizer_state(opt,device):
    for st in opt.state.values():
        for k,v in list(st.items()):
            if torch.is_tensor(v):
                st[k]=v.to(device)


def train(args):
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    pairs=load_training_pairs(args.train_file)
    tr,va=h.split_pairs(pairs)
    data_hash=hashlib.sha256("\n".join(f"{q}={a}" for q,a in pairs).encode()).hexdigest()
    device="cuda" if args.device in ("auto","cuda") and torch.cuda.is_available() else "cpu"
    cfg=dict(h.PRESETS[args.preset])
    ck=h.checkpoint_load(args.model_file) if args.resume else None
    rng=random.Random(args.seed)

    if ck:
        if ck.get("version") != "hgai-v10" or ck.get("preset") != args.preset:
            raise SystemExit("checkpoint version/preset mismatch")
        if ck.get("dataset_hash") != data_hash:
            raise SystemExit("dataset changed; refusing unsafe resume")
        tok=h.Tokenizer(ck["vocab"]); cfg=dict(ck["config"])
        model=h.HGAIModel(len(tok.vocab),cfg).to(device)
        model.load_state_dict(ck.get("resume_model") or ck["model"])
        start=int(ck.get("step",0)); best=float(ck.get("best_val",999.0)); best_step=int(ck.get("best_step",start))
        best_state={k:v.detach().cpu().clone() for k,v in ck["model"].items()}
        if ck.get("rng_state") is not None:
            try: rng.setstate(ck["rng_state"])
            except Exception: pass
    else:
        tok=h.Tokenizer(h.build_vocab(tr,cfg["max_vocab"]))
        model=h.HGAIModel(len(tok.vocab),cfg).to(device)
        start=0; best=999.0; best_step=0; best_state=None

    trset=set(tr)
    core=[x for x in h.expand_pairs(h.curated_pairs(),max_variants=8) if x in trset]
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.1)
    if ck and ck.get("optimizer"):
        try:
            opt.load_state_dict(ck["optimizer"]); move_optimizer_state(opt,device)
        except Exception as e:
            print("optimizer resume skipped:",e)

    total=max(1,start+args.steps)
    warm=max(20,min(300,total//20))
    ev=max(25,min(250,args.steps//10 if args.steps>=10 else 1))
    eval_seed=args.seed+9173
    last_loss=None; last_val=999.0
    print(f"HGAI quality trainer preset={args.preset} params={h.count_params(model):,} pairs={len(pairs):,} core={len(core):,} start={start} device={device}")

    for local in range(args.steps):
        step=start+local
        progress=(step+1)/max(1,total)
        focus_prob=.45 if progress < .45 else (.32 if progress < .80 else .22)
        x,y=make_batch(tok,cfg,tr,device,rng,ctx_prob=.30,focus_pairs=core,focus_prob=focus_prob)
        _,loss=model(x,y); last_loss=float(loss.detach())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        p=min(1,max(0,(step-warm)/max(1,total-warm)))
        scale=(step+1)/warm if step < warm else .1+.9*.5*(1+math.cos(math.pi*p))
        for group in opt.param_groups: group["lr"]=args.lr*max(.05,scale)
        opt.step()

        if local==0 or (local+1)%ev==0 or local==args.steps-1:
            last_val=eval_raw_loss(model,tok,cfg,va,device,eval_seed)
            if last_val < best:
                best=last_val; best_step=step+1; best_state=cpu_state_dict(model)
            print(f"step {step+1} train={last_loss:.4f} raw_val={last_val:.4f} best={best:.4f}@{best_step}")
            resume_state=cpu_state_dict(model)
            torch.save({
                "version":"hgai-v10","preset":args.preset,"config":cfg,"vocab":tok.vocab,
                "model":best_state if best_state is not None else resume_state,
                "resume_model":resume_state,"optimizer":opt.state_dict(),"rng_state":rng.getstate(),
                "step":step+1,"best_step":best_step,"best_val":best,"last_val":last_val,
                "params":h.count_params(model),"dataset_hash":data_hash,"artifact_ready":False,
                "trainer":"quality-v1"
            },args.model_file)

    result={"step":start+args.steps,"best_step":best_step,"best_val":best,"last_val":last_val,
            "last_loss":last_loss,"params":h.count_params(model),"pairs":len(pairs),"core_pairs":len(core),
            "dataset_hash":data_hash,"trainer":"quality-v1"}
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-file",default="train.txt")
    ap.add_argument("--model-file",default="hgai_model_v10.pth")
    ap.add_argument("--preset",default="main",choices=sorted(h.PRESETS))
    ap.add_argument("--steps",type=int,default=2500)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--seed",type=int,default=20260822)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args(); train(args)

if __name__=="__main__": main()
