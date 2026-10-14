#!/usr/bin/env python3
"""HGAIM v10.3: continue the *latest main-run* released v10 weights with bilingual game Q&A.

Do not confuse training a 4.7M standalone PyTorch model with training v15 Qwen3.
The input is a final inference checkpoint (optimizer state stripped), so this is
a fresh low-LR optimizer *continuation*, not an Adam momentum resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import hgai_core as h
from nyang_style_v14 import nyangify

def make_source_examples(facts, seed=20261014):
    rows = json.loads(Path(facts).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    train, validation = [], []
    groups_train, groups_val = set(), set()
    seen = set()
    for entry in rows:
        if entry["game"] == "DevilConnection":
            # Unknown identity: source references absent, don't make up gameplay.
            continue
        group = f"{entry['game']}:{entry['id']}"
        # Whole fact groups (both languages) stay in a single partition.
        val = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10 == 0
        target, group_set = (validation, groups_val) if val else (train, groups_train)
        for lang in ("ko", "en"):
            q = entry["q_"+lang].strip()
            answer = entry["a_"+lang].strip()
            if not q or not answer:continue
            # Real source-linked facts, never repeat 3 near-identical labels to
            # pretend we have thousands of new independent knowledge items.
            answer = nyangify(answer, lang)
            # Truncate extremely long answers for old v10 context capacity.
            if len(answer) > 210: answer = answer[:210].rsplit(" ",1)[0].rstrip(" ,;") + ("다냥." if lang=="ko" else ", meow.")
            tag=(q.casefold(),answer.casefold())
            if tag not in seen:
                target.append((q,answer))
                seen.add(tag)
        group_set.add(group)
    if groups_train & groups_val:
        raise ValueError("Fact group leakage")
    if len(train) < 110 or len(validation) < 10:
        raise ValueError(f"Game QA corpus unexpectedly small: train={len(train)} val={len(validation)}")
    rng.shuffle(train)
    return train, validation, groups_train, groups_val

def make_batch(model, tok, cfg, pairs, rng, context_rate=.0):
    examples=[h.example(tok,cfg,*pairs[rng.randrange(len(pairs))],rng.random()<context_rate) for _ in range(cfg["batch_size"])]
    T=max(len(x[0]) for x in examples)
    xx=torch.full((len(examples),T),tok.pad,dtype=torch.long)
    yy=torch.full((len(examples),T),-100,dtype=torch.long)
    for j,(x,y) in enumerate(examples):
        xx[j,:len(x)]=torch.tensor(x)
        yy[j,:len(y)]=torch.tensor(y)
    return xx,yy

@torch.inference_mode()
def loss_sample(model,tok,cfg,pairs,seed,limit=24):
    rng=random.Random(seed)
    model.eval()
    samples=[]
    if len(pairs)<limit: limit=len(pairs)
    selected=rng.sample(pairs, limit)
    for q,a in selected:
        x,y=h.example(tok,cfg,q,a,False)
        if sum(int(t!=-100) for t in y)<4:continue
        logits,loss=model(torch.tensor([x],dtype=torch.long),torch.tensor([y],dtype=torch.long))
        samples.append(float(loss))
    model.train()
    return sum(samples)/len(samples) if samples else 999.

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--init",default="hgai_model_latest_main.pth")
    p.add_argument("--output",default="hgai_model_v10_3.pth")
    p.add_argument("--report",default="V10_3_TRAINING_REPORT.json")
    p.add_argument("--facts",default="game_facts_v13.json")
    p.add_argument("--steps",type=int,default=1600)
    p.add_argument("--lr",type=float,default=1.5e-5)
    p.add_argument("--seed",type=int,default=20261014)
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if not 1<=args.steps<=3000 or not 0<args.lr<=1e-4:p.error("Outside safe training range")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    previous=h.checkpoint_load(args.init)
    if not previous or previous.get("version")!="hgai-v10" or previous.get("preset")!="main":
        p.error("Cannot load exact v10-main predecessor checkpoint")
    cfg=dict(previous["config"])
    tok=h.Tokenizer(previous["vocab"])
    model=h.HGAIModel(len(tok.vocab),cfg)
    model.load_state_dict(previous["model"])
    prev_params=h.count_params(model)

    general=h.training_pairs("train.txt")
    general_train,general_validation=h.split_pairs(general)
    # Curated English + Korean are sampled alongside the generic dataset.
    curated=h.curated_pairs()
    if not curated:raise ValueError("Curated training data missing")
    game_train,game_val,game_train_groups,game_valid_groups=make_source_examples(args.facts,args.seed)
    # Factual questions that happen to match a general reference question are
    # excluded from the general sampler to preserve exact-string split safety.
    game_val_q={q.strip().casefold() for q,_ in game_val}
    general_train=[x for x in general_train if x[0].strip().casefold() not in game_val_q]
    curated=[x for x in curated if x[0].strip().casefold() not in game_val_q]

    # The original checkpoint's source domain is used for regression checking.
    base_general=loss_sample(model,tok,cfg,general_validation,args.seed+21,limit=32)
    base_game=loss_sample(model,tok,cfg,game_val,args.seed+29,limit=24)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.015)
    best_score=0.
    best_step=0
    best_snapshot=None
    history=[]
    epochs=max(1,args.steps)
    for step in range(1,args.steps+1):
        roll=rng.random()
        if roll < .40:
            pool=game_train
        elif roll < .65:
            pool=curated
        else:
            pool=general_train
        x,y=make_batch(model,tok,cfg,pool,rng,0.02 if pool is game_train else 0.08)
        model.train()
        _,loss=model(x,y)
        if not torch.isfinite(loss): raise RuntimeError("Invalid loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        rate=args.lr*(.2+.8*.5*(1+math.cos(math.pi*step/epochs)))
        for g in opt.param_groups:g["lr"]=rate
        opt.step()
        if step%200==0 or step==args.steps:
            general_loss=loss_sample(model,tok,cfg,general_validation,args.seed+21,limit=32)
            game_loss=loss_sample(model,tok,cfg,game_val,args.seed+29,limit=24)
            # Do not select a checkpoint that seriously degrades the original
            # general language validation relative to the latest main model.
            safe=(general_loss<=base_general+.14)
            improvement=(base_game-game_loss)
            if safe and improvement>best_score+.012:
                best_score=improvement
                best_step=step
                best_snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            rec={"step":step,"training_loss":float(loss),"general_holdout_loss":general_loss,
                 "game_holdout_loss":game_loss,"passes_general_retention":bool(safe),
                 "selected_step_so_far":best_step}
            history.append(rec)
            print(json.dumps(rec,ensure_ascii=False),flush=True)

    selected=best_snapshot if best_snapshot is not None else previous["model"]
    ck=dict(previous)
    ck["model"]=selected
    ck["step"]=int(previous.get("step",0))+ (best_step if best_step else 0)
    ck["best_step"]=ck["step"]
    ck["v10_3_train_steps_attempted"]=args.steps
    ck["v10_3_selected_training_steps"]=best_step
    ck["v10_3_initializer"]=f"GitHub Actions main run 38484724746 artifact 11859211226"
    ck["v10_3_game_facts_evaluated_only_not_runtime_trained"]=False
    ck["v10_3_released"]=False
    # Do not copy semantic readiness certificates from the earlier checkpoint.
    # Independent free-generation tests on the final model must re-establish it.
    ck["artifact_ready"]=False
    ck["semantic_ready"]=False
    ck["raw_neural_gate"]=False
    ck["hybrid_gate"]=False
    for unwanted in ("optimizer","resume_model","rng_state"):
        ck.pop(unwanted,None)
    torch.save(ck,args.output)
    summary={
        "model":"HGAIM-v10.3-standalone-4.7M",
        "latest_initializer":"main run 38484724746, artifact 11859211226",
        "params":prev_params,"vocabulary":len(tok.vocab),
        "attempted_steps":args.steps,"selected_step":best_step,
        "selected_new_weights":best_snapshot is not None,
        "baseline_general_holdout_loss":base_general,
        "baseline_game_holdout_loss":base_game,
        "general_regression_max_delta":.14,
        "game_holdout_fact_groups":len(game_valid_groups),
        "game_train_fact_groups":len(game_train_groups),
        "game_train_examples":len(game_train),
        "game_valid_examples":len(game_val),
        "game_sources_are_reviewer_cards_not_verbatim_localization":True,
        "validation_caveat":"Lower teacher-forced loss does not establish correct raw-neural conversation. See independent generated reports.",
        "history":history,
    }
    Path(args.report).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="history"},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
