#!/usr/bin/env python3
"""v16.2 experimental neural continuation from the latest stable v10 model.

The existing Pi's CPU-friendly HGAIModel architecture, vocabulary IDs and
checkpoint format are preserved. Facts from v13 are split by provenance group,
never from paraphrases of the same fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import hgai_core as h
import train_v10_quality as quality

def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def read_pairs(rows, max_count, seed):
    rng=random.Random(seed)
    by_group={}
    for row in rows:
        q,a=row["user"].strip(),row["assistant"].strip()
        if not q or not a or len(q)>205 or len(a)>280:
            continue
        key=(row["group"],row["language"])
        if key not in by_group:
            by_group[key]=(q,a)
    vals=list(by_group.values())
    rng.shuffle(vals)
    return vals[:max_count],len(by_group)

def simple_eval(model,tok,cfg,examples,seed,batches=12):
    # Measure teacher-forced likelihood, never call the answer router.
    return quality.eval_raw_loss(model,tok,cfg,examples,"cpu",seed,batches=batches)

def evaluate_replies(model,tok,cfg,questions,seed):
    model.eval()
    examples=[]
    for i,q in enumerate(questions):
        out=h.neural_generate(model,tok,"cpu",cfg,q,None,[],
                              temperature=.55,top_k=20,top_p=.85,
                              max_new_tokens=58,seed=seed+i)
        examples.append({"question":q,"raw_neural":out or ""})
    return examples

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-model",default="latest/hgai_model_v10.pth")
    ap.add_argument("--game-train",default="v13_game_data/game_qa_v13_train.jsonl")
    ap.add_argument("--game-holdout",default="v13_game_data/game_qa_v13_holdout.jsonl")
    ap.add_argument("--out",default="hgai_model_v16_2.pth")
    ap.add_argument("--steps",type=int,default=900)
    ap.add_argument("--eval-every",type=int,default=150)
    ap.add_argument("--lr",type=float,default=4e-5)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20261014)
    args=ap.parse_args()
    if not 1<=args.steps<=3600 or args.eval_every<1 or args.threads<1:
        ap.error("invalid resource bounds")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); random.seed(args.seed)
    rng=random.Random(args.seed)
    ck=h.checkpoint_load(args.base_model)
    if not ck or ck.get("preset")!="main" or ck.get("version")!="hgai-v10":
        raise SystemExit("A real compatible v10 main checkpoint is required")
    cfg=dict(ck["config"])
    tok=h.Tokenizer(ck["vocab"])
    model=h.HGAIModel(len(tok.vocab),cfg).to("cpu")
    model.load_state_dict(ck["model"])
    initial_step=int(ck.get("step",0))
    if initial_step<1500:raise SystemExit("Initializer is not the trained v10 main model")
    all_general=h.training_pairs("train.txt",max_variants=2)
    general_train,general_valid=h.split_pairs(all_general,pct=3)
    if len(general_train)<1000 or not general_valid:
        raise SystemExit("General training source unexpectedly missing")
    game_train,distinct_train=read_pairs(load_rows(args.game_train),max_count=1800,seed=args.seed)
    game_valid,distinct_valid=read_pairs(load_rows(args.game_holdout),max_count=120,seed=args.seed+3)
    if len(game_train)<350 or len(game_valid)<30:
        raise SystemExit("v13 game training or holdout dataset missing")
    original_train=load_rows(args.game_train)
    original_holdout=load_rows(args.game_holdout)
    if {x["group"] for x in original_train} & {x["group"] for x in original_holdout}:
        raise SystemExit("GAME FACT LEAKAGE DETECTED")
    # Preserve original vocabulary IDs: neither reset the tokenizer nor randomly
    # initialize the old language model! This stays compatible with Raspberry Pi.
    # Any newly unseen terms have reduced expressiveness (logged below).
    all_tokens=sum(len(tok.encode(q+" "+a)) for q,a in game_train)
    unknown_tokens=sum(sum(t==tok.unk for t in tok.encode(q+" "+a)) for q,a in game_train)
    print("GAME TOKEN OOV rate",round(unknown_tokens/max(1,all_tokens),4),flush=True)
    curated_game=[(row["user"],row["assistant"]) for row in original_train
                  if row["group"].startswith("game:") and len(row["assistant"])<=280]
    if not curated_game:raise SystemExit("Curated game facts not found")
    print(json.dumps({"init_step":initial_step,"init_params":h.count_params(model),
                      "vocab":len(tok.vocab),
                      "general_train":len(general_train),"general_val":len(general_valid),
                      "game_train":len(game_train),"game_val":len(game_valid),
                      "game_train_groups":distinct_train,"game_val_groups":distinct_valid,
                      "curated_examples":len(curated_game)},ensure_ascii=False),flush=True)
    model.train()
    base_general=simple_eval(model,tok,cfg,general_valid,args.seed+27,batches=16)
    base_game=simple_eval(model,tok,cfg,game_valid,args.seed+39,batches=12)
    print("BASELINE general=%.4f game=%.4f"%(base_general,base_game),flush=True)
    pool={"general":general_train,"game":game_train,"curated":curated_game}
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.015)
    best_score=float("inf"); best_state=None; best_step=0
    history=[]
    for step in range(1,args.steps+1):
        choice=rng.random()
        category="general" if choice<.53 else ("game" if choice<.83 else "curated")
        x,y=quality.make_batch(tok,cfg,pool[category],"cpu",rng,ctx_prob=0.04)
        model.train(); logits,loss=model(x,y)
        if not torch.isfinite(loss):raise RuntimeError("non-finite training loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        for group in opt.param_groups:
            group["lr"]=args.lr*(.18+.82*(1-step/(args.steps+1)))
        opt.step()
        if step%args.eval_every!=0 and step!=args.steps:continue
        vg=simple_eval(model,tok,cfg,general_valid,args.seed+27,batches=16)
        vx=simple_eval(model,tok,cfg,game_valid,args.seed+39,batches=12)
        # Relative score discourages catastrophic general language forgetting.
        score=.65*(vg/base_general)+.35*(vx/base_game)
        allowed=vg<=base_general+0.22
        selected=allowed and score<best_score
        if selected:
            best_score=score;best_step=step
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rec={"step":step,"loss":round(float(loss),4),
             "general_holdout":round(vg,4),"game_holdout":round(vx,4),
             "score":round(score,4),"general_regression_allowed":allowed,
             "selected":selected}
        history.append(rec)
        print(json.dumps(rec,ensure_ascii=False),flush=True)
    if best_state is None:
        raise SystemExit("General regression exceeded ceiling at all checkpoints")
    model.load_state_dict(best_state)
    model.eval()
    game_best=simple_eval(model,tok,cfg,game_valid,args.seed+39,batches=16)
    general_best=simple_eval(model,tok,cfg,general_valid,args.seed+27,batches=16)
    # Raw samples are diagnostics only. Source-router answers do not count here.
    probes=[
       "C:U에서 SAW 프로젝트가 뭔지 알려줘.",
       "Gunsaw는 누가 개발했어?",
       "로블록스는 어떤 언어로 스크립트를 써?",
       "Minecraft 레드스톤은 뭐에 쓰여?",
       "What's the difference between TCP and UDP?",
       "What language does Roblox use for scripting?",
       "I am building a tiny game. How should I test it?",
       "게임의 프레임이 떨어질 때 뭘 확인할까?",
    ]
    raw=evaluate_replies(model,tok,cfg,probes,args.seed)
    improvement=(general_best<=base_general+0.10 and game_best<base_game-0.05)
    checkpoint={
      "version":"hgai-v10","preset":"main","config":cfg,"vocab":tok.vocab,
      "model":best_state,"step":initial_step+best_step,
      "best_step":initial_step+best_step,"best_val":float(general_best),
      "last_val":float(general_best),"params":h.count_params(model),
      "dataset_hash":hashlib.sha256((str(ck.get("dataset_hash",""))+
                       Path(args.game_train).read_text(encoding="utf-8")[:1000]).encode()).hexdigest(),
      "semantic_ready":False,"artifact_ready":False,"hybrid_gate":False,
      "raw_neural_gate":False,"trainer":"continued-v16.2-experimental",
      "source_training_run":38484724746,
      "source_init_step":initial_step,
      "experimental_quality_ok":improvement,
      "note":"Not promoted: must review raw responses and compare independently. Source-fact retrieval is separate from neural knowledge."
    }
    torch.save(checkpoint,args.out)
    reopened=h.checkpoint_load(args.out)
    assert reopened["trainer"]=="continued-v16.2-experimental"
    verify=h.HGAIModel(len(reopened["vocab"]),reopened["config"])
    verify.load_state_dict(reopened["model"]);verify.eval()
    report={
        "status":"trained_experimental_not_deployed",
        "base_run":38484724746,"init_step":initial_step,
        "new_optimizer_steps":args.steps,"chosen_checkpoint_step":best_step,
        "result_step":initial_step+best_step,
        "parameters":h.count_params(model),
        "vocab_size":len(tok.vocab),"game_unknown_token_rate":unknown_tokens/max(1,all_tokens),
        "train_records":len(game_train)+len(general_train)+len(curated_game),
        "game_source_group_overlap":False,
        "general_loss_baseline":base_general,"general_loss_candidate":general_best,
        "game_loss_baseline":base_game,"game_loss_candidate":game_best,
        "automatic_promote":False,"token_loss_improvement_gate":improvement,
        "history":history,"neural_probes":raw,
        "caution":"Raw open-domain correctness is not established by teacher-forced losses. Use reviewed v16 source routing for verified game facts."
    }
    Path("V16_2_TRAINING_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("history","neural_probes")},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
