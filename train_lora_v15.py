#!/usr/bin/env python3
"""HGAIM v15 Qwen3 LoRA, CPU-safe and assistant-span-correct multi-turn training.

Trains all assistant messages rather than only the last response.
Game-source holdout remains outside gradient updates.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="Qwen/Qwen3-0.6B"

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def encode_dialogue(tok, sample, max_length):
    messages=sample["messages"]
    if messages[-1]["role"]!="assistant":
        raise ValueError("Last turn must be an assistant response")
    full=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    encoding=tok(full,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoding["input_ids"]
    offsets=encoding["offset_mapping"]
    # Locate literal assistant content inside the fully rendered chat.
    # This handles Qwen3's special no-thinking preamble without mistaking it
    # for the answer, and supervises every assistant turn in a multi-turn chat.
    spans=[]
    cursor=0
    for m in messages:
        content=m["content"]
        start=full.find(content,cursor)
        if start<0:
            raise ValueError("Could not match message in Qwen3 chat template")
        end=start+len(content)
        if m["role"]=="assistant":
            closer_candidates=[i for tag in ("<|im_end|>","<|fim_suffix|>") if (i:=full.find(tag,end))>=0]
            if closer_candidates:
                end=min(closer_candidates)+len("<|im_end|>") if min(closer_candidates)==full.find("<|im_end|>",end) else min(closer_candidates)+len("<|fim_suffix|>")
            spans.append((start,end))
        cursor=start+len(content)
    labels=[tid if any(a<=st and en<=b and st<en for a,b in spans) else -100
            for tid,(st,en) in zip(ids,offsets)]
    start=max(0,len(ids)-max_length)
    cut_ids=ids[start:]
    cut_labels=labels[start:]
    if sum(x!=-100 for x in cut_labels)<5:
        return None
    return {"ids":cut_ids,"labels":cut_labels,
            "supervised_tokens":sum(x!=-100 for x in cut_labels),
            "assistant_turns":len(spans)}

def collate(records,pad):
    m=max(len(x["ids"]) for x in records)
    ids=torch.full((len(records),m),pad,dtype=torch.long)
    mask=torch.zeros_like(ids)
    y=torch.full((len(records),m),-100,dtype=torch.long)
    for i,r in enumerate(records):
        n=len(r["ids"])
        ids[i,:n]=torch.tensor(r["ids"])
        mask[i,:n]=1
        y[i,:n]=torch.tensor(r["labels"])
    return {"input_ids":ids,"attention_mask":mask,"labels":y}

@torch.inference_mode()
def heldout_loss(model,records,pad):
    was_training=model.training;model.eval()
    total=0.
    tokens=0
    for rec in records:
        batch=collate([rec],pad)
        loss=float(model(**batch).loss)
        count=max(1,rec["supervised_tokens"])
        total+=loss*count;tokens+=count
    if was_training:model.train()
    return total/max(1,tokens)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default=BASE)
    p.add_argument("--train",default="v15_data/train.jsonl")
    p.add_argument("--valid",default="v15_data/holdout.jsonl")
    p.add_argument("--output",default="hgaim_v15_lora")
    p.add_argument("--steps",type=int,default=960)
    p.add_argument("--eval-every",type=int,default=160)
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--lr",type=float,default=6e-5)
    p.add_argument("--seed",type=int,default=20261013)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--eval-count",type=int,default=24)
    args=p.parse_args()
    if args.steps<1 or args.steps>3000 or args.eval_every<1 or not(128<=args.max_length<=512) or args.threads<1:
        p.error("Invalid CPU-safety bounds")
    torch.manual_seed(args.seed);rng=random.Random(args.seed);torch.set_num_threads(args.threads)
    tok=AutoTokenizer.from_pretrained(args.base,trust_remote_code=False,use_fast=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    all_train=rows(args.train)
    all_valid=rows(args.valid)
    train=[(r,t) for r in all_train if (t:=encode_dialogue(tok,r,args.max_length))]
    valid=[(r,t) for r in all_valid if (t:=encode_dialogue(tok,r,args.max_length))]
    if len(train)<500 or len(valid)<20:
        raise ValueError(f"Too few encoded samples train={len(train)}, valid={len(valid)}")
    # The holdout is never sampled for gradients. Use a deterministic, balanced,
    # multilingual subset for checkpoints instead of first N records only.
    rng.shuffle(valid)
    en=[t for r,t in valid if r["language"]=="en"]
    ko=[t for r,t in valid if r["language"]=="ko"]
    limit=max(2,args.eval_count//2)
    selected_valid=en[:limit]+ko[:limit]
    if len(selected_valid)<8:raise ValueError("Evaluation split missing a language")
    pools={
      "cu":[t for r,t in train if r["source_group"].startswith(("cu:","game:Casualties: Unknown:"))],
      "named_games":[t for r,t in train if r["source_group"].startswith("game:") and not r["source_group"].startswith("game:Casualties: Unknown:")],
      "dialogues":[t for r,t in train if r["kind"]=="dialogue"],
    }
    if any(not v for v in pools.values()):raise ValueError("Curriculum pool empty")
    # Repeated factual paraphrases don't count as unique knowledge.
    train_groups={r["source_group"] for r,t in train}
    if any(r["source_group"] in train_groups for r,t in valid):
        raise ValueError("Train/holdout source group overlap")
    model=AutoModelForCausalLM.from_pretrained(args.base,dtype=torch.float32,low_cpu_mem_usage=True,trust_remote_code=False)
    model.config.use_cache=False
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM",bias="none")
    model=get_peft_model(model,lora)
    model.train()
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=args.lr,weight_decay=.02)
    baseline=heldout_loss(model,selected_valid,tok.pad_token_id)
    best=baseline;best_step=0
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    progress=[]
    print(json.dumps({"base":args.base,"train":len(train),"holdout":len(valid),
       "heldout_sampled":len(selected_valid),
       "supervised_multi_turn_records":sum(1 for r,t in train if t["assistant_turns"]>1),
       "pools":{k:len(v) for k,v in pools.items()},
       "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
       "baseline_holdout_loss":baseline}),flush=True)
    for step in range(1,args.steps+1):
        sample_roll=rng.random()
        pool="cu" if sample_roll<.35 else ("named_games" if sample_roll<.68 else "dialogues")
        batch=collate([rng.choice(pools[pool])],tok.pad_token_id)
        output=model(**batch)
        loss=output.loss
        if not torch.isfinite(loss):raise RuntimeError("Training diverged")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
        lr=args.lr*(.2+.8*.5*(1+math.cos(math.pi*step/max(1,args.steps))))
        for g in optimizer.param_groups:g["lr"]=lr
        optimizer.step()
        if step%args.eval_every and step!=args.steps:continue
        val=heldout_loss(model,selected_valid,tok.pad_token_id)
        improved=(val<best-.005)
        if improved:
            best=val;best_step=step
            model.save_pretrained(out)
        record={"step":step,"train_loss":float(loss),"holdout_loss":val,"selected":improved}
        progress.append(record)
        print(json.dumps(record),flush=True)
    if best_step==0:model.save_pretrained(out)
    tok.save_pretrained(out)
    report={
      "version":"HGAIM-v15-Qwen3","base":args.base,"steps_completed":args.steps,
      "best_step":best_step,"baseline_holdout_loss":baseline,
      "best_holdout_loss":best if best_step else None,
      "improved_holdout":bool(best_step),
      "train_samples":len(train),"holdout_samples":len(valid),
      "max_length":args.max_length,
      "supervised_multi_turn_records":sum(1 for r,t in train if t["assistant_turns"]>1),
      "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
      "history":progress,
      "note":"Holdout loss is teacher-forced, not a generated factuality or fluency score. Adapter is not GGUF. Preserve live Pi deployment."
    }
    (out/"training_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="history"},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
