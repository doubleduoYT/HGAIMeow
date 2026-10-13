#!/usr/bin/env python3
"""Merge an evaluated Qwen3 v15 adapter before an *external* GGUF conversion.

Use a PC/Actions runner, NEVER the 1GB Raspberry Pi. Does not auto-deploy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE="Qwen/Qwen3-0.6B"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--adapter",default="hgaim_v15_lora")
    p.add_argument("--output",default="hgaim_v15_merged")
    p.add_argument("--acknowledge-experimental",action="store_true")
    args=p.parse_args()
    root=Path(args.adapter)
    if not args.acknowledge_experimental:
        p.error("Requires --acknowledge-experimental; manual factual/quality review is required")
    report_path=root/"training_report.json"
    config_path=root/"adapter_config.json"
    if not report_path.is_file() or not config_path.is_file():
        p.error("Tracked trained adapter and report are required")
    report=json.loads(report_path.read_text(encoding="utf-8"))
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if report.get("base")!=BASE or config.get("base_model_name_or_path")!=BASE:
        p.error("Adapter and trained report do not match Qwen3-0.6B")
    model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,str(root))
    merged=model.merge_and_unload(safe_merge=True)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    merged.save_pretrained(dest,safe_serialization=True)
    tok=AutoTokenizer.from_pretrained(BASE)
    tok.save_pretrained(dest)
    (dest/"HGAIM_ORIGIN.json").write_text(json.dumps({
        "base":BASE,"adapter":str(root),
        "selected_step":report.get("best_step"),"human_factual_accuracy_approved":False,
        "format":"HuggingFace FP16, NOT GGUF",
        "warning":"Convert to GGUF and quantize on another machine; test Pi RAM and output semantics before serving."
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Wrote research-only merged FP16 checkpoint:",dest)
    print("Use a compatible Qwen3 llama.cpp converter separately. This output is NOT a .gguf.")

if __name__=="__main__":main()
