#!/usr/bin/env python3
"""Merge an independently-reviewed HGAIM v14 LoRA adapter into HF fp16 weights.

Run this on a PC or GitHub Actions runner, NEVER on a 1GB Raspberry Pi.
After merging, convert the result to GGUF with llama.cpp's converter.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="hgaim_v14_lora")
    parser.add_argument("--output", default="hgaim_v14_merged")
    parser.add_argument("--require-holdout-improvement", action="store_true", default=True)
    parser.add_argument("--force-experimental-export", action="store_true")
    args = parser.parse_args()
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        parser.error("Expected a trained PEFT adapter directory")
    report_path = adapter / "training_report.json"
    if not report_path.exists():
        parser.error("Adapter training_report.json missing. Not exporting an untracked artifact")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["base_model"] != "Qwen/Qwen2.5-0.5B-Instruct":
        parser.error("Unknown base model: will not merge wrong architecture")
    if not report.get("improved_holdout") and not args.force_experimental_export:
        parser.error("Holdout loss did not improve; use --force-experimental-export for research only")
    model = AutoModelForCausalLM.from_pretrained(
        report["base_model"], torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload(safe_merge=True)
    out = Path(args.output)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(report["base_model"])
    tokenizer.save_pretrained(out)
    (out / "HGAIM_V14_ORIGIN.json").write_text(
        json.dumps({
            "base": report["base_model"], "source_adapter": str(adapter),
            "improved_holdout": report.get("improved_holdout", False),
            "human_approved": False, "format": "HF merged fp16, NOT GGUF",
        }, indent=2)+"\n", encoding="utf-8")
    print("Merged to", out, "-- not a GGUF yet, manual quality check required.")


if __name__ == "__main__":
    main()
