#!/usr/bin/env python3
"""Fail-closed audit of *already completed* HGAIM training artifacts.

This does not train, publish, or deploy a model. Keyword pass rates and
teacher-forced loss cannot establish factual correctness or fluency.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile


def read_json(archive: ZipFile, name: str) -> dict:
    return json.loads(archive.read(name).decode("utf-8"))


def audit_candidate(base_ko: dict, base_en: dict, new_ko: dict, new_en: dict,
                    evidence: dict | None = None) -> dict:
    evidence = evidence or {}
    for label, record in (("base_ko", base_ko), ("base_en", base_en),
                          ("new_ko", new_ko), ("new_en", new_en)):
        if not isinstance(record.get("passed"), int) or not isinstance(record.get("total"), int):
            raise ValueError(f"{label}: missing pass count")
        if not 0 <= record["passed"] <= record["total"]:
            raise ValueError(f"{label}: invalid pass count")
        if len(record.get("tests", [])) != record["total"]:
            raise ValueError(f"{label}: missing per-prompt raw outputs")
    if base_ko["total"] != new_ko["total"] or base_en["total"] != new_en["total"]:
        raise ValueError("Baseline/candidate question counts differ")
    ko0, ko1 = base_ko["passed"], new_ko["passed"]
    en0, en1 = base_en["passed"], new_en["passed"]
    mixed = sum(bool(re.search(r"[가-힣]", item.get("answer", "")))
                for item in new_en["tests"])
    # These are *minimum review thresholds*, not proof of correctness.
    reasons = []
    if ko1 < ko0:
        reasons.append("Korean raw-neural pass count regressed")
    if en1 < en0:
        reasons.append("English raw-neural pass count regressed")
    if ko1 < 8 or en1 < 10:
        reasons.append("Raw-neural review floors (KO 8/10, EN 10/15) not met")
    if mixed:
        reasons.append(f"English raw outputs contain Hangul in {mixed} prompts")
    for item in ("independent_game_fact_accuracy", "multi_turn_raw_neural",
                 "nyang_style_review", "train_validation_provenance_audit"):
        if evidence.get(item) is not True:
            reasons.append(f"Missing verified evidence: {item}")
    return {
        "baseline_ko": f"{ko0}/{base_ko['total']}",
        "candidate_ko": f"{ko1}/{new_ko['total']}",
        "baseline_en": f"{en0}/{base_en['total']}",
        "candidate_en": f"{en1}/{new_en['total']}",
        "english_hangul_mixed_prompts": mixed,
        "review_eligible": not reasons,
        "automatic_promotion": False,
        "reasons": reasons,
    }


def audit_archives(v10_path: str, v16_path: str) -> dict:
    with ZipFile(v10_path) as z10, ZipFile(v16_path) as z16:
        baseline_ko = read_json(z10, "V10_3_BASE_KO.json")
        baseline_en = read_json(z10, "V10_3_BASE_EN.json")
        v10_ko = read_json(z10, "V10_3_TUNED_KO.json")
        v10_en = read_json(z10, "V10_3_TUNED_EN.json")
        v16_ko = read_json(z16, "V16_2_KO_NEURAL.json")
        v16_en = read_json(z16, "V16_2_EN_NEURAL.json")
        selected = read_json(z10, "V10_3_SELECTION.json")
        trained = read_json(z16, "V16_2_TRAINING_REPORT.json")
    result10 = audit_candidate(baseline_ko, baseline_en, v10_ko, v10_en)
    result16 = audit_candidate(baseline_ko, baseline_en, v16_ko, v16_en)
    if selected.get("promoted_as_safe") and not result10["review_eligible"]:
        result10["reasons"].append(
            "Existing v10.3 workflow labeled new weights safe despite failing this audit"
        )
    return {
        "baseline_run": 38484724746,
        "v10_3_run": 38524946883,
        "v16_2_run": 38525176474,
        "v10_3": result10,
        "v16_2": result16,
        "v16_2_source_group_nonoverlap_reported": trained.get("game_source_group_overlap") is False,
        "v16_2_independent_leakage_audit": "not established by available artifacts",
        "deployment_authorized": False,
        "retraining_started": False,
        "note": "No model promotion or expensive retraining. Raw generations fail quality review; source-routed answers are not raw-neural accuracy.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10-artifact", required=True)
    ap.add_argument("--v16-artifact", required=True)
    ap.add_argument("--output", default="HGAIM_QUALITY_AUDIT.json")
    args = ap.parse_args()
    result = audit_archives(args.v10_artifact, args.v16_artifact)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
