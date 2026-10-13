# HGAIM v12 — Generative Quality (experimental)

Goal: make the **standalone** neural model produce better Korean/English dialogue, with **no retrieval in raw-neural inference**. This is NOT a claim that a 4.7M-parameter network has achieved general conversation.

## What has actually changed

- New `train_dialogues_v12.jsonl`: **98 independent situations** (49 Korean / 49 English), each with two user formulations and two assistant turns, for **196 dialogues / 392 training turns**. The two formulations share target answers, so do not confuse rewritten samples with independent knowledge.
- Combined with the previous two dialogue files, the continuation-trainer sees **601 dialogue turns**.
- `validate_dialogues_v12.jsonl` contains **40 distinct held-out sessions / 80 turns**, used for checkpoint selection, never for gradient updates.
- `train_neural_v12.py` continues from the completed **v11.1** checkpoint instead of returning to the v10 baseline. It uses a longer schedule (5,400 optimizer steps), effective batch size 8, and cosine-decayed learning rate. We keep the v10 checkpoint format for the Raspberry Pi's existing model loader.
- `hgai_core.py` now prevents generating the padding/unknown/internal role tokens, hard-masks Hangul word tokens in English-generation mode, and avoids free-standing English letters that caused unpronounceable character soup. Korean continues to allow technical English with a reduced penalty.
- `evaluate_generation_v12.py` runs 20 new, standalone two-turn Korean/English test scenarios, compares outputs before and after training and reports surface-level English spelling/word-likeness (using `wordfreq` in evaluation only). This does not use retrieval and **does not** claim to measure semantics.
- `.github/workflows/train_neural_v12.yml` runs a model-format smoke test, downloads [the successful v11.1 workflow checkpoint](https://github.com/doubleduoYT/HGAIMeow/actions/runs/38368251370), trains the candidate, compares independent generation and uploads an artifact without overwriting a working model.

## How to run in GitHub

Branch: `feature/hgaim-v12-neural-quality`.

Workflow: `HGAIM v12 Generative Quality Experiment`. Branch pushes trigger the workflow, if Actions are enabled. GitHub Actions uses a CPU runner. Training and evaluation can take considerable time, and workflows have resource limits. Check the actual run results; do not assume a push means that a model was trained successfully.

## Results / release policy

The workflow saves the experimental files:
- `hgai_model_v12.pth`
- `hgai_model_v12.training.json`
- `V12_BASELINE_GENERATION.json`
- `V12_GENERATION_EVAL.json`
- `V12_DIALOGUE_EVAL.json`
- `V12_KO_RAW.json`
- `V12_EN_RAW.json`

A lower held-out teacher-forced loss can coexist with worse generated text: this happened in v11.1. Inspect per-prompt outputs and report failures, especially unnatural Korean text, invented English words, topical irrelevance, and missing context. A checkpoint cannot be automatically promoted based on loss or word frequencies.

**Do not automatically deploy** this candidate to the Raspberry Pi. The user's `run_hgai.py --serve` is a custom persistent-daemon build; the GitHub CLI version lacks `--serve`. Keep both prior checkpoint and daemon before testing new weights. The Pi 3B RAM limit must be considered.

## Longer-term constraints

A 4.7M-parameter transformer fine-tuned on several hundred dialogue turns will still have severe open-domain limits. Truly natural, wide-ranging standalone dialogue will need a substantially larger diverse text corpus and, likely, more model capacity and capable pretraining compute. No amount of output filtering can substitute for that.
