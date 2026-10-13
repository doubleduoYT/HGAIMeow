# HGAIM v11 Neural-First experiment

This is an **experimental branch**, not a claim of a completed v11 model.

## Goals

- Use the original v10 Preview 4 small Transformer checkpoint as initialization.
- Continue supervised fine-tuning with **no retrieved context in the training input**.
- Add Korean/English **multi-turn examples**, using previous turns inside the model context.
- Explicitly sample English question-answer pairs and reduce fact-only oversampling.
- Penalize incidental cross-language vocabulary during neural generation (the model still makes the answer).
- Assess the candidate with independent raw-neural single-turn and multi-turn tests. Search/hybrid success does not count as neural success.

## Training

The branch's `.github/workflows/train_neural_v11.yml` should start on pushes to `feature/hgaim-v11-neural-first`, provided GitHub Actions are enabled.

It downloads release `v10.1.0-bilingual-preview.4` and trains **1,200 continuation steps** on the standard GitHub-hosted CPU runner. The trained checkpoint is stored as a **GitHub Actions artifact**, not committed to the repository or automatically deployed.

Local equivalent:

```bash
pip install -r requirements.txt
python build_dataset.py --max-variants 12
python train_neural_v11.py --init-model-file hgai_model_v10_preview.pth --model-file hgai_model_v11.pth --steps 1200 --threads 2
python evaluate_neural_v11.py --model-file hgai_model_v11.pth
python evaluate_neural_v10.py --model-file hgai_model_v11.pth --min-pass 0
python evaluate_bilingual_v10.py --mode raw-neural --model-file hgai_model_v11.pth --min-pass 0
```

`train_neural_v11.py` keeps the v10 checkpoint format. It uses the existing token vocabulary/architecture where possible, and evaluates without retrieval. It does not rewrite the model architecture into a general LLM.

## Compatibility / important limitations

- **Do not replace** the Raspberry Pi's persistent `run_hgai.py --serve` with the GitHub repository's original CLI-only `run_hgai.py`. The deployed daemon is a custom variant. Install the resulting checkpoint separately and point the daemon's `--model-file` at it after testing.
- This first stage has only 61 hand-written multi-turn training turns, plus the original ~98k expanded training pairs. It cannot establish fluent open-domain reasoning or robust uncertainty handling.
- The original validation loss may contain similar paraphrase variants across train and validation. Treat it as a training signal, **not an unbiased generalization metric**.
- Automated string-match scores and language mixing diagnostics are not substitute for human review. The workflow deliberately uploads a model candidate even when qualitative results are poor; check both baseline and candidate results before deploying.
- A model that does not know a factual answer should express uncertainty, not invent facts. Reducing repetitive fallback output is not the same as eliminating honest uncertainty.
- GitHub Actions CPU time and account quotas may limit completion. See the run status and logs rather than assuming training finished.

## Next steps

1. Review the v10 and v11 per-prompt outputs in the Actions artifact.
2. Add substantially more **genuine, reviewed, varied dialogue** and held-out sessions.
3. Improve tokenizer, context length, independent split rules and model size only after checking quality and Raspberry Pi RAM.
4. If the trained candidate is better, deploy after making a backup; update the persistent socket server separately for per-channel history isolation.
