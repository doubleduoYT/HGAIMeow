# HGAIM v14: pretrained bilingual model + reliable nyang speech

**Status: experimental; no claim of training success, quality improvement, or Raspberry Pi deployment before reviewing Actions reports.**

## Why v13 ignored the cat-speech rule

1. `hgai_core.fix_nyang` cleaned whitespace and repeated `냥`, but **never appended `냥` to an ordinary sentence**. The model was free to return plain text in every mode, especially `raw-neural`.
2. C:U game localization Q/A entries used unstyled neutral text, repeatedly teaching the network factual source explanations without the cat voice.
3. The 6.74M parameter scratch-trained model consistently failed both Korean and English raw neural generation despite lower teacher-forced loss. Prompt-only style instructions cannot guarantee a tiny model will obey.

v14 adds `nyang_style_v14.py`, a deterministic postprocessor for natural-language clauses (preserving fenced/inline code). The **legacy** `HGAIEngine.reply` now uses that postprocessor in all modes, including search and fallback, while the **new** `hgaim_v14_client.py` uses it on local model responses.

This guarantees a cat-like clause ending for many ordinary Korean statements, not semantic correctness. Some Korean endings will still sound unusual until the pretrained model produces naturally phrased text. English responses use a light `meow`, without Korean mixing.

## New language model

- Base: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache-2.0, 0.49 billion parameters, pre-trained and instruction-tuned, Korean and English support).
- Training: CPU-bounded LoRA r8 on q/v attention projections. The **pre-trained model remains frozen**, only the LoRA weights are trained.
- Data: generated from v13's existing C:U localized game Q/A artifacts and 29 other games, plus existing HGAIM conversations. Natural-language outputs are converted to cat speech. One prompt per (source fact, language) reduces pseudo-data duplication. Original localization sentences remain credited to their authors. Train-vs-holdout game groups are inherited from v13, so no facts leak across splits.
- Benchmark: raw base model vs. LoRA with prompts and output **before and after** cat-speech processing. Loss, fixed fallback rate, and string similarity cannot by themselves show natural conversation improvement.

## GitHub Actions

`HGAIM v14 Pretrained Bilingual LoRA & Nyang Fix` on branch `feature/hgaim-v14-pretrained-nyang`:

1. Run dependency-free style, code-block and local-network safety tests.
2. Download v13 game Q/A from the successful v13 workflow run, generate balanced SFT data.
3. Install PyTorch + Transformers + PEFT from PyPI.
4. Load the actual pretrained Qwen model, execute **one training step** as a smoke test, then 48 LoRA fine-tuning steps on the GitHub CPU runner.
5. Independently generate Korean and English answers before/after training, save model adapter + reports as an Actions artifact.

48 steps is a **small first experiment** appropriate for CPU Actions quotas, not full fine-tuning of every fact. If the adapter is not better, use the original pretrained model rather than forcing the adapter into production.

**Important:** A LoRA adapter is **not** a GGUF model. The Pi cannot use `hgaim_v14_lora/` with llama.cpp without merging it into Qwen model weights and converting/quantizing to GGUF. Do not rename or feed the adapter as a `.gguf` file.

## Raspberry Pi 3B (1GB) deployment plan — NOT YET EXECUTED

The standalone 0.5B base model has community GGUF quantizations around 350MB (`Q4_0_4_4` for ARM) or 400MB (`Q4_K_M` generic). These are smaller than loading full FP16 Qwen, but 1GB Pi RAM, slow Cortex-A53 CPU and two live bots make inference tight. Test it **separately** from the existing HGAIM daemon with a low context window, one request at a time. Running another heavyweight PyTorch model on the Pi is not recommended.

On a compatible 64-bit Pi OS with `llama.cpp` compiled to expose `llama-server` (verify the system architecture first), an illustrative manual command is:

```bash
./llama-server -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -c 384 -t 2 --parallel 1 --host 127.0.0.1 --port 8086
python3 hgaim_v14_client.py --question "C:U가 뭐야?" --max-tokens 72
```

Install/download the GGUF from its upstream release using its exact name and license. Never expose the local HTTP endpoint to the network; the v14 client rejects any remote/non-loopback URL.

**No existing** `bot.service`, `botpro.service` or `hgaim.service` **was changed**. The Raspberry Pi's persistent `run_hgai.py --serve` is a custom edition, and the GitHub CLI-only original must not replace it. After semantic review, create a **separate, versioned local backend bridge** that keeps the existing Unix socket protocol and per-user conversation contexts.

## Copyright and data quality

The original C:U game strings are the intellectual property of their creators, and Korean translations have attribution/license conditions. Access to public source files does not grant blanket rights for unrestricted redistribution or commercial model training. Confirm relevant licensing and keep noncommercial fan-project and author credits as appropriate. Generated source-linked Q/A are *about* documented fictional effects, not real medical advice.

C:U plot facts derived from in-world notes should be explicitly attributed as that character's or document's claims. `DevilConnection` identity is not independently verified, so its data intentionally avoids fabricated game details.
