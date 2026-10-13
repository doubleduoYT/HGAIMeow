# HGAIM v15 — Qwen3.0-0.6B: better bilingual language training and game factuality

**Status: experimental.** The model must pass real generation checks; a lower token loss is insufficient to claim an expert C:U chatbot.

## Why a new approach?

The v14.1 Qwen2.5-0.5B adapter used 953 records and 720 one-example steps. Its held-out token loss improved (2.755 to 2.090) but it **hallucinated Gunsaw's developer, C:U OSPREY's purpose, Roblox's scripting language, and ADOFAI's mechanics**. These are unacceptable answers for a game-knowledge bot. We explicitly do **not** deploy v14.1.

v15 tests a *new* base pretrained model rather than extrapolating its failed factuality:
- **Qwen/Qwen3-0.6B** multilingual weights, Apache 2.0. This is an alternative architecture, not a direct numeric upgrade of Qwen2.5 with an interchangeable LoRA adapter. The v14.1 adapter is not compatible and cannot be loaded onto Qwen3.
- For inference and fine-tuning, `enable_thinking=False` disables hidden reasoning text, improving resource suitability for the Raspberry Pi target and allowing direct assistant-answer supervision.
- Qwen3 needs `transformers >= 4.51` and up-to-date `llama.cpp` for a GGUF deployment.
- A compatible Q4_K_M GGUF is expected around **380–480MB depending on quantization build**, but actual 1GB Raspberry Pi 3B RAM and tokens/second still need measurement. This experiment does not alter live daemons.

## Crucial multi-turn training bug fixed

v14/v14.1 tokenization masked the system and user tokens correctly, but **only the final assistant turn** had loss. When multiple user/assistant turns were present, preceding assistant responses were prompt context only.

`train_lora_v15.py` now uses **Qwen3's fully rendered chat template** and fast tokenizer character offsets to compute a token-loss mask for *every assistant reply*, while ignoring user text. The workflow contains a preflight assertion that tokens for the first **and** second assistant answer are supervised, and that the intervening user message is not. On real Qwen3, this must pass before any expensive fine-tuning starts.

## Bilingual game knowledge corpus

The training job reuses the **exact v13 C:U localized sources and reviewed game fact cards** (Minecraft, Roblox, ADOFAI, Gunsaw, Changed, DevilConnection and other games) from the successfully built v13 artifact.

`prepare_sft_v15.py`:
- Preserves v14's 953 records including ordinary conversations and locale-derived C:U knowledge.
- Adds one or two prompts per reviewed fact and language, with source-group IDs, but **only if that fact's group belongs to v13's training partition**.
- Does **not** leak previously held-out facts into training. The source-group disjointness is asserted and reported.
- Maintains honest uncertainty about DevilConnection's identity rather than inventing a plot or developer.
- Converts factual answers into natural conversational nyang/meow format using `nyang_style_v14.py`.

This is source-based SFT. It does not search or paste retrieval results into generation-time prompts.

## Training

- ~0.6B base model. The base weights are frozen.
- New rank-8 LoRA on q/k/v/o attention projections (not compatible with the v14 Qwen2.5 adapter).
- Up to 960 CPU training steps on GitHub Actions, maximum sample input 384 tokens.
- Sources in each training step: **35% C:U**, **33% other games**, **32% general/multi-turn chat**.
- Every evaluation uses a deterministic, stratified English/Korean subset of the original group-held-out examples. No examples from that holdout are used for gradients.
- Save best adapter by held-out token loss, but require separate review of factual answers before approving deployment.

## Evaluation

`evaluate_v15.py` uses *new wordings* of game questions not simply copied from the game-facts JSON, and captures the **raw** generated answer and the **nyang-styled** answer. It checks for broad required concepts and a few known hallucinated claims. This **is not an automatic claim of correctness**: a false answer can include all the keywords. Inspect each generated explanation.

Separate general-dialogue questions expose catastrophic forgetting, while game test questions include source-linked SAW-03, OSPREY, yellow blood, Gunsaw, Changed, ADOFAI, Roblox, Minecraft and DevilConnection. For C:U, in-world claims must be attributed to the actual document/character rather than presented as universal canon.

## Running

Open the Actions workflow `HGAIM v15 Qwen3 Bilingual Factuality and Multi-turn` on branch `feature/hgaim-v15-qwen3-grounded`. Once it succeeds, open `hgaim-v15-qwen3-bilingual-...` under **Artifacts** for the LoRA adapter and bilingual before/after logs.

The model is **not** integrated with the Raspberry Pi's `hgaim.service`, `bot.service`, `botpro.service`. The Pi uses a customized persistent `run_hgai.py --serve` that must not be overwritten. Only after independent qualitative evaluation should an adapter be merged into the correct Qwen3 base, converted and quantized to a Qwen3 GGUF, tested for memory/speed, and bridged to the existing socket protocol.

### Explicit limitations
- Extra steps cannot create knowledge not present in validated source data.
- Source-localized game strings include fictional descriptions that are not always real-world facts; keep provenance.
- The older uploaded Korean file differs from the current public locale used for the v13 artifact; do not misrepresent which revision trained the model.
- Please confirm original game text and translated text rights, licensing and credits before redistributing trained game-specific assets.
