# HGAIM v16 — immediate game factuality hotfix and regression gate

**Status:** GitHub-side offline source-answer implementation and tests. No new neural weights were trained; no Pi daemon or Discord bot was modified.

## Problem actually found in v15

The 960-step Qwen3-0.6B LoRA job completed, yet its independent factual generation check fell from 3/12 rough concept passes (base) to 0/12 (adapted). Held-out teacher-forced loss went from ~3.381 to ~1.852, but this did not mean better conversational accuracy. In particular:
- OSPREY's source group `game:Casualties: Unknown:lore-osprey` was **in the 8% held-out split**, so v15 had never been supervised on that specific answer and the old benchmark described the claim as "seen" inaccurately.
- Other simple information such as Gunsaw's developer, Roblox's Luau, and SAW-03's fur was also incorrectly generated, despite being represented in the curated facts.
- The prompt said "냥체" repeatedly and the model began including the literal word in unrelated answers, so v16 keeps style formatting in the postprocessor rather than in the model's factual prompt.

## Fix implemented now

`hgaim_v16_grounded.py` has conservative intent recognition for game/source facts. It uses **local `game_facts_v13.json`** in-memory, with a provenance field for each response. This is neither an internet search nor a claim that the neural model has internalized the facts. It does NOT retrieve facts from GitHub or use a model API at runtime. No false confidence:
- A high-confidence game fact query returns the exact Korean or English reviewed fact, with `source`, `game`, and `fact_id` metadata.
- Other game fact questions without suitable evidence return a clear "not verified" response, instead of hallucinating developers, levels, dates or mechanics.
- General non-factual conversation is sent to the local-only llama.cpp client, which must be started separately. The poor v15 LoRA is NOT automatically promoted. You can test the guardrail without any neural model running.
- English game names adjacent to Korean particles (`Gunsaw라는`, `Roblox에서`, `Puro는`) are matched. Code/URL blocks remain protected by the nyang formatter; natural Korean `-야` and `-해` are improved to `-다냥` and `-한다냥` when appropriate.

**Known coverage:** high-confidence concept rules for C:U OSPREY/SAW-03/yellow blood, Gunsaw creator, Roblox scripting, ADOFAI rhythm input, Changed/Puro, Minecraft redstone, Terraria type and DevilConnection's unverified identity. Near-identical phrasing of other reviewed game cards can also be answered if an unambiguous game anchor is detected. C:U gameplay knowledge beyond these supported source cards is **not guaranteed**. The complete 4k localized v13 Q/A artifact is NOT distributed as part of this local default registry.

### No-Pi-risk offline example

```bash
python3 hgaim_v16_client.py --question "C:U의 OSPREY는 어떤 건물을 옮기려고 했어?" --metadata
python3 hgaim_v16_client.py --question "Gunsaw 개발자가 누구야?" --metadata
python3 hgaim_v16_client.py --question "Gunsaw 출시 연도가 몇 년이야?" --metadata
python3 -m unittest -v test_v14_style test_v16_grounded
```

**Expected routes:** `source`, `source`, `abstain`. On non-factual/general questions, the client uses the local loopback `llama.cpp` server; that server must run a *proper GGUF*, not a LoRA adapter, and is **not installed here**.

### Regression-gate tests

`evaluate_quality_gate_v16.py` compares the v15 baseline and tuned **identical prompts** and reports:
- factual concept heuristic values
- unsupported contradictions
- English/Hangul mixing
- whether the generated responses are even eligible for further human review

Crucially, `automatic_promote` is **always false**. This avoids making the same mistake as choosing a model solely by decreasing teacher-forced loss. The gate should detect that the existing v15 3/12 → 0/12 generation is a regression. Human semantic review is mandatory even if heuristic results improve.

CI workflow: `HGAIM v16 Offline Factuality Fix and Regression Gate`. It runs only standard-library tests, downloads the prior existing v15 evaluation artifact and produces `V16_QUALITY_GATE.json` and `V16_GROUNDED_DEMO.json`. It **does not** start costly fine-tuning on every push.

## Actual verified examples

- Q: `C:U 문서에 등장하는 OSPREY 장치는 어떤 물체를 옮기려는 목적으로 설계됐니?`
- A: `사고 기록에 따르면 OSPREY는 건물 전체를 옮길 수 있도록 설계된 순간이동 설비다냥.`
- source: `user-upload:EN.json.txt:pdaNotes.6`

The cited source above is the *user-provided C:U fictional research note*, not an independent official statement about everything in the game's future releases.

## Deployment safety

1. Do not overwrite the Raspberry Pi's custom `run_hgai.py --serve`.
2. Do not merge or deploy the v15 LoRA, which degraded actual factuality.
3. Evaluate other C:U lore, updated localization keys and general non-game conversation separately before promoting the new routing.
4. Keep source version hashes/creator and translation credits for licensed game text.
5. For the Pi 3B, an offline sourced answer does not require any heavy neural model, but the neural fallback still needs separate RAM/speed checks.

Next model-training research should use more genuinely varied original samples (not repeated templated lore + instructions), reduced learning rate, more general instruction/chat mix, and **generation-based gating** at candidate checkpoints instead of selecting the lowest synthetic-data loss alone.
