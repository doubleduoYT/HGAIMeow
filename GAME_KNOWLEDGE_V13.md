# HGAIM v13 — Game Specialist / C:U bilingual

This is a *training experiment* with provenance-tagged game Q/A, not a certified game expert or a completed checkpoint. No external retrieval is required during raw-neural generation.

## Inputs and sources
- The **English** locale `EN.json.txt` uploaded in chat was inspected, has `version: 1.0-Pre3`, and SHA-256 `3d804e7491a5fc961d6c0a9d3d63ce168ada1d70d0716b30dcb738a6e9436aae`. This SHA matches the publicly hosted `EN.json` at `Kedr2806/scavgame-locale` at setup time. GitHub Actions asserts the hash before training.
- The uploaded Korean `ko-KR(2).json` has SHA-256 `fd17679283563da844adce06e2320cc5bb8e96fddeaed2841e21425c61de3262`, and no `version` field. The public upstream `ko-KR.json` is a **different, more complete revision** (`version: 7.7.9` at review time). **GitHub Actions currently trains on the public upstream Korean, not the exact uploaded Korean file**. Do not conflate those. For exact Korean training use the user-generated JSONL dataset from the chat or supply the old Korean file to the runner.
- C:U localized items, building descriptions, status effects, layer names and layer modifiers become paired English/Korean Q/A entries using each localization's own strings.
- 20 more C:U lore Q/A facts are derived from user-supplied EN PDA notes and survivor notes. Some are explicitly framed as a document's *claim* rather than established canon.
- 95 reviewed bilingual Q/A fact cards cover Minecraft, Roblox, ADOFAI, Gunsaw, Changed, DevilConnection and many other games. Total 115 fact cards including C:U lore, each with an authored Korean and English question and answer. `DevilConnection` has **unverified title identity**, so two clarification responses are used instead of invented mechanics.
- External sources are listed for each fact in `game_facts_v13.json`; the most important official sources are 7th Beat Games, Roblox Creator Hub and the Gunsaw itch.io author page.

### Data volume and provenance
`build_games_v13.py` makes 2 or 3 different prompts for a single localized C:U description plus localized-name questions and curated game facts. These **must not be counted as independent new facts**. Questions are grouped by `cu:section:key` or `game:title:id` before an 8% hash holdout split, keeping all variants/translations of the same fact in one partition. It saves:

- `v13_game_data/game_qa_v13_train.jsonl`
- `v13_game_data/game_qa_v13_holdout.jsonl`
- `v13_game_data/REPORT_v13.json` with source SHA-256 values and group/language counts.

Source Q/A generation is deterministic, and does **not** execute retrieved information at runtime.

### Model upgrade
`train_neural_v13.py` uses the selected v12 checkpoint, extends the existing vocabulary using C:U and game words, and optionally extends RoPE context to 384 positions while keeping the Transformer layer dimensions unchanged. Existing token embeddings and layer weights are transferred by token string through `warmstart_model`. Model output remains in the `hgai-v10` checkpoint format for the existing engine.

The workflow `HGAIM v13 Game Knowledge Neural Training` runs a 2-step smoke, then 3,600 optimizer steps with gradient accumulation, independent dialogue and knowledge benchmarks, and uploads artifacts. It **does not deploy** the trained model to the Raspberry Pi.

The baseline is a roughly 4.7M-parameter model with known serious generation failures. Vocabulary expansion + more Q/A **does not by itself guarantee fluent conversation**. Inspect `V13_GENERATION_EVAL.json` and `V13_DIALOGUE_EVAL.json` for examples, not only validation loss or keyword scores. A substantially larger, well-pretrained language model would likely be needed for broad, robust open-domain answers.

### Licensing / caution
The original English game writing is copyright Orsoniks; the Korean translation is credited to community translators in the locale file. Please review the `scavgame-locale/LICENSE.md` Volunteer Translator License and attribute the translators for significant use. Do not assume public source access equals unrestricted re-licensing or commercial training rights. C:U medicine descriptions are **fictional in-game effects**, not medical advice.

### Run
[GitHub Actions](https://github.com/doubleduoYT/HGAIMeow/actions) → `HGAIM v13 Game Knowledge Neural Training`, then examine `hgaim-v13-bilingual-game-expert-...` under the successful run's Artifacts.

Local reproduction (with locally provided original files):
```bash
python build_games_v13.py --en EN.json.txt --ko 'ko-KR(2).json' --facts game_facts_v13.json --output v13_game_data
python train_neural_v13.py --model-file hgai_model_v13.pth --init-model-file hgai_model_v12.pth --steps 3600 --threads 2
python evaluate_generation_v12.py --model-file hgai_model_v13.pth --report V13_GENERATION_EVAL.json
```
