# HGAIM v16.1 — Offline game-source routing reliability

**Scope:** v16.1 is a source-route and dialogue-context reliability patch, **not** a newly trained or deployed language model. Raspberry Pi services, custom `run_hgai.py --serve`, and v15 LoRA weights remain untouched.

## Bugs reproduced and corrected

1. **C:U in Korean sentences.** The previous \b word boundaries failed to recognize an ASCII game abbreviation followed by Hangul, such as `C:U의`. Use ASCII-specific boundaries for game titles.
2. **Comparisons involving two games.** `Gunsaw와 C:U의 관계가 뭐야?` used to return no source even though the reviewed Gunsaw relationship fact exists. Now the explicit Gunsaw–C:U relationship question maps to `game_facts_v13.json` `(Gunsaw, relation)`. Other ambiguous multi-game queries remain unresolved; the source attributes this claim to a fan wiki.
3. **Short follow-up questions.** If a prior *user* message names exactly one game, a question such as `그 게임 개발자는 누구야?` can reuse the title. For a C:U question involving OSPREY or SAW-03, entity identity may also be carried into an explicit `그 장비`/similar follow-up. This requires the calling application to supply `history`; it does **not** persist histories itself.
4. **Wrong editor author.** `Gunsaw 레벨 에디터 만든 사람 누구야?` previously returned the *game* developer, which was unsupported evidence for the *editor* creator. The query now abstains instead of asserting an identity.
5. **Unsupported personal metadata.** Developer ages and birthdays, unknown prices and unverified release dates no longer route to an unrelated game-developer or generic description.
6. **Open-ended requests.** `Minecraft에서 뭐 만들면 좋을까?` is creative advice, not a fact lookup, and is passed to the configured local neural service rather than returned as an unsupported-lore warning.
7. **Changed is a verb too.** `I changed my settings` is not a question about DragonSnow's game, while `What is Changed?` and `Who is Puro?` are.
8. **Fragile working directories.** The default reviewed fact JSON now resolves relative to the module file, not the shell working directory or systemd's current working directory.
9. **Korean name variants.** `Gunsaw라는`, `Roblox에서`, `OSPREY라는`, `만든`, `옮겨` are covered by new tests and source rules.

## Coverage and boundaries

`hgaim_v16_client.py` is a *local-only* hybrid:
- `route=source`: exact answer from the reviewed `game_facts_v13.json` entry, accompanied by its source, game, and fact ID. No neural inference is performed. This is not proof that the neural model memorized the fact.
- `route=abstain`: a factual question about a recognized game does not have an adequate reviewed source.
- `route=neural`: creative/general conversation is forwarded to a separately running loopback-only llama.cpp service. **No such service was installed or launched by this PR.**

Source coverage is currently the existing reviewed game cards (~115 facts), not the entire C:U localization corpus. The developer should add more author-reviewed local cards and targeted disambiguation tests before expanding coverage. Recent game changes require manually updated sources.

Language style continues to be handled by `nyang_style_v14.py`. It can make text sound catlike but cannot repair a false answer.

## Reproduction

```bash
python3 -m unittest -v test_v14_style test_v16_grounded test_v16_1_routing
python3 hgaim_v16_client.py --question "Gunsaw와 C:U의 관계가 뭐야?" --metadata
python3 hgaim_v16_client.py --question "Gunsaw 출시일은 언제야?" --metadata
```

CI workflow `HGAIM v16.1 Source Routing Reliability` on this feature branch verifies 29 unit tests and emits `V16_1_ROUTING_DEMO.json` with five example source/abstain routes. Successful run: https://github.com/doubleduoYT/HGAIMeow/actions/runs/38481347294

## Release policy

- No automatic merge to `main`, production Pi install, daemon restart or Discord bot modification.
- Do not promote v15's 960-step LoRA: its separate generated-game-answer diagnostics regressed.
- After deployment approval, integrate **explicitly** with the user's existing Unix-socket protocol and per-user Discord history, preserve the customized `run_hgai.py --serve`, and test performance under the Pi 3B's 1GB RAM constraint.
- The hourly improvement automation should not repeatedly retrain just because of a lower teacher-forced loss or because this source-router test succeeded.
