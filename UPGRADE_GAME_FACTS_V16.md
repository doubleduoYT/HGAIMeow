# HGAIM v10.2 — reviewed v16.1 game facts backport

This release **keeps the deployed v10 model/checkpoint and all existing CLI signatures**. Only small offline source-response modules and a short wrapper in `hgai_core.py` were added. No v14/v15/Qwen weights are merged or downloaded.

## Available now

- Source-backed `Casualties: Unknown` facts (OSPREY, SAW-03, yellow blood etc.), Gunsaw developer, Roblox's Luau, Changed/Puro, Minecraft redstone and the other reviewed Q/A cards in `game_facts_v13.json`.
- Source-preserving replies in Korean/English, with cat-speech style.
- Multi-turn unambiguous game follow-ups from the existing v10 `HGAIEngine.history` list. The engine can still answer ordinary questions through its original rule/calculation/retrieval/Transformer pipeline.
- Clear refusal when a game has no supported developer age/release date, rather than confidently guessing one.
- The original **raw-neural mode stays free of retrieval**, allowing future honest independent model evaluation.
- 16 independent offline stable-v10 regression tests including two actual `run_hgai.py --once` smoke tests.

## Important compatibility details

`model_info().version` continues to return `hgai-v10` because **the neural architecture and checkpoint file format did not change**. The game-fact enhancement is a runtime engine backport, not a new trained 0.6B model.

There are no new pip dependencies and no runtime access to the Internet. Game fact data are read from the module directory, independent of the current working directory.

All previously verified game facts retain their `source` metadata in `hgaim_v16_grounded.py`. **These local reviewed answers do not prove that the v10 neural model memorized the facts.**

## Raspberry Pi 3B update procedure — manual and backup-first

**No Raspberry Pi or live Discord bot was remotely modified by this GitHub release.**

The user's Pi uses a CUSTOM `run_hgai.py --serve` and persistent Unix socket, which does not match the regular GitHub CLI file. **Never `git reset --hard` or blindly copy GitHub's `run_hgai.py` over the Pi copy.** Back up first.

From Termux, run:

```bash
ssh doubleduo@100.77.196.60
```

Then, on the Pi:

```bash
set -e
BASE="$HOME/bot/hgaimeow-v10-preview"
test -d "$BASE" || { echo "Check the Pi engine path before proceeding"; exit 1; }
test -f "$BASE/run_hgai.py" || { echo "Custom Pi --serve entrypoint missing"; exit 1; }
BACKUP="$HOME/hgaim-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP" "$BASE/.v16_stage"
cp -a "$BASE/hgai_core.py" "$BASE/run_hgai.py" "$BACKUP/"
FILES="hgai_core.py hgaim_v16_context.py hgaim_v16_grounded.py nyang_style_v14.py game_facts_v13.json"
for f in $FILES; do
  test ! -f "$BASE/$f" || cp -a "$BASE/$f" "$BACKUP/"
  curl -fLsS --retry 3 "https://raw.githubusercontent.com/doubleduoYT/HGAIMeow/main/$f" -o "$BASE/.v16_stage/$f"
done
for f in $FILES; do
  test -s "$BASE/.v16_stage/$f"
done
python3 -m py_compile "$BASE/.v16_stage/"*.py
for f in $FILES; do cp -a "$BASE/.v16_stage/$f" "$BASE/$f"; done
cd "$BASE"
python3 -m py_compile hgai_core.py hgaim_v16_context.py hgaim_v16_grounded.py nyang_style_v14.py
python3 - <<'PY'
from hgai_core import HGAIEngine
h = HGAIEngine(load_model=False)
assert 'Orsoniks' in h.reply('Gunsaw 개발자 누구야?')
assert '건물 전체' in h.reply('C:U의 OSPREY는 뭘 옮겨?')
print('Source-based v16.1 update smoke test passed')
PY
echo "Backup directory: $BACKUP"
echo "Custom run_hgai.py untouched."
```

Only after confirming the tests and the Pi's customized service setup, restart the existing `hgaim.service` manually (if the user has appropriate sudo permission):

```bash
sudo systemctl restart hgaim.service
sudo systemctl status hgaim.service --no-pager
```

If the service fails to start, restore `hgai_core.py` and the previous game modules from `$BACKUP` and restart the service. The backup includes `run_hgai.py` for safety but the installation deliberately does **not** replace that customized file.

## Source and rights

Game source material is an experimental paraphrased FAQ and retains attribution links in `game_facts_v13.json`. The game descriptions and translations are credited to their respective authors/contributors, and inclusion in the open repo does not change their underlying licensing. The `DevilConnection` title is still unverified and is not given fabricated lore.

## CI

Workflow: [HGAIM v10.2 Stable Game Facts Backport](https://github.com/doubleduoYT/HGAIMeow/actions).

The production Raspberry Pi has only 1GB of RAM; this runtime improvement adds no heavyweight language model or extra memory-intensive model loader.
