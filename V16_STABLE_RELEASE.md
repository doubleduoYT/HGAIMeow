# HGAIM v16.1 Stable — game-source answer backport

### Compatibility
This is a patch to the **v10 runtime engine**, not a promotion of v15 Qwen3 or a replacement of any Raspberry Pi weights. It adds an offline reviewed fact registry and routes recognized factual questions before fuzzy retrieval.

**Existing files changed**
- `hgai_core.py`: routes supported factual questions and follow-up context in hybrid/search modes; optional unknown-fact abstention for unsupported precise metadata; applies cat endings to user-facing answers, leaving `raw-neural` unchanged.
- `README.md`: usage.

**New files**: `game_facts_v13.json`, `nyang_style_v14.py`, `hgaim_v16_grounded.py`, `hgaim_v16_client.py`, the local-only Qwen test clients (optional, not invoked by v10), and source-routing regression tests.

### Coverage
C:U lore includes source-linked research notes (e.g. OSPREY transports entire buildings; SAW-03 and yellow blood), plus reviewed cards for Gunsaw, Roblox, Minecraft, ADOFAI, Changed, DevilConnection and others. Unsupported claims are not filled by invented details. The more extensive C:U localized dataset is **not** silently loaded into this runtime; it still needs independent review and source alignment.

A sourced response is not evidence of neural learning. Running the unmodified standalone `raw-neural` mode will still reveal baseline weights' limitations.

### Verification
```sh
python3 -m unittest -v test_v14_style test_v16_grounded test_v16_1_routing test_v16_stable_integration
python3 run_hgai.py --once "Gunsaw 개발자는 누구야?"
python3 run_hgai.py --once "C:U의 OSPREY는 뭘 옮겨?"
python3 run_hgai.py --once "Who developed Gunsaw?"
```

### Raspberry Pi 3B deployment safety

Do **not** directly overwrite the custom Raspberry Pi installation. Its
`/home/doubleduo/bot/hgaimeow-v10-preview/run_hgai.py` has a local persistent
`--serve` Unix-socket implementation that is **not present in GitHub's CLI file**.

1. Backup the live directory and model checkpoint.
2. Test the updated Python modules and JSON in a disposable copy.
3. Keep the customized `run_hgai.py` and existing model weights unchanged.
4. Test source Q&A and low-memory performance with the same service account.
5. Only if approved, switch the live engine and check `hgaim.service`, `bot.service`, `botpro.service` logs. This PR **does not** issue SSH commands or restart services.

No Qwen server, external API, internet search, hosted model, or additional runtime pip dependencies are required.

### Limitations
- Source registry entries are short reviewed Q&A records; not every question in the game locale or every game mechanic has an entry.
- In general chat, the original v10 small model/retriever still determines answer quality.
- `raw-neural` responses will still be weak; the fix targets the stable hybrid experience.
- Style postprocessing is a presentation layer, not a substitute for semantic accuracy.
