# Fine-tuning Gemma for the MagickHome kitchen designer

This folder teaches **Gemma 4 E4B** the MagickHome way of designing kitchens, so the knowledge lives in the
model's weights instead of being sent in every prompt. Everything about the approach, the training data, the
method and how to run and retrain it is in this one file.

---

## 1. Why fine-tune

Today every design request sends the whole MagickHome knowledge pack to the model:

| Part of the prompt | Size | Fine-tuned model |
|---|---|---|
| Designer playbook + worked examples (`knowledge/`) | ~3,000 tokens | **dropped** - learned in training |
| Cabinet menu with a designer note per option | ~1,200 tokens | kept, **without** the notes |
| Room facts with rule hints ("hob centre range …") | ~600 tokens | kept, **without** the hints |
| Customer needs | ~300 tokens | kept |
| **Total** | **~6,600 tokens** | **~1,700 tokens** |

A shorter prompt means faster designs (especially the first design after the model loads, which re-reads the
whole prompt) and no per-request cost for knowledge the model already has.

**What does not change:** the answer is still locked to a JSON schema (the model can only pick real cabinet codes
and real wall names), and the deterministic **rule engine still checks and corrects every design**. Fine-tuning
changes how good the model's choices are, never whether the rules are enforced.

---

## 2. The approach in one picture

```
                          ┌──────────────────────── TRAINING DATA ────────────────────────┐
 random rooms + briefs ─► │ 1. rules teacher   (rules_dataset.py)   - available now       │
                          │ 2. LLM teacher     (build_dataset.py)   - Gemma / Gemini with  │
                          │                                           the FULL knowledge   │
 web-UI designs ────────► │ 3. harvested       (harvest_designs.py)                        │
 designer samples ──────► │ 4. samples         (samples/*.json, 3× weight)                 │
                          └────────────────────────────┬───────────────────────────────────┘
                                                       │  each example = SHORT prompt → ideal answer
                                                       ▼
                               quality gate: replay the answer through the real rule engine;
                               keep only 0 corrections, 0 issues, scorecard ≥ 90
                                                       ▼
                               merge.py → train.jsonl (90 %) + val.jsonl (10 %)
                                                       ▼
                 Colab T4: QLoRA fine-tune of unsloth/gemma-4-E4B-it (Unsloth, SFT, answers only)
                                                       ▼
                       merge LoRA → GGUF q4_k_m → Ollama "magick-gemma" (Colab tunnel or local)
                                                       ▼
          app: GEMMA_MODEL=magick-gemma, GEMMA_USE_KNOWLEDGE=0 → short prompt → schema-locked answer
                                                       ▼
                                   rule engine (unchanged) → 3D scene
```

The idea is **distillation**: a "teacher" that has the full knowledge (the rule engine + playbook, or an LLM
given the full prompt) produces correct designs; the "student" (fine-tuned Gemma) learns to produce the same
answers from the short prompt alone.

---

## 3. The training data

### 3.1 What one example looks like

Each line of `data/*.jsonl` is one example: three chat messages.

| Message | Content | Size |
|---|---|---|
| `system` | The plain designer role (`PLAIN_SYSTEM_PROMPT` in `ai_designer.py`) - no playbook | ~870 chars |
| `user` | Cabinet menu (codes only), room facts (no hints), customer needs, the answer-field list | ~6,000 chars |
| `assistant` | The ideal answer: JSON in the exact answer format | ~2,500 chars |

The system and user messages are produced by the **same function the app uses at run time**
(`ai_designer.build_messages(..., knowledge=False)`), so training and production prompts are identical.

Example user message (shortened, real data):

```
CABINET LIST - choose by CODE (code | width x depth | catalogue type):
SINK UNITS:
  SINK-900 | 900 x 560 mm | Sink Double Shutter
  SINK-1050 | 1050 x 560 mm | Sink Double Shutter
  ...
ROOM (mm measured along each wall from its start; walls listed in order around the room):
- Wall 1: 3500 mm, from the Wall 4 corner to the Wall 2 corner. Solid. Free run 50-3450 (3400 mm).
- Wall 2: 5400 mm, from the Wall 1 corner to the Wall 3 corner. Solid. Free run 50-5350 (5300 mm).
- Wall 3: 3500 mm, ... DOOR 350-1100 (near the Wall 2 corner). Free run 1200-3450 (2250 mm).
- Wall 4: 5400 mm, ... WINDOW 3900-5100 (centre 4500). Free run 50-5350 (5300 mm).
- Floor area 194 sqft.
CUSTOMER NEEDS (stated needs are fixed; 'not stated' means you decide and say why):
- Gas: uses an LPG cylinder - needs a cylinder unit.
- Base cabinets: prefers shutter base cabinets.
...
```

Example answer (shortened, real data):

```json
{
  "room_reading": "Window on Wall 4 (3900-5100 mm); door on Wall 3 near the Wall 2 corner; long solid walls: Wall 1, Wall 2.",
  "design_plan": "Wet zone: sink on Wall 4 under the window. Cooking: hob on Wall 2 with the chimney above. ...",
  "layout_why": "Large U-Shape kitchen (194 sqft): Wall 2 + Wall 3 + Wall 4 - the window wall Wall 4 anchors the run; aisle 2280 mm (needs 1200+)",
  "cabinet_walls": ["Wall 2", "Wall 3", "Wall 4"],
  "sink_why": "Centred under the Wall 4 window for light and air; ...",
  "sink_wall": "Wall 4", "sink": "SINK-800",
  "hob_why": "Wall 2 is a solid wall, centred at 2300 mm with counter both sides. Rule R69: 900 mm Triple Drawer - A unit.",
  "hob_wall": "Wall 2", "hob": "HOB-900-3DRAWER", "hob_center_mm": 2300,
  "lpg_why": "The customer uses LPG: the cylinder unit sits beside the hob.", "lpg_cylinder": "LPG-600",
  "tall_units": [{"wall": "Wall 3", "unit": "PANTRY-600"}],
  "base_storage": "shutters",
  "...": "purifier, display, open unit, feature units, wall-cabinet sizes, wall doors, finishes, lofts"
}
```

Every decision has a `*_why` written **before** it - the model states its reason, then commits to the option.

### 3.2 The four data sources

| Source | Script | Teacher | Status | Strength | Weakness |
|---|---|---|---|---|---|
| **Rules** | `rules_dataset.py` | Quick-rules designer + rule book + rule engine | **726 examples (used in the first training)** | Fast (~0.5 s/room, offline), 100 % rule-compliant, unlimited | Templated reasons; only as smart as the rules designer |
| **LLM teacher** | `build_dataset.py` | Gemma / Gemini given the full knowledge pack | ready, not yet run | Real design judgement and natural reasons | Needs a GPU / API; ~45 s per room on Colab |
| **Harvested** | `harvest_designs.py` | Past AI designs made in the web UI | 6 examples | Real rooms | Few; older ones don't match today's answer format (rejected automatically) |
| **Samples** | `add_sample.py`, `samples/` | MagickHome designers by hand | 0 so far | Best quality - how *your* designers design | Slow to make |

#### How the rules examples are made (`rules_dataset.py`)
1. Generate a random room and customer brief (`rooms.py`, see 3.3).
2. Run the full pipeline with the rule-based designer: proposal → rule book module sizes → rule engine layout.
3. Read the **final, checked layout** (where the sink, hob, tall units, wall cabinets, lofts actually ended up)
   and write it in the answer format. Reasons come from the playbook and the rule book (rule numbers are cited,
   e.g. "Rule R69: 900 mm Triple Drawer - A unit").
4. **Replay**: feed that answer back through the real pipeline exactly as if the model had given it.
5. Keep it only if the quality gate passes (3.4).

#### How the LLM-teacher examples are made (`build_dataset.py`)
Random room → Gemma (on Colab) or Gemini designs it with the **full** knowledge prompt → the same quality gate →
the example stores the **short** prompt with the teacher's answer. The student learns to match a teacher that had
far more context than the student will get.

### 3.3 The generated rooms and briefs (`rooms.py`)

| Property | Range |
|---|---|
| Room shape | Rectangle (4 walls), built from the production sample scene |
| Wall lengths | 2.0-5.5 m and 1.8-5.0 m (50 mm steps) |
| Ceiling | 2.4-3.0 m (decides lofts and wall/tall heights) |
| Windows | 0 (10 %), 1 (72 %), 2 (18 %); 600-1800 mm wide, any wall |
| Doors | 1 (82 %), 2 (18 %); 750-1000 mm; 40 % placed near a corner |
| Brief | Mostly "not stated" (the designer decides), with stated needs mixed in: door style, layout (auto / single / L / U), tall units, fuel (LPG / piped / induction), water purifier, built-in appliances, base storage style, cooking, household size, lofts, budget, glass display, crockery unit |

Rooms are reproducible: the same `--seed` and index always give the same room.

### 3.4 The quality gate

An answer becomes training data only if, when run through the real pipeline:

| Check | Bar | Setting |
|---|---|---|
| Rule-engine corrections (sink moved, hob moved, units dropped …) | **0** | `FINETUNE_MAX_CORRECTIONS` |
| Layout validation issues (overlaps, collisions, corner rule …) | **0** | - |
| Designer scorecard (`designer_benchmark.score_design`, 14 checks) | **≥ 90 %** | `FINETUNE_MIN_SCORE` |
| Answer matches today's answer format (schema) | must validate | - |

The scorecard checks things like: sink on the window wall, hob on a solid wall, sink and hob apart, hob clear of
the door, sink size suits the wall, drawer hob unit, tall unit well placed, walls connected, feature units zoned.

### 3.5 First dataset - numbers

| | |
|---|---|
| Rooms tried | 1,100 (seed 101) |
| Kept | **726** (66 %) + 6 harvested = 732 → 731 after removing 1 duplicate |
| Rejected | 374 - most for rule corrections (142), schema mismatches such as "tall unit wanted but no room for one" (117), feature zoning (74), tall placement (60), hob on a window wall (57) |
| Scorecard of kept examples | 100 % for 550, 90-93 % for the rest |
| Sinks in the answers | SINK-800 248, SINK-900 183, SINK-1200 176, SINK-1050 119 |
| Tokens per example | ~3,300 (Gemma tokenizer), max ~3,450 |

### 3.6 Train / validation split

`merge.py` splits **90 % train / 10 % validation** → **660 / 71**.

- **Why 90/10:** with a few hundred examples, every training example counts; 71 validation examples × ~1,200
  answer tokens is ~85,000 scored tokens - plenty for a stable validation loss. The real quality test is the
  benchmark (below), not validation.
- **Stable split:** by a hash of each example's content, so an example never moves between train and validation
  when data is added later - results stay comparable across re-trains.
- **Duplicates** are removed before splitting.
- **Designer samples always go to training**, repeated 3× (`--sample-weight`).
- **Test set:** the 8 hand-built rooms of `designer_benchmark.py`. They are not among the random training rooms.

Change it with `python finetune/merge.py --val-share 0.05` (95/5 - sensible once there are 2,000+ examples).

### 3.7 Known data-quality issues

Dataset v1 had three wording bugs in reasons (the design decisions were correct): a misleading sink reason when
the engine shrank the sink, drawers named in `features_why` without a drawer unit, and "Storage: None" in the plan
text. The generator is fixed and the data regenerated (v2, 0 occurrences). See `TRAINING_DATA.md` section 7 for
the remaining limitations (one teacher, templated reasons, never-seen choices).

---

## 4. The fine-tuning method

**Supervised fine-tuning (SFT) with QLoRA** (4-bit LoRA), using [Unsloth](https://github.com/unslothai/unsloth)
on a free Google Colab T4 GPU.

| Part | Setting | Why |
|---|---|---|
| Base model | `unsloth/gemma-4-E4B-it` (instruction-tuned Gemma 4 E4B, ~8 B parameters, Apache-2.0, not gated) | Same family as the `gemma4:E4B` the app already runs |
| Loading | 4-bit quantized (the "Q" in QLoRA) | ~16 GB → ~5 GB, fits the 15 GB T4 |
| Adapter | LoRA rank 16, alpha 16, dropout 0; attention + MLP layers of the language model; vision layers frozen | Only **34.9 M of 7.98 B parameters (0.44 %)** are trained; the original weights stay frozen |
| Loss | **Answers only** (`train_on_responses_only`) - the prompt tokens are masked | The model learns to write designs, not to memorise menus and room descriptions |
| Chat format | Gemma's own chat template (`<\|turn>user` … `<\|turn>model`); markers detected automatically | Matches how Ollama formats prompts at run time |
| Packing | On - several examples per 4,096-token row, no attention across examples | Better GPU use |
| Optimizer | AdamW 8-bit, learning rate 2e-4, linear schedule, 5 warm-up steps, weight decay 0.01 | Unsloth's standard LoRA recipe |
| Batch | 1 × 8 gradient accumulation = 8 examples per step | Fits T4 memory |
| Epochs | 1 (first, rules-only dataset: avoids memorising templated reasons); 2-3 once most data is teacher / designer data | `EPOCHS` in the notebook |
| Precision | float32 compute | The T4 has no bfloat16 and Gemma 4 does not work in float16 - slower but stable |
| Max length | 4,096 tokens | Short prompt ~1,700 + answer ~1,200 (word-level); ~3,300 Gemma tokens in total |

First run: 660 examples, **83 steps, ~1 h 36 min** on the T4, plus ~20 min for the check and export.

**Not used (and why):** full fine-tuning (needs far more GPU memory); RLHF / DPO preference training (needs pairs
of "better / worse" designs - possible later from designers' accept / reject decisions).

### After training
1. **Quick check:** generate answers for 3 validation rooms without the schema lock and report whether they are
   valid JSON.
2. **Export:** merge the LoRA into the model and save **GGUF q4_k_m** (~5 GB) - the format Ollama runs.
3. **Serve:** `ollama create magick-gemma` with Ollama's built-in Gemma 4 prompt renderer and output parser
   (`RENDERER gemma4`, `PARSER gemma4`, temperature 0), behind the same password proxy + Cloudflare tunnel as
   the other Colab notebooks.

---

## 5. Running it

### 5.1 Make the data (laptop)
```bash
cd aidesignernew
# rules examples - offline, ~8 min for 1,100 rooms
MAGICKHOME_ASSETS_API_URL= python3 finetune/rules_dataset.py --n 1100 --seed 101
# optional: LLM-teacher examples (teacher Gemma running on Colab - see colab/README.md)
OLLAMA_HOST=https://<tunnel> OLLAMA_API_KEY=<password> python3 finetune/build_dataset.py --n 300 --teacher gemma
# optional: good designs from the web UI
python3 finetune/harvest_designs.py
# build train / val
python3 finetune/merge.py
git add finetune/data/train.jsonl finetune/data/val.jsonl && git commit -m "Fine-tune data" && git push
```
All generators are resumable - stop and re-run the same command; finished rooms are skipped.

### 5.2 Train (Colab, free T4)
Open
`https://colab.research.google.com/github/thenushans-magicktech/qwen2.5/blob/gemma/finetune/Gemma_finetune_on_Colab.ipynb`
→ Runtime → Change runtime type → **T4 GPU** → **Run all**. It reads `train.jsonl` / `val.jsonl` from the
`gemma` branch (or set `UPLOAD_INSTEAD = True` to upload them).

Already have a connected Colab session (e.g. the Gemma serving notebook)? Run the fine-tune there in one cell:
```python
!git clone -q --depth 1 -b gemma https://github.com/thenushans-magicktech/qwen2.5.git /content/ft
exec(open("/content/ft/finetune/colab_run.py").read())
```

Keep the tab open (answer Colab's "are you still there?" prompt). With `SAVE_TO_DRIVE = True` the model is copied to
Google Drive; the next session can set `SKIP_TRAINING = True` and serve it in minutes.

### 5.3 Use it
**On the Colab GPU** (fast, ~30-45 s per design) - the notebook's last cell prints:
```bash
python3 colab/connect_colab.py https://<tunnel>.trycloudflare.com <password> --finetuned
```
**On this laptop** - download the `.gguf` from Colab (Files panel, `gguf_out/`), then:
```bash
bash finetune/install_local.sh ~/Downloads/<model>.gguf          # creates Ollama model "magick-gemma"
GEMMA_MODEL=magick-gemma GEMMA_USE_KNOWLEDGE=0 MAGICKHOME_AI_PROVIDER=gemma python3 main.py --serve
```
Training on this laptop is not practical (no GPU, 14 GB RAM: the model does not fit for training, and a CPU would
need days); running the trained model locally is fine.

### 5.4 Check it
Compare the fine-tuned model with the full-prompt Gemma on the 8 benchmark rooms:
```bash
GEMMA_MODEL=magick-gemma GEMMA_USE_KNOWLEDGE=0 python3 designer_benchmark.py --designer gemma --versions v2
```
Reference: full-prompt Gemma 4 E4B scored ~97.6 % (3 rooms, 1 rule fix on average). Use the fine-tuned model
day to day only when it matches.

---

## 6. Retraining later (new rules, new designs)

1. **Rules or playbook changed** (`knowledge/`, rule workbook): regenerate with the teachers - they read the new
   knowledge. After a big change, move the old `data/*.jsonl` away first so old answers don't teach old rules.
2. **Designer samples**: add them to `samples/` (see `samples/README.md`); they count 3× in training.
3. `merge.py` → push → Run all on Colab with `SKIP_TRAINING = False`.
4. Benchmark again before switching.

Until a re-train is done, the app can keep using the full-prompt Gemma (`--designer gemma`, `GEMMA_USE_KNOWLEDGE=1`),
which picks up knowledge changes immediately.

**Recommended path to a better model:** add 300-500 LLM-teacher examples (judgement, natural reasons) and a few
dozen designer samples on top of the rules examples, then train 2 epochs.

---

## 7. Files

| File | Purpose |
|---|---|
| `README.md` | This guide |
| `TRAINING_DATA.md` | Deep dive on the training data: formats, every answer field, quality gate, current dataset numbers, limitations |
| `rules_dataset.py` | Examples from the current rules (offline) → `data/rules.jsonl` |
| `build_dataset.py` | Examples from an LLM teacher with the full knowledge → `data/generated.jsonl` |
| `harvest_designs.py` | Examples from good web-UI designs → `data/harvested.jsonl` |
| `add_sample.py`, `samples/` | Hand-made designer samples (create, check) |
| `merge.py` | All sources → `data/train.jsonl`, `data/val.jsonl`, `data/stats.json` |
| `rooms.py` | Random room and customer-brief generator |
| `common.py` | Short prompt, quality gate, row format |
| `Gemma_finetune_on_Colab.ipynb` | Colab: QLoRA fine-tune, GGUF export, Ollama serving |
| `colab_run.py` | Run that notebook inside an already-connected Colab session |
| `install_local.sh` | Install a downloaded GGUF into local Ollama |
| `data/` | Datasets (`*_rejected.jsonl` explain every rejected room) |

## 8. Settings

| Variable | Default | Meaning |
|---|---|---|
| `GEMMA_MODEL` | `gemma4:E4B` | Ollama model for the Gemma designer (`magick-gemma` = fine-tuned) |
| `GEMMA_USE_KNOWLEDGE` | `1` | `0` = short prompt (for the fine-tuned model) |
| `FINETUNE_MIN_SCORE` | `90` | Scorecard bar for a training example |
| `FINETUNE_MAX_CORRECTIONS` | `0` | Rule-engine corrections allowed in a training example |
| Notebook: `EPOCHS`, `LORA_R`, `LEARNING_RATE`, `MAX_SEQ` | 1, 16, 2e-4, 4096 | Training settings |
| Notebook: `SKIP_TRAINING`, `SAVE_TO_DRIVE` | False, True | Reuse a model saved on Google Drive |

## 9. Glossary

| Term | Meaning |
|---|---|
| **Fine-tuning** | Further training an existing model on your own examples |
| **SFT** | Supervised fine-tuning: learn from (prompt → ideal answer) pairs |
| **LoRA** | Train small add-on matrices instead of the whole model; the base stays frozen |
| **QLoRA** | LoRA on a 4-bit quantized model, so it fits a small GPU |
| **Distillation** | A teacher with more knowledge produces answers a student learns to copy |
| **Epoch** | One pass over all training examples |
| **Validation loss** | How wrong the model is on held-out examples; falling = learning, rising = memorising |
| **GGUF** | The model file format Ollama / llama.cpp run |
| **q4_k_m** | 4-bit quantization level for the GGUF - small and good quality |
