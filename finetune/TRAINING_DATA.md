# Training data for the fine-tuned Gemma designer

What the training data is, where every example comes from, what is inside one, how it is checked, what the
current dataset contains (and does not), and how to inspect, extend and clean it. For the overall approach and
the training method see [README.md](README.md).

---

## 1. The idea in one paragraph

The fine-tuned model must answer the **short prompt** (cabinet menu + room facts + customer needs, no playbook)
as well as a designer who knows all the MagickHome rules. So each training example pairs **exactly the short
prompt the app will send** with **a design that is known to be correct** - checked by running it through the
real rule engine. The model never sees the playbook text; it learns its effect from hundreds of correct designs.

---

## 2. Files in `data/`

| File | Written by | One line = | Used for training |
|---|---|---|---|
| `rules.jsonl` | `rules_dataset.py` | a kept example from the current rules | yes (via merge) |
| `rules_rejected.jsonl` | `rules_dataset.py` | a rejected room and the reason | no - diagnostics |
| `generated.jsonl` | `build_dataset.py` | a kept example from an LLM teacher | yes (via merge) |
| `rejected.jsonl` | `build_dataset.py` | a rejected room and the reason | no - diagnostics |
| `harvested.jsonl` | `harvest_designs.py` | a kept example from a web-UI design | yes (via merge) |
| `train.jsonl` | `merge.py` | `{"messages": [...]}` only | **yes - what the notebook trains on** |
| `val.jsonl` | `merge.py` | `{"messages": [...]}` only | validation loss |
| `stats.json` | `merge.py` | counts per source / teacher, token sizes | - |
| `../samples/*.json` | designers (`add_sample.py`) | one hand-made design | yes (3× weight, via merge) |

All `.jsonl` files are UTF-8, one JSON object per line. Every generator **appends** and **skips rooms already
present** (kept or rejected), so it can be stopped and restarted safely.

### 2.1 A source row (`rules.jsonl`, `generated.jsonl`, `harvested.jsonl`)

```json
{
  "id": "rules-s101-00003",
  "source": "rules",                       // rules | generated | ui | sample
  "teacher": "rules-teacher",              // or the teacher model, e.g. gemma4:E4B, gemini-2.5-flash
  "score": 100.0,                          // designer scorecard of the replayed design
  "messages": [
    {"role": "system",    "content": "You are an experienced modular-kitchen designer ..."},
    {"role": "user",      "content": "CABINET LIST - choose by CODE ...\nROOM ...\nCUSTOMER NEEDS ..."},
    {"role": "assistant", "content": "{\"room_reading\": \"...\", ...}"}
  ],
  "meta": {
    "room": "5.4 x 3.5 m, 1 window(s), 1 door(s), 2.8 m high",
    "preferences": {"door_style": "Shaker -1", "fuel": "lpg", "base_storage": "shutters", "household_size": 3},
    "prompt_version": "v2",
    "knowledge": false                     // the prompt is the SHORT one
  }
}
```

`meta.preferences` lists only the needs that differ from the defaults ("not stated"). The `id` encodes the source,
the seed and the room index, so the same room can always be rebuilt: `rules-s<seed>-<index>`,
`gen-s<seed>-<index>`, `ui-<job id>`, `sample-<file name>`.

### 2.2 A rejected row

```json
{"id": "rules-s101-00002",
 "reason": "score 91.7 failed ['hob_position_sane'] fixes ['tall unit(s) dropped: they left no legal hob position', ...] issues 0",
 "room": "4.9 x 1.8 m, 2 window(s), 0 door(s), 2.5 m high"}
```

### 2.3 A training row (`train.jsonl`, `val.jsonl`)

Only `{"messages": [system, user, assistant]}` - the format the Colab notebook reads.

---

## 3. Inside one example

### 3.1 The prompt (what the model reads)

Built by `ai_designer.build_messages(geometry, groups, prefs, "v2", knowledge=False)` - the same call the app
makes at run time with `GEMMA_USE_KNOWLEDGE=0`. Training and production prompts are therefore identical.

| Section | Content | Not included (vs. the full prompt) |
|---|---|---|
| **System** (~870 chars) | The designer's role, how the answer is used (a placement engine sets exact positions, fills runs, adds corner units and the chimney), stated needs are fixed, write each `*_why` first, answer only JSON | The designer playbook, the 3 worked examples |
| **Cabinet list** | ~36 options by **code**: `SINK-900 \| 900 x 560 mm \| Sink Double Shutter`, grouped (sink, hob, spice, LPG, tall, feature base units, wall units, purifier, display, open shelf) | The designer note on each option ("standard", "only with 3600+ mm free" …) |
| **Finishes** | The shutter-finish codes allowed for the budget | Finish advice |
| **Room** | Per wall: length, which corners it joins, openings with mm positions (DOOR 350-1100, WINDOW 3900-5100), "Solid", the free run; floor area; wall height | Rule hints: hob-centre range, "wet zone: sink centred at …", U-shape aisle, rule-book size band |
| **Customer needs** | Each need: stated (fixed) or "not stated" (the designer decides); forced walls; the free-text brief | - |
| **Task** | The answer fields in order, what each means | Designer guidance on how to fill them |

Typical size: user message 5,600-6,400 characters (mean 6,078); with the system message ~1,700 tokens
(word-level estimate), ~3,300 Gemma tokens including the answer.

**The cabinet menu changes with the door style** (Slab / Shaker) and with the catalogue - which is why it stays
in the prompt and is not learned.

### 3.2 The answer (what the model learns to write)

JSON with these fields, **in this order** (each reason comes before its decision). The same schema is enforced at
run time by Ollama, so the model can only produce valid values.

| Field | Type / allowed values | Limits |
|---|---|---|
| `room_reading` | text: window wall, door wall and its corner, long solid walls | ≤ 240 chars |
| `design_plan` | text: zones and work triangle | ≤ 200 chars |
| `layout_why` | reason for the cabinet walls | ≤ 180 chars |
| `cabinet_walls` | list of wall names from the prompt (`Wall 1` …) | 1-3 |
| `sink_why`, `sink_wall`, `sink` | reason; wall name; `SINK-800 / 900 / 1050 / 1200` | |
| `hob_why`, `hob_wall`, `hob` | reason; wall name; `HOB-900-3DRAWER`, `HOB-900-2DRAWER`, `HOB-750-3DRAWER`, `HOB-900-SHUTTER` | |
| `hob_center_mm` | integer, mm from the hob wall's start | 0-20,000 |
| `spice` | `SPICE-300`, `SPICE-150`, `NONE` | |
| `lpg_why`, `lpg_cylinder` | reason; `LPG-600` or `NONE` | |
| `tall_why`, `tall_units` | reason; list of `{wall, unit}` with `PANTRY-600`, `MICROWAVE-600`, `OVEN-600`, `OVEN-MICRO-600`, `CROCKERY-600` … | 0-2; **at least 1** when the customer wants tall storage |
| `purifier_why`, `water_purifier` | reason; `AQUAGUARD-600`, `AQUAGUARD-450`, `NONE` | |
| `display_why`, `display_unit` | reason; `GLASS-600` … or `NONE` | |
| `open_why`, `open_unit` | reason; open shelf code or `NONE` | |
| `storage_why`, `base_storage` | reason; `drawers`, `mixed`, `shutters` | |
| `features_why`, `base_feature_units` | reason; list of `{wall, units}` with `DRAWERS4-600`, `GRAIN-600`, wicker … | ≤ 3 walls, ≤ 3 units each |
| `wall_cabinets_why`, `wall_cabinet_sizes` | reason; list of `{wall, units}` with `WALL-600`, `WALL-900`, `WALL-450-1DOOR` … | ≤ 3 walls, ≤ 2 sizes each |
| `wall_style_why`, `wall_style` | reason; `shutters` or `lift_up` | |
| `finish_why`, `base_finish`, `wall_finish` | reason; finish codes from the prompt (allowed by the budget) | |
| `loft_why`, `lofts` | reason; `yes` / `no` (`no` only when the ceiling is too low) | |

Every `*_why` is at most 180 characters.

**Stated customer needs narrow the allowed values.** If the customer says "piped gas", `lpg_cylinder` can only be
`NONE`; "drawers" forces `base_storage = drawers`; a chosen hob wall fixes `hob_wall`. So in the data, a stated need
is always obeyed, and "not stated" is where the model learns to decide.

What the model does **not** decide (the rule engine does, so it is not in the answer): exact cabinet positions,
corner units, fillers, hinges, plain storage cabinets, countertop, backsplash, chimney, toe kick, heights.

---

## 4. Where the answers come from

### 4.1 Rules teacher - `rules_dataset.py` (the current dataset)

```
random room + brief
   → Quick-rules designer proposes (walls, sink, hob, tall, features)
   → rule book sets module sizes (e.g. R69: hob 900 mm Triple Drawer in a Large U-shape)
   → rule engine lays everything out (clearances, corners, landing space, fill)
   → read the FINAL layout → write it as the answer (codes, walls, positions)
   → reasons from the playbook / rule book, citing rule numbers
   → REPLAY the answer through the real pipeline as if the model gave it
   → quality gate (section 5)
```

Because the answer is read from the **final** layout, it already contains every correction the engine would make
- the replay confirms the engine needs **zero** further corrections. The design decisions are therefore always
buildable; the reasons are templates (see the known issues in section 7).

Speed: ~0.5 s per room on the laptop CPU, no network, no GPU.

### 4.2 LLM teacher - `build_dataset.py`

Gemma (on the Colab GPU) or Gemini designs the room with the **full** knowledge prompt; the answer goes through the
same quality gate; the stored example pairs the **short** prompt with that answer. This adds real judgement and
natural reasons that templates cannot. ~45 s per room on a Colab T4.

### 4.3 Web-UI designs - `harvest_designs.py`

Finished AI designs saved in `designs/*.json` (Qwen, Gemma, Gemini with the full knowledge). Skipped
automatically: quick-rules designs, failed jobs, short-prompt answers (so the fine-tuned model never trains on its
own output), and answers in an older answer format. Of 71 saved designs, 6 passed.

### 4.4 Designer samples - `samples/*.json`

Hand-checked designs from MagickHome designers (see `samples/README.md`). Checked against the answer format by
`add_sample.py check` and again by `merge.py`; counted **3×** in training; always in the training split.

---

## 5. The quality gate

An answer is kept only if, after the replay through the real pipeline:

| Condition | Bar |
|---|---|
| Rule-engine corrections (sink moved, hob moved, re-chosen walls, units dropped / skipped …) | 0 (`FINETUNE_MAX_CORRECTIONS`) |
| Layout validation issues (overlaps, 3D collisions, corner rule, chimney alignment …) | 0 |
| Designer scorecard | ≥ 90 % (`FINETUNE_MIN_SCORE`) |
| Answer matches today's schema | must validate |

The designer scorecard (`designer_benchmark.score_design`) - each check counts only when it applies to the room:

| Check | Passes when |
|---|---|
| `reads_room` | the room reading names the window wall and the door wall correctly |
| `sink_on_window_wall` | the sink is on the window wall |
| `hob_on_solid_wall` | the hob is on a solid wall (when a long solid wall exists) |
| `sink_hob_apart` | sink and hob on different walls, or 600+ mm of counter between them |
| `hob_clear_of_door` | the hob is 600+ mm from any door on its wall |
| `hob_position_sane` | the hob centre leaves 300+ mm of counter on both sides inside a free run |
| `sink_size_suits_wall` | 1050 needs 3000+ mm free, 1200 needs 3600+ mm |
| `hob_unit_has_drawers` | a drawer hob unit (pans lie flat) |
| `spice_beside_hob` | a spice pull-out is chosen |
| `tall_well_placed` | tall units on a cabinet wall, not crowding the hob wall |
| `walls_connected` | the cabinet walls form one connected run including the window wall |
| `uses_the_room` | L or U when the window wall has a usable neighbouring wall |
| `features_zoned` | drawers on the hob wall, wicker by the sink, grain away from the hob |
| `wall_cabinets_planned` | wall-cabinet sizes chosen, only for cabinet walls |

---

## 6. The current dataset (first training run)

### 6.1 Summary

| | |
|---|---|
| Source | `rules.jsonl` (seed 101) + `harvested.jsonl` |
| Rooms tried | 1,100 |
| Kept | **726** rules (66 %) + 6 UI = 732 → **731** after de-duplication |
| Split | **660 train / 71 validation** (90 / 10, stable by content hash) |
| Scorecard of kept examples | 100 % for 550; 90-93 % for the rest |
| Size | user message 5,617-6,373 chars (mean 6,078); answer 1,990-2,580 chars (mean 2,303); ~3,300 Gemma tokens per example |
| File size | `train.jsonl` 6.1 MB, `val.jsonl` 0.7 MB |

### 6.2 Why rooms were rejected (374)

| Reason | Rooms | What it means |
|---|---|---|
| Rule-engine corrections | 142 | e.g. tall unit dropped to free a legal hob position, purifier skipped (no wall space near the sink) |
| Schema mismatch | 117 | mostly: the customer wants tall storage but the room had no free run end for one (the answer must name ≥ 1) |
| `features_zoned` failed | 74 | feature units landed on the wrong wall |
| `tall_well_placed` failed | 60 | tall unit crowding the hob wall |
| `hob_on_solid_wall` failed | 57 | hob ended up on a window wall although a solid wall existed |
| `sink_on_window_wall` failed | 23 | |
| `uses_the_room` / `walls_connected` failed | 19 / 16 | |
| Engine could not build the room | 12 | walls too short or fully blocked |

(One room can have several reasons.) Rejections are expected: the gate removes the rules designer's weak designs
so the model only learns from good ones.

### 6.3 What the rooms and briefs look like

| Property | Distribution in the kept examples |
|---|---|
| Windows | 1: 535 · 0: 107 · 2: 84 |
| Doors | 1: 572 · 2: 99 · 0: 55 (the generator could not place one) |
| Ceiling (m) | 2.5: 185 · 2.7: 184 · 2.8: 103 · 2.6: 94 · 2.4: 82 · 3.0: 78 |
| Door style | Slab: 574 · Shaker: 152 |
| Layout need | not stated: 667 · L: 30 · U: 25 · single: 4 |
| Tall storage | wanted: 603 · not wanted: 123 |
| Fuel | not stated: 428 · LPG: 152 · piped gas: 94 · induction: 52 |
| Water purifier | not stated: 496 · yes: 168 · no: 62 |
| Built-in appliances | not stated: 438 · microwave: 121 · oven: 70 · none: 62 · oven + microwave: 35 |
| Base storage | not stated: 511 · drawers: 109 · mixed: 65 · shutters: 41 |
| Cooking | not stated: 462 · heavy: 195 · light: 69 |
| Household size | not stated: 366 · 2-6 people: 360 |
| Lofts | not stated: 543 · yes: 106 · no: 77 |
| Budget | not stated: 522 · essential: 83 · premium: 80 · luxury: 41 |

### 6.4 What the answers contain

| Decision | Distribution |
|---|---|
| Cabinet walls | 3 (U): 489 · 2 (L): 233 · 1: 4 |
| Sink | 800: 248 · 900: 183 · 1200: 176 · 1050: 119 |
| Hob | HOB-900-3DRAWER: 410 · HOB-750-3DRAWER: 316 |
| Spice | 300: 556 · 150: 170 |
| LPG cylinder | yes: 580 · none: 146 |
| Tall units | 1: 531 · 0: 123 · 2: 72 - pantry 539, microwave 55, crockery 35, oven 33, oven + micro 13 |
| Water purifier | Aquaguard 600: 554 · none: 147 · Aquaguard 450: 25 |
| Glass display | none: 675 · yes: 51 |
| Open shelf | none: 726 |
| Base storage | mixed: 348 · drawers: 287 · shutters: 91 |
| Feature units | grain trolley in 722 · 4-drawer unit in 665 · wicker: 0 |
| Wall-cabinet sizes | 600: 1,178 · 450: 264 · 700: 227 · 800: 191 · 900: 153 · 1000: 56 |
| Wall doors | shutters: 726 |
| Lofts | no: 400 · yes: 326 |

---

## 7. Known limitations of the current data

| Limitation | Effect on the model | How to fix |
|---|---|---|
| **One teacher (the rules designer)** - 99 % of the data | Learns the rules designer's habits, not richer judgement | Add LLM-teacher examples (`build_dataset.py`) and designer samples |
| **Templated reasons** | Reasons are correct-ish but repetitive | LLM-teacher and designer data bring natural reasons |
| **Wording bugs in dataset v1** (sink reason when the engine shrank the sink, drawers named but not chosen in 61 examples, "Storage: None" in 301 plans) | Fixed: the generator was corrected and `rules.jsonl` regenerated (v2, 0 occurrences). The first, interrupted training run used v1 | - |
| **Never-seen choices**: `wall_style` is always `shutters`, `open_unit` always `NONE`, no wicker basket, almost no one-wall kitchens (4) | The model will rarely choose lift-up doors, open shelves, wicker or a single-wall layout on its own | Randomise `wall_style` / `open_unit` in `rooms.py`; add samples with these choices; more narrow rooms |
| **Cabinet coverage**: the prompt menu has 40 codes (from 1,065 SKUs per door style, 3,684 in the full catalogue); the training answers use 28. Never chosen: HOB-900-2DRAWER, HOB-900-SHUTTER, CROCKERY-900, TALL-WICKER-600, DRAWER-SHUTTER-800, GRAIN-450, WICKER-450, AQUAGUARD-750, GLASS-800, GLASS-900, OPEN-450, OPEN-600 | The model can pick them (they are in the menu and the schema) but rarely will on its own. The full catalogue is never learned - the rule engine maps decisions to real SKUs and fills plain storage from it | Designer samples using these units; vary the briefs in `rooms.py`; LLM-teacher data |
| **Almost always grain trolley + drawers** as feature units | Little variety in feature units | LLM-teacher data / samples |
| **Rectangular rooms only** | Unknown behaviour on L-shaped or angled rooms (the rule engine still keeps the layout valid) | Add real customer floor plans via the web UI → `harvest_designs.py`, or samples |
| **Rooms that need a correction are excluded** | The model sees fewer "difficult" rooms | Fine for now - the engine still handles them at run time |

---

## 8. Working with the data

### Look at it
```bash
cd aidesignernew
# how many examples, from where, how big
cat finetune/data/stats.json
# one example, pretty-printed answer
python3 -c "import json; r=json.loads(open('finetune/data/rules.jsonl').readline()); print(r['meta']); print(json.dumps(json.loads(r['messages'][-1]['content']), indent=1))"
# why rooms were rejected
cut -c1-200 finetune/data/rules_rejected.jsonl | head
# the exact prompt for one example
python3 -c "import json; r=json.loads(open('finetune/data/rules.jsonl').readline()); print(r['messages'][1]['content'])"
```

### Add more
```bash
MAGICKHOME_ASSETS_API_URL= python3 finetune/rules_dataset.py --n 1000 --seed 102     # new rooms: new seed
OLLAMA_HOST=... OLLAMA_API_KEY=... python3 finetune/build_dataset.py --n 300 --teacher gemma
python3 finetune/harvest_designs.py
python3 finetune/add_sample.py new --scene plan.json --name my_case --job <job id>   # then edit + check
python3 finetune/merge.py
```

### Start over (after a rule change)
Move the old source files away so old answers don't teach old rules, then regenerate:
```bash
mkdir -p finetune/data/old && mv finetune/data/{rules,rules_rejected,generated,rejected}.jsonl finetune/data/old/
```
Keep `samples/` unless the change makes a sample wrong - `merge.py` refuses samples that no longer match the
answer format.

### How much data is enough

| Examples | What to expect |
|---|---|
| ~300 | The model learns the answer format and the basic rules |
| 600-1,000 (now: 731) | Reliable rule-following on rooms like the training rooms |
| 1,000-3,000, mixed sources | Better judgement and variety; worth training 2-3 epochs |

Quality and variety matter more than count: 50 good designer samples improve the model more than 500 more
template examples.

---

## 9. Data rules

1. **The prompt in a training example must be exactly what the app sends** - always build it with
   `common.short_messages()` / `build_messages(..., knowledge=False)`, never by hand.
2. **Every answer must pass the quality gate** - no example goes in without the replay check.
3. **Never train on the fine-tuned model's own answers** (`harvest_designs.py` skips short-prompt answers).
4. **One answer format at a time** - when the schema changes, old examples are rejected by the validation in
   `harvest_designs.py`, `add_sample.py` and `merge.py`.
5. **No personal data** - rooms are geometry and catalogue codes; customer names, addresses and file names are not
   put in prompts or answers (`meta.filename` of harvested rows stays out of `train.jsonl`).
6. **Keep the validation split stable** - don't edit `merge.py`'s hashing; examples must not move between train
   and validation between runs.
