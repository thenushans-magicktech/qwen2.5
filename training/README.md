# MagickHome kitchen designer — training dataset

Training data for a kitchen-design model **trained from scratch** on MagickHome's own knowledge:
the rule workbook, the cabinet catalogue (assets), the scene JSON format, and kitchen design practice.

Room facts and customer needs go in; design decisions come out (which walls, sink and hob wall and
position, unit sizes, tall units, purifier/display/open units, storage style, wall-door style,
finishes, lofts). Every example also has the **final cabinet-by-cabinet layout** (catalogue AIDs and
positions) built by the MagickHome placement engine, and the designer's reason for each decision.

Make or remake it with:

```
python training/make_dataset.py --n 1000              # about 1-1.5 h on a laptop (6 workers)
python training/make_dataset.py --n 50 --out training/sample   # a quick sample
```

The same `--seed` always gives the same dataset.

## What is in `training/dataset/`

| Path | What | Used for |
|---|---|---|
| `scratch/train.jsonl`, `scratch/val.jsonl` | One structured example per line (input, output, reasons, rules applied, checks) | **Training the from-scratch model** |
| `scratch/features.csv` | One row per example: 72 room and customer features (walls as slots Wall 1-4) | Classic ML (gradient boosting, small neural nets) |
| `scratch/labels.csv` | One row per example: 27 design decisions | The targets for the features above |
| `knowledge/cabinets.json` | The catalogue the designs use: AID, category, width/depth/height, combination, opening, handle... (1,065 units, Slab door style, from the Assets API copy) | Asset knowledge: turning decisions into real cabinets |
| `knowledge/codes.json` | The option codes a designer chooses (SINK-1050, HOB-900-3DRAWER, OVEN-600...) mapped to AIDs and sizes | The decision vocabulary |
| `knowledge/rules.json` | All 81 rules of `Kitchen_Rule_Engine_V1.xlsx`, the module-width table (by kitchen type and size band) and the rule groups | Rule knowledge |
| `knowledge/finishes.json` | 174 shutter finishes with codes, categories, colours, and the budget tiers | Finish knowledge |
| `scenes/` | Floor plan JSON and built 3D kitchen JSON (MagickHome scene format) for the first 20 examples | Reference: see a design in the app's own format |
| `llm/train.jsonl`, `llm/val.jsonl` | The same examples as chat messages (the production prompt and the design JSON answer) | Only if a language model is trained later |
| `meta.jsonl`, `stats.json` | Per-example notes (room, customer, checklist score) and dataset totals | Audit and balance checks |

The split is 95% train, 5% validation.

## One example (`scratch/*.jsonl`)

```jsonc
{
  "id": "room0042",
  "room_spec": {"width_mm": 3600, "depth_mm": 2600, "height_m": 2.8,
                "windows": [[1, 1290, 700]], "doors": [[3, 1890, 950]]},   // [wall, centre mm, width mm]
  "input": {
    "room": {
      "floor_area_sqft": 94, "size_band": "Medium", "ceiling_mm": 2800, "lofts_possible": true,
      "walls": [{"wall": "Wall 1", "length_mm": 2600, "corner_at_start": "Wall 4", "corner_at_end": "Wall 2",
                 "windows": [{"start_mm": 940, "end_mm": 1640, "width_mm": 700, "center_mm": 1290}],
                 "doors": [], "free_runs_mm": [[50, 2550]], "longest_free_mm": 2500,
                 "can_carry_cabinets": true}, "..."]
    },
    "customer": {"layout": "U", "budget": "luxury", "fuel": "lpg", "base_storage": "drawers",
                 "built_in": "auto", "include_loft": true, "cooking_wall": null, "sink_wall": null,
                 "brief": "We store big vessels and festival items. Prefer dark, rich colours.",
                 "brief_flags": {"big_vessels": true, "...": false}, "brief_colour": "dark", "...": "..."}
  },
  "output": {
    "decisions": {"cabinet_walls": ["Wall 1", "Wall 2", "Wall 4"], "sink_wall": "Wall 2", "sink": "SINK-1050",
                  "sink_center_mm": 2720, "hob_wall": "Wall 4", "hob": "HOB-750-3DRAWER", "hob_center_mm": 2540,
                  "spice": "SPICE-300", "lpg_cylinder": "LPG-600", "tall_units": [{"wall": "Wall 4", "unit": "PANTRY-600"}],
                  "water_purifier": "NONE", "display_unit": "NONE", "open_unit": "NONE", "base_storage": "drawers",
                  "base_feature_units": [], "wall_cabinet_sizes": [], "wall_style": "shutters",
                  "base_finish": "...", "wall_finish": "...", "lofts": "yes",
                  "code_aids": {"SINK-1050": "MTBABSU1050...", "...": "..."}},
    "layout": [{"wall": "Wall 2", "cabinets": [{"tier": "base", "role": "sink", "aid": "MTBABSU1050...",
                "category": "Base Sink Unit", "width_mm": 1050, "start_mm": 2195, "end_mm": 3245, "opening": "Both"}, "..."]}],
    "heights": {"wall_mm": 600, "tall_mm": 1940, "lofts": true, "ceiling_mm": 2800, "top_mm": 2658},
    "kitchen_type": "U-Shape"
  },
  "reasons": {"room_reading": "...", "design_plan": "...", "sink_why": "...", "hob_why": "...", "...": "..."},
  "rules_applied": [{"rule": 66, "role": "hob", "width_mm": 800, "text": "In a Medium U-Shape kitchen ..."}],
  "checks": {"engine_corrections": 0, "checklist_score": 92.9, "checklist_failed": ["..."], "scene_verified": true}
}
```

`decisions` is what the model should learn to predict. `layout` is what the engine built from them
(the exact cabinets). A model can be trained on either; the decisions are the practical target,
because the engine turns them into a full, rule-correct layout.

## How the examples were made

1. **Rooms**: rectangles from 2.1 to 5.0 m a side, built from the production sample floor plan. They are
   sampled by floor area so Small (60-80 sqft), Medium (81-120) and Large (121-180) kitchens all occur,
   with ceilings from 2.4 to 3.2 m, 1-3 doors (sometimes 3 on one wall) and 0-2 windows that do not overlap.
2. **Customers**: random mixes of "let the designer decide" and stated choices: kitchen shape, ticked walls,
   cooking and sink walls, purifier, built-in oven or microwave, display and open units, crockery unit,
   storage style, wall doors, cooking style, household size, lofts, tall storage, fuel and budget. 45%
   have an English brief, with habits and colour wishes that the form has no field for.
3. **Design**: a designer policy (`design()` in `make_dataset.py`) reads each room and customer and decides,
   following the rule workbook and common Indian kitchen practice:
   - sink under the window for light;
   - hob on a solid wall for the chimney, away from doors and the sink;
   - LPG unit and spice pull-out beside the hob;
   - tall units at run ends, clear of windows;
   - drawers for heavy cooking or big families;
   - storage style and finishes within the budget tier;
   - finishes from the brief's colour words;
   - lofts when the ceiling allows.

   It picks among several good options at random, so similar rooms get different designs.
4. **Check by the real pipeline**: every design goes through the production rule book, placement engine
   and 3D build.
   - **First pass**: whatever the engine had to change (a module width set by the rule book, a final
     position) is taken into the answer. When the plan could not be built, a designer's repair is tried:
     move the tall unit, a narrower hob or sink, sink to one end of a single wall, another hob wall.
   - **Second pass**: the example is kept only when the engine makes **no correction**, the 3D scene
     verifies (no collisions, every AID in the catalogue), and the designer checklist passes
     (`designer_benchmark.CHECKS`; the average is about 95%).
5. **Reasons** are written last, from the final decisions, the room facts and the rules the rule book
   actually applied, so a reason never contradicts its decision.

## Points to know before training

- **Synthetic, not real projects.** The rooms are generated rectangles and the decisions come from a design
  policy, not MagickHome designers. A model trained on this learns that policy within the rules: reliable
  and rule-correct, but no better than the policy. To learn real designer judgement, add real MagickHome
  designs or designer-corrected examples. Even 200-300 of those matter more than more synthetic data.
- **Rectangular rooms only.** Every room has 4 walls (Wall 1-4 in order). Real plans with more walls need
  the variable-length `walls` list (already in the JSON), not the 4-slot CSV.
- **Codes vs rule widths.** The rule book sets some widths the code list cannot name. For example, R63 sets
  a 600 mm hob for a Small U-shape, but the codes offer 750 and 900. The decision holds the closest code,
  and `layout` and `rules_applied` hold the width actually built. A model that predicts decisions should
  leave the exact width to the rule book, as production does.
- **Balance.** About 55% L-shape, 35% U-shape and 10% straight. Straight kitchens are rarer because many of
  the generated rooms cannot fit a sink, hob, LPG unit and their landings on one wall. Check `stats.json`
  for the exact counts.
- **One door style.** Slab (Plain) with external handles. Shaker designs would need another run with that
  door style.
- **The "why" texts follow patterns.** They are accurate but less varied than a designer's own words.

## Suggested way to train

- **Decisions (recommended first)**: `features.csv` to `labels.csv`.
  - One classifier per categorical label: shape, the cabinet-wall flags, sink and hob wall, unit codes,
    storage, wall style, lofts, finishes.
  - A regressor for `sink_center_mm` and `hob_center_mm`. Predicting them relative to the wall length works better.
  - Gradient boosting (LightGBM, XGBoost or scikit-learn) trains in minutes on a laptop CPU.
  - Hold out `split == "val"`.
- **Then let the engine build**: feed the predicted decisions to the placement engine
  (`rule_engine.solve_layout`) the same way the app does with Qwen's answer. That gives the full,
  rule-correct cabinet layout and the 3D scene.
- **Measure**: run the model's designs through `designer_benchmark.py` and count engine corrections.
  Compare against Qwen and the quick rules on the same rooms.
