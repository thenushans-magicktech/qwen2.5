# Designer sample designs

Each `*.json` here is one hand-checked design: "for this room and this customer, this is the right answer".
`merge.py` adds every valid sample to the training data (3× weight) and refuses broken ones.

## Format

```json
{
  "scene": "magickhomes-scene_2026-11-18_10-24-05.json",
  "preferences": {"fuel": "png", "water_purifier": "yes"},
  "note": "Why this is a good design (for reviewers; not shown to the model).",
  "answer": { "...": "the design, in the same JSON format the AI designer answers in" }
}
```

- `scene`: an empty floor plan (path relative to the project folder, or absolute). Existing cabinets are ignored.
- `preferences`: the customer's stated needs (same fields as the web UI / `KitchenPreferences`); omit = not stated.
- `answer`: the design. Codes (`SINK-900`, `HOB-900-3DRAWER` …) and wall names (`Wall 1` …) must be ones the
  prompt offers; write each `*_why` as a short plain reason.

## Making one

```bash
# start from a finished web-UI design you like (best), or from an empty template
python finetune/add_sample.py new --scene my_room.json --name l_shape_tall_by_door --job <job id>
# edit finetune/samples/l_shape_tall_by_door.json, then
python finetune/add_sample.py check finetune/samples/l_shape_tall_by_door.json
```

`new` prints the exact prompt the model will see, so you can pick codes and wall names from it.
The job id is the `?job=...` part of a web-UI link (its file is `designs/<id>.json`).
