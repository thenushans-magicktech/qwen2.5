# Making the training dataset with Qwen 2.5 14B on Colab

`training/qwen_dataset.py` makes training examples for the MagickHome designer model. Qwen 14B
designs each kitchen on a free Colab GPU. Your laptop does everything else: floor plans, customers,
the prompt, the engine check and saving.

## Every example

1. **Floor plan.** The plan comes from one of three sources, each of them in the MagickHome scene format:
   - a real room from your database export (`training/floorplans/db/`, 112 rooms, cabinets removed);
   - a variation of a real room (stretched or shrunk, doors and windows moved, other ceilings);
   - a generated room.

   The door style is **Slab (Plain)** or **Shaker 1**.
2. **Customer**: needs, budget, fuel, brief and chosen walls (random mix).
3. **Prompt**: the 81 rules of the rule sheet, the cabinet codes for that door style (each code is a real
   AID), the room facts and the customer's needs. `knowledge/designer_advice.md` is added as
   **advice, not rules**.
4. **Qwen designs it**, twice per room (`--best-of 2`). The answer is streamed through the tunnel.
5. **Your engine checks it**, twice:
   - pass 1: whatever the rule book or the engine changed goes into the answer;
   - pass 2: the example is kept only if there are zero corrections and the 3D scene verifies.

   Of the two designs, the one the engine had to change least is kept.
6. **Saved**: room facts, customer, decisions, the full cabinet layout (AIDs and positions), Qwen's
   reasons and **reason codes** per decision, plus the plan file. Progress is saved after every example.

## Reason codes (from the rule sheet, the customer and the room only)

| Code | Meaning |
|---|---|
| `RULE_Rn` | Rule n of the rule sheet decided or limited it |
| `CUSTOMER_*` | What the customer chose or asked for (form or brief), including budget and fuel |
| `ROOM_CEILING_*` | The ceiling height decides lofts |
| `ENGINE_ADJUSTED` | The engine changed Qwen's choice to fit the room or a rule |
| `DESIGNER_CHOICE` | Nothing above decided it: Qwen's own judgement (see its written reason) |

Full list: `knowledge/reason_codes.json` in the output.

## How to run it

1. Open `colab/Qwen_on_Colab_GPU.ipynb` in Google Colab. Choose **Runtime → Change runtime type → T4 GPU**,
   then **Runtime → Run all** (5–10 minutes the first time; it downloads Qwen 2.5 14B).
2. The last cell prints a command. Run it in PowerShell on the laptop, in the project folder:
   ```
   python training/qwen_dataset.py --url https://xxxx.trycloudflare.com --token XXXX --model qwen2.5:14b --n 1000
   ```
3. Leave the Colab tab open and the laptop awake. Each kept example prints one line with the time left.
4. **When Colab stops** (idle, or after about 12 h), the tool prints `PAUSED` and exits with progress saved.
   Open the notebook, **Run all** again, and run the **new** command. It continues from where it stopped.
5. To stop it yourself, press **Ctrl+C**. Progress is saved; run the same command to continue.

Start with a pilot: `--n 50` (about 3 hours). Review it, then run again with `--n 1000`. That run
continues from the 50 already made.

## Output (`training/dataset_qwen/`)

| Path | What |
|---|---|
| `scratch/train.jsonl`, `scratch/val.jsonl` | The examples: input (room + customer), output (decisions + full layout), reasons, reason codes, checks |
| `scratch/features.csv`, `scratch/labels.csv` | Tables for training (up to 8 walls per room; one `why_*` column per decision) |
| `plans/` | The floor plan of every example (MagickHome scene JSON) |
| `knowledge/` | All 3,684 cabinets, codes per door style, the 81 rules, finishes, reason codes, the designer advice |
| `progress.json`, `run.log`, `stats.json` | Where the run is, its log, and why designs were rejected |

## Test without Colab

```
python training/qwen_dataset.py --designer policy --n 5 --out training/try   # stand-in designer
python training/qwen_dataset.py --designer mock --n 3 --out training/try     # tests the streaming proxy
```

## Not yet included

- **Corner sink units.** The catalogue has 16, but the engine and the code list do not use them yet.
- **Parallel kitchens.** The engine does not build them yet.
- **The other 850 projects' floor plans.** They are not in the `ai_design` database.

Adding the first two needs engine work; after that, the pipeline uses them with no other change.
