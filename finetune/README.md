# Fine-tuning Gemma for the MagickHome kitchen designer

**Goal:** put the MagickHome designer knowledge (playbook, worked examples, cabinet notes, rule hints) into
Gemma's weights, so every design request sends the **short prompt** instead of the full knowledge pack.

| | Today (knowledge in the prompt) | Fine-tuned Gemma (knowledge in the weights) |
|---|---|---|
| Prompt per design | ~6,600 tokens | ~1,700 tokens (room + cabinet menu + customer needs) |
| Edit the playbook | takes effect on the next design | needs new training data and a re-train |
| Rule engine | checks and corrects every design | unchanged - still checks every design |

The cabinet menu stays in the prompt (the catalogue changes; the answer must use today's codes), and the
answer format is still locked by the JSON schema, so the model cannot invent a cabinet.

## How it works

```
 random rooms ─► teacher (Gemma / Gemini WITH the full knowledge pack) ─► rule engine ─► scorecard
                                                                                   │
                         keep only designs with 0 rule corrections, 0 issues, score ≥ 90
                                                                                   ▼
        training example =  SHORT prompt (what the student will see)  ─►  the teacher's answer
                                                                                   │
 good UI designs (designs/*.json) ─────────────────────────────────────────────────┤
 designer samples (finetune/samples/*.json, weighted ×3) ──────────────────────────┤
                                                                                   ▼
                     merge.py ─► data/train.jsonl + data/val.jsonl ─► Colab: LoRA fine-tune (Unsloth)
                                                                                   │
                               GGUF ─► Ollama "magick-gemma" ─► tunnel ─► laptop app (--finetuned)
```

## Files

| File | What it does |
|---|---|
| `build_dataset.py` | Random rooms → the teacher designs them with the full knowledge → good ones become examples (`data/generated.jsonl`; failures in `data/rejected.jsonl`). Resumable. |
| `harvest_designs.py` | Good AI designs made in the web UI (`designs/*.json`) → examples (`data/harvested.jsonl`). |
| `add_sample.py` | Create / check a hand-made sample design in `samples/` (see `samples/README.md`). |
| `merge.py` | Everything → `data/train.jsonl`, `data/val.jsonl`, `data/stats.json`. Checks every sample. |
| `rooms.py` | The random room and customer-brief generator. |
| `common.py` | Short prompt, quality judge, row format. |
| `Gemma_finetune_on_Colab.ipynb` | Colab T4: LoRA fine-tune of `unsloth/gemma-4-E4B-it`, export GGUF, save to Google Drive, serve with Ollama + password proxy + tunnel. |

## Step by step

### 1. Make training data (laptop + teacher on Colab)
Start the teacher: open `colab/Gemma_on_Colab_GPU.ipynb` in Colab and Run all (see `colab/README.md`). Then, with
the URL and password it prints:

```bash
OLLAMA_HOST=https://<tunnel>.trycloudflare.com OLLAMA_API_KEY=<password> \
  python finetune/build_dataset.py --n 300 --teacher gemma --seed 1
```

~45 s per room on the T4; about 60-80 % of rooms pass the quality bar. Stop any time and run the same
command again - finished rooms are skipped. Use `--seed 2`, `3` ... for more, different rooms.
Gemini works as a teacher too: `GEMINI_API_KEY=... python finetune/build_dataset.py --teacher gemini`.

Also add good designs already made in the web UI:

```bash
python finetune/harvest_designs.py
```

How many examples? 300 is a useful start; 800-1,000 is better. Quality matters more than quantity.

### 2. Build the training files

```bash
python finetune/merge.py
git add finetune/data/train.jsonl finetune/data/val.jsonl && git commit -m "Fine-tune data" && git push
```

(The Colab notebook reads the files from the `gemma` branch on GitHub, or you can upload them.)

### 3. Fine-tune on Colab
Open `finetune/Gemma_finetune_on_Colab.ipynb` in Colab
(`https://colab.research.google.com/github/thenushans-magicktech/qwen2.5/blob/gemma/finetune/Gemma_finetune_on_Colab.ipynb`),
choose the T4 GPU, Run all. About 1-2 h for 300-500 examples. The model is saved to Google Drive
(`MyDrive/magick-gemma`).

### 4. Use it
The notebook's last cell prints the command for the laptop:

```bash
python colab/connect_colab.py https://<tunnel>.trycloudflare.com <password> --finetuned
```

This sets `GEMMA_MODEL=magick-gemma` and `GEMMA_USE_KNOWLEDGE=0` (short prompt). Next Colab session: set
`SKIP_TRAINING = True` in the notebook - it loads the model from Drive and serves it in a few minutes.

### 5. Check it is as good as the teacher
Benchmark the fine-tuned model against the full-prompt Gemma on the same rooms (with the laptop app connected
to the fine-tuned model's Colab):

```bash
GEMMA_MODEL=magick-gemma GEMMA_USE_KNOWLEDGE=0 OLLAMA_HOST=... OLLAMA_API_KEY=... \
  python designer_benchmark.py --designer gemma --versions v2
```

Switch to it for daily use only when its score matches the teacher's (about 97 % on the 3-room run).

## Changing it later (new rules, new samples)

1. Edit the playbook / rules / knowledge pack as usual (`knowledge/`), or add designer samples (`samples/`).
2. Re-generate examples with the **teacher** (it reads the new knowledge): `build_dataset.py --seed <new>`.
   After a big rule change, start a fresh dataset (move the old `data/*.jsonl` away) so old answers don't
   teach the old rules.
3. `merge.py`, push, Run all on Colab with `SKIP_TRAINING = False`.

Until the re-train is done, the app can keep using the full-prompt Gemma (`--designer gemma`), which picks up
knowledge changes immediately.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `GEMMA_MODEL` | `gemma4:E4B` | Ollama model for the Gemma designer (`magick-gemma` = fine-tuned) |
| `GEMMA_USE_KNOWLEDGE` | `1` | `0` = short prompt (fine-tuned model) |
| `FINETUNE_MIN_SCORE` | `90` | scorecard bar for an example |
| `FINETUNE_MAX_CORRECTIONS` | `0` | rule-engine corrections allowed in an example |
