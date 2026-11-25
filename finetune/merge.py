"""Build the final training files from every source.

    python finetune/merge.py            # -> finetune/data/train.jsonl, val.jsonl, stats.json

Sources: data/generated.jsonl (teacher on random rooms), data/harvested.jsonl (good UI designs) and
samples/*.json (hand-made designs; checked against the answer format, broken ones are refused).
Samples are repeated (--sample-weight, default 3) because a designer's sample is worth more than a
generated one. About 10% of generated / harvested rows go to val.jsonl to measure the fine-tune.
The train / val files hold only {"messages": [...]} - the format the Colab notebook reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import common  # noqa: F401  (sets sys.path)
from common import DATA, make_row, read_jsonl
from add_sample import SAMPLES, _load_scene, check
from ai_designer import KitchenPreferences


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample-weight", type=int, default=3)
    ap.add_argument("--val-share", type=float, default=0.1)
    args = ap.parse_args()

    rows = read_jsonl(DATA / "generated.jsonl") + read_jsonl(DATA / "harvested.jsonl")
    samples, refused = [], []
    for path in sorted(SAMPLES.glob("*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        problems = check(s)
        if problems:
            refused.append(f"{path.name}: {problems[0][:300]}")
            continue
        prefs = KitchenPreferences(**s.get("preferences", {}))
        samples.append(make_row(f"sample-{path.stem}", "sample", _load_scene(s["scene"]), prefs, s["answer"],
                                "designer", None, {"note": s.get("note", "")}))

    seen, train, val = set(), [], []
    for r in rows:
        key = hashlib.sha1(json.dumps(r["messages"], sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        bucket = int(key[:8], 16) / 0xFFFFFFFF  # stable split: a row never moves between train and val
        (val if bucket < args.val_share else train).append(r)
    train += samples * max(1, args.sample_weight)

    for name, part in (("train", train), ("val", val)):
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    chars = [sum(len(m["content"]) for m in r["messages"]) for r in train + val] or [0]
    stats = {
        "train_rows": len(train), "val_rows": len(val), "samples": len(samples), "refused_samples": refused,
        "by_source": dict(Counter(r["source"] for r in rows + samples)),
        "by_teacher": dict(Counter(r["teacher"] for r in rows)),
        "approx_tokens_per_row": {"max": max(chars) // 4, "mean": sum(chars) // len(chars) // 4},
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if refused:
        print("\nSome samples were refused - fix them with: python finetune/add_sample.py check <file>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
