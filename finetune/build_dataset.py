"""Generate training examples: random rooms -> teacher designs them with the FULL knowledge pack ->
keep only designs the rule engine did not have to correct -> save as SHORT prompt -> answer.

    # teacher = Gemma on Colab (start colab/Gemma_on_Colab_GPU.ipynb first; ~45 s per room)
    OLLAMA_HOST=https://<tunnel> OLLAMA_API_KEY=<password> python finetune/build_dataset.py --n 300 --teacher gemma

    # teacher = Gemini (needs GEMINI_API_KEY; free tier is rate-limited)
    python finetune/build_dataset.py --n 100 --teacher gemini --seed 2

Resumable: rooms already in data/generated.jsonl or data/rejected.jsonl are skipped, so stop it any
time and run the same command again. Each --seed gives a different set of rooms.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback

import common  # noqa: F401  (sets sys.path)
from common import DATA, append_jsonl, judge, make_row, read_jsonl
import config
from rooms import random_prefs, random_room

GENERATED, REJECTED = DATA / "generated.jsonl", DATA / "rejected.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50, help="rooms to try this run (default 50)")
    ap.add_argument("--seed", type=int, default=1, help="room set; use a new seed for new rooms")
    ap.add_argument("--start", type=int, default=0, help="first room index")
    ap.add_argument("--teacher", default="gemma", choices=["gemma", "qwen", "gemini"])
    args = ap.parse_args()

    from main import generate_kitchen
    from ai_designer import KitchenPreferences  # noqa: F401

    # The teacher always gets the full MagickHome knowledge pack.
    config.GEMMA_USE_KNOWLEDGE = True
    config.GEMINI_USE_KNOWLEDGE = True

    done = {r["id"] for r in read_jsonl(GENERATED)} | {r["id"] for r in read_jsonl(REJECTED)}
    kept = rejected = 0
    t_all = time.time()
    for i in range(args.start, args.start + args.n):
        rid = f"gen-s{args.seed}-{i:05d}"
        if rid in done:
            continue
        rng = random.Random(args.seed * 1_000_003 + i)
        room = random_room(rng, rid)
        prefs = random_prefs(rng, ai_mode="ollama", ai_provider=args.teacher)
        doc = room.build()
        t = time.time()
        try:
            _, report = generate_kitchen(doc, prefs)
        except Exception as exc:  # teacher unreachable, room the engine cannot build, ...
            print(f"{rid}  {time.time() - t:5.0f}s  ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            if "Unavailable" in type(exc).__name__ or "not reachable" in str(exc):
                # not the room's fault: do not record it, so the next run retries this room
                print("The teacher is not reachable - stopping. Start it and run the same command again.")
                return 1
            append_jsonl(REJECTED, {"id": rid, "reason": f"{type(exc).__name__}: {exc}"[:400], "room": room.title})
            rejected += 1
            continue
        ai = report.get("ai") or {}
        try:
            verdict = judge(doc, prefs, report)
        except Exception:
            traceback.print_exc()
            verdict = {"ok": False, "score": None, "failed": ["judge error"], "corrections": []}
        raw = ai.get("raw")
        msg = (f"score {verdict['score']}  fixes {len(verdict['corrections'])}  "
               f"failed {verdict['failed']}  issues {len(verdict.get('validation_issues') or [])}")
        meta = {"room": room.title, "teacher_seconds": ai.get("seconds"), "failed_checks": verdict["failed"]}
        if verdict["ok"] and raw:
            row = make_row(rid, "generated", doc, prefs.model_copy(update={"ai_mode": "auto", "ai_provider": None}),
                           raw, ai.get("model", args.teacher), verdict["score"], meta)
            append_jsonl(GENERATED, row)
            kept += 1
            print(f"{rid}  {time.time() - t:5.0f}s  KEEP    {msg}  | {room.title}", flush=True)
        else:
            why = "fallback designer was used" if ai.get("mode") == "fallback" else msg
            append_jsonl(REJECTED, {"id": rid, "reason": why, "room": room.title, "corrections": verdict["corrections"]})
            rejected += 1
            print(f"{rid}  {time.time() - t:5.0f}s  REJECT  {why}  | {room.title}", flush=True)
            if ai.get("mode") == "fallback":
                print(f"  teacher failed: {ai.get('fallback_reason', '')[:200]}")
    total = len(read_jsonl(GENERATED))
    print(f"\nThis run: kept {kept}, rejected {rejected} in {(time.time() - t_all) / 60:.1f} min. "
          f"Dataset now has {total} generated examples ({GENERATED}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
