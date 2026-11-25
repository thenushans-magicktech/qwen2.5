"""Turn designs made in the web UI (designs/*.json) into training examples.

Only AI designs (Qwen / Gemma / Gemini) that pass the same quality bar as build_dataset.py are kept;
quick-rules designs and failed jobs are skipped. Run it any time - already harvested designs are skipped.

    python finetune/harvest_designs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import common  # noqa: F401  (sets sys.path)
from common import DATA, ROOT, append_jsonl, judge, make_row, read_jsonl
from ai_designer import KitchenPreferences
from add_sample import schema_model

HARVESTED = DATA / "harvested.jsonl"


def main() -> int:
    seen = {r["id"] for r in read_jsonl(HARVESTED)}
    kept = skipped = 0
    for path in sorted((ROOT / "designs").glob("*.json")):
        rid = f"ui-{path.stem}"
        if rid in seen:
            continue
        job = json.loads(path.read_text(encoding="utf-8"))
        report, result = job.get("report") or {}, job.get("result") or {}
        ai = report.get("ai") or {}
        if job.get("status") != "done" or ai.get("mode") not in ("ollama", "gemma", "gemini") or not ai.get("raw"):
            skipped += 1
            continue
        if ai.get("knowledge") is False:  # short-prompt answers (Gemini "own judgement", the fine-tuned student)
            skipped += 1
            continue
        prefs = KitchenPreferences(**{k: v for k, v in (job.get("preferences") or {}).items()
                                      if k in KitchenPreferences.model_fields})
        doc = result  # the finished scene; its cabinets are stripped before the room is read
        try:  # answers from an older prompt version / answer format do not fit today's schema: skip them
            schema_model(doc, prefs).model_validate(ai["raw"])
        except Exception:
            print(f"{rid}: answer does not match today's answer format (prompt {ai.get('prompt_version')}) - skipped")
            skipped += 1
            continue
        try:
            verdict = judge(doc, prefs, report)
        except Exception as exc:
            print(f"{rid}: cannot score ({exc})")
            skipped += 1
            continue
        if not verdict["ok"]:
            print(f"{rid}: below the bar - score {verdict['score']}, fixes {len(verdict['corrections'])}")
            skipped += 1
            continue
        append_jsonl(HARVESTED, make_row(rid, "ui", doc, prefs, ai["raw"], ai.get("model", "?"), verdict["score"],
                                         {"filename": job.get("filename")}))
        kept += 1
        print(f"{rid}: kept ({ai.get('model')}, score {verdict['score']})")
    print(f"\nKept {kept}, skipped {skipped}. {HARVESTED} has {len(read_jsonl(HARVESTED))} examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
