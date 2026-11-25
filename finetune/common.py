"""Shared helpers for building the fine-tuning dataset.

A training row teaches the student model to answer the SHORT prompt (no playbook, no worked
examples, no cabinet notes, no rule hints) with an answer as good as the teacher's answer to the
FULL prompt. Row format (one JSON object per line in finetune/data/*.jsonl):

    {"id": ..., "source": "generated" | "ui" | "sample", "teacher": "<model>", "score": 100.0,
     "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "<answer JSON>"}],
     "meta": {...}}

The system + user messages are exactly what ``ai_designer.build_messages(..., knowledge=False)``
sends at run time, so the fine-tuned model sees the same prompt in training and in production.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from ai_designer import KitchenPreferences, build_messages, build_shortlist_v2  # noqa: E402
from catalog_loader import load_filtered_catalog  # noqa: E402
from scene_parser import parse_scene, strip_items  # noqa: E402

# Quality bar for a teacher answer to become a training example.
MIN_SCORE = float(os.getenv("FINETUNE_MIN_SCORE", "90"))
MAX_CORRECTIONS = int(os.getenv("FINETUNE_MAX_CORRECTIONS", "0"))


@lru_cache(maxsize=8)
def catalog_and_groups(door_style: str, handle_type: str):
    catalog = load_filtered_catalog(door_style, handle_type)
    return catalog, build_shortlist_v2(catalog)


def short_messages(doc: dict, prefs: KitchenPreferences) -> list[dict]:
    """The system + user messages the fine-tuned model gets at run time (knowledge pack left out)."""
    geometry = parse_scene(strip_items(doc))
    _, groups = catalog_and_groups(prefs.door_style, prefs.handle_type)
    messages, _, _ = build_messages(geometry, groups, prefs, "v2", knowledge=False)
    return messages


def judge(doc: dict, prefs: KitchenPreferences, report: dict) -> dict:
    """Score a finished design the way the designer benchmark does. ``ok`` = good enough to train on."""
    from designer_benchmark import score_design

    geometry = parse_scene(strip_items(doc))
    catalog, _ = catalog_and_groups(prefs.door_style, prefs.handle_type)
    s = score_design(geometry, catalog, report, prefs)
    issues = report.get("validation_issues") or []
    s["validation_issues"] = issues
    s["ok"] = (not issues and len(s["corrections"]) <= MAX_CORRECTIONS and s["score"] >= MIN_SCORE)
    return s


def make_row(row_id: str, source: str, doc: dict, prefs: KitchenPreferences, answer: dict,
             teacher: str, score: Optional[float], meta: Optional[dict] = None) -> dict:
    msgs = short_messages(doc, prefs)
    msgs.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return {"id": row_id, "source": source, "teacher": teacher, "score": score, "messages": msgs,
            "meta": dict(meta or {}, preferences=prefs.model_dump(exclude_defaults=True),
                         prompt_version="v2", knowledge=False)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
