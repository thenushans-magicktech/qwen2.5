"""Hand-made sample designs (a designer's "this is how it should be done").

1. Create a sample from an empty floor plan - it is pre-filled with a draft answer (quick rules, or a
   finished UI job's answer) for you to edit:

       python finetune/add_sample.py new --scene plan.json --name tall_by_door [--prefs '{"fuel": "png"}'] [--job <job id>]

   This writes finetune/samples/tall_by_door.json and prints the prompt the model will see.
2. Edit "answer" in that file (codes and wall names must come from the prompt; reasons in plain words).
3. Check it:

       python finetune/add_sample.py check finetune/samples/tall_by_door.json

   merge.py also checks every sample and refuses broken ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common  # noqa: F401  (sets sys.path)
from common import ROOT, catalog_and_groups, short_messages
from ai_designer import KitchenPreferences, build_messages
from scene_parser import parse_scene, strip_items

SAMPLES = Path(__file__).resolve().parent / "samples"


def _load_scene(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8"))


def schema_model(doc: dict, prefs: KitchenPreferences):
    geometry = parse_scene(strip_items(doc))
    _, groups = catalog_and_groups(prefs.door_style, prefs.handle_type)
    _, Model, _ = build_messages(geometry, groups, prefs, "v2", knowledge=False)
    return Model


def check(sample: dict) -> list[str]:
    """Problems with a sample ([] = fine)."""
    try:
        doc = _load_scene(sample["scene"])
        prefs = KitchenPreferences(**sample.get("preferences", {}))
        schema_model(doc, prefs).model_validate(sample["answer"])
    except Exception as exc:
        return [str(exc)[:1500]]
    return []


def draft_answer(doc: dict, prefs: KitchenPreferences, job_id: str | None) -> dict:
    if job_id:
        job = json.loads((ROOT / "designs" / f"{job_id}.json").read_text(encoding="utf-8"))
        raw = (job.get("report") or {}).get("ai", {}).get("raw")
        if raw:
            return raw
        print("That job has no AI answer - starting from an empty template instead.")
    schema = schema_model(doc, prefs).model_json_schema()
    return {k: f"<{v.get('description') or v.get('type') or 'fill in'}>" for k, v in schema.get("properties", {}).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--scene", required=True, help="empty floor plan (magickhomes-scene JSON)")
    n.add_argument("--name", required=True)
    n.add_argument("--prefs", default="{}", help="customer needs as JSON, e.g. '{\"fuel\": \"png\"}'")
    n.add_argument("--job", help="start from this UI job's answer (designs/<id>.json)")
    c = sub.add_parser("check")
    c.add_argument("file")
    args = ap.parse_args()

    if args.cmd == "check":
        sample = json.loads(Path(args.file).read_text(encoding="utf-8"))
        problems = check(sample)
        print("OK - ready for merge.py" if not problems else "NOT VALID:\n" + "\n".join(problems))
        return 0 if not problems else 1

    doc = _load_scene(args.scene)
    prefs = KitchenPreferences(**json.loads(args.prefs))
    sample = {
        "scene": args.scene,
        "preferences": json.loads(args.prefs),
        "note": "Why this is a good design (for the people reviewing samples; not shown to the model).",
        "answer": draft_answer(doc, prefs, args.job),
    }
    SAMPLES.mkdir(exist_ok=True)
    out = SAMPLES / f"{args.name}.json"
    if out.exists():
        print(f"{out} already exists - pick another --name or edit it directly.")
        return 1
    out.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    msgs = short_messages(doc, prefs)
    print(msgs[1]["content"])
    print(f"\nWrote {out}. Edit its \"answer\" using the codes and wall names above, then run:\n"
          f"  python finetune/add_sample.py check {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
