"""Designer knowledge taught to Qwen through the prompt (no training, no GPU).

The knowledge lives in editable files under ``config.KNOWLEDGE_DIR``:

* ``designer_playbook.md`` - zones, work triangle, sizing and layout rules (system prompt)
* ``worked_examples.md``   - three rooms with the designer's thinking and decisions
* ``cabinet_guide.json``   - what every catalogue category is for; its short notes are
  printed next to each option Qwen can choose

MagickHome designers can correct these files to change how Qwen designs; no code
change is needed. ``python design_knowledge.py`` regenerates the human-readable
``knowledge/CABINET_GUIDE.md`` from the guide plus the live catalogue.
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import config
from catalog_loader import CatalogItem, canon, clean


def _path(name: str) -> Path:
    return Path(config.KNOWLEDGE_DIR) / name


@lru_cache(maxsize=1)
def playbook() -> str:
    return _path("designer_playbook.md").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def worked_examples() -> str:
    return _path("worked_examples.md").read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def cabinet_guide() -> dict[str, dict]:
    """Category entries keyed by canonical category name (dash / case insensitive)."""
    raw = json.loads(_path("cabinet_guide.json").read_text(encoding="utf-8"))
    out = {}
    for name, entry in raw.get("categories", {}).items():
        e = dict(entry)
        e["name"] = name
        e["combinations"] = {canon(k): v for k, v in entry.get("combinations", {}).items()}
        e["width_notes"] = {str(k): v for k, v in entry.get("width_notes", {}).items()}
        out[canon(name)] = e
    return out


def reload() -> None:
    """Pick up edited knowledge files without restarting the server."""
    for fn in (playbook, worked_examples, cabinet_guide):
        fn.cache_clear()


def note_for(item: CatalogItem) -> str:
    """Short designer note shown to Qwen next to one menu option."""
    entry = cabinet_guide().get(item.cat)
    if not entry:
        return ""
    parts = [entry["combinations"].get(canon(item.cabinet_combination)) or entry.get("prompt_note", "")]
    width = entry["width_notes"].get(str(item.width))
    if width:
        parts.append(width)
    return "; ".join(p for p in parts if p)


def system_prompt() -> str:
    return f"{playbook()}\n\n{worked_examples()}"


# ---------------------------------------------------------------------------
# Human-readable guide (for designers reviewing what Qwen is taught)
# ---------------------------------------------------------------------------

_STATUS = {"offered": "Qwen can choose it", "auto": "added by the solver", "not_yet": "not used by the generator yet"}


def render_cabinet_guide_md(catalog) -> str:
    """Markdown guide: every catalogue category, its sizes per door style and the designer notes."""
    guide = cabinet_guide()
    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "family": "", "styles": defaultdict(set), "heights": set(), "depths": set(), "combos": set()})
    for it in catalog.items:
        s = stats[it.cat]
        s["n"] += 1
        s["name"] = it.category
        s["family"] = clean(it.unit_family)
        s["styles"][clean(it.door_style) or "-"].add(it.width)
        s["heights"].add(it.height)
        s["depths"].add(it.depth)
        if it.cabinet_combination:
            s["combos"].add(clean(it.cabinet_combination))

    lines = [
        "# MagickHome cabinet guide",
        "",
        "What Qwen is taught about each catalogue category. Edit `knowledge/cabinet_guide.json` to change it,",
        "then run `python design_knowledge.py` to regenerate this page. Items marked **confirm** need a",
        "MagickHome designer to check the note.",
        "",
        f"Catalogue: {len(catalog.items)} SKUs in {len(stats)} categories ({catalog.source}).",
        "",
        "| Category | Family | Status | SKUs |",
        "|---|---|---|---|",
    ]
    order = sorted(stats, key=lambda c: (stats[c]["family"], c))
    for c in order:
        e = guide.get(c, {})
        lines.append(f"| {stats[c]['name']} | {stats[c]['family']} | {_STATUS.get(e.get('status'), 'no note yet')} | {stats[c]['n']} |")
    lines.append("")
    for c in order:
        s, e = stats[c], guide.get(c)
        lines += [f"## {s['name']}", ""]
        if e:
            flag = " **confirm**" if e.get("confirm") else ""
            lines += [
                f"- **Status:** {_STATUS.get(e.get('status'), e.get('status'))}{flag}",
                f"- **Zone:** {e.get('zone', '-')}",
                f"- **What it is:** {e.get('purpose', '-')}",
                f"- **Choose when:** {e.get('choose_when', '-')}",
                f"- **Avoid:** {e.get('avoid', '-')}",
            ]
            if e["combinations"]:
                lines.append("- **Combinations:** " + "; ".join(f"{k}: {v}" for k, v in e["combinations"].items()))
            if e["width_notes"]:
                lines.append("- **Sizes:** " + "; ".join(f"{k} mm: {v}" for k, v in e["width_notes"].items()))
        else:
            lines.append("- No designer note yet (add one to `cabinet_guide.json`).")
        widths = "; ".join(f"{st}: {', '.join(map(str, sorted(ws)))}" for st, ws in sorted(s["styles"].items()))
        lines += [
            f"- **Widths by door style (mm):** {widths}",
            f"- **Heights:** {', '.join(map(str, sorted(s['heights'])))} | **Depths:** {', '.join(map(str, sorted(s['depths'])))}",
            f"- **Catalogue combinations:** {', '.join(sorted(s['combos'])) or '-'}",
            "",
        ]
    return "\n".join(lines)


def _offline_catalog():
    """Local Assets-API snapshot if present (no network), else the xlsx."""
    from catalog_loader import Catalog, _items_from_records, _read_cache, load_catalog

    cached = _read_cache()
    if cached and cached[0]:
        return Catalog(_items_from_records(cached[0], "api-cache"), source="api-cache")
    return load_catalog(prefer_api=False)


if __name__ == "__main__":
    cat = _offline_catalog()
    out = _path("CABINET_GUIDE.md")
    out.write_text(render_cabinet_guide_md(cat), encoding="utf-8")
    missing = sorted({it.category for it in cat.items if it.cat not in cabinet_guide()})
    print(f"wrote {out} ({len(cat.items)} SKUs)")
    if missing:
        print("categories without a designer note:", missing)
