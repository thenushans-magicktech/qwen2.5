"""Shutter finishes: the catalogue, the budget-tier rules and the palette applied to a design.

* Catalogue: the MagickHome finish records in ``data/material_presets.json`` (shutterTextures),
  one record per finish ``aid`` (e.g. MHS135 "Pure Legno Oak", Laminate / Merino). The cabinets
  API carries no finishes, so this list grows when more finish records are added there.
* Rules: ``knowledge/finish_rules.json`` - which finish categories (Laminate / PU) each budget
  tier allows on the base run (base + tall units) and the wall run (wall units + lofts), how many
  runs may be PU, the default palette per tier, and the finish-match rules.
* Palette: ``{"base": aid, "wall": aid}`` - the customer's stated finish wins, else the AI
  designer's choice, else the tier default; then the tier rules are enforced.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Optional

import config

RUNS = ("base", "wall")
RUN_OF_TIER = {"base": "base", "tall": "base", "wall": "wall", "loft": "wall"}


@lru_cache(maxsize=1)
def catalogue() -> dict[str, dict]:
    """aid -> finish record (shutter texture), in preset order."""
    try:
        presets = json.loads(config.MATERIAL_PRESETS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict] = {}
    for tier in ("base", "wall", "tall", "loft"):
        for t in presets.get("shutterTextures", {}).get(tier, []):
            if t.get("aid"):
                out.setdefault(t["aid"], t)
    return out


@lru_cache(maxsize=1)
def rules() -> dict:
    try:
        return json.loads((config.KNOWLEDGE_DIR / "finish_rules.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tiers": {}, "defaults": {}, "match": []}


def code(aid: str) -> str:
    """Readable option code for prompts and schema enums, e.g. ``OAK-LEGNO-LAMINATE``."""
    rec = catalogue().get(aid, {})
    name = "-".join(w for w in (rec.get("name") or aid).upper().replace("_", " ").split())
    return f"{name}-{(rec.get('category') or 'FINISH').upper()}"


def describe(aid: Optional[str]) -> str:
    rec = catalogue().get(aid or "")
    if not rec:
        return "standard finish"
    wood = ", wood texture" if (rec.get("family") or "").lower() == "wooden" else ""
    return f"{rec['name']} ({rec.get('category')} {(rec.get('shutter') or '').lower()}{wood}, {rec.get('brand')})"


def summary(aid: str) -> dict:
    rec = catalogue().get(aid, {})
    return {"aid": aid, "code": code(aid), "name": rec.get("name"), "category": rec.get("category"),
            "family": rec.get("family"), "shutter": rec.get("shutter"), "brand": rec.get("brand"),
            "color": rec.get("colorCode") if rec.get("colorCode") not in (None, "Nil") else None,
            "thumbnail": rec.get("thumbnailUrl"), "texture": bool(rec.get("useTexture"))}


def allowed(run: str, budget: Optional[str]) -> list[str]:
    """Finish aids the tier allows on ``run`` (all finishes when no budget is stated)."""
    cats = (rules().get("tiers", {}).get(budget or "", {}) or {}).get(run)
    return [aid for aid, rec in catalogue().items() if cats is None or rec.get("category") in cats]


def default(budget: Optional[str]) -> tuple[dict, str]:
    d = rules().get("defaults", {}).get(budget or "auto") or rules().get("defaults", {}).get("auto", {})
    pal = {run: d.get(run) for run in RUNS if d.get(run) in catalogue()}
    for run in RUNS:  # data changed: fall back to the first allowed finish
        if run not in pal and allowed(run, budget):
            pal[run] = allowed(run, budget)[0]
    return pal, d.get("why", "")


def resolve(stated: dict, designer: dict, budget: Optional[str]) -> tuple[dict, dict, list[str]]:
    """Final palette, the source of each run (customer / designer / default) and rule notes."""
    pal, src, notes = {}, {}, []
    dflt, _ = default(budget)
    tier = rules().get("tiers", {}).get(budget or "", {})
    for run in RUNS:
        ok = allowed(run, budget)
        if stated.get(run) in catalogue():
            pal[run], src[run] = stated[run], "customer"
            if stated[run] not in ok:
                notes.append(f"{run} run: {describe(stated[run])} is outside the {budget} tier - kept (customer's choice)")
        elif designer.get(run) in ok:
            pal[run], src[run] = designer[run], "designer"
        else:
            if designer.get(run):
                notes.append(f"{run} run: {describe(designer[run])} is not allowed in the {budget} tier - tier default used")
            if dflt.get(run):
                pal[run], src[run] = dflt[run], "rules"
    max_pu = tier.get("max_pu_runs")
    pu = [run for run in RUNS if catalogue().get(pal.get(run, ""), {}).get("category") == "PU"]
    if max_pu is not None and len(pu) > max_pu:
        # too many PU runs for the tier: the run the designer (not the customer) chose goes back to laminate
        for run in sorted(pu, key=lambda r: src.get(r) == "customer")[: len(pu) - max_pu]:
            lam = next((a for a in allowed(run, budget) if catalogue()[a].get("category") != "PU"), None)
            if lam and src.get(run) != "customer":
                notes.append(f"{run} run: only {max_pu} PU run allowed in the {budget} tier - "
                             f"{describe(pal[run])} changed to {describe(lam)}")
                pal[run], src[run] = lam, "rules"
            elif src.get(run) == "customer":
                notes.append(f"{run} run: PU on more runs than the {budget} tier's {max_pu} - kept (customer's choice)")
    return pal, src, notes


def texture_list(available: list[dict], aid: Optional[str], drop_sku: bool = False) -> list[dict]:
    """``available`` finish records with the chosen finish first (the scene applies textures[0])."""
    rec = catalogue().get(aid or "")
    if rec is None:
        return copy.deepcopy(available)
    first = copy.deepcopy(next((t for t in available if t.get("aid") == aid), rec))
    if drop_sku:
        first.pop("SKU_ID", None)
    return [first] + [copy.deepcopy(t) for t in available if t.get("aid") != aid]
