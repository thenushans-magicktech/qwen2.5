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
SURFACES = ("countertop", "backsplash")  # worktop and backsplash finishes (not budget-tiered: no rules yet)


API_TYPES = {"shutter": "Shutter", "countertop": "Countertop", "backsplash": "Backsplash", "floor": "Floor"}


@lru_cache(maxsize=1)
def api_records() -> tuple[list[dict], str]:
    """All MagickHome finish records from the textures API (read-only GET), else the last cache.
    ([], "presets") when neither is available - the sample-scene presets are used then."""
    import logging

    import requests

    log = logging.getLogger(__name__)
    url = config.TEXTURES_API_URL
    if url:
        try:
            items, page = [], 1
            while True:
                r = requests.get(url, params={"page": page, "limit": 1000}, timeout=20,
                                 headers={"Accept": "application/json", "User-Agent": "magickhome-kitchen-generator"})
                r.raise_for_status()
                data = r.json().get("data") or {}
                batch = data.get("items") or []
                items += batch
                if not batch or len(items) >= (data.get("total") or 0) or page >= 20:
                    break
                page += 1
            if items:
                try:
                    config.TEXTURES_CACHE_PATH.write_text(json.dumps({"url": url, "items": items}, ensure_ascii=False),
                                                          encoding="utf-8")
                except OSError:
                    pass
                return items, "api"
        except Exception as exc:  # offline / API down -> cache
            log.warning("textures API unavailable (%s)", exc)
    try:
        return json.loads(config.TEXTURES_CACHE_PATH.read_text(encoding="utf-8"))["items"], "api-cache"
    except (OSError, ValueError, KeyError):
        return [], "presets"


def _presets() -> dict:
    try:
        return json.loads(config.MATERIAL_PRESETS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _from_api(kind: str) -> dict[str, dict]:
    want = API_TYPES[kind]
    return {t["aid"]: t for t in api_records()[0]
            if t.get("aid") and t.get("type") == want and t.get("isActive", True) and t.get("isAvailable", True)}


@lru_cache(maxsize=4)
def surface_options(kind: str) -> dict[str, dict]:
    """aid -> finish record for a surface (countertop / backsplash / floor): the textures API,
    else the sample-scene presets."""
    out = _from_api(kind)
    if out:
        return out
    for t in _presets().get("fixtureTextures", {}).get(kind, []):
        if t.get("aid"):
            out.setdefault(t["aid"], t)
    return out


@lru_cache(maxsize=1)
def catalogue() -> dict[str, dict]:
    """aid -> shutter finish record (ShutterFrontM): the textures API (174 today: Merino laminates,
    MagickHome PU, Rehau acrylic), else the sample-scene presets."""
    out = _from_api("shutter")
    if out:
        return out
    for tier in ("base", "wall", "tall", "loft"):
        for t in _presets().get("shutterTextures", {}).get(tier, []):
            if t.get("aid"):
                out.setdefault(t["aid"], t)
    return out


def refresh() -> None:
    """Forget the cached finish data (re-read the API on next use)."""
    for f in (api_records, catalogue, surface_options, codes):
        f.cache_clear()


def is_premium(aid: Optional[str]) -> bool:
    """PU / acrylic (the tier rules' premium_categories), as opposed to laminate."""
    cats = rules().get("premium_categories") or ["PU"]
    return catalogue().get(aid or "", {}).get("category") in cats


def ai_shortlist(run: str, budget: Optional[str], per_group: int = 4) -> list[str]:
    """A small, varied set of finishes for the AI designer's prompt (174 is too many): the tier
    defaults, then per allowed category a spread of solids from light to dark plus a few woods."""
    ok = allowed(run, budget)
    out = [a for a in default(budget)[0].values() if a in ok]
    groups: dict[tuple, list[str]] = {}
    for a in ok:
        r = catalogue()[a]
        fam = "Wooden" if (r.get("family") or "") == "Wooden" else "Solids" if (r.get("family") or "") == "Solids" else None
        if fam:
            groups.setdefault((r.get("category"), fam), []).append(a)

    def light(a):
        c = (catalogue()[a].get("colorCode") or "").lstrip("#")
        try:
            return sum(int(c[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return 0

    for (cat, fam), aids in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        aids = sorted(aids, key=light)
        n = per_group if fam == "Solids" else max(1, per_group // 2)
        step = max(1, len(aids) // n)
        out += [a for a in aids[::step][:n] if a not in out]
    return out


@lru_cache(maxsize=1)
def rules() -> dict:
    try:
        return json.loads((config.KNOWLEDGE_DIR / "finish_rules.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tiers": {}, "defaults": {}, "match": []}


def _base_code(rec: dict, aid: str) -> str:
    name = "-".join("".join(ch for ch in w if ch.isalnum()) for w in (rec.get("name") or aid).upper().replace("_", " ").split())
    return f"{name}-{(rec.get('category') or 'FINISH').upper()}"


@lru_cache(maxsize=1)
def codes() -> dict[str, str]:
    """aid -> unique readable code (e.g. ``PURE-LEGNO-OAK-LAMINATE``; a clash adds the finish and aid)."""
    out, seen = {}, {}
    for aid, rec in catalogue().items():
        c = _base_code(rec, aid)
        seen[c] = seen.get(c, 0) + 1
        out[aid] = c
    for aid, c in list(out.items()):
        if seen[c] > 1:
            rec = catalogue()[aid]
            out[aid] = f"{c}-{(rec.get('shutter') or '').upper().replace(' ', '-')}-{aid}".replace("--", "-")
    return out


def code(aid: str) -> str:
    """Readable option code for prompts and schema enums, e.g. ``PURE-LEGNO-OAK-LAMINATE``."""
    return codes().get(aid) or _base_code({}, aid)


def describe(aid: Optional[str]) -> str:
    rec = catalogue().get(aid or "")
    if not rec:
        return "standard finish"
    wood = ", wood texture" if (rec.get("family") or "").lower() == "wooden" else ""
    return f"{rec['name']} ({rec.get('category')} {(rec.get('shutter') or '').lower()}{wood}, {rec.get('brand')})"


def summary(aid: str, records: Optional[dict] = None) -> dict:
    rec = (records if records is not None else catalogue()).get(aid, {})
    return {"aid": aid, "code": code(aid), "name": rec.get("name"), "category": rec.get("category"),
            "family": rec.get("family"), "shutter": rec.get("shutter"), "brand": rec.get("brand"),
            "color": rec.get("colorCode") if rec.get("colorCode") not in (None, "Nil") else None,
            "thumbnail": rec.get("thumbnailUrl"), "texture": bool(rec.get("useTexture")),
            "texture_width": rec.get("textureWidth"), "texture_height": rec.get("textureHeight")}


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
    max_pu = tier.get("max_premium_runs", tier.get("max_pu_runs"))
    pu = [run for run in RUNS if is_premium(pal.get(run))]
    if max_pu is not None and len(pu) > max_pu:
        # too many PU / acrylic runs for the tier: the run the designer (not the customer) chose goes back to laminate
        for run in sorted(pu, key=lambda r: src.get(r) == "customer")[: len(pu) - max_pu]:
            lam = (dflt.get(run) if dflt.get(run) in allowed(run, budget) and not is_premium(dflt.get(run)) else None) \
                or next((a for a in allowed(run, budget) if not is_premium(a)), None)
            if lam and src.get(run) != "customer":
                notes.append(f"{run} run: only {max_pu} PU / acrylic run allowed in the {budget} tier - "
                             f"{describe(pal[run])} changed to {describe(lam)}")
                pal[run], src[run] = lam, "rules"
            elif src.get(run) == "customer":
                notes.append(f"{run} run: PU / acrylic on more runs than the {budget} tier's {max_pu} - kept (customer's choice)")
    return pal, src, notes


def texture_list(available: list[dict], aid: Optional[str], drop_sku: bool = False,
                 records: Optional[dict] = None) -> list[dict]:
    """``available`` finish records with the chosen finish first (the scene applies textures[0])."""
    rec = (records if records is not None else catalogue()).get(aid or "")
    if rec is None:
        return copy.deepcopy(available)
    first = copy.deepcopy(next((t for t in available if t.get("aid") == aid), rec))
    if drop_sku:
        first.pop("SKU_ID", None)
    return [first] + [copy.deepcopy(t) for t in available if t.get("aid") != aid]


def surface_choice(stated: dict) -> dict:
    """The customer's valid worktop / backsplash choices (others keep the scene default)."""
    return {k: stated[k] for k in SURFACES if stated.get(k) in surface_options(k)}


def apply_to_scene(doc: dict, palette: dict) -> int:
    """Re-apply a palette to an existing generated scene (a finish changed after the design):
    cabinet fronts by run, their slabs / fillers, the worktop and the backsplash. Returns the
    number of nodes changed."""
    changed = 0
    for n in (doc.get("scene") or doc).get("nodes", {}).values():
        meta = n.get("metadata") or {}
        tex = meta.get("textures")
        if n.get("type") != "item" or not tex:
            continue
        kind = meta.get("fixtureKind")
        if (n.get("asset") or {}).get("category") == "kitchen":
            new = texture_list(tex, palette.get(RUN_OF_TIER.get((n.get("placement") or {}).get("level"), "")))
        elif kind in ("slab", "filler"):
            new = texture_list(tex, palette.get(RUN_OF_TIER.get(meta.get("level"), "")), drop_sku=True)
        elif kind in SURFACES:
            new = texture_list(tex, palette.get(kind), records=surface_options(kind))
        else:
            continue
        if new and new[0].get("aid") != tex[0].get("aid"):
            meta["textures"] = new
            changed += 1
    return changed
