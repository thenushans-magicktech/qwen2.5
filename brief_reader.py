"""Customer brief reader: Gemini understands the customer's free text - any language, any wording
(English, Tamil, Sinhala, Hindi, Tanglish, short notes) - and turns it into

* the form settings it clearly asks for (shape, walls, oven tower, lofts, drawers, fuel, budget...),
* a short, clear English brief for the designer (Qwen), and
* the wishes the generator cannot build (an island, a dishwasher...), so they are shown, not lost.

Only settings the customer left on "decide" are filled: a choice made in the form always wins over
the text (the clash is reported). Without ``GEMINI_API_KEY`` - or when Gemini is busy - nothing
changes and the designer gets the customer's own words, as before.

The brief text and the wall names / sizes are sent to Google (Gemini API) when this runs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import config
import rulebook

log = logging.getLogger(__name__)

# reading field -> (preferences field, the form's "not stated" value, label for the customer)
CHOICES: dict[str, tuple[str, Any, str]] = {
    "layout": ("layout", "auto", "Kitchen type"),
    "built_in": ("built_in", "auto", "Built-in appliances"),
    "water_purifier": ("water_purifier", "auto", "Water purifier unit"),
    "display_unit": ("display_unit", "auto", "Glass display wall unit"),
    "open_unit": ("open_unit", "auto", "Open shelf unit"),
    "crockery_unit": ("crockery_unit", "auto", "Tall crockery unit"),
    "base_storage": ("base_storage", "auto", "Base cabinets"),
    "wall_style": ("wall_style", "auto", "Wall cabinet doors"),
    "cooking": ("cooking", "auto", "Cooking"),
    "fuel": ("fuel", "auto", "Hob fuel"),
    "budget": ("budget", "auto", "Budget"),
}
ENUMS: dict[str, list[str]] = {
    "layout": ["auto", "single", "L", "U"],
    "built_in": ["auto", "none", "microwave", "oven", "oven_microwave"],
    "water_purifier": ["auto", "yes", "no"],
    "display_unit": ["auto", "yes", "no"],
    "open_unit": ["auto", "yes", "no"],
    "crockery_unit": ["auto", "yes", "no"],
    "base_storage": ["auto", "drawers", "mixed", "shutters"],
    "wall_style": ["auto", "shutters", "lift_up"],
    "cooking": ["auto", "heavy", "light"],
    "fuel": ["auto", "lpg", "png", "induction"],
    "budget": ["auto", "essential", "premium", "luxury"],
    "lofts": ["auto", "yes", "no"],
    "tall_units": ["auto", "yes", "no"],
}
VALUE_LABELS = {
    "single": "Straight (one wall)", "L": "L-shape", "U": "U-shape", "none": "none",
    "microwave": "microwave tower", "oven": "oven tower", "oven_microwave": "oven + microwave tower",
    "drawers": "more drawers", "mixed": "drawer + shutter mix", "shutters": "shutters",
    "lift_up": "lift-up (flap-up)", "heavy": "heavy, daily", "light": "light",
    "lpg": "LPG cylinder", "png": "piped gas (PNG)", "induction": "induction",
    "essential": "Essential", "premium": "Premium", "luxury": "Luxury", "yes": "yes", "no": "no",
}
CANNOT_BUILD = ("a kitchen island or breakfast counter", "a dishwasher, fridge or washing machine slot",
                "a parallel (two facing walls) kitchen", "moving or sizing one particular cabinet",
                "custom cabinet heights or depths", "lighting, flooring, wall paint or tiles")

SYSTEM = """You read a home kitchen customer's brief for a modular-kitchen designer. The brief may be in any
language or mix (English, Tamil, Sinhala, Hindi, Tanglish, short notes, spelling mistakes). Understand what the
family wants and fill the JSON:

- Set a field ONLY when the brief clearly asks for it (directly or by an obvious need: "we bake every weekend"
  -> an oven; "reheat food a lot" -> a microwave; "big family, need storage" -> drawers). Otherwise "auto".
- Tall cabinets / tall units / tall cupboards / pantry towers mean tall_units=yes. "No tall cabinets", "minimal
  cupboards", or "no tall storage" means tall_units=no. Tall cabinets should normally be included by the
  designer when the customer asks for them.
- Never guess walls: set cooking_wall / sink_wall / cabinet_walls only when the brief names a wall, or a wall
  feature that identifies exactly one wall (e.g. "sink near the window" when only one wall has a window).
  The sink wall must be a wall that meets the cooking wall at a corner (listed as "corners with"), or the same
  wall in a straight (one-wall) kitchen.
- english_brief: a short, clear English summary for the designer of EVERYTHING design-relevant in the brief,
  including wishes with no field (colours, look, who cooks, habits). Plain sentences, no invention.
- understood: one line per setting you filled, each with the customer's own words that led to it.
- not_possible: wishes this generator cannot build (it cannot build: """ + "; ".join(CANNOT_BUILD) + """).
- language: the language(s) the brief is written in."""


def _schema(wall_names: list[str]) -> dict:
    walls = ["auto"] + wall_names
    props: dict[str, dict] = {
        "language": {"type": "string"},
        "english_brief": {"type": "string", "description": "Clear English summary for the designer, max 450 characters"},
        "understood": {"type": "array", "maxItems": 15, "items": {
            "type": "object", "properties": {"setting": {"type": "string"}, "customer_words": {"type": "string"}},
            "required": ["setting", "customer_words"]}},
        "not_possible": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "household_size": {"type": "integer", "minimum": 0, "maximum": 20, "description": "0 = not stated"},
        "cooking_wall": {"type": "string", "enum": walls},
        "sink_wall": {"type": "string", "enum": walls},
        "cabinet_walls": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": wall_names},
                          "description": "Walls the customer wants cabinets on; empty = not stated"},
    }
    for name, values in ENUMS.items():
        props[name] = {"type": "string", "enum": values}
    return {"type": "object", "properties": props, "propertyOrdering": list(props), "required": list(props)}


def _room_lines(geometry) -> list[str]:
    lines = []
    for wid in geometry.wall_order:
        w = geometry.walls[wid]
        tags = (["window"] if w.windows else []) + (["door"] if w.doors else [])
        near = [geometry.walls[c].name for c in rulebook.near_walls(geometry, wid)]
        lines.append(f"- {w.name}: {w.length_mm / 1000:.1f} m{', ' + ', '.join(tags) if tags else ''}"
                     f"{'; corners with ' + ', '.join(near) if near else ''}")
    return lines


def available() -> bool:
    return config.BRIEF_READER == "gemini" and bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _post(model: str, body: dict, key: str) -> dict:
    """One try per model with a short timeout: reading a brief must not hold the design up - a busy
    model (or one out of quota) moves on to the next model at once. Tests stand in for this."""
    import requests

    from gemini_designer import API_BASE, ModelUnavailable, _daily_quota_used

    try:
        r = requests.post(f"{API_BASE}/models/{model}:generateContent", json=body, timeout=config.BRIEF_TIMEOUT_S,
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    except requests.exceptions.RequestException as exc:
        raise ModelUnavailable(f"{model}: no answer within {config.BRIEF_TIMEOUT_S:.0f} s ({exc.__class__.__name__})") from exc
    if r.status_code == 200:
        return r.json()
    if r.status_code == 429:
        raise ModelUnavailable(f"{model}: " + ("free daily quota used up (resets at midnight Pacific time)"
                                                if _daily_quota_used(r.text) else "rate limited"))
    if r.status_code in (500, 503):
        raise ModelUnavailable(f"{model}: busy (HTTP {r.status_code})")
    raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")


def ask_gemini(text: str, geometry) -> tuple[dict, dict]:
    """Gemini's reading of the brief (validated JSON) and call info. Raises RuntimeError when unavailable."""
    from gemini_designer import ModelUnavailable, model_chain

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    names = [geometry.walls[w].name for w in geometry.wall_order]
    prompt = "ROOM WALLS:\n" + "\n".join(_room_lines(geometry)) + f"\n\nCUSTOMER BRIEF:\n{text.strip()}"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": _schema(names),
                                 "maxOutputTokens": 4096, "temperature": 0.2}}
    skipped, t0 = [], time.time()
    for model in model_chain(config.BRIEF_MODEL or None):
        try:
            data = _post(model, body, key)
        except ModelUnavailable as exc:
            skipped.append(str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - e.g. a retired model (404): try the next one
            skipped.append(f"{model}: {str(exc)[:160]}")
            continue
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        try:
            reading = json.loads(raw)
        except ValueError as exc:
            skipped.append(f"{model}: reply was not JSON")
            continue
        if not isinstance(reading, dict):
            skipped.append(f"{model}: reply was not an object")
            continue
        return reading, {"model": data.get("modelVersion") or model, "seconds": round(time.time() - t0, 1),
                         "model_notes": skipped}
    raise RuntimeError("Gemini unavailable: " + "; ".join(skipped))


def apply_reading(reading: dict, prefs, geometry, text: str) -> tuple[Any, dict]:
    """Fill the preferences the customer left on "decide" from Gemini's reading. Returns
    ``(new_prefs, info)``; info lists what was applied, what the form kept, and what cannot be built."""
    by_name = {geometry.walls[w].name: w for w in geometry.wall_order}
    upd: dict[str, Any] = {}
    applied: list[dict] = []
    kept: list[str] = []

    def want(field: str, value: Any, label: str, shown: str, unset: bool, current_shown: str = ""):
        if unset:
            upd[field] = value
            applied.append({"setting": label, "value": shown, "field": field})
        elif getattr(prefs, field) != value:
            kept.append(f"{label}: the brief asks for {shown}, the form says {current_shown or getattr(prefs, field)} - form kept")

    for key, (field, unset_value, label) in CHOICES.items():
        v = reading.get(key)
        if v in ENUMS[key] and v != "auto":
            want(field, v, label, VALUE_LABELS.get(v, v), getattr(prefs, field) == unset_value,
                 VALUE_LABELS.get(getattr(prefs, field), str(getattr(prefs, field))))
    lofts = reading.get("lofts")
    if lofts in ("yes", "no"):
        want("include_loft", lofts == "yes", "Lofts", lofts, prefs.include_loft is None,
             "yes" if prefs.include_loft else "no")
    tall_units = reading.get("tall_units")
    if tall_units in ("yes", "no"):
        requested = tall_units == "yes"
        if tall_units == "yes" and not prefs.include_tall:
            kept.append("Tall units: the brief asks for tall cabinets, but the form says no - form kept")
        elif tall_units == "no" and prefs.include_tall:
            upd["include_tall"] = False
            applied.append({"setting": "Tall units", "value": tall_units, "field": "include_tall"})
        else:
            applied.append({"setting": "Tall units", "value": tall_units, "field": "include_tall"})
    size = reading.get("household_size")
    if isinstance(size, int) and 1 <= size <= 20:
        want("household_size", size, "Household", f"{size} people", prefs.household_size is None, f"{prefs.household_size} people")

    # walls: names -> ids; the shape's wall count and the corner rule are checked again in the pipeline
    walls = [by_name[n] for n in dict.fromkeys(reading.get("cabinet_walls") or []) if n in by_name]
    shape = upd.get("layout", prefs.layout)
    need = {"single": 1, "L": 2, "U": 3}.get(shape)
    if walls and not prefs.cabinet_walls and (need is None or len(walls) == need):
        upd["cabinet_walls"] = walls
        applied.append({"setting": "Kitchen walls", "value": ", ".join(geometry.walls[w].name for w in walls), "field": "cabinet_walls"})
    elif walls and prefs.cabinet_walls and set(walls) != set(prefs.cabinet_walls):
        kept.append("Kitchen walls: the brief names other walls than the ones ticked - ticked walls kept")
    for key, field, label in (("cooking_wall", "primary_wall", "Cooking (hob) wall"), ("sink_wall", "secondary_wall", "Sink wall")):
        wid = by_name.get(reading.get(key) or "")
        if wid:
            cur = getattr(prefs, field)
            want(field, wid, label, geometry.walls[wid].name, cur is None,
                 geometry.walls[cur].name if cur in geometry.walls else str(cur))

    english = " ".join(str(reading.get("english_brief") or "").split())[:500]
    if english:
        upd["style_notes"] = english
    info = {
        "original": text,
        "language": str(reading.get("language") or ""),
        "english_brief": english,
        "applied": applied,
        "understood": [f"{u.get('setting', '')} - \"{u.get('customer_words', '')}\""
                       for u in (reading.get("understood") or []) if isinstance(u, dict)][:15],
        "kept_form": kept,
        "not_possible": [str(x) for x in (reading.get("not_possible") or [])][:8],
        "changes": {k: v for k, v in upd.items() if k != "style_notes"},
    }
    return prefs.model_copy(update=upd), info


def read_brief(prefs, geometry) -> tuple[Any, Optional[dict]]:
    """The pipeline step: ``(prefs, info)``. info is None when there is no brief; when Gemini is not
    set up or fails, prefs are unchanged and info says why (the designer then reads the raw text)."""
    text = (prefs.style_notes or "").strip()
    if not text or prefs.read_brief is False:
        return prefs, None
    if not available():
        return prefs, {"original": text, "skipped": "Gemini is not set up (GEMINI_API_KEY) - the designer read the brief as written"}
    try:
        reading, call = ask_gemini(text, geometry)
    except RuntimeError as exc:
        log.warning("brief reader: %s", exc)
        return prefs, {"original": text, "skipped": f"{exc} - the designer read the brief as written"}
    new, info = apply_reading(reading, prefs, geometry, text)
    info.update(call)
    return new, info


def explanation_entry(info: dict) -> dict:
    """The "Customer brief" line of the design explanation."""
    if info.get("skipped"):
        return {"item": "Customer brief", "decision": None, "built": "read as written", "why": info["skipped"],
                "source": "customer", "details": [f"Brief: {info['original']}"], "rules": [], "need": None}
    details = [f"Your words: {info['original']}"]
    if info.get("english_brief") and info["english_brief"] != info["original"]:
        details.append(f"For the designer: {info['english_brief']}")
    details += [f"Set from the brief - {a['setting']}: {a['value']}" for a in info.get("applied", [])]
    details += info.get("kept_form", [])
    details += [f"Cannot be built here: {x}" for x in info.get("not_possible", [])]
    n = len(info.get("applied", []))
    return {"item": "Customer brief", "decision": None,
            "built": f"understood{' (' + info['language'] + ')' if info.get('language') else ''}; {n} setting{'s' if n != 1 else ''} filled",
            "why": f"Gemini ({info.get('model', '')}) read the brief; the form's own choices were kept.",
            "source": "customer", "details": details, "rules": [], "need": None}
