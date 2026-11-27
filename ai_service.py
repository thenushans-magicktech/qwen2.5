"""AI decision service for other apps (e.g. the MagickTech design backend).

The fine-tuned designer (magick-lfm3 / magick-gemma ...) makes the high-level kitchen decisions - which walls get
cabinets, sink and hob wall + position, tall units, lofts, storage style - with a reason for each. The calling app keeps
its own rule engine for the exact placement; it only takes these decisions as inputs.

    POST /api/ai/decisions  {scene, preferences}                    -> decisions (+ state_id for later changes)
    POST /api/ai/change     {state_id, message, focus?}             -> what the customer asked, rule-book conflicts,
                                                                       new decisions (rules) and override decisions

``scene`` is a MagickHomes / Pascal scene ({nodes, rootNodeIds} or the exported {scene: {...}} document);
``preferences`` uses the design backend's intake keys (kitchenType, budgetTier, hobFuel, appliancesRequired,
cabinetWallIds, householdSize, cookingIntensity) - unknown keys are ignored.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any, Optional

import config
from ai_designer import KitchenPreferences

STATES: "OrderedDict[str, dict]" = OrderedDict()  # state_id -> pseudo job (scene + preferences + report)
KEEP_STATES = 50

_LAYOUT = {"straight": "single", "l_shape": "L", "u_shape": "U"}
_FUEL = {"cylinder lpg": "lpg", "lpg": "lpg", "piped gas": "png", "png": "png", "induction": "induction",
         "electric": "induction"}


def scene_doc(scene: dict) -> dict:
    """Accept the inner {nodes, rootNodeIds} or the exported {format, scene: {...}} document."""
    if "scene" in scene and isinstance(scene["scene"], dict):
        return scene
    return {"format": "magickhomes-scene", "version": 1, "scene": scene}


def to_preferences(p: dict, model: Optional[str] = None) -> KitchenPreferences:
    """Design-backend intake keys -> this designer's preferences (the AI must answer: no rule-based fallback)."""
    p = p or {}
    out: dict[str, Any] = {"ai_mode": "ollama", "ai_provider": "local", "local_model": model or config.LOCAL_MODEL}
    if p.get("kitchenType") in _LAYOUT:
        out["layout"] = _LAYOUT[p["kitchenType"]]
    if p.get("cabinetWallIds"):
        out["cabinet_walls"] = list(p["cabinetWallIds"])
    if str(p.get("budgetTier", "")).lower() in ("essential", "premium", "luxury"):
        out["budget"] = str(p["budgetTier"]).lower()
    fuel = _FUEL.get(str(p.get("hobFuel", "")).strip().lower())
    if fuel:
        out["fuel"] = fuel
    cook = str(p.get("cookingIntensity", "")).lower()
    if cook in ("heavy", "light"):
        out["cooking"] = cook
    size = str(p.get("householdSize", ""))
    if size:
        out["household_size"] = 5 if "5" in size else 4 if "3" in size or "4" in size else 2
    apps = {str(a).lower() for a in (p.get("appliancesRequired") or [])}
    oven, micro = any("oven" in a for a in apps), any("microwave" in a for a in apps)
    if apps:
        out["built_in"] = "oven_microwave" if oven and micro else "oven" if oven else "microwave" if micro else "none"
    return KitchenPreferences(**out)


def _decisions(report: dict) -> dict:
    """The design's decisions in wall ids and metres (the caller's coordinate frame: along the wall from its start)."""
    prop, lay = report.get("proposal") or {}, report.get("layout") or {}
    placements = lay.get("placements") or {}

    def unit(role: str) -> Optional[dict]:
        for wid, items in placements.items():
            for it in items:
                if it.get("role") == role:
                    return {"wall_id": wid, "center_m": round((it["start_mm"] + it["end_mm"]) / 2000, 3),
                            "width_mm": it.get("width"), "category": it.get("category")}
        return None

    reasons = prop.get("reasons") or {}
    # the walls that really carry cabinets (the layout solver keeps the customer's layout type)
    walls = [w for w in (lay.get("cabinet_walls") or prop.get("cabinet_walls") or []) if placements.get(w)]
    sink, hob = unit("sink"), unit("hob")
    for d, key in ((sink, "sink"), (hob, "hob")):
        if d:
            d["reason"] = reasons.get(key, "")
    return {
        "kitchen_type": {1: "straight", 2: "l_shape"}.get(len(walls), "u_shape"),
        "cabinet_wall_ids": walls,
        "sink": sink, "hob": hob,
        "tall_units": [{"wall_id": wid, "category": it["category"], "width_mm": it["width"],
                        "center_m": round((it["start_mm"] + it["end_mm"]) / 2000, 3)}
                       for wid, items in placements.items() for it in items if it.get("tier") == "tall"],
        "lofts": bool((report.get("heights") or {}).get("lofts")),
        "base_storage": prop.get("base_storage"), "wall_style": prop.get("wall_style"),
        "reasons": reasons, "notes": prop.get("notes", ""),
    }


def _remember(job: dict) -> str:
    sid = uuid.uuid4().hex[:12]
    STATES[sid] = job
    while len(STATES) > KEEP_STATES:
        STATES.popitem(last=False)
    return sid


def decisions(scene: dict, preferences: dict, model: Optional[str] = None) -> dict:
    import main

    t0 = time.time()
    doc, prefs = scene_doc(scene), to_preferences(preferences, model)
    out, report = main.generate_kitchen(doc, prefs)
    ai = report.get("ai") or {}
    job = {"id": "svc", "result": out, "report": report, "preferences": prefs.model_dump(), "version": 0}
    return {"state_id": _remember(job), "model": ai.get("model"), "seconds": round(time.time() - t0, 1),
            "ai_seconds": ai.get("seconds"), "decisions": _decisions(report),
            "rulebook_corrections": [{"rule": f"R{a.get('rule')}", "text": a.get("text"), "designer": a.get("designer")}
                                     for a in (report.get("rulebook") or {}).get("applied", []) if a.get("changed")]}


def change(state_id: str, message: str, focus: Optional[dict] = None) -> dict:
    """The customer's change request -> new decisions, checked against the rule book."""
    import design_edit

    job = STATES.get(state_id)
    if job is None:
        raise KeyError("unknown or expired state_id - ask for decisions again")
    t0 = time.time()
    r = design_edit.propose(job, message, focus)
    if r is None:
        return {"type": "no_change", "seconds": round(time.time() - t0, 1),
                "answer": "I couldn't find a design change in that - try e.g. 'move the hob to Wall 1'."}
    pend = design_edit.PENDING[r["id"]]
    out = {k: r.get(k) for k in ("summary", "changes", "conflicts", "not_built", "issues", "guidelines",
                                 "override_issues", "answer", "choices", "model", "fields")}
    out["type"] = "change"
    out["seconds"] = round(time.time() - t0, 1)
    for name, v in pend["variants"].items():  # "rules" and, when a rule was broken, "override"
        new_job = {**job, "result": v["scene"], "report": v["report"], "preferences": v["prefs"]}
        out[f"{name}_decisions"] = _decisions(v["report"])
        out[f"{name}_state_id"] = _remember(new_job)
    return out


def state(state_id: str) -> dict:
    """The decisions of an earlier answer (e.g. the variant the customer chose after a change)."""
    job = STATES.get(state_id)
    if job is None:
        raise KeyError("unknown or expired state_id - ask for decisions again")
    ai = (job.get("report") or {}).get("ai") or {}
    return {"state_id": state_id, "model": ai.get("model"), "decisions": _decisions(job["report"]),
            "edits": (job.get("report") or {}).get("edits", [])}


def ask(state_id: str, question: str, focus: Optional[dict] = None, history: Optional[list] = None) -> dict:
    """Answer a question about the design ("why is the sink here?"), from the design's own decisions and reasons."""
    import chat

    job = STATES.get(state_id)
    if job is None:
        raise KeyError("unknown or expired state_id - ask for decisions again")
    # the calling app's rule engine placed the final cabinets: answer from decisions + reasons, not our own layout
    r = chat.answer(job, question, history or [], focus, placements=False)
    return {"type": "answer", "answer": r["answer"], "model": r.get("model"), "seconds": r.get("seconds"),
            "sources": r.get("sources", [])}
