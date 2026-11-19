"""Explain the design: for every key item, what was built, why, and who decided.

Sources:
* ``customer`` - the household stated this need in the preferences (enforced),
* ``qwen`` / ``rules-designer`` - the designer chose it; ``why`` is its own one-line reason,
* ``rules`` notes - where the rule engine changed or dropped the designer's choice.
"""
from __future__ import annotations

import re
from typing import Optional

from rule_engine import KitchenLayout, Proposal

_NAMES = (
    ("aquaguard", "Water purifier cabinet"), ("glass unit", "Glass display cabinet"),
    ("oven & micro", "Oven + microwave tower"), ("built-in oven", "Oven tower"), ("built-in microwave", "Microwave tower"),
    ("built-in wicker", "Basket pantry"), ("tall unit", "Tall pantry"), ("sink unit", "Sink unit"), ("hob unit", "Hob unit"),
    ("cylinder", "LPG cylinder unit"), ("spice", "Spice pull-out"), ("grain", "Grain trolley"), ("wicker", "Vegetable basket"),
)


def _name(category: str) -> str:
    c = (category or "").lower()
    return next((n for k, n in _NAMES if k in c), category)


def _desc(p) -> str:
    combo = p.item.cabinet_combination
    return f"{_name(p.item.category)} {p.item.width} mm" + (f" ({combo})" if combo else "")


def explain_design(proposal: Proposal, layout: KitchenLayout, prefs, geometry, catalog) -> list[dict]:
    W = geometry.walls

    def wname(wid: Optional[str]) -> str:
        return W[wid].name if wid in W else str(wid)

    roles: dict[str, list] = {}
    for p in layout.placements:
        roles.setdefault(p.role, []).append(p)
    reasons = proposal.reasons or {}
    designer = "qwen" if proposal.source == "ollama" else "rules-designer"
    notes = [m for m in layout.decisions + layout.warnings if "hinged" not in m]

    def rules(*keys: str) -> list[str]:
        # placement log lines ("Wall 3: display unit ... at 700-1300 mm") are not corrections
        return [m for m in notes if any(k in m for k in keys) and not re.search(r": (display|purifier) unit .* at \d", m)]

    out: list[dict] = []

    def add(item, built, why, source, rule_notes=(), need=None):
        out.append({"item": item, "built": built, "why": why or "", "source": source,
                    "rules": list(rule_notes), "need": need})

    # layout
    shape = {1: "One wall", 2: "L-shape", 3: "U-shape"}.get(len(layout.cabinet_walls), "Kitchen")
    walls = ", ".join(wname(w) for w in layout.cabinet_walls)
    cust_layout = bool(prefs.cabinet_walls) or prefs.layout != "auto"
    add("Layout", f"{shape} on {walls}", reasons.get("layout"), "customer" if cust_layout else designer,
        rules("cabinet walls changed", "added window wall", "re-chosen", "chosen by solver", "left without cabinets"),
        need=f"Customer asked for {prefs.layout} layout" if prefs.layout != "auto" else None)

    # sink
    sink = (roles.get("sink") or [None])[0]
    sink_rules = rules("sink moved", "sink offset")
    prop = catalog.get(proposal.sink_aid) if proposal.sink_aid else None
    if sink and prop and prop.width != sink.item.width:
        sink_rules.append(f"Designer chose {prop.width} mm; the rules used {sink.item.width} mm so it centres under the window.")
    add("Sink", f"{_desc(sink)} on {wname(sink.wall_id)}" if sink else "None", reasons.get("sink"), designer, sink_rules)

    # hob
    hob = (roles.get("hob") or [None])[0]
    add("Hob", f"{_desc(hob)} on {wname(hob.wall_id)}, chimney above" if hob else "Not placed",
        reasons.get("hob"), designer, rules("hob moved", "hob unit narrowed", "hob could not", "spice unit narrowed"))

    # LPG
    cyl = (roles.get("cylinder") or [None])[0]
    gas_need = {True: "Customer uses LPG", False: "Customer has piped gas"}.get(prefs.include_cylinder)
    add("LPG cylinder", _desc(cyl) if cyl else "None", reasons.get("lpg"),
        "customer" if prefs.include_cylinder is not None else designer, rules("cylinder unit dropped"), need=gas_need)

    # tall units
    talls = roles.get("tall") or []
    tower_need = {"none": "No built-in appliances", "microwave": "Needs a microwave", "oven": "Needs an oven",
                  "oven_microwave": "Needs an oven and a microwave"}.get(prefs.built_in)
    add("Tall units", ", ".join(f"{_desc(p)} on {wname(p.wall_id)}" for p in talls) or "None",
        reasons.get("tall"), "customer" if prefs.built_in != "auto" else designer,
        rules("tall unit"), need=tower_need)

    # water purifier
    pur = (roles.get("purifier") or [None])[0]
    pur_need = {"yes": "Needs a water purifier", "no": "No water purifier wanted"}.get(prefs.water_purifier)
    add("Water purifier", f"{_desc(pur)} on {wname(pur.wall_id)}" if pur else "None", reasons.get("purifier"),
        "customer" if prefs.water_purifier != "auto" else designer, rules("water purifier"), need=pur_need)

    # display
    disp = (roles.get("display") or [None])[0]
    disp_need = {"yes": "Wants a glass display", "no": "No glass display wanted"}.get(prefs.display_unit)
    add("Glass display", f"{_desc(disp)} on {wname(disp.wall_id)}" if disp else "None", reasons.get("display"),
        "customer" if prefs.display_unit != "auto" else designer, rules("display unit"), need=disp_need)

    # base cabinet style (drawers vs shutters) across the plain storage runs
    from ai_designer import base_storage_need

    store, how = base_storage_need(prefs)
    base_store = [p for p in layout.placements if p.tier == "base" and p.role == "storage"]
    n_drawer = sum(1 for p in base_store if "drawer" in (p.item.cabinet_combination or "").lower())
    add("Base cabinets", f"{n_drawer} of {len(base_store)} storage cabinets have drawers",
        {"drawers": "More drawers requested: storage runs use drawer units.",
         "shutters": "Shutter cabinets requested."}.get(store, "Standard shutter storage; drawers only where the designer asked."),
        "customer" if how else designer,
        need={"form": f"Asked for {store}", "brief": "The brief asks for drawers"}.get(how))

    # feature base units (what the designer asked for, per wall)
    feats = []
    for wid, aids in (proposal.base_prefs or {}).items():
        items = [catalog.get(a) for a in aids]
        names = ", ".join(f"{_name(i.category)} {i.width}" + (f" {i.cabinet_combination.lower()}" if i.cabinet_combination else "")
                          for i in items if i is not None)
        if names:
            feats.append(f"{wname(wid)}: {names}")
    add("Feature units", "; ".join(feats) or "Standard units only", reasons.get("features"), designer,
        rules("requested feature unit"))

    # corners (always the rules)
    corner_txt = []
    for cp in layout.corner_plans:
        units = [u for u in (cp.base_unit, cp.wall_unit, cp.loft_unit) if u is not None]
        tag = f"{wname(cp.primary)} / {wname(cp.secondary)}"
        corner_txt.append(f"{tag}: {len(units)} corner cabinet{'s' if len(units) != 1 else ''}" if units else f"{tag}: kept open")
    if corner_txt:
        add("Corners", "; ".join(corner_txt), "Corners take only corner cabinets, facing the room; "
            "a corner stays open when a door or window leaves no room for one.", "rules")
    return out
