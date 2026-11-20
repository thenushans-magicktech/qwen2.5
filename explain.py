"""Explain the design: every decision, what was built, why, and the facts behind it.

Each entry:

* ``item``      - the decision (Layout, Sink, Hob, ...)
* ``decision``  - what the designer (Qwen or the quick rules) chose, when it differs from what was built
* ``built``     - what is in the final kitchen
* ``why``       - the designer's own reason (Qwen writes it before choosing)
* ``details``   - facts measured by the system: room geometry, catalogue data, rule checks,
                  the other options that were available. Not written by Qwen.
* ``rules``     - where the rule engine changed or dropped the designer's choice
* ``source``    - ``customer`` (a stated need), ``qwen`` / ``rules-designer`` (the designer), ``rules``
* ``need``      - the customer's stated need, if any
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

import config
from rule_engine import KitchenLayout, Proposal

_NAMES = (
    ("tall glass", "Crockery unit"), ("wall open unit", "Open shelf unit"), ("lift up", "Lift-up wall cabinet"),
    ("aquaguard", "Water purifier cabinet"), ("glass unit", "Glass display cabinet"),
    ("oven & micro", "Oven + microwave tower"), ("built-in oven", "Oven tower"), ("built-in microwave", "Microwave tower"),
    ("built-in wicker", "Basket pantry"), ("tall unit", "Tall pantry"), ("sink unit", "Sink unit"), ("hob unit", "Hob unit"),
    ("cylinder", "LPG cylinder unit"), ("spice", "Spice pull-out"), ("grain", "Grain trolley"), ("wicker", "Vegetable basket"),
    ("wall l corner", "Corner wall cabinet"), ("loft", "Loft cabinet"), ("blind corner", "Blind corner cabinet"),
    ("wall unit", "Wall cabinet"), ("base unit", "Base cabinet"),
)


def _name(category: str) -> str:
    c = (category or "").lower().replace("–", "-")
    return next((n for k, n in _NAMES if k in c), category)


def _item(it) -> str:
    combo = it.cabinet_combination
    return f"{_name(it.category)} {it.width} mm" + (f" ({combo})" if combo else "")


def _desc(p) -> str:
    return _item(p.item)


def _size(it) -> str:
    return f"{it.width} x {it.depth} x {it.height} mm (W x D x H)"


def explain_design(proposal: Proposal, layout: KitchenLayout, prefs, geometry, catalog) -> list[dict]:
    from ai_designer import _free_segments, base_storage_need, build_menu, build_shortlist_v2
    from design_knowledge import note_for

    W = geometry.walls

    def wname(wid: Optional[str]) -> str:
        return W[wid].name if wid in W else str(wid)

    def free_run(wid: str) -> float:
        return max((b - a for a, b in _free_segments(geometry, wid)), default=0.0)

    def wall_fact(wid: str) -> str:
        w = W[wid]
        bits = [f"{w.length_mm} mm long"]
        bits += [f"window {o.start_mm:.0f}-{o.end_mm:.0f} mm" for o in w.windows]
        bits += [f"door {o.start_mm:.0f}-{o.end_mm:.0f} mm" for o in w.doors]
        bits.append(f"longest free run {free_run(wid):.0f} mm")
        return f"{w.name}: " + ", ".join(bits)

    def world(p, depth_m: float = 0.3):
        w = W[p.wall_id]
        return w.to_world(p.center_mm / 1000.0, w.thickness_m / 2 + depth_m)

    def neighbours(p):
        same = sorted((q for q in layout.placements if q.wall_id == p.wall_id and q.tier in ("base", "tall") and q is not p),
                      key=lambda q: q.start_mm)
        left = next((q for q in reversed(same) if q.end_mm <= p.start_mm + 1), None)
        right = next((q for q in same if q.start_mm >= p.end_mm - 1), None)
        return left, right

    def what(it) -> str:
        note = note_for(it)
        return f"{_size(it)}" + (f" - {note}" if note else "")

    try:
        menu = build_menu(build_shortlist_v2(catalog))
    except Exception:  # pragma: no cover - catalogue without a v2 menu
        menu = {}

    def offered(group: str) -> str:
        opts = [f"{it.width}" + (f" ({note_for(it).split('; ')[-1]})" if note_for(it) else "") for _, it in menu.get(group, [])]
        return "; ".join(opts)

    roles: dict[str, list] = {}
    for p in layout.placements:
        roles.setdefault(p.role, []).append(p)
    reasons = proposal.reasons or {}
    designer = {"ollama": "qwen", "gemini": "gemini"}.get(proposal.source, "rules-designer")
    notes = [m for m in layout.decisions + layout.warnings if "hinged" not in m]

    def rules(*keys: str) -> list[str]:
        # placement log lines ("Wall 3: display unit ... at 700-1300 mm") are not corrections
        return [m for m in notes if any(k in m for k in keys)
                and not re.search(r": (display|purifier|open) unit .* at \d", m) and not m.startswith("heights:")]

    out: list[dict] = []

    def add(item, built, why, source, details=(), rule_notes=(), need=None, decision=None):
        out.append({"item": item, "decision": decision, "built": built, "why": why or "", "source": source,
                    "details": [d for d in details if d], "rules": list(rule_notes), "need": need})

    sink = (roles.get("sink") or [None])[0]
    hob = (roles.get("hob") or [None])[0]

    # -- layout -------------------------------------------------------------------------
    shape = {1: "One wall", 2: "L-shape", 3: "U-shape"}.get(len(layout.cabinet_walls), "Kitchen")
    walls = ", ".join(wname(w) for w in layout.cabinet_walls)
    cust_layout = bool(prefs.cabinet_walls) or prefs.layout != "auto"
    det = [wall_fact(w) for w in layout.cabinet_walls]
    unused = [w for w in geometry.wall_order if w not in layout.cabinet_walls]
    if unused:
        det.append("Not used: " + "; ".join(wall_fact(w) for w in unused))
    if len(layout.cabinet_walls) == 3 and sink is not None:
        lo, hi = W[sink.wall_id].usable
        det.append(f"U-shape aisle between the side runs: {(hi - lo) - 2 * config.CORNER_DEPTH_MM:.0f} mm "
                   f"(needs {config.MIN_AISLE_MM}+)")
    det.append(f"{len(layout.corner_plans)} inside corner(s) between cabinet runs")
    asked = [wname(w) for w in proposal.cabinet_walls if w in W]
    add("Layout", f"{shape} on {walls}", reasons.get("layout"), "customer" if cust_layout else designer, det,
        rules("cabinet walls changed", "added window wall", "re-chosen", "chosen by solver", "left without cabinets"),
        need=f"Customer asked for {prefs.layout} layout" if prefs.layout != "auto" else None,
        decision=f"Walls {', '.join(asked)}" if asked and set(proposal.cabinet_walls) != set(layout.cabinet_walls) else None)

    # -- sink ---------------------------------------------------------------------------
    det = []
    sink_rules = rules("sink moved", "sink offset")
    prop = catalog.get(proposal.sink_aid) if proposal.sink_aid else None
    if sink:
        det.append("Unit: " + what(sink.item))
        win = layout.sink_window
        if win is not None:
            det.append(f"Window {win.width_mm:.0f} mm wide, centre at {win.center_mm:.0f} mm; sink centre at "
                       f"{sink.center_mm:.0f} mm ({'centred' if abs(sink.center_mm - win.center_mm) < 1 else 'offset'})")
        det.append(f"{wname(sink.wall_id)} free run {free_run(sink.wall_id):.0f} mm")
        if menu.get("sink"):
            det.append("Sizes on offer: " + offered("sink"))
        if prop and prop.width != sink.item.width:
            sink_rules.append(f"Designer chose {prop.width} mm; the rules used {sink.item.width} mm so it centres under the window.")
    add("Sink", f"{_desc(sink)} on {wname(sink.wall_id)}" if sink else "None", reasons.get("sink"), designer, det, sink_rules,
        decision=_item(prop) if prop and sink and prop.aid != sink.item.aid else None)

    # -- hob ----------------------------------------------------------------------------
    det = []
    prop = catalog.get(proposal.hob_aid) if proposal.hob_aid else None
    if hob:
        hw = W[hob.wall_id]
        det.append("Unit: " + what(hob.item))
        det.append(f"{hw.name} has {'no window - a solid wall for the chimney' if not hw.windows else 'a window'}")
        if hw.doors:
            gap = min(max(d.start_mm - hob.end_mm, hob.start_mm - d.end_mm) for d in hw.doors)
            det.append(f"Nearest door on this wall is {gap:.0f} mm from the hob (keep 600+)")
        else:
            det.append("No door on this wall")
        if proposal.hob_center_mm is not None and proposal.hob_wall_id == hob.wall_id:
            det.append(f"Asked for the hob centre at {proposal.hob_center_mm:.0f} mm; built at {hob.center_mm:.0f} mm")
        left, right = neighbours(hob)
        det.append(f"Beside it: {(_desc(left) if left else 'free end')} / {(_desc(right) if right else 'free end')}")
        det.append(f"Chimney {config.CHIMNEY_WIDTH_MM} mm wide above it at {config.POS_Y_M['wall']:.2f} m; "
                   f"no wall cabinets in that zone")
        if sink is not None:
            dist = math.dist(world(sink), world(hob))
            same = sink.wall_id == hob.wall_id
            det.append(f"Sink to hob: {dist:.1f} m ({'same wall' if same else 'different walls'}; comfortable range 1.2-2.7 m)")
    add("Hob", f"{_desc(hob)} on {wname(hob.wall_id)}, chimney above" if hob else "Not placed",
        reasons.get("hob"), designer, det,
        rules("hob moved", "hob unit narrowed", "hob could not"),
        decision=_item(prop) if prop and hob and prop.aid != hob.item.aid else None)

    # -- spice --------------------------------------------------------------------------
    spice = (roles.get("spice") or [None])[0]
    det = []
    if spice and hob:
        left, right = neighbours(hob)
        det.append("Unit: " + what(spice.item))
        det.append("Right beside the hob" if spice in (left, right) else "Not directly beside the hob")
    add("Spice pull-out", _desc(spice) if spice else "None",
        "Masalas and oils within reach of the cook." if spice else "", designer, det, rules("spice unit narrowed"))

    # -- LPG ----------------------------------------------------------------------------
    cyl = (roles.get("cylinder") or [None])[0]
    gas_need = {True: "Customer uses LPG", False: "Customer has piped gas"}.get(prefs.include_cylinder)
    det = []
    if cyl:
        det.append("Unit: " + what(cyl.item))
        det.append(f"On {wname(cyl.wall_id)} at {cyl.start_mm:.0f}-{cyl.end_mm:.0f} mm, beside the hob")
    elif hob:
        det.append(f"Hob wall free run {free_run(hob.wall_id):.0f} mm (a cylinder unit needs about 600 mm more)")
    add("LPG cylinder", _desc(cyl) if cyl else "None", reasons.get("lpg"),
        "customer" if prefs.include_cylinder is not None else designer, det, rules("cylinder unit dropped"), need=gas_need,
        decision=_item(catalog.get(proposal.cylinder_aid)) if proposal.cylinder_aid and not cyl and catalog.get(proposal.cylinder_aid) else None)

    # -- tall units ---------------------------------------------------------------------
    talls = roles.get("tall") or []
    tower_need = {"none": "No built-in appliances", "microwave": "Needs a microwave", "oven": "Needs an oven",
                  "oven_microwave": "Needs an oven and a microwave"}.get(prefs.built_in)
    det = []
    for t in talls:
        wall = W[t.wall_id]
        run = sorted((q for q in layout.placements if q.wall_id == t.wall_id and q.tier in ("base", "tall")), key=lambda q: q.start_mm)
        at_end = bool(run) and (t is run[0] or t is run[-1])
        in_window = any(o.start_mm < t.end_mm and t.start_mm < o.end_mm for o in wall.windows)
        hob_gap = ""
        if hob is not None and hob.wall_id == t.wall_id:
            hob_gap = f"; {max(t.start_mm - hob.end_mm, hob.start_mm - t.end_mm):.0f} mm from the hob"
        det.append(f"{_desc(t)}: {what(t.item)}")
        det.append(f"On {wall.name} at {t.start_mm:.0f}-{t.end_mm:.0f} mm - "
                   f"{'end of the run' if at_end else 'inside the run'}, "
                   f"{'IN FRONT OF A WINDOW' if in_window else 'clear of windows'}{hob_gap}")
    if menu.get("tall"):
        det.append("Tall units on offer: " + ", ".join(_name(it.category) for _, it in menu["tall"]))
    asked = [catalog.get(a) for _, a in proposal.tall_units]
    crockery_need = {"yes": "Wants a crockery unit", "no": "No crockery unit wanted"}.get(prefs.crockery_unit)
    tower_need = "; ".join(n for n in (tower_need, crockery_need) if n) or None
    add("Tall units", ", ".join(f"{_desc(p)} on {wname(p.wall_id)}" for p in talls) or "None",
        reasons.get("tall"), "customer" if prefs.built_in != "auto" or prefs.crockery_unit != "auto" else designer,
        det, rules("tall unit"), need=tower_need,
        decision=", ".join(_item(i) for i in asked if i) if len([i for i in asked if i]) != len(talls) else None)

    # -- water purifier -----------------------------------------------------------------
    pur = (roles.get("purifier") or [None])[0]
    pur_need = {"yes": "Needs a water purifier", "no": "No water purifier wanted"}.get(prefs.water_purifier)
    det = []
    if pur:
        det.append("Unit: " + what(pur.item))
        det.append(f"On {wname(pur.wall_id)} at {pur.start_mm:.0f}-{pur.end_mm:.0f} mm, wall-cabinet height")
        if sink is not None:
            det.append(f"{math.dist(world(sink), world(pur, 0.165)) * 1000:.0f} mm from the sink centre (water and power nearby)")
    if menu.get("purifier"):
        det.append("Purifier sizes on offer: " + ", ".join(str(it.width) for _, it in menu["purifier"]))
    add("Water purifier", f"{_desc(pur)} on {wname(pur.wall_id)}" if pur else "None", reasons.get("purifier"),
        "customer" if prefs.water_purifier != "auto" else designer, det, rules("water purifier"), need=pur_need,
        decision=_item(catalog.get(proposal.purifier_aid)) if proposal.purifier_aid and not pur and catalog.get(proposal.purifier_aid) else None)

    # -- glass display ------------------------------------------------------------------
    disp = (roles.get("display") or [None])[0]
    disp_need = {"yes": "Wants a glass display", "no": "No glass display wanted"}.get(prefs.display_unit)
    det = []
    if disp:
        det.append("Unit: " + what(disp.item))
        det.append(f"On {wname(disp.wall_id)} at {disp.start_mm:.0f}-{disp.end_mm:.0f} mm, at the end of a wall-cabinet run")
    add("Glass display", f"{_desc(disp)} on {wname(disp.wall_id)}" if disp else "None", reasons.get("display"),
        "customer" if prefs.display_unit != "auto" else designer, det, rules("display unit"), need=disp_need,
        decision=_item(catalog.get(proposal.display_aid)) if proposal.display_aid and not disp and catalog.get(proposal.display_aid) else None)

    # -- base cabinet style ---------------------------------------------------------------
    store, how = base_storage_need(prefs)
    style = store if how else (proposal.base_storage or "standard")
    style_name = {"drawers": "Drawer units", "mixed": "Drawer + shutter units", "shutters": "Shutter units"}.get(style, "Standard units")
    base_store = [p for p in layout.placements if p.tier == "base" and p.role == "storage"]
    n_drawer = sum(1 for p in base_store if "drawer" in (p.item.cabinet_combination or "").lower())
    if how:
        why = {"drawers": "More drawers requested: storage runs use drawer units.",
               "mixed": "Drawer + shutter mix requested.", "shutters": "Shutter cabinets requested."}[store]
    else:
        why = reasons.get("storage") or "Standard storage."
    mix = Counter(f"{_name(p.item.category)} - {p.item.cabinet_combination}" for p in base_store)
    det = [f"{n} x {k}" for k, n in mix.most_common()]
    det.append("Options: drawers (pans and vessels stay reachable), mixed (cutlery drawer on top, shelf below), shutters (lowest cost)")
    add("Base cabinets", f"{style_name}: {n_drawer} of {len(base_store)} storage cabinets have drawers", why,
        "customer" if how else designer, det,
        need={"form": f"Asked for {store}", "brief": "The brief asks for drawers"}.get(how))

    # -- feature units: where each requested unit ended up ----------------------------------
    det, missing = [], []
    built_by_aid: dict[str, list] = {}
    for p in base_store:
        built_by_aid.setdefault(p.item.aid, []).append(p)
    seen: set = set()
    for wid, aids in (proposal.base_prefs or {}).items():
        for aid in aids:
            it = catalog.get(aid)
            if it is None:
                continue
            spots = [p for p in built_by_aid.get(aid, []) if id(p) not in seen]
            if spots:
                p = spots[0]
                seen.add(id(p))
                moved = "" if p.wall_id == wid else f" (moved from {wname(wid)}: no room there)"
                det.append(f"{_item(it)}: asked for {wname(wid)} -> built on {wname(p.wall_id)} at "
                           f"{p.start_mm:.0f}-{p.end_mm:.0f} mm{moved} - {note_for(it)}")
            else:
                missing.append(_item(it))
                det.append(f"{_item(it)}: asked for {wname(wid)} -> did not fit anywhere")
    add("Feature units", f"{len(det) - len(missing)} of {len(det)} requested units built" if det else "None requested",
        reasons.get("features"), designer, det, rules("requested feature unit"))

    # -- wall cabinets ----------------------------------------------------------------------
    det = []
    for wid in layout.cabinet_walls:
        ws = [p for p in layout.placements if p.wall_id == wid and p.tier == "wall" and p.role == "wall"]
        asked = [catalog.get(a) for a in (proposal.wall_prefs or {}).get(wid, [])]
        asked_txt = ", ".join(str(i.width) for i in asked if i) or "no preference"
        clear = []
        clear += [f"window {o.start_mm:.0f}-{o.end_mm:.0f}" for o in W[wid].windows]
        if layout.chimney and layout.chimney["wall_id"] == wid:
            clear.append("chimney zone")
        det.append(f"{wname(wid)}: {len(ws)} cabinets ({', '.join(str(p.item.width) for p in sorted(ws, key=lambda p: p.start_mm)) or '-'} mm); "
                   f"sizes asked {asked_txt}" + (f"; kept clear: {', '.join(clear)}" if clear else ""))
    add("Wall cabinets", f"{sum(1 for p in layout.placements if p.tier == 'wall' and p.role == 'wall')} wall cabinets",
        reasons.get("wall_cabinets") or "Wall cabinets above every base run, kept clear of windows and the chimney.",
        designer, det)

    # -- wall cabinet doors (hinged / lift-up) -------------------------------------------------
    wall_units = [p for p in layout.placements if p.tier == "wall" and p.role == "wall"]
    n_lift = sum(1 for p in wall_units if "lift up" in p.item.cat)
    door_style = prefs.wall_style if prefs.wall_style != "auto" else (proposal.wall_style or "shutters")
    det = []
    if door_style == "lift_up":
        built = f"Lift-up doors on {n_lift} of {len(wall_units)} wall cabinets"
        if n_lift < len(wall_units):
            det.append("Lift-up units exist only in some widths at this height; the other widths keep hinged doors")
    else:
        built = "Hinged shutter doors"
    det.append("Options: hinged shutters (standard, lowest cost) or lift-up / bi-fold doors (no door at head height)")
    add("Wall cabinet doors", built,
        reasons.get("wall_style") or ("Customer's choice." if prefs.wall_style != "auto" else "Standard hinged doors."),
        "customer" if prefs.wall_style != "auto" else designer, det,
        need={"shutters": "Wants hinged shutters", "lift_up": "Wants lift-up doors"}.get(prefs.wall_style))

    # -- open shelf unit ------------------------------------------------------------------------
    op = (roles.get("open") or [None])[0]
    det = []
    if op:
        det.append("Unit: " + what(op.item))
        det.append(f"On {wname(op.wall_id)} at {op.start_mm:.0f}-{op.end_mm:.0f} mm, at the end of a wall-cabinet run, "
                   "away from the hob")
    add("Open shelf unit", f"{_desc(op)} on {wname(op.wall_id)}" if op else "None", reasons.get("open"),
        "customer" if prefs.open_unit != "auto" else designer, det, rules("open shelf unit"),
        need={"yes": "Wants an open shelf unit", "no": "No open shelves wanted"}.get(prefs.open_unit),
        decision=_item(catalog.get(proposal.open_unit_aid)) if proposal.open_unit_aid and not op and catalog.get(proposal.open_unit_aid) else None)

    # -- lofts and cabinet heights ------------------------------------------------------------------
    h = layout.heights or {}
    lofts = [p for p in layout.placements if p.tier == "loft"]
    wanted = prefs.include_loft if prefs.include_loft is not None else bool(proposal.lofts)
    loft_top = round(config.LOFT_TOP_M * 1000)
    det = [f"Ceiling (lowest wall) {h.get('ceiling_mm', '?')} mm; a loft needs {loft_top} mm"]
    if lofts:
        det.append(f"{len(lofts)} loft cabinets on {', '.join(wname(w) for w in layout.loft_walls)}, "
                   f"{lofts[0].item.height} mm high above the wall cabinets")
    add("Lofts", f"{len(lofts)} loft cabinets" if lofts else "None",
        reasons.get("lofts") or ("Customer's choice." if prefs.include_loft is not None else ""),
        "customer" if prefs.include_loft is not None else designer, det,
        [h["why"]] if wanted and not lofts and h.get("why") else [],
        need={True: "Wants lofts", False: "No lofts wanted"}.get(prefs.include_loft),
        decision="Lofts" if wanted and not lofts else None)
    if h:
        add("Cabinet heights", f"Wall cabinets {h['wall_mm']} mm, tall units {h['tall_mm']} mm", h.get("why"), "rules",
            [f"Wall cabinets hang from 1440 mm up to {h.get('top_mm')} mm; tall units stand on the 100 mm plinth and "
             "end at the same line",
             "Height pairs in the catalogue: 600 / 1940, 700 / 2040, 900 / 2240 mm (wall / tall)"])

    # -- corners (always the rules) -------------------------------------------------------------
    det = []
    for cp in layout.corner_plans:
        tag = f"{wname(cp.primary)} / {wname(cp.secondary)}"
        units = [(lvl, u) for lvl, u in (("base", cp.base_unit), ("wall", cp.wall_unit), ("loft", cp.loft_unit)) if u is not None]
        if units:
            det.append(f"{tag}: " + "; ".join(f"{lvl} - {_desc(u)} on {wname(u.wall_id)}" for lvl, u in units)
                       + (f"; open tiers: {', '.join(sorted(cp.open_tiers))}" if cp.open_tiers else ""))
        else:
            det.append(f"{tag}: kept open (a door or window leaves no room for a corner cabinet)")
    if layout.corner_plans:
        add("Corners", f"{len(layout.corner_plans)} corner(s)", "Corners take only corner cabinets, facing the room; "
            "a corner stays open when a door or window leaves no room for one.", "rules", det)
    return out
