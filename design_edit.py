"""Design chat edits: "change this cabinet to a pantry", "move the hob to Wall 1", "no gas cylinder".

1. The chat model turns the customer's words into a small, schema-locked change (only real cabinet codes and
   wall names - the same codes the AI designer uses).
2. The change is applied to the design's proposal and the normal pipeline runs again (rule book -> layout
   solver -> 3D scene -> report). No AI design call: this takes about a second.
3. If the rule book changes the customer's choice back (R11, module widths R1-R21 ...), the customer is told
   which rule, what the rules would build instead, and what overriding it means - then chooses:
   follow the rules, override, or cancel. Things that physically do not fit are reported (cannot be overridden).
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field, create_model

log = logging.getLogger("design_edit")

CHANGE_WORDS = re.compile(
    r"\b(change|replace|swap|switch|move|shift|remove|delete|drop|add|put|use|make|want|need|instead|"
    r"bigger|smaller|wider|narrower|larger|get rid|don'?t want|no more|convert|turn)\b", re.I)
QUESTION_START = re.compile(r"^\s*(why|what|how|explain|does|is|are|which|where|when|who|can you explain|tell me)\b", re.I)

PENDING: dict[str, dict] = {}  # change id -> the computed variants, waiting for the customer's choice
KEEP_PENDING = 30
KEEP_HISTORY = 3  # earlier versions kept per design for "Undo"


def looks_like_change(text: str) -> bool:
    return bool(CHANGE_WORDS.search(text or "")) and not QUESTION_START.match(text or "")


# ---------------------------------------------------------------------------
# The design's working context (floor plan, catalogue, menu of codes)
# ---------------------------------------------------------------------------

def _context(job: dict) -> dict:
    from ai_designer import KitchenPreferences, build_menu, build_shortlist_v2, wall_labels
    from catalog_loader import load_filtered_catalog
    from scene_parser import parse_scene, strip_items

    doc = strip_items(job["result"])  # every generated node is an item: this is the empty floor plan again
    geometry = parse_scene(doc)
    prefs = KitchenPreferences(**job["preferences"])
    catalog = load_filtered_catalog(prefs.door_style, prefs.handle_type)
    menu = build_menu(build_shortlist_v2(catalog))
    labels = wall_labels(geometry)  # "Wall 1" -> wall id
    return {"doc": doc, "geometry": geometry, "prefs": prefs, "catalog": catalog, "menu": menu, "labels": labels,
            "names": {wid: label for label, wid in labels.items()},
            "code_to_aid": {c: it.aid for opts in menu.values() for c, it in opts},
            "aid_to_code": {it.aid: c for opts in menu.values() for c, it in opts}}


def proposal_from_dict(d: dict):
    from rule_engine import Proposal

    return Proposal(
        layout_type=d.get("layout_type", "auto"), cabinet_walls=list(d.get("cabinet_walls") or []),
        sink_wall_id=(d.get("sink") or {}).get("wall_id"), sink_aid=(d.get("sink") or {}).get("aid"),
        sink_center_mm=(d.get("sink") or {}).get("center_mm"),
        hob_wall_id=(d.get("hob") or {}).get("wall_id"), hob_aid=(d.get("hob") or {}).get("aid"),
        hob_center_mm=(d.get("hob") or {}).get("center_mm"),
        spice_aid=d.get("spice_aid"), cylinder_aid=d.get("cylinder_aid"),
        tall_units=[(t["wall_id"], t["aid"]) for t in d.get("tall_units") or []],
        base_prefs={k: list(v) for k, v in (d.get("base_prefs") or {}).items()},
        wall_prefs={k: list(v) for k, v in (d.get("wall_prefs") or {}).items()},
        purifier_aid=d.get("purifier_aid"), purifier_required=bool(d.get("purifier_required")),
        display_aid=d.get("display_aid"), open_unit_aid=d.get("open_unit_aid"),
        base_storage=d.get("base_storage"), lofts=d.get("lofts"), wall_style=d.get("wall_style"),
        finishes=dict(d.get("finishes") or {}), reasons=dict(d.get("reasons") or {}),
        notes=d.get("notes", ""), source=d.get("source", "fallback"),
    )


def _lit(values):
    from typing import Literal

    return Literal[tuple(values)]  # type: ignore[valid-type]


def _code(ctx, aid) -> str:
    if not aid:
        return "NONE"
    if aid in ctx["aid_to_code"]:
        return ctx["aid_to_code"][aid]
    it = ctx["catalog"].get(aid)
    return f"{it.category} {it.width} mm" if it else aid


def current_design(ctx: dict, p) -> str:
    n = ctx["names"]
    w = lambda wid: n.get(wid, wid)  # noqa: E731
    lines = [
        f"cabinet_walls: {', '.join(w(x) for x in p.cabinet_walls)}",
        f"sink: {_code(ctx, p.sink_aid)} on {w(p.sink_wall_id)}",
        f"hob: {_code(ctx, p.hob_aid)} on {w(p.hob_wall_id)} at {round(p.hob_center_mm or 0)} mm",
        f"spice: {_code(ctx, p.spice_aid)}", f"lpg_cylinder: {_code(ctx, p.cylinder_aid)}",
        "tall_units: " + (", ".join(f"{_code(ctx, a)} on {w(x)}" for x, a in p.tall_units) or "none"),
        f"water_purifier: {_code(ctx, p.purifier_aid)}", f"display_unit: {_code(ctx, p.display_aid)}",
        f"open_unit: {_code(ctx, p.open_unit_aid)}", f"base_storage: {p.base_storage}",
        f"wall_style: {p.wall_style}", f"lofts: {'yes' if p.lofts else 'no'}",
        "base_feature_units: " + ("; ".join(f"{w(x)}: {', '.join(_code(ctx, a) for a in v)}"
                                            for x, v in p.base_prefs.items()) or "none"),
    ]
    return "\n".join(lines)


def _menu_text(ctx: dict) -> str:
    from ai_designer import _menu_lines

    return "\n".join(_menu_lines(ctx["menu"], knowledge=False))


ITEMS = {  # item kind -> menu group of its codes
    "sink": "sink", "hob": "hob", "spice": "spice", "lpg_cylinder": "cylinder", "tall_unit": "tall",
    "water_purifier": "purifier", "display_unit": "display", "open_unit": "open", "feature_unit": "base",
    "wall_cabinet": "wall", "base_storage": None, "wall_style": None, "lofts": None, "cabinet_wall": None,
}
VALUES = ["drawers", "mixed", "shutters", "lift_up", "yes", "no"]

EDIT_SYSTEM = """You turn a customer's request into ONE edit of their MagickHome kitchen design.
action: replace (swap a unit for another code), add, remove, or move (to another wall).
item: which part of the kitchen. new_code: the code from the CABINET LIST for the new unit (replace / add), or \
drawers / mixed / shutters (base_storage), shutters / lift_up (wall_style), yes / no (lofts); null for remove / move.
wall: the wall for move / add, else null. "this" means the item the customer clicked. Reply with JSON only."""


def edit_model(ctx: dict) -> type[BaseModel]:
    codes = [c for opts in ctx["menu"].values() for c, _ in opts]
    return create_model(
        "DesignEdit",
        summary=(str, Field(max_length=160, description="the edit in a few plain words")),
        action=(_lit(["replace", "add", "remove", "move"]), ...),
        item=(_lit(list(ITEMS)), ...),
        new_code=(Optional[_lit(codes + VALUES)], None),
        wall=(Optional[_lit(list(ctx["labels"]))], None),
    )


def understand(ctx: dict, p, request: str, focus: Optional[dict]) -> dict:
    """The customer's words -> one edit -> a change dict for apply_change."""
    import chat

    Model = edit_model(ctx)
    user = (f"WALLS: {', '.join(ctx['labels'])}\n\n{_menu_text(ctx)}\n\nCURRENT DESIGN:\n{current_design(ctx, p)}\n\n")
    focus_aid = None
    if focus:
        bits = [focus.get("title"), focus.get("sub")] + list(focus.get("lines") or [])[:6]
        focus_aid = next((a for a in list(ctx["aid_to_code"]) + [x for _, x in p.tall_units]
                          if any(a in str(b) for b in bits)), None)
        code = ctx["aid_to_code"].get(focus_aid) if focus_aid else None
        user += ("THE ITEM THE CUSTOMER CLICKED (\"this\"): " + " | ".join(str(b) for b in bits if b)
                 + (f" (code {code})" if code else "") + "\n\n")
    user += f"CUSTOMER REQUEST: {request}"
    msgs = [{"role": "system", "content": EDIT_SYSTEM}, {"role": "user", "content": user}]
    model = chat._pick_model()
    if not model:
        raise RuntimeError("no AI model is running to understand the change")
    op = Model.model_validate(chat.ask_json(model, msgs, Model.model_json_schema())).model_dump()
    # small models often say "Remove ..." in the summary but pick replace / add with no new code
    removing = re.search(r"\b(remove|delete|drop|get rid|don'?t want|do not want|no more|without)\b|^\s*no\b",
                         f"{request} {op.get('summary') or ''}", re.I)
    if op["action"] in ("replace", "add") and not op.get("new_code") and removing:
        op["action"] = "remove"
    return {"model": model, "op": op, "change": op_to_change(ctx, p, op, focus_aid, request)}


def _group_of(ctx) -> dict:
    return {c: g for g, opts in ctx["menu"].items() for c, _ in opts}


def _fit_code(ctx, item: str, code: Optional[str], request: str) -> Optional[str]:
    """A code of the right group for ``item``: the model's, else the same width in that group, else a width
    named in the request ("600 mm"), else the group's first option."""
    group = ITEMS.get(item)
    opts = ctx["menu"].get(group, []) if group else []
    if not opts:
        return code
    if code in [c for c, _ in opts]:
        return code
    width = None
    other = next((it for g in ctx["menu"].values() for c, it in g if c == code), None)
    if other is not None:
        width = other.width
    m = re.search(r"(\d{3,4})\s*mm", request)
    if m:
        width = int(m.group(1))
    same = [c for c, it in opts if width and it.width == width]
    return (same or [opts[0][0]])[0]


def op_to_change(ctx: dict, p, op: dict, focus_aid: Optional[str], request: str) -> dict:
    """One edit (action, item, new code, wall) -> the change fields apply_change understands."""
    a, item, wall = op["action"], op["item"], op.get("wall")
    names = ctx["names"]
    code = _fit_code(ctx, item, op.get("new_code"), request) if a in ("replace", "add") and ITEMS.get(item) else op.get("new_code")
    aid_code = lambda x: ctx["aid_to_code"].get(x)  # noqa: E731
    ch: dict = {"summary": op.get("summary")}
    single = {"sink": "sink", "hob": "hob", "spice": "spice", "lpg_cylinder": "lpg_cylinder",
              "water_purifier": "water_purifier", "display_unit": "display_unit", "open_unit": "open_unit"}
    if item in single:
        if a == "remove":
            ch[single[item]] = "NONE"
        elif a == "move" and item in ("sink", "hob"):
            ch[f"{item}_wall"] = wall
        elif code:
            ch[single[item]] = code
            if wall and item in ("sink", "hob"):
                ch[f"{item}_wall"] = wall
    elif item == "tall_unit":
        talls = [{"wall": names.get(w, w), "unit": aid_code(x) or x} for w, x in p.tall_units]
        target = next((i for i, (_, x) in enumerate(p.tall_units) if x == focus_aid), 0 if talls else None)
        if a == "add" and code:
            talls.append({"wall": wall or names.get(p.hob_wall_id), "unit": code})
        elif a == "remove" and target is not None:
            talls.pop(target)
        elif a == "replace" and code:
            if target is None:
                talls.append({"wall": wall or names.get(p.hob_wall_id), "unit": code})
            else:
                talls[target]["unit"] = code
        elif a == "move" and target is not None and wall:
            talls[target]["wall"] = wall
        ch["tall_units"] = [t for t in talls if t["unit"] in ctx["code_to_aid"]][:2]
    elif item in ("feature_unit", "wall_cabinet"):
        attr = "base_prefs" if item == "feature_unit" else "wall_prefs"
        runs = {names.get(w, w): [aid_code(x) for x in v if aid_code(x)] for w, v in getattr(p, attr).items()}
        old = aid_code(focus_aid) if focus_aid else None
        if a == "add" and code:
            runs.setdefault(wall or next(iter(runs), names.get(p.sink_wall_id)), []).append(code)
        elif a == "remove":
            for v in runs.values():
                for c in ([old] if old else list(v)):
                    if c in v:
                        v.remove(c)
        elif a == "replace" and code:
            done = False
            for v in runs.values():
                if old in v:
                    v[v.index(old)] = code
                    done = True
            if not done:
                runs.setdefault(wall or next(iter(runs), names.get(p.sink_wall_id)), []).append(code)
        key = "base_feature_units" if item == "feature_unit" else "wall_cabinet_sizes"
        ch[key] = [{"wall": w, "units": v} for w, v in runs.items()]
    elif item in ("base_storage", "wall_style", "lofts"):
        ok = {"base_storage": ("drawers", "mixed", "shutters"), "wall_style": ("shutters", "lift_up"), "lofts": ("yes", "no")}[item]
        val = op.get("new_code") if op.get("new_code") in ok else ("no" if item == "lofts" and a == "remove" else
                                                                    "yes" if item == "lofts" and a == "add" else None)
        if val:
            ch[item] = val
    elif item == "cabinet_wall" and wall:
        walls = [names.get(w, w) for w in p.cabinet_walls]
        if a == "add" and wall not in walls:
            walls.append(wall)
        elif a == "remove" and wall in walls and len(walls) > 1:
            walls.remove(wall)
        ch["cabinet_walls"] = walls
    return ch


# ---------------------------------------------------------------------------
# Apply the change and re-run the pipeline
# ---------------------------------------------------------------------------

def apply_change(ctx: dict, p0, change: dict):
    """Returns (proposal, prefs, requested, what) - only fields that really differ from the current design."""
    from ai_designer import KitchenPreferences

    p = copy.deepcopy(p0)
    pref = ctx["prefs"].model_dump()
    L, aid = ctx["labels"], ctx["code_to_aid"]
    get = lambda c: None if c in (None, "NONE") else aid.get(c)  # noqa: E731
    requested, what = {}, []
    items = {c: it for opts in ctx["menu"].values() for c, it in opts}

    def desc(code):  # "SINK-800" -> "SINK-800 (Base Sink Unit, 800 mm)"
        it = items.get(code)
        return f"{code} ({it.cabinet_combination or it.category}, {it.width} mm)" if it else str(code)

    def set_unit(attr, code, label, pref_key=None):
        if code is None:
            return
        new = get(code)
        if new == getattr(p, attr):
            return
        setattr(p, attr, new)
        requested[attr] = new
        what.append(f"{label} → {desc(code)}" if new else f"{label} removed")
        if pref_key:
            pref[pref_key] = "yes" if new else "no"

    set_unit("sink_aid", change.get("sink"), "sink")
    set_unit("hob_aid", change.get("hob"), "hob")
    set_unit("spice_aid", change.get("spice"), "spice pull-out")
    set_unit("purifier_aid", change.get("water_purifier"), "water purifier", "water_purifier")
    set_unit("display_aid", change.get("display_unit"), "glass display unit", "display_unit")
    set_unit("open_unit_aid", change.get("open_unit"), "open shelf unit", "open_unit")
    if change.get("lpg_cylinder") is not None and get(change["lpg_cylinder"]) != p.cylinder_aid:
        set_unit("cylinder_aid", change["lpg_cylinder"], "LPG cylinder unit")
        if p.cylinder_aid:  # the customer wants the cylinder: LPG
            pref.update(fuel="lpg", include_cylinder=True)
    for key, attr, pref_key in (("sink_wall", "sink_wall_id", "secondary_wall"), ("hob_wall", "hob_wall_id", "primary_wall")):
        if change.get(key) and L.get(change[key]) and L[change[key]] != getattr(p, attr):
            setattr(p, attr, L[change[key]])
            requested[attr] = L[change[key]]
            pref[pref_key] = L[change[key]]  # R24 / R25: the customer's own wall choice
            what.append(f"{key.replace('_', ' ')} → {change[key]}")
            if L[change[key]] not in p.cabinet_walls:
                p.cabinet_walls.append(L[change[key]])
    if change.get("hob_center_mm") is not None and abs((p.hob_center_mm or 0) - change["hob_center_mm"]) > 1:
        p.hob_center_mm = float(change["hob_center_mm"])
        requested["hob_center_mm"] = p.hob_center_mm
        what.append(f"hob centre: {change['hob_center_mm']} mm")
    if change.get("tall_units") is not None:
        talls = [(L[t["wall"]], aid[t["unit"]]) for t in change["tall_units"] if t["wall"] in L and t["unit"] in aid]
        if talls != p.tall_units:
            p.tall_units = talls
            requested["tall_units"] = talls
            pref["include_tall"] = bool(talls)
            what.append("tall units → " + (", ".join(f"{desc(t['unit'])} on {t['wall']}" for t in change["tall_units"]) or "none"))
    for key, pkey, val in (("base_storage", "base_storage", None), ("wall_style", "wall_style", None)):
        v = change.get(key)
        if v and v != getattr(p, key):
            setattr(p, key, v)
            pref[pkey] = v
            requested[key] = v
            what.append(f"{key.replace('_', ' ')} → {v.replace('_', '-')}")
    if change.get("lofts") and (change["lofts"] == "yes") != bool(p.lofts):
        p.lofts = change["lofts"] == "yes"
        pref["include_loft"] = p.lofts
        requested["lofts"] = p.lofts
        what.append(f"lofts → {change['lofts']}")
    if change.get("cabinet_walls"):
        walls = list(dict.fromkeys(L[w] for w in change["cabinet_walls"] if w in L))
        if walls and set(walls) != set(p.cabinet_walls):
            p.cabinet_walls = walls
            p.layout_type = {1: "single", 2: "L"}.get(len(walls), "U")
            pref["cabinet_walls"] = walls
            requested["cabinet_walls"] = walls
            what.append("cabinet walls: " + ", ".join(change["cabinet_walls"]))
    for key, attr in (("base_feature_units", "base_prefs"), ("wall_cabinet_sizes", "wall_prefs")):
        if change.get(key) is not None:
            new = {}
            for run in change[key]:
                if run["wall"] in L:
                    new.setdefault(L[run["wall"]], []).extend(aid[u] for u in run["units"] if u in aid)
            if new != getattr(p, attr):
                setattr(p, attr, new)
                requested[attr] = new
                what.append(f"{key.replace('_', ' ')}: " + "; ".join(
                    f"{r['wall']}: {', '.join(r['units']) or 'none'}" for r in change[key]))
    return p, KitchenPreferences(**pref), requested, what


_RULE_ROLE = {"sink_aid": ("sink",), "hob_aid": ("hob",), "spice_aid": ("spice",),
              "cylinder_aid": ("cylinder", "cylinder_included")}


def _build(ctx, proposal, prefs, requested, override: bool, job: dict):
    """Run the pipeline. Returns (scene, report, conflicts)."""
    import main

    conflicts = []

    def after_rules(prop, facts):
        for attr, roles in _RULE_ROLE.items():
            if attr not in requested or getattr(prop, attr) == requested[attr]:
                continue
            rule = next((a for a in facts.applied if a.get("role") in roles), {})
            got = ctx["catalog"].get(getattr(prop, attr)) if getattr(prop, attr) else None
            want = ctx["catalog"].get(requested[attr]) if requested[attr] else None
            conflicts.append({
                "field": attr, "rule": rule.get("rule"), "text": rule.get("text", ""),
                "you_asked": f"{want.category} {want.width} mm" if want else "no unit",
                "rules_build": f"{got.category} {got.width} mm" if got else "no unit",
            })
            if override:
                setattr(prop, attr, requested[attr])

    ai_info = dict((job.get("report") or {}).get("ai") or {})
    out, report = main.build_kitchen(ctx["doc"], ctx["geometry"], ctx["catalog"], prefs, proposal, ai_info,
                                     lambda _m: None, time.time(), after_rules=after_rules)
    return out, report, conflicts


def _placed(report: dict) -> set[tuple]:
    """(category, width) of every placed unit: the solver may use another height variant of the same unit."""
    return {(i["category"], i["width"]) for items in (report.get("layout") or {}).get("placements", {}).values()
            for i in items}


def _not_built(ctx, requested: dict, report: dict, skip=()) -> list[str]:
    """Units the customer asked for that the layout could not fit (physical limits - cannot be overridden).
    ``skip``: fields the rule book changed (reported as rule conflicts instead)."""
    placed, out = _placed(report), []
    wanted = [requested.get(a) for a in ("sink_aid", "hob_aid", "spice_aid", "cylinder_aid", "purifier_aid",
                                         "display_aid", "open_unit_aid") if requested.get(a) and a not in skip]
    wanted += [a for _, a in requested.get("tall_units", [])]
    for v in (requested.get("base_prefs") or {}).values():
        wanted += v
    for a in dict.fromkeys(wanted):
        it = ctx["catalog"].get(a)
        if it is not None and (it.category, it.width) not in placed:
            out.append(f"{it.category} {it.width} mm")
    if requested.get("lofts") and not (report.get("heights") or {}).get("lofts"):
        out.append("lofts - " + ((report.get("heights") or {}).get("why") or "the ceiling is too low"))
    return out


def guideline_issues(ctx: dict, report: dict) -> list[str]:
    """MagickHome designer guidelines (not numbered rules) the layout breaks: hob or tall unit at a window,
    sink and hob too close on one wall."""
    out, walls = [], ctx["geometry"].walls
    names = ctx["names"]
    for wid, items in ((report.get("layout") or {}).get("placements") or {}).items():
        wins = [(o.start_mm, o.start_mm + o.width_mm) for o in walls[wid].openings if o.kind == "window"] if wid in walls else []
        hob = next((i for i in items if i.get("role") == "hob" or str(i.get("category", "")).startswith("Base Hob")), None)
        sink = next((i for i in items if i.get("role") == "sink" or str(i.get("category", "")).startswith("Base Sink")), None)
        for a, b in wins:
            if hob and hob["start_mm"] < b and hob["end_mm"] > a:
                out.append(f"The hob is under the window on {names.get(wid, wid)}: no chimney can go there, and wind "
                           "can blow out the flame or blow the curtains towards it.")
            for t in (i for i in items if i.get("tier") == "tall"):
                if t["start_mm"] < b and t["end_mm"] > a:
                    out.append(f"A tall unit stands in front of the window on {names.get(wid, wid)} and blocks light.")
        if hob and sink:
            gap = max(sink["start_mm"] - hob["end_mm"], hob["start_mm"] - sink["end_mm"])
            if gap < 600:
                out.append(f"Only {max(0, round(gap))} mm of counter between the sink and the hob on "
                           f"{names.get(wid, wid)} (designers keep at least 600 mm): water splashes onto the flame "
                           "and there is no space to prepare food between them.")
    return list(dict.fromkeys(out))


def _plain_warning(w: str) -> str:
    """Layout-engine warnings in customer words."""
    m = re.match(r"(\w+) unit narrowed (\d+) -> (\d+) mm to fit the room", w)
    if m:
        return f"Not enough space: the {m.group(1)} unit had to shrink from {m.group(2)} to {m.group(3)} mm."
    m = re.match(r"(\w+) landing under 450 mm \((R\d+)\): (\w+) side (\d+) mm", w)
    if m:
        return (f"Breaks rule {m.group(2)}: only {m.group(4)} mm of counter beside the {m.group(1)} "
                f"({m.group(3)} side) - at least 450 mm is needed to put pots and plates down.")
    return w


def _consequence(c: dict) -> str:
    """What overriding this rule means for the customer, in plain words."""
    f, asked, rules = c["field"], c["you_asked"], c["rules_build"]
    num = lambda t: int(re.search(r"(\d+) mm", t).group(1)) if re.search(r"(\d+) mm", t) else None  # noqa: E731
    aw, rw = num(asked), num(rules)
    if f == "cylinder_aid" and asked == "no unit":
        return ("No cabinet for an LPG cylinder: if you cook with LPG the cylinder stands in the open or in a normal "
                "cabinet with no ventilation. Fine if you have piped gas (PNG) or induction - tell me and I'll set that.")
    if f == "sink_aid" and aw and rw:
        return (f"A {aw} mm sink instead of {rw} mm: less washing space for big vessels and pressure cookers in a "
                "kitchen this size." if aw < rw else f"A {aw} mm sink takes counter and storage space from the run.")
    if f == "hob_aid" and aw and rw:
        return (f"A {aw} mm hob instead of {rw} mm: fewer burners or less room for big pots, and the chimney may be "
                "wider than the hob." if aw < rw else f"A {aw} mm hob takes counter and storage space beside it.")
    return f"You get {asked} instead of the rule-book {rules}."


_REASON_KEY = {"sink_aid": "sink", "sink_wall_id": "sink", "hob_aid": "hob", "hob_wall_id": "hob",
               "hob_center_mm": "hob", "cylinder_aid": "lpg", "tall_units": "tall", "purifier_aid": "purifier",
               "display_aid": "display", "open_unit_aid": "open", "base_storage": "storage",
               "wall_style": "wall_style", "lofts": "lofts", "cabinet_walls": "layout", "base_prefs": "features",
               "wall_prefs": "wall_cabinets"}


def _set_reasons(proposal, requested: dict, request: str, conflicts: list, override: bool):
    """The changed decisions' reasons become the customer's request (the explanation is rebuilt from them)."""
    note = f"Changed at your request: \"{request.strip()[:140]}\""
    for attr in requested:
        key = _REASON_KEY.get(attr)
        if not key:
            continue
        rule = next((c for c in conflicts if c["field"] == attr), None)
        if rule and override:
            proposal.reasons[key] = f"{note} - you chose to override rule R{rule.get('rule')}."
        elif rule:
            proposal.reasons[key] = f"{note}, but rule R{rule.get('rule')} keeps the rule-book unit: {rule.get('text', '')}"[:400]
        else:
            proposal.reasons[key] = f"{note}."


def _log_edit(old_report: dict, report: dict, what: list[str], request: str, override: bool):
    report["edits"] = list(old_report.get("edits") or []) + [
        {"t": time.time(), "request": request, "changes": what, "override": override}]


def propose(job: dict, request: str, focus: Optional[dict] = None) -> Optional[dict]:
    """Understand and evaluate a change. None = it was not a design change (answer it as a question)."""
    t0 = time.time()
    ctx = _context(job)
    p0 = proposal_from_dict((job.get("report") or {}).get("proposal") or {})
    got = understand(ctx, p0, request, focus)
    proposal, prefs, requested, what = apply_change(ctx, p0, got["change"])
    if not requested:
        return None
    old_rep = job.get("report") or {}
    before_issues = set(old_rep.get("validation_issues") or []) | set((old_rep.get("layout") or {}).get("warnings") or [])
    probe = copy.deepcopy(proposal)
    _set_reasons(probe, requested, request, [], False)
    out, rep, conflicts = _build(ctx, probe, prefs, requested, False, job)
    if conflicts:  # rebuild with reasons that say the rule kept its unit
        rules_p = copy.deepcopy(proposal)
        _set_reasons(rules_p, requested, request, conflicts, False)
        out, rep, _ = _build(ctx, rules_p, prefs, requested, False, job)
    variants = {"rules": {"scene": out, "report": rep, "prefs": prefs.model_dump()}}
    result = {
        "type": "change", "id": uuid.uuid4().hex[:10], "summary": "; ".join(what),
        "changes": what, "conflicts": conflicts,
        "not_built": _not_built(ctx, requested, rep, skip={c["field"] for c in conflicts}),
        "issues": [_plain_warning(i) for i in (rep.get("validation_issues") or []) + ((rep.get("layout") or {}).get("warnings") or [])
                   if i not in before_issues],
        "model": got["model"],
    }
    before_guides = set(guideline_issues(ctx, job.get("report") or {}))
    result["guidelines"] = [g for g in guideline_issues(ctx, rep) if g not in before_guides]
    _log_edit(old_rep, rep, what, request, False)
    if conflicts:
        over_p = copy.deepcopy(proposal)
        _set_reasons(over_p, requested, request, conflicts, True)
        o_out, o_rep, _ = _build(ctx, over_p, prefs, requested, True, job)
        _log_edit(old_rep, o_rep, what, request, True)
        variants["override"] = {"scene": o_out, "report": o_rep, "prefs": prefs.model_dump()}
        result["override_issues"] = (
            [_consequence(c) for c in conflicts]
            + [g for g in guideline_issues(ctx, o_rep) if g not in before_guides]
            + [_plain_warning(i) for i in (o_rep.get("validation_issues") or []) + ((o_rep.get("layout") or {}).get("warnings") or [])
               if i not in before_issues]
            + [f"Could not fit: {x}" for x in _not_built(ctx, requested, o_rep)]
            + ["The design report will show this as a rule you overrode."])
    result["seconds"] = round(time.time() - t0, 1)
    result["answer"] = _message(result)
    if conflicts:
        result["choices"] = [{"id": "override", "label": "Override the rule"},
                             {"id": "rules", "label": "Follow the rules"}, {"id": "cancel", "label": "Cancel"}]
    elif result["issues"] or result["guidelines"]:
        result["choices"] = [{"id": "rules", "label": "Apply anyway"}, {"id": "cancel", "label": "Cancel"}]
    else:
        result["choices"] = [{"id": "rules", "label": "Apply change"}, {"id": "cancel", "label": "Cancel"}]
    PENDING[result["id"]] = {"job": job["id"], "version": job.get("version", 0), "variants": variants,
                             "request": request, "t": time.time()}
    while len(PENDING) > KEEP_PENDING:
        PENDING.pop(next(iter(PENDING)))
    return result


def _message(r: dict) -> str:
    lines = [f"I understood: {r['summary']}."]
    if r["conflicts"]:
        for c in r["conflicts"]:
            rule = f"rule R{c['rule']}" if c.get("rule") else "the rule book"
            lines.append(f"⚠ This doesn't follow {rule}: {c['text']}".rstrip(": "))
            lines.append(f"You asked for {c['you_asked']}; following the rules I'd build {c['rules_build']}.")
        lines.append("If you override, you get what you asked for, with these issues:")
        lines += [f"• {i}" for i in r.get("override_issues") or []]
        lines.append("Do you want to override the rule, follow the rules, or cancel?")
    elif r.get("guidelines") or r["issues"]:
        problems = r.get("guidelines", []) + r["issues"]
        lines.append("⚠ This breaks MagickHome rules - if you apply it, you get these issues:"
                     if any(x.startswith("Breaks rule") for x in problems)
                     else "⚠ The rules allow this, but it goes against MagickHome design guidance - you'd get:")
        lines += [f"• {g}" for g in r.get("guidelines", []) + r["issues"]]
        lines.append("Do you still want to apply it?")
    else:
        lines.append("✓ This follows the MagickHome rules.")
        lines.append("Apply this change?")
    if r["not_built"]:
        lines.insert(1, "Note: these don't fit in this room and would be left out: " + "; ".join(r["not_built"]) + ".")
    return "\n".join(lines)


def apply(job: dict, change_id: str, override: bool) -> dict:
    pend = PENDING.get(change_id)
    if not pend or pend["job"] != job["id"]:
        raise KeyError("this change has expired - please ask again")
    if pend["version"] != job.get("version", 0):
        raise ValueError("the design changed since this was proposed - please ask again")
    v = pend["variants"]["override" if override and "override" in pend["variants"] else "rules"]
    hist = job.setdefault("history", [])
    hist.append({"result": job["result"], "report": job["report"], "preferences": job["preferences"]})
    del hist[:-KEEP_HISTORY]
    job.update(result=v["scene"], report=v["report"], preferences=v["prefs"], version=job.get("version", 0) + 1)
    PENDING.pop(change_id, None)
    return job


def undo(job: dict) -> dict:
    hist = job.get("history") or []
    if not hist:
        raise ValueError("nothing to undo")
    prev = hist.pop()
    job.update(result=prev["result"], report=prev["report"], preferences=prev["preferences"],
               version=job.get("version", 0) + 1)
    return job
