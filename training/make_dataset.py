"""Training dataset for a MagickHome kitchen-designer model (supervised fine-tuning, chat format).

Each example is exactly one production design request and a rule-checked answer:

* system + user = the prompt the app sends Qwen today (``ai_designer.build_messages``, prompt v2,
  ``DESIGN_RULES=rulebook``): the 81 rules of the MagickHome rule workbook, the cabinet codes, the
  room facts and the customer's needs / brief. Same text the model will see when it is used.
* assistant = the design JSON (same schema Qwen fills), written by a designer policy
  (:func:`design`) that follows the rule workbook, the playbook and common Indian kitchen practice,
  with variety (several good answers exist for one room).

Every answer is CHECKED by the real pipeline (rule book + layout engine + 3D build):

1. pass 1 - run the answer; whatever the rule book or the engine had to change (module widths by
   kitchen size, a unit that did not fit, the final sink / hob positions...) is taken INTO the
   answer, so the answer is what a rule-following designer would have said;
2. pass 2 - run the corrected answer; the example is kept only if the engine needs NO correction,
   the 3D scene verifies, and the designer checklist (designer_benchmark.CHECKS) passes;
3. the *_why reasons are written last, from the final decisions, the room facts and the rules the
   rule book actually applied (so a reason never contradicts its decision).

Rooms are synthetic rectangles (2.1-5.0 m walls, 2.4-3.2 m ceilings, 1-3 doors, 0-2 windows) built
from the production sample plan; customers are random mixes of "decide" and stated needs, picked
walls, budgets, fuels and English briefs. Offline: the catalogue and finishes come from the last
Assets / textures API copies in data/ (no API calls).

    python training/make_dataset.py --n 1000            # ~30-45 min
    python training/make_dataset.py --n 20 --out training/sample   # quick look

Output (in --out): train.jsonl, val.jsonl ({"messages": [system, user, assistant]}), meta.jsonl
(room, customer, checks per example) and stats.json.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import itertools
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
os.environ.setdefault("MAGICKHOME_DESIGN_RULES", "rulebook")

import config  # noqa: E402
import catalog_loader  # noqa: E402

# offline and repeatable: the last Assets API / textures API copies, no network calls
config.TEXTURES_API_URL = ""
catalog_loader.fetch_api_records = lambda url="": catalog_loader._read_cache()[0]

import finishes  # noqa: E402
import main  # noqa: E402
import rulebook  # noqa: E402
from ai_designer import (  # noqa: E402
    KitchenPreferences, _free_segments, _hob_range, build_menu, build_messages, build_shortlist_v2, lofts_possible,
    wall_labels,
)
from designer_benchmark import CORRECTION_PATTERNS, DOOR_ID, Room, score_design  # noqa: E402
from scene_parser import parse_scene  # noqa: E402
from test_pipeline import verify_ai_scene  # noqa: E402

logging.disable(logging.WARNING)
MIN_WALL = config.MIN_CABINET_WALL_MM

# ---------------------------------------------------------------------------
# rooms
# ---------------------------------------------------------------------------


def sample_room(rng: random.Random, i: int) -> Room:
    """A rectangle 2.1-5.0 m a side with 1-3 doors and 0-2 windows that do not overlap."""
    lo, hi = rng.choices([(50, 60), (60, 80), (81, 120), (121, 180), (181, 230)], [6, 36, 30, 23, 5])[0]
    area_m2 = rng.uniform(lo, hi) / 10.7639
    width = min(5000, max(2100, round(rng.uniform(2100, min(4800, area_m2 * 1000 / 2.1)) / 50) * 50))
    depth = min(5000, max(2100, round(area_m2 * 1e6 / width / 50) * 50))
    height = rng.choice([2.4, 2.5, 2.5, 2.6, 2.7, 2.7, 2.75, 2.8, 2.9, 3.0, 3.0, 3.2])
    length = {1: depth, 2: width, 3: depth, 4: width}  # Room.build: walls 1 / 3 = depth, 2 / 4 = width
    taken: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)

    def place(kind: str) -> Optional[tuple[int, int, int]]:
        for _ in range(40):
            wall = rng.randint(1, 4)
            w = rng.randrange(750, 1001, 50) if kind == "door" else rng.randrange(600, 1801, 100)
            if w > length[wall] - 600:
                continue
            lo, hi = w / 2 + (150 if kind == "door" else 300), length[wall] - w / 2 - (150 if kind == "door" else 300)
            if hi <= lo:
                continue
            c = round(rng.uniform(lo, hi) / 10) * 10
            a, b = c - w / 2, c + w / 2
            if any(a < y + 250 and b > x - 250 for x, y in taken[wall]):
                continue
            taken[wall].append((a, b))
            return wall, c, w
        return None

    n_doors = rng.choices([1, 2, 3], [70, 22, 8])[0]
    n_windows = rng.choices([0, 1, 2], [10, 70, 20])[0]
    doors = [d for d in (place("door") for _ in range(n_doors)) if d]
    windows = [w for w in (place("window") for _ in range(n_windows)) if w]
    title = (f"{width / 1000:.2f} x {depth / 1000:.2f} m, {height:.2f} m high, {len(doors)} door(s), "
             f"{len(windows)} window(s)")
    return Room(f"room{i:04d}", title, width, depth, windows=windows, doors=doors, height_m=height)


# ---------------------------------------------------------------------------
# customers
# ---------------------------------------------------------------------------

COLOURS = {  # brief words -> what to look for in the finish summary (category / colour / family)
    "wood": ("a warm wood look", ("wood", "oak", "walnut", "teak", "maple", "ash")),
    "white": ("white and bright", ("white", "ivory", "frost", "cream", "snow")),
    "grey": ("a grey, modern look", ("grey", "gray", "slate", "ash", "graphite")),
    "dark": ("dark, rich colours", ("black", "dark", "walnut", "charcoal", "wine", "berry")),
    "blue": ("a touch of blue", ("blue", "navy", "teal", "aqua")),
}


def connected(geo, ws: list[str]) -> bool:
    if len(ws) <= 1:
        return True
    seen, todo = {ws[0]}, [ws[0]]
    while todo:
        a = todo.pop()
        for b in ws:
            if b not in seen and geo.corner_between(a, b):
                seen.add(b)
                todo.append(b)
    return len(seen) == len(ws)


def free(geo, wid: str) -> float:
    return max((b - a for a, b in _free_segments(geo, wid)), default=0.0)


def sample_prefs(rng: random.Random, geo) -> KitchenPreferences:
    p: dict[str, Any] = {"ai_mode": "auto", "ai_provider": "qwen"}
    p["layout"] = rng.choices(["auto", "single", "L", "U"], [46, 18, 23, 13])[0]
    for f, vals, w_auto in (
        ("water_purifier", ["yes", "no"], 0.6), ("built_in", ["none", "microwave", "oven", "oven_microwave"], 0.55),
        ("display_unit", ["yes", "no"], 0.65), ("base_storage", ["drawers", "mixed", "shutters"], 0.65),
        ("wall_style", ["shutters", "lift_up"], 0.65), ("open_unit", ["yes", "no"], 0.7),
        ("crockery_unit", ["yes", "no"], 0.75), ("cooking", ["heavy", "light"], 0.5),
        ("fuel", ["lpg", "png", "induction"], 0.5), ("budget", ["essential", "premium", "luxury"], 0.45),
    ):
        p[f] = "auto" if rng.random() < w_auto else rng.choice(vals)
    if rng.random() < 0.45:
        p["household_size"] = rng.randint(1, 8)
    if rng.random() < 0.3:
        p["include_loft"] = rng.random() < 0.6
    p["include_tall"] = rng.random() < 0.92
    usable = [w for w in geo.wall_order if free(geo, w) >= MIN_WALL]
    need = {"single": 1, "L": 2, "U": 3}.get(p["layout"])
    r = rng.random()
    if need and r < 0.18:  # the customer ticked the walls for the shape
        sets = [list(s) for s in itertools.combinations(usable, need) if connected(geo, list(s))]
        if sets:
            p["cabinet_walls"] = rng.choice(sets)
    if r < 0.5 and usable:  # the customer chose the cooking wall (and maybe the sink wall)
        pool = p.get("cabinet_walls") or usable
        p["primary_wall"] = rng.choice(pool)
        if rng.random() < 0.55:
            if p["layout"] == "single" or len(p.get("cabinet_walls") or []) == 1:
                p["secondary_wall"] = p["primary_wall"]
            else:
                near = [w for w in rulebook.near_walls(geo, p["primary_wall"]) if w in pool]
                if near:
                    p["secondary_wall"] = rng.choice(near)
    if rng.random() < 0.45:
        p["style_notes"] = sample_brief(rng, p)
    return KitchenPreferences(**p)


def sample_brief(rng: random.Random, p: dict) -> str:
    """An English brief like the ones the brief reader writes (it may repeat stated needs and add
    wishes the form has no field for - colours, habits - which the designer should act on)."""
    bits = []
    n = p.get("household_size")
    if n:
        bits.append(rng.choice([f"We are a family of {n}.", f"{n} people live here.", f"Household of {n}."]))
    if p.get("cooking") == "heavy" or (p.get("cooking") == "auto" and rng.random() < 0.3):
        bits.append(rng.choice(["We cook full Indian meals twice a day.", "Heavy cooking every day, lots of vessels.",
                                "Daily cooking with pressure cookers and big kadais."]))
    elif p.get("cooking") == "light":
        bits.append(rng.choice(["We mostly make simple meals.", "Light cooking, mostly breakfast and dinner."]))
    extra = [
        ("bake", 0.15, "We bake cakes on weekends."),
        ("microwave", 0.15, "We reheat food in the microwave a lot."),
        ("drawers", 0.15, "We want lots of drawers."),
        ("vessels", 0.15, "We store big vessels and festival items."),
        ("water", 0.12, "Purified drinking water is a must for us."),
        ("crockery", 0.1, "We have good crockery we like to show."),
        ("minimal", 0.12, "Please keep it clean and minimal."),
    ]
    for _, prob, text in extra:
        if rng.random() < prob:
            bits.append(text)
    if rng.random() < 0.45:
        key = rng.choice(list(COLOURS))
        bits.append(rng.choice(["We like ", "We would love ", "Prefer "]) + COLOURS[key][0] + ".")
    if not bits:
        bits.append(rng.choice(["Keep it practical.", "A practical kitchen for daily use.", "Storage matters most."]))
    rng.shuffle(bits)
    return " ".join(bits)[:480]


# ---------------------------------------------------------------------------
# the designer policy
# ---------------------------------------------------------------------------


def schema_enums(Model) -> dict[str, list]:
    sch = Model.model_json_schema()
    defs = sch.get("$defs", {})

    def node_of(n):
        while "$ref" in n:
            n = defs[n["$ref"].split("/")[-1]]
        return n

    def en(n):
        n = node_of(n)
        if "enum" in n:
            return list(n["enum"])
        if "const" in n:
            return [n["const"]]
        if n.get("type") == "array":
            return en(n["items"])
        return None

    props = sch["properties"]
    out = {k: en(v) for k, v in props.items()}
    for key, field in (("tall_units", "unit"), ("base_feature_units", "units"), ("wall_cabinet_sizes", "units")):
        item = node_of(node_of(props[key])["items"])
        out[f"{key}.{field}"] = en(item["properties"][field])
        out[f"{key}.wall"] = en(item["properties"]["wall"])
    return out


def wpick(rng, options: list, weights: list):
    pairs = [(o, w) for o, w in zip(options, weights) if o is not None and w > 0]
    return rng.choices([o for o, _ in pairs], [w for _, w in pairs])[0] if pairs else None


def code_width(code: str) -> int:
    m = re.search(r"-(\d{3,4})", code or "")
    return int(m.group(1)) if m else 0


def brief_has(prefs, *words) -> bool:
    t = (prefs.style_notes or "").lower()
    return any(w in t for w in words)


def design(geo, prefs, Model, rng: random.Random) -> dict:
    """A designer's answer for this room and customer (before the engine check)."""
    E = schema_enums(Model)
    labels = wall_labels(geo)
    L = {wid: lab for lab, wid in labels.items()}
    usable = [w for w in geo.wall_order if free(geo, w) >= MIN_WALL]
    win_walls = sorted([w for w in usable if geo.walls[w].windows],
                       key=lambda w: -max(o.width_mm for o in geo.walls[w].windows))
    fixed_hob = labels[E["hob_wall"][0]] if len(E["hob_wall"]) == 1 else None
    fixed_sink = labels[E["sink_wall"][0]] if len(E["sink_wall"]) == 1 else None
    area = rulebook.floor_area_sqft(geo)

    # -- cabinet walls --------------------------------------------------------
    if prefs.cabinet_walls:
        walls = list(prefs.cabinet_walls)
    else:
        n = {"single": 1, "L": 2, "U": 3}.get(prefs.layout)
        if n is None:
            anchor = win_walls[0] if win_walls else (max(usable, key=lambda w: free(geo, w)) if usable else None)
            nb = [c for c in rulebook.near_walls(geo, anchor) if c in usable] if anchor else []
            lo, hi = geo.walls[anchor].usable if anchor else (0, 0)
            u_ok = len(nb) == 2 and (hi - lo) - 2 * config.CORNER_DEPTH_MM >= config.MIN_AISLE_MM and area >= 95
            w1, w2, w3 = (30, 60, 10) if area < 81 else (14, 60, 26) if area < 121 else (6, 50, 44)
            n = wpick(rng, [1, 2 if nb else None, 3 if u_ok else None], [w1 if nb else 100, w2, w3])
        must = [w for w in (fixed_hob, fixed_sink) if w]
        best, best_score = None, -1e9
        for k in (n, n - 1, n + 1, 1):
            for ws in itertools.combinations(usable, max(1, min(k, 3))):
                ws = list(ws)
                if not set(must) <= set(ws) or not connected(geo, ws):
                    continue
                s = sum(free(geo, w) for w in ws) / 1000 + (3 if win_walls and win_walls[0] in ws else 0)
                s += rng.uniform(0, 1.5)
                if s > best_score:
                    best, best_score = ws, s
            if best:
                break
        walls = best or (usable[:1] or geo.wall_order[:1])
        walls = list(dict.fromkeys(walls + [w for w in must if w not in walls]))

    # -- sink wall ------------------------------------------------------------
    if fixed_sink in walls:
        sink_wall = fixed_sink
    else:
        wins = [w for w in win_walls if w in walls and w != fixed_hob]
        cands = [w for w in walls if w != fixed_hob] or walls
        sink_wall = wins[0] if wins and rng.random() < 0.85 else max(cands, key=lambda w: free(geo, w) + rng.uniform(0, 600))
    # -- hob wall -------------------------------------------------------------
    if fixed_hob in walls:
        hob_wall = fixed_hob
    else:
        others = [w for w in walls if w != sink_wall and free(geo, w) >= 1800] or [w for w in walls if w != sink_wall]
        if others:
            hob_wall = max(others, key=lambda w: free(geo, w) - (900 if geo.walls[w].windows else 0)
                           - 500 * len(geo.walls[w].doors) + rng.uniform(0, 500))
        else:
            hob_wall = sink_wall

    # -- sink unit + position ---------------------------------------------------
    sink_codes = [c for c in E["sink"] if code_width(c)]
    seg_s = max(_free_segments(geo, sink_wall), key=lambda t: t[1] - t[0])
    f = seg_s[1] - seg_s[0]
    want = 800 if f < 2400 else 900 if f < 3000 else 1050 if f < 3600 else 1200
    sink = min(sink_codes, key=lambda c: (abs(code_width(c) - want) + rng.uniform(0, 160)))
    sw = code_width(sink)
    win = max(geo.walls[sink_wall].windows, key=lambda o: o.width_mm) if geo.walls[sink_wall].windows else None

    def corner_margin(wid, end):
        nb = geo.walls[wid].neighbor_end if end else geo.walls[wid].neighbor_start
        return 700 if nb in walls else 100

    def spot(wid, seg, width):
        lo = seg[0] + width / 2 + (corner_margin(wid, False) if seg[0] <= geo.walls[wid].usable[0] + 1 else 150)
        hi = seg[1] - width / 2 - (corner_margin(wid, True) if seg[1] >= geo.walls[wid].usable[1] - 1 else 150)
        return (lo, hi) if hi > lo else ((seg[0] + seg[1]) / 2,) * 2

    if win and seg_s[0] <= win.center_mm <= seg_s[1] and rng.random() < 0.8:
        sink_c = win.center_mm
    else:
        lo, hi = spot(sink_wall, seg_s, sw)
        sink_c = rng.uniform(lo, hi)
    sink_c = int(round(sink_c / 10) * 10)

    # -- hob unit + position ----------------------------------------------------
    hob_codes = list(E["hob"])
    seg_h = max(_free_segments(geo, hob_wall), key=lambda t: t[1] - t[0])
    hob = wpick(rng, hob_codes, [60 if "3DRAWER" in c else 22 if "2DRAWER" in c else 8 for c in hob_codes])
    if seg_h[1] - seg_h[0] < 2400:
        smaller = [c for c in hob_codes if code_width(c) <= 750]
        hob = rng.choice(smaller) if smaller and rng.random() < 0.6 else hob
    hw = code_width(hob)
    if hob_wall == sink_wall:  # keep 600+ mm of counter between sink and hob
        gap = (sw + hw) / 2 + 650
        left, right = (seg_h[0], sink_c - gap + hw / 2), (sink_c + gap - hw / 2, seg_h[1])
        side = max((left, right), key=lambda s: s[1] - s[0] + rng.uniform(0, 300))
        lo, hi = spot(hob_wall, side, hw)
    else:
        rng_h = _hob_range([seg_h])
        lo, hi = rng_h if rng_h else spot(hob_wall, seg_h, hw)
    hob_c = int(round(rng.uniform(min(lo, hi), max(lo, hi)) / 10) * 10)

    # -- small units ------------------------------------------------------------
    spice = wpick(rng, E["spice"], [75 if c == "SPICE-300" else 15 if c != "NONE" else 10 for c in E["spice"]])
    lpg = E["lpg_cylinder"][0] if len(E["lpg_cylinder"]) == 1 else next((c for c in E["lpg_cylinder"] if c != "NONE"), "NONE")

    # -- tall units -------------------------------------------------------------
    talls: list[dict] = []
    tall_codes = E.get("tall_units.unit") or []
    if prefs.include_tall and tall_codes:
        def first(prefix):
            return next((c for c in tall_codes if c.startswith(prefix)), None)
        want_units = []
        tower = {"oven": "OVEN-", "microwave": "MICROWAVE-", "oven_microwave": "OVEN-MICRO-"}.get(prefs.built_in)
        if tower:
            want_units.append(first(tower))
        elif prefs.built_in == "auto" and brief_has(prefs, "bake"):
            want_units.append(first("OVEN-"))
        elif prefs.built_in == "auto" and brief_has(prefs, "microwave"):
            want_units.append(first("MICROWAVE-"))
        if prefs.crockery_unit == "yes" or (prefs.crockery_unit == "auto" and brief_has(prefs, "crockery") and rng.random() < 0.6):
            want_units.append(first("CROCKERY-"))
        total_free = sum(free(geo, w) for w in walls)
        extra = 0 if total_free < 3600 else wpick(rng, [0, 1, 2], [25, 55, 20 if total_free >= 6000 else 0])
        while len(want_units) < min(2, len([u for u in want_units if u]) + extra):
            want_units.append(wpick(rng, [first("PANTRY-"), first("TALL-WICKER-"), first("CROCKERY-") if prefs.crockery_unit != "no" else None],
                                    [60, 20, 20]))
        want_units = [u for u in dict.fromkeys(u for u in want_units if u)][:2]
        if not want_units:  # the answer form asks for at least one tall unit
            want_units = [first("PANTRY-") or tall_codes[0]]
        tall_walls = [w for w in walls if w != hob_wall or free(geo, w) >= 3600] or walls
        tw = max(tall_walls, key=lambda w: free(geo, w) - (800 if w == sink_wall else 0) + rng.uniform(0, 700))
        talls = [{"wall": L[tw], "unit": u} for u in want_units]

    # -- wall-side units ----------------------------------------------------------
    def opt(codes, p_yes, prefer=None):
        if codes == ["NONE"]:
            return "NONE"
        real = [c for c in codes if c != "NONE"]
        if "NONE" in codes and rng.random() >= p_yes:
            return "NONE"
        if prefer:
            pref = [c for c in real if prefer in c]
            if pref and rng.random() < 0.7:
                return pref[0]
        return rng.choice(real) if real else "NONE"

    purifier = opt(E["water_purifier"], 0.85 if brief_has(prefs, "water") else 0.7, "600")
    display = opt(E["display_unit"], 0.6 if brief_has(prefs, "crockery") else (0.3 if len(walls) >= 2 else 0.12))
    open_u = opt(E["open_unit"], 0.25 if not brief_has(prefs, "minimal") else 0.1)

    # -- base cabinet style ---------------------------------------------------------
    st = E["base_storage"]
    if len(st) == 1:
        storage = st[0]
    elif prefs.budget == "essential":
        storage = "shutters"
    elif prefs.budget in ("premium", "luxury"):
        storage = wpick(rng, st, [60 if s == "drawers" else 40 if s == "mixed" else 0 for s in st])
    elif prefs.cooking == "heavy" or (prefs.household_size or 0) >= 5:
        storage = wpick(rng, st, [70 if s == "drawers" else 30 if s == "mixed" else 0 for s in st])
    elif prefs.cooking == "light":
        storage = wpick(rng, st, [0 if s == "drawers" else 40 if s == "mixed" else 60 for s in st])
    else:
        storage = wpick(rng, st, [30 if s == "drawers" else 50 if s == "mixed" else 20 for s in st])

    feat_codes = E.get("base_feature_units.units") or []
    features = []
    drawer = [c for c in feat_codes if c.startswith("DRAWERS4-")]
    if drawer and rng.random() < 0.75 and free(geo, hob_wall) >= 2400:
        features.append({"wall": L[hob_wall], "units": [rng.choice(drawer)]})
    near_sink = [c for c in feat_codes if c.startswith(("WICKER-", "GRAIN-"))]
    if near_sink and sink_wall != hob_wall and rng.random() < 0.5 and free(geo, sink_wall) >= 2400:
        features.append({"wall": L[sink_wall], "units": [rng.choice(near_sink)]})

    wall_codes = E.get("wall_cabinet_sizes.units") or []
    sizes = []
    for w in walls:
        if wall_codes and rng.random() < 0.7:
            pool = [c for c in wall_codes if code_width(c) in (600, 800, 900)] or wall_codes
            sizes.append({"wall": L[w], "units": rng.sample(pool, k=min(len(pool), rng.choice([1, 1, 2])))})

    ws = E["wall_style"]
    if len(ws) == 1:
        wall_style = ws[0]
    else:
        p_lift = 0.6 if brief_has(prefs, "minimal") else 0.5 if prefs.budget == "luxury" else 0.25
        wall_style = "lift_up" if rng.random() < p_lift else "shutters"

    base_fin, wall_fin = pick_finishes(rng, prefs, E["base_finish"], E["wall_finish"])
    lo_opts = E["lofts"]
    if len(lo_opts) == 1:
        lofts = lo_opts[0]
    else:
        lofts = "yes" if rng.random() < (0.92 if brief_has(prefs, "vessels") else 0.78) else "no"

    return {
        "room_reading": "", "design_plan": "", "layout_why": "", "cabinet_walls": [L[w] for w in walls],
        "sink_why": "", "sink_wall": L[sink_wall], "sink": sink, "sink_center_mm": sink_c,
        "hob_why": "", "hob_wall": L[hob_wall], "hob": hob, "hob_center_mm": hob_c, "spice": spice,
        "lpg_why": "", "lpg_cylinder": lpg, "tall_why": "", "tall_units": talls,
        "purifier_why": "", "water_purifier": purifier, "display_why": "", "display_unit": display,
        "open_why": "", "open_unit": open_u, "storage_why": "", "base_storage": storage,
        "features_why": "", "base_feature_units": features, "wall_cabinets_why": "", "wall_cabinet_sizes": sizes,
        "wall_style_why": "", "wall_style": wall_style, "finish_why": "", "base_finish": base_fin,
        "wall_finish": wall_fin, "loft_why": "", "lofts": lofts,
    }


_BY_CODE = None


def finish_info(code: str) -> dict:
    global _BY_CODE
    if _BY_CODE is None:
        _BY_CODE = {finishes.code(a): a for a in finishes.catalogue()}
    aid = _BY_CODE.get(code)
    return finishes.summary(aid) if aid else {}


def finish_words(code: str) -> str:
    s = finish_info(code)
    return " ".join(str(s.get(k) or "") for k in ("name", "color", "family", "category", "shutter")).lower() + " " + code.lower()


def brief_colour(prefs) -> Optional[str]:
    t = (prefs.style_notes or "").lower()
    for key, (phrase, _) in COLOURS.items():
        if phrase in t:
            return key
    return None


def pick_finishes(rng, prefs, base_codes, wall_codes):
    want = brief_colour(prefs)

    def choose(codes):
        if len(codes) == 1:
            return codes[0]
        if want:
            hits = [c for c in codes if any(w in finish_words(c) for w in COLOURS[want][1])]
            if hits:
                return rng.choice(hits)
        return rng.choice(codes)

    base = choose(base_codes)
    if base in wall_codes and rng.random() < 0.45:
        return base, base
    light = [c for c in wall_codes if any(w in finish_words(c) for w in COLOURS["white"][1])]
    wall = rng.choice(light) if light and rng.random() < 0.5 else choose(wall_codes)
    return base, wall


# ---------------------------------------------------------------------------
# reasons (written from the final, checked decisions)
# ---------------------------------------------------------------------------


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return cut[: end + 1] if end > n // 2 else cut[: cut.rfind(" ")].rstrip(",;") + "."


def write_reasons(ans: dict, geo, prefs, report: dict, rng: random.Random) -> dict:
    labels = wall_labels(geo)
    W = {lab: geo.walls[wid] for lab, wid in labels.items()}
    L = {wid: lab for lab, wid in labels.items()}
    say = rng.choice
    applied = {a.get("role"): a for a in report["rulebook"].get("applied", [])}
    band = report["rulebook"].get("size_band")
    area = report["rulebook"].get("floor_area_sqft")
    walls = ans["cabinet_walls"]
    shape = {1: "Straight", 2: "L-shape", 3: "U-shape"}[len(walls)]
    sw, hw = ans["sink_wall"], ans["hob_wall"]

    def fr(lab):
        return int(free(geo, labels[lab]))

    # room reading: windows, doors (with corner), long solid walls
    parts = []
    for lab, w in W.items():
        for o in w.windows:
            parts.append(f"window on {lab} (centre {o.center_mm:.0f})")
    for lab, w in W.items():
        for o in w.doors:
            near = w.neighbor_start if o.center_mm < w.length_mm / 2 else w.neighbor_end
            parts.append(f"door on {lab}" + (f" near the {L[near]} corner" if near in L else ""))
    solid = [lab for lab, w in W.items() if not w.openings and fr(lab) >= 2400]
    if solid:
        parts.append(f"{' and '.join(solid)} solid" + (f" ({fr(solid[0])} mm)" if len(solid) == 1 else ""))
    if not any(w.windows for w in W.values()):
        parts.insert(0, "no window")
    reading = say(["", "Room: ", "Reading the plan: "]) + "; ".join(parts) + f"; {area:.0f} sqft, {band}."
    ans["room_reading"] = clip(reading[0].upper() + reading[1:], 240)

    talls = ans["tall_units"]
    TALL_WORDS = {"OVEN-MICRO": "oven + microwave", "OVEN": "oven", "MICROWAVE": "microwave", "PANTRY": "pantry",
                  "TALL-WICKER": "wicker", "CROCKERY": "crockery"}

    def tall_word(code):
        return next((v for k, v in TALL_WORDS.items() if code.startswith(k + "-")), code.lower())

    tall_txt = (", " + " and ".join(tall_word(t["unit"]) for t in talls)
                + f" tall unit{'s' if len(talls) > 1 else ''} on {talls[0]['wall']}") if talls else ""
    plan = (f"{shape} on {', '.join(walls)}: sink on {sw}, hob on {hw}"
            + (" with the spice pull-out" if ans["spice"] != "NONE" else "")
            + (" and LPG unit beside it" if ans["lpg_cylinder"] != "NONE" else "") + tall_txt + ".")
    ans["design_plan"] = clip(plan, 200)

    fixed = [x for x in (prefs.primary_wall, prefs.secondary_wall) if x]
    if prefs.cabinet_walls:
        lw = f"The customer chose {', '.join(walls)} for the {shape.lower()}."
    else:
        why = []
        if prefs.layout != "auto":
            why.append(f"customer asked for {shape.lower()}")
        if fixed:
            why.append("keeps the customer's " + " and ".join(
                ("cooking wall " + L[prefs.primary_wall]) if x == prefs.primary_wall else ("sink wall " + L[x]) for x in dict.fromkeys(fixed)))
        win_in = [lab for lab in walls if W[lab].windows]
        if win_in and not fixed:
            why.append(f"includes the window wall {win_in[0]}")
        why.append(say([f"the longest free runs ({', '.join(f'{lab} {fr(lab)} mm' for lab in walls)})",
                        f"{sum(fr(lab) for lab in walls)} mm of usable wall"]))
        lw = f"{shape}: " + "; ".join(why) + "."
    ans["layout_why"] = clip(lw, 180)

    sink_w = code_width(ans["sink"])
    win = max(W[sw].windows, key=lambda o: o.width_mm) if W[sw].windows else None
    s = []
    if prefs.secondary_wall:
        s.append(f"{sw} is the customer's sink wall (R25)")
    elif win:
        s.append(say([f"{sw} has the window: daylight while washing", f"window wall {sw} for light and air at the sink"]))
    else:
        s.append(f"{sw} has a {fr(sw)} mm free run")
    if win and abs(ans["sink_center_mm"] - win.center_mm) <= 30:
        s.append("centred under the window")
    rule = applied.get("sink")
    rw = int((rule or {}).get("width") or 0)
    if rule and rule.get("rule") and rw and rw != sink_w:
        s.append(f"{ans['sink']}; the rule book sets {rw} mm for a {band} kitchen (R{rule['rule']})")
    elif rule and rule.get("rule"):
        s.append(f"{sink_w} mm sink per R{rule['rule']} for a {band} kitchen")
    else:
        s.append(f"{sink_w} mm sink suits the {fr(sw)} mm run")
    ans["sink_why"] = clip("; ".join(s) + ".", 180)

    h = []
    if prefs.primary_wall:
        h.append(f"{hw} is the customer's cooking wall (R24)")
    elif not W[hw].openings:
        h.append(say([f"{hw} is solid, good for the chimney", f"solid {hw} carries the hob and chimney"]))
    else:
        h.append(f"{hw} has the longest free run left ({fr(hw)} mm)")
    if hw == sw:
        h.append("over 600 mm of counter from the sink")
    elif not prefs.primary_wall:
        h.append("away from the sink so two people can work")
    rule = applied.get("hob")
    rw = int((rule or {}).get("width") or 0)
    unit = ans["hob"].replace("HOB-", "").replace("-", " ").lower() + " unit"
    if rule and rule.get("rule") and rw and rw != code_width(ans["hob"]):
        h.append(f"{unit}; the rule book sets {rw} mm here (R{rule['rule']})")
    else:
        h.append(unit + (f" per R{rule['rule']}" if rule and rule.get("rule") else ""))
    ans["hob_why"] = clip("; ".join(h) + ".", 180)

    if ans["lpg_cylinder"] == "NONE":
        ans["lpg_why"] = ("Piped gas: no cylinder unit (R11)." if prefs.fuel == "png" else
                          "Induction hob: no cylinder unit needed (R11)." if prefs.fuel == "induction" else
                          "No LPG cylinder unit (R11).")
    else:
        ans["lpg_why"] = ("Customer uses an LPG cylinder: unit next to the hob (R11)." if prefs.fuel == "lpg" else
                          "Fuel not stated, so LPG is assumed and the cylinder unit sits by the hob (R11).")

    if not prefs.include_tall:
        ans["tall_why"] = "Customer does not want tall storage."
    elif talls:
        names = [t["unit"] for t in talls]
        why = []
        if prefs.built_in not in ("auto", "none"):
            why.append(f"customer needs a {prefs.built_in.replace('_', ' + ')} tower")
        elif any(n.startswith(("OVEN", "MICROWAVE")) for n in names) and prefs.style_notes:
            why.append("the brief mentions " + ("baking" if any(n.startswith("OVEN") for n in names) else "the microwave"))
        if any(n.startswith("CROCKERY") for n in names):
            why.append("crockery unit for display")
        why.append(f"at the end of {talls[0]['wall']}'s run, clear of the window and the hob")
        ans["tall_why"] = clip(f"{' + '.join(names)}: " + "; ".join(why) + ".", 180)
    else:
        ans["tall_why"] = clip(say(["No tall unit: the runs are needed for counter and base storage.",
                                    f"Runs too short for a tall unit ({sum(fr(lab) for lab in walls)} mm in all)."]), 180)

    def need_reason(field, unit, yes_txt, no_txt, auto_yes, auto_no):
        need = getattr(prefs, field)
        if unit == "NONE":
            return no_txt if need == "no" else auto_no
        return yes_txt if need == "yes" else auto_yes

    ans["purifier_why"] = clip(need_reason(
        "water_purifier", ans["water_purifier"], "Customer needs a water purifier: wall unit near the sink.",
        "Customer does not want a purifier unit.",
        "Purifier unit near the sink" + (": the brief asks for purified water." if brief_has(prefs, "water") else
                                         ": standard in Indian kitchens."),
        "No purifier unit: not requested, and the wall space is used for storage."), 180)
    ans["display_why"] = clip(need_reason(
        "display_unit", ans["display_unit"], "Customer wants a glass display unit.", "Customer does not want a glass display.",
        "Glass display" + (" for the crockery in the brief." if brief_has(prefs, "crockery") else " to break the long wall run."),
        "No glass display: not requested."), 180)
    ans["open_why"] = clip(need_reason(
        "open_unit", ans["open_unit"], "Customer wants an open shelf unit.", "Customer does not want an open shelf.",
        "Open shelf at a run end for everyday jars.",
        "No open shelf" + (": the brief asks for a clean look." if brief_has(prefs, "minimal") else ": closed storage keeps it dust-free.")), 180)

    stg = ans["base_storage"]
    if prefs.base_storage != "auto":
        sw_txt = f"Customer asked for {stg}."
    elif brief_has(prefs, "drawer"):
        sw_txt = "The brief asks for lots of drawers."
    elif prefs.budget == "essential":
        sw_txt = "Essential budget: shutter base units."
    elif prefs.budget in ("premium", "luxury"):
        sw_txt = f"{prefs.budget.title()} budget: {'drawer' if stg == 'drawers' else 'drawer + shutter'} base units."
    elif prefs.cooking == "heavy" or (prefs.household_size or 0) >= 5:
        sw_txt = f"{'Heavy cooking' if prefs.cooking == 'heavy' else 'Big family'}: {stg} keep many vessels reachable."
    elif stg == "shutters":
        sw_txt = "Shutter base units: simple and cost-effective" + (" for light cooking." if prefs.cooking == "light" else ".")
    else:
        sw_txt = f"{'Drawers' if stg == 'drawers' else 'Drawer + shutter units'}: easy reach for daily vessels."
    ans["storage_why"] = clip(sw_txt, 180)

    if ans["base_feature_units"]:
        bits = [f"{', '.join(r['units'])} on {r['wall']}" for r in ans["base_feature_units"]]
        ans["features_why"] = clip("; ".join(bits) + ": drawers by the hob for pans, baskets by the sink.", 180)
    else:
        ans["features_why"] = "No special base units: the plain base cabinets fill the runs."
    if ans["wall_cabinet_sizes"]:
        ans["wall_cabinets_why"] = clip("Preferred sizes: " + "; ".join(
            f"{r['wall']} {', '.join(u.replace('WALL-', '') for u in r['units'])} mm" for r in ans["wall_cabinet_sizes"])
            + " - even module rhythm.", 180)
    else:
        ans["wall_cabinets_why"] = "Standard wall cabinets: the engine sizes them to the runs."
    if prefs.wall_style != "auto":
        ans["wall_style_why"] = f"Customer chose {'lift-up' if ans['wall_style'] == 'lift_up' else 'hinged shutter'} doors."
    elif ans["wall_style"] == "lift_up":
        ans["wall_style_why"] = "Lift-up doors: " + ("the clean, minimal look the brief asks for." if brief_has(prefs, "minimal")
                                                      else "no door swings into the head space.")
    else:
        ans["wall_style_why"] = "Hinged shutters: the standard, lowest-cost wall door."

    b, w = finish_info(ans["base_finish"]), finish_info(ans["wall_finish"])
    colour = brief_colour(prefs)
    fw = (f"{COLOURS[colour][0].capitalize()} as the brief asks; " if colour else "")
    fw += (f"{(b.get('category') or '')} base" + (f" and {(w.get('category') or '')} wall" if ans['wall_finish'] != ans['base_finish'] else " and wall")
           + (f", allowed for the {prefs.budget} budget." if prefs.budget != "auto" else "."))
    if ans["wall_finish"] != ans["base_finish"]:
        fw += " Lighter wall run keeps the kitchen open." if any(x in finish_words(ans["wall_finish"]) for x in COLOURS["white"][1]) else ""
    ans["finish_why"] = clip(fw, 180)

    hgt = report["heights"]
    if ans["lofts"] == "yes":
        ans["loft_why"] = clip(("Customer wants lofts" if prefs.include_loft else
                                "The brief mentions big vessels" if brief_has(prefs, "vessels") else
                                "Lofts store big vessels and seasonal items")
                               + f"; the {hgt['ceiling_mm']} mm ceiling allows them (R48).", 180)
    elif prefs.include_loft is False:
        ans["loft_why"] = "Customer does not want lofts."
    elif hgt["ceiling_mm"] < config.LOFT_TOP_M * 1000:
        ans["loft_why"] = f"Ceiling {hgt['ceiling_mm']} mm is too low for lofts (R47): wall units go taller instead."
    else:
        ans["loft_why"] = "No lofts: a lighter look, wall units carry the storage."
    return ans


# ---------------------------------------------------------------------------
# the pipeline check
# ---------------------------------------------------------------------------

CTX: dict = {}


def _design_kitchen(geometry, catalog, prefs, version=None):
    """Stands in for the AI designer: builds the production prompt, answers with the policy (pass 1)
    or the corrected answer (pass 2), validates it against the schema, returns the proposal."""
    groups = build_shortlist_v2(catalog)
    messages, Model, to_proposal = build_messages(geometry, groups, prefs, "v2")
    CTX.update(messages=messages, Model=Model, prefs=prefs, menu=build_menu(groups))
    # pass 1: the designer answers (the policy here, or another designer such as Qwen on Colab set in
    # CTX["designer"]); repairs and pass 2 reuse the answer already in CTX
    designer = CTX.get("designer") or (lambda g, p, M_, msgs: design(g, p, M_, CTX["rng"]))
    ans = CTX.get("answer") or designer(geometry, prefs, Model, messages)
    d = Model.model_validate(ans)
    CTX["answer"] = json.loads(d.model_dump_json())
    proposal = to_proposal(d)
    proposal.source = "ollama"
    return proposal, {"mode": "dataset", "prompt_version": "v2"}


main.design_kitchen = _design_kitchen


_HYDRATE = main.hydrate_scene


def run(doc: dict, prefs: KitchenPreferences, scene: bool = True):
    """The production pipeline; scene=False skips the 3D build (pass 1 only needs the layout)."""
    main.hydrate_scene = _HYDRATE if scene else (lambda d, layout, seed=None, include_floor=True: (d, {}))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return main.generate_kitchen(doc, prefs)
    finally:
        main.hydrate_scene = _HYDRATE


def placements(report) -> list[dict]:
    return [dict(p, wall=wid) for wid, ps in report["layout"]["placements"].items() for p in ps]


def adopt(ans: dict, report: dict, geo) -> tuple[dict, list[str]]:
    """Take what the rule book / engine built into the answer. Returns (answer, problems that
    cannot be adopted)."""
    labels = wall_labels(geo)
    L = {wid: lab for lab, wid in labels.items()}
    E = schema_enums(CTX["Model"])
    menu = CTX["menu"]
    cat_of = {c: it.cat for opts in menu.values() for c, it in opts}
    pls = placements(report)
    bad = []

    def one(role):
        return next((p for p in pls if p["role"] == role), None)

    def code_for(choices, width, cat=None):
        hits = [c for c in choices if c != "NONE" and code_width(c) == width and (cat is None or cat_of.get(c) == cat)]
        return hits[0] if hits else None

    final_walls = [L[w] for w in report["layout"]["cabinet_walls"]]
    if set(final_walls) - set(E["cabinet_walls"] or final_walls):
        bad.append("walls outside the schema")
    ans["cabinet_walls"] = final_walls
    for role, wall_key, unit_key, c_key in (("sink", "sink_wall", "sink", "sink_center_mm"), ("hob", "hob_wall", "hob", "hob_center_mm")):
        p = one(role)
        if p is None:
            bad.append(f"no {role} built")
            continue
        if L[p["wall"]] != ans[wall_key]:
            if len(E[wall_key]) > 1 and L[p["wall"]] in E[wall_key]:
                ans[wall_key] = L[p["wall"]]  # not the customer's wall: the engine's wall is a valid choice
            else:
                bad.append(f"{role} moved to {L[p['wall']]}")  # the customer chose that wall (R24 / R25)
        rule = next((a for a in report["rulebook"].get("applied", []) if a.get("role") == role), {})
        item_of = {x: it for opts in menu.values() for x, it in opts}
        # the rule book's size wins when the code list has it (the engine may still narrow it to fit)
        by_rule = [x for x in E[unit_key] if x != "NONE" and rule.get("width") and code_width(x) == int(rule["width"])]
        same = by_rule or [x for x in E[unit_key] if x != "NONE" and code_width(x) == int(p["width"])]
        c = (next((x for x in same if item_of.get(x) is not None and item_of[x].aid == p["aid"]), None)
             or next((x for x in same if item_of.get(x) is not None and rule.get("combination")
                      and (item_of[x].cabinet_combination or "").lower() == str(rule["combination"]).lower()), None)
             or (same[0] if same else None))
        if c and c != ans[unit_key]:
            ans[unit_key] = c
        elif not c:  # e.g. R69 wants an 800 mm hob, the list offers 750 / 900: closest code, rule sets width
            ans[unit_key] = min(E[unit_key], key=lambda x: (abs(code_width(x) - int(p["width"])), x != ans[unit_key]))
        ans[c_key] = int(round((p["start_mm"] + p["end_mm"]) / 2 / 10) * 10)
    sp = one("spice")
    if sp is None and ans["spice"] != "NONE":
        if "NONE" in E["spice"]:
            ans["spice"] = "NONE"
        else:
            bad.append("spice dropped")
    elif sp is not None:
        ans["spice"] = code_for(E["spice"], int(sp["width"])) or ans["spice"]
    cyl = one("cylinder")
    if cyl is None and ans["lpg_cylinder"] != "NONE":
        bad.append("cylinder dropped")
    for key, role in (("water_purifier", "purifier"), ("display_unit", "display"), ("open_unit", "open")):
        p = one(role)
        if p is None and ans[key] != "NONE":
            if "NONE" in E[key]:
                ans[key] = "NONE"
            else:
                bad.append(f"{role} dropped")
        elif p is not None:
            ans[key] = code_for(E[key], int(p["width"])) or ans[key]
    tall_codes = E.get("tall_units.unit") or []
    built_talls = [p for p in pls if p["tier"] == "tall"]
    new_talls = []
    for t in ans["tall_units"]:
        hit = next((p for p in built_talls if code_for([t["unit"]], int(p["width"]), cat_of.get(t["unit"]))
                    and p["category"] and p not in [x[1] for x in new_talls]), None)
        if hit:
            new_talls.append(({"wall": L[hit["wall"]], "unit": t["unit"]}, hit))
    ans["tall_units"] = [t for t, _ in new_talls]
    if len(ans["tall_units"]) < (1 if CTX["prefs"].built_in not in ("auto", "none") and CTX["prefs"].include_tall else 0):
        bad.append("needed tower not built")
    lofts = "yes" if report["heights"].get("lofts") else "no"
    if lofts != ans["lofts"]:
        if lofts in E["lofts"]:
            ans["lofts"] = lofts
        else:
            bad.append("lofts differ")
    _ = tall_codes
    return ans, bad


def corrections(report, ans: dict) -> list[str]:
    lay = report["layout"]
    out = [m for m in lay["decisions"] + lay["warnings"] if any(p in m for p in CORRECTION_PATTERNS)]
    E = schema_enums(CTX["Model"])
    for a in report["rulebook"].get("applied", []):
        if str(a.get("changed")) != "True":
            continue
        key = {"sink": "sink", "hob": "hob", "spice": "spice", "cylinder": "lpg_cylinder"}.get(a.get("role"))
        if key and not any(code_width(c) == int(a.get("width") or 0) for c in E[key]):
            continue  # the rule's width has no code: the rule book sets it (as in production)
        out.append(f"R{a.get('rule')} changed {a.get('role')}")
    return out


def repair(ans: dict, bad: list[str], adopted: dict, geo, tried: set) -> Optional[dict]:
    """What a designer would change when the engine could not build the plan: move the tall unit
    to another wall, the hob to another wall (or a narrower hob) away from tall units, recentre.
    None = nothing left to try."""
    E = schema_enums(CTX["Model"])
    labels = wall_labels(geo)
    ans = json.loads(json.dumps(ans))
    walls = ans["cabinet_walls"]

    def mid(lab, width):
        s, e = max(_free_segments(geo, labels[lab]), key=lambda t: t[1] - t[0])
        return int(round((s + e) / 2 / 10) * 10)

    hob_trouble = any(b.startswith(("no hob", "hob moved", "cylinder dropped")) for b in bad)
    tall_lost = len(adopted["tall_units"]) < len(ans["tall_units"]) or any(b.startswith("needed tower") for b in bad)
    if hob_trouble:
        # tall units off the hob wall first, then a narrower hob, then another hob wall
        if any(t["wall"] == ans["hob_wall"] for t in ans["tall_units"]) and ("talls-off-hob", ans["hob_wall"]) not in tried:
            tried.add(("talls-off-hob", ans["hob_wall"]))
            other = [w for w in walls if w != ans["hob_wall"]]
            if other:
                ans["tall_units"] = [dict(t, wall=other[0]) for t in ans["tall_units"]]
            else:
                ans["tall_units"] = ans["tall_units"][:1]
            ans["hob_center_mm"] = mid(ans["hob_wall"], code_width(ans["hob"]))
            return ans
        narrow = min(E["hob"], key=code_width)
        if code_width(narrow) < code_width(ans["hob"]) and ("narrow", ans["hob_wall"]) not in tried:
            tried.add(("narrow", ans["hob_wall"]))
            ans["hob"], ans["hob_center_mm"] = narrow, mid(ans["hob_wall"], code_width(narrow))
            return ans
        if ans["hob_wall"] == ans["sink_wall"] and ("ends",) not in tried:
            # one wall carries both: sink near one end, hob towards the other, so the hob gets its landings
            tried.add(("ends",))
            s, e = max(_free_segments(geo, labels[ans["sink_wall"]]), key=lambda t: t[1] - t[0])
            sw_, hw_ = code_width(ans["sink"]), code_width(ans["hob"])
            ans["sink_center_mm"] = int(round((s + sw_ / 2 + 150) / 10) * 10)
            ans["hob_center_mm"] = int(round((e - hw_ / 2 - 900) / 10) * 10)
            ans["tall_units"] = ans["tall_units"][:1]
            return ans
        if ans["hob_wall"] == ans["sink_wall"]:  # one wall carries both: a narrower sink, then no spice pull-out
            small_sink = min(E["sink"], key=code_width)
            if code_width(small_sink) < code_width(ans["sink"]) and ("narrow-sink",) not in tried:
                tried.add(("narrow-sink",))
                ans["sink"] = small_sink
                return ans
        if ans["spice"] != "NONE" and "NONE" in E["spice"] and ("no-spice",) not in tried:
            tried.add(("no-spice",))
            ans["spice"] = "NONE"
            return ans
        if len(E["hob_wall"]) > 1:
            for lab in sorted(walls, key=lambda w: -free(geo, labels[w])):
                if lab != ans["hob_wall"] and ("hob-wall", lab) not in tried and lab in E["hob_wall"]:
                    tried.add(("hob-wall", lab))
                    ans["hob_wall"], ans["hob_center_mm"] = lab, mid(lab, code_width(ans["hob"]))
                    ans["tall_units"] = [dict(t, wall=next((w for w in walls if w != lab), t["wall"])) for t in ans["tall_units"]]
                    return ans
        return None
    if tall_lost:
        for lab in walls:
            if ("tall-wall", lab) not in tried and any(t["wall"] != lab for t in ans["tall_units"]):
                tried.add(("tall-wall", lab))
                ans["tall_units"] = [dict(t, wall=lab) for t in ans["tall_units"]]
                return ans
        if len(ans["tall_units"]) > 1 and ("one-tall",) not in tried:
            tried.add(("one-tall",))
            ans["tall_units"] = ans["tall_units"][:1]
            return ans
        return None
    if any(b.startswith("sink moved") for b in bad) and ("sink-mid",) not in tried:
        tried.add(("sink-mid",))
        ans["sink_center_mm"] = mid(ans["sink_wall"], code_width(ans["sink"]))
        return ans
    return None


FATAL: tuple = ()  # errors that must stop the run instead of rejecting one example (e.g. Colab gone)


def first_pass(doc, geo, prefs, stats) -> tuple[Optional[dict], str]:
    """Pass 1 with up to 6 repairs. Returns (answer with the engine's result adopted, reason)."""
    CTX["answer"] = None
    try:
        _, rep1 = run(doc, prefs, scene=False)
    except Exception as exc:  # noqa: BLE001
        if FATAL and isinstance(exc, FATAL):
            raise
        return None, f"pass 1 {type(exc).__name__}"
    tried: set = set()
    for _ in range(7):
        ans = CTX["answer"]
        adopted, bad = adopt(json.loads(json.dumps(ans)), rep1, geo)
        need = 1 if (CTX["prefs"].include_tall and (schema_enums(CTX["Model"]).get("tall_units.unit") or [])) else 0
        if not bad and len(adopted["tall_units"]) >= need:
            try:
                return json.loads(CTX["Model"].model_validate(adopted).model_dump_json()), ""
            except Exception:  # noqa: BLE001
                return None, "adopted answer invalid"
        if len(adopted["tall_units"]) < need and not bad:
            bad = ["tall unit dropped"]
        fixed = repair(ans, bad, adopted, geo, tried)
        if fixed is None:
            return None, bad[0]
        stats["repairs"] += 1
        try:
            CTX["answer"] = json.loads(CTX["Model"].model_validate(fixed).model_dump_json())
            _, rep1 = run(doc, prefs, scene=False)
        except Exception as exc:  # noqa: BLE001
            return None, f"repair {type(exc).__name__}"
    return None, "repairs exhausted"


BRIEF_FLAGS = {"bake": ("bake",), "microwave": ("microwave",), "lots_of_drawers": ("drawer",),
               "big_vessels": ("vessels",), "purified_water": ("water",), "show_crockery": ("crockery",),
               "minimal_look": ("minimal",)}
REASON_KEYS = ("room_reading", "design_plan")


def _openings(items) -> list[dict]:
    return [{"start_mm": round(o.start_mm), "end_mm": round(o.end_mm), "width_mm": round(o.width_mm),
             "center_mm": round(o.center_mm)} for o in sorted(items, key=lambda o: o.start_mm)]


def structured_record(room: Room, geo, prefs, ans: dict, report: dict, score: dict) -> dict:
    """One example for a model trained from scratch: structured input (room + customer), the design
    decisions, the final cabinet-by-cabinet layout (AIDs from the MagickHome catalogue), the reasons
    and the rules the rule book applied."""
    labels = wall_labels(geo)
    L = {wid: lab for lab, wid in labels.items()}
    walls = []
    for wid in geo.wall_order:
        w = geo.walls[wid]
        segs = _free_segments(geo, wid)
        longest = max((b - a for a, b in segs), default=0.0)
        walls.append({
            "wall": L[wid], "length_mm": w.length_mm, "height_mm": round(w.height_m * 1000),
            "corner_at_start": L.get(w.neighbor_start), "corner_at_end": L.get(w.neighbor_end),
            "windows": _openings(w.windows), "doors": _openings(w.doors),
            "free_runs_mm": [[round(a), round(b)] for a, b in segs], "longest_free_mm": round(longest),
            "can_carry_cabinets": longest >= MIN_WALL,
        })
    rb = report["rulebook"]
    brief = prefs.style_notes or ""
    customer = {k: getattr(prefs, k) for k in (
        "layout", "budget", "fuel", "water_purifier", "built_in", "display_unit", "base_storage", "wall_style",
        "open_unit", "crockery_unit", "cooking", "household_size", "include_loft", "include_tall",
        "door_style", "handle_type")}
    customer.update(
        cabinet_walls=[L[w] for w in prefs.cabinet_walls] if prefs.cabinet_walls else None,
        cooking_wall=L.get(prefs.primary_wall), sink_wall=L.get(prefs.secondary_wall), brief=brief,
        brief_flags={k: any(x in brief.lower() for x in v) for k, v in BRIEF_FLAGS.items()},
        brief_colour=brief_colour(prefs))
    menu = CTX["menu"]
    aid_of = {c: it.aid for opts in menu.values() for c, it in opts}
    decisions = {k: v for k, v in ans.items() if not k.endswith("_why") and k not in REASON_KEYS}
    chosen = [decisions["sink"], decisions["hob"], decisions["spice"], decisions["lpg_cylinder"],
              decisions["water_purifier"], decisions["display_unit"], decisions["open_unit"]]
    chosen += [t["unit"] for t in decisions["tall_units"]]
    chosen += [u for r in decisions["base_feature_units"] for u in r["units"]]
    chosen += [u for r in decisions["wall_cabinet_sizes"] for u in r["units"]]
    decisions["code_aids"] = {c: aid_of[c] for c in dict.fromkeys(chosen) if c in aid_of}
    layout = []
    for wid, ps in report["layout"]["placements"].items():
        cabs = sorted(ps, key=lambda q: ({"base": 0, "tall": 1, "wall": 2, "loft": 3}.get(q["tier"], 4), q["start_mm"]))
        layout.append({"wall": L[wid], "cabinets": [
            {"tier": q["tier"], "role": q["role"], "aid": q["aid"], "category": q["category"], "width_mm": q["width"],
             "start_mm": round(q["start_mm"]), "end_mm": round(q["end_mm"]), "opening": q.get("opening")}
            for q in cabs]})
    layout.sort(key=lambda r: r["wall"])
    return {
        "id": room.key,
        "room_spec": {"width_mm": room.width_mm, "depth_mm": room.depth_mm, "height_m": room.height_m,
                      "windows": room.windows, "doors": room.doors,
                      "note": "rebuild the floor plan with designer_benchmark.Room(**room_spec).build()"},
        "input": {
            "room": {"floor_area_sqft": rb.get("floor_area_sqft"), "size_band": rb.get("size_band"),
                     "ceiling_mm": report["heights"]["ceiling_mm"], "lofts_possible": report["heights"]["ceiling_mm"] + 1 >= config.LOFT_TOP_M * 1000,
                     "walls": walls},
            "customer": customer,
        },
        "output": {
            "decisions": decisions,
            "layout": layout,
            "heights": {k: report["heights"][k] for k in ("wall_mm", "tall_mm", "lofts", "ceiling_mm", "top_mm")},
            "kitchen_type": rb.get("kitchen_type"),
        },
        "reasons": {k: v for k, v in ans.items() if k.endswith("_why") or k in REASON_KEYS},
        "rules_applied": [{"rule": a.get("rule"), "role": a.get("role"), "width_mm": a.get("width"),
                           "text": a.get("text")} for a in rb.get("applied", [])],
        "checks": {"engine_corrections": 0, "checklist_score": score["score"], "checklist_failed": score["failed"],
                   "scene_verified": True},
    }


def make_example(i: int, rng: random.Random, stats: collections.Counter) -> Optional[dict]:
    for attempt in range(8):
        room = sample_room(rng, i)
        doc = room.build()
        geo = parse_scene(doc)
        if not any(free(geo, w) >= MIN_WALL for w in geo.wall_order):
            stats["reject: no usable wall"] += 1
            continue
        prefs = sample_prefs(rng, geo)
        CTX.clear()
        CTX["rng"] = rng
        ans, why = first_pass(doc, geo, prefs, stats)
        if ans is None and prefs.include_tall and ("tall" in why or "tower" in why or "hob" in why):
            # no room for tall storage here: the same room for a customer without tall units
            prefs = prefs.model_copy(update={"include_tall": False,
                                             "built_in": "none" if prefs.built_in != "auto" else "auto",
                                             "crockery_unit": "no" if prefs.crockery_unit == "yes" else prefs.crockery_unit})
            CTX.clear()
            CTX["rng"] = rng
            ans, why = first_pass(doc, geo, prefs, stats)
            stats["no-tall variant"] += 1
        if ans is None:
            stats["reject: " + " ".join(why.split(" ")[:3])] += 1
            continue
        CTX["answer"] = ans
        try:
            out, rep2 = run(doc, prefs)
        except Exception as exc:  # noqa: BLE001
            stats[f"reject: pass 2 {type(exc).__name__}"] += 1
            continue
        fixes = corrections(rep2, ans)
        if fixes:
            stats["reject: engine corrected: " + re.sub(r"[\d.]+", "#", fixes[0])[:60]] += 1
            continue
        pls = placements(rep2)
        shift = [abs((p["start_mm"] + p["end_mm"]) / 2 - ans[k]) for p in pls for r, k in
                 (("sink", "sink_center_mm"), ("hob", "hob_center_mm")) if p["role"] == r]
        if any(s > 60 for s in shift):
            stats["reject: position shifted"] += 1
            continue
        loft_y = (config.POS_Y_M["wall"] * 1000 + rep2["heights"]["wall_mm"] + config.LOFT_BOTTOM_MM) / 1000
        errs = [e for e in verify_ai_scene(out, expect_door=DOOR_ID)
                if not (": loft y=" in e and abs(float(e.split("loft y=")[1].split()[0]) - loft_y) < 0.005)]
        if errs:
            stats["reject: scene check: " + errs[0][:50]] += 1
            continue
        geo2 = parse_scene(doc)
        catalog = catalog_loader.load_filtered_catalog(CTX["prefs"].door_style, CTX["prefs"].handle_type)
        ans = write_reasons(ans, geo2, CTX["prefs"], rep2, rng)
        ans = json.loads(CTX["Model"].model_validate(ans).model_dump_json())
        rep2["proposal"]["notes"] = f"{ans['room_reading']} {ans['design_plan']}"
        sc = score_design(geo2, catalog, rep2, CTX["prefs"])
        must = ("walls_connected", "hob_position_sane", "sink_hob_apart", "reads_room")
        if any(sc["checks"].get(k) is False for k in must) or sc["score"] < 75:
            stats["reject: checklist " + ",".join(sc["failed"][:2])] += 1
            continue
        stats["kept"] += 1
        structured = structured_record(room, geo2, CTX["prefs"], ans, rep2, sc)
        if SCENE_DIR:  # the floor plan and the built 3D scene, for reference
            sd = Path(SCENE_DIR)
            sd.mkdir(parents=True, exist_ok=True)
            (sd / f"{room.key}_floorplan.json").write_text(json.dumps(doc), encoding="utf-8")
            (sd / f"{room.key}_kitchen.json").write_text(json.dumps(out), encoding="utf-8")
        msgs = CTX["messages"]
        assistant = json.dumps(ans, ensure_ascii=False)
        return {
            "structured": structured,
            "messages": [msgs[0], msgs[1], {"role": "assistant", "content": assistant}],
            "meta": {
                "id": room.key, "room": room.title, "attempt": attempt + 1,
                "prefs": json.loads(CTX["prefs"].model_dump_json(exclude={"ai_mode", "ai_provider", "seed", "read_brief"})),
                "shape": len(ans["cabinet_walls"]), "score": sc["score"], "failed_checks": sc["failed"],
                "size_band": rep2["rulebook"].get("size_band"), "lofts": ans["lofts"],
                "rules_applied": sorted({str(a.get("rule")) for a in rep2["rulebook"].get("applied", [])}),
                "chars": {m["role"]: len(m["content"]) for m in msgs} | {"assistant": len(assistant)},
            },
        }
    return None


SCENE_DIR: Optional[str] = None


def make_one(args: tuple) -> tuple[Optional[dict], dict]:
    """Worker: example ``i`` with its own seed (the same i and seed give the same example)."""
    global SCENE_DIR
    i, seed, SCENE_DIR = args
    stats: collections.Counter = collections.Counter()
    ex = make_example(i, random.Random(seed * 1_000_003 + i), stats)
    return ex, dict(stats)


FEATURE_COLUMNS: list[str] = []
LABEL_COLUMNS: list[str] = []


def flat_rows(rec: dict) -> tuple[dict, dict]:
    """One row of features (room + customer) and one row of labels (the decisions) for classic ML.
    Walls are the four slots Wall 1-4 in order around the room."""
    room, cust, dec = rec["input"]["room"], rec["input"]["customer"], rec["output"]["decisions"]
    f: dict[str, Any] = {"id": rec["id"], "width_mm": rec["room_spec"]["width_mm"], "depth_mm": rec["room_spec"]["depth_mm"],
                         "ceiling_mm": room["ceiling_mm"], "floor_area_sqft": room["floor_area_sqft"],
                         "size_band": room["size_band"], "lofts_possible": int(room["lofts_possible"])}
    for k, w in enumerate(room["walls"][:4], 1):
        win = max(w["windows"], key=lambda o: o["width_mm"]) if w["windows"] else None
        f.update({f"w{k}_length_mm": w["length_mm"], f"w{k}_windows": len(w["windows"]), f"w{k}_doors": len(w["doors"]),
                  f"w{k}_window_width_mm": win["width_mm"] if win else 0, f"w{k}_window_center_mm": win["center_mm"] if win else -1,
                  f"w{k}_door_width_mm": sum(d["width_mm"] for d in w["doors"]),
                  f"w{k}_longest_free_mm": w["longest_free_mm"], f"w{k}_free_runs": len(w["free_runs_mm"]),
                  f"w{k}_can_carry": int(w["can_carry_cabinets"])})
    for k in ("layout", "budget", "fuel", "water_purifier", "built_in", "display_unit", "base_storage", "wall_style",
              "open_unit", "crockery_unit", "cooking"):
        f[f"cust_{k}"] = cust[k]
    f["cust_household_size"] = cust["household_size"] or 0
    f["cust_include_loft"] = {None: "auto", True: "yes", False: "no"}[cust["include_loft"]]
    f["cust_include_tall"] = int(cust["include_tall"])
    for k in ("cooking_wall", "sink_wall"):
        f[f"cust_{k}"] = cust[k] or "auto"
    for k in range(1, 5):
        f[f"cust_ticked_w{k}"] = int(f"Wall {k}" in (cust["cabinet_walls"] or []))
    for k, v in cust["brief_flags"].items():
        f[f"brief_{k}"] = int(v)
    f["brief_colour"] = cust["brief_colour"] or "none"
    talls = dec["tall_units"] + [{"wall": "none", "unit": "NONE"}] * 2
    lab: dict[str, Any] = {"id": rec["id"], "shape": {1: "straight", 2: "L", 3: "U"}[len(dec["cabinet_walls"])]}
    for k in range(1, 5):
        lab[f"cabinets_w{k}"] = int(f"Wall {k}" in dec["cabinet_walls"])
    lab.update({"sink_wall": dec["sink_wall"], "sink": dec["sink"], "sink_center_mm": dec["sink_center_mm"],
                "hob_wall": dec["hob_wall"], "hob": dec["hob"], "hob_center_mm": dec["hob_center_mm"],
                "spice": dec["spice"], "lpg_cylinder": dec["lpg_cylinder"],
                "tall1_wall": talls[0]["wall"], "tall1_unit": talls[0]["unit"],
                "tall2_wall": talls[1]["wall"], "tall2_unit": talls[1]["unit"],
                "water_purifier": dec["water_purifier"], "display_unit": dec["display_unit"], "open_unit": dec["open_unit"],
                "base_storage": dec["base_storage"], "wall_style": dec["wall_style"], "base_finish": dec["base_finish"],
                "wall_finish": dec["wall_finish"], "lofts": dec["lofts"]})
    return f, lab


def write_knowledge(out: Path) -> dict:
    """The MagickHome knowledge the examples are built on: cabinets (assets), the option codes the
    designer picks, the rule workbook and the finishes."""
    kd = out / "knowledge"
    kd.mkdir(parents=True, exist_ok=True)
    catalog = catalog_loader.load_filtered_catalog(config.DEFAULT_DOOR_STYLE, config.DEFAULT_HANDLE_TYPE)
    keep = ("aid", "description", "width", "depth", "height", "unit_family", "category", "door_type",
            "cabinet_combination", "shutter_count", "drawer_count", "open_box_count", "door_style",
            "opening_direction", "handle_type", "internals", "is_accessory")
    cabinets = [{k: getattr(it, k) for k in keep} for it in catalog.items]
    (kd / "cabinets.json").write_text(json.dumps({"source": catalog.source, "door_style": config.DEFAULT_DOOR_STYLE,
                                                   "count": len(cabinets), "cabinets": cabinets}, indent=1), encoding="utf-8")
    menu = build_menu(build_shortlist_v2(catalog))
    codes = {g: [{"code": c, "aid": it.aid, "category": it.category, "width": it.width, "depth": it.depth,
                  "height": it.height, "combination": it.cabinet_combination, "opening": it.opening_direction}
                 for c, it in opts] for g, opts in menu.items()}
    (kd / "codes.json").write_text(json.dumps(codes, indent=1), encoding="utf-8")
    rb = rulebook.load()
    rules = {"source": rb.source,
             "rules": [{"rule": n, "text": t} for n, t in sorted(rb.texts.items())],
             "module_widths": [{"rule": w.rule, "unit": w.trigger, "kitchen_type": w.kitchen_type, "size_band": w.size,
                                "width_mm": w.width, "combination": w.combination} for w in rb.widths],
             "groups": {k: list(v) for k, v in (rb.groups or {}).items()}}
    (kd / "rules.json").write_text(json.dumps(rules, indent=1, ensure_ascii=False), encoding="utf-8")
    fins = [dict(finishes.summary(a), aid=a) for a in finishes.catalogue()]
    (kd / "finishes.json").write_text(json.dumps({"tiers": finishes.rules(), "finishes": fins}, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    return {"cabinets": len(cabinets), "codes": sum(len(v) for v in codes.values()), "rules": len(rules["rules"]),
            "module_width_rules": len(rules["module_widths"]), "finishes": len(fins)}


def main_cli() -> int:
    import csv
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--val", type=float, default=0.05, help="share kept for validation")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--scenes", type=int, default=20, help="save the floor plan + 3D kitchen JSON of this many examples")
    ap.add_argument("--out", default=str(ROOT / "training" / "dataset"))
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("scratch", "llm", "scenes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    knowledge = write_knowledge(out)
    stats: collections.Counter = collections.Counter()
    rows, t0, i = [], time.time(), 0
    files = {"train": open(out / "llm" / "train.jsonl", "w", encoding="utf-8"),
             "val": open(out / "llm" / "val.jsonl", "w", encoding="utf-8"),
             "s_train": open(out / "scratch" / "train.jsonl", "w", encoding="utf-8"),
             "s_val": open(out / "scratch" / "val.jsonl", "w", encoding="utf-8"),
             "meta": open(out / "meta.jsonl", "w", encoding="utf-8")}
    feats, labs = [], []
    every = max(2, round(1 / args.val)) if args.val > 0 else 0
    try:
        with mp.Pool(args.workers) as pool:
            while len(rows) < args.n:
                batch = [(k, args.seed, str(out / "scenes") if k < args.scenes * 2 else None)
                         for k in range(i, i + args.workers * 4)]
                i += len(batch)
                for ex, st in pool.imap(make_one, batch):
                    stats.update(st)
                    if ex is None or len(rows) >= args.n:
                        continue
                    split = "val" if every and len(rows) % every == every - 1 else "train"
                    stats[split] += 1
                    files[split].write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
                    rec = dict(ex["structured"], split=split)
                    files["s_" + split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    files["meta"].write(json.dumps(dict(ex["meta"], split=split), ensure_ascii=False) + "\n")
                    f_row, l_row = flat_rows(rec)
                    feats.append(dict(f_row, split=split))
                    labs.append(dict(l_row, split=split))
                    rows.append(ex["meta"])
                for f in files.values():
                    f.flush()
                el = time.time() - t0
                print(f"{len(rows)}/{args.n} kept (example slots tried {i}) - {el / 60:.1f} min, "
                      f"~{el / max(1, len(rows)) * (args.n - len(rows)) / 60:.0f} min left", flush=True)
    finally:
        for f in files.values():
            f.close()
    for name, table in (("features.csv", feats), ("labels.csv", labs)):
        if table:
            with open(out / "scratch" / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
    kept_scenes = {r["id"] for r in rows if (out / "scenes" / f"{r['id']}_kitchen.json").exists()}
    kept_scenes = set(sorted(kept_scenes)[: args.scenes])
    for fp in (out / "scenes").glob("*.json"):  # scenes of examples not kept (tried in parallel) are removed
        if fp.name.split("_")[0] not in kept_scenes:
            fp.unlink()
    summary = {
        "examples": len(rows), "train": stats["train"], "val": stats["val"], "example_slots_tried": i,
        "minutes": round((time.time() - t0) / 60, 1),
        "rejected": {k: v for k, v in stats.most_common() if k.startswith("reject")},
        "shapes": dict(collections.Counter({1: "straight", 2: "L", 3: "U"}[r["shape"]] for r in rows)),
        "size_bands": dict(collections.Counter(r["size_band"] for r in rows)),
        "lofts": dict(collections.Counter(r["lofts"] for r in rows)),
        "with_brief": sum(1 for r in rows if r["prefs"].get("style_notes")),
        "customer_cooking_wall": sum(1 for r in rows if r["prefs"].get("primary_wall")),
        "customer_walls_ticked": sum(1 for r in rows if r["prefs"].get("cabinet_walls")),
        "budgets": dict(collections.Counter(r["prefs"].get("budget") for r in rows)),
        "checklist_score_avg": round(sum(r["score"] for r in rows) / max(1, len(rows)), 1),
        "avg_chars": {k: round(sum(r["chars"][k] for r in rows) / max(1, len(rows))) for k in ("system", "user", "assistant")},
        "repairs": stats["repairs"], "no_tall_variants": stats["no-tall variant"],
        "knowledge": knowledge,
        "scenes_saved": len(list((out / "scenes").glob("*_kitchen.json"))),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
