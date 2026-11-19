"""AI interior designer: local Ollama Qwen 2.5 Coder (``config.OLLAMA_MODEL``) with a deterministic fallback.

Qwen's job is *design intent* - which walls carry cabinets, where the sink / hob /
tall units go and which SKUs to use. It never emits coordinates that are
trusted blindly: :mod:`rule_engine` snaps and validates everything.

Hallucination guard: the response format is a JSON schema generated from a
Pydantic model whose fields are ``Literal`` enums built from the filtered
catalogue shortlist (35-45 SKUs) and the scene's walls. Ollama compiles the
schema into a grammar, so the model can only produce valid options; the reply
is then re-validated with the same Pydantic model.

Two prompt versions (``config.AI_PROMPT_VERSION``):

* ``v2`` (default) teaches Qwen with the knowledge pack in ``knowledge/``
  (designer playbook, worked examples, a designer note on every option) and
  uses short readable option codes (``SINK-900``, ``HOB-900-3DRAWER``) and wall
  names instead of 30-character AIDs and wall ids. It also offers the water
  purifier, glass display and built-in microwave / oven-micro units.
* ``v1`` is the original prompt, kept for A/B comparison
  (``designer_benchmark.py``).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, create_model

import config
import design_knowledge
from catalog_loader import (
    CAT_AQUAGUARD,
    CAT_BASE,
    CAT_CYLINDER,
    CAT_GRAIN,
    CAT_HOB,
    CAT_MICRO,
    CAT_OVEN,
    CAT_OVEN_MICRO,
    CAT_SINK,
    CAT_SPICE,
    CAT_TALL,
    CAT_TALL_WICKER,
    CAT_WALL,
    CAT_WICKER,
    WALL_GLASS_CATS,
    Catalog,
    CatalogItem,
    canon,
)
from rule_engine import Proposal, SolverError, suggest_cabinet_walls
from scene_parser import RoomGeometry

log = logging.getLogger(__name__)


class AIUnavailable(RuntimeError):
    """Ollama not reachable, model missing, or the reply was unusable."""


class KitchenPreferences(BaseModel):
    """Customer preferences accepted by the API / CLI."""

    door_style: str = Field(config.DEFAULT_DOOR_STYLE, description="'Slab (Plain)' or 'Shaker -1'")
    handle_type: str = Field(config.DEFAULT_HANDLE_TYPE)
    layout: Literal["auto", "single", "L", "U"] = "auto"
    cabinet_walls: Optional[list[str]] = Field(None, description="Force these wall ids to carry cabinets")
    include_loft: bool = Field(False, description="Add loft units when the wall height allows (>= 2.658 m)")
    include_tall: bool = True
    include_cylinder: Optional[bool] = Field(None, description="LPG cylinder unit beside the hob (None = auto)")
    sink_under_window: bool = True
    style_notes: str = Field("", max_length=500, description="Free-text brief passed to the AI designer")
    # Household needs. "auto" = not stated: Qwen decides and explains why. Any other
    # value is the customer's requirement and is enforced (Qwen cannot override it).
    water_purifier: Literal["auto", "yes", "no"] = Field("auto", description="Wall purifier (Aquaguard) cabinet")
    built_in: Literal["auto", "none", "microwave", "oven", "oven_microwave"] = Field(
        "auto", description="Built-in appliance tower(s) the household needs"
    )
    display_unit: Literal["auto", "yes", "no"] = Field("auto", description="Glass display wall cabinet")
    cooking: Literal["auto", "heavy", "light"] = Field("auto", description="Daily heavy cooking or light cooking")
    household_size: Optional[int] = Field(None, ge=1, le=20, description="People living in the home")
    ai_mode: Literal["auto", "ollama", "fallback"] = Field(
        "auto", description="auto = Ollama with deterministic fallback; ollama = require the LLM"
    )
    strip_existing_items: bool = True
    include_floor: bool = True
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Catalogue shortlist (what the LLM may choose from)
# ---------------------------------------------------------------------------

def _picker(catalog: Catalog):
    def pick(cat, widths, opening=None, combination=None, height=None, depth=None, limit=None):
        out = []
        for w in widths:
            found = catalog.find(cat, width=w, opening=opening, combination=combination, height=height, depth=depth)
            if found:
                out.append(found[0])
        return out[:limit] if limit else out

    return pick


def build_shortlist(catalog: Catalog, max_items: int = config.AI_SHORTLIST_MAX) -> dict[str, list[CatalogItem]]:
    """v1: ~35-45 representative SKUs grouped by design slot."""
    pick = _picker(catalog)

    groups = {
        "sink": pick((CAT_SINK,), [1200, 1050, 900, 800], opening="Both", height=700),
        "hob": (
            pick((CAT_HOB,), [900, 750], combination="Triple Drawer - A", height=700)
            + pick((CAT_HOB,), [900], combination="Hob Double Drawer", height=700)
            + pick((CAT_HOB,), [900], combination="Hob Double Shutter", height=700)
        ),
        "spice": pick((CAT_SPICE,), [300, 150], height=700),
        "cylinder": pick((CAT_CYLINDER,), [600], opening="Right", height=800),
        # Standard / feature sizes first: small models favour the first enum options.
        "base": (
            pick((CAT_BASE,), [600, 800], combination="Four Drawer", height=700)
            + pick((CAT_BASE,), [900], combination="Double Drawer", height=700)
            + pick((CAT_GRAIN,), [600], height=700)
            + pick((CAT_WICKER,), [600], height=700)
            + pick((CAT_BASE,), [600, 700, 750, 800, 900, 1000, 1200], combination="Double Shutter", height=700)
            + pick((CAT_BASE,), [300, 400, 450, 500], opening="Left", combination="Single Shutter", height=700)
        ),
        "wall": (
            pick((CAT_WALL,), [600, 700, 750, 800, 900, 1000], combination="Double Shutter", height=600, depth=330)
            + pick((CAT_WALL,), [300, 400, 450, 500, 550], opening="Left", height=600, depth=330)
        ),
        "tall": (
            pick((CAT_TALL,), [600], opening="Left", combination="Single Shutter", height=1940)
            + pick((CAT_TALL,), [600], combination="Double Shutter", height=1940)
            + pick((CAT_OVEN,), [600], opening="Left", height=1940, limit=1)
        ),
    }
    # Trim the largest groups if we exceed the budget.
    total = sum(len(v) for v in groups.values())
    for key in ("base", "wall"):
        while total > max_items and len(groups[key]) > 6:
            groups[key].pop(len(groups[key]) // 2)
            total -= 1
    missing = [k for k in ("sink", "hob", "base", "wall") if not groups[k]]
    if missing:
        raise SolverError(f"catalogue shortlist is missing required groups: {missing}")
    return groups


def shortlist_items(groups: dict[str, list[CatalogItem]]) -> list[CatalogItem]:
    seen, out = set(), []
    for items in groups.values():
        for it in items:
            if it.aid not in seen:
                seen.add(it.aid)
                out.append(it)
    return out


def build_shortlist_v2(catalog: Catalog) -> dict[str, list[CatalogItem]]:
    """v2 menu: designer-default option first in every group (small models favour
    the first options), only *feature* base units (the solver fills plain storage),
    plus the purifier, glass display and built-in microwave / oven-micro units."""
    pick = _picker(catalog)

    def first_of(cat, width, **kw):
        # prefer a left-hinged single shutter; the hinge pass flips it where needed
        return pick(cat, [width], opening="Left", **kw) or pick(cat, [width], **kw)

    groups = {
        "sink": pick((CAT_SINK,), [900, 1050, 1200, 800], opening="Both", height=700),
        "hob": (
            pick((CAT_HOB,), [900], combination="Triple Drawer - A", height=700)
            + pick((CAT_HOB,), [900], combination="Hob Double Drawer", height=700)
            + pick((CAT_HOB,), [750], combination="Triple Drawer - A", height=700)
            + pick((CAT_HOB,), [900], combination="Hob Double Shutter", height=700)
        ),
        "spice": pick((CAT_SPICE,), [300, 150], height=700),
        "cylinder": pick((CAT_CYLINDER,), [600], opening="Right", height=800),
        "tall": (
            pick((CAT_TALL,), [600], opening="Left", combination="Single Shutter", height=1940)
            + pick((CAT_TALL,), [800, 900], combination="Double Shutter", height=1940, limit=1)
            + first_of((CAT_OVEN,), 600, height=1940)[:1]
            + first_of((CAT_MICRO,), 600, height=1940)[:1]
            + first_of((CAT_OVEN_MICRO,), 600, height=1940)[:1]
            + first_of((CAT_TALL_WICKER,), 600, height=1940)[:1]
        ),
        "base": (
            pick((CAT_BASE,), [600, 800], combination="Four Drawer", height=700)
            + pick((CAT_BASE,), [900], combination="Double Drawer", height=700)
            + pick((CAT_BASE,), [800], combination="Drawer with Double Shutter", height=700)
            + pick((CAT_GRAIN,), [600, 450], height=700)
            + pick((CAT_WICKER,), [600, 450], height=700)
        ),
        "wall": (
            pick((CAT_WALL,), [600, 700, 800, 900, 1000], combination="Double Shutter", height=600, depth=330)
            + pick((CAT_WALL,), [450], opening="Left", height=600, depth=330)
        ),
        "purifier": [x for w in (600, 450, 750) for x in first_of((CAT_AQUAGUARD,), w, height=600, depth=330)],
        "display": pick(WALL_GLASS_CATS, [600, 800, 900], combination=None, height=600, depth=330),
    }
    missing = [k for k in ("sink", "hob", "base", "wall") if not groups[k]]
    if missing:
        raise SolverError(f"catalogue shortlist is missing required groups: {missing}")
    return groups


# Short, readable option codes for the v2 prompt ---------------------------------

V2_GROUP_ORDER = ("sink", "hob", "spice", "cylinder", "tall", "base", "wall", "purifier", "display")
V2_GROUP_TITLES = {
    "sink": "SINK UNIT - one, centred under the window",
    "hob": "HOB UNIT - one, on a solid wall; chimney above",
    "spice": "SPICE PULL-OUT - right beside the hob (or NONE)",
    "cylinder": "LPG CYLINDER UNIT - beside the hob (or NONE)",
    "tall": "TALL UNITS - end of a run, near the entrance",
    "base": "BASE FEATURE UNITS - 1-3 per wall with a purpose; the solver fills the rest",
    "wall": "WALL CABINET SIZES - preferred sizes per wall",
    "purifier": "WATER PURIFIER UNIT - wall cabinet beside the sink window (or NONE)",
    "display": "GLASS DISPLAY WALL UNIT - one, at the end of a run (or NONE)",
}


def option_code(group: str, it: CatalogItem) -> str:
    c, w = canon(it.cabinet_combination), it.width
    if group == "sink":
        return f"SINK-{w}"
    if group == "hob":
        return f"HOB-{w}-" + ("3DRAWER" if "triple" in c else "2DRAWER" if "drawer" in c else "SHUTTER")
    if group == "spice":
        return f"SPICE-{w}"
    if group == "cylinder":
        return f"LPG-{w}"
    if group == "base":
        if it.cat == CAT_GRAIN:
            return f"GRAIN-{w}"
        if it.cat == CAT_WICKER:
            return f"WICKER-{w}"
        if c == "four drawer":
            return f"DRAWERS4-{w}"
        if c == "double drawer":
            return f"DRAWERS2-{w}"
        if "drawer with" in c:
            return f"DRAWER-SHUTTER-{w}"
        return f"SHUTTER-{w}"
    if group == "wall":
        return f"WALL-{w}" + ("" if it.opening_direction == "Both" else "-1DOOR")
    if group == "purifier":
        return f"AQUAGUARD-{w}"
    if group == "display":
        return f"GLASS-{w}"
    if group == "tall":
        tag = {CAT_OVEN: "OVEN", CAT_MICRO: "MICROWAVE", CAT_OVEN_MICRO: "OVEN-MICRO", CAT_TALL_WICKER: "TALL-WICKER"}.get(it.cat)
        return f"{tag}-{w}" if tag else f"PANTRY-{w}" + ("-2DOOR" if it.opening_direction == "Both" else "")
    return f"{group.upper()}-{w}"


def build_menu(groups: dict[str, list[CatalogItem]]) -> dict[str, list[tuple[str, CatalogItem]]]:
    """group -> [(code, item)] with codes unique across the whole menu."""
    seen: set[str] = set()
    menu = {}
    for g in V2_GROUP_ORDER:
        opts = []
        for it in groups.get(g, []):
            code = base = option_code(g, it)
            k = 2
            while code in seen:
                code, k = f"{base}-{k}", k + 1
            seen.add(code)
            opts.append((code, it))
        menu[g] = opts
    return menu


def wall_labels(geometry: RoomGeometry) -> dict[str, str]:
    """label -> wall id: the scene's wall names when unique ("Wall 1"), else W1, W2, ..."""
    names = [(geometry.walls[w].name or "").strip() for w in geometry.wall_order]
    if all(names) and len(set(names)) == len(names) and all(len(n) <= 24 for n in names):
        return dict(zip(names, geometry.wall_order))
    return {f"W{i + 1}": w for i, w in enumerate(geometry.wall_order)}


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

def _lit(values: list[str]):
    return Literal.__getitem__(tuple(dict.fromkeys(values)))


def build_design_model(groups: dict[str, list[CatalogItem]], wall_ids: list[str], require_tall: bool = False) -> type[BaseModel]:
    """Pydantic model whose enums only admit shortlist AIDs and real wall ids."""
    Wall = _lit(wall_ids)
    Sink = _lit([i.aid for i in groups["sink"]])
    Hob = _lit([i.aid for i in groups["hob"]])
    Spice = _lit([i.aid for i in groups["spice"]] or ["NONE"])
    Base = _lit([i.aid for i in groups["base"]])
    WallUnit = _lit([i.aid for i in groups["wall"]])
    Tall = _lit([i.aid for i in groups["tall"]] or ["NONE"])
    Cyl = _lit([i.aid for i in groups["cylinder"]] or ["NONE"])

    BaseRun = create_model(
        "BaseRun",
        wall_id=(Wall, Field(description="wall that carries this base run")),
        aids=(list[Base], Field(max_length=3, description="feature base units wanted on this wall")),
    )
    WallRun = create_model(
        "WallRun",
        wall_id=(Wall, ...),
        aids=(list[WallUnit], Field(max_length=2, description="preferred wall-cabinet sizes on this wall")),
    )
    TallChoice = create_model("TallChoice", wall_id=(Wall, ...), aid=(Tall, ...))
    # Field order matters: the model writes its reading of the room and its plan
    # *before* committing to any decision (cheap chain-of-thought inside JSON).
    return create_model(
        "KitchenDesign",
        room_reading=(str, Field(max_length=300, description="which wall has the window, the door, which are solid")),
        design_plan=(str, Field(max_length=300, description="work-triangle plan: sink wall, hob wall, tall storage, why")),
        layout_type=(Literal["single", "L", "U"], ...),
        cabinet_walls=(list[Wall], Field(min_length=1, max_length=3)),
        sink_wall_id=(Wall, ...),
        sink_aid=(Sink, ...),
        hob_wall_id=(Wall, ...),
        hob_aid=(Hob, ...),
        hob_center_mm=(int, Field(ge=0, le=20000, description="desired hob centre along its wall, mm from wall start")),
        spice_aid=(Spice, ...),
        cylinder_aid=(Optional[Cyl], None),
        # Small models skip optional lists even when their plan mentions them.
        tall_units=(list[TallChoice], Field(min_length=1 if require_tall and groups["tall"] else 0, max_length=2)),
        base_runs=(list[BaseRun], Field(max_length=3)),
        wall_runs=(list[WallRun], Field(max_length=3)),
    )


TALL_KIND = {CAT_OVEN: "oven", CAT_MICRO: "microwave", CAT_OVEN_MICRO: "oven_microwave"}
PANTRY_CATS = (CAT_TALL, CAT_TALL_WICKER)
# Tall towers that satisfy each built-in appliance need.
BUILT_IN_CATS = {
    "microwave": (CAT_MICRO, CAT_OVEN_MICRO),
    "oven": (CAT_OVEN, CAT_OVEN_MICRO),
    "oven_microwave": (CAT_OVEN_MICRO,),
}


def allowed_codes(menu, group: str, prefs: Optional["KitchenPreferences"]) -> list[str]:
    """Option codes Qwen may use for ``group`` after applying the customer's stated needs.

    "NONE" is included when leaving the item out is allowed. A stated need removes the
    other choices from the schema, so the model cannot contradict the customer.
    """
    opts = menu.get(group, [])
    codes = [c for c, _ in opts]
    p = prefs
    if group == "purifier":
        need = p.water_purifier if p else "auto"
        return ["NONE"] if need == "no" or not codes else codes if need == "yes" else codes + ["NONE"]
    if group == "display":
        need = p.display_unit if p else "auto"
        return ["NONE"] if need == "no" or not codes else codes if need == "yes" else codes + ["NONE"]
    if group == "cylinder":
        need = p.include_cylinder if p else None
        return ["NONE"] if need is False or not codes else codes if need else codes + ["NONE"]
    if group == "tall":
        need = p.built_in if p else "auto"
        if need == "auto":
            return codes
        keep = PANTRY_CATS + (BUILT_IN_CATS.get(need, ()))
        return [c for c, it in opts if it.cat in keep] or codes
    if group == "spice":
        return codes + ["NONE"]
    return codes


def build_design_model_v2(
    menu: dict[str, list[tuple[str, CatalogItem]]],
    walls: list[str],
    require_tall: bool = False,
    prefs: Optional["KitchenPreferences"] = None,
) -> type[BaseModel]:
    """v2 schema: enums of option codes and wall names.

    Every key decision is preceded by a short ``*_why`` field: the model states its
    reason first and then commits to the option (explainable, and the reason steers
    the choice). Customer needs narrow the enums (see :func:`allowed_codes`).
    """
    Wall = _lit(walls)

    def codes(group: str):
        return _lit(allowed_codes(menu, group, prefs) or ["NONE"])

    def why(what: str):
        return (str, Field(max_length=110, description=f"one short sentence: the room fact or customer need behind the {what}"))

    Base, WallUnit, Tall = codes("base"), codes("wall"), codes("tall")
    FeatureRun = create_model(
        "FeatureRun",
        wall=(Wall, ...),
        units=(list[Base], Field(max_length=3, description="feature base units for this wall")),
    )
    WallSizes = create_model(
        "WallSizes",
        wall=(Wall, ...),
        units=(list[WallUnit], Field(max_length=2, description="preferred wall-cabinet sizes for this wall")),
    )
    TallChoice = create_model("TallChoiceV2", wall=(Wall, ...), unit=(Tall, ...))
    include_tall = prefs.include_tall if prefs else True
    need_tall = 1 if require_tall and include_tall and menu.get("tall") else 0
    return create_model(
        "KitchenDesignV2",
        room_reading=(str, Field(max_length=240, description="window wall, door wall and its corner, long solid walls")),
        design_plan=(str, Field(max_length=200, description="zones and work triangle in one or two sentences")),
        layout_why=why("choice of cabinet walls"),
        cabinet_walls=(list[Wall], Field(min_length=1, max_length=3)),
        sink_why=why("sink unit and its wall"),
        sink_wall=(Wall, ...),
        sink=(codes("sink"), ...),
        hob_why=why("hob unit, its wall and position"),
        hob_wall=(Wall, ...),
        hob=(codes("hob"), ...),
        hob_center_mm=(int, Field(ge=0, le=20000, description="hob centre, mm from the hob wall's start")),
        spice=(codes("spice"), ...),
        lpg_why=why("LPG cylinder unit decision"),
        lpg_cylinder=(codes("cylinder"), ...),
        tall_why=why("tall units"),
        tall_units=(list[TallChoice], Field(min_length=need_tall, max_length=2 if include_tall else 0)),
        purifier_why=why("water purifier decision"),
        water_purifier=(codes("purifier"), ...),
        display_why=why("glass display decision"),
        display_unit=(codes("display"), ...),
        features_why=why("feature base units"),
        base_feature_units=(list[FeatureRun], Field(max_length=3)),
        wall_cabinet_sizes=(list[WallSizes], Field(max_length=3)),
    )


SYSTEM_PROMPT = """You are a senior MagickHome modular-kitchen interior designer.
Think like a designer: first read the room, then plan the work triangle (sink - hob - tall storage), then pick units.
Choose ONLY catalogue AIDs and wall ids given by the user. A solver snaps exact positions afterwards.
Design rules:
- The sink goes on the WINDOW wall, centred under the window (light + view while washing).
- The hob goes on a SOLID wall (no window) with a long free run, on a different wall from the sink when possible,
  so the chimney can be ducted and the cook is not in the door's path.
- A 300 mm spice pull-out sits right beside the hob; an LPG cylinder unit may sit on its other side.
- Tall units / built-in ovens go at the FAR END of a run, never in front of a window or blocking a door.
- layout_type must match cabinet_walls: single = 1 wall, L = window wall + 1 adjacent wall, U = window wall + both adjacent walls.
- Corners take ONLY corner cabinets (blind corner units, Wall L Corner, Loft L Corner); the solver adds them
  automatically - never list them and never expect a regular unit in a corner.
- base_runs: only a few FEATURE units you really want (wide drawers near the hob, grain trolley or wicker basket near the sink).
  The solver fills the remaining space. Do NOT copy the catalogue list.
Reply with JSON only."""


def room_brief(geometry: RoomGeometry) -> list[str]:
    """Plain-language facts about the room, so a small model need not do geometry."""
    lines = []
    names = {wid: geometry.walls[wid].name for wid in geometry.wall_order}
    for wid in geometry.wall_order:
        w = geometry.walls[wid]
        free = _free_len(geometry, wid)
        facts = [f"{w.name} (id {wid}): {w.length_mm} mm long"]
        facts += [f"WINDOW at {o.start_mm:.0f}-{o.end_mm:.0f} mm (centre {o.center_mm:.0f})" for o in w.windows]
        facts += [f"DOOR at {o.start_mm:.0f}-{o.end_mm:.0f} mm" for o in w.doors]
        facts.append(f"longest free cabinet run {free:.0f} mm")
        if w.windows and free >= config.MIN_CABINET_WALL_MM:
            hint = "window wall -> SINK goes here"
        elif free >= 2400 and not w.doors:
            hint = "solid wall, long run -> ideal for HOB + chimney or tall storage"
        elif free >= 2400:
            hint = "usable for hob or storage, keep the door clear"
        elif free >= config.MIN_CABINET_WALL_MM:
            hint = "short run -> storage only"
        else:
            hint = "too short for cabinets"
        lines.append(f"- {'; '.join(facts)}. {hint}.")
    window_walls = [wid for wid in geometry.wall_order if geometry.walls[wid].windows]
    for ww in window_walls:
        adj = [c.other(ww)[1] for c in geometry.corners_of(ww)]
        if adj:
            lines.append(f"- Walls adjacent to {names[ww]} (for L or U shapes): {', '.join(names[a] for a in adj)}.")
    lines.append("- Corners: " + "; ".join(f"{names[c.wall_a]} meets {names[c.wall_b]}" for c in geometry.corners) + ".")
    h = min(w.height_m for w in geometry.walls.values())
    lines.append(f"- Wall height {h * 1000:.0f} mm (lofts {'possible' if h >= config.LOFT_TOP_M else 'not possible'}).")
    return lines


def _user_prompt(geometry: RoomGeometry, groups, prefs: KitchenPreferences) -> str:
    lines = ["ROOM FACTS (mm measured from each wall's start):", *room_brief(geometry), "", "CATALOGUE (AID | category | WxDxH | combination | opening):"]
    for key in ("sink", "hob", "spice", "cylinder", "base", "wall", "tall"):
        for it in groups.get(key, []):
            lines.append(f"[{key}] {it.short()}")
    lines += [
        "",
        f"CUSTOMER: door style {prefs.door_style}; layout preference {prefs.layout}; "
        f"{'wants' if prefs.include_tall else 'no'} tall storage; "
        f"{'' if prefs.include_cylinder is not False else 'no '}LPG cylinder unit.",
    ]
    if prefs.cabinet_walls:
        names = [geometry.walls[w].name for w in prefs.cabinet_walls if w in geometry.walls]
        lines.append(f"Use exactly these cabinet walls: {names} {prefs.cabinet_walls}")
    if prefs.style_notes:
        lines.append(f"Brief: {prefs.style_notes}")
    lines += [
        "",
        "Fill the JSON fields in order:",
        "1. room_reading - which wall has the window, which has the door, which walls are solid and long.",
        "2. design_plan - where the sink, hob and tall storage go and why (work triangle).",
        "3. the decisions: layout_type, cabinet_walls, sink, hob (hob_center_mm = mm from that wall's start),",
        "   spice, cylinder (or null), tall_units, base_runs (max 3 feature units per wall), wall_runs (max 2 sizes per wall).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v2 prompt: knowledge pack + readable room facts
# ---------------------------------------------------------------------------

def _hob_range(segs: list[tuple[float, float]]) -> Optional[tuple[int, int]]:
    """Comfortable hob-centre range on the longest free run: room for a corner
    unit or landing + spice pull-out on each side (about 1050 mm)."""
    if not segs:
        return None
    s, e = max(segs, key=lambda t: t[1] - t[0])
    if e - s < 1800:
        return None
    m = min(1050.0, (e - s) / 2 - 150)
    return int(round((s + m) / 50) * 50), int(round((e - m) / 50) * 50)


def room_brief_v2(geometry: RoomGeometry, labels: dict[str, str]) -> list[str]:
    """Facts a designer reads off the plan: wall order, corners, openings, free runs, zones."""
    name = {wid: label for label, wid in labels.items()}
    lines = []
    for wid in geometry.wall_order:
        w = geometry.walls[wid]
        a, b = name.get(w.neighbor_start), name.get(w.neighbor_end)
        run = (f"from the {a} corner" if a else "from a free end") + (f" to the {b} corner" if b else " to a free end")
        facts = [f"{name[wid]}: {w.length_mm} mm, {run}."]
        if not w.openings:
            facts.append("Solid.")
        for o in sorted(w.windows, key=lambda o: o.start_mm):
            facts.append(f"WINDOW {o.start_mm:.0f}-{o.end_mm:.0f} (centre {o.center_mm:.0f}).")
        for o in sorted(w.doors, key=lambda o: o.start_mm):
            near = a if o.center_mm < w.length_mm / 2 else b
            facts.append(f"DOOR {o.start_mm:.0f}-{o.end_mm:.0f}" + (f" (near the {near} corner)." if near else "."))
        segs = [(s, e) for s, e in _free_segments(geometry, wid) if e - s >= 300]
        free = max((e - s for s, e in segs), default=0.0)
        if segs:
            facts.append("Free run " + ", ".join(f"{s:.0f}-{e:.0f} ({e - s:.0f} mm)" for s, e in segs) + ".")
        else:
            facts.append("No free run.")
        if free < config.MIN_CABINET_WALL_MM:
            facts.append("Too short for cabinets.")
        elif w.windows:
            win = max(w.windows, key=lambda o: o.width_mm)
            facts.append(f"Wet zone: sink centred at {win.center_mm:.0f}.")
        else:
            rng = _hob_range(segs)
            facts.append(f"Hob centre range {rng[0]}-{rng[1]}." if rng else "Too short for the hob: storage only.")
        lines.append("- " + " ".join(facts))
    for wid in geometry.wall_order:
        w = geometry.walls[wid]
        adj = [c.other(wid)[1] for c in geometry.corners_of(wid)]
        if w.windows and len(adj) == 2:
            lo, hi = w.usable
            aisle = (hi - lo) - 2 * config.CORNER_DEPTH_MM
            lines.append(
                f"- U-shape on {name[wid]}: aisle between the {name[adj[0]]} and {name[adj[1]]} runs "
                f"would be {aisle:.0f} mm (needs {config.MIN_AISLE_MM}+)."
            )
    h = min(w.height_m for w in geometry.walls.values())
    lines.append(f"- Wall height {h * 1000:.0f} mm.")
    return lines


def _menu_lines(menu: dict[str, list[tuple[str, CatalogItem]]]) -> list[str]:
    lines = ["CATALOGUE MENU - choose by CODE (code | width | type | designer note):"]
    for g in V2_GROUP_ORDER:
        if not menu.get(g):
            continue
        lines.append(f"{V2_GROUP_TITLES[g]}:")
        for code, it in menu[g]:
            note = design_knowledge.note_for(it)
            kind = it.cabinet_combination or it.category
            lines.append(f"  {code} | {it.width} mm | {kind}" + (f" | {note}" if note else ""))
    return lines


def customer_needs(prefs: KitchenPreferences) -> list[str]:
    """The household's needs in plain words (shown to Qwen and in the report)."""
    p = prefs
    pur = {"yes": "NEEDS a wall water-purifier (Aquaguard) cabinet", "no": "does NOT want a water-purifier cabinet"}.get(
        p.water_purifier, "water purifier not stated")
    gas = "uses an LPG cylinder - needs a cylinder unit" if p.include_cylinder else (
        "has piped gas - no cylinder unit" if p.include_cylinder is False else "gas supply not stated")
    tower = {
        "none": "no built-in oven or microwave", "microwave": "NEEDS a built-in microwave tower",
        "oven": "NEEDS a built-in oven tower", "oven_microwave": "NEEDS a built-in oven + microwave tower",
    }.get(p.built_in, "built-in appliances not stated")
    disp = {"yes": "WANTS one glass display cabinet", "no": "does NOT want glass display cabinets"}.get(
        p.display_unit, "glass display not stated")
    cook = {"heavy": "heavy daily cooking (many vessels, pressure cookers)", "light": "light cooking"}.get(
        p.cooking, "cooking style not stated")
    size = f"{p.household_size} people in the home" if p.household_size else "household size not stated"
    return [
        f"- Door style {p.door_style}; layout preference {p.layout}; {'wants' if p.include_tall else 'NO'} tall storage.",
        f"- Water purifier: {pur}.",
        f"- Gas: {gas}.",
        f"- Built-in appliances: {tower}.",
        f"- Glass display: {disp}.",
        f"- Cooking: {cook}; {size}.",
    ]


def _user_prompt_v2(geometry: RoomGeometry, menu, labels: dict[str, str], prefs: KitchenPreferences) -> str:
    # The menu comes first: it is identical for every room of one door style, so
    # Ollama can reuse the cached prompt prefix between requests.
    lines = _menu_lines(menu)
    lines += ["", "ROOM (mm measured along each wall from its start; walls listed in order around the room):"]
    lines += room_brief_v2(geometry, labels)
    lines += ["", "CUSTOMER NEEDS (stated needs are fixed; 'not stated' means you decide and say why):"]
    lines += customer_needs(prefs)
    if prefs.cabinet_walls:
        inv = {wid: label for label, wid in labels.items()}
        names = [inv[w] for w in prefs.cabinet_walls if w in inv]
        lines.append(f"- Use exactly these cabinet walls: {', '.join(names)}.")
    if prefs.style_notes:
        lines.append(f"- Brief from the customer: {prefs.style_notes}")
    lines += [
        "",
        "TASK - fill the JSON fields in order:",
        "1. room_reading: the window wall, the door wall and which corner the door is near, the long solid walls.",
        "2. design_plan: the zones and the work triangle in one or two sentences.",
        "3. the decisions, using ONLY codes and wall names from this message. Before each decision write its",
        "   *_why: one short sentence naming the room fact or customer need that decides it.",
        "   hob_center_mm is measured from the hob wall's start and should sit inside that wall's hob centre range.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _client():
    import ollama

    return ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_S)


def _model_names(client) -> set[str]:
    resp = client.list()
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    names = set()
    for m in models or []:
        name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
        if name:
            names.add(name)
    return names


_PULL_LOCK = threading.Lock()


def ensure_model(client=None, auto_pull: bool = config.OLLAMA_AUTO_PULL, wait: bool = True) -> None:
    """Raise :class:`AIUnavailable` unless the Ollama model is ready (pull if missing).

    ``wait=False`` fails fast while another thread is downloading the model, so
    API requests in ``auto`` mode fall back instead of blocking on a multi-GB pull.
    """
    client = client or _client()
    try:
        names = _model_names(client)
    except Exception as exc:
        raise AIUnavailable(f"Ollama not reachable at {config.OLLAMA_HOST}: {exc}") from exc
    want = config.OLLAMA_MODEL
    if want in names or (":" not in want and f"{want}:latest" in names):
        return
    if not auto_pull:
        raise AIUnavailable(f"model {want} not installed (run: ollama pull {want})")
    if not _PULL_LOCK.acquire(blocking=wait):
        raise AIUnavailable(f"model {want} is still downloading")
    try:
        if want in _model_names(client):  # another thread finished the pull
            return
        log.info("pulling %s (one-time download, several GB)...", want)
        last = -1
        for prog in client.pull(want, stream=True):
            total = getattr(prog, "total", None) or 0
            done = getattr(prog, "completed", None) or 0
            pct = int(done * 100 / total) if total else -1
            if pct >= 0 and pct // 10 != last // 10:
                log.info("  %s %d%%", getattr(prog, "status", ""), pct)
                last = pct
    except AIUnavailable:
        raise
    except Exception as exc:
        raise AIUnavailable(f"failed to pull {want}: {exc}") from exc
    finally:
        _PULL_LOCK.release()


def _secs(resp, key: str) -> Optional[float]:
    ns = getattr(resp, key, None)
    return round(ns / 1e9, 1) if isinstance(ns, (int, float)) else None


def build_messages(geometry: RoomGeometry, groups, prefs: KitchenPreferences, version: str):
    """``(messages, Model, to_proposal)`` for one prompt version."""
    if version == "v1":
        Model = build_design_model(groups, geometry.wall_order, require_tall=prefs.include_tall)
        system, user = SYSTEM_PROMPT, _user_prompt(geometry, groups, prefs)

        def to_proposal(d):
            return _design_to_proposal(d, prefs)
    else:
        menu, labels = build_menu(groups), wall_labels(geometry)
        Model = build_design_model_v2(menu, list(labels), require_tall=prefs.include_tall, prefs=prefs)
        system, user = design_knowledge.system_prompt(), _user_prompt_v2(geometry, menu, labels, prefs)

        def to_proposal(d):
            return _design_to_proposal_v2(d, menu, labels, prefs)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return messages, Model, to_proposal


def design_with_ollama(
    geometry: RoomGeometry, groups, prefs: KitchenPreferences, client=None, version: Optional[str] = None
) -> tuple[Proposal, dict]:
    version = version or config.AI_PROMPT_VERSION
    client = client or _client()
    ensure_model(client, wait=prefs.ai_mode == "ollama")
    messages, Model, to_proposal = build_messages(geometry, groups, prefs, version)
    schema = Model.model_json_schema()
    errors = []
    for attempt in range(config.AI_MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat(
                model=config.OLLAMA_MODEL,
                messages=messages,
                format=schema,
                options=dict(config.OLLAMA_OPTIONS),
                keep_alive=config.OLLAMA_KEEP_ALIVE,
            )
        except Exception as exc:
            raise AIUnavailable(f"Ollama chat failed: {exc}") from exc
        content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
        try:
            design = Model.model_validate_json(content)
        except ValidationError as exc:
            errors.append(str(exc)[:400])
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Invalid JSON for the schema: {exc.errors()[:3]}. Reply again."})
            continue
        info = {
            "mode": "ollama",
            "model": config.OLLAMA_MODEL,
            "prompt_version": version,
            "attempts": attempt + 1,
            "seconds": round(time.time() - t0, 1),
            "prompt_tokens": getattr(resp, "prompt_eval_count", None),
            "prompt_seconds": _secs(resp, "prompt_eval_duration"),
            "eval_count": getattr(resp, "eval_count", None),
            "eval_seconds": _secs(resp, "eval_duration"),
            "load_seconds": _secs(resp, "load_duration"),
            "raw": json.loads(design.model_dump_json()),
        }
        return to_proposal(design), info
    raise AIUnavailable(f"model reply failed schema validation: {errors[-1] if errors else 'unknown'}")


def _design_to_proposal(d, prefs: KitchenPreferences) -> Proposal:
    base_prefs: dict[str, list[str]] = {}
    for run in d.base_runs:
        base_prefs.setdefault(run.wall_id, []).extend(run.aids)
    wall_prefs: dict[str, list[str]] = {}
    for run in d.wall_runs:
        wall_prefs.setdefault(run.wall_id, []).extend(run.aids)
    talls = [(t.wall_id, t.aid) for t in d.tall_units if t.aid != "NONE"] if prefs.include_tall else []
    cyl = d.cylinder_aid if d.cylinder_aid not in (None, "NONE") and prefs.include_cylinder is not False else None
    return Proposal(
        # The label is derived from the walls actually chosen (small models mislabel it).
        layout_type={1: "single", 2: "L"}.get(len(set(d.cabinet_walls)), "U"),
        cabinet_walls=list(dict.fromkeys(prefs.cabinet_walls or d.cabinet_walls)),
        sink_wall_id=d.sink_wall_id,
        sink_aid=d.sink_aid,
        hob_wall_id=d.hob_wall_id,
        hob_aid=d.hob_aid,
        hob_center_mm=float(d.hob_center_mm),
        spice_aid=None if d.spice_aid == "NONE" else d.spice_aid,
        cylinder_aid=cyl,
        tall_units=talls,
        base_prefs=base_prefs,
        wall_prefs=wall_prefs,
        notes=f"{d.room_reading.strip()} {d.design_plan.strip()}".strip(),
        source="ollama",
    )


def _design_to_proposal_v2(d, menu, labels: dict[str, str], prefs: KitchenPreferences) -> Proposal:
    code = {c: it.aid for opts in menu.values() for c, it in opts}

    def aid(c):
        return None if c in (None, "NONE") else code[c]

    base_prefs: dict[str, list[str]] = {}
    for run in d.base_feature_units:
        base_prefs.setdefault(labels[run.wall], []).extend(code[u] for u in run.units)
    wall_prefs: dict[str, list[str]] = {}
    for run in d.wall_cabinet_sizes:
        wall_prefs.setdefault(labels[run.wall], []).extend(code[u] for u in run.units)
    walls = list(dict.fromkeys(labels[w] for w in d.cabinet_walls))
    return Proposal(
        layout_type={1: "single", 2: "L"}.get(len(walls), "U"),
        cabinet_walls=list(dict.fromkeys(prefs.cabinet_walls or walls)),
        sink_wall_id=labels[d.sink_wall],
        sink_aid=code[d.sink],
        hob_wall_id=labels[d.hob_wall],
        hob_aid=code[d.hob],
        hob_center_mm=float(d.hob_center_mm),
        spice_aid=aid(d.spice),
        cylinder_aid=aid(d.lpg_cylinder) if prefs.include_cylinder is not False else None,
        tall_units=[(labels[t.wall], code[t.unit]) for t in d.tall_units] if prefs.include_tall else [],
        base_prefs=base_prefs,
        wall_prefs=wall_prefs,
        purifier_aid=aid(d.water_purifier),
        display_aid=aid(d.display_unit),
        notes=f"{d.room_reading.strip()} {d.design_plan.strip()}".strip(),
        source="ollama",
    )


# ---------------------------------------------------------------------------
# Deterministic fallback designer
# ---------------------------------------------------------------------------

def _free_segments(geometry: RoomGeometry, wid: str) -> list[tuple[float, float]]:
    """Usable spans of a wall's base zone after door clearances."""
    wall = geometry.walls[wid]
    lo, hi = wall.usable
    cuts = sorted(d.interval(config.DOOR_CLEARANCE_MM) for d in wall.doors)
    segs, cur = [], lo
    for a, b in cuts:
        if min(a, hi) > cur:
            segs.append((cur, min(a, hi)))
        cur = max(cur, b)
    if hi > cur:
        segs.append((cur, hi))
    return segs


def _free_len(geometry: RoomGeometry, wid: str) -> float:
    return max((b - a for a, b in _free_segments(geometry, wid)), default=0.0)


def fallback_proposal(geometry: RoomGeometry, groups, prefs: KitchenPreferences) -> Proposal:
    """Rule-of-thumb design used for automated tests and when Ollama is unavailable."""
    walls = list(prefs.cabinet_walls or suggest_cabinet_walls(geometry, prefs.layout))
    window_walls = [w for w in walls if geometry.walls[w].windows]
    sink_wall = max(window_walls, key=lambda w: max(o.width_mm for o in geometry.walls[w].windows)) if window_walls else max(
        walls, key=lambda w: _free_len(geometry, w)
    )
    others = [w for w in walls if w != sink_wall]
    hob_wall = max(others, key=lambda w: _free_len(geometry, w)) if others else sink_wall
    rest = [w for w in walls if w not in (sink_wall, hob_wall)]
    tall_wall = rest[0] if rest else (sink_wall if sink_wall != hob_wall else hob_wall)

    by = {k: {it.aid: it for it in v} for k, v in groups.items()}
    sink_free = _free_len(geometry, sink_wall)
    # playbook sizes: 800 small, 900 standard, 1050 from 3000 mm free, 1200 from 3600 mm free
    want = 800 if sink_free < 2400 else 900 if sink_free < 3000 else 1050 if sink_free < 3600 else 1200
    sinks = sorted(groups["sink"], key=lambda it: (abs(it.width - want), -it.width))
    sink = sinks[0]
    hob = groups["hob"][0]
    spice = groups["spice"][0] if groups["spice"] else None
    cyl = None
    if prefs.include_cylinder or (prefs.include_cylinder is None and _free_len(geometry, hob_wall) >= 2700):
        cyl = groups["cylinder"][0].aid if groups["cylinder"] else None
    talls = []
    if prefs.include_tall and groups["tall"]:
        tall_free = _free_len(geometry, tall_wall)
        if tall_free >= 2400 or len(walls) > 1:
            talls.append((tall_wall, groups["tall"][0].aid))

    def first(group, pred):
        return next((it.aid for it in groups[group] if pred(it)), None)

    drawer = first("base", lambda i: canon(i.cabinet_combination) == "four drawer" and i.width == 600)
    trolley = first("base", lambda i: i.cat == CAT_GRAIN)
    base_prefs = {}
    if drawer:
        base_prefs[hob_wall] = [drawer]
    if trolley and sink_wall != hob_wall:
        base_prefs[sink_wall] = [trolley]

    layout_type = {1: "single", 2: "L", 3: "U"}.get(len(walls), "U")
    return Proposal(
        layout_type=layout_type,
        cabinet_walls=walls,
        sink_wall_id=sink_wall,
        sink_aid=sink.aid,
        hob_wall_id=hob_wall,
        hob_aid=hob.aid,
        hob_center_mm=None,
        spice_aid=spice.aid if spice else None,
        cylinder_aid=cyl,
        tall_units=talls,
        base_prefs=base_prefs,
        wall_prefs={},
        # water purifier beside the sink window (v2 shortlist only)
        purifier_aid=groups["purifier"][0].aid if groups.get("purifier") and window_walls else None,
        notes=(
            f"{layout_type}-shaped kitchen: sink on {geometry.walls[sink_wall].name}"
            f"{' under the window' if window_walls else ''}, hob on {geometry.walls[hob_wall].name}"
            f"{', tall storage on ' + geometry.walls[tall_wall].name if talls else ''}."
        ),
        source="fallback",
    )


def design_kitchen(
    geometry: RoomGeometry, catalog: Catalog, prefs: KitchenPreferences, version: Optional[str] = None
) -> tuple[Proposal, dict]:
    """Return ``(proposal, ai_info)`` according to ``prefs.ai_mode``."""
    version = version or config.AI_PROMPT_VERSION
    groups = build_shortlist_v2(catalog) if version == "v2" else build_shortlist(catalog)
    n = len(shortlist_items(groups))
    info: dict = {"shortlist_size": n, "prompt_version": version}
    if prefs.ai_mode in ("auto", "ollama"):
        try:
            proposal, ai = design_with_ollama(geometry, groups, prefs, version=version)
            info.update(ai)
            return proposal, info
        except AIUnavailable as exc:
            if prefs.ai_mode == "ollama":
                raise
            log.warning("AI designer unavailable, using deterministic fallback: %s", exc)
            info["fallback_reason"] = str(exc)
    proposal = fallback_proposal(geometry, groups, prefs)
    info["mode"] = "fallback"
    return proposal, info
