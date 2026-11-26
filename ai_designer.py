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
import re
import logging
import threading
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, create_model, model_validator

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
    CAT_TALL_GLASS,
    CAT_TALL_WICKER,
    CAT_WALL,
    CAT_WALL_OPEN,
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
    include_loft: Optional[bool] = Field(
        None, description="Lofts above the wall cabinets when the wall height allows (>= 2.658 m); None = the designer decides"
    )
    include_tall: bool = True
    include_cylinder: Optional[bool] = Field(
        None, description="LPG cylinder unit beside the hob. None = not stated: LPG is assumed and the unit is "
                          "included (rule R11); False = piped gas (PNG) or induction"
    )
    fuel: Literal["auto", "lpg", "png", "induction"] = Field(
        "auto", description="Hob fuel (rule R11): lpg keeps the cylinder unit, png (piped gas) and induction drop it; "
                            "auto = not stated (LPG assumed). Overrides include_cylinder when set"
    )
    size_band: Literal["auto", "small", "medium", "large"] = Field(
        "auto", description="Rule-book size band; auto = from the floor area (Small 60-80, Medium 81-120, Large 121-180 sqft)"
    )
    budget: Literal["auto", "essential", "premium", "luxury"] = Field(
        "auto", description="Budget tier (rule workbook): Essential = shutter base units, Premium / Luxury = drawer "
                            "base units and L-corner wall units. auto = not stated (the designer's base style is used)"
    )
    base_finish: Optional[str] = Field(
        None, description="Shutter finish aid for the base run (base + tall units), e.g. MHS135; None = the designer "
                          "chooses within the budget tier's finish rules (knowledge/finish_rules.json)"
    )
    wall_finish: Optional[str] = Field(None, description="Shutter finish aid for the wall run (wall units + lofts)")
    countertop_finish: Optional[str] = Field(None, description="Worktop finish aid (e.g. MHC015); None = scene default")
    backsplash_finish: Optional[str] = Field(None, description="Backsplash finish aid (e.g. MHB021); None = scene default")
    primary_wall: Optional[str] = Field(None, description="Wall id the customer chose for the hob / cooking (rule R24)")
    secondary_wall: Optional[str] = Field(None, description="Wall id the customer chose for the sink (rule R25)")
    sink_under_window: bool = True
    style_notes: str = Field("", max_length=500, description="Free-text brief passed to the AI designer")
    # Household needs. "auto" = not stated: Qwen decides and explains why. Any other
    # value is the customer's requirement and is enforced (Qwen cannot override it).
    water_purifier: Literal["auto", "yes", "no"] = Field("auto", description="Wall purifier (Aquaguard) cabinet")
    built_in: Literal["auto", "none", "microwave", "oven", "oven_microwave"] = Field(
        "auto", description="Built-in appliance tower(s) the household needs"
    )
    display_unit: Literal["auto", "yes", "no"] = Field("auto", description="Glass display wall cabinet")
    base_storage: Literal["auto", "drawers", "mixed", "shutters"] = Field(
        "auto", description="Plain base cabinets: drawers, drawer + shutter mix, or shutters (auto = Qwen decides)"
    )
    wall_style: Literal["auto", "shutters", "lift_up"] = Field(
        "auto", description="Wall cabinet doors: hinged shutters or lift-up (flap-up) doors"
    )
    open_unit: Literal["auto", "yes", "no"] = Field("auto", description="Open wall shelf unit at a run end")
    crockery_unit: Literal["auto", "yes", "no"] = Field("auto", description="Tall glass crockery unit")
    cooking: Literal["auto", "heavy", "light"] = Field("auto", description="Daily heavy cooking or light cooking")
    household_size: Optional[int] = Field(None, ge=1, le=20, description="People living in the home")
    ai_mode: Literal["auto", "ollama", "fallback"] = Field(
        "auto", description="auto = AI designer with deterministic fallback; ollama = require the AI designer"
    )
    ai_provider: Optional[Literal["qwen", "gemma", "gemini", "local"]] = Field(
        None, description="Which AI designer: qwen or gemma (Ollama, local/Colab), gemini (Google API), or local "
                          "(any other installed Ollama model, named in local_model); None = config default"
    )
    local_model: Optional[str] = Field(
        None, max_length=200, description="Ollama model name for ai_provider='local', e.g. qwen3.5:2b or gemma4:e2b"
    )
    strip_existing_items: bool = True
    include_floor: bool = True
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _fuel_sets_cylinder(self):
        """R11: the fuel decides the cylinder unit - LPG keeps it, piped gas / induction drop it."""
        if self.fuel == "lpg":
            self.include_cylinder = True
        elif self.fuel in ("png", "induction"):
            self.include_cylinder = False
        return self


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
            # crockery unit: glass-front tall unit (single 600, double 800/900)
            + first_of((CAT_TALL_GLASS,), 600, height=1940)[:1]
            + pick((CAT_TALL_GLASS,), [900, 800], opening="Both", height=1940, limit=1)
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
        # open shelf unit for daily-use jars (same finish for every door style)
        "open": pick((CAT_WALL_OPEN,), [600, 450], combination="Open Unit", height=600, depth=330)
        + pick((CAT_WALL_OPEN,), [900], combination="Open Unit with 2 Vertical Partitions", height=600, depth=330),
    }
    missing = [k for k in ("sink", "hob", "base", "wall") if not groups[k]]
    if missing:
        raise SolverError(f"catalogue shortlist is missing required groups: {missing}")
    return groups


# Short, readable option codes for the v2 prompt ---------------------------------

V2_GROUP_ORDER = ("sink", "hob", "spice", "cylinder", "tall", "base", "wall", "purifier", "display", "open")
V2_GROUP_TITLES = {
    "sink": "SINK UNIT - one, centred under the window",
    "hob": "HOB UNIT - one, on a solid wall; chimney above",
    "spice": "SPICE PULL-OUT - right beside the hob (or NONE)",
    "cylinder": "LPG CYLINDER UNIT - beside the hob (or NONE)",
    "tall": "TALL UNITS - end of a run, near the entrance (CROCKERY = glass crockery unit)",
    "base": "BASE FEATURE UNITS - 1-3 per wall with a purpose; the solver fills the rest",
    "wall": "WALL CABINET SIZES - preferred sizes per wall",
    "purifier": "WATER PURIFIER UNIT - wall cabinet beside the sink window (or NONE)",
    "display": "GLASS DISPLAY WALL UNIT - one, at the end of a run (or NONE)",
    "open": "OPEN SHELF WALL UNIT - one, at a run end away from the hob (or NONE)",
}
# Neutral titles for a designer given no MagickHome knowledge (Gemini's "own judgement" mode).
V2_GROUP_TITLES_PLAIN = {
    "sink": "SINK UNITS",
    "hob": "HOB UNITS",
    "spice": "SPICE PULL-OUTS",
    "cylinder": "LPG CYLINDER UNITS",
    "tall": "TALL UNITS",
    "base": "BASE UNITS",
    "wall": "WALL CABINETS",
    "purifier": "WATER PURIFIER WALL UNITS",
    "display": "GLASS DISPLAY WALL UNITS",
    "open": "OPEN SHELF WALL UNITS",
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
    if group == "open":
        return f"OPEN-{w}" + ("-PARTITIONS" if "partition" in c else "")
    if group == "tall":
        tag = {CAT_OVEN: "OVEN", CAT_MICRO: "MICROWAVE", CAT_OVEN_MICRO: "OVEN-MICRO", CAT_TALL_WICKER: "TALL-WICKER",
               CAT_TALL_GLASS: "CROCKERY"}.get(it.cat)
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
    if group == "open":
        need = p.open_unit if p else "auto"
        return ["NONE"] if need == "no" or not codes else codes if need == "yes" else codes + ["NONE"]
    if group == "cylinder":
        # R11: LPG is assumed unless the customer has piped gas / induction - no NONE then
        need = p.include_cylinder if p else None
        return ["NONE"] if need is False or not codes else codes
    if group == "tall":
        crockery = p.crockery_unit if p else "auto"
        if crockery == "no":  # customer does not want a crockery unit: not offered
            opts = [(c, it) for c, it in opts if it.cat != CAT_TALL_GLASS] or opts
            codes = [c for c, _ in opts]
        need = p.built_in if p else "auto"
        if need == "auto":
            return codes
        keep = PANTRY_CATS + (BUILT_IN_CATS.get(need, ())) + ((CAT_TALL_GLASS,) if crockery != "no" else ())
        return [c for c, it in opts if it.cat in keep] or codes
    if group == "spice":
        return codes + ["NONE"]
    return codes


def build_design_model_v2(
    menu: dict[str, list[tuple[str, CatalogItem]]],
    walls: list[str],
    require_tall: bool = False,
    prefs: Optional["KitchenPreferences"] = None,
    lofts_possible: bool = True,
    fixed_walls: Optional[dict[str, str]] = None,
) -> type[BaseModel]:
    """v2 schema: enums of option codes and wall names.

    Every key decision is preceded by a short ``*_why`` field: the model states its
    reason first and then commits to the option (explainable, and the reason steers
    the choice). Customer needs narrow the enums (see :func:`allowed_codes`); lofts are
    "no" only when the ceiling is too low for them.
    """
    Wall = _lit(walls)
    fixed = fixed_walls or {}  # R24 / R25: the customer's hob and sink walls
    HobWall = _lit([fixed["hob_wall"]]) if fixed.get("hob_wall") in walls else Wall
    SinkWall = _lit([fixed["sink_wall"]]) if fixed.get("sink_wall") in walls else Wall

    def codes(group: str):
        return _lit(allowed_codes(menu, group, prefs) or ["NONE"])

    def why(what: str):
        return (str, Field(
            max_length=180,
            description=f"one or two short sentences on the {what}: the deciding room fact or customer need, "
                        "and why this option rather than the others",
        ))

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
        sink_wall=(SinkWall, ...),
        sink=(codes("sink"), ...),
        hob_why=why("hob unit, its wall and position"),
        hob_wall=(HobWall, ...),
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
        open_why=why("open shelf unit decision"),
        open_unit=(codes("open"), ...),
        storage_why=why("base cabinet style (drawers / mixed / shutters)"),
        base_storage=(_lit(allowed_storage(prefs)), ...),
        features_why=why("feature base units"),
        base_feature_units=(list[FeatureRun], Field(max_length=3)),
        wall_cabinets_why=why("wall cabinet sizes"),
        wall_cabinet_sizes=(list[WallSizes], Field(max_length=3)),
        wall_style_why=why("wall cabinet doors (hinged shutters / lift-up)"),
        wall_style=(_lit(allowed_wall_styles(prefs)), ...),
        finish_why=why("shutter finishes of the base run and the wall run"),
        base_finish=(_lit(finish_codes("base", prefs)), ...),
        wall_finish=(_lit(finish_codes("wall", prefs)), ...),
        loft_why=why("loft decision"),
        lofts=(_lit(allowed_lofts(prefs, lofts_possible)), ...),
    )


def finish_codes(run: str, prefs: Optional["KitchenPreferences"]) -> list[str]:
    """Finish codes the designer may pick for ``run``: the customer's stated finish, else every
    finish the budget tier allows (knowledge/finish_rules.json)."""
    import finishes

    stated = getattr(prefs, f"{run}_finish", None) if prefs else None
    if stated in finishes.catalogue():
        return [finishes.code(stated)]
    budget = prefs.budget if prefs and prefs.budget != "auto" else None
    # a varied shortlist (the API has 174 shutter finishes - too many for the prompt); the customer
    # can still pick any of them in the UI
    return [finishes.code(a) for a in finishes.ai_shortlist(run, budget)] or ["STANDARD"]


WALL_STYLES = ("shutters", "lift_up")


def allowed_wall_styles(prefs: Optional["KitchenPreferences"]) -> list[str]:
    """Wall-door styles the designer may pick: both unless the customer chose one."""
    stated = prefs.wall_style if prefs else "auto"
    return list(WALL_STYLES) if stated == "auto" else [stated]


def allowed_lofts(prefs: Optional["KitchenPreferences"], possible: bool = True) -> list[str]:
    """``yes`` / ``no``: only ``no`` when the ceiling is too low; the customer's answer otherwise wins."""
    if not possible:
        return ["no"]
    stated = prefs.include_loft if prefs else None
    return ["yes", "no"] if stated is None else ["yes" if stated else "no"]


def lofts_possible(geometry: RoomGeometry) -> bool:
    """A loft needs the ceiling (lowest wall) to reach the loft top (2658 mm)."""
    return min(w.height_m for w in geometry.walls.values()) + 1e-6 >= config.LOFT_TOP_M


STORAGE_STYLES = ("drawers", "mixed", "shutters")


def allowed_storage(prefs: Optional["KitchenPreferences"]) -> list[str]:
    """Base cabinet styles Qwen may pick: all three when the customer left it to the
    designer, else only the style the customer asked for (form or brief)."""
    stated = base_storage_need(prefs)[0] if prefs else "auto"
    return list(STORAGE_STYLES) if stated == "auto" else [stated]


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


def room_brief_v2(geometry: RoomGeometry, labels: dict[str, str], hints: bool = True) -> list[str]:
    """Facts a designer reads off the plan: wall order, corners, openings, free runs, zones.

    ``hints=False`` gives only the measured facts - no zone advice computed from the
    MagickHome rules (wet zone, hob centre range, too short, U-aisle check)."""
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
        if not hints:
            pass
        elif free < config.MIN_CABINET_WALL_MM:
            facts.append("Too short for cabinets.")
        elif w.windows:
            win = max(w.windows, key=lambda o: o.width_mm)
            facts.append(f"Wet zone: sink centred at {win.center_mm:.0f}.")
        else:
            rng = _hob_range(segs)
            facts.append(f"Hob centre range {rng[0]}-{rng[1]}." if rng else "Too short for the hob: storage only.")
        lines.append("- " + " ".join(facts))
    for wid in geometry.wall_order if hints else ():
        w = geometry.walls[wid]
        adj = [c.other(wid)[1] for c in geometry.corners_of(wid)]
        if w.windows and len(adj) == 2:
            lo, hi = w.usable
            aisle = (hi - lo) - 2 * config.CORNER_DEPTH_MM
            lines.append(
                f"- U-shape on {name[wid]}: aisle between the {name[adj[0]]} and {name[adj[1]]} runs "
                f"would be {aisle:.0f} mm (needs {config.MIN_AISLE_MM}+)."
            )
    import rulebook

    area = rulebook.floor_area_sqft(geometry)
    if hints:
        band, note = rulebook.size_band(area)
        lines.append(f"- Floor area {area:.0f} sqft: a {band} kitchen in the rule book"
                     + (f" ({note})" if note else "") + " - module sizes follow its table (playbook RULE BOOK).")
    else:
        lines.append(f"- Floor area {area:.0f} sqft.")
    h = min(w.height_m for w in geometry.walls.values())
    if hints:
        top = config.LOFT_TOP_M * 1000
        lines.append(f"- Wall height {h * 1000:.0f} mm: " + (
            f"lofts possible (they need {top:.0f} mm)." if h * 1000 + 1e-3 >= top else
            f"too low for lofts (they need {top:.0f} mm); wall and tall units grow towards the ceiling instead."))
    else:
        lines.append(f"- Wall height {h * 1000:.0f} mm.")
    return lines


def _finish_lines(knowledge: bool = True, prefs: Optional["KitchenPreferences"] = None) -> list[str]:
    """Shutter finishes on offer (a shortlist of the MagickHome textures API for the budget tier).
    With knowledge: the budget-tier finish rules too."""
    import finishes

    budget = prefs.budget if prefs and prefs.budget != "auto" else None
    offered = list(dict.fromkeys(finishes.ai_shortlist("base", budget) + finishes.ai_shortlist("wall", budget)))
    for run in finishes.RUNS:  # the customer's stated finishes are always listed
        aid = getattr(prefs, f"{run}_finish", None) if prefs else None
        if aid in finishes.catalogue() and aid not in offered:
            offered.append(aid)
    lines = ["SHUTTER FINISHES - choose by CODE (code | type | look | brand | colour):"]
    for aid in offered:
        s = finishes.summary(aid)
        look = f"{(s['shutter'] or '').lower()}" + (", wood texture" if (s["family"] or "").lower() == "wooden" else "")
        lines.append(f"  {s['code']} | {s['category']} | {look} | {s['brand']} | {s['color'] or 'texture'}")
    if knowledge:
        for tier, t in finishes.rules().get("tiers", {}).items():
            lines.append(f"  {tier.title()} budget: base run {'/'.join(t.get('base', []))}, wall run "
                         f"{'/'.join(t.get('wall', []))} - {t.get('note', '')}")
        lines += [f"  {m}" for m in finishes.rules().get("match", [])]
    return lines


def _menu_lines(menu: dict[str, list[tuple[str, CatalogItem]]], knowledge: bool = True) -> list[str]:
    """The cabinet menu. ``knowledge=False``: plain catalogue facts only (code, width x depth,
    catalogue type) under neutral titles - no designer notes or placement advice."""
    if not knowledge:
        lines = ["CABINET LIST - choose by CODE (code | width x depth | catalogue type):"]
        for g in V2_GROUP_ORDER:
            if menu.get(g):
                lines.append(f"{V2_GROUP_TITLES_PLAIN[g]}:")
                lines += [f"  {code} | {it.width} x {it.depth} mm | {it.cabinet_combination or it.category}"
                          for code, it in menu[g]]
        return lines
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


def base_storage_need(prefs: KitchenPreferences) -> tuple[str, str]:
    """The customer's stated base-cabinet style: (``auto``|``drawers``|``mixed``|``shutters``, source).
    ``auto`` means not stated - the designer (Qwen) decides. A brief that asks for drawers
    counts as stated, so "lots of drawers" in the text is honoured even if the form says auto."""
    if prefs.base_storage != "auto":
        return prefs.base_storage, "form"
    if re.search(r"\bdrawers?\b", prefs.style_notes or "", re.I):
        return "drawers", "brief"
    return "auto", ""


def customer_needs(prefs: KitchenPreferences) -> list[str]:
    """The household's needs in plain words (shown to Qwen and in the report)."""
    p = prefs
    pur = {"yes": "NEEDS a wall water-purifier (Aquaguard) cabinet", "no": "does NOT want a water-purifier cabinet"}.get(
        p.water_purifier, "water purifier not stated")
    gas = "uses an LPG cylinder - needs a cylinder unit" if p.include_cylinder else (
        {"png": "has piped gas (PNG) - no cylinder unit", "induction": "cooks on induction - no cylinder unit"}.get(
            p.fuel, "has piped gas (PNG) or induction - no cylinder unit") if p.include_cylinder is False
        else "not stated - LPG is assumed, so the cylinder unit is included (rule R11)")
    tower = {
        "none": "no built-in oven or microwave", "microwave": "NEEDS a built-in microwave tower",
        "oven": "NEEDS a built-in oven tower", "oven_microwave": "NEEDS a built-in oven + microwave tower",
    }.get(p.built_in, "built-in appliances not stated")
    disp = {"yes": "WANTS one glass display cabinet", "no": "does NOT want glass display cabinets"}.get(
        p.display_unit, "glass display not stated")
    cook = {"heavy": "heavy daily cooking (many vessels, pressure cookers)", "light": "light cooking"}.get(
        p.cooking, "cooking style not stated")
    size = f"{p.household_size} people in the home" if p.household_size else "household size not stated"
    store = {"drawers": "wants MORE DRAWERS (base runs are filled with drawer units; ask for drawer feature units)",
             "mixed": "wants a drawer + shutter mix",
             "shutters": "prefers shutter base cabinets"}.get(
        base_storage_need(p)[0], "base cabinet style not stated - you choose drawers, mixed or shutters")
    loft = {True: "WANTS lofts (when the ceiling allows)", False: "does NOT want lofts"}.get(
        p.include_loft, "lofts not stated")
    wall_doors = {"shutters": "wants hinged shutter doors", "lift_up": "wants LIFT-UP (flap-up) doors"}.get(
        p.wall_style, "door type not stated - you choose hinged shutters or lift-up")
    open_unit = {"yes": "WANTS an open shelf unit", "no": "does NOT want open shelves"}.get(
        p.open_unit, "open shelf unit not stated")
    crockery = {"yes": "WANTS a tall glass crockery unit", "no": "does NOT want a crockery unit"}.get(
        p.crockery_unit, "crockery unit not stated")
    return [
        f"- Door style {p.door_style}; layout preference {p.layout}; {'wants' if p.include_tall else 'NO'} tall storage.",
        f"- Water purifier: {pur}.",
        f"- Gas: {gas}.",
        f"- Built-in appliances: {tower}.",
        f"- Glass display: {disp}.",
        f"- Cooking: {cook}; {size}.",
        f"- Base cabinets: {store}.",
        f"- Wall cabinets: {wall_doors}; {open_unit}.",
        f"- Crockery: {crockery}.",
        f"- Lofts: {loft}.",
        f"- Budget: " + ({"essential": "Essential (shutter base units, blind corners, laminate fronts)",
                           "premium": "Premium (drawer base units, L corners, laminate + one PU run)",
                           "luxury": "Luxury (drawer base units, L corners, PU fronts)"}.get(p.budget) or "not stated") + ".",
        "- Finishes: " + "; ".join(
            f"{run} run {_finish_need(val)}" for run, val in (("base", p.base_finish), ("wall", p.wall_finish)))
        + " (the codes allowed are the ones the budget tier permits).",
    ]


def _finish_need(aid: Optional[str]) -> str:
    import finishes

    return f"{finishes.describe(aid)} (customer's choice)" if aid in finishes.catalogue() else "not stated - you choose"


PLAIN_SYSTEM_PROMPT = """You are an experienced modular-kitchen designer working on homes in India for MagickHome.
Design this kitchen with your own professional judgement: read the room, choose which walls carry
cabinets, where the sink and the hob go, and which units this household needs.

How your answer is used: you pick cabinets by CODE from the cabinet list and walls by name, both
exactly as written in the message. A placement engine then sets the exact positions, puts corner
units where two cabinet runs meet, fills the remaining base and wall space with plain cabinets in
your chosen base style, and adds a chimney above the hob. It only corrects choices that cannot be built.

The customer's stated needs are fixed. Where a need is 'not stated', you decide.
Write each *_why before its decision: your reason in one or two short sentences, in plain words.
Answer only with the JSON."""


def _plain_task() -> list[str]:
    """What each answer field means - the answer format only, no design advice."""
    return [
        "",
        "TASK - fill the JSON fields in order, using ONLY codes and wall names from this message:",
        "1. room_reading: what you read from this room.",
        "2. design_plan: your design idea for this kitchen.",
        "3. the decisions, each after its *_why (your reason).",
        "   hob_center_mm = the hob's centre, in mm from the start of the hob wall.",
        "   base_storage = the style of the plain base cabinets that fill the runs.",
        "   base_feature_units = special base units you want on a wall (the plain base cabinets fill the rest).",
        "   wall_cabinet_sizes = wall-cabinet widths you prefer on a wall.",
        "   water_purifier, display_unit, open_unit, spice, lpg_cylinder = whether and which unit; the engine picks",
        "   its wall and position, so do not name a wall for them in your reason.",
        "   wall_style = the doors of the plain wall cabinets: hinged shutters or lift-up (flap-up).",
        "   lofts = storage cabinets above the wall cabinets, up to the ceiling. Without lofts the engine makes",
        "   the wall and tall units as tall as the ceiling allows.",
        "   base_finish / wall_finish = shutter finish of the base run (base + tall units) and of the wall run",
        "   (wall units + lofts); panels and fillers take the same finish.",
        "   NONE = leave that unit out.",
    ]


def _user_prompt_v2(geometry: RoomGeometry, menu, labels: dict[str, str], prefs: KitchenPreferences,
                    knowledge: bool = True) -> str:
    # The menu comes first: it is identical for every room of one door style, so
    # Ollama can reuse the cached prompt prefix between requests.
    lines = _menu_lines(menu, knowledge=knowledge) + _finish_lines(knowledge, prefs)
    lines += ["", "ROOM (mm measured along each wall from its start; walls listed in order around the room):"]
    lines += room_brief_v2(geometry, labels, hints=knowledge)
    lines += ["", "CUSTOMER NEEDS (stated needs are fixed; 'not stated' means you decide and say why):"]
    lines += customer_needs(prefs)
    inv = {wid: label for label, wid in labels.items()}
    if prefs.cabinet_walls:
        names = [inv[w] for w in prefs.cabinet_walls if w in inv]
        lines.append(f"- Use exactly these cabinet walls: {', '.join(names)}.")
    if prefs.primary_wall in inv:
        lines.append(f"- The hob goes on {inv[prefs.primary_wall]} (the customer's cooking wall).")
    if prefs.secondary_wall in inv:
        lines.append(f"- The sink goes on {inv[prefs.secondary_wall]} (the customer's choice).")
    if prefs.style_notes:
        lines.append(f"- Brief from the customer: {prefs.style_notes}")
    if not knowledge:
        return "\n".join(lines + _plain_task())
    lines += [
        "",
        "TASK - fill the JSON fields in order:",
        "1. room_reading: the window wall, the door wall and which corner the door is near, the long solid walls.",
        "2. design_plan: the zones and the work triangle in one or two sentences.",
        "3. the decisions, using ONLY codes and wall names from this message. Before each decision write its",
        "   *_why: one or two short sentences - the room fact or customer need that decides it, and why this option",
        "   rather than the other options on the menu (e.g. a smaller or larger size, or NONE). Use only the needs",
        "   listed above for THIS customer - never facts from the worked examples (other families).",
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


def ollama_model_for(provider: Optional[str], local_model: Optional[str] = None) -> str:
    """The Ollama model behind an AI provider name ("gemma" -> Gemma, "local" -> the chosen model, else Qwen)."""
    if provider == "local":
        return local_model or config.LOCAL_MODEL
    return config.GEMMA_MODEL if provider == "gemma" else config.OLLAMA_MODEL


LLAMACPP_PREFIX = "llamacpp:"


def local_knowledge(model: str) -> bool:
    """Full knowledge prompt for a local model? No for fine-tuned ones (FINETUNED_MODEL_TAGS), else LOCAL_USE_KNOWLEDGE."""
    name = (model or "").lower()
    if any(tag in name for tag in config.FINETUNED_MODEL_TAGS):
        return False
    return config.LOCAL_USE_KNOWLEDGE


def _llamacpp_headers() -> dict:
    return {"Authorization": f"Bearer {config.LLAMACPP_API_KEY}"} if config.LLAMACPP_API_KEY else {}


def list_llamacpp_models() -> list[dict]:
    """Models served by the optional llama.cpp servers (``config.LLAMACPP_URLS``), as ``llamacpp:<id>``.
    Servers that do not answer are skipped (raises only when none answers)."""
    out, errors = [], []
    for model_id, _url in _llamacpp_servers(errors).items():
        out.append({"name": LLAMACPP_PREFIX + model_id, "size_gb": None, "family": "llama.cpp"})
    if not out and errors:
        raise AIUnavailable("; ".join(errors))
    return out


def _llamacpp_servers(errors: Optional[list] = None) -> dict[str, str]:
    """model id -> base URL of the llama.cpp server that serves it."""
    import requests

    found = {}
    for url in config.LLAMACPP_URLS:
        try:
            r = requests.get(f"{url}/v1/models", headers=_llamacpp_headers(), timeout=5)
            r.raise_for_status()
            for m in r.json().get("data", []):
                found.setdefault(m["id"], url)
        except Exception as exc:
            if errors is not None:
                errors.append(f"{url}: {exc}")
    return found


def list_local_models(client=None) -> list[dict]:
    """Choices for the 'Local model' designer: installed Ollama models plus llama.cpp server models.
    Raises only when neither source answers."""
    out, errors = [], []
    try:
        out += _ollama_models(client)
    except Exception as exc:
        errors.append(f"Ollama: {exc}")
    try:
        out += list_llamacpp_models()
    except Exception as exc:
        errors.append(f"llama.cpp: {exc}")
    if not out and errors:
        raise AIUnavailable("; ".join(errors))
    return out


# Model architectures and the first Ollama version that runs them. Installed models of these families are hidden
# from the list on older Ollama versions (they fail with "unknown model architecture"); serve them with
# run_llamacpp.sh instead, or update Ollama.
OLLAMA_MIN_VERSION = {"lfm3": (0, 40, 12)}


def _ollama_version() -> tuple:
    import requests

    try:
        v = requests.get(f"{config.OLLAMA_HOST}/api/version", timeout=3).json().get("version", "")
        return tuple(int(x) for x in v.split("-")[0].split(".")[:3])
    except Exception:
        return ()


def _ollama_models(client=None) -> list[dict]:
    """Installed Ollama models (name, size GB, family) that this Ollama version can run."""
    client = client or _client()
    version = _ollama_version()
    resp = client.list()
    models = getattr(resp, "models", None) or (resp.get("models", []) if isinstance(resp, dict) else [])
    out = []
    for m in models:
        get = (lambda k: getattr(m, k, None)) if not isinstance(m, dict) else m.get
        name = get("model") or get("name")
        if not name:
            continue
        det = get("details")
        fam = getattr(det, "family", None) if det is not None and not isinstance(det, dict) else (det or {}).get("family")
        need = OLLAMA_MIN_VERSION.get(fam or "")
        if need and version and version < need:
            continue  # this Ollama cannot load it - offered through llama.cpp instead
        out.append({"name": name, "size_gb": round((get("size") or 0) / 1e9, 1), "family": fam})
    return sorted(out, key=lambda x: x["name"])


def _can_think(client, model: str) -> bool:
    """True when the Ollama model has a thinking mode (Qwen 3.x, Gemma 4 ...) - it is then switched off."""
    try:
        caps = getattr(client.show(model), "capabilities", None) or []
        return "thinking" in caps
    except Exception:
        return False


def ensure_model(client=None, auto_pull: bool = config.OLLAMA_AUTO_PULL, wait: bool = True,
                 model: Optional[str] = None) -> None:
    """Raise :class:`AIUnavailable` unless the Ollama model is ready (pull if missing).

    ``model`` defaults to ``config.OLLAMA_MODEL`` (Qwen). ``wait=False`` fails fast while
    another thread is downloading the model, so API requests in ``auto`` mode fall back
    instead of blocking on a multi-GB pull.
    """
    client = client or _client()
    try:
        names = _model_names(client)
    except Exception as exc:
        raise AIUnavailable(f"Ollama not reachable at {config.OLLAMA_HOST}: {exc}") from exc
    want = model or config.OLLAMA_MODEL
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


def build_messages(geometry: RoomGeometry, groups, prefs: KitchenPreferences, version: str,
                   knowledge: bool = True):
    """``(messages, Model, to_proposal)`` for one prompt version.

    ``knowledge=False`` (v2 only) leaves out the MagickHome knowledge pack: no playbook or
    worked examples, no cabinet notes, no rule hints in the room facts - the designer works
    from its own judgement. The answer schema (allowed codes and walls) is the same."""
    if version == "v1":
        Model = build_design_model(groups, geometry.wall_order, require_tall=prefs.include_tall)
        system, user = SYSTEM_PROMPT, _user_prompt(geometry, groups, prefs)

        def to_proposal(d):
            return _design_to_proposal(d, prefs)
    else:
        menu, labels = build_menu(groups), wall_labels(geometry)
        inv = {wid: label for label, wid in labels.items()}
        fixed = {"hob_wall": inv.get(prefs.primary_wall), "sink_wall": inv.get(prefs.secondary_wall)}
        Model = build_design_model_v2(menu, list(labels), require_tall=prefs.include_tall, prefs=prefs,
                                      lofts_possible=lofts_possible(geometry), fixed_walls=fixed)
        system = design_knowledge.system_prompt() if knowledge else PLAIN_SYSTEM_PROMPT
        user = _user_prompt_v2(geometry, menu, labels, prefs, knowledge=knowledge)

        def to_proposal(d):
            return _design_to_proposal_v2(d, menu, labels, prefs)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return messages, Model, to_proposal


def design_with_llamacpp(geometry: RoomGeometry, groups, prefs: KitchenPreferences, model_id: str,
                         version: Optional[str] = None) -> tuple[Proposal, dict]:
    """Design with a model on a llama.cpp server (OpenAI-compatible): same prompt and schema as Ollama.
    The schema is enforced by the server's JSON-schema grammar; thinking is switched off."""
    import requests

    version = version or config.AI_PROMPT_VERSION
    if not config.LLAMACPP_URLS:
        raise AIUnavailable("LLAMACPP_URL is not set")
    server = _llamacpp_servers().get(model_id)
    if not server:
        raise AIUnavailable(f"no llama.cpp server is serving '{model_id}' (start it with ./run_llamacpp.sh)")
    knowledge = local_knowledge(model_id)
    messages, Model, to_proposal = build_messages(geometry, groups, prefs, version, knowledge=knowledge)
    schema = Model.model_json_schema()
    errors = []
    for attempt in range(config.AI_MAX_RETRIES + 1):
        t0 = time.time()
        body = {
            "model": model_id, "messages": messages, "temperature": 0, "seed": config.OLLAMA_OPTIONS.get("seed", 42),
            "max_tokens": config.LLAMACPP_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "kitchen_design", "schema": schema}},
        }
        try:
            r = requests.post(f"{server}/v1/chat/completions", json=body, headers=_llamacpp_headers(),
                              timeout=config.LLAMACPP_TIMEOUT_S)
            data = r.json()
            if r.status_code != 200 or "error" in data:
                raise RuntimeError(str(data.get("error", data))[:300])
        except Exception as exc:
            raise AIUnavailable(f"llama.cpp server failed: {exc}") from exc
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        try:
            design = Model.model_validate_json(content)
        except ValidationError as exc:
            note = " (answer cut off - raise LLAMACPP_MAX_TOKENS)" if choice.get("finish_reason") == "length" else ""
            errors.append(str(exc)[:400] + note)
            messages = messages + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": f"Invalid JSON for the schema: {exc.errors()[:3]}. Reply again."}]
            continue
        usage, timings = data.get("usage") or {}, data.get("timings") or {}
        info = {
            "mode": "local", "provider": "local", "model": LLAMACPP_PREFIX + model_id, "server": "llama.cpp",
            "knowledge": knowledge, "prompt_version": version, "attempts": attempt + 1,
            "seconds": round(time.time() - t0, 1),
            "prompt_tokens": usage.get("prompt_tokens"), "eval_count": usage.get("completion_tokens"),
            "prompt_seconds": round(timings["prompt_ms"] / 1000, 1) if timings.get("prompt_ms") else None,
            "eval_seconds": round(timings["predicted_ms"] / 1000, 1) if timings.get("predicted_ms") else None,
            "raw": json.loads(design.model_dump_json()),
        }
        proposal = to_proposal(design)
        proposal.source = "local"
        return proposal, info
    raise AIUnavailable(f"model reply failed schema validation: {errors[-1] if errors else 'unknown'}")


def design_with_ollama(
    geometry: RoomGeometry, groups, prefs: KitchenPreferences, client=None, version: Optional[str] = None,
    provider: str = "qwen",
) -> tuple[Proposal, dict]:
    """Design with a local / Colab Ollama model. ``provider`` "qwen" (default), "gemma" or "local" (any
    installed model, ``prefs.local_model``): same prompt, knowledge pack and schema; only the model differs.
    Models with a thinking mode get it switched off (the answer already carries a reason per decision)."""
    version = version or config.AI_PROMPT_VERSION
    client = client or _client()
    model = ollama_model_for(provider, prefs.local_model)
    # a local model must already be installed: no multi-GB download in the middle of a design
    ensure_model(client, wait=prefs.ai_mode == "ollama", model=model, auto_pull=provider != "local")
    # A fine-tuned Gemma (finetune/) has the knowledge pack in its weights: send the short prompt.
    knowledge = (config.GEMMA_USE_KNOWLEDGE if provider == "gemma"
                 else local_knowledge(model) if provider == "local" else True)
    messages, Model, to_proposal = build_messages(geometry, groups, prefs, version, knowledge=knowledge)
    schema = Model.model_json_schema()
    if provider == "gemma":
        extra = {"think": config.GEMMA_THINK}
    elif provider == "local" and _can_think(client, model):
        extra = {"think": False}
    else:
        extra = {}
    errors = []
    for attempt in range(config.AI_MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat(
                model=model,
                messages=messages,
                format=schema,
                options=dict(config.OLLAMA_OPTIONS),
                keep_alive=config.OLLAMA_KEEP_ALIVE,
                **extra,
            )
        except Exception as exc:
            msg = str(exc)
            if "unknown model architecture" in msg:
                msg = (f"Ollama {'.'.join(map(str, _ollama_version())) or ''} cannot run {model} ({msg.split(':')[-1].strip()[:60]}). "
                       "Update Ollama, or serve it with ./run_llamacpp.sh and pick its llamacpp: entry.")
            raise AIUnavailable(f"Ollama chat failed: {msg}") from exc
        content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
        try:
            design = Model.model_validate_json(content)
        except ValidationError as exc:
            errors.append(str(exc)[:400])
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Invalid JSON for the schema: {exc.errors()[:3]}. Reply again."})
            continue
        info = {
            "mode": {"gemma": "gemma", "local": "local"}.get(provider, "ollama"),
            "provider": provider,
            "model": model,
            "knowledge": knowledge,
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
        proposal = to_proposal(design)
        if provider in ("gemma", "local"):
            proposal.source = provider
        return proposal, info
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
    reasons = {
        "layout": d.layout_why, "sink": d.sink_why, "hob": d.hob_why, "lpg": d.lpg_why, "tall": d.tall_why,
        "purifier": d.purifier_why, "display": d.display_why, "features": d.features_why,
        "storage": d.storage_why, "wall_cabinets": d.wall_cabinets_why,
        "open": d.open_why, "wall_style": d.wall_style_why, "lofts": d.loft_why, "finish": d.finish_why,
    }
    import finishes

    by_code = {finishes.code(a): a for a in finishes.catalogue()}
    chosen_finishes = {run: by_code[c] for run, c in (("base", d.base_finish), ("wall", d.wall_finish)) if c in by_code}
    talls = [(labels[t.wall], code[t.unit]) for t in d.tall_units] if prefs.include_tall else []
    host = labels[d.hob_wall] if d.hob_wall in labels else None
    talls, added = _ensure_built_in(talls, menu, prefs, host)
    talls, added2 = _ensure_crockery(talls, menu, prefs, host)
    for note in (added, added2):
        if note:
            reasons["tall"] = f"{reasons['tall']} {note}".strip()
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
        tall_units=talls,
        base_prefs=base_prefs,
        wall_prefs=wall_prefs,
        purifier_aid=aid(d.water_purifier) if prefs.water_purifier != "no" else None,
        purifier_required=prefs.water_purifier == "yes",
        display_aid=aid(d.display_unit) if prefs.display_unit != "no" else None,
        open_unit_aid=aid(d.open_unit) if prefs.open_unit != "no" else None,
        base_storage=d.base_storage,
        lofts=d.lofts == "yes",
        wall_style=d.wall_style,
        finishes=chosen_finishes,
        reasons={k: (v or "").strip() for k, v in reasons.items()},
        notes=f"{d.room_reading.strip()} {d.design_plan.strip()}".strip(),
        source="ollama",
    )


def _ensure_crockery(talls: list[tuple[str, str]], menu, prefs: KitchenPreferences, wall: Optional[str]):
    """Guarantee the crockery unit the customer asked for, keeping a needed appliance tower
    (at most two tall units: the crockery unit replaces a plain pantry). Returns (talls, note)."""
    if prefs.crockery_unit != "yes" or not prefs.include_tall:
        return talls, ""
    by_aid = {it.aid: it for _, opts in menu.items() for _, it in opts}
    if any(by_aid.get(a) is not None and by_aid[a].cat == CAT_TALL_GLASS for _, a in talls):
        return talls, ""
    unit = next((it for _, it in menu.get("tall", []) if it.cat == CAT_TALL_GLASS), None)
    if unit is None:
        return talls, "(The catalogue has no crockery unit for this door style.)"
    host = talls[0][0] if talls else wall
    if host is None:
        return talls, ""
    towers = BUILT_IN_CATS.get(prefs.built_in, ())
    keep = [t for t in talls if by_aid.get(t[1]) is not None and by_aid[t[1]].cat in towers]
    rest = [t for t in talls if t not in keep]
    return (keep + [(host, unit.aid)] + rest)[:2], "Added a crockery unit because the customer wants one."


def _ensure_built_in(talls: list[tuple[str, str]], menu, prefs: KitchenPreferences, wall: Optional[str]):
    """Guarantee the appliance tower the customer asked for (the schema allows it, the
    model might still skip it). Returns (talls, note)."""
    want = BUILT_IN_CATS.get(prefs.built_in) if prefs.include_tall else None
    if not want:
        return talls, ""
    by_aid = {it.aid: it for _, opts in menu.items() for _, it in opts}
    if any(by_aid.get(a) is not None and by_aid[a].cat in want for _, a in talls):
        return talls, ""
    tower = next((it for _, it in menu.get("tall", []) if it.cat in want), None)
    if tower is None:
        return talls, "(The catalogue has no matching appliance tower for this door style.)"
    host = talls[0][0] if talls else wall
    if host is None:
        return talls, ""
    talls = [(host, tower.aid)] + talls[:1]
    return talls, f"Added a {tower.category} because the customer needs it."


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
    if prefs.include_cylinder is not False:  # R11: LPG assumed unless piped gas / induction
        cyl = groups["cylinder"][0].aid if groups["cylinder"] else None
    talls = []
    if prefs.include_tall and groups["tall"]:
        tall_free = _free_len(geometry, tall_wall)
        if tall_free >= 2400 or len(walls) > 1:
            want_cats = BUILT_IN_CATS.get(prefs.built_in, ())
            tower = next((it for it in groups["tall"] if it.cat in want_cats), groups["tall"][0])
            talls.append((tall_wall, tower.aid))
            crockery = next((it for it in groups["tall"] if it.cat == CAT_TALL_GLASS), None)
            if prefs.crockery_unit == "yes" and crockery is not None:
                talls.append((tall_wall, crockery.aid))

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
    # water purifier beside the sink window (v2 shortlist only), honouring the customer's answer
    has_pur = bool(groups.get("purifier")) and prefs.water_purifier != "no" and (window_walls or prefs.water_purifier == "yes")
    has_disp = bool(groups.get("display")) and prefs.display_unit == "yes"
    has_open = bool(groups.get("open")) and prefs.open_unit == "yes"
    # Lofts are standard in Indian kitchens when the ceiling allows (big vessels, seasonal items).
    can_loft = lofts_possible(geometry)
    lofts = can_loft if prefs.include_loft is None else (prefs.include_loft and can_loft)
    wall_style = prefs.wall_style if prefs.wall_style != "auto" else "shutters"
    W = geometry.walls
    reasons = {
        "layout": f"{layout_type}-shape on the walls with the longest free runs.",
        "sink": f"{sink.width} mm suits a {sink_free:.0f} mm window-wall run." if window_walls else "Longest free wall.",
        "hob": f"Drawer hob unit on {W[hob_wall].name}, a different wall from the sink.",
        "lpg": "Customer uses LPG." if prefs.include_cylinder else (
            "Customer has piped gas or induction." if prefs.include_cylinder is False else
            "Fuel not stated: LPG is assumed, so the cylinder unit is included (rule R11)."),
        "tall": "Matches the customer's built-in appliance need." if prefs.built_in not in ("auto", "none") else "Pantry at a run end.",
        "purifier": "Customer needs a water purifier." if prefs.water_purifier == "yes" else (
            "Customer does not want one." if prefs.water_purifier == "no" else
            ("Standard beside the sink window." if has_pur else "No window wall for it.")),
        "display": "Customer wants a glass display." if has_disp else "Not requested.",
        "features": "Drawers by the hob, grain trolley on the sink wall.",
        "open": "Customer wants an open shelf unit." if has_open else "Not requested.",
        "wall_style": "Customer's choice." if prefs.wall_style != "auto" else "Hinged shutters: the standard, lowest-cost door.",
        "lofts": ("Customer wants lofts." if prefs.include_loft else "Lofts store big vessels and seasonal items.")
        if lofts else ("Customer does not want lofts." if prefs.include_loft is False else "Ceiling too low for lofts."),
        "finish": _fallback_finish_reason(prefs),
    }
    tall_aids = {a for _, a in talls}
    if any(it.cat == CAT_TALL_GLASS and it.aid in tall_aids for it in groups["tall"]):
        reasons["tall"] += " Crockery unit as the customer wants."
    stated = base_storage_need(prefs)[0]
    if stated != "auto":
        storage = stated
    elif prefs.cooking == "heavy" or (prefs.household_size or 0) >= 5:
        storage, reasons["storage"] = "drawers", "Heavy cooking or a big family: drawers keep many vessels reachable."
    elif prefs.cooking == "light":
        storage, reasons["storage"] = "shutters", "Light cooking: shutter storage is enough."
    else:
        storage, reasons["storage"] = "mixed", "Drawer + shutter units: a cutlery drawer on top, shelf space below."
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
        purifier_aid=groups["purifier"][0].aid if has_pur else None,
        purifier_required=prefs.water_purifier == "yes",
        display_aid=groups["display"][0].aid if has_disp else None,
        open_unit_aid=groups["open"][0].aid if has_open else None,
        base_storage=storage,
        lofts=lofts,
        wall_style=wall_style,
        reasons=reasons,
        notes=(
            f"{layout_type}-shaped kitchen: sink on {geometry.walls[sink_wall].name}"
            f"{' under the window' if window_walls else ''}, hob on {geometry.walls[hob_wall].name}"
            f"{', tall storage on ' + geometry.walls[tall_wall].name if talls else ''}."
        ),
        source="fallback",
    )


def _fallback_finish_reason(prefs: KitchenPreferences) -> str:
    import finishes

    budget = prefs.budget if prefs.budget != "auto" else None
    return finishes.default(budget)[1] or "Standard MagickHome palette."


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
            provider = prefs.ai_provider or config.AI_PROVIDER
            if provider == "gemini":
                from gemini_designer import design_with_gemini

                proposal, ai = design_with_gemini(geometry, groups, prefs, version=version)
            elif provider == "local" and (prefs.local_model or "").startswith(LLAMACPP_PREFIX):
                proposal, ai = design_with_llamacpp(geometry, groups, prefs, prefs.local_model[len(LLAMACPP_PREFIX):],
                                                    version=version)
            else:
                proposal, ai = design_with_ollama(geometry, groups, prefs, version=version,
                                                  provider=provider if provider in ("gemma", "local") else "qwen")
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
