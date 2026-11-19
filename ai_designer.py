"""AI interior designer: local Ollama ``qwen2.5-coder:3b`` with a deterministic fallback.

Qwen's job is *design intent* - which walls carry cabinets, where the sink / hob /
tall units go and which SKUs to use. It never emits coordinates that are
trusted blindly: :mod:`rule_engine` snaps and validates everything.

Hallucination guard: the response format is a JSON schema generated from a
Pydantic model whose fields are ``Literal`` enums built from the filtered
catalogue shortlist (35-45 SKUs) and the scene's wall ids. Ollama compiles the
schema into a grammar, so the model can only produce valid AIDs / wall ids;
the reply is then re-validated with the same Pydantic model.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, create_model

import config
from catalog_loader import (
    CAT_BASE,
    CAT_CYLINDER,
    CAT_GRAIN,
    CAT_HOB,
    CAT_OVEN,
    CAT_SINK,
    CAT_SPICE,
    CAT_TALL,
    CAT_WALL,
    CAT_WICKER,
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
    ai_mode: Literal["auto", "ollama", "fallback"] = Field(
        "auto", description="auto = Ollama with deterministic fallback; ollama = require the LLM"
    )
    strip_existing_items: bool = True
    include_floor: bool = True
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Catalogue shortlist (what the LLM may choose from)
# ---------------------------------------------------------------------------

def build_shortlist(catalog: Catalog, max_items: int = config.AI_SHORTLIST_MAX) -> dict[str, list[CatalogItem]]:
    """~35-45 representative SKUs grouped by design slot."""

    def pick(cat, widths, opening=None, combination=None, height=None, depth=None, limit=None):
        out = []
        for w in widths:
            found = catalog.find(cat, width=w, opening=opening, combination=combination, height=height, depth=depth)
            if found:
                out.append(found[0])
        return out[:limit] if limit else out

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
    API requests in ``auto`` mode fall back instead of blocking on a 2 GB pull.
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
        log.info("pulling %s (one-time download, ~2 GB)...", want)
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


def design_with_ollama(geometry: RoomGeometry, groups, prefs: KitchenPreferences, client=None) -> tuple[Proposal, dict]:
    client = client or _client()
    ensure_model(client, wait=prefs.ai_mode == "ollama")
    Model = build_design_model(groups, geometry.wall_order, require_tall=prefs.include_tall)
    schema = Model.model_json_schema()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(geometry, groups, prefs)},
    ]
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
            "attempts": attempt + 1,
            "seconds": round(time.time() - t0, 1),
            "eval_count": getattr(resp, "eval_count", None),
            "raw": json.loads(design.model_dump_json()),
        }
        return _design_to_proposal(design, prefs), info
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


# ---------------------------------------------------------------------------
# Deterministic fallback designer
# ---------------------------------------------------------------------------

def _free_len(geometry: RoomGeometry, wid: str) -> float:
    wall = geometry.walls[wid]
    lo, hi = wall.usable
    cuts = sorted(d.interval(config.DOOR_CLEARANCE_MM) for d in wall.doors)
    best, cur = 0.0, lo
    for a, b in cuts:
        best = max(best, min(a, hi) - cur)
        cur = max(cur, b)
    return max(best, hi - cur)


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
    sink = next((it for it in groups["sink"] if it.width <= max(900, sink_free - 1800)), groups["sink"][-1])
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
        notes=(
            f"{layout_type}-shaped kitchen: sink on {geometry.walls[sink_wall].name}"
            f"{' under the window' if window_walls else ''}, hob on {geometry.walls[hob_wall].name}"
            f"{', tall storage on ' + geometry.walls[tall_wall].name if talls else ''}."
        ),
        source="fallback",
    )


def design_kitchen(geometry: RoomGeometry, catalog: Catalog, prefs: KitchenPreferences) -> tuple[Proposal, dict]:
    """Return ``(proposal, ai_info)`` according to ``prefs.ai_mode``."""
    groups = build_shortlist(catalog)
    n = len(shortlist_items(groups))
    info: dict = {"shortlist_size": n}
    if prefs.ai_mode in ("auto", "ollama"):
        try:
            proposal, ai = design_with_ollama(geometry, groups, prefs)
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
