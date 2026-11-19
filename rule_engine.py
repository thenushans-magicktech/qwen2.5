"""Deterministic spatial & ergonomic rule solver.

The AI designer (or the deterministic fallback) produces a :class:`Proposal`:
which walls carry cabinets and which catalogue SKUs to use. This module turns
that proposal into an exact, collision-free :class:`KitchenLayout`, enforcing:

1. Sink-under-window: the sink is centred on the window of its wall and the
   window span is kept clear of wall / tall / loft cabinets.
2. Door clearance: no cabinet within ``DOOR_CLEARANCE_MM`` of a door frame on
   its wall; hinged-door swing zones are also kept clear on adjacent walls.
3. Zero collision & exact packing: cabinets stay inside the wall's interior
   span, never overlap on a tier, and are packed flush. Runs are filled with
   catalogue widths (dynamic programming over the 50 mm module); leftovers go
   to an open end (no filler) or become a filler strip against a wall/corner.
4. Ergonomics: spice pull-out beside the hob, chimney zone (900 mm + side
   clearance) free of wall units, tall units at the far ends of runs, blind
   corner units at L/U corners with 560 mm reserved on the perpendicular wall,
   >= 300 mm landing beside the hob, >= 600 mm between sink and hob.
5. Door swing direction: single-shutter units next to a wall, door opening,
   corner or tall unit use the variant hinged on that side.

All coordinates are millimetres along the wall from its ``start`` point.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

import config
from catalog_loader import (
    BASE_STORAGE_CATS,
    CAT_BASE,
    CAT_BCU_L,
    CAT_BCU_R,
    CAT_CYLINDER,
    CAT_HOB,
    CAT_LOFT,
    CAT_LOFT_CORNER,
    CAT_SINK,
    CAT_SPICE,
    CAT_WALL,
    CAT_WALL_CORNER,
    CORNER_CATS,
    TALL_CATS,
    Catalog,
    CatalogItem,
)
from scene_parser import Corner, Opening, RoomGeometry, Wall

log = logging.getLogger(__name__)

EPS = 0.5  # mm tolerance
# corner_open = a corner between two cabinet runs that is left empty because no
# corner cabinet fits there (regular cabinets are never used in corners).
SOFT_KINDS = frozenset({"door", "swing", "open", "window", "chimney", "corner_open"})
BLOCKING_KINDS = ("door", "swing", "window", "chimney", "corner", "corner_open")
HARD_KINDS = frozenset({"wall", "corner"})
STRONG_HINGE_OBSTACLES = frozenset({"wall", "door", "swing", "corner", "tall"})

# Vertical extent of each placement zone (metres) - used for opening clashes.
ZONE_Y = {"base": (0.0, 0.84), "wall": (1.44, 2.04), "loft": (2.058, config.LOFT_TOP_M)}
TIER_ZONES = {"base": ("base",), "tall": ("base", "wall"), "wall": ("wall",), "loft": ("loft",)}
ZONE_DEPTH_MM = {"base": 560, "wall": 330, "loft": 330}


class SolverError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    """Designer intent (from Qwen or the deterministic fallback)."""

    layout_type: str = "auto"
    cabinet_walls: list[str] = field(default_factory=list)
    sink_wall_id: Optional[str] = None
    sink_aid: Optional[str] = None
    sink_center_mm: Optional[float] = None
    hob_wall_id: Optional[str] = None
    hob_aid: Optional[str] = None
    hob_center_mm: Optional[float] = None
    spice_aid: Optional[str] = None
    cylinder_aid: Optional[str] = None
    tall_units: list[tuple[str, str]] = field(default_factory=list)  # (wall_id, aid)
    base_prefs: dict[str, list[str]] = field(default_factory=dict)
    wall_prefs: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""
    source: str = "fallback"

    def to_dict(self) -> dict:
        return {
            "layout_type": self.layout_type,
            "cabinet_walls": self.cabinet_walls,
            "sink": {"wall_id": self.sink_wall_id, "aid": self.sink_aid, "center_mm": self.sink_center_mm},
            "hob": {"wall_id": self.hob_wall_id, "aid": self.hob_aid, "center_mm": self.hob_center_mm},
            "spice_aid": self.spice_aid,
            "cylinder_aid": self.cylinder_aid,
            "tall_units": [{"wall_id": w, "aid": a} for w, a in self.tall_units],
            "base_prefs": self.base_prefs,
            "wall_prefs": self.wall_prefs,
            "notes": self.notes,
            "source": self.source,
        }


@dataclass
class SolverPrefs:
    include_loft: bool = False
    layout: str = "auto"  # auto | single | L | U
    cabinet_walls: Optional[list[str]] = None
    sink_under_window: bool = True


@dataclass
class Placement:
    item: CatalogItem
    wall_id: str
    tier: str  # base | tall | wall | loft
    start_mm: float
    role: str  # sink hob spice cylinder corner storage tall wall wall_corner loft loft_deep
    corner_end: Optional[str] = None
    note: str = ""

    @property
    def width_mm(self) -> int:
        return self.item.width

    @property
    def end_mm(self) -> float:
        return self.start_mm + self.item.width

    @property
    def center_mm(self) -> float:
        return self.start_mm + self.item.width / 2.0

    @property
    def zones(self) -> tuple[str, ...]:
        return TIER_ZONES[self.tier]

    def pos_y(self) -> float:
        if self.tier == "base":
            return config.POS_Y_M["base_floor"] if self.item.floor_standing else config.POS_Y_M["base"]
        return config.POS_Y_M[self.tier]

    def pos_z(self, wall: Wall) -> float:
        """Centre depth from the wall centre-line: face + depth/2."""
        return round(wall.thickness_m / 2.0 + self.item.depth / 2000.0, 6)

    def label(self) -> str:
        return f"{self.item.category} {self.item.width} [{self.start_mm:.0f}-{self.end_mm:.0f}]"


@dataclass
class Feature:
    lo: float
    hi: float
    kind: str
    ref: str = ""


@dataclass
class CornerPlan:
    """A corner where two cabinet runs meet. Each tier gets a corner cabinet or stays open."""

    corner: Corner
    primary: str
    primary_end: str
    secondary: str
    secondary_end: str
    base_unit: Optional[Placement] = None
    wall_unit: Optional[Placement] = None
    loft_unit: Optional[Placement] = None
    open_tiers: set = field(default_factory=set)
    wall_reserve_mm: float = 0.0

    def end_of(self, wid: str) -> str:
        return self.primary_end if wid == self.primary else self.secondary_end


@dataclass
class Gap:
    wall_id: str
    lo: float
    hi: float
    kind_lo: str
    kind_hi: str
    caps_lo: list = field(default_factory=list)  # CatalogItems flush at lo (ordered lo->hi)
    caps_hi: list = field(default_factory=list)  # CatalogItems flush at hi (ordered lo->hi)

    @property
    def length(self) -> float:
        return self.hi - self.lo


@dataclass
class FillResult:
    items: list[tuple[CatalogItem, float, str]]  # (item, start_mm, role)
    leftover: float
    leftover_at: str  # lo | hi | none
    penalty: float
    used_prefs: int = 0


@dataclass
class KitchenLayout:
    geometry: RoomGeometry
    placements: list[Placement]
    cabinet_walls: list[str]
    corner_plans: list[CornerPlan]
    features: dict[tuple[str, str], list[Feature]]
    chimney: Optional[dict]
    sink: Optional[Placement]
    hob: Optional[Placement]
    sink_window: Optional[Opening]
    loft_walls: list[str]
    proposal: Proposal
    decisions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0

    def on(self, wall_id: str, *zones: str) -> list[Placement]:
        zs = set(zones) if zones else {"base", "wall", "loft"}
        return sorted(
            (p for p in self.placements if p.wall_id == wall_id and zs.intersection(p.zones)),
            key=lambda p: p.start_mm,
        )

    def boundary(self, wall_id: str, zone: str, x: float, direction: int, exclude: Iterable[Placement] = ()):
        """Nearest obstacle from ``x`` towards ``direction`` (+1 = wall end).

        Returns ``(kind, coord, distance_mm)``; kinds: wall/open (wall bounds),
        door/swing/window/chimney/corner/tall (features) or ``"cabinet"``.
        """
        wall = self.geometry.walls[wall_id]
        lo, hi = wall.usable
        excl = {id(p) for p in exclude}
        best = None

        def consider(kind, coord):
            nonlocal best
            dist = (coord - x) * direction
            if dist < -EPS:
                return
            if best is None or dist < best[2] - 1e-9:
                best = (kind, coord, max(0.0, dist))

        if direction > 0:
            consider("wall" if wall.neighbor_end else "open", hi)
        else:
            consider("wall" if wall.neighbor_start else "open", lo)
        for f in self.features.get((wall_id, zone), []):
            consider(f.kind, f.lo if direction > 0 else f.hi)
        for p in self.placements:
            if p.wall_id != wall_id or zone not in p.zones or id(p) in excl:
                continue
            consider("tall" if p.tier == "tall" else "cabinet", p.start_mm if direction > 0 else p.end_mm)
        return best

    def summary(self) -> dict:
        walls = {}
        for wid in self.cabinet_walls:
            walls[wid] = [
                {
                    "aid": p.item.aid,
                    "category": p.item.category,
                    "tier": p.tier,
                    "role": p.role,
                    "width": p.item.width,
                    "start_mm": round(p.start_mm, 1),
                    "end_mm": round(p.end_mm, 1),
                    "opening": p.item.opening_direction,
                }
                for p in sorted(self.on(wid), key=lambda p: (p.tier, p.start_mm))
            ]
        return {
            "cabinet_walls": self.cabinet_walls,
            "placements": walls,
            "chimney": self.chimney,
            "corners": [
                {"walls": [c.primary, c.secondary],
                 "base_unit": c.base_unit.item.aid if c.base_unit else None,
                 "wall_unit": c.wall_unit.item.aid if c.wall_unit else None,
                 "loft_unit": c.loft_unit.item.aid if c.loft_unit else None,
                 "open_tiers": sorted(c.open_tiers)}
                for c in self.corner_plans
            ],
            "loft_walls": self.loft_walls,
            "score": round(self.score, 2),
            "decisions": self.decisions,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def overlaps(a_lo: float, a_hi: float, b_lo: float, b_hi: float, tol: float = EPS) -> bool:
    return a_lo < b_hi - tol and b_lo < a_hi - tol


def merge_intervals(iv: Iterable[tuple[float, float]], join: float = EPS) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1] + join:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def convex_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]], tol: float = 1e-4) -> bool:
    """Separating-axis test for two convex polygons (touching != overlapping)."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            nx, nz = -(p2[1] - p1[1]), p2[0] - p1[0]
            ln = math.hypot(nx, nz)
            if ln < 1e-12:
                continue
            nx, nz = nx / ln, nz / ln
            pa = [p[0] * nx + p[1] * nz for p in a]
            pb = [p[0] * nx + p[1] * nz for p in b]
            if max(pa) <= min(pb) + tol or max(pb) <= min(pa) + tol:
                return False
    return True


def footprint(wall: Wall, x0_mm: float, x1_mm: float, z0_m: float, z1_m: float) -> list[tuple[float, float]]:
    return [
        wall.to_world(x0_mm / 1000.0, z0_m),
        wall.to_world(x1_mm / 1000.0, z0_m),
        wall.to_world(x1_mm / 1000.0, z1_m),
        wall.to_world(x0_mm / 1000.0, z1_m),
    ]


def placement_box(p: Placement, wall: Wall):
    """World footprint polygon + vertical span (metres) of a placement."""
    face = wall.thickness_m / 2.0
    poly = footprint(wall, p.start_mm, p.end_mm, face, face + p.item.depth / 1000.0)
    y0 = p.pos_y()
    return poly, (y0, y0 + p.item.height / 1000.0)


# ---------------------------------------------------------------------------
# Width packing (dynamic programming over the 50 mm catalogue module)
# ---------------------------------------------------------------------------

def _unit_cost(width: int, narrow: bool = False) -> float:
    c = 1.0 + abs(width - 600) / 5000.0
    if width < 450:
        c += 0.35
    if width > 900:
        c += 0.25
    if narrow:
        c += 3.0
    return c


@lru_cache(maxsize=200_000)
def _dp_fill(n_steps: int, widths_costs: tuple[tuple[int, float], ...]) -> tuple[int, ...]:
    """Cheapest multiset of widths with the largest reachable sum <= n_steps*GRID."""
    g = config.GRID_MM
    best: list[Optional[tuple[float, tuple[int, ...]]]] = [None] * (n_steps + 1)
    best[0] = (0.0, ())
    for k in range(1, n_steps + 1):
        cand = None
        for w, c in widths_costs:
            j = k - w // g
            if j < 0 or best[j] is None:
                continue
            val = best[j][0] + c
            if cand is None or val < cand[0] - 1e-9:
                cand = (val, best[j][1] + (w,))
        best[k] = cand
    for k in range(n_steps, -1, -1):
        if best[k] is not None:
            return tuple(sorted(best[k][1], reverse=True))
    return ()


class Pool:
    """Default SKU per width for one fill context (e.g. base storage)."""

    def __init__(self, items_by_width: dict[int, CatalogItem], narrow: Optional[dict[int, CatalogItem]] = None):
        self.by_width = dict(items_by_width)
        self.narrow = {w: it for w, it in (narrow or {}).items() if w not in self.by_width}
        costs = [(w, _unit_cost(w)) for w in self.by_width] + [(w, _unit_cost(w, True)) for w in self.narrow]
        self.widths_costs = tuple(sorted(c for c in costs if c[0] % config.GRID_MM == 0))
        self.min_width = min((w for w, _ in self.widths_costs), default=10**9)
        # Narrow units (e.g. 150 mm pull-outs) are a last resort for the DP only;
        # preferred SKUs must leave a remainder that regular units can fill.
        self.min_regular = min(self.by_width, default=10**9)

    def item(self, width: int) -> CatalogItem:
        return self.by_width.get(width) or self.narrow[width]

    def fillable(self, r: float) -> bool:
        return r < config.GRID_MM - EPS or r >= self.min_regular - EPS

    def _dp(self, r: float) -> list[CatalogItem]:
        n = int(math.floor((r + EPS) / config.GRID_MM))
        return [self.item(w) for w in _dp_fill(n, self.widths_costs)] if n >= 1 else []

    def choose(self, target: float, prefs: list[CatalogItem]) -> tuple[list[CatalogItem], int]:
        """Units for ``target`` mm: designer-preferred SKUs when they fit well, else pure DP.

        Preferences are honoured only if they do not make the run worse than the
        optimal packing: no extra narrow (< 450 mm) units, at most one extra unit,
        and no extra leftover. This stops a small model's habit of picking the
        first (narrowest) catalogue items from fragmenting runs.
        """
        best = self._dp(target)
        if not prefs:
            return best, 0

        def narrow(units):
            return sum(1 for u in units if u.width < 450)

        def left(units):
            return target - sum(u.width for u in units)

        def attempt(cands):
            chosen, r = [], target
            for p in cands:
                if p.width <= r + EPS and self.fillable(r - p.width):
                    chosen.append(p)
                    r -= p.width
            return chosen, chosen + self._dp(r)

        for cands in [prefs] + [[p] for p in prefs]:
            chosen, units = attempt(cands)
            if (
                chosen
                and len(units) <= len(best) + 1
                and narrow(units) <= narrow(best)
                and left(units) <= left(best) + EPS
            ):
                return units, len(chosen)
        return best, 0


def fill_gap(gap: Gap, pool: Pool, prefs: list[CatalogItem], role: str = "storage") -> Optional[FillResult]:
    caps_w = sum(it.width for it in gap.caps_lo) + sum(it.width for it in gap.caps_hi)
    target = gap.length - caps_w
    if target < -EPS:
        return None
    units, used_prefs = pool.choose(max(0.0, target), prefs)
    left = max(0.0, target - sum(u.width for u in units))
    soft_lo, soft_hi = gap.kind_lo in SOFT_KINDS, gap.kind_hi in SOFT_KINDS
    hard_lo, hard_hi = gap.kind_lo in HARD_KINDS, gap.kind_hi in HARD_KINDS

    if left <= EPS:
        at, kind = "none", None
    elif soft_lo and not soft_hi:
        at, kind = "lo", "soft"
    elif soft_hi:
        at, kind = "hi", "soft"
    elif hard_lo and not hard_hi:
        at, kind = "lo", "filler"
    elif hard_hi:
        at, kind = "hi", "filler"
    else:
        at, kind = "hi", "mid"

    seq = [(it, "tall" if it.tier == "tall" else role) for it in gap.caps_lo]
    seq += [(it, role) for it in units]
    seq += [(it, "tall" if it.tier == "tall" else role) for it in gap.caps_hi]
    items = []
    x = gap.lo + (left if at == "lo" else 0.0)
    for it, r in seq:
        items.append((it, x, r))
        x += it.width

    pen = 2.0 * len(units) - 3.0 * used_prefs
    if kind == "soft":
        pen += 0.02 * left
    elif kind in ("filler", "mid"):
        if left > config.MAX_FILLER_MM:
            pen += 1000.0 + left
        elif left > config.GRID_MM:
            pen += 15.0 + left
        else:
            pen += 5.0 + 0.2 * left
        if kind == "mid":
            pen += 10.0
    return FillResult(items, left, at, pen, used_prefs)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class _BasePlan:
    score: float
    placements: list[Placement]
    corner_plans: list[CornerPlan]
    reserves: dict[str, list[Feature]]
    sink: Optional[Placement]
    hob: Optional[Placement]
    chimney: Optional[dict]
    decisions: list[str]
    warnings: list[str]


class LayoutSolver:
    def __init__(self, geometry: RoomGeometry, catalog: Catalog, proposal: Proposal, prefs: SolverPrefs):
        self.g = geometry
        self.cat = catalog
        self.proposal = proposal
        self.prefs = prefs
        self.decisions: list[str] = []
        self.warnings: list[str] = []

        self.base_pool = self._pool((CAT_BASE,), "base", height=700, narrow_cats=(CAT_SPICE,))
        self.wall_pool = self._pool((CAT_WALL,), "wall", height=600, depth=330)
        self.loft_pool = self._pool((CAT_LOFT,), "loft", height=600, depth=330)
        self.loft_deep_pool = self._pool((CAT_LOFT,), "loft", height=600, depth=560)
        if not self.base_pool.by_width:
            raise SolverError("catalogue has no 700 mm 'Base Unit' SKUs for this door style")

        self.walls = self._resolve_cabinet_walls()
        self.static = self._static_features()

    # -- catalogue helpers -------------------------------------------------
    def _pool(self, cats, tier, height=None, depth=None, narrow_cats=()) -> Pool:
        by_w: dict[int, CatalogItem] = {}
        for it in self.cat.find(cats, tier=tier, height=height, depth=depth):
            by_w.setdefault(it.width, it)
        narrow: dict[int, CatalogItem] = {}
        for it in self.cat.find(narrow_cats, tier=tier, height=height) if narrow_cats else []:
            narrow.setdefault(it.width, it)
        return Pool(by_w, narrow)

    def _resolve(self, aid: Optional[str], cats: Iterable[str], what: str, tier: Optional[str] = None) -> Optional[CatalogItem]:
        if not aid:
            return None
        it = self.cat.get(aid)
        cats = set(cats)
        if it is None:
            self.warnings.append(f"{what}: AID {aid} is not in the filtered catalogue; using default")
            return None
        if it.cat not in cats or (tier and it.tier != tier):
            self.warnings.append(f"{what}: {aid} is a '{it.category}', not a valid {what}; using default")
            return None
        return it

    def _default(self, cats, width=None, opening=None, tier=None, height=None, depth=None) -> Optional[CatalogItem]:
        found = self.cat.find(cats, width=width, opening=opening, tier=tier, height=height, depth=depth)
        return found[0] if found else None

    # -- walls & static features ---------------------------------------------
    def free_base_segments(self, wall: Wall) -> list[tuple[float, float]]:
        lo, hi = wall.usable
        blocked = [f for f in self._door_features(wall, "base")]
        return _subtract([(lo, hi)], [(f.lo, f.hi) for f in blocked])

    def _resolve_cabinet_walls(self) -> list[str]:
        walls = self.g.walls
        requested = self.prefs.cabinet_walls or self.proposal.cabinet_walls or []
        chosen = []
        for wid in requested:
            if wid not in walls:
                self.warnings.append(f"cabinet wall {wid} does not exist; ignored")
                continue
            if wid in chosen:
                continue
            free = max((b - a for a, b in self.free_base_segments(walls[wid])), default=0)
            if free < config.MIN_CABINET_WALL_MM:
                self.warnings.append(f"{walls[wid].name}: only {free:.0f} mm free base run; cannot carry cabinets")
                continue
            chosen.append(wid)
        expected = {"single": 1, "L": 2, "U": 3}.get(self.prefs.layout)
        if chosen and expected and not self.prefs.cabinet_walls and len(chosen) != expected:
            self.decisions.append(
                f"designer proposed {len(chosen)} wall(s) but the customer asked for layout {self.prefs.layout}; re-chosen"
            )
            chosen = []
        if not chosen:
            chosen = suggest_cabinet_walls(self.g, self.prefs.layout)
            self.decisions.append(f"cabinet walls chosen by solver: {[walls[w].name for w in chosen]}")

        if self.prefs.sink_under_window and not self.prefs.cabinet_walls:
            if not any(walls[w].windows for w in chosen):
                for win in sorted(self.g.windows, key=lambda o: -o.width_mm):
                    wid = win.wall_id
                    free = max((b - a for a, b in self.free_base_segments(walls[wid])), default=0)
                    if free < config.MIN_CABINET_WALL_MM:
                        continue
                    adjacent = any(self.g.corner_between(wid, c) for c in chosen)
                    if len(chosen) < 3 and (adjacent or len(chosen) == 0):
                        chosen.append(wid)
                    else:
                        chosen = [wid] + [c for c in chosen if self.g.corner_between(wid, c)][:2]
                    self.decisions.append(f"added window wall {walls[wid].name} so the sink can sit under the window")
                    break
        order = {w: i for i, w in enumerate(self.g.wall_order)}
        return sorted(chosen, key=lambda w: order.get(w, 99))

    def _door_features(self, wall: Wall, zone: str) -> list[Feature]:
        out = []
        y0, _ = ZONE_Y[zone]
        for d in wall.doors:
            if d.top_m > y0 + 1e-6:
                lo, hi = d.interval(config.DOOR_CLEARANCE_MM)
                out.append(Feature(lo, hi, "door", d.id))
        return out

    def _static_features(self) -> dict[tuple[str, str], list[Feature]]:
        feats: dict[tuple[str, str], list[Feature]] = {}
        for wid, wall in self.g.walls.items():
            for zone in ("base", "wall", "loft"):
                fl = self._door_features(wall, zone)
                y0, y1 = ZONE_Y[zone]
                if zone != "base":
                    for w in wall.windows:
                        if w.top_m > y0 + 1e-6 and w.bottom_m < y1 - 1e-6:
                            lo, hi = w.interval(config.WINDOW_CLEARANCE_MM)
                            fl.append(Feature(lo, hi, "window", w.id))
                feats[(wid, zone)] = fl
        if config.DOOR_SWING_CHECK:
            for door in self.g.doors:
                if not door.hinged_inward:
                    continue
                poly = self._swing_polygon(door)
                for wid, wall in self.g.walls.items():
                    if wid == door.wall_id:
                        continue
                    for zone in ("base", "wall", "loft"):
                        if door.top_m <= ZONE_Y[zone][0]:
                            continue
                        for lo, hi in self._project(poly, wall, ZONE_DEPTH_MM[zone]):
                            feats[(wid, zone)].append(Feature(lo, hi, "swing", door.id))
        return feats

    def _swing_polygon(self, door: Opening):
        wall = self.g.walls[door.wall_id]
        face = wall.thickness_m / 2.0
        lo, hi = door.interval(config.DOOR_CLEARANCE_MM)
        return footprint(wall, lo, hi, face, face + door.width_mm / 1000.0)

    @staticmethod
    def _project(poly, wall: Wall, depth_mm: float, step: int = 10) -> list[tuple[float, float]]:
        face = wall.thickness_m / 2.0
        hits = []
        for x in range(0, wall.length_mm, step):
            strip = footprint(wall, x, x + step, face, face + depth_mm / 1000.0)
            if convex_overlap(strip, poly):
                hits.append((float(x), float(min(x + step, wall.length_mm))))
        return merge_intervals(hits, join=1.0)

    # -- main entry ------------------------------------------------------------
    def solve(self) -> KitchenLayout:
        corners = [
            c for c in self.g.corners if c.wall_a in self.walls and c.wall_b in self.walls
        ]
        sink_wall, sink_window = self._sink_target()
        sink_options = self._sink_options()
        hob_item = self._resolve(self.proposal.hob_aid, (CAT_HOB,), "hob", "base") or self._default(
            (CAT_HOB,), width=900, opening="Pull Out", height=700
        )
        if hob_item is None:
            raise SolverError("no Base Hob Unit available in the catalogue")
        spice_item = self._resolve(self.proposal.spice_aid, (CAT_SPICE,), "spice pull-out", "base") or self._default(
            (CAT_SPICE,), width=300
        )
        cyl_item = self._resolve(self.proposal.cylinder_aid, (CAT_CYLINDER,), "cylinder unit", "base")
        talls = []
        for wid, aid in self.proposal.tall_units[:2]:
            it = self._resolve(aid, TALL_CATS, "tall unit", "tall")
            if it is not None:
                talls.append((wid, it))

        best: Optional[_BasePlan] = None
        # Each corner: blind corner unit on wall_a or wall_b; "open" (None) only
        # when neither wall can take one - never a regular cabinet in the corner.
        options = []
        for c in corners:
            opts = [w for w in (c.wall_a, c.wall_b) if self._bcu_fits(c, w)]
            options.append(opts or [None])
        assignments = list(itertools.product(*options)) or [()]
        for assign in assignments:
            for sink_item in sink_options or [None]:
                plan = self._plan_base(corners, assign, sink_wall, sink_window, sink_item, hob_item, spice_item, cyl_item, talls)
                if plan and (best is None or plan.score < best.score - 1e-9):
                    best = plan
        if best is None:
            raise SolverError("no feasible base layout (walls too short or fully blocked)")

        self.decisions.extend(best.decisions)
        self.warnings.extend(best.warnings)
        layout = KitchenLayout(
            geometry=self.g,
            placements=list(best.placements),
            cabinet_walls=self.walls,
            corner_plans=best.corner_plans,
            features={k: list(v) for k, v in self.static.items()},
            chimney=best.chimney,
            sink=best.sink,
            hob=best.hob,
            sink_window=sink_window,
            loft_walls=[],
            proposal=self.proposal,
            decisions=self.decisions,
            warnings=self.warnings,
            score=best.score,
        )
        for wid, fl in best.reserves.items():
            layout.features[(wid, "base")].extend(fl)
        if best.chimney:
            c = best.chimney["center_mm"]
            half = config.CHIMNEY_WIDTH_MM / 2 + config.CHIMNEY_SIDE_CLEARANCE_MM
            layout.features[(best.chimney["wall_id"], "wall")].append(Feature(c - half, c + half, "chimney", "chimney"))
        for p in layout.placements:
            if p.tier == "tall":
                layout.features[(p.wall_id, "wall")].append(Feature(p.start_mm, p.end_mm, "tall", p.item.aid))

        self._plan_wall_tier(layout)
        if self.prefs.include_loft:
            self._plan_loft(layout)
        self._apply_hinges(layout)
        self._repair(layout)
        return layout

    # -- sink ---------------------------------------------------------------
    def _sink_target(self) -> tuple[Optional[str], Optional[Opening]]:
        prop_wall = self.proposal.sink_wall_id
        if self.prefs.sink_under_window:
            windows = [w for w in self.g.windows if w.wall_id in self.walls]
            if windows:
                on_prop = [w for w in windows if w.wall_id == prop_wall]
                win = max(on_prop or windows, key=lambda o: o.width_mm)
                if prop_wall and prop_wall != win.wall_id:
                    self.decisions.append(
                        f"sink moved from {self.g.walls[prop_wall].name if prop_wall in self.g.walls else prop_wall} "
                        f"to {self.g.walls[win.wall_id].name} to sit under {win.id}"
                    )
                return win.wall_id, win
        if prop_wall in self.walls:
            return prop_wall, None
        longest = max(self.walls, key=lambda w: max((b - a for a, b in self.free_base_segments(self.g.walls[w])), default=0))
        return longest, None

    def _sink_options(self) -> list[CatalogItem]:
        prop = self._resolve(self.proposal.sink_aid, (CAT_SINK,), "sink", "base")
        cands = [it for it in self.cat.find((CAT_SINK,), height=700) if it.opening_direction == "Both"]
        if prop is None:
            prop = next((c for c in cands if c.width == 1200), cands[0] if cands else None)
        if prop is None:
            return []
        same = [c for c in cands if canon_eq(c.cabinet_combination, prop.cabinet_combination) and c.aid != prop.aid]
        same.sort(key=lambda c: (abs(c.width - prop.width), -c.width))
        return [prop] + same

    # -- base tier ----------------------------------------------------------------
    def _plan_base(self, corners, assign, sink_wall, sink_window, sink_item, hob_item, spice_item, cyl_item, talls):
        decisions: list[str] = []
        warnings: list[str] = []
        score = 0.0
        anchors: dict[str, list[Placement]] = {w: [] for w in self.walls}
        reserves: dict[str, list[Feature]] = {w: [] for w in self.walls}
        cplans: list[CornerPlan] = []

        # 1. corners: blind corner unit on the primary, 560 mm reserve on the secondary
        for corner, primary in zip(corners, assign):
            if primary is None:
                # No corner cabinet fits: leave the corner square empty on both walls.
                for wid in (corner.wall_a, corner.wall_b):
                    end = corner.other(wid)[0]
                    r = self._corner_strip(wid, end, config.CORNER_DEPTH_MM)
                    reserves[wid].append(Feature(r[0], r[1], "corner_open", corner.wall_a + "|" + corner.wall_b))
                cp = CornerPlan(corner, corner.wall_a, corner.end_a, corner.wall_b, corner.end_b)
                cp.open_tiers.update({"base", "wall", "loft"})
                cplans.append(cp)
                warnings.append(
                    f"corner {self.g.walls[corner.wall_a].name}/{self.g.walls[corner.wall_b].name} left open: "
                    "no blind corner unit fits (doors/openings)"
                )
                score += 500
                continue
            p_end, secondary, s_end = corner.other(primary)
            pw, sw = self.g.walls[primary], self.g.walls[secondary]
            cat = CAT_BCU_L if p_end == "end" else CAT_BCU_R
            bcu = self._default((cat,), width=1050, tier="base", height=700) or self._default((cat,), tier="base", height=700)
            if bcu is None:
                return None
            lo, hi = pw.usable
            start = hi - bcu.width if p_end == "end" else lo
            iv = (start, start + bcu.width)
            if any(overlaps(*iv, f.lo, f.hi) for f in self.static[(primary, "base")]):
                return None
            if any(overlaps(*iv, f.lo, f.hi) for f in reserves[primary]):
                return None
            if any(overlaps(*iv, a.start_mm, a.end_mm) for a in anchors[primary]):
                return None
            pl = Placement(bcu, primary, "base", start, "corner", corner_end=p_end)
            anchors[primary].append(pl)
            depth = config.CORNER_DEPTH_MM
            r = (0.0, sw.inset_start_mm + depth) if s_end == "start" else (sw.length_mm - sw.inset_end_mm - depth, float(sw.length_mm))
            if any(overlaps(*r, a.start_mm, a.end_mm) for a in anchors[secondary]):
                return None
            reserves[secondary].append(Feature(r[0], r[1], "corner", corner.wall_a + "|" + corner.wall_b))
            cplans.append(CornerPlan(corner, primary, p_end, secondary, s_end, base_unit=pl))

        def base_feats(wid):
            return self.static[(wid, "base")] + reserves[wid]

        # 2. sink
        sink = None
        if sink_item is not None and sink_wall:
            wall = self.g.walls[sink_wall]
            if sink_window is not None:
                c = sink_window.center_mm
                for off in [0] + [s * k for k in range(25, 801, 25) for s in (1, -1)]:
                    st = c + off - sink_item.width / 2
                    if self._fits(wall, st, sink_item.width, base_feats(sink_wall), anchors[sink_wall]):
                        sink = Placement(sink_item, sink_wall, "base", st, "sink")
                        if off:
                            # Alignment is mandatory: any narrower sink that centres wins.
                            warnings.append(f"sink offset {off:+.0f} mm from window centre (obstruction)")
                            score += 1000 + abs(off)
                        break
            else:
                gaps = self._gaps(sink_wall, base_feats(sink_wall), anchors[sink_wall])
                target = self.proposal.sink_center_mm
                res = self._search_block(sink_wall, gaps, [(sink_item, "sink")], target, [], hob_rules=False)
                if res:
                    sink = res[1][0]
                    score += res[0]
            if sink is None:
                return None
            anchors[sink_wall].append(sink)
            if self.proposal.sink_aid and sink_item.aid != self.proposal.sink_aid:
                prop = self.cat.get(self.proposal.sink_aid)
                if prop:
                    score += 0.2 * abs(prop.width - sink_item.width)

        # 3+4. tall end-caps, then the hob + spice (+ cylinder) block. The hob is
        # essential; optional items (cylinder, tall units) are dropped one by one
        # if they leave no legal hob position.
        expected_hob_wall = self.proposal.hob_wall_id if self.proposal.hob_wall_id in self.walls else None
        relax = [(False, False)]
        if cyl_item is not None:
            relax.append((True, False))
        if talls:
            relax += [(False, True)] + ([(True, True)] if cyl_item is not None else [])
        for drop_cyl, drop_talls in relax:
            caps: dict[tuple[str, float], list[CatalogItem]] = {}
            cap_warn: list[str] = []
            cap_pen = 0.0
            for t_wall, t_item in ([] if drop_talls else talls):
                choice = self._tall_slot(t_item, t_wall, expected_hob_wall, sink, base_feats, anchors, caps)
                if choice is None:
                    cap_warn.append(f"tall unit {t_item.aid}: no free run end without blocking a window/door; skipped")
                    cap_pen += 60
                    continue
                key, at_lo = choice
                caps.setdefault(key, [])
                if at_lo:
                    caps[key].insert(0, t_item)
                else:
                    caps[key].append(t_item)
            best_hob = self._place_hob(hob_item, spice_item, None if drop_cyl else cyl_item, sink, sink_wall, base_feats, anchors, caps)
            if best_hob is not None:
                break
        warnings.extend(cap_warn)
        score += cap_pen
        if best_hob is not None and drop_cyl:
            warnings.append("cylinder unit dropped: no room beside the hob with chimney/sink clearances")
            score += 80
        if best_hob is not None and drop_talls:
            warnings.append("tall unit(s) dropped: they left no legal hob position")
            score += 80 * len(talls)
        hob = None
        chimney = None
        if best_hob is None:
            warnings.append("hob could not be placed with chimney/landing clearances")
            score += 2000
        else:
            score += best_hob[0]
            wid = best_hob[2]
            anchors[wid].extend(best_hob[1])
            hob = next(p for p in best_hob[1] if p.role == "hob")
            chimney = {"wall_id": wid, "center_mm": hob.center_mm}
            if self.proposal.hob_wall_id and wid != self.proposal.hob_wall_id:
                decisions.append(f"hob moved to {self.g.walls[wid].name} (clearances on proposed wall failed)")

        # 5. fill every remaining gap
        placements: list[Placement] = [a for wl in anchors.values() for a in wl]
        for wid in self.walls:
            prefs = self._base_prefs(wid)
            for gap in self._gaps(wid, base_feats(wid), anchors[wid], caps):
                res = fill_gap(gap, self.base_pool, prefs)
                if res is None:
                    score += 500
                    continue
                score += res.penalty
                del prefs[: res.used_prefs]
                for it, st, role in res.items:
                    tier = "tall" if it.tier == "tall" else "base"
                    placements.append(Placement(it, wid, tier, st, role))
        # Ergonomics penalties
        if hob is not None and not self._has_landing(hob, placements):
            score += 300
        return _BasePlan(score, placements, cplans, reserves, sink, hob, chimney, decisions, warnings)

    def _corner_strip(self, wid: str, end: str, depth_mm: float) -> tuple[float, float]:
        """Span of ``wid`` from its ``end`` that lies in front of the perpendicular wall's cabinets."""
        wall = self.g.walls[wid]
        if end == "start":
            return (0.0, wall.inset_start_mm + depth_mm)
        return (wall.length_mm - wall.inset_end_mm - depth_mm, float(wall.length_mm))

    def _bcu_fits(self, corner: Corner, primary: str) -> bool:
        """Static check: can a blind corner unit sit on ``primary`` at this corner?"""
        p_end, secondary, s_end = corner.other(primary)
        cat = CAT_BCU_L if p_end == "end" else CAT_BCU_R
        bcu = self._default((cat,), width=1050, tier="base", height=700) or self._default((cat,), tier="base", height=700)
        if bcu is None:
            return False
        lo, hi = self.g.walls[primary].usable
        start = hi - bcu.width if p_end == "end" else lo
        if any(overlaps(start, start + bcu.width, f.lo, f.hi) for f in self.static[(primary, "base")]):
            return False
        r = self._corner_strip(secondary, s_end, config.CORNER_DEPTH_MM)
        return not any(overlaps(*r, f.lo, f.hi) for f in self.static[(secondary, "base")])

    def _place_hob(self, hob_item, spice_item, cyl_item, sink, sink_wall, base_feats, anchors, caps):
        """Best (score, placements, wall) for the hob block over all cabinet walls."""
        hob_block = [(hob_item, "hob"), (spice_item, "spice")] if spice_item else [(hob_item, "hob")]
        best = None
        for wid in self.walls:
            gaps = self._gaps(wid, base_feats(wid), anchors[wid], caps)
            target = self.proposal.hob_center_mm if wid == self.proposal.hob_wall_id else None
            bonus = 0.0 if wid == self.proposal.hob_wall_id else 40.0
            if wid == sink_wall and len(self.walls) > 1:
                bonus += 30.0
            prefs = self._base_prefs(wid)
            for spice_side in ("hi", "lo"):
                elements = list(hob_block) if spice_side == "hi" else list(reversed(hob_block))
                if cyl_item is not None:
                    elements = elements + [(cyl_item, "cylinder")] if spice_side == "lo" else [(cyl_item, "cylinder")] + elements
                res = self._search_block(wid, gaps, elements, target, prefs, hob_rules=True, sink=sink)
                if res and (best is None or res[0] + bonus < best[0]):
                    best = (res[0] + bonus, res[1], wid)
        return best

    def _base_prefs(self, wid: str) -> list[CatalogItem]:
        out = []
        for aid in self.proposal.base_prefs.get(wid, []):
            it = self.cat.get(aid)
            if it is not None and it.cat in BASE_STORAGE_CATS and it.tier == "base" and it.width % config.GRID_MM == 0:
                out.append(it)
        return out

    def _wall_prefs(self, wid: str) -> list[CatalogItem]:
        out = []
        for aid in self.proposal.wall_prefs.get(wid, []):
            it = self.cat.get(aid)
            if it is not None and it.cat == CAT_WALL and it.depth == 330 and it.width % config.GRID_MM == 0:
                out.append(it)
        return out

    def _fits(self, wall: Wall, start: float, width: float, feats: list[Feature], anchors: list[Placement]) -> bool:
        lo, hi = wall.usable
        end = start + width
        if start < lo - EPS or end > hi + EPS:
            return False
        if any(overlaps(start, end, f.lo, f.hi) for f in feats):
            return False
        return not any(overlaps(start, end, a.start_mm, a.end_mm) for a in anchors)

    def _gaps(self, wid, feats: list[Feature], anchors: list[Placement], caps=None) -> list[Gap]:
        """Free intervals of the wall's usable span, labelled with what bounds them."""
        wall = self.g.walls[wid]
        lo, hi = wall.usable
        end_kind = "wall" if wall.neighbor_end else "open"
        blocks = sorted(
            [(f.lo, f.hi, f.kind) for f in feats] + [(a.start_mm, a.end_mm, "anchor") for a in anchors],
            key=lambda b: (b[0], b[1]),
        )
        raw: list[tuple[float, float, str, str]] = []
        cur, cur_kind = lo, ("wall" if wall.neighbor_start else "open")
        for b_lo, b_hi, kind in blocks:
            if b_hi <= cur + EPS:
                continue  # already covered
            if b_lo >= hi - EPS:
                break  # beyond the usable span
            if b_lo > cur + EPS:
                raw.append((cur, b_lo, cur_kind, kind))
            cur, cur_kind = max(cur, b_hi), kind
            if cur >= hi - EPS:
                break
        if cur < hi - EPS:
            raw.append((cur, hi, cur_kind, end_kind))
        gaps = []
        for g_lo, g_hi, k_lo, k_hi in raw:
            g = Gap(wid, g_lo, g_hi, k_lo, k_hi)
            if caps:
                g.caps_lo = list(caps.get((wid, round(g.lo, 3)), []))
                g.caps_hi = list(caps.get((wid, round(g.hi, 3)), []))
            gaps.append(g)
        return gaps

    def _tall_slot(self, item, pref_wall, hob_wall, sink, base_feats, anchors, caps):
        best = None
        for wid in self.walls:
            wall = self.g.walls[wid]
            for gap in self._gaps(wid, base_feats(wid), anchors[wid], caps):
                used = sum(i.width for i in gap.caps_lo + gap.caps_hi)
                for at_lo in (True, False):
                    kind = gap.kind_lo if at_lo else gap.kind_hi
                    if kind not in ("wall", "open", "door", "swing"):
                        continue
                    existing = gap.caps_lo if at_lo else gap.caps_hi
                    if existing and len(existing) >= 2:
                        continue
                    remain = gap.length - used - item.width
                    if remain < -EPS:
                        continue
                    if 0 < remain < config.MIN_UNIT_WIDTH_MM and kind not in SOFT_KINDS:
                        continue
                    off = sum(i.width for i in existing)
                    st = gap.lo + off if at_lo else gap.hi - off - item.width
                    iv = (st, st + item.width)
                    if any(overlaps(*iv, f.lo, f.hi) for f in self.static[(wid, "wall")]):
                        continue  # window / door above
                    s = 0.0
                    s += 500 if wid == pref_wall else 0
                    s -= 300 if wid == hob_wall else 0
                    s -= 200 if (sink and wid == sink.wall_id) else 0
                    if sink and wid == sink.wall_id:
                        s += abs((st + item.width / 2) - sink.center_mm) / 10.0
                    s += 100 if kind == "wall" else 0
                    s += min(remain, 2400) / 20.0
                    if best is None or s > best[0]:
                        best = (s, (wid, round(gap.lo if at_lo else gap.hi, 3)), at_lo)
        return None if best is None else (best[1], best[2])

    def _chimney_ok(self, wid: str, center: float) -> bool:
        wall = self.g.walls[wid]
        half = config.CHIMNEY_WIDTH_MM / 2 + config.CHIMNEY_SIDE_CLEARANCE_MM
        lo, hi = center - half, center + half
        ulo, uhi = wall.usable
        if lo < ulo - EPS or hi > uhi + EPS:
            return False
        if any(overlaps(lo, hi, f.lo, f.hi) for f in self.static[(wid, "wall")]):
            return False
        for c in self.g.corners_of(wid):
            my_end, other, _ = c.other(wid)
            if other not in self.walls:
                continue
            zone = config.WALL_CORNER_DEPTH_MM
            r = (0.0, wall.inset_start_mm + zone) if my_end == "start" else (wall.length_mm - wall.inset_end_mm - zone, wall.length_mm)
            if overlaps(lo, hi, *r):
                return False
        return True

    def _search_block(self, wid, gaps, elements, target, prefs, hob_rules, sink=None):
        """Best position of a contiguous block of anchor cabinets inside ``gaps``."""
        block_w = sum(it.width for it, _ in elements)
        best = None
        for gap in gaps:
            cap_lo = sum(i.width for i in gap.caps_lo)
            cap_hi = sum(i.width for i in gap.caps_hi)
            lo, hi = gap.lo + cap_lo, gap.hi - cap_hi - block_w
            if hi < lo - EPS:
                continue
            g = config.GRID_MM
            starts = {round(lo, 3), round(hi, 3)}
            k = 1
            while lo + k * g <= hi + EPS and k <= 80:
                starts.add(round(lo + k * g, 3))
                starts.add(round(hi - k * g, 3))
                k += 1
            tgt = target if target is not None else (gap.lo + gap.hi) / 2
            for st in sorted(starts):
                if st < lo - EPS or st > hi + EPS:
                    continue
                pls, x = [], st
                for it, role in elements:
                    pls.append(Placement(it, wid, "base", x, role))
                    x += it.width
                main = pls[0] if not hob_rules else next(p for p in pls if p.role == "hob")
                if hob_rules:
                    if not self._chimney_ok(wid, main.center_mm):
                        continue
                    half = config.CHIMNEY_WIDTH_MM / 2 + config.CHIMNEY_SIDE_CLEARANCE_MM
                    c_lo, c_hi = main.center_mm - half, main.center_mm + half
                    # tall end-caps may float by < 1 module; keep the hood clear of them
                    if cap_lo and c_lo < gap.lo + cap_lo + g:
                        continue
                    if cap_hi and c_hi > gap.hi - cap_hi - g:
                        continue
                    if sink is not None and sink.wall_id == wid:
                        d = max(sink.start_mm - main.end_mm, main.start_mm - sink.end_mm)
                        if d < config.MIN_SINK_HOB_GAP_MM - EPS:
                            continue
                left = Gap(wid, gap.lo, st, gap.kind_lo, "anchor", gap.caps_lo, [])
                right = Gap(wid, st + block_w, gap.hi, "anchor", gap.kind_hi, [], gap.caps_hi)
                pen = 0.0
                ok = True
                fills = []
                pcopy = list(prefs)
                for sub in (left, right):
                    if sub.length < -EPS:
                        ok = False
                        break
                    r = fill_gap(sub, self.base_pool, pcopy)
                    if r is None:
                        ok = False
                        break
                    del pcopy[: r.used_prefs]
                    pen += r.penalty
                    fills.append(r)
                if not ok:
                    continue
                if hob_rules:
                    # landing: hob side without spice/cylinder needs a storage unit
                    hob_idx = next(i for i, p in enumerate(pls) if p.role == "hob")
                    open_side = "lo" if hob_idx == 0 else ("hi" if hob_idx == len(pls) - 1 else None)
                    if open_side is not None:
                        r = fills[0] if open_side == "lo" else fills[1]
                        units = [it for it, _, role in r.items if role == "storage"]
                        if not units or units[0 if open_side == "hi" else -1].width < config.MIN_LANDING_MM:
                            pen += 400
                pen += 0.01 * abs(main.center_mm - tgt)
                if best is None or pen < best[0] - 1e-9:
                    best = (pen, pls)
        return best

    def _has_landing(self, hob: Placement, placements: list[Placement]) -> bool:
        same = [p for p in placements if p.wall_id == hob.wall_id and "base" in p.zones and p is not hob]
        left = any(abs(p.end_mm - hob.start_mm) <= config.GRID_MM and p.tier == "base" for p in same)
        right = any(abs(p.start_mm - hob.end_mm) <= config.GRID_MM and p.tier == "base" for p in same)
        return left and right

    # -- wall tier ----------------------------------------------------------------
    def _place_corner_tier(self, layout: KitchenLayout, cp: CornerPlan, zone: str) -> None:
        """Corner rule for the wall / loft tier: a corner cabinet, or the corner stays empty.

        Regular cabinets are never placed where two runs meet. Wall tier tries a
        Wall L Corner (650x650) on either wall, then a wall blind corner unit
        (730/830, L or R by wall end). Loft tier uses a Loft L Corner above a
        Wall L Corner. Otherwise both walls keep the corner square clear.
        """
        walls = self.g.walls
        depth_reg = config.WALL_UNIT_DEPTH_MM
        a, a_end, b, b_end = cp.primary, cp.primary_end, cp.secondary, cp.secondary_end
        tag = f"{walls[a].name}/{walls[b].name}"

        def blocked(wid, iv):
            if any(f.kind in BLOCKING_KINDS + ("tall",) and overlaps(*iv, f.lo, f.hi) for f in layout.features[(wid, zone)]):
                return True
            return any(p.wall_id == wid and zone in p.zones and overlaps(*iv, p.start_mm, p.end_mm) for p in layout.placements)

        def leave_open(reason):
            for wid, end in ((a, a_end), (b, b_end)):
                r = self._corner_strip(wid, end, depth_reg)
                layout.features[(wid, zone)].append(Feature(r[0], r[1], "corner_open", tag))
            cp.open_tiers.add(zone)
            layout.decisions.append(f"corner {tag}: {zone} tier left open ({reason}) - no regular cabinet in the corner")

        if "base" in cp.open_tiers:
            return leave_open("base corner is open")
        if blocked(a, self._corner_strip(a, a_end, depth_reg)) or blocked(b, self._corner_strip(b, b_end, depth_reg)):
            return leave_open("window/door/chimney at the corner")

        candidates: list[tuple[str, object]] = []
        order = [(a, a_end, b, b_end), (b, b_end, a, a_end)]
        if zone == "wall":
            lc = self._default((CAT_WALL_CORNER,), tier="wall", height=600)
            if lc is not None:
                candidates.append(("L", lc))
            candidates += [("blind", 730), ("blind", 830)]
        elif cp.wall_unit is not None and cp.wall_unit.item.cat == CAT_WALL_CORNER:
            w = cp.wall_unit
            lc = self._default((CAT_LOFT_CORNER,), tier="loft", height=600, width=w.item.width) or self._default(
                (CAT_LOFT_CORNER,), tier="loft", height=600
            )
            if lc is not None:
                candidates.append(("L", lc))
            order = [o for o in order if o[0] == w.wall_id]  # stack above the wall L corner
        for kind, spec in candidates:
            for x, x_end, y, y_end in order:
                if kind == "L":
                    item, reserve = spec, spec.depth
                else:
                    cat = CAT_BCU_L if x_end == "end" else CAT_BCU_R
                    item, reserve = self._default((cat,), width=spec, tier="wall", height=600), depth_reg
                    if item is None:
                        continue
                lo, hi = walls[x].usable
                st = hi - item.width if x_end == "end" else lo
                r = self._corner_strip(y, y_end, reserve)
                if blocked(x, (st, st + item.width)) or blocked(y, r):
                    continue
                role = "wall_corner" if zone == "wall" else "loft_corner"
                pl = Placement(item, x, zone, st, role, corner_end=x_end)
                layout.placements.append(pl)
                layout.features[(y, zone)].append(Feature(r[0], r[1], "corner", tag))
                if zone == "wall":
                    cp.wall_unit, cp.wall_reserve_mm = pl, reserve
                else:
                    cp.loft_unit = pl
                if kind == "blind":
                    layout.decisions.append(f"corner {tag}: Wall L Corner blocked; wall blind corner {item.aid} used")
                return
        leave_open("no corner cabinet fits")

    def _plan_wall_tier(self, layout: KitchenLayout) -> None:
        for cp in layout.corner_plans:
            self._place_corner_tier(layout, cp, "wall")

        for wid in self.walls:
            cover = self._coverage(layout, wid, "base", extra_corner=True)
            if layout.chimney and layout.chimney["wall_id"] == wid:
                f = next(f for f in layout.features[(wid, "wall")] if f.kind == "chimney")
                cover = merge_intervals(cover + [(f.lo, f.hi)], join=config.MAX_FILLER_MM + 10)
            self._fill_tier(layout, wid, "wall", cover, self.wall_pool, self._wall_prefs(wid), "wall")

    def _coverage(self, layout, wid, zone, extra_corner=False) -> list[tuple[float, float]]:
        iv = [(p.start_mm, p.end_mm) for p in layout.placements if p.wall_id == wid and zone in p.zones]
        if extra_corner:
            iv += [(f.lo, f.hi) for f in layout.features[(wid, "base")] if f.kind == "corner"]
        return merge_intervals(iv, join=config.MAX_FILLER_MM + 10)

    def _fill_tier(self, layout, wid, zone, cover, pool: Pool, prefs, role, tier=None) -> None:
        tier = tier or zone
        wall = self.g.walls[wid]
        feats = layout.features[(wid, zone)]
        anchors = [p for p in layout.placements if p.wall_id == wid and zone in p.zones]
        for c_lo, c_hi in cover:
            for gap in self._gaps(wid, feats, anchors):
                lo, hi = max(gap.lo, c_lo), min(gap.hi, c_hi)
                if hi - lo < pool.min_width - EPS:
                    continue
                k_lo = gap.kind_lo if abs(lo - gap.lo) <= EPS else "open"
                k_hi = gap.kind_hi if abs(hi - gap.hi) <= EPS else "open"
                res = fill_gap(Gap(wid, lo, hi, k_lo, k_hi), pool, prefs, role)
                if res is None:
                    continue
                del prefs[: res.used_prefs]
                for it, st, r in res.items:
                    layout.placements.append(Placement(it, wid, tier, st, r))
                if res.leftover > EPS and res.leftover_at != "none":
                    kinds = k_lo if res.leftover_at == "lo" else k_hi
                    if kinds not in SOFT_KINDS and res.leftover > config.MAX_FILLER_MM:
                        layout.warnings.append(f"{wall.name} {zone}: {res.leftover:.0f} mm gap left unfilled")

    # -- loft tier ------------------------------------------------------------------
    def _plan_loft(self, layout: KitchenLayout) -> None:
        for wid in self.walls:
            wall = self.g.walls[wid]
            if wall.height_m + 1e-6 < config.LOFT_TOP_M:
                layout.decisions.append(
                    f"{wall.name}: no loft (wall height {wall.height_m:.3f} m < {config.LOFT_TOP_M:.3f} m)"
                )
            else:
                layout.loft_walls.append(wid)
        for cp in layout.corner_plans:
            on = [w for w in (cp.primary, cp.secondary) if w in layout.loft_walls]
            if len(on) == 2:
                self._place_corner_tier(layout, cp, "loft")
            elif on:  # only one wall is tall enough: its lofts still stop short of the corner
                wid = on[0]
                r = self._corner_strip(wid, cp.end_of(wid), config.WALL_UNIT_DEPTH_MM)
                layout.features[(wid, "loft")].append(Feature(r[0], r[1], "corner_open", "loft-corner"))
                cp.open_tiers.add("loft")
        for wid in layout.loft_walls:
            # Deep lofts above tall units (same width, 560 deep).
            for t in [p for p in layout.placements if p.wall_id == wid and p.tier == "tall"]:
                deep = self.loft_deep_pool.by_width.get(t.item.width)
                if deep and not any(overlaps(t.start_mm, t.end_mm, f.lo, f.hi) for f in layout.features[(wid, "loft")]):
                    layout.placements.append(Placement(deep, wid, "loft", t.start_mm, "loft_deep"))
            iv = [(p.start_mm, p.end_mm) for p in layout.placements if p.wall_id == wid and p.tier in ("wall", "tall")]
            iv += [(f.lo, f.hi) for f in layout.features[(wid, "wall")] if f.kind == "chimney"]
            cover = merge_intervals(iv, join=config.MAX_FILLER_MM + 10)
            self._fill_tier(layout, wid, "loft", cover, self.loft_pool, [], "loft")

    # -- door swing direction -----------------------------------------------------
    def _apply_hinges(self, layout: KitchenLayout) -> None:
        for i, p in enumerate(layout.placements):
            if not p.item.is_single_shutter or p.role in ("corner", "wall_corner", "loft_corner", "sink", "hob"):
                continue
            sides = {}
            for side, direction, x in (("start", -1, p.start_mm), ("end", 1, p.end_mm)):
                kinds = set()
                for zone in p.zones:
                    b = layout.boundary(p.wall_id, zone, x, direction, exclude=[p])
                    if b is None:
                        continue
                    kind, _, dist = b
                    if kind in ("wall", "corner") and dist <= config.MAX_FILLER_MM:
                        kinds.add(kind)
                    elif kind in ("door", "swing") and dist <= config.DOOR_CLEARANCE_MM + config.MAX_FILLER_MM:
                        kinds.add(kind)
                    elif kind == "tall" and dist <= config.GRID_MM and p.tier != "tall":
                        kinds.add(kind)
                    elif kind == "chimney" and dist <= config.MAX_FILLER_MM:
                        kinds.add("chimney")
                sides[side] = kinds
            strong = {s for s, k in sides.items() if k & STRONG_HINGE_OBSTACLES}
            weak = {s for s, k in sides.items() if "chimney" in k}
            hinge = None
            if len(strong) == 1:
                hinge = next(iter(strong))
            elif not strong and len(weak) == 1:
                hinge = next(iter(weak))
            if hinge is None:
                continue
            want = next(o for o, s in config.HINGE_SIDE_FOR_OPENING.items() if s == hinge)
            if p.item.opening_direction == want:
                continue
            v = self.cat.variant(p.item, want)
            if v is None:
                layout.warnings.append(f"no '{want}' variant for {p.item.aid}; hinge left as {p.item.opening_direction}")
                continue
            layout.placements[i] = Placement(v, p.wall_id, p.tier, p.start_mm, p.role, p.corner_end, p.note)
            layout.decisions.append(
                f"{self.g.walls[p.wall_id].name}: {p.item.category} {p.item.width} hinged {want} "
                f"(towards {'/'.join(sorted(sides[hinge]))}) -> {v.aid}"
            )

    # -- validation & repair ------------------------------------------------------
    def _repair(self, layout: KitchenLayout) -> None:
        for _ in range(50):
            issues = validate_layout(layout)
            if not issues:
                return
            removable = [(msg, p) for msg, p in issues if p is not None and p.role in ("storage", "wall", "loft", "loft_deep")]
            if not removable:
                for msg, _ in issues:
                    layout.warnings.append("unresolved: " + msg)
                return
            msg, p = removable[0]
            layout.placements.remove(p)
            layout.warnings.append(f"removed {p.label()} on {self.g.walls[p.wall_id].name}: {msg}")


def canon_eq(a: Optional[str], b: Optional[str]) -> bool:
    from catalog_loader import canon

    return canon(a) == canon(b)


def _subtract(base: list[tuple[float, float]], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = list(base)
    for c_lo, c_hi in cuts:
        nxt = []
        for lo, hi in out:
            if c_hi <= lo or c_lo >= hi:
                nxt.append((lo, hi))
                continue
            if c_lo > lo:
                nxt.append((lo, c_lo))
            if c_hi < hi:
                nxt.append((c_hi, hi))
        out = nxt
    return out


# ---------------------------------------------------------------------------
# Validation (also used by tests)
# ---------------------------------------------------------------------------

def validate_layout(layout: KitchenLayout) -> list[tuple[str, Optional[Placement]]]:
    """Return ``[(message, offending_placement_or_None), ...]``; empty = valid."""
    g = layout.geometry
    issues: list[tuple[str, Optional[Placement]]] = []
    for p in layout.placements:
        wall = g.walls[p.wall_id]
        lo, hi = wall.usable
        if p.start_mm < lo - EPS or p.end_mm > hi + EPS:
            issues.append((f"{p.label()} outside wall span [{lo:.0f}, {hi:.0f}]", p))
        for zone in p.zones:
            for f in layout.features.get((p.wall_id, zone), []):
                if f.kind in ("door", "swing", "window", "chimney") and overlaps(p.start_mm, p.end_mm, f.lo, f.hi):
                    issues.append((f"{p.label()} intrudes into {f.kind} zone {f.ref} [{f.lo:.0f}, {f.hi:.0f}]", p))
                if f.kind in ("corner", "corner_open") and overlaps(p.start_mm, p.end_mm, f.lo, f.hi):
                    issues.append((f"{p.label()} intrudes into corner reserve [{f.lo:.0f}, {f.hi:.0f}]", p))
    # corner rule: only corner cabinets where two runs meet (checked per tier)
    for cp in layout.corner_plans:
        for zone, depth in (("base", config.CORNER_DEPTH_MM), ("wall", config.WALL_UNIT_DEPTH_MM), ("loft", config.WALL_UNIT_DEPTH_MM)):
            for wid in (cp.primary, cp.secondary):
                wall = g.walls[wid]
                if cp.end_of(wid) == "start":
                    strip = (0.0, wall.inset_start_mm + depth)
                else:
                    strip = (wall.length_mm - wall.inset_end_mm - depth, float(wall.length_mm))
                for p in layout.on(wid, zone):
                    if overlaps(p.start_mm, p.end_mm, *strip) and p.item.cat not in CORNER_CATS:
                        issues.append((f"regular cabinet {p.label()} in the {g.walls[cp.primary].name}/"
                                       f"{g.walls[cp.secondary].name} corner ({zone} tier)", p))
    # same wall / zone overlaps
    for wid in layout.cabinet_walls:
        for zone in ("base", "wall", "loft"):
            ps = layout.on(wid, zone)
            for a, b in zip(ps, ps[1:]):
                if overlaps(a.start_mm, a.end_mm, b.start_mm, b.end_mm):
                    victim = b if b.role in ("storage", "wall", "loft", "loft_deep") else a
                    issues.append((f"{a.label()} overlaps {b.label()} on {zone} tier", victim))
    # world-space 3D collisions (corners between walls)
    boxes = [(p, *placement_box(p, g.walls[p.wall_id])) for p in layout.placements]
    for i in range(len(boxes)):
        pa, polya, ya = boxes[i]
        for j in range(i + 1, len(boxes)):
            pb, polyb, yb = boxes[j]
            if pa.wall_id == pb.wall_id:
                continue
            if ya[0] >= yb[1] - 1e-4 or yb[0] >= ya[1] - 1e-4:
                continue
            if convex_overlap(polya, polyb):
                victim = pb if pb.role in ("storage", "wall", "loft", "loft_deep") else pa
                issues.append((f"3D collision {pa.label()} x {pb.label()}", victim))
    # chimney vs wall cabinets (3D)
    if layout.chimney:
        wall = g.walls[layout.chimney["wall_id"]]
        c = layout.chimney["center_mm"]
        half = config.CHIMNEY_WIDTH_MM / 2
        for p in layout.on(wall.id, "wall"):
            if overlaps(p.start_mm, p.end_mm, c - half, c + half):
                issues.append((f"{p.label()} obstructs chimney", p))
    if layout.sink is not None and layout.sink_window is not None:
        if abs(layout.sink.center_mm - layout.sink_window.center_mm) > EPS:
            issues.append((f"sink not centred under {layout.sink_window.id}", None))
    return issues


# ---------------------------------------------------------------------------
# Wall selection heuristics (used by the fallback designer and as a safety net)
# ---------------------------------------------------------------------------

def suggest_cabinet_walls(g: RoomGeometry, layout: str = "auto") -> list[str]:
    def free_len(wid):
        wall = g.walls[wid]
        lo, hi = wall.usable
        cuts = [d.interval(config.DOOR_CLEARANCE_MM) for d in wall.doors]
        return max((b - a for a, b in _subtract([(lo, hi)], cuts)), default=0.0)

    cands = [w for w in g.wall_order if free_len(w) >= config.MIN_CABINET_WALL_MM]
    if not cands:
        raise SolverError("no wall has enough free length for a kitchen run")
    window_walls = [w for w in cands if g.walls[w].windows]
    anchor = max(window_walls or cands, key=free_len)
    neighbors = sorted(
        [w for w in cands if w != anchor and g.corner_between(anchor, w)], key=free_len, reverse=True
    )
    layout = (layout or "auto").upper() if layout and layout.lower() in ("l", "u") else (layout or "auto").lower()
    if layout == "single" or not neighbors:
        return [anchor]
    if layout == "L":
        return [anchor, neighbors[0]]
    if layout == "U":
        return [neighbors[0], anchor] + neighbors[1:2]
    # auto: U when both legs exist and the aisle between them is comfortable
    aisle = g.walls[anchor].usable[1] - g.walls[anchor].usable[0] - 2 * config.CORNER_DEPTH_MM
    if len(neighbors) >= 2 and aisle >= config.MIN_AISLE_MM and all(free_len(n) >= 1800 for n in neighbors[:2]):
        return [neighbors[0], anchor, neighbors[1]]
    return [anchor, neighbors[0]]


def solve_layout(geometry: RoomGeometry, catalog: Catalog, proposal: Proposal, prefs: Optional[SolverPrefs] = None) -> KitchenLayout:
    return LayoutSolver(geometry, catalog, proposal, prefs or SolverPrefs()).solve()
