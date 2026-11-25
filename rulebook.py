"""MagickHome Kitchen Rule Engine V1 (the rule workbook) applied by the generator.

Source: ``knowledge/Kitchen_Rule_Engine_V1.xlsx``, sheet ``Rules`` (81 active rules). Every rule
this module applies is cited by its workbook row number ``R<S.No>`` so explanations can say
which rule decided something.

* Tabular rules are read from the workbook at runtime, so an updated workbook takes effect
  without code changes: the module widths by kitchen type and size band (``SET_VALUE width``
  rules with no condition, e.g. R1-R5, R12-R21, R61-R69).
* Rules that are logic rather than a table are implemented here and cite their rows:
  size bands from the floor area, kitchen type from the cabinet walls, the budget-tier fill
  rules (R6-R10, R31-R33, R38-R43), the hob step-down (R55-R57), lofts (R47-R54) and corners
  by budget (R71-R75).

If the workbook is missing or unreadable a built-in copy of the module-width table is used.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import config
from catalog_loader import CAT_BCU_L, CAT_BCU_R, CAT_CYLINDER, CAT_HOB, CAT_SINK, CAT_SPICE, canon

log = logging.getLogger(__name__)

SQFT_PER_M2 = 10.7639
# Size bands (R*: Size Category sheet): Small 60-80, Medium 81-120, Large 121-180 sqft.
SIZE_BANDS = (("Small", 60, 80), ("Medium", 81, 120), ("Large", 121, 180))
BUDGETS = ("essential", "premium", "luxury")

# Workbook trigger object -> catalogue category (canonical) for the module-width rules.
TRIGGER_CATS = {
    "base hob unit": (CAT_HOB,),
    "base sink unit": (CAT_SINK,),
    "base spice pull out": (CAT_SPICE,),
    "base cylinder unit": (CAT_CYLINDER,),
    "blind corner unit - l/r": (CAT_BCU_L, CAT_BCU_R),
}
ROLE_OF_TRIGGER = {"base hob unit": "hob", "base sink unit": "sink", "base spice pull out": "spice",
                   "base cylinder unit": "cylinder", "blind corner unit - l/r": "corner"}

# Built-in copy of the module-width rules (used only when the workbook cannot be read).
_BUILTIN_WIDTHS = [
    # (rule, trigger, kitchen_type, size, width, combination)
    (1, "base hob unit", "L-Shape", "Small", 600, "Triple Drawer - A"),
    (2, "base sink unit", "L-Shape", "Small", 600, "Sink Single Shutter"),
    (3, "base spice pull out", "L-Shape", "Small", 150, "Spice Pullout"),
    (4, "base cylinder unit", "L-Shape", "Small", 450, "Cylinder Single Shutter"),
    (5, "blind corner unit - l/r", "L-Shape", "Small", 1050, None),
    (12, "base hob unit", "L-Shape", "Medium", 800, "Triple Drawer - A"),
    (13, "base sink unit", "L-Shape", "Medium", 900, "Sink Double Shutter"),
    (14, "base spice pull out", "L-Shape", "Medium", 200, "Spice Pullout"),
    (15, "base cylinder unit", "L-Shape", "Medium", 500, "Cylinder Single Shutter"),
    (16, "blind corner unit - l/r", "L-Shape", "Medium", 1050, None),
    (17, "base hob unit", "L-Shape", "Large", 900, "Triple Drawer - A"),
    (18, "base sink unit", "L-Shape", "Large", 1200, "Sink Double Shutter"),
    (19, "base spice pull out", "L-Shape", "Large", 300, "Spice Pullout"),
    (20, "base cylinder unit", "L-Shape", "Large", 600, "Cylinder Single Shutter"),
    (21, "blind corner unit - l/r", "L-Shape", "Large", 1050, None),
    (61, "base hob unit", "Straight", "Small", 600, "Triple Drawer - A"),
    (62, "base hob unit", "Parallel", "Small", 750, "Triple Drawer - A"),
    (63, "base hob unit", "U-Shape", "Small", 600, "Triple Drawer - A"),
    (64, "base hob unit", "Straight", "Medium", 800, "Triple Drawer - A"),
    (65, "base hob unit", "Parallel", "Medium", 800, "Triple Drawer - A"),
    (66, "base hob unit", "U-Shape", "Medium", 800, "Triple Drawer - A"),
    (67, "base hob unit", "Straight", "Large", 900, "Triple Drawer - A"),
    (68, "base hob unit", "Parallel", "Large", 1050, "Triple Drawer - A"),
    (69, "base hob unit", "U-Shape", "Large", 900, "Triple Drawer - A"),
]


@dataclass(frozen=True)
class WidthRule:
    rule: int
    trigger: str          # canonical workbook trigger, e.g. "base hob unit"
    kitchen_type: str     # "L-Shape" | "Straight" | "Parallel" | "U-Shape" | "All"
    size: str             # "Small" | "Medium" | "Large" | "All"
    width: int
    combination: Optional[str]
    text: str = ""

    @property
    def role(self) -> str:
        return ROLE_OF_TRIGGER.get(self.trigger, self.trigger)


@dataclass
class Rulebook:
    source: str
    widths: list[WidthRule]
    texts: dict[int, str] = field(default_factory=dict)  # rule number -> workbook description
    groups: dict[str, list[int]] = field(default_factory=dict)  # "Category > Subcategory" -> rule numbers

    def prompt_lines(self) -> list[str]:
        """The workbook's active rules for the AI designer's prompt, grouped as in the sheet."""
        out = []
        for group, nos in self.groups.items():
            out.append(f"{group}:")
            seen = set()
            for n in nos:
                t = " ".join((self.texts.get(n) or "").split())
                if t and t not in seen:
                    seen.add(t)
                    out.append(f"  R{n}: {t}")
        return out

    def cite(self, rule: int) -> str:
        """``R12: <workbook description>`` for explanations."""
        t = self.texts.get(rule, "")
        return f"R{rule}" + (f": {t}" if t else "")

    def module_width(self, role: str, kitchen_type: str, size: str) -> Optional[WidthRule]:
        """Most specific width rule for ``role`` (hob / sink / spice / cylinder / corner)."""
        cands = [r for r in self.widths if r.role == role
                 and r.kitchen_type in (kitchen_type, "All") and r.size in (size, "All")]
        if not cands:
            return None
        return max(cands, key=lambda r: ((r.kitchen_type == kitchen_type) * 2 + (r.size == size), r.rule))


def _clean(x):
    if x is None:
        return None
    s = str(x).strip()
    return None if s in ("", "None") else s


@lru_cache(maxsize=1)
def load() -> Rulebook:
    """Read the workbook once per process (built-in table if it cannot be read)."""
    path = config.RULEBOOK_PATH
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(wb["Rules"].iter_rows(values_only=True))
        hdr = [str(h) for h in rows[0]]

        def v(r, name):
            return _clean(r[hdr.index(name)]) if name in hdr else None

        widths, texts, groups = [], {}, {}
        for r in rows[1:]:
            no = v(r, "S.No")
            if no is None or (v(r, "status") or "").lower() != "active":
                continue
            no = int(float(no))
            texts[no] = v(r, "description") or ""
            groups.setdefault(f"{v(r, 'category') or 'Other'} > {v(r, 'subcategory') or 'General'}", []).append(no)
            unconditional = not any(v(r, f"when_fact_1_{p}") for p in ("subtype", "family", "property"))
            if (v(r, "action") == "SET_VALUE" and (v(r, "action_target_property") or "").lower() == "width"
                    and unconditional and v(r, "action_min")):
                trig = canon(v(r, "trigger_object")).replace("–", "-")
                if trig not in TRIGGER_CATS:
                    continue
                combo = v(r, "mfg_subcategory")
                widths.append(WidthRule(no, trig, v(r, "kitchen_type") or "All", v(r, "size_category") or "All",
                                        int(float(v(r, "action_min"))),
                                        None if trig.startswith("blind") else combo, texts[no]))
        if not widths:
            raise ValueError("no module-width rules found")
        log.info("rule workbook: %d rules, %d module-width rules from %s", len(texts), len(widths), path.name)
        return Rulebook(f"workbook {path.name}", widths, texts, groups)
    except Exception as exc:  # missing file, renamed sheet, ... -> built-in copy
        log.warning("rule workbook unavailable (%s) - using the built-in module-width table", exc)
        return Rulebook("built-in table", [WidthRule(*row) for row in _BUILTIN_WIDTHS])


# ---------------------------------------------------------------------------
# Facts derived from the floor plan
# ---------------------------------------------------------------------------

def floor_area_sqft(geometry) -> float:
    """Interior floor area: the room outline (wall centre-lines) less half a wall all round."""
    pts = geometry.outline
    if len(pts) < 3:
        return 0.0
    a = 0.5 * abs(sum(x0 * z1 - x1 * z0 for (x0, z0), (x1, z1) in zip(pts, pts[1:] + pts[:1])))
    perim = sum(math.dist(p, q) for p, q in zip(pts, pts[1:] + pts[:1]))
    t = sum(w.thickness_m for w in geometry.walls.values()) / max(1, len(geometry.walls))
    return max(0.0, a - perim * t / 2) * SQFT_PER_M2


def size_band(sqft: float) -> tuple[str, str]:
    """(band, note). Outside the workbook's 60-180 sqft range the nearest band is used."""
    if sqft < 81:
        return "Small", ("below the rule book's 60 sqft minimum - Small rules used" if sqft < 60 else "")
    if sqft < 121:
        return "Medium", ""
    return "Large", ("above the rule book's 180 sqft maximum - Large rules used" if sqft > 180 else "")


def kitchen_type(geometry, walls: list[str]) -> str:
    """Workbook kitchen type from the cabinet walls: Straight, L-Shape, U-Shape or Parallel."""
    if len(walls) <= 1:
        return "Straight"
    if len(walls) == 2:
        return "L-Shape" if geometry.corner_between(walls[0], walls[1]) else "Parallel"
    return "U-Shape"


def working_aisle_mm(geometry, wid: str, cabinet_walls: list[str]) -> Optional[float]:
    """Clear floor in front of ``wid``'s counter: distance to the facing wall less both runs'
    depths (600 mm counter; the facing run only if it carries cabinets). None = no facing wall."""
    w = geometry.walls[wid]
    mid = w.to_world(w.length_m / 2, 0.0)
    best = None
    for oid, o in geometry.walls.items():
        if oid == wid or w.direction[0] * o.direction[0] + w.direction[1] * o.direction[1] > -0.9:
            continue
        d = (o.start[0] - mid[0]) * w.normal[0] + (o.start[1] - mid[1]) * w.normal[1]
        if d <= 0:
            continue
        clear = d * 1000 - (w.thickness_m + o.thickness_m) * 500 - 600 - (600 if oid in cabinet_walls else 0)
        best = clear if best is None else min(best, clear)
    return best


@dataclass
class RuleFacts:
    """What the rule book decided for one kitchen (reported and explained)."""
    area_sqft: float
    band: str
    band_note: str
    kitchen_type: str
    source: str
    applied: list[dict] = field(default_factory=list)  # one entry per rule applied to the design

    def for_role(self, role: str) -> Optional[dict]:
        return next((a for a in self.applied if a["role"] == role), None)

    def to_dict(self) -> dict:
        return {"floor_area_sqft": round(self.area_sqft, 1), "size_band": self.band, "size_note": self.band_note,
                "kitchen_type": self.kitchen_type, "source": self.source, "applied": self.applied}


_ROLE_SLOTS = (  # role, Proposal attribute, catalogue categories, carcass height
    ("hob", "hob_aid", (CAT_HOB,), 700),
    ("sink", "sink_aid", (CAT_SINK,), 700),
    ("spice", "spice_aid", (CAT_SPICE,), 700),
    ("cylinder", "cylinder_aid", (CAT_CYLINDER,), 800),
)


def apply_module_rules(proposal, geometry, catalog, prefs) -> RuleFacts:
    """HARD rules on the designer's proposal, before the layout is solved:

    * R11 - the LPG cylinder unit is included unless the customer has piped gas / induction;
    * module widths by kitchen type and size band (R1-R5, R12-R21, R61-R69): hob, sink,
      spice pull-out and cylinder unit get the rule-book width and combination.
    The designer's own choice is kept in ``applied[].designer`` for the explanation.
    """
    rb = load()
    walls = list(prefs.cabinet_walls or proposal.cabinet_walls)
    area = floor_area_sqft(geometry)
    band, note = size_band(area)
    chosen = (getattr(prefs, "size_band", "auto") or "auto").title()
    if chosen != "Auto":  # the customer / designer overrides the band read from the floor area
        if chosen != band:
            note = f"set to {chosen} by the customer (the floor area gives {band})"
        band = chosen
    facts = RuleFacts(area, band, note, kitchen_type(geometry, walls), rb.source)

    def item_for(rule: WidthRule, cats, height, like=None):
        items = catalog.find(cats, width=rule.width, combination=rule.combination, height=height) if rule.combination else []
        items = items or catalog.find(cats, width=rule.width, height=height)
        if like is not None:  # keep the designer's opening side where the same unit exists
            items.sort(key=lambda i: i.opening_direction != like.opening_direction)
        return items[0] if items else None

    # R11: LPG is assumed when the customer has not said piped gas / induction.
    if prefs.include_cylinder is not False and proposal.cylinder_aid is None:
        r = rb.module_width("cylinder", facts.kitchen_type, band)
        it = (item_for(r, (CAT_CYLINDER,), 800) if r else None) or next(
            iter(catalog.find((CAT_CYLINDER,), width=600, height=800)), None)
        if it is not None:
            proposal.cylinder_aid = it.aid
            facts.applied.append({"role": "cylinder_included", "rule": 11, "text": rb.texts.get(11, ""),
                                  "designer": "no cylinder unit", "built": f"{it.category} {it.width} mm"})
    for role, attr, cats, height in _ROLE_SLOTS:
        cur_aid = getattr(proposal, attr)
        if cur_aid is None:  # not in this kitchen (no spice / piped gas): nothing to size
            continue
        r = rb.module_width(role, facts.kitchen_type, band)
        if r is None:
            continue
        cur = catalog.get(cur_aid)
        it = item_for(r, cats, height, like=cur)
        if it is None:
            facts.applied.append({"role": role, "rule": r.rule, "text": r.text, "width": r.width,
                                  "missing": f"no {r.width} mm {r.combination or ''} unit in the catalogue - designer's unit kept"})
            continue
        changed = cur is None or cur.width != it.width or canon(cur.cabinet_combination) != canon(it.cabinet_combination)
        facts.applied.append({"role": role, "rule": r.rule, "text": r.text, "width": r.width,
                              "combination": it.cabinet_combination, "changed": changed,
                              "designer": f"{cur.width} mm {cur.cabinet_combination}" if cur and changed else None})
        setattr(proposal, attr, it.aid)
    return facts


# ---------------------------------------------------------------------------
# What the UI offers: the filters the workbook's rules are scoped by, with their effects
# ---------------------------------------------------------------------------

KITCHEN_TYPES = (  # rule-book kitchen type -> generator layout value ("supported": the engine builds it)
    ("Straight", "single", True), ("L-Shape", "L", True), ("U-Shape", "U", True), ("Parallel", "parallel", False),
)


def options() -> dict:
    """Filters the rule workbook scopes its rules by, each with the rules it switches on.
    Only filters some active rule uses are listed (the workbook's vastu / user-count / region
    columns are empty for every rule)."""
    rb = load()
    cite = rb.cite
    table: dict = {}
    for kt, _, _ in KITCHEN_TYPES:
        table[kt] = {}
        for band, _, _ in SIZE_BANDS:
            row = {}
            for role in ("hob", "sink", "spice", "cylinder", "corner"):
                r = rb.module_width(role, kt, band)
                if r:
                    row[role] = {"width": r.width, "combination": r.combination, "rule": r.rule}
            table[kt][band] = row
    return {
        "source": rb.source,
        "rules": len(rb.texts),
        "budget": [
            {"value": "auto", "label": "Not stated", "effects": ["The designer chooses the base cabinet style (drawers / mixed / shutters)."], "rules": []},
            {"value": "essential", "label": "Essential",
             "effects": ["Base units: single shutter under 600 mm, double shutter 600-900 mm.",
                         "A 600 mm gap: double shutter when the working aisle is under 1000 mm, single shutter otherwise.",
                         "Wall corner: blind corner unit; the loft corner stays an open 330 x 330 mm void."],
             "rules": [cite(n) for n in (6, 7, 31, 32, 43, 75)]},
            {"value": "premium", "label": "Premium",
             "effects": ["Base units: double drawer from 600 mm, single shutter under 600 mm.",
                         "Wall corner: L corner unit with a loft corner unit above."],
             "rules": [cite(n) for n in (8, 9, 33, 39, 40, 71)]},
            {"value": "luxury", "label": "Luxury",
             "effects": ["Base units: double drawer from 600 mm, single shutter under 600 mm.",
                         "Wall corner: L corner unit with a loft corner unit above."],
             "rules": [cite(n) for n in (10, 38, 41, 71)]},
        ],
        "fuel": [
            {"value": "auto", "label": "Not stated - LPG assumed", "effects": ["Cylinder unit included."], "rules": [cite(11)]},
            {"value": "lpg", "label": "LPG cylinder", "effects": ["Cylinder unit on the hob's left (R28)."], "rules": [cite(11), cite(28)]},
            {"value": "png", "label": "Piped gas (PNG)", "effects": ["No cylinder unit."], "rules": [cite(11)]},
            {"value": "induction", "label": "Induction", "effects": ["No cylinder unit."], "rules": [cite(11)]},
        ],
        "size_band": [{"value": "auto", "label": "From the floor area", "range": ""}]
        + [{"value": b.lower(), "label": b, "range": f"{lo}-{hi} sqft"} for b, lo, hi in SIZE_BANDS],
        "kitchen_type": [{"label": kt, "layout": lay, "supported": ok} for kt, lay, ok in KITCHEN_TYPES],
        "walls": {"primary": cite(24), "secondary": cite(25)},
        "module_widths": table,
        "always": [cite(n) for n in (26, 27, 28, 29, 44, 45, 46, 47, 48, 53, 54)],
    }


def _free_run_mm(geometry, wid: str) -> float:
    """Longest wall span free of doors (with their clearance)."""
    w = geometry.walls[wid]
    lo, hi = w.usable
    segs, cur = [], lo
    for a, b in sorted(d.interval(config.DOOR_CLEARANCE_MM) for d in w.doors):
        if min(a, hi) > cur:
            segs.append(min(a, hi) - cur)
        cur = max(cur, b)
    if hi > cur:
        segs.append(hi - cur)
    return max(segs, default=0.0)


def near_walls(geometry, wid: Optional[str]) -> list[str]:
    """The walls meeting a wall at its corners - end corner first, then start corner. Only these can
    be the sink (secondary) wall for that cooking (primary) wall."""
    w = geometry.walls.get(wid or "")
    if w is None:
        return []
    out = []
    for c in (w.neighbor_end, w.neighbor_start):
        if c and c in geometry.walls and c != wid and c not in out:
            out.append(c)
    return out


def next_wall(geometry, wid: Optional[str], among: Optional[list[str]] = None) -> Optional[str]:
    """The secondary (sink) wall that goes with a primary (cooking) wall: the next wall round the
    room - the one meeting the primary wall at its end corner - or, if that one cannot carry
    cabinets (too short / a door), the wall at its start corner (R24 / R25: hob on the primary
    wall, sink on the secondary wall)."""
    cands = [c for c in near_walls(geometry, wid) if not among or c in among]  # among: the customer's walls
    for c in cands:
        if _free_run_mm(geometry, c) >= config.MIN_CABINET_WALL_MM:
            return c
    return cands[0] if cands else None


def plan_facts(geometry, cabinet_walls: Optional[list[str]] = None) -> dict:
    """Rule-book facts read from a floor plan: floor area and band, ceiling, lofts, walls, aisles."""
    area = floor_area_sqft(geometry)
    band, note = size_band(area)
    ceiling = round(min(w.height_m for w in geometry.walls.values()) * 1000)
    walls = list(cabinet_walls or geometry.wall_order)
    return {
        "floor_area_sqft": round(area, 1),
        "size_band": band,
        "size_note": note,
        "ceiling_mm": ceiling,
        "lofts_possible": ceiling + 1 >= round(config.LOFT_TOP_M * 1000),
        "walls": [{"id": wid, "name": geometry.walls[wid].name, "length_mm": geometry.walls[wid].length_mm,
                   "window": bool(geometry.walls[wid].windows), "door": bool(geometry.walls[wid].doors),
                   "next": next_wall(geometry, wid),  # the sink wall that goes with this cooking wall
                   "near": near_walls(geometry, wid),  # the only walls the sink may go on with this cooking wall
                   "aisle_mm": (round(a) if (a := working_aisle_mm(geometry, wid, walls)) is not None else None)}
                  for wid in geometry.wall_order],
    }


# ---------------------------------------------------------------------------
# Budget tier: plain base-unit fill (R6-R10, R31-R33, R38-R43) and corners (R71, R75)
# ---------------------------------------------------------------------------

BASE_FILL_MAX_MM = 900  # R7: 600-900 mm gaps take one unit; wider gaps are tiled (R34, R42)


def base_fill_combination(budget: str, width: int, aisle_mm: Optional[float]) -> tuple[str, int]:
    """(cabinet combination, rule) for a plain Base Unit of ``width`` mm in ``budget``."""
    if width < 600:
        return "Single Shutter", {"essential": 6, "premium": 39, "luxury": 41}[budget]
    if budget == "essential":
        if width == 600 and aisle_mm is not None:
            # R31 / R32: a 600 mm gap: double shutter in a tight aisle (smaller door swing)
            return ("Double Shutter", 31) if aisle_mm < 1000 else ("Single Shutter", 32)
        return "Double Shutter", 7
    if budget == "premium":
        return "Double Drawer", 33 if width == 600 else 40
    return "Double Drawer", 38


def wall_corner_kind(budget: Optional[str]) -> Optional[str]:
    """R71 / R75: Essential kitchens use a blind wall corner (open loft void above);
    Premium / Luxury a Wall L Corner (loft corner unit above). None = no budget stated."""
    if not budget:
        return None
    return "blind" if budget == "essential" else "L"


# ---------------------------------------------------------------------------
# Lofts (R47-R54): height coupled to the ceiling
# ---------------------------------------------------------------------------

LOFT_HEIGHTS_MM = (850, 800, 750, 700, 650, 600)
LOFT_REVEAL_MM = (18, 100)  # R48: gap left between the loft top and the ceiling


def loft_size(ceiling_mm: float, wall_top_mm: float) -> Optional[tuple[int, int]]:
    """R48: the largest standard loft height leaving an 18-100 mm reveal to the ceiling.
    Returns (loft height, reveal) or None when no standard height fits that reveal."""
    lo, hi = LOFT_REVEAL_MM
    for h in LOFT_HEIGHTS_MM:
        reveal = ceiling_mm - wall_top_mm - config.LOFT_BOTTOM_MM - h
        if lo <= reveal <= hi:
            return h, round(reveal)
    return None
