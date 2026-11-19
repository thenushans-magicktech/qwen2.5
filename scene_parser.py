"""Parse a ``magickhomes-scene`` v1 document into solver-friendly room geometry.

Coordinate conventions (verified against production exports):

* Walls are centre-lines ``start`` -> ``end`` in world ``[x, z]`` metres.
* Wall-attached nodes use a wall-local frame: ``x`` = distance from ``start``
  along the wall, ``y`` = height above floor, ``z`` = distance from the wall
  centre-line towards the wall's front (interior) side.
* ``wall_length_mm = round(hypot(x2 - x1, z2 - z1) * 1000)`` and
  ``wallT = center_mm / wall_length_mm``.
* Door / window ``position = [center_along_wall_m, center_height_m, 0]``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import config

FORMAT = "magickhomes-scene"
VERSION = 1


class SceneError(ValueError):
    pass


def wall_length_mm(start, end) -> int:
    return round(math.hypot(end[0] - start[0], end[1] - start[1]) * 1000)


@dataclass
class Opening:
    id: str
    kind: str  # "door" | "window"
    wall_id: str
    center_mm: float
    width_mm: float
    bottom_m: float
    top_m: float
    hinged_inward: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def start_mm(self) -> float:
        return self.center_mm - self.width_mm / 2

    @property
    def end_mm(self) -> float:
        return self.center_mm + self.width_mm / 2

    def interval(self, margin_mm: float = 0.0) -> tuple[float, float]:
        return (self.start_mm - margin_mm, self.end_mm + margin_mm)


@dataclass
class Wall:
    id: str
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    length_mm: int
    thickness_m: float
    height_m: float
    direction: tuple[float, float]   # unit vector start -> end (world x, z)
    normal: tuple[float, float]      # unit vector towards the room (local +z)
    openings: list[Opening] = field(default_factory=list)
    # Distance from each wall end to the face of a connecting wall (mm).
    inset_start_mm: float = 0.0
    inset_end_mm: float = 0.0
    neighbor_start: Optional[str] = None
    neighbor_end: Optional[str] = None

    @property
    def length_m(self) -> float:
        return self.length_mm / 1000.0

    @property
    def doors(self) -> list[Opening]:
        return [o for o in self.openings if o.kind == "door"]

    @property
    def windows(self) -> list[Opening]:
        return [o for o in self.openings if o.kind == "window"]

    @property
    def usable(self) -> tuple[float, float]:
        """Interior face-to-face span along the wall (mm)."""
        return (self.inset_start_mm, self.length_mm - self.inset_end_mm)

    def to_world(self, x_m: float, z_m: float) -> tuple[float, float]:
        """Wall-local (x along wall, z into room) -> world (x, z), metres."""
        return (
            self.start[0] + self.direction[0] * x_m + self.normal[0] * z_m,
            self.start[1] + self.direction[1] * x_m + self.normal[1] * z_m,
        )

    def end_coord(self, end: str) -> float:
        return 0.0 if end == "start" else float(self.length_mm)


@dataclass
class Corner:
    """Interior corner where ``wall_a``'s ``end_a`` meets ``wall_b``'s ``end_b``."""

    wall_a: str
    end_a: str
    wall_b: str
    end_b: str
    point: tuple[float, float]
    angle_deg: float

    def other(self, wall_id: str) -> tuple[str, str, str]:
        """(my_end, other_wall, other_end) seen from ``wall_id``."""
        if wall_id == self.wall_a:
            return self.end_a, self.wall_b, self.end_b
        return self.end_b, self.wall_a, self.end_a

    def involves(self, wall_id: str) -> bool:
        return wall_id in (self.wall_a, self.wall_b)


@dataclass
class RoomGeometry:
    walls: dict[str, Wall]
    wall_order: list[str]
    corners: list[Corner]
    openings: list[Opening]
    level_id: Optional[str]
    ceiling_height_m: Optional[float]
    outline: list[tuple[float, float]]
    centroid: tuple[float, float]

    @property
    def doors(self) -> list[Opening]:
        return [o for o in self.openings if o.kind == "door"]

    @property
    def windows(self) -> list[Opening]:
        return [o for o in self.openings if o.kind == "window"]

    def corner_between(self, a: str, b: str) -> Optional[Corner]:
        for c in self.corners:
            if {c.wall_a, c.wall_b} == {a, b}:
                return c
        return None

    def corners_of(self, wall_id: str) -> list[Corner]:
        return [c for c in self.corners if c.involves(wall_id)]

    def summary(self) -> dict:
        """Compact geometry for prompts / reports."""
        out = []
        for wid in self.wall_order:
            w = self.walls[wid]
            out.append(
                {
                    "id": wid,
                    "name": w.name,
                    "length_mm": w.length_mm,
                    "height_mm": round(w.height_m * 1000),
                    "usable_mm": [round(v) for v in w.usable],
                    "doors": [[round(o.start_mm), round(o.end_mm)] for o in w.doors],
                    "windows": [[round(o.start_mm), round(o.end_mm)] for o in w.windows],
                    "start_meets": w.neighbor_start,
                    "end_meets": w.neighbor_end,
                }
            )
        return {
            "walls": out,
            "corners": [[c.wall_a, c.wall_b] for c in self.corners],
            "ceiling_mm": round((self.ceiling_height_m or 0) * 1000) or None,
        }


# ---------------------------------------------------------------------------

def _nodes(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise SceneError("scene document must be a JSON object")
    if doc.get("format") != FORMAT:
        raise SceneError(f"format must be {FORMAT!r}, got {doc.get('format')!r}")
    if doc.get("version") != VERSION:
        raise SceneError(f"version must be {VERSION}, got {doc.get('version')!r}")
    nodes = (doc.get("scene") or {}).get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise SceneError("scene.nodes missing or empty")
    return nodes


def _unit(dx: float, dz: float) -> tuple[float, float]:
    n = math.hypot(dx, dz)
    if n < 1e-9:
        raise SceneError("degenerate wall (zero length)")
    return (dx / n, dz / n)


def parse_scene(doc: dict) -> RoomGeometry:
    nodes = _nodes(doc)
    raw_walls = [n for n in nodes.values() if n.get("type") == "wall"]
    if not raw_walls:
        raise SceneError("scene has no wall nodes")

    ceiling = next((n for n in nodes.values() if n.get("type") == "ceiling"), None)
    ceiling_h = float(ceiling["height"]) if ceiling and ceiling.get("height") else None
    level = next((n for n in nodes.values() if n.get("type") == "level"), None)

    # Room centroid: ceiling/slab polygon when present, else wall midpoints.
    outline: list[tuple[float, float]] = []
    if ceiling and isinstance(ceiling.get("polygon"), list) and len(ceiling["polygon"]) >= 3:
        outline = [tuple(p) for p in ceiling["polygon"]]
    else:
        slab = next((n for n in nodes.values() if n.get("type") == "slab"), None)
        poly = (slab or {}).get("polygon")
        if isinstance(poly, list) and len(poly) >= 3:
            outline = [tuple(p) for p in poly]
    if not outline:
        outline = [tuple(w["start"]) for w in raw_walls]
    cx = sum(p[0] for p in outline) / len(outline)
    cz = sum(p[1] for p in outline) / len(outline)

    walls: dict[str, Wall] = {}
    order: list[str] = []
    level_children = (level or {}).get("children") or []
    for w in sorted(raw_walls, key=lambda n: level_children.index(n["id"]) if n["id"] in level_children else 1e9):
        start, end = tuple(map(float, w["start"])), tuple(map(float, w["end"]))
        d = _unit(end[0] - start[0], end[1] - start[1])
        # Interior normal: the perpendicular that points at the room centroid.
        n1 = (-d[1], d[0])
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        if (cx - mid[0]) * n1[0] + (cz - mid[1]) * n1[1] < 0:
            n1 = (d[1], -d[0])
        walls[w["id"]] = Wall(
            id=w["id"],
            name=w.get("name") or w["id"],
            start=start,
            end=end,
            length_mm=wall_length_mm(start, end),
            thickness_m=float(w.get("thickness") or config.DEFAULT_WALL_THICKNESS_M),
            height_m=float(w.get("height") or ceiling_h or config.DEFAULT_WALL_HEIGHT_M),
            direction=d,
            normal=n1,
        )
        order.append(w["id"])

    # Openings -------------------------------------------------------------
    openings: list[Opening] = []
    for n in nodes.values():
        kind = n.get("type")
        if kind not in ("door", "window"):
            continue
        wid = n.get("wallId") or n.get("parentId")
        if wid not in walls:
            continue
        pos = n.get("position") or [0, 0, 0]
        width = float(n.get("width") or 0.9)
        height = float(n.get("height") or (2.1 if kind == "door" else 1.2))
        cy = float(pos[1]) if len(pos) > 1 else height / 2
        op = Opening(
            id=n["id"],
            kind=kind,
            wall_id=wid,
            center_mm=float(pos[0]) * 1000.0,
            width_mm=width * 1000.0,
            bottom_m=cy - height / 2,
            top_m=cy + height / 2,
            hinged_inward=(
                kind == "door"
                and (n.get("doorType") or "hinged") == "hinged"
                and (n.get("swingDirection") or "inward") == "inward"
            ),
            raw=n,
        )
        walls[wid].openings.append(op)
        openings.append(op)

    # Corners ----------------------------------------------------------------
    corners: list[Corner] = []
    tol = 0.02  # 2 cm endpoint snapping
    ids = list(walls)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            wa, wb = walls[a], walls[b]
            for end_a in ("start", "end"):
                pa = wa.start if end_a == "start" else wa.end
                for end_b in ("start", "end"):
                    pb = wb.start if end_b == "start" else wb.end
                    if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) > tol:
                        continue
                    # Direction of each wall pointing away from the shared point.
                    da = wa.direction if end_a == "start" else (-wa.direction[0], -wa.direction[1])
                    db = wb.direction if end_b == "start" else (-wb.direction[0], -wb.direction[1])
                    cosang = max(-1.0, min(1.0, da[0] * db[0] + da[1] * db[1]))
                    angle = math.degrees(math.acos(cosang))
                    # Interior corner: each wall runs along the other's room side.
                    interior = (db[0] * wa.normal[0] + db[1] * wa.normal[1]) > 0.5 and (
                        da[0] * wb.normal[0] + da[1] * wb.normal[1]
                    ) > 0.5
                    if 60.0 <= angle <= 120.0 and interior:
                        corners.append(Corner(a, end_a, b, end_b, pa, angle))
                    # Any joint (even straight) trims the usable span by half
                    # the connecting wall's thickness.
                    if angle > 30.0:
                        inset_a = wb.thickness_m * 500.0
                        inset_b = wa.thickness_m * 500.0
                        if end_a == "start":
                            wa.inset_start_mm, wa.neighbor_start = max(wa.inset_start_mm, inset_a), b
                        else:
                            wa.inset_end_mm, wa.neighbor_end = max(wa.inset_end_mm, inset_a), b
                        if end_b == "start":
                            wb.inset_start_mm, wb.neighbor_start = max(wb.inset_start_mm, inset_b), a
                        else:
                            wb.inset_end_mm, wb.neighbor_end = max(wb.inset_end_mm, inset_b), a

    return RoomGeometry(
        walls=walls,
        wall_order=order,
        corners=corners,
        openings=openings,
        level_id=(level or {}).get("id"),
        ceiling_height_m=ceiling_h,
        outline=outline,
        centroid=(cx, cz),
    )


def strip_items(doc: dict, predicate: Optional[Callable[[dict], bool]] = None) -> dict:
    """Deep-copy ``doc`` without ``item`` nodes (all, or those matching ``predicate``).

    Removed ids are also dropped from every ``children`` list and ``rootNodeIds``.
    """
    out = copy.deepcopy(doc)
    nodes = _nodes(out)
    pred = predicate or (lambda n: n.get("type") == "item")
    doomed = {nid for nid, n in nodes.items() if pred(n)}
    # Children of removed items go too (e.g. sink under a sink cabinet).
    changed = True
    while changed:
        changed = False
        for nid, n in nodes.items():
            if nid not in doomed and n.get("parentId") in doomed:
                doomed.add(nid)
                changed = True
    for nid in doomed:
        nodes.pop(nid, None)
    for n in nodes.values():
        if isinstance(n.get("children"), list):
            n["children"] = [c for c in n["children"] if c not in doomed]
    scene = out["scene"]
    if isinstance(scene.get("rootNodeIds"), list):
        scene["rootNodeIds"] = [r for r in scene["rootNodeIds"] if r not in doomed]
    return out
