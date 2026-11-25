"""Turn a solved :class:`KitchenLayout` into a hydrated ``magickhomes-scene`` v1 document.

Adds to ``scene.nodes`` (matching production exports):

* ``item_*`` cabinet nodes (``asset.category == "kitchen"``) parented to their wall,
* appliance nodes: ``Sink90`` / ``Hob90`` as children of their host cabinets and a
  wall-mounted ``Chimney90`` centred above the hob,
* procedural ``fixture`` nodes: toe-kicks, countertops (with sink/hob cut-outs),
  backsplashes, end slabs, fillers, tall side panels, loft bottom/top/side panels
  and the room floor,

and appends every new wall-level id to ``scene.nodes[wallId].children``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import string
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import config
import finishes
from rule_engine import EPS, KitchenLayout, Placement
from scene_parser import Wall

_ID_ALPHABET = string.ascii_lowercase + string.digits
RUN_JOIN_MM = config.MAX_FILLER_MM  # gaps up to this are closed with filler panels


@lru_cache(maxsize=1)
def load_presets() -> dict:
    try:
        return json.loads(config.MATERIAL_PRESETS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"shutterTextures": {}, "fixtureTextures": {}, "fixtureColors": {}, "appliances": {}}


class IdFactory:
    """Deterministic ``item_<16 chars>`` ids (same input + seed -> same ids)."""

    def __init__(self, seed: int, taken: set[str]):
        self.rng = random.Random(seed)
        self.taken = set(taken)

    def item(self) -> str:
        while True:
            nid = "item_" + "".join(self.rng.choice(_ID_ALPHABET) for _ in range(16))
            if nid not in self.taken:
                self.taken.add(nid)
                return nid


def scene_seed(doc: dict, extra: str = "") -> int:
    nodes = sorted((doc.get("scene") or {}).get("nodes", {}))
    h = hashlib.sha256((json.dumps(nodes) + extra).encode()).hexdigest()
    return int(h[:12], 16)


def _r(v: float, nd: int = 6) -> float:
    return round(float(v), nd)


# ---------------------------------------------------------------------------

class SceneHydrator:
    def __init__(self, doc: dict, layout: KitchenLayout, seed: int, include_floor: bool = True):
        self.doc = copy.deepcopy(doc)
        self.nodes: dict = self.doc["scene"]["nodes"]
        self.layout = layout
        self.g = layout.geometry
        self.ids = IdFactory(seed, set(self.nodes))
        self.presets = load_presets()
        self.include_floor = include_floor
        self.cab_ids: dict[int, str] = {}
        self.stats: dict[str, int] = {}

    # -- generic helpers ------------------------------------------------------
    def _attach(self, node: dict, parent_id: str) -> None:
        self.nodes[node["id"]] = node
        parent = self.nodes.get(parent_id)
        if parent is not None:
            parent.setdefault("children", []).append(node["id"])
        kind = (node.get("asset") or {}).get("category", "?")
        key = kind if kind != "fixture" else f"fixture:{node['metadata'].get('fixtureKind')}"
        self.stats[key] = self.stats.get(key, 0) + 1

    def _zshift(self, wall: Wall) -> float:
        """Production fixture z values assume a 100 mm wall (face at 0.05 m)."""
        return wall.thickness_m / 2.0 - 0.05

    # -- cabinets -------------------------------------------------------------------
    def _cabinet_node(self, p: Placement) -> dict:
        wall = self.g.walls[p.wall_id]
        it = p.item
        w_m, h_m, d_m = it.width / 1000.0, it.height / 1000.0, it.depth / 1000.0
        meta = it.metadata()
        if not meta.get("textures"):
            # the run's finish first (MagickHome applies textures[0]): base + tall share the base finish,
            # wall units + lofts the wall finish (finish rules)
            meta["textures"] = finishes.texture_list(self.presets.get("shutterTextures", {}).get(p.tier, []),
                                                     self._finish(p.tier))
        nid = self.ids.item()
        # Rooms on a wall's back side (frontSide == "exterior"): mirror depth and rotation.
        placement = {"level": p.tier, "side": wall.side_sign, "designSource": config.DESIGN_SOURCE}
        if p.corner_end:
            # hand of the corner unit as seen from the room ("left" = door left, blind right)
            placement["cornerOrientation"] = "left" if wall.side_of(p.corner_end) == "right" else "right"
            placement["cornerAtEnd"] = p.corner_end == "end"
        return {
            "object": "node",
            "id": nid,
            "type": "item",
            "name": it.aid,
            "parentId": wall.id,
            "visible": True,
            "metadata": meta,
            "position": [p.center_mm / 1000.0, p.pos_y(), wall.side_sign * p.pos_z(wall)],
            "rotation": _facing(wall),
            "scale": [1, 1, 1],
            "children": [],
            "wallId": wall.id,
            "wallT": p.center_mm / wall.length_mm,
            "asset": {
                "id": meta.get("_id") or it.aid,
                "category": "kitchen",
                "name": it.aid,
                # 3D model + thumbnail URLs come straight from the Assets API record;
                # the S3 pattern is only a fallback for the offline xlsx catalogue.
                "thumbnail": meta.get("thumbnail") or config.CABINET_THUMBNAIL_URL.format(aid=it.aid),
                "source": "library",
                "src": meta.get("modelUrl") or config.CABINET_MODEL_URL.format(aid=it.aid),
                "dimensions": [w_m, h_m, d_m],
                "offset": [0, h_m / 2.0, 0],
                "rotation": [0, 0, 0],
                "scale": [0.01, 0.01, 0.01],
            },
            "placement": placement,
        }

    # -- appliances ------------------------------------------------------------------
    def _appliance_node(self, kind: str, parent_id: str, wall_id: str, wall_t: float, position, rotation, host_id: str,
                        host_width_mm: float = 900) -> dict:
        cfg = config.appliance_for(kind, host_width_mm)  # the model that fits the host cabinet
        preset = self.presets.get("appliances", {}).get(kind, {})
        w, h, d = cfg["dims_mm"]
        meta = copy.deepcopy(preset.get("metadata") or {})
        if meta.get("aid") not in (None, cfg["aid"]):  # preset belongs to another size: keep only its textures
            meta = {"textures": meta.get("textures", [])}
        meta.update(
            {
                "aid": cfg["aid"],
                "category": cfg["category"],
                "description": cfg["name"],
                "width": w,
                "depth": d,
                "height": h,
                "modelUrl": cfg["src"],
                "modelKey": cfg["src"].split(".amazonaws.com/", 1)[-1],
                "thumbnail": cfg["thumbnail"],
                "isActive": True,
                "applianceKind": kind,
                "hostCabinetId": host_id,
                "designSource": config.DESIGN_SOURCE_APPLIANCE,
                "level": cfg["level"],
                "subCategory": cfg["category"],
            }
        )
        meta.setdefault("_id", cfg["aid"])
        meta.setdefault("textures", [])
        return {
            "object": "node",
            "id": self.ids.item(),
            "type": "item",
            "name": cfg["name"],
            "parentId": parent_id,
            "visible": True,
            "metadata": meta,
            "position": list(position),
            "rotation": list(rotation),
            "scale": [1, 1, 1],
            "children": [],
            "wallId": wall_id,
            "wallT": wall_t,
            "asset": {
                "id": meta["_id"],
                "category": "appliance",
                "name": cfg["name"],
                "thumbnail": cfg["thumbnail"],
                "source": "library",
                "src": cfg["src"],
                "dimensions": [w / 1000.0, h / 1000.0, d / 1000.0],
                "offset": [0, h / 2000.0, 0],
                "rotation": [0, 0, 0],
                "scale": [0.01, 0.01, 0.01],
            },
        }

    def _add_appliances(self) -> None:
        lay = self.layout
        if lay.sink is not None and id(lay.sink) in self.cab_ids:
            host = self.nodes[self.cab_ids[id(lay.sink)]]
            node = self._appliance_node(
                "sink", host["id"], host["wallId"], host["wallT"], config.APPLIANCES["sink"]["local_position"], [0, 0, 0], host["id"],
                host_width_mm=lay.sink.item.width,
            )
            self._attach(node, host["id"])
        if lay.hob is not None and id(lay.hob) in self.cab_ids:
            host = self.nodes[self.cab_ids[id(lay.hob)]]
            node = self._appliance_node(
                "hob", host["id"], host["wallId"], host["wallT"], config.APPLIANCES["hob"]["local_position"], [0, 0, 0], host["id"],
                host_width_mm=lay.hob.item.width,
            )
            self._attach(node, host["id"])
            wall = self.g.walls[host["wallId"]]
            y, z = config.APPLIANCES["chimney"]["wall_position_yz"]
            chim = self._appliance_node(
                "chimney",
                wall.id,
                wall.id,
                host["wallT"],
                [host["position"][0], y, _r(wall.side_sign * (z + self._zshift(wall)))],
                _facing(wall),
                host["id"],
                host_width_mm=lay.hob.item.width,
            )
            self._attach(chim, wall.id)

    def _finish(self, tier: Optional[str]) -> Optional[str]:
        """The finish aid chosen for the run this tier belongs to (None = scene default)."""
        return (getattr(self.layout, "finishes", None) or {}).get(finishes.RUN_OF_TIER.get(tier or "", ""))

    # -- fixtures ---------------------------------------------------------------------
    def _wall_zone(self, spec: dict) -> dict:
        """A fixture spec made for 600 mm wall units, stretched to this kitchen's wall-cabinet height."""
        h = (self.layout.heights or {}).get("wall_mm", 600) / 1000.0
        return dict(spec, y=config.POS_Y_M["wall"] + h / 2, thickness=h)

    def _tall_zone(self, spec: dict) -> dict:
        """A fixture spec made for 1940 mm tall units, stretched to this kitchen's tall-unit height."""
        h = (self.layout.heights or {}).get("tall_mm", 1940) / 1000.0
        return dict(spec, y=config.POS_Y_M["tall"] + h / 2, thickness=h)

    def _fixture(self, wall: Wall, kind: str, suffix: str, x0_mm: float, x1_mm: float, y: float, z: float,
                 thickness: float, depth: float, level: Optional[str] = None, **extra) -> Optional[dict]:
        width = (x1_mm - x0_mm) / 1000.0
        if width <= 0.0005:
            return None
        center_mm = (x0_mm + x1_mm) / 2.0
        fixture_id = f"fixture_{kind.replace('-', '')}_{wall.id}_{suffix}"
        slot = {"toe-kick": "toeKick", "countertop": "counterTop", "backsplash": "backsplash"}.get(kind, "door")
        tex_key = f"{kind}:{level}" if level and kind in ("slab", "filler") else kind
        textures = copy.deepcopy(self.presets.get("fixtureTextures", {}).get(tex_key, []))
        if kind in ("slab", "filler") and level:  # panels and fillers match the run they close
            textures = finishes.texture_list(textures, self._finish(level), drop_sku=True)
        elif kind in finishes.SURFACES:  # the customer's worktop / backsplash
            textures = finishes.texture_list(textures, (getattr(self.layout, "finishes", None) or {}).get(kind),
                                             records=finishes.surface_options(kind))
        meta = {
            "fixtureKind": kind,
            "fixtureId": fixture_id,
            "widthM": _r(width),
            "thicknessM": _r(thickness),
            "depthM": _r(depth),
            "color": self.presets.get("fixtureColors", {}).get(kind) or config.FIXTURE_COLORS.get(kind, "#e8e8e8"),
            "materialSlot": slot,
            "side": wall.side_sign,
            "wallT": center_mm / wall.length_mm,
            "designSource": config.DESIGN_SOURCE_FIXTURE,
            "fromDesignResponse": True,
        }
        if level:
            meta["level"] = level
        meta.update(extra)
        if textures:
            meta["textures"] = textures
        node = {
            "object": "node",
            "id": self.ids.item(),
            "type": "item",
            "name": f"{kind.capitalize() if kind != 'toe-kick' else 'Toe-kick'} {fixture_id}",
            "parentId": wall.id,
            "visible": True,
            "metadata": meta,
            "position": [center_mm / 1000.0, _r(y), _r(wall.side_sign * (z + self._zshift(wall)))],
            "rotation": list(config.FIXTURE_ROTATION),
            "scale": [1, 1, 1],
            "children": [],
            "wallId": wall.id,
            "wallT": center_mm / wall.length_mm,
            "asset": {
                "id": fixture_id,
                "category": "fixture",
                "name": kind,
                "thumbnail": "",
                "source": "library",
                "src": f"asset://fixture/{kind}",
                "dimensions": [_r(width), _r(thickness), _r(depth)],
                "offset": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
            },
        }
        self._attach(node, wall.id)
        return node

    @staticmethod
    def _runs(ps: list[Placement], join: float = RUN_JOIN_MM) -> list[list[Placement]]:
        runs: list[list[Placement]] = []
        for p in sorted(ps, key=lambda p: p.start_mm):
            if runs and p.start_mm - runs[-1][-1].end_mm <= join + EPS:
                runs[-1].append(p)
            else:
                runs.append([p])
        return runs

    def _end(self, wall: Wall, zone: str, run: list[Placement], side: str):
        """Classify one end of a run -> (kind, target_mm, distance).

        kind: flush | filler (to wall/corner) | neighbor (cabinet/tall) | exposed
        """
        x = run[0].start_mm if side == "lo" else run[-1].end_mm
        b = self.layout.boundary(wall.id, zone, x, -1 if side == "lo" else 1, exclude=run)
        if b is None:
            return "exposed", x, 0.0
        kind, coord, dist = b
        if kind in ("wall", "corner") and dist <= config.MAX_FILLER_MM:
            return ("flush" if dist <= EPS else "filler"), coord, dist
        if kind in ("cabinet", "tall") and dist <= config.MAX_FILLER_MM:
            return "neighbor", coord, dist
        return "exposed", x, dist

    def _corner_feature(self, wall: Wall, zone: str, coord: float):
        for f in self.layout.features.get((wall.id, zone), []):
            if f.kind == "corner" and (abs(f.lo - coord) <= EPS or abs(f.hi - coord) <= EPS):
                return f
        return None

    def _add_wall_fixtures(self, wall: Wall) -> None:
        lay = self.layout
        P = config.PANEL_T_M * 1000.0  # 18 mm
        face = 0.05  # production z values are relative to a 100 mm wall; _zshift adapts
        base_ps = [p for p in lay.on(wall.id, "base") if p.tier == "base"]
        talls = [p for p in lay.on(wall.id, "base") if p.tier == "tall"]
        counter_spans: list[tuple[float, float]] = []

        # ---- floor runs: toe-kicks under base + tall units ------------------------
        for i, run in enumerate(self._runs(base_ps + talls)):
            lo_kind, lo_t, _ = self._end(wall, "base", run, "lo")
            hi_kind, hi_t, _ = self._end(wall, "base", run, "hi")
            x0 = lo_t if lo_kind in ("filler", "flush") else run[0].start_mm
            x1 = hi_t if hi_kind in ("filler", "flush") else run[-1].end_mm
            tk = config.TOEKICK
            self._fixture(wall, "toe-kick", f"run{i}", x0, x1, tk["y"], tk["z"], tk["thickness"], tk["depth"])
            for side, kind in (("lo", lo_kind), ("hi", hi_kind)):
                if kind == "exposed":
                    x = run[0].start_mm if side == "lo" else run[-1].end_mm - P
                    self._fixture(wall, "toe-kick", f"side_run{i}_{side}", x, x + P, tk["y"], tk["side_z"], tk["thickness"], tk["side_depth"])

        # ---- counter runs: countertop, backsplash, base slabs & fillers -----------
        for i, run in enumerate(self._runs(base_ps)):
            ends = {}
            for side in ("lo", "hi"):
                kind, target, dist = self._end(wall, "base", run, side)
                x = run[0].start_mm if side == "lo" else run[-1].end_mm
                if kind == "filler" or (kind == "neighbor" and dist > EPS):
                    a, b = (target, x) if side == "lo" else (x, target)
                    bf = config.BASE_FILLER
                    self._fixture(wall, "filler", f"base_{side}_run{i}", a, b, bf["y"], bf["z"], bf["thickness"],
                                  config.PANEL_T_M, level="base", frontShiftM=0)
                elif kind == "exposed":
                    a = x - P if side == "lo" else x
                    bs = config.BASE_SLAB
                    self._fixture(wall, "slab", f"base_{side}_run{i}", a, a + P, bs["y"], bs["z"], bs["thickness"],
                                  bs["depth"], level="base", frontShiftM=0, extraDepthM=config.PANEL_T_M)
                ends[side] = (kind, target, x)
            # mid-run gaps -> filler panels
            for j, (a, b) in enumerate(zip(run, run[1:])):
                if b.start_mm - a.end_mm > EPS:
                    bf = config.BASE_FILLER
                    self._fixture(wall, "filler", f"base_mid_run{i}_{j}", a.end_mm, b.start_mm, bf["y"], bf["z"],
                                  bf["thickness"], config.PANEL_T_M, level="base", frontShiftM=0)
            # countertop span
            ct0 = self._counter_edge(wall, ends["lo"], "lo", P)
            ct1 = self._counter_edge(wall, ends["hi"], "hi", P)
            counter_spans.append((ct0, ct1))
            ct = config.COUNTERTOP
            holes = self._holes(wall, run, (ct0 + ct1) / 2.0)
            extra = {"frontEdge": dict(ct["edge"])}
            if holes:
                extra["holes"] = holes
            self._fixture(wall, "countertop", f"ct{i}", ct0, ct1, ct["y"], ct["z"], ct["thickness"], ct["depth"], **extra)
            # low backsplash (skips windows)
            bs0 = ct0
            if ends["lo"][0] in ("filler", "flush") and self._corner_feature(wall, "base", ends["lo"][1]):
                bs0 = wall.usable[0] if ends["lo"][1] < wall.length_mm / 2 else bs0
            bs1 = ct1
            if ends["hi"][0] in ("filler", "flush") and self._corner_feature(wall, "base", ends["hi"][1]):
                bs1 = wall.usable[1] if ends["hi"][1] > wall.length_mm / 2 else bs1
            spans = [(bs0, bs1)]
            for w in wall.windows:
                if w.bottom_m < 1.44 and w.top_m > 0.84:
                    spans = _cut(spans, w.start_mm, w.end_mm)
            bk = config.BACKSPLASH
            for k, (a, b) in enumerate(spans):
                if b - a >= 10:
                    self._fixture(wall, "backsplash", f"bs{i}_{k}", a, b, bk["y"], bk["z"], bk["thickness"], bk["depth"])

        # ---- tall units: side panels & fillers ------------------------------------------
        for i, run in enumerate(self._runs(talls, join=EPS)):
            for side in ("lo", "hi"):
                x = run[0].start_mm if side == "lo" else run[-1].end_mm
                kind, target, dist = self._end(wall, "base", run, side)
                tf = self._tall_zone(config.TALL_FILLER)
                if kind == "filler":
                    a, b = (target, x) if side == "lo" else (x, target)
                    self._fixture(wall, "filler", f"tall_{side}_run{i}", a, b, tf["y"], tf["z"], tf["thickness"],
                                  config.PANEL_T_M, level="tall", frontShiftM=tf["front_shift"])
                    continue
                if kind == "flush":
                    continue
                a = x - P if side == "lo" else x
                base_next = kind == "neighbor"
                wall_b = self.layout.boundary(wall.id, "wall", x, -1 if side == "lo" else 1, exclude=run)
                wall_next = wall_b is not None and wall_b[0] in ("cabinet",) and wall_b[2] <= config.MAX_FILLER_MM
                if base_next:
                    s = config.TALL_SIDE_SLAB_COUNTER
                    self._fixture(wall, "slab", f"tall_side_{side}_run{i}_counter", a, a + P, s["y"], s["z"], s["thickness"],
                                  s["depth"], level="tall", extraDepthM=config.PANEL_T_M)
                    s = self._wall_zone(config.TALL_SIDE_SLAB_WALL if wall_next else dict(config.WALL_SLAB, depth=0.578, z=0.339))
                    self._fixture(wall, "slab", f"tall_side_{side}_run{i}_upper", a, a + P, s["y"], s["z"], s["thickness"],
                                  s["depth"], level="tall", extraDepthM=config.PANEL_T_M)
                else:
                    s = self._tall_zone(config.TALL_SIDE_SLAB_FULL)
                    self._fixture(wall, "slab", f"tall_side_{side}_run{i}_full", a, a + P, s["y"], s["z"], s["thickness"],
                                  s["depth"], level="tall", extraDepthM=config.PANEL_T_M)

        # ---- wall units: slabs, fillers, upper backsplash ----------------------------
        wall_ps = [p for p in lay.on(wall.id, "wall") if p.tier == "wall"]
        occupied: list[tuple[float, float]] = [(p.start_mm, p.end_mm) for p in lay.on(wall.id, "wall")]
        for i, run in enumerate(self._runs(wall_ps)):
            for side in ("lo", "hi"):
                kind, target, dist = self._end(wall, "wall", run, side)
                x = run[0].start_mm if side == "lo" else run[-1].end_mm
                end_unit = run[0] if side == "lo" else run[-1]
                depth = end_unit.item.depth / 1000.0
                if kind == "filler" or (kind == "neighbor" and dist > EPS):
                    a, b = (target, x) if side == "lo" else (x, target)
                    wf = self._wall_zone(config.WALL_FILLER)
                    front_z = face + depth + config.PANEL_T_M / 2
                    self._fixture(wall, "filler", f"wall_{side}_run{i}", a, b, wf["y"], front_z, wf["thickness"],
                                  config.PANEL_T_M, level="wall", frontShiftM=0)
                    self._fixture(wall, "filler", f"wall_{side}_run{i}_bottom", a, b, wf["bottom_y"], face + depth / 2,
                                  config.PANEL_T_M, depth, level="wall", frontShiftM=0)
                    occupied.append((min(a, b), max(a, b)))
                elif kind == "exposed":
                    a = x - P if side == "lo" else x
                    d = depth + config.PANEL_T_M
                    ws = self._wall_zone(config.WALL_SLAB)
                    self._fixture(wall, "slab", f"wall_{side}_run{i}", a, a + P, ws["y"], face + d / 2, ws["thickness"], d,
                                  level="wall", frontShiftM=0, extraDepthM=config.PANEL_T_M)
                    occupied.append((a, a + P))
            for j, (a, b) in enumerate(zip(run, run[1:])):
                if b.start_mm - a.end_mm > EPS:
                    wf = self._wall_zone(config.WALL_FILLER)
                    self._fixture(wall, "filler", f"wall_mid_run{i}_{j}", a.end_mm, b.start_mm, wf["y"], wf["z"],
                                  wf["thickness"], config.PANEL_T_M, level="wall", frontShiftM=0)
                    occupied.append((a.end_mm, b.start_mm))
        # upper backsplash: counter spans not covered by wall-tier items or windows
        bk = config.BACKSPLASH
        up = self._wall_zone({"y": bk["upper_y"], "thickness": bk["thickness"]})
        # the perpendicular wall's corner cabinets occupy the wall-tier corner zone
        occupied += [(f.lo, f.hi) for f in lay.features.get((wall.id, "wall"), []) if f.kind == "corner"]
        for k, (c0, c1) in enumerate(counter_spans):
            spans = [(c0, c1)]
            for a, b in occupied:
                spans = _cut(spans, a, b)
            for w in wall.windows:
                if w.top_m > up["y"] - up["thickness"] / 2 and w.bottom_m < up["y"] + up["thickness"] / 2:
                    spans = _cut(spans, w.start_mm, w.end_mm)
            for m, (a, b) in enumerate(spans):
                if b - a >= 10:
                    self._fixture(wall, "backsplash", f"upper{k}_{m}", a, b, up["y"], bk["z"], up["thickness"], bk["depth"])

        # ---- lofts -----------------------------------------------------------------
        loft_ps = lay.on(wall.id, "loft")
        if loft_ps:
            # loft zone from the height plan (R48): bottom panel on the wall run, carcass, then the reveal
            hp = lay.heights or {}
            y0 = hp.get("loft_y_mm", round(config.POS_Y_M["loft"] * 1000)) / 1000.0
            lh = hp.get("loft_mm", 600) / 1000.0
            loft_top = y0 + lh
            top_gap = max(0.0, wall.height_m - loft_top)
            lb = dict(config.LOFT_BOTTOM, y=round(y0 - config.LOFT_BOTTOM["thickness"] / 2, 4))
            ls = dict(config.LOFT_SIDE, y=round(y0 + lh / 2, 4), thickness=lh)
            for i, run in enumerate(self._runs(loft_ps)):
                # bottom / top panels per contiguous same-depth segment
                segs: list[list[Placement]] = []
                for p in run:
                    if segs and segs[-1][-1].item.depth == p.item.depth:
                        segs[-1].append(p)
                    else:
                        segs.append([p])
                for k, seg in enumerate(segs):
                    d = seg[0].item.depth / 1000.0
                    a, b = seg[0].start_mm, seg[-1].end_mm
                    self._fixture(wall, "slab", f"loft_bottom_run{i}_{k}", a, b, lb["y"], face + (d + 0.018) / 2,
                                  lb["thickness"], d + config.PANEL_T_M, level="loft")
                    if top_gap > 0.001:
                        self._fixture(wall, "filler", f"loft_top_run{i}_{k}", a, b, loft_top + top_gap / 2,
                                      face + d + 0.009, top_gap, config.PANEL_T_M, level="loft")
                    # shallow/deep junction ("bank side")
                    if k > 0:
                        prev = segs[k - 1]
                        dp, dc = prev[0].item.depth / 1000.0, d
                        if abs(dp - dc) > 1e-6:
                            shallow, deep_d = min(dp, dc), max(dp, dc)
                            x = seg[0].start_mm
                            a2 = x - P if dc > dp else x
                            span = deep_d - shallow  # from shallow front to deep front (shutters included)
                            zc = face + shallow + config.PANEL_T_M + span / 2
                            self._fixture(wall, "slab", f"loft_bank_side_run{i}_{k}", a2, a2 + P, ls["y"], zc, ls["thickness"],
                                          span, level="loft")
                for side in ("lo", "hi"):
                    kind, target, dist = self._end(wall, "loft", run, side)
                    x = run[0].start_mm if side == "lo" else run[-1].end_mm
                    end_unit = run[0] if side == "lo" else run[-1]
                    d = end_unit.item.depth / 1000.0
                    if kind == "filler":
                        a, b = (target, x) if side == "lo" else (x, target)
                        self._fixture(wall, "filler", f"loft_{side}_run{i}", a, b, ls["y"], face + d + 0.009, ls["thickness"],
                                      config.PANEL_T_M, level="loft")
                    elif kind == "exposed":
                        a = x - P if side == "lo" else x
                        dd = d + config.PANEL_T_M
                        self._fixture(wall, "slab", f"loft_side_{side}_run{i}", a, a + P, ls["y"], face + dd / 2, ls["thickness"],
                                      dd, level="loft")
                        if top_gap > 0.001:
                            self._fixture(wall, "filler", f"loft_side_top_{side}_run{i}", a, a + P,
                                          loft_top + top_gap / 2, face + dd / 2, top_gap, dd, level="loft")

    def _counter_edge(self, wall: Wall, end, side: str, P: float) -> float:
        kind, target, x = end
        sign = -1 if side == "lo" else 1
        if kind == "exposed":
            return x + sign * P
        if kind in ("filler", "flush"):
            f = self._corner_feature(wall, "base", target)
            if f is not None:
                # Stop at the perpendicular countertop's front edge (590 mm deep).
                overhang = (config.COUNTERTOP["depth"] * 1000.0) - config.CORNER_DEPTH_MM
                return target - sign * overhang
            return target
        return target if kind == "neighbor" else x

    def _holes(self, wall: Wall, run: list[Placement], ct_center_mm: float) -> list[dict]:
        holes = []
        ct_z = config.COUNTERTOP["z"] + self._zshift(wall)
        for p in run:
            kind = {"sink": "sink", "hob": "hob"}.get(p.role)
            if kind is None:
                continue
            cfg = config.appliance_for(kind, p.item.width)
            w, _, d = cfg["dims_mm"]
            inset = cfg["cutout_inset_mm"]
            hw, hd = (w - 2 * inset) / 2000.0, (d - 2 * inset) / 2000.0
            cx = (p.center_mm - ct_center_mm) / 1000.0
            cz = wall.side_sign * (p.pos_z(wall) - ct_z)
            holes.append({"outline": _rounded_rect(cx - hw, cx + hw, cz - hd, cz + hd, 0.015)})
        return holes

    def _add_floor(self) -> None:
        if not self.include_floor or not self.g.level_id or self.g.level_id not in self.nodes:
            return
        if any((n.get("metadata") or {}).get("fixtureKind") == "floor" for n in self.nodes.values()):
            return
        xs = [p[0] for p in self.g.outline]
        zs = [p[1] for p in self.g.outline]
        w, d = max(xs) - min(xs), max(zs) - min(zs)
        meta = {
            "fixtureKind": "floor",
            "fixtureId": "fixture_floor_room",
            "outline": [list(p) for p in self.g.outline],
            "widthM": w,
            "thicknessM": 0.01,
            "depthM": d,
            "color": config.FIXTURE_COLORS["floor"],
            "materialSlot": "floor",
            "designSource": config.DESIGN_SOURCE_FIXTURE,
            "fromDesignResponse": True,
        }
        tex = self.presets.get("fixtureTextures", {}).get("floor")
        if tex:
            meta["textures"] = copy.deepcopy(tex)
        node = {
            "object": "node",
            "id": self.ids.item(),
            "type": "item",
            "name": "Floor",
            "parentId": self.g.level_id,
            "visible": True,
            "metadata": meta,
            "position": [(min(xs) + max(xs)) / 2, -0.005, (min(zs) + max(zs)) / 2],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1],
            "children": [],
            "wallT": 0,
            "asset": {
                "id": "fixture_floor_room",
                "category": "fixture",
                "name": "floor",
                "thumbnail": "",
                "source": "library",
                "src": "asset://fixture/floor",
                "dimensions": [w, 0.01, d],
                "offset": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
            },
        }
        self._attach(node, self.g.level_id)

    # -- main --------------------------------------------------------------------------
    def hydrate(self) -> dict:
        order = {w: i for i, w in enumerate(self.g.wall_order)}
        tier_order = {"base": 0, "tall": 1, "wall": 2, "loft": 3}
        for p in sorted(self.layout.placements, key=lambda p: (order.get(p.wall_id, 99), tier_order[p.tier], p.start_mm)):
            node = self._cabinet_node(p)
            self._attach(node, p.wall_id)
            self.cab_ids[id(p)] = node["id"]
        self._add_appliances()
        for wid in self.layout.cabinet_walls:
            self._add_wall_fixtures(self.g.walls[wid])
        self._add_floor()
        self.doc["exportedAt"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return self.doc


def _facing(wall: Wall) -> list[float]:
    """Cabinet / chimney rotation so the front faces the room.

    Catalogue models face -z; [0, pi, 0] turns them to the wall's front (+z). When the
    room is on the wall's back side they keep [0, 0, 0] and sit at negative z.
    """
    return list(config.CABINET_ROTATION) if wall.front_interior else [0, 0, 0]


def _cut(spans: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    out = []
    for lo, hi in spans:
        if b <= lo or a >= hi:
            out.append((lo, hi))
            continue
        if a > lo:
            out.append((lo, a))
        if b < hi:
            out.append((b, hi))
    return out


def _rounded_rect(x0: float, x1: float, z0: float, z1: float, r: float, seg: int = 8) -> list[list[float]]:
    """Closed outline (production ordering: starts bottom-right, counter-clockwise arcs)."""
    pts = []
    corners = [
        (x1 - r, z0 + r, -90.0),
        (x1 - r, z1 - r, 0.0),
        (x0 + r, z1 - r, 90.0),
        (x0 + r, z0 + r, 180.0),
    ]
    for cx, cz, a0 in corners:
        for k in range(seg + 1):
            a = math.radians(a0 + 90.0 * k / seg)
            pts.append([cx + r * math.cos(a), cz + r * math.sin(a)])
    return pts


def hydrate_scene(doc: dict, layout: KitchenLayout, seed: Optional[int] = None, include_floor: bool = True) -> tuple[dict, dict]:
    """Return ``(hydrated_doc, node_stats)``."""
    h = SceneHydrator(doc, layout, seed if seed is not None else scene_seed(doc), include_floor=include_floor)
    out = h.hydrate()
    return out, h.stats
