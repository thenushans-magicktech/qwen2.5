"""End-to-end verification of the AI kitchen layout generator.

Strips every ``item_*`` node from the production sample
``magickhomes-scene_2026-11-18_10-38-22.json`` and regenerates the kitchen, then
checks the output JSON *independently* of the solver's internal state:

* magickhomes-scene v1 schema and node-graph integrity,
* every cabinet AID exists in ``Finalised_Asset_Catalogue_Sheet.xlsx``,
* Base Sink Unit centred under ``window_a6jgraczhlas4vwp`` with child ``Sink90`` (APS02.glb),
* Base Hob Unit with child ``Hob90`` (APH02.glb) and ``Chimney90`` (APC02.glb) at y = 1.44,
* zero overlap with ``door_6a3cugqojq1bwey8`` (+100 mm clearance) and zero 3D cabinet collisions.

AI mode: deterministic fallback by default (CI-safe). Use the live local model with
    python test_pipeline.py --ai-mode ollama        (or KITCHEN_TEST_AI_MODE=ollama pytest)
``--ai-mode auto`` tries Ollama and falls back. The mocked-Ollama tests always run.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ai_designer  # noqa: E402
from ai_designer import KitchenPreferences  # noqa: E402
from main import generate_kitchen  # noqa: E402
from scene_parser import strip_items  # noqa: E402

PRIMARY = ROOT / "magickhomes-scene_2026-11-18_10-38-22.json"
SECONDARY = ROOT / "magickhomes-scene_2026-11-18_10-24-05.json"
XLSX = ROOT / "Finalised_Asset_Catalogue_Sheet.xlsx"
OUT_DIR = ROOT / "test_output"
WINDOW_ID = "window_a6jgraczhlas4vwp"
DOOR_ID = "door_6a3cugqojq1bwey8"
W1, W2, W3 = "wall_1oy5ecjq6k11pfoo", "wall_56i56ymve0xf69fm", "wall_49fdugi4g7vj12pf"
PI = math.pi
TOL = 1e-6

AI_MODE = os.getenv("KITCHEN_TEST_AI_MODE", "fallback")


# ---------------------------------------------------------------------------
# Independent helpers (do not reuse solver internals)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def xlsx_aids() -> frozenset[str]:
    import pandas as pd

    xls = pd.ExcelFile(XLSX)
    aids = set()
    for sheet in ("Kitchen Combinations - Slab", "Kitchen Combinations - Shaker 1", "Kitchen Combinations - Common"):
        df = pd.read_excel(xls, sheet)
        col = next(c for c in df.columns if str(c).replace(" ", "") == "AIDCombinedAssetID")
        aids |= {str(a).strip() for a in df[col].dropna()}
    return frozenset(aids)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_plan(path: Path = PRIMARY) -> dict:
    return strip_items(load(path), lambda n: n.get("type") == "item" and n["id"].startswith("item_"))


class Frame:
    """MagickHome wall-local -> world transform, derived straight from the JSON.

    Local +x runs start -> end and local +z is the wall's FRONT side, (-dz, dx) in world
    x/z. ``sign`` is +1 when the room is on the front (``frontSide == "interior"``) and
    -1 when it is on the back, so cabinets must sit at ``sign * (face + depth / 2)``.
    """

    def __init__(self, wall: dict, outline, centroid):
        (x1, z1), (x2, z2) = wall["start"], wall["end"]
        self.length = math.hypot(x2 - x1, z2 - z1)
        self.o = (x1, z1)
        self.d = ((x2 - x1) / self.length, (z2 - z1) / self.length)
        self.n = (-self.d[1], self.d[0])  # front normal = local +z
        fs, bs = str(wall.get("frontSide", "")).lower(), str(wall.get("backSide", "")).lower()
        if fs == "interior" or bs == "exterior":
            self.sign = 1
        elif bs == "interior" or fs == "exterior":
            self.sign = -1
        else:  # no side info: probe the room outline, else the centroid
            mid = ((x1 + x2) / 2, (z1 + z2) / 2)
            probe = (mid[0] + self.n[0] * 0.25, mid[1] + self.n[1] * 0.25)
            if outline:
                self.sign = 1 if point_in_polygon(probe, outline) else -1
            else:
                self.sign = 1 if (centroid[0] - mid[0]) * self.n[0] + (centroid[1] - mid[1]) * self.n[1] >= 0 else -1
        self.inward = (self.n[0] * self.sign, self.n[1] * self.sign)

    def world(self, x, z):
        return (self.o[0] + self.d[0] * x + self.n[0] * z, self.o[1] + self.d[1] * x + self.n[1] * z)


def point_in_polygon(pt, poly) -> bool:
    x, z = pt
    inside = False
    for i in range(len(poly)):
        (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % len(poly)]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1) + x1:
            inside = not inside
    return inside


def room_outline(nodes: dict):
    for kind in ("slab", "ceiling"):
        for n in nodes.values():
            if n["type"] == kind and isinstance(n.get("polygon"), list) and len(n["polygon"]) >= 3:
                return [tuple(p) for p in n["polygon"]]
    return None


def frames(nodes: dict) -> dict[str, Frame]:
    walls = {k: n for k, n in nodes.items() if n["type"] == "wall"}
    pts = [p for w in walls.values() for p in (w["start"], w["end"])]
    c = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    outline = room_outline(nodes)
    return {k: Frame(w, outline, c) for k, w in walls.items()}


def sat(a, b, tol=1e-4) -> bool:
    for poly in (a, b):
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            ax, az = -(q[1] - p[1]), q[0] - p[0]
            ln = math.hypot(ax, az) or 1.0
            ax, az = ax / ln, az / ln
            pa = [v[0] * ax + v[1] * az for v in a]
            pb = [v[0] * ax + v[1] * az for v in b]
            if max(pa) <= min(pb) + tol or max(pb) <= min(pa) + tol:
                return False
    return True


def box(node: dict, fr: Frame):
    """World footprint + y-span of a wall-parented item (position = bottom-centre + offset)."""
    w, h, d = node["asset"]["dimensions"]
    x, y, z = node["position"]
    oy = node["asset"]["offset"][1]
    y0 = y + oy - h / 2
    poly = [fr.world(x - w / 2, z - d / 2), fr.world(x + w / 2, z - d / 2), fr.world(x + w / 2, z + d / 2), fr.world(x - w / 2, z + d / 2)]
    return poly, (y0, y0 + h)


def cabinets(nodes):
    return [n for n in nodes.values() if n["type"] == "item" and n["asset"]["category"] == "kitchen"]


def by_category(nodes, category):
    return [n for n in cabinets(nodes) if n["metadata"]["category"] == category]


def interval(n):
    x, w = n["position"][0], n["asset"]["dimensions"][0]
    return x - w / 2, x + w / 2


def appliance_cfg(kind, host_node):
    """The appliance model expected on / above a host cabinet node (by its width)."""
    import config

    return config.appliance_for(kind, round(host_node["asset"]["dimensions"][0] * 1000))


# ---------------------------------------------------------------------------
# The verification suite (returns a list of failures so it can run anywhere)
# ---------------------------------------------------------------------------

def verify_scene(doc: dict, expect_window: str | None = WINDOW_ID, expect_door: str | None = DOOR_ID) -> list[str]:
    errs: list[str] = []
    E = errs.append

    # -- schema -----------------------------------------------------------------
    if doc.get("format") != "magickhomes-scene":
        E(f"format is {doc.get('format')!r}")
    if doc.get("version") != 1:
        E(f"version is {doc.get('version')!r}")
    if not isinstance(doc.get("exportedAt"), str):
        E("exportedAt missing")
    scene = doc.get("scene") or {}
    nodes = scene.get("nodes")
    if not isinstance(nodes, dict):
        return errs + ["scene.nodes missing"]
    for r in scene.get("rootNodeIds", []):
        if r not in nodes:
            E(f"rootNodeId {r} missing")
    for nid, n in nodes.items():
        if n.get("id") != nid or n.get("object") != "node":
            E(f"node {nid}: id/object mismatch")
        for c in n.get("children", []):
            if c not in nodes:
                E(f"{nid}: child {c} missing")
            elif nodes[c]["type"] == "item" and nodes[c]["parentId"] != nid:
                E(f"{c}: parentId {nodes[c]['parentId']} but listed under {nid}")
        if n["type"] == "item":
            p = n.get("parentId")
            if p not in nodes or nid not in nodes[p].get("children", []):
                E(f"{nid}: not registered in parent {p}.children")
            for k in ("position", "rotation", "scale", "asset", "metadata"):
                if k not in n:
                    E(f"{nid}: missing {k}")
            a = n["asset"]
            if len(a.get("dimensions", [])) != 3 or min(a["dimensions"]) <= 0:
                E(f"{nid}: bad dimensions {a.get('dimensions')}")
            if any(isinstance(v, float) and not math.isfinite(v) for v in n["position"]):
                E(f"{nid}: non-finite position")

    fr = frames(nodes)
    walls = {k: n for k, n in nodes.items() if n["type"] == "wall"}
    cabs = cabinets(nodes)
    if not cabs:
        return errs + ["no cabinets generated"]

    # -- cabinet nodes ------------------------------------------------------------
    aids = xlsx_aids()
    y_by_level = {"tall": 0.1, "wall": 1.44, "loft": 2.058}
    for n in cabs:
        aid = n["metadata"]["aid"]
        a = n["asset"]
        if aid not in aids:
            E(f"{n['id']}: AID {aid} not in Finalised_Asset_Catalogue_Sheet.xlsx")
        if n["name"] != aid or a["name"] != aid:
            E(f"{n['id']}: name/asset.name != aid")
        api_url = n["metadata"].get("modelUrl")  # present when the catalogue came from the Assets API
        want_src = api_url or f"https://aid-assertcatalogue-dev.s3.ap-south-1.amazonaws.com/catalogue-models/{aid}.glb"
        if a["src"] != want_src:
            E(f"{n['id']}: asset.src {a['src']} != API modelUrl {want_src}")
        if not a["src"].endswith(".glb"):
            E(f"{n['id']}: asset.src is not a .glb")
        if n["parentId"] != n["wallId"] or n["wallId"] not in walls:
            E(f"{n['id']}: parentId/wallId mismatch")
        want_rot = [0, PI, 0] if fr[n["wallId"]].sign > 0 else [0, 0, 0]  # front faces the room
        if n["rotation"] != want_rot or n["scale"] != [1, 1, 1] or a["scale"] != [0.01, 0.01, 0.01]:
            E(f"{n['id']}: rotation/scale mismatch")
        if n["placement"].get("side") != fr[n["wallId"]].sign:
            E(f"{n['id']}: placement.side {n['placement'].get('side')} but the room is on side {fr[n['wallId']].sign}")
        w, h, d = a["dimensions"]
        if abs(a["offset"][1] - h / 2) > TOL:
            E(f"{n['id']}: offset {a['offset']} != h/2")
        lvl = n["placement"]["level"]
        if n["placement"].get("designSource") != "ai-kitchen-v1":
            E(f"{n['id']}: placement block {n['placement']}")
        want_y = (0.0 if n["metadata"]["height"] >= 800 else 0.1) if lvl == "base" else y_by_level[lvl]
        if abs(n["position"][1] - want_y) > TOL:
            E(f"{n['id']}: {lvl} y={n['position'][1]} expected {want_y}")
        face = walls[n["wallId"]].get("thickness", 0.1) / 2  # back of cabinet on the wall face
        want_z = fr[n["wallId"]].sign * (face + d / 2)  # 0.33 / 0.215 / 0.375 on the room side
        if abs(n["position"][2] - want_z) > TOL:
            E(f"{n['id']}: z={n['position'][2]} expected {want_z}")
        L = fr[n["wallId"]].length
        L_mm = round(L * 1000)  # spec: wallT = center_mm / round(hypot(dx, dz) * 1000)
        if abs(n["wallT"] - n["position"][0] * 1000 / L_mm) > 1e-9:
            E(f"{n['id']}: wallT inconsistent")
        lo, hi = interval(n)
        if lo < -TOL or hi > L + TOL:
            E(f"{n['id']}: outside wall [0, {L}]")

    # -- every cabinet stands inside the room (MagickHome wall frames + room outline) -----
    outline = room_outline(nodes)
    if outline:
        for n in cabs:
            poly, _ = box(n, fr[n["wallId"]])
            cx = sum(p[0] for p in poly) / 4
            cz = sum(p[1] for p in poly) / 4
            shrunk = [(cx + (p[0] - cx) * 0.98, cz + (p[1] - cz) * 0.98) for p in poly]
            if not all(point_in_polygon(p, outline) for p in shrunk):
                E(f"{n['metadata']['category']} {n['id']} on {walls[n['wallId']].get('name')} is outside the room")

    # -- sink under window ------------------------------------------------------------
    sinks = by_category(nodes, "Base Sink Unit")
    if len(sinks) != 1:
        E(f"expected 1 Base Sink Unit, got {len(sinks)}")
    else:
        s = sinks[0]
        if expect_window:
            win = nodes[expect_window]
            if s["wallId"] != win["wallId"]:
                E(f"sink on {s['wallId']} but {expect_window} is on {win['wallId']}")
            if abs(s["position"][0] - win["position"][0]) > 0.001:
                E(f"sink centre {s['position'][0]:.4f} not under window {win['position'][0]:.4f}")
        kids = [nodes[c] for c in s["children"]]
        want = appliance_cfg("sink", s)  # the sink model that fits this cabinet (Sink60 / 90 / 120)
        if len(kids) != 1 or kids[0]["name"] != want["name"] or kids[0]["asset"]["src"] != want["src"]:
            E(f"sink children wrong: {[k['name'] for k in kids]} (expected {want['name']})")
        else:
            k = kids[0]
            if k["parentId"] != s["id"] or k["position"] != [0, 0.55, 0] or k["rotation"] != [0, 0, 0]:
                E("sink appliance not a correctly placed child of the sink unit")
            if k["wallId"] != s["wallId"] or abs(k["wallT"] - s["wallT"]) > 1e-12:
                E("sink appliance wallId/wallT differ from host")

    # -- hob, spice, chimney ------------------------------------------------------------
    hobs = by_category(nodes, "Base Hob Unit")
    chims = [n for n in nodes.values() if n["type"] == "item" and n["name"].startswith("Chimney")]
    if len(hobs) != 1:
        E(f"expected 1 Base Hob Unit, got {len(hobs)}")
    else:
        hb = hobs[0]
        kids = [nodes[c] for c in hb["children"]]
        want = appliance_cfg("hob", hb)
        if len(kids) != 1 or kids[0]["name"] != want["name"] or kids[0]["asset"]["src"] != want["src"]:
            E(f"hob children wrong: {[k['name'] for k in kids]} (expected {want['name']})")
        elif kids[0]["position"] != [0, 0.74, 0] or kids[0]["parentId"] != hb["id"]:
            E("hob appliance not a correctly placed child of the hob unit")
        if len(chims) != 1:
            E(f"expected 1 chimney, got {len(chims)}")
        else:
            c = chims[0]
            if c["asset"]["src"] != appliance_cfg("chimney", hb)["src"]:
                E("chimney src wrong")
            if c["parentId"] != hb["wallId"] or c["wallId"] != hb["wallId"] or c["id"] not in walls[hb["wallId"]]["children"]:
                E("chimney not a child of the hob's wall")
            cz = 0.32 + walls[hb["wallId"]].get("thickness", 0.1) / 2 - 0.05
            if abs(c["position"][0] - hb["position"][0]) > TOL or abs(c["position"][1] - 1.44) > TOL or abs(c["position"][2] - cz) > TOL:
                E(f"chimney position {c['position']} not above hob x={hb['position'][0]}")
            if c["rotation"] != [0, PI, 0] or abs(c["wallT"] - hb["wallT"]) > 1e-12:
                E("chimney rotation/wallT wrong")
            # chimney zone must be free of wall-tier cabinets
            cx, half = c["position"][0], c["asset"]["dimensions"][0] / 2  # hood width: 600 or 900
            for n in cabs:
                if n["wallId"] == hb["wallId"] and n["placement"]["level"] in ("wall", "tall"):
                    lo, hi = interval(n)
                    if lo < cx + half - TOL and hi > cx - half + TOL:
                        E(f"{n['metadata']['category']} {n['id']} obstructs the chimney")
        spice = [n for n in by_category(nodes, "Base Spice Pull Out") if n["wallId"] == hb["wallId"]]
        h_lo, h_hi = interval(hb)
        if not any(abs(interval(s)[1] - h_lo) < 0.001 or abs(interval(s)[0] - h_hi) < 0.001 for s in spice):
            E("no Base Spice Pull Out immediately adjacent to the hob")

    # -- doors -----------------------------------------------------------------------
    doors = [n for n in nodes.values() if n["type"] == "door"]
    if expect_door and expect_door not in nodes:
        E(f"{expect_door} missing")
    for dnode in doors:
        dlo = dnode["position"][0] - dnode["width"] / 2 - 0.1
        dhi = dnode["position"][0] + dnode["width"] / 2 + 0.1
        for n in cabs:
            if n["wallId"] == dnode["wallId"]:
                lo, hi = interval(n)
                if lo < dhi - TOL and hi > dlo + TOL:
                    E(f"{n['metadata']['category']} {n['id']} overlaps {dnode['id']} clearance [{dlo:.3f}, {dhi:.3f}]")

    # -- windows: no wall/tall/loft cabinets in front --------------------------------------
    for wn in (n for n in nodes.values() if n["type"] == "window"):
        wlo, whi = wn["position"][0] - wn["width"] / 2, wn["position"][0] + wn["width"] / 2
        for n in cabs:
            if n["wallId"] == wn["wallId"] and n["placement"]["level"] in ("wall", "tall", "loft"):
                lo, hi = interval(n)
                if lo < whi - TOL and hi > wlo + TOL:
                    E(f"{n['metadata']['category']} {n['id']} blocks {wn['id']}")

    # -- zero 3D collisions ---------------------------------------------------------------
    solids = [(n, *box(n, fr[n["wallId"]])) for n in cabs]
    for c in chims:
        solids.append((c, *box(c, fr[c["wallId"]])))
    for i in range(len(solids)):
        a, pa, ya = solids[i]
        for j in range(i + 1, len(solids)):
            b, pb, yb = solids[j]
            if ya[0] >= yb[1] - 1e-4 or yb[0] >= ya[1] - 1e-4:
                continue
            if sat(pa, pb):
                E(f"3D collision: {a['name']} ({a['id']}) x {b['name']} ({b['id']})")

    base_walls = {n["wallId"] for n in cabs if n["placement"]["level"] in ("base",)}

    # -- corner rule: where two runs meet, only corner cabinets (per tier) -------------------
    def is_corner_unit(n):
        c = n["metadata"]["category"].replace("–", "-")
        return c.startswith("Blind corner unit") or "L Corner" in c

    tier_levels = {"base": ("base", "tall"), "wall": ("wall", "tall"), "loft": ("loft",)}
    tier_depth = {"base": 0.56, "wall": 0.33, "loft": 0.33}
    for a in walls:
        for b in walls:
            if a >= b:
                continue
            for end_a in ("start", "end"):
                for end_b in ("start", "end"):
                    if math.dist(walls[a][end_a], walls[b][end_b]) > 0.02:
                        continue
                    fa, fb = fr[a], fr[b]
                    da = fa.d if end_a == "start" else (-fa.d[0], -fa.d[1])  # away from the joint
                    db = fb.d if end_b == "start" else (-fb.d[0], -fb.d[1])
                    if db[0] * fa.inward[0] + db[1] * fa.inward[1] < 0.5 or da[0] * fb.inward[0] + da[1] * fb.inward[1] < 0.5:
                        continue  # outside (reflex) corner or straight joint: no corner unit needed
                    for tier, levels in tier_levels.items():
                        def near(wid, end):
                            L = fr[wid].length
                            out = []
                            for n in cabs:
                                if n["wallId"] != wid or n["placement"]["level"] not in levels:
                                    continue
                                lo, hi = interval(n)
                                d = lo if end == "start" else L - hi
                                if d < 1.2:
                                    out.append((n, lo, hi))
                            return out
                        na, nb = near(a, end_a), near(b, end_b)
                        if not na or not nb:
                            continue  # not a corner between two runs on this tier
                        for wid, end, other, items in ((a, end_a, b, na), (b, end_b, a, nb)):
                            L = fr[wid].length
                            strip = tier_depth[tier] + walls[other].get("thickness", 0.1) / 2 - 0.001
                            for n, lo, hi in items:
                                inside = lo < strip if end == "start" else hi > L - strip
                                if inside and not is_corner_unit(n):
                                    E(f"regular {n['metadata']['category']} {n['id']} in the "
                                      f"{walls[a]['name']}/{walls[b]['name']} corner ({tier} tier)")

    # -- corner models face the right way (from the GLB geometry) -------------------------
    # Seen from inside the room: L-shaped corner units have one hand (corner on the viewer's
    # RIGHT); blind corner "- L" = door left / blind right -> corner on the right; "- R" -> left.
    # On a wall whose room side is the back, the viewer's right is the wall START.
    for n in cabs:
        c = n["metadata"]["category"].replace("–", "-").lower().strip()
        if "l corner" in c:
            hand = "right"
        elif c.startswith("blind corner"):
            hand = "right" if c.endswith("- l") else "left" if c.endswith("- r") else None
        else:
            hand = None
        if hand is None:
            continue
        need = "end" if (hand == "right") == (fr[n["wallId"]].sign > 0) else "start"
        lo, hi = interval(n)
        L = fr[n["wallId"]].length
        at = "start" if lo < 0.2 else "end" if hi > L - 0.2 else "middle"
        if at != need or n["placement"].get("cornerAtEnd") != (need == "end"):
            E(f"{n['metadata']['category']} {n['id']} sits at the wall {at} but its model only fits the wall {need}")

    # -- fixtures --------------------------------------------------------------------------
    fixtures = [n for n in nodes.values() if n["type"] == "item" and n["asset"]["category"] == "fixture"]
    for wid in base_walls:
        kinds = {f["metadata"]["fixtureKind"] for f in fixtures if f.get("wallId") == wid}
        for k in ("toe-kick", "countertop", "backsplash"):
            if k not in kinds:
                E(f"{walls[wid]['name']}: no {k} fixture")
    for f in fixtures:
        if f.get("wallId") and f["id"] not in walls[f["wallId"]]["children"]:
            E(f"fixture {f['id']} not in its wall's children")
    return errs


# ---------------------------------------------------------------------------
# pytest scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "auto": dict(),
    "reference_L_walls": dict(cabinet_walls=[W1, W2]),
    "shaker_U": dict(door_style="Shaker -1", layout="U"),
    "single_wall": dict(layout="single"),
}


def run(prefs: KitchenPreferences, source: Path = PRIMARY, name: str = "out"):
    out, report = generate_kitchen(empty_plan(source), prefs)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"generated_kitchen_{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / f"report_{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, report


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name):
    prefs = KitchenPreferences(ai_mode=AI_MODE, **SCENARIOS[name])
    out, report = run(prefs, name=name)
    errs = verify_scene(out)
    assert not errs, "\n".join(errs)
    assert not report["validation_issues"], report["validation_issues"]
    if name == "reference_L_walls":
        assert set(report["layout"]["cabinet_walls"]) == {W1, W2}
        door_wall_cabs = [n for n in cabinets(out["scene"]["nodes"]) if n["wallId"] == W1]
        assert door_wall_cabs, "door wall must carry cabinets so the door rule is exercised"
        corner = report["layout"]["corners"][0]
        assert corner["base_unit"] and corner["wall_unit"], "the L corner should get corner cabinets"


def test_strip_removes_all_items():
    plan = empty_plan()
    nodes = plan["scene"]["nodes"]
    assert not [n for n in nodes.values() if n["type"] == "item"]
    for n in nodes.values():
        assert all(c in nodes for c in n.get("children", []))


def test_loft_on_tall_walls():
    """Secondary sample has 2.7 m walls: lofts allowed, deep loft above the tall unit."""
    prefs = KitchenPreferences(ai_mode=AI_MODE, include_loft=True, cabinet_walls=[W1, W2])
    out, report = run(prefs, SECONDARY, "loft")
    assert not verify_scene(out), verify_scene(out)
    lofts = [n for n in cabinets(out["scene"]["nodes"]) if n["placement"]["level"] == "loft"]
    assert lofts, "expected loft units"
    for n in lofts:
        top = n["position"][1] + n["asset"]["dimensions"][1]
        assert top <= 2.7 + TOL
    fx = {n["metadata"].get("fixtureId", "") for n in out["scene"]["nodes"].values() if n["type"] == "item"}
    assert any("loft_top" in f for f in fx), "expected loft top fillers up to the 2.7 m wall height"


def test_no_loft_when_wall_too_low():
    prefs = KitchenPreferences(ai_mode="fallback", include_loft=True)
    out, report = run(prefs, PRIMARY, "loft_low")
    assert not [n for n in cabinets(out["scene"]["nodes"]) if n["placement"]["level"] == "loft"]
    assert any("no loft" in d for d in report["layout"]["decisions"])


def _level(out, level):
    return [n for n in cabinets(out["scene"]["nodes"]) if n["placement"]["level"] == level]


def test_heights_follow_the_ceiling():
    """2.5 m room (no lofts possible): 900 wall + 2240 tall units with level tops.
    2.7 m room: lofts by default (standard 600 / 1940 below them); lofts declined -> 900 / 2240."""
    out, report = run(KitchenPreferences(ai_mode="fallback"), PRIMARY, "heights_low")
    assert not verify_scene(out), verify_scene(out)
    assert not report["validation_issues"], report["validation_issues"]
    h = report["heights"]
    assert (h["wall_mm"], h["tall_mm"], h["lofts"]) == (900, 2240, False)
    walls, talls = _level(out, "wall"), _level(out, "tall")
    assert walls and talls and not _level(out, "loft")
    assert all(abs(n["asset"]["dimensions"][1] - 0.9) < 1e-6 for n in walls), "every wall-tier unit is 900 high"
    tops = {round(n["position"][1] + n["asset"]["dimensions"][1], 3) for n in walls + talls}
    assert tops == {2.34}, tops
    ex = {e["item"]: e for e in report["explanation"]}
    assert "900" in ex["Cabinet heights"]["built"] and ex["Lofts"]["built"] == "None"

    out, report = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]), SECONDARY, "heights_loft")
    assert not verify_scene(out), verify_scene(out)
    assert report["heights"]["lofts"] and report["heights"]["wall_mm"] == 600
    assert _level(out, "loft"), "lofts are the default when the ceiling allows"
    assert report["proposal"]["lofts"] is True

    out, report = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2], include_loft=False), SECONDARY, "heights_noloft")
    assert not verify_scene(out), verify_scene(out)
    assert not _level(out, "loft") and report["heights"]["wall_mm"] == 900
    assert next(e for e in report["explanation"] if e["item"] == "Lofts")["source"] == "customer"


def test_lift_up_doors_open_shelf_and_crockery_unit():
    """Stated needs for the new cabinet types are built and explained."""
    prefs = KitchenPreferences(ai_mode="fallback", wall_style="lift_up", open_unit="yes", crockery_unit="yes")
    out, report = run(prefs, PRIMARY, "upper_needs")
    assert not verify_scene(out), verify_scene(out)
    assert not report["validation_issues"], report["validation_issues"]
    nodes = out["scene"]["nodes"]
    assert by_category(nodes, "Wall Lift Up Unit"), "lift-up wall cabinets expected"
    opens = by_category(nodes, "Wall Open Unit")
    assert len(opens) == 1
    assert by_category(nodes, "Tall Glass Unit"), "crockery unit expected"
    chim = report["layout"]["chimney"]
    if chim and chim["wall_id"] == opens[0]["wallId"]:
        lo, hi = interval(opens[0])
        assert min(abs(lo * 1000 - chim["center_mm"]), abs(hi * 1000 - chim["center_mm"])) > 450, "open shelf beside the hob"
    ex = {e["item"]: e for e in report["explanation"]}
    assert ex["Wall cabinet doors"]["source"] == "customer" and "Lift-up" in ex["Wall cabinet doors"]["built"]
    assert ex["Open shelf unit"]["built"].startswith("Open shelf unit")
    assert "Crockery unit" in ex["Tall units"]["built"]

    # the same needs declined: none of them appear
    prefs = KitchenPreferences(ai_mode="fallback", wall_style="shutters", open_unit="no", crockery_unit="no")
    out, _ = run(prefs, PRIMARY, "upper_needs_no")
    nodes = out["scene"]["nodes"]
    assert not by_category(nodes, "Wall Lift Up Unit") and not by_category(nodes, "Wall Open Unit")
    assert not by_category(nodes, "Tall Glass Unit")


def test_rulebook_reads_the_workbook():
    """The MagickHome rule workbook drives module widths; the logic rules match its rows."""
    import rulebook

    rb = rulebook.load()
    assert rb.source.startswith("workbook") and len(rb.texts) >= 81
    want = {("hob", "L-Shape", "Small"): (600, 1), ("sink", "L-Shape", "Medium"): (900, 13),
            ("spice", "L-Shape", "Large"): (300, 19), ("cylinder", "L-Shape", "Small"): (450, 4),
            ("hob", "Parallel", "Large"): (1050, 68), ("hob", "U-Shape", "Medium"): (800, 66)}
    for key, (width, rule) in want.items():
        r = rb.module_width(*key)
        assert (r.width, r.rule) == (width, rule), key
    assert [rulebook.size_band(a)[0] for a in (70, 80, 81, 120, 121, 175)] == ["Small"] * 2 + ["Medium"] * 2 + ["Large"] * 2
    assert rulebook.loft_size(2700, 2040) == (600, 42) and rulebook.loft_size(3000, 2040) == (850, 92)  # R48
    fill = rulebook.base_fill_combination
    assert fill("essential", 450, 2000) == ("Single Shutter", 6) and fill("essential", 800, 2000)[0] == "Double Shutter"
    assert fill("essential", 600, 900) == ("Double Shutter", 31) and fill("essential", 600, 1500) == ("Single Shutter", 32)
    assert fill("premium", 600, None) == ("Double Drawer", 33) and fill("luxury", 500, None)[0] == "Single Shutter"


def test_rulebook_module_widths_lpg_and_one_spice(monkeypatch):
    """Medium L-Shape: R12-R15 set hob 800 / sink 900 / spice 200 / cylinder 500 over the designer's
    choice; R11 adds the cylinder when the fuel is not stated; piped gas drops it; one spice only."""
    import rulebook
    from catalog_loader import load_filtered_catalog
    from scene_parser import parse_scene

    geo = parse_scene(empty_plan())
    cat = load_filtered_catalog()
    monkeypatch.setattr(rulebook, "floor_area_sqft", lambda g: 100.0)  # a Medium kitchen
    groups = ai_designer.build_shortlist_v2(cat)
    prefs = KitchenPreferences(cabinet_walls=[W1, W2])
    prop = ai_designer.fallback_proposal(geo, groups, prefs)
    prop.cylinder_aid = None
    facts = rulebook.apply_module_rules(prop, geo, cat, prefs)
    assert (facts.band, facts.kitchen_type) == ("Medium", "L-Shape")
    got = {role: cat.get(getattr(prop, attr)) for role, attr in (("hob", "hob_aid"), ("sink", "sink_aid"),
                                                                  ("spice", "spice_aid"), ("cylinder", "cylinder_aid"))}
    assert {r: it.width for r, it in got.items()} == {"hob": 800, "sink": 900, "spice": 200, "cylinder": 500}
    assert got["hob"].cabinet_combination == "Triple Drawer - A"
    assert {a["rule"] for a in facts.applied} >= {11, 12, 13, 14, 15}

    monkeypatch.undo()  # the real sample room: 172 sqft, a Large L-Shape (R17-R20)
    out, report = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]), name="rulebook_L")
    assert not verify_scene(out), verify_scene(out)
    nodes = out["scene"]["nodes"]
    assert len(by_category(nodes, "Base Spice Pull Out")) == 1, "R44: one spice pull-out"
    assert by_category(nodes, "Base Cylinder Unit"), "R11: LPG assumed"
    assert report["rulebook"]["kitchen_type"] == "L-Shape" and report["explanation"][0]["item"] == "Rule book"
    hob = next(e for e in report["explanation"] if e["item"] == "Hob")
    assert any(r.startswith("R17:") for r in hob["rules"]), hob["rules"]
    out, _ = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2], include_cylinder=False), name="rulebook_png")
    assert not by_category(out["scene"]["nodes"], "Base Cylinder Unit")


def test_budget_tiers_fill_base_runs_and_pick_corners():
    """Essential: shutter base units, blind wall corner (R75). Premium: double-drawer units from
    600 mm, L wall corner (R71). Plain units are never wider than 900 mm (R7 / R34)."""
    for budget, corner in (("essential", "Blind corner unit"), ("premium", "Wall L Corner")):
        out, report = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2], budget=budget), name=f"budget_{budget}")
        assert not verify_scene(out), verify_scene(out)
        assert not report["validation_issues"], report["validation_issues"]
        nodes = out["scene"]["nodes"]
        feature = {a for aids in report["proposal"]["base_prefs"].values() for a in aids}
        plain = [n for n in by_category(nodes, "Base Unit") if n["metadata"]["aid"] not in feature]
        assert plain and all(n["asset"]["dimensions"][0] <= 0.9 + 1e-6 for n in plain)
        for n in plain:
            w, combo = round(n["asset"]["dimensions"][0] * 1000), n["metadata"]["cabinetCombination"]
            if w < 600:
                assert combo == "Single Shutter", (budget, w, combo)
            elif budget == "essential":
                assert "Drawer" not in combo, (budget, w, combo)
            else:
                assert combo == "Double Drawer", (budget, w, combo)
        wall_corners = [n["metadata"]["category"] for n in cabinets(nodes)
                        if n["placement"]["level"] == "wall" and "orner" in n["metadata"]["category"]]
        assert wall_corners and all(c.startswith(corner) for c in wall_corners), (budget, wall_corners)
        base = next(e for e in report["explanation"] if e["item"] == "Base cabinets")
        assert base["source"] == "customer" and base["rules"]


def test_rulebook_filters_endpoints_and_overrides():
    """The UI's rule-book filters: options with their rules, plan facts, a size-band override
    (Small L-Shape -> hob 600 / sink 600, R1 / R2) and the fuel choice (PNG drops the cylinder)."""
    from fastapi.testclient import TestClient

    import main

    c = TestClient(main.app)
    o = c.get("/api/rulebook").json()
    assert [b["value"] for b in o["budget"]] == ["auto", "essential", "premium", "luxury"]
    assert [f["value"] for f in o["fuel"]] == ["auto", "lpg", "png", "induction"]
    assert o["module_widths"]["L-Shape"]["Small"]["hob"] == {"width": 600, "combination": "Triple Drawer - A", "rule": 1}
    assert any(r.startswith("R75:") for r in o["budget"][1]["rules"])
    f = c.post("/api/plan-facts", json={"scene": empty_plan()}).json()
    assert f["size_band"] == "Large" and f["ceiling_mm"] == 2500 and not f["lofts_possible"] and len(f["walls"]) == 4

    out, report = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2], size_band="small", fuel="png"),
                      name="small_png")
    assert not verify_scene(out), verify_scene(out)
    assert report["rulebook"]["size_band"] == "Small" and "set to Small" in report["rulebook"]["size_note"]
    nodes = out["scene"]["nodes"]
    hob, sink = by_category(nodes, "Base Hob Unit")[0], by_category(nodes, "Base Sink Unit")[0]
    assert round(hob["asset"]["dimensions"][0] * 1000) == 600 and round(sink["asset"]["dimensions"][0] * 1000) == 600
    assert [nodes[k]["name"] for k in hob["children"]] == ["Hob60"] and [nodes[k]["name"] for k in sink["children"]] == ["Sink60"]
    assert not by_category(nodes, "Base Cylinder Unit"), "piped gas: no cylinder unit"


def test_finishes_follow_the_budget_tier():
    """Essential = laminate only, Premium = at most one PU run, Luxury = PU; the customer's finish
    wins; the scene applies the run's finish first on cabinets and on their panels and fillers."""
    import finishes

    lam = {a for a, r in finishes.catalogue().items() if r["category"] == "Laminate"}
    assert set(finishes.allowed("base", "essential")) == lam
    assert {"MHS224", "MHS212"} <= set(finishes.allowed("wall", "luxury"))
    assert all(finishes.is_premium(a) for a in finishes.allowed("wall", "luxury"))
    codes = ai_designer.finish_codes("base", KitchenPreferences(budget="essential"))
    assert codes and all("-LAMINATE" in c for c in codes) and len(codes) <= 30, "a shortlist, not every finish"
    assert ai_designer.finish_codes("wall", KitchenPreferences(wall_finish="MHS064")) == [finishes.code("MHS064")]

    pal, src, notes = finishes.resolve({}, {"base": "MHS224", "wall": "MHS212"}, "premium")  # two PU runs
    assert sum(finishes.is_premium(a) for a in pal.values()) == 1 and notes
    pal, src, notes = finishes.resolve({}, {"base": "MHS224"}, "essential")  # PU not allowed in Essential
    assert finishes.catalogue()[pal["base"]]["category"] == "Laminate" and src["base"] == "rules" and notes
    pal, src, _ = finishes.resolve({"base": "MHS224"}, {}, "essential")  # the customer's choice wins
    assert pal["base"] == "MHS224" and src["base"] == "customer"

    out, report = run(KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2], budget="essential"), name="finish_essential")
    assert not verify_scene(out), verify_scene(out)
    want = finishes.rules()["defaults"]["essential"]
    assert (report["finishes"]["base"]["aid"], report["finishes"]["wall"]["aid"]) == (want["base"], want["wall"])
    nodes = out["scene"]["nodes"].values()
    for n in nodes:
        if n["type"] != "item" or not (n["metadata"].get("textures")):
            continue
        level = n["placement"]["level"] if n["asset"]["category"] == "kitchen" else n["metadata"].get("level")
        run_ = finishes.RUN_OF_TIER.get(level or "")
        if run_ and (n["asset"]["category"] == "kitchen" or n["metadata"].get("fixtureKind") in ("slab", "filler")):
            assert n["metadata"]["textures"][0]["aid"] == want[run_], (level, n["metadata"].get("fixtureKind"))
    fin = next(e for e in report["explanation"] if e["item"] == "Finishes")
    assert "Laminate" in fin["built"] and fin["rules"][0].startswith("Essential tier")


def test_finishes_change_after_the_design_and_surfaces():
    """Worktop / backsplash choices reach the scene; POST /api/jobs/{id}/finishes re-applies new
    finishes to a finished design (fronts, their panels, worktop, backsplash) and its report."""
    from fastapi.testclient import TestClient

    import finishes
    import main

    def first(out, pred):
        return {n["metadata"]["textures"][0]["aid"] for n in out["scene"]["nodes"].values()
                if n["type"] == "item" and n["metadata"].get("textures") and pred(n)}

    surf = lambda k: lambda n: n["metadata"].get("fixtureKind") == k  # noqa: E731
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="fallback", backsplash_finish="MHB021"))
    assert first(out, surf("backsplash")) == {"MHB021"} and report["finishes"]["backsplash"]["source"] == "customer"
    assert first(out, surf("countertop")) == {"MHC015"}

    c = TestClient(main.app)
    job = main.JOBS.submit(empty_plan(), KitchenPreferences(ai_mode="fallback", budget="premium"), "t.json")
    for _ in range(200):
        if main.JOBS.get(job["id"])["status"] in ("done", "error"):
            break
        import time
        time.sleep(0.1)
    r = c.post(f"/api/jobs/{job['id']}/finishes", json={"base_finish": "MHS064", "backsplash_finish": "MHB038"})
    assert r.status_code == 200 and r.json()["changed_nodes"] > 0
    scene = c.get(f"/api/jobs/{job['id']}/scene").json()
    base_run = lambda n: n["asset"]["category"] == "kitchen" and n["placement"]["level"] in ("base", "tall")  # noqa: E731
    assert first(scene, base_run) == {"MHS064"} and first(scene, surf("backsplash")) == {"MHB038"}
    assert first(scene, lambda n: n["metadata"].get("fixtureKind") in ("slab", "filler") and n["metadata"].get("level") in ("base", "tall")) <= {"MHS064"}
    rep = c.get(f"/api/jobs/{job['id']}/report").json()
    assert rep["finishes"]["base"]["aid"] == "MHS064" and rep["finishes"]["base"]["source"] == "customer"
    assert next(e for e in rep["explanation"] if e["item"] == "Finishes")["why"].startswith("Changed by the customer")
    assert {f["aid"] for f in c.get("/api/finishes").json()["surfaces"]["backsplash"]} == set(finishes.surface_options("backsplash"))


def test_quick_rules_never_wait_behind_an_ai_design(monkeypatch):
    """A slow AI (Qwen on CPU) design must not hold up a quick-rules design: separate lanes."""
    import threading
    import time

    import main

    release = threading.Event()
    real = main.generate_kitchen

    def slow_or_real(doc, prefs, progress=None):
        if prefs.ai_mode != "fallback":
            release.wait(20)  # stands in for a multi-minute Qwen design (never contacts Qwen)
            return real(doc, prefs.model_copy(update={"ai_mode": "fallback"}), progress)
        return real(doc, prefs, progress)

    monkeypatch.setattr(main, "generate_kitchen", slow_or_real)
    store = main.JobStore()
    monkeypatch.setattr(store, "_save", lambda job: None)
    try:
        ai = store.submit(empty_plan(), KitchenPreferences(ai_mode="ollama"), "a.json")
        t0 = time.time()
        quick = store.submit(empty_plan(), KitchenPreferences(ai_mode="fallback"), "q.json")
        while store.get(quick["id"])["status"] not in ("done", "error") and time.time() - t0 < 15:
            time.sleep(0.05)
        assert store.get(quick["id"])["status"] == "done", "quick rules waited behind the AI design"
        assert store.get(ai["id"])["status"] in ("queued", "running")
        assert store.public(store.get(quick["id"]))["queue_position"] == 0
    finally:
        release.set()


def test_customer_primary_and_secondary_walls():
    """R24 / R25: the hob goes on the customer's primary wall, the sink on the secondary wall."""
    W3 = "wall_49fdugi4g7vj12pf"
    out, report = run(KitchenPreferences(ai_mode="fallback", primary_wall=W3, secondary_wall=W2), name="primary_wall")
    assert not verify_scene(out), verify_scene(out)
    nodes = out["scene"]["nodes"]
    assert by_category(nodes, "Base Hob Unit")[0]["wallId"] == W3
    assert by_category(nodes, "Base Sink Unit")[0]["wallId"] == W2
    assert W3 in report["layout"]["cabinet_walls"]


def test_lofts_follow_the_ceiling_one_per_unit():
    """R48: 850 mm lofts under a 3.0 m ceiling (92 mm reveal); R53: a same-width loft above every
    wall unit; R47: frame-with-shutter lofts; R52: 560 mm deep over tall units."""
    doc = empty_plan(SECONDARY)
    for n in doc["scene"]["nodes"].values():
        if n.get("type") in ("wall", "ceiling"):
            n["height"] = 3.0
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]))
    assert not verify_scene(out), verify_scene(out)
    h = report["heights"]
    assert (h["loft_mm"], h["reveal_mm"], h["lofts"]) == (850, 92, True)
    nodes = out["scene"]["nodes"]
    lofts = _level(out, "loft")
    assert lofts and all(n["metadata"]["category"].startswith(("Loft - Frame", "Loft – L Corner - Frame")) for n in lofts)
    assert all(abs(n["asset"]["dimensions"][1] - 0.85) < 1e-6 for n in lofts)
    for n in _level(out, "wall"):
        if "orner" in n["metadata"]["category"]:
            continue
        assert any(l["wallId"] == n["wallId"] and abs(l["position"][0] - n["position"][0]) < 1e-3
                   and abs(l["asset"]["dimensions"][0] - n["asset"]["dimensions"][0]) < 1e-3 for l in lofts), n["metadata"]["category"]
    for t in _level(out, "tall"):
        assert any(abs(l["position"][0] - t["position"][0]) < 1e-3 and abs(l["asset"]["dimensions"][2] - 0.56) < 1e-6
                   for l in lofts if l["wallId"] == t["wallId"]), "560 mm deep loft over the tall unit"


def test_rotated_and_translated_room():
    """Same room rotated 30 deg and moved: wall-local maths must be frame independent."""
    a = math.radians(30)

    def f(x, z):
        return [x * math.cos(a) - z * math.sin(a) + 5.0, x * math.sin(a) + z * math.cos(a) - 2.0]

    doc = empty_plan()
    for n in doc["scene"]["nodes"].values():
        if n["type"] == "wall":
            n["start"], n["end"] = f(*n["start"]), f(*n["end"])
        if n["type"] in ("ceiling", "slab") and isinstance(n.get("polygon"), list):
            n["polygon"] = [f(*p) for p in n["polygon"]]
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]))
    assert not verify_scene(out), verify_scene(out)
    assert not report["validation_issues"]


def _move_opening(doc, oid, wall_id, center_m, width_m=None):
    nodes = doc["scene"]["nodes"]
    o = nodes[oid]
    for n in nodes.values():
        if oid in n.get("children", []):
            n["children"].remove(oid)
    o["wallId"] = o["parentId"] = wall_id
    o["position"][0] = center_m
    if width_m:
        o["width"] = width_m
    nodes[wall_id]["children"].append(oid)
    return doc


def _layout_corners(report):
    return report["layout"]["corners"]


def test_corner_wall_tier_uses_blind_corner_when_l_corner_blocked():
    """Window 550 mm from the Wall 1/Wall 2 corner: no room for a 650 Wall L Corner on Wall 2,
    so the wall tier must use a wall blind corner unit (never a regular wall unit)."""
    doc = _move_opening(empty_plan(), WINDOW_ID, W2, 0.55 + 0.6, 1.2)  # window 550-1750 on Wall 2
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]))
    errs = verify_scene(out)
    assert not errs, "\n".join(errs)
    corner = _layout_corners(report)[0]
    wall_aid = corner["wall_unit"]
    assert wall_aid and ("WBCL" in wall_aid or "WBCR" in wall_aid), corner
    assert corner["base_unit"], corner


def test_corner_left_open_when_no_corner_unit_fits():
    """Doors right at the Wall 1/Wall 2 corner on both walls: no blind corner unit can sit there,
    so the corner must stay empty instead of getting regular cabinets."""
    doc = empty_plan()
    _move_opening(doc, DOOR_ID, W1, 4.05 - 0.55, 0.9)  # door 3050-3950 on Wall 1 (corner end)
    nodes = doc["scene"]["nodes"]
    d2 = copy.deepcopy(nodes[DOOR_ID])
    d2["id"] = "door_secondcorner0001"
    nodes[d2["id"]] = d2
    _move_opening(doc, d2["id"], W2, 0.05 + 0.55, 0.9)  # door 150-1050 on Wall 2 (corner start)
    _move_opening(doc, WINDOW_ID, W2, 2.6, 1.2)
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]))
    errs = verify_scene(out, expect_window=WINDOW_ID, expect_door=DOOR_ID)
    assert not errs, "\n".join(errs)
    corner = _layout_corners(report)[0]
    assert corner["base_unit"] is None and "base" in corner["open_tiers"], corner
    assert any("left open" in w for w in report["layout"]["warnings"])


# -- walls whose room side is the back, and non-rectangular rooms ---------------------------

DATA = ROOT / "test_data"
REAL_ROOMS = {
    # L-shaped room exported from MagickHome; its window wall is drawn the other way round
    # (frontSide "exterior", backSide "interior") - cabinets there used to land outside.
    "l_shaped_room_reversed_wall": ("window_t6m3s0qnha8vb0d4", "door_2sqpkgv0swc2anh4"),
    "eight_wall_room": ("window_c99hnxmmb65wwkj1", "door_2sqpkgv0swc2anh4"),
}


@pytest.mark.parametrize("room", list(REAL_ROOMS))
@pytest.mark.parametrize("layout", ["auto", "L", "U", "single"])
def test_real_non_rectangular_rooms(room, layout):
    window, door = REAL_ROOMS[room]
    doc = empty_plan(DATA / f"{room}.json")
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", layout=layout))
    errs = verify_scene(out, expect_window=window, expect_door=door)
    assert not errs, "\n".join(errs)
    assert not report["validation_issues"], report["validation_issues"]


def test_l_shaped_room_sides_and_outside_corner():
    from scene_parser import parse_scene

    g = parse_scene(empty_plan(DATA / "l_shaped_room_reversed_wall.json"))
    assert not g.walls["wall_unzzbbh7z7n6uka1"].front_interior, "window wall: room on its back side"
    assert all(g.walls[w].front_interior for w in g.walls if w != "wall_unzzbbh7z7n6uka1")
    # Wall 5 / Wall 6 meet at an OUTSIDE corner of the L: never a corner-cabinet corner.
    pairs = {frozenset((c.wall_a, c.wall_b)) for c in g.corners}
    assert frozenset(("wall_9kj3t56v1k400q7j", "wall_918glvxpk8snjsft")) not in pairs, pairs


def test_reversed_window_wall_puts_cabinets_on_the_room_side():
    """Same sample room with Wall 2 drawn end -> start: its cabinets must move to the wall's
    back side (negative z, unrotated) and the corner units must swap hands."""
    doc = empty_plan()
    nodes = doc["scene"]["nodes"]
    w2 = nodes[W2]
    L = math.dist(w2["start"], w2["end"])
    w2["start"], w2["end"] = w2["end"], w2["start"]
    w2["frontSide"], w2["backSide"] = "exterior", "interior"
    win = nodes[WINDOW_ID]
    win["position"][0] = L - win["position"][0]
    win["side"] = "back"
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2, W3]))
    errs = verify_scene(out)
    assert not errs, "\n".join(errs)
    on_w2 = [n for n in cabinets(out["scene"]["nodes"]) if n["wallId"] == W2]
    assert on_w2 and all(n["position"][2] < 0 and n["rotation"] == [0, 0, 0] and n["placement"]["side"] == -1 for n in on_w2)
    fx = [n for n in out["scene"]["nodes"].values() if n["type"] == "item" and n["asset"]["category"] == "fixture" and n.get("wallId") == W2]
    assert fx and all(n["position"][2] < 0 for n in fx), "countertop / panels must follow the cabinets"


def test_verifier_catches_violations():
    """The checks above must actually fail on broken scenes (guards against vacuous passes)."""
    good, _ = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="fallback", cabinet_walls=[W1, W2]))
    assert not verify_scene(good)

    def mutated(fn):
        doc = copy.deepcopy(good)
        fn(doc["scene"]["nodes"])
        return verify_scene(doc)

    def shift_sink(nodes):
        by_category(nodes, "Base Sink Unit")[0]["position"][0] += 0.05

    def into_door(nodes):
        cab = min((n for n in cabinets(nodes) if n["wallId"] == W1), key=lambda n: n["position"][0])
        cab["position"][0] = nodes[DOOR_ID]["position"][0] + 0.3

    def collide(nodes):
        a, b = [n for n in cabinets(nodes) if n["wallId"] == W2 and n["placement"]["level"] == "base"][:2]
        b["position"][0] = a["position"][0] + 0.1

    def fake_aid(nodes):
        cabinets(nodes)[0]["metadata"]["aid"] = "MTFAKE123"

    def regular_in_corner(nodes):
        bcu = next(n for n in cabinets(nodes) if n["metadata"]["category"].startswith("Blind corner unit"))
        bcu["metadata"]["category"] = "Base Unit"

    def drop_chimney(nodes):
        cid = next(k for k, n in nodes.items() if (n.get("name") or "").startswith("Chimney"))
        wall = nodes[nodes[cid]["parentId"]]
        wall["children"].remove(cid)
        del nodes[cid]

    def l_corner_at_start(nodes):
        lc = next(n for n in cabinets(nodes) if n["metadata"]["category"] == "Wall L Corner")
        L = frames(nodes)[lc["wallId"]].length
        lc["position"][0] = L - lc["position"][0]  # mirror it to the other end of its wall
        lc["placement"]["cornerAtEnd"] = False

    assert any("only fits the wall end" in e for e in mutated(l_corner_at_start))

    assert any("not under window" in e for e in mutated(shift_sink))
    assert any(DOOR_ID in e for e in mutated(into_door))
    assert any("3D collision" in e for e in mutated(collide))
    assert any("not in Finalised_Asset_Catalogue_Sheet" in e for e in mutated(fake_aid))
    assert any("expected 1 chimney" in e for e in mutated(drop_chimney))
    assert any("corner" in e and "regular Base Unit" in e for e in mutated(regular_in_corner))


def test_assets_api_loader_paginates_normalises_and_caches(monkeypatch, tmp_path):
    """POST ?page=N&limit=M pagination, 'None' strings, 'Shaker 1' == 'Shaker -1', offline cache fallback."""
    import catalog_loader as CL
    import config
    import requests

    def rec(i, style):
        return {"_id": f"id{i}", "aid": f"MTTEST{i}", "description": "t", "width": 600, "depth": 560, "height": 700,
                "unitFamily": "Base", "category": "Base Unit ", "doorType": "Shutter", "cabinetCombination": "Double Shutter",
                "doorStyle": style, "openingDirection": "None" if i == 0 else "Both", "handleType": "External Handle",
                "internals": "One Shelf", "isApplianceAccessory": False, "isActive": True}

    records = [rec(i, "Shaker 1" if i % 2 else "Slab (Plain)") for i in range(5)]
    calls = []

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_post(url, params=None, json=None, headers=None, timeout=None):
        calls.append(params)
        page, limit = params["page"], params["limit"]
        items = records[(page - 1) * limit: page * limit]
        return Resp({"success": True, "message": "Cabinets fetched", "data": {"items": items, "total": 5, "page": page, "limit": limit}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(config, "ASSETS_API_PAGE_SIZE", 2)
    monkeypatch.setattr(config, "ASSETS_API_METHOD", "POST")
    monkeypatch.setattr(config, "ASSETS_API_URL", "https://example.invalid/cabinets")
    monkeypatch.setattr(config, "CATALOG_CACHE_PATH", tmp_path / "cache.json")
    try:
        CL._load_catalog.cache_clear()
        cat = CL.load_catalog()
        assert cat.source == "api" and len(cat) == 5
        assert [c["page"] for c in calls] == [1, 2, 3]
        assert cat.get("MTTEST0").opening_direction is None
        assert cat.get("MTTEST0").category == "Base Unit"
        assert len(cat.filter(door_style="Shaker -1")) == 2  # API spells it 'Shaker 1'
        assert cat.get("MTTEST1").metadata()["_id"] == "id1"
        assert (tmp_path / "cache.json").exists()

        def down(*a, **k):
            raise requests.ConnectionError("offline")

        monkeypatch.setattr(requests, "post", down)
        CL._load_catalog.cache_clear()
        cached = CL.load_catalog()
        assert cached.source == "api-cache" and len(cached) == 5
    finally:
        CL._load_catalog.cache_clear()  # later tests reload the real catalogue


def test_web_ui_job_flow(monkeypatch, tmp_path):
    """Upload -> background job -> poll -> download, exactly as the web UI does it."""
    import time as _time

    import config
    from fastapi.testclient import TestClient

    from main import JOBS, app

    monkeypatch.setattr(config, "JOBS_DIR", tmp_path)  # finished designs are saved here

    with TestClient(app) as c:
        page = c.get("/")
        assert page.status_code == 200 and "MagickHome Kitchen Designer" in page.text and "Design my kitchen" in page.text
        plan = c.get("/api/sample-floorplan").json()
        assert not [n for n in plan["scene"]["nodes"].values() if n["type"] == "item"]
        r = c.post("/api/jobs", json={"scene": plan, "preferences": {"ai_mode": "fallback"}, "filename": "my plan.json"})
        assert r.status_code == 202, r.text
        jid = r.json()["id"]
        for _ in range(300):
            s = c.get(f"/api/jobs/{jid}").json()
            if s["status"] in ("done", "error"):
                break
            _time.sleep(0.1)
        assert s["status"] == "done", s
        assert [e["msg"] for e in s["log"]][0] == "Reading the floor plan"
        d = c.get(f"/api/jobs/{jid}/scene?download=1")
        assert 'filename="my_plan_kitchen.json"' in d.headers["content-disposition"]
        assert not verify_scene(d.json())
        assert c.get(f"/api/jobs/{jid}/report?download=1").json()["validation_issues"] == []
        assert (tmp_path / f"{jid}.json").is_file(), "finished design must be saved to disk"
        with JOBS.lock:
            JOBS.jobs.pop(jid)  # as after a server restart
        again = c.get(f"/api/jobs/{jid}").json()
        assert again["status"] == "done" and again["report"]["explanation"], "design must reload from disk"
        assert c.get("/").headers.get("cache-control") == "no-cache"
        assert c.post("/api/jobs", json={"scene": {"format": "nope", "version": 1, "scene": {"nodes": {}}}}).status_code == 400
        assert c.get("/api/jobs/doesnotexist").status_code == 404


def test_deterministic_output():
    prefs = KitchenPreferences(ai_mode="fallback")
    a, _ = generate_kitchen(empty_plan(), prefs)
    b, _ = generate_kitchen(empty_plan(), prefs)
    a.pop("exportedAt"), b.pop("exportedAt")
    assert a == b


# -- mocked Ollama (exercises the structured-output path without a server) ------------

class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.message = _Msg(content)
        self.eval_count = 123


class _Model:
    def __init__(self, name):
        self.model = name


class FakeOllama:
    """Returns a deliberately poor design to prove the rule engine repairs it."""

    def __init__(self, design: dict, installed=True):
        self.design = design
        self.installed = installed
        self.calls = []
        self.pulled = False

    def list(self):
        import config

        return type("L", (), {"models": [_Model(config.OLLAMA_MODEL)] if self.installed else []})()

    def pull(self, model, stream=False):
        self.pulled = True
        self.installed = True
        return iter([type("P", (), {"status": "success", "total": 1, "completed": 1})()])

    def chat(self, model, messages, format=None, options=None, keep_alive=None):
        self.calls.append({"model": model, "format": format, "options": options, "messages": messages})
        return _Resp(json.dumps(self.design))


def _bad_design(groups):
    g = groups
    return {
        "room_reading": "Wall 2 has the window, Wall 1 has the door.",
        "design_plan": "Sink on the plain wall, hob under the window, tall unit in the middle.",
        "layout_type": "L",
        "cabinet_walls": [W2, W3],
        "sink_wall_id": W3,  # wrong: no window on Wall 3
        "sink_aid": g["sink"][0].aid,
        "hob_wall_id": W2,  # wrong: window wall -> chimney impossible there
        "hob_aid": g["hob"][0].aid,
        "hob_center_mm": 1990,
        "spice_aid": g["spice"][0].aid,
        "cylinder_aid": None,
        "tall_units": [{"wall_id": W2, "aid": g["tall"][0].aid}],
        "base_runs": [{"wall_id": W3, "aids": [g["base"][0].aid, g["base"][5].aid]}],
        "wall_runs": [{"wall_id": W3, "aids": [g["wall"][6].aid]}],
    }


def test_mocked_ollama_design_is_repaired(monkeypatch):
    from catalog_loader import load_filtered_catalog

    import config

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v1")  # original prompt (A/B baseline)
    groups = ai_designer.build_shortlist(load_filtered_catalog())
    fake = FakeOllama(_bad_design(groups), installed=False)
    monkeypatch.setattr(ai_designer, "_client", lambda: fake)
    out, report = run(KitchenPreferences(ai_mode="ollama"), name="mocked_ollama")
    assert fake.pulled, "missing model should be auto-pulled"
    call = fake.calls[0]
    assert call["options"]["temperature"] == 0
    # schema is enum-locked to the shortlist
    sink_enum = call["format"]["properties"]["sink_aid"]["enum"]
    assert set(sink_enum) == {i.aid for i in groups["sink"]}
    assert report["ai"]["mode"] == "ollama"
    errs = verify_scene(out)
    assert not errs, "\n".join(errs)
    hob = by_category(out["scene"]["nodes"], "Base Hob Unit")[0]
    assert hob["wallId"] != W2, "hob must not stay under the window"


def test_customer_layout_overrides_ai_wall_count(monkeypatch):
    """Qwen proposes an L (2 walls) but the customer asked for a U."""
    from catalog_loader import load_filtered_catalog

    import config

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v1")
    groups = ai_designer.build_shortlist(load_filtered_catalog())
    fake = FakeOllama(_bad_design(groups))
    monkeypatch.setattr(ai_designer, "_client", lambda: fake)
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="ollama", layout="U"))
    assert len(report["layout"]["cabinet_walls"]) == 3
    assert any("layout U" in d for d in report["layout"]["decisions"])
    assert not verify_scene(out)


def test_schema_rejects_hallucinated_aid():
    from catalog_loader import load_filtered_catalog
    from pydantic import ValidationError

    from scene_parser import parse_scene

    groups = ai_designer.build_shortlist(load_filtered_catalog())
    g = parse_scene(empty_plan())
    Model = ai_designer.build_design_model(groups, g.wall_order)
    bad = _bad_design(groups)
    bad["sink_aid"] = "MTBABSU9999560700FAKE"
    with pytest.raises(ValidationError):
        Model.model_validate(bad)
    bad = _bad_design(groups)
    bad["hob_wall_id"] = "wall_doesnotexist"
    with pytest.raises(ValidationError):
        Model.model_validate(bad)
    n = len(ai_designer.shortlist_items(groups))
    assert 35 <= n <= 45, n


# -- v2 prompt: knowledge pack, option codes, new feature units ---------------------

def _v2_menu(door_style=None):
    from catalog_loader import load_filtered_catalog

    cat = load_filtered_catalog(door_style) if door_style else load_filtered_catalog()
    return ai_designer.build_menu(ai_designer.build_shortlist_v2(cat))


def _v2_design(menu, **over):
    first = {g: opts[0][0] for g, opts in menu.items() if opts}
    codes = {g: [c for c, _ in opts] for g, opts in menu.items()}
    d = {
        "room_reading": "Wall 2 has the window, Wall 1 the door near the Wall 4 corner, Wall 3 and Wall 4 are solid.",
        "design_plan": "Sink under the window on Wall 2, hob on solid Wall 3, tall storage at the end of Wall 3.",
        "layout_why": "L-shape: window wall plus the solid wall away from the door.",
        "cabinet_walls": ["Wall 2", "Wall 3"],
        "sink_why": "Window on Wall 2, standard 900 sink.",
        "sink_wall": "Wall 2",
        "sink": first["sink"],
        "hob_why": "Wall 3 is solid and far from the door.",
        "hob_wall": "Wall 3",
        "hob": first["hob"],
        "hob_center_mm": 2000,
        "spice": first["spice"],
        "lpg_why": "Gas supply not stated, so LPG is assumed (R11).",
        "lpg_cylinder": first["cylinder"],
        "tall_why": "Reheats daily, microwave tower at the run end.",
        "tall_units": [{"wall": "Wall 3", "unit": next(c for c in codes["tall"] if c.startswith("MICROWAVE"))}],
        "purifier_why": "Not stated; most homes use a wall purifier.",
        "water_purifier": first["purifier"],
        "display_why": "Long window wall can frame the window.",
        "display_unit": first["display"],
        "storage_why": "Not stated; a cutlery drawer on top and shelves below suit most homes.",
        "base_storage": "mixed",
        "features_why": "Drawers by the hob, basket by the sink.",
        "wall_cabinets_why": "Wide 900 units over the hob wall for a calm, aligned look.",
        "base_feature_units": [{"wall": "Wall 3", "units": [codes["base"][0]]}, {"wall": "Wall 2", "units": [codes["base"][-2]]}],
        "wall_cabinet_sizes": [{"wall": "Wall 3", "units": [codes["wall"][3]]}],
        "open_why": "Not asked for; open shelves gather grease.",
        "open_unit": "NONE",
        "wall_style_why": "Standard hinged doors.",
        "wall_style": "shutters",
        "loft_why": "The ceiling is too low for lofts.",
        "lofts": "no",
        "finish_why": "Warm oak below, a light PU wall run to keep the room bright.",
        "base_finish": ai_designer.finish_codes("base", None)[-1],  # from the AI shortlist
        "wall_finish": ai_designer.finish_codes("wall", None)[0],
    }
    d.update(over)
    return d


def test_v2_prompt_teaches_and_places_new_units(monkeypatch):
    """v2: knowledge pack in the prompt, code enums in the schema, and the solver places
    the purifier beside the window, the glass display at a run end and the microwave tall unit."""
    import config
    import design_knowledge

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v2")
    menu = _v2_menu()
    fake = FakeOllama(_v2_design(menu))
    monkeypatch.setattr(ai_designer, "_client", lambda: fake)
    out, report = run(KitchenPreferences(ai_mode="ollama"), name="mocked_ollama_v2")
    call = fake.calls[0]
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "WORK TRIANGLE" in system and "WORKED EXAMPLES" in system
    assert "SINK-900 |" in user and "Hob centre range" in user and "Wet zone: sink centred at" in user
    assert design_knowledge.note_for(menu["sink"][0][1]) in user
    props = call["format"]["properties"]
    assert set(props["sink"]["enum"]) == {c for c, _ in menu["sink"]}
    assert report["ai"]["prompt_version"] == "v2"
    errs = verify_scene(out)
    assert not errs, "\n".join(errs)

    nodes = out["scene"]["nodes"]
    win = nodes[WINDOW_ID]
    aqua = by_category(nodes, "Wall Aquaguard Unit")
    assert len(aqua) == 1 and aqua[0]["wallId"] == win["wallId"]
    w_lo, w_hi = win["position"][0] - win["width"] / 2, win["position"][0] + win["width"] / 2
    a_lo, a_hi = interval(aqua[0])
    assert min(abs(a_lo - w_hi), abs(w_lo - a_hi)) < 0.2, "purifier must hang right beside the window"
    glass = [n for n in cabinets(nodes) if "Glass" in n["metadata"]["category"]]
    assert len(glass) == 1 and glass[0]["placement"]["level"] == "wall"
    assert by_category(nodes, "Tall Built-In Microwave"), "Qwen's microwave tall unit must be built"


def test_gemini_designs_without_the_knowledge_pack(monkeypatch):
    """Default Gemini mode: no playbook, worked examples, cabinet notes, rule hints or schema
    hint texts - only plain room facts, the cabinet list, the needs and the same enum-locked
    schema. Qwen's prompt keeps the full knowledge pack."""
    import config
    import design_knowledge
    import gemini_designer
    from catalog_loader import load_filtered_catalog
    from scene_parser import parse_scene

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v2")
    monkeypatch.setattr(config, "GEMINI_USE_KNOWLEDGE", False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    menu = _v2_menu()
    sent = []

    def fake_post(model, body, key):
        sent.append(body)
        parts = [{"text": "The window is on Wall 2.", "thought": True}, {"text": json.dumps(_v2_design(menu))}]
        return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}

    monkeypatch.setattr(gemini_designer, "_post", fake_post)
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="ollama", ai_provider="gemini"))
    system = sent[0]["systemInstruction"]["parts"][0]["text"]
    user = sent[0]["contents"][0]["parts"][0]["text"]
    assert system == ai_designer.PLAIN_SYSTEM_PROMPT
    assert "WORK TRIANGLE" not in system + user and "WORKED EXAMPLES" not in system + user
    for hint in ("Wet zone", "Hob centre range", "Too short", "U-shape on", "centred under the window",
                 "designer note", "beside the hob", "worked examples"):
        assert hint not in user, hint
    notes = {design_knowledge.note_for(it) for opts in menu.values() for _, it in opts} - {"", None}
    assert notes and not [n for n in notes if n in user]
    assert "SINK-900 |" in user and "WINDOW" in user and "CUSTOMER NEEDS" in user
    schema = sent[0]["generationConfig"]["responseJsonSchema"]
    assert '"description"' not in json.dumps(schema)
    assert set(schema["properties"]["sink"]["enum"]) == {c for c, _ in menu["sink"]}
    assert report["ai"]["mode"] == "gemini" and report["ai"]["knowledge"] is False
    assert report["ai"]["thoughts"] == "The window is on Wall 2."
    assert not verify_scene(out)

    # Qwen (and Gemini with GEMINI_USE_KNOWLEDGE=1) get the knowledge; the answer schema is identical.
    geo, groups, prefs = parse_scene(empty_plan()), ai_designer.build_shortlist_v2(load_filtered_catalog()), KitchenPreferences()
    (msgs_k, Model_k, _), (msgs_p, Model_p, _) = (
        ai_designer.build_messages(geo, groups, prefs, "v2", knowledge=k) for k in (True, False))
    assert "WORK TRIANGLE" in msgs_k[0]["content"] and "Hob centre range" in msgs_k[1]["content"]
    assert Model_k.model_json_schema() == Model_p.model_json_schema()


def test_gemini_survives_dropped_connections_busy_models_and_quota(monkeypatch):
    """A dropped connection is retried; a model that stays busy (503) or has used its daily quota
    (429 PerDay) is skipped for the next model - the design still comes from Gemini, not the rules."""
    import requests

    import config
    import gemini_designer

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v2")
    monkeypatch.setattr(config, "GEMINI_MODEL", "main-model")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", ["busy-model", "quota-model", "good-model"])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini_designer.time, "sleep", lambda s: None)
    menu = _v2_menu()
    calls = []

    class Resp:
        def __init__(self, code, payload=None, text=""):
            self.status_code, self._payload, self.text = code, payload, text

        def json(self):
            return self._payload

    def fake_post(url, **kw):
        model = url.split("/models/")[1].split(":")[0]
        calls.append(model)
        if model == "main-model":
            raise requests.exceptions.SSLError("EOF occurred in violation of protocol")
        if model == "busy-model":
            return Resp(503, text='{"error": {"message": "high demand"}}')
        if model == "quota-model":
            return Resp(429, text='{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}')
        parts = [{"text": json.dumps(_v2_design(menu))}]
        return Resp(200, {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                          "usageMetadata": {}, "modelVersion": model})

    monkeypatch.setattr(gemini_designer.requests, "post", fake_post)
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="auto", ai_provider="gemini", layout="U"))
    assert report["ai"]["mode"] == "gemini", report["ai"].get("fallback_reason")
    assert report["ai"]["model"] == "good-model"
    assert calls.count("main-model") == 4, "dropped connections are retried before moving on"
    assert calls.count("quota-model") == 1, "a used-up daily quota is not retried"
    notes = " ".join(report["ai"]["model_notes"])
    assert "connection dropped" in notes and "busy" in notes and "daily quota" in notes
    assert not verify_scene(out)


def test_v2_schema_rejects_unknown_codes_and_walls():
    from pydantic import ValidationError

    from scene_parser import parse_scene

    menu = _v2_menu()
    labels = ai_designer.wall_labels(parse_scene(empty_plan()))
    Model = ai_designer.build_design_model_v2(menu, list(labels), require_tall=True)
    Model.model_validate(_v2_design(menu))
    for bad in (
        {"sink": "SINK-950"},
        {"hob_wall": "Wall 9"},
        {"sink": menu["sink"][0][1].aid},  # raw AIDs are not accepted in v2
        {"tall_units": []},  # customer wants tall storage
    ):
        with pytest.raises(ValidationError):
            Model.model_validate(_v2_design(menu, **bad))


def test_stated_needs_narrow_the_schema():
    """A need the customer states is not Qwen's to change: the enum only allows that answer."""
    menu = _v2_menu()
    walls = ["Wall 1", "Wall 2", "Wall 3", "Wall 4"]

    def enum(prefs, field):
        props = ai_designer.build_design_model_v2(menu, walls, prefs=prefs).model_json_schema()
        f = props["properties"][field]
        return set(f.get("enum") or [f.get("const")])

    assert enum(KitchenPreferences(water_purifier="no"), "water_purifier") == {"NONE"}
    assert "NONE" not in enum(KitchenPreferences(water_purifier="yes"), "water_purifier")
    assert "NONE" in enum(KitchenPreferences(), "water_purifier")  # not stated: Qwen decides
    assert enum(KitchenPreferences(include_cylinder=False), "lpg_cylinder") == {"NONE"}
    assert enum(KitchenPreferences(display_unit="no"), "display_unit") == {"NONE"}
    talls = ai_designer.allowed_codes(menu, "tall", KitchenPreferences(built_in="none"))
    assert talls and all(c.startswith(("PANTRY", "TALL-WICKER", "CROCKERY")) for c in talls)
    talls = ai_designer.allowed_codes(menu, "tall", KitchenPreferences(built_in="none", crockery_unit="no"))
    assert talls and not [c for c in talls if c.startswith("CROCKERY")]
    # new needs: open shelves, wall doors, lofts (and the ceiling decides lofts too)
    assert enum(KitchenPreferences(open_unit="no"), "open_unit") == {"NONE"}
    assert "NONE" not in enum(KitchenPreferences(open_unit="yes"), "open_unit")
    assert enum(KitchenPreferences(wall_style="lift_up"), "wall_style") == {"lift_up"}
    assert enum(KitchenPreferences(), "wall_style") == {"shutters", "lift_up"}
    assert enum(KitchenPreferences(include_loft=False), "lofts") == {"no"}
    assert enum(KitchenPreferences(), "lofts") == {"yes", "no"}
    low = ai_designer.build_design_model_v2(menu, walls, prefs=KitchenPreferences(include_loft=True), lofts_possible=False)
    f = low.model_json_schema()["properties"]["lofts"]
    assert set(f.get("enum") or [f.get("const")]) == {"no"}


def test_qwen_cannot_skip_a_needed_tower_and_reasons_are_reported(monkeypatch):
    """Customer needs a microwave + purifier; the (mocked) model only proposes a pantry.
    The microwave tower is added, the purifier is built, and every key item carries a reason."""
    import config

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v2")
    menu = _v2_menu()
    pantry = next(c for c, _ in menu["tall"] if c.startswith("PANTRY"))
    fake = FakeOllama(_v2_design(menu, tall_units=[{"wall": "Wall 3", "unit": pantry}], lpg_cylinder="NONE"))
    monkeypatch.setattr(ai_designer, "_client", lambda: fake)
    prefs = KitchenPreferences(ai_mode="ollama", built_in="microwave", water_purifier="yes", include_cylinder=False)
    out, report = generate_kitchen(empty_plan(), prefs)
    assert not verify_scene(out), verify_scene(out)
    user = fake.calls[0]["messages"][1]["content"]
    assert "NEEDS a wall water-purifier" in user and "piped gas" in user and "NEEDS a built-in microwave tower" in user
    nodes = out["scene"]["nodes"]
    assert by_category(nodes, "Tall Built-In Microwave"), "needed microwave tower must be built"
    assert by_category(nodes, "Wall Aquaguard Unit"), "needed purifier must be built"
    assert not by_category(nodes, "Base Cylinder Unit"), "piped gas: no cylinder unit"
    ex = {e["item"]: e for e in report["explanation"]}
    assert ex["Water purifier"]["source"] == "customer" and ex["LPG cylinder"]["source"] == "customer"
    assert ex["Sink"]["source"] == "qwen" and ex["Sink"]["why"]
    assert "customer needs it" in ex["Tall units"]["why"]


@pytest.mark.parametrize("needs, present, absent", [
    (dict(water_purifier="no", include_cylinder=False, display_unit="no", built_in="none"),
     [], ["Wall Aquaguard Unit", "Base Cylinder Unit", "Wall Glass Unit", "Tall Built-In Microwave", "Tall Built-In Oven"]),
    (dict(water_purifier="yes", include_cylinder=True, display_unit="yes", built_in="oven"),
     ["Wall Aquaguard Unit", "Base Cylinder Unit", "Wall Glass Unit", "Tall Built-In Oven"], []),
])
def test_quick_rules_designer_follows_needs(needs, present, absent):
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="fallback", **needs))
    assert not verify_scene(out), verify_scene(out)
    nodes = out["scene"]["nodes"]
    for cat in present:
        assert by_category(nodes, cat), f"{cat} should be built for {needs}"
    for cat in absent:
        assert not by_category(nodes, cat), f"{cat} should not be built for {needs}"
    assert all(e["why"] for e in report["explanation"]), report["explanation"]


def test_left_to_qwen_the_base_cabinet_style_is_qwens_decision(monkeypatch):
    """Base cabinets on "Let Qwen decide": Qwen picks the style (here drawers) with its own
    reason, and the solver builds it. A stated style leaves Qwen only that one option."""
    import config

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v2")
    menu = _v2_menu()
    reason = "Heavy daily cooking: drawers keep pans and pressure cookers reachable."
    fake = FakeOllama(_v2_design(menu, base_storage="drawers", storage_why=reason))
    monkeypatch.setattr(ai_designer, "_client", lambda: fake)
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="ollama"))
    assert not verify_scene(out), verify_scene(out)
    props = fake.calls[0]["format"]["properties"]
    assert set(props["base_storage"]["enum"]) == {"drawers", "mixed", "shutters"}
    store = [n for n in cabinets(out["scene"]["nodes"]) if n["metadata"]["category"] == "Base Unit"]
    assert store and all("drawer" in (n["metadata"].get("cabinetCombination") or "").lower() for n in store)
    ex = {e["item"]: e for e in report["explanation"]}["Base cabinets"]
    assert ex["source"] == "qwen" and ex["why"] == reason and ex["built"].startswith("Drawer units")
    stated = ai_designer.build_design_model_v2(menu, ["Wall 1"], prefs=KitchenPreferences(base_storage="shutters"))
    f = stated.model_json_schema()["properties"]["base_storage"]
    assert set(f.get("enum") or [f.get("const")]) == {"shutters"}


def test_every_decision_is_explained_with_facts(monkeypatch):
    """Each of Qwen's decisions carries its own reason plus facts measured by the system."""
    import config

    monkeypatch.setattr(config, "AI_PROMPT_VERSION", "v2")
    fake = FakeOllama(_v2_design(_v2_menu()))
    monkeypatch.setattr(ai_designer, "_client", lambda: fake)
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="ollama"))
    ex = {e["item"]: e for e in report["explanation"]}
    for item in ("Layout", "Sink", "Hob", "Spice pull-out", "LPG cylinder", "Tall units", "Water purifier",
                 "Glass display", "Base cabinets", "Feature units", "Wall cabinets", "Corners"):
        assert item in ex, item
    facts = lambda item: " | ".join(ex[item]["details"])  # noqa: E731
    assert "centred" in facts("Sink") and "Sizes on offer" in facts("Sink")
    assert "Sink to hob" in facts("Hob") and "door" in facts("Hob").lower() and "Chimney" in facts("Hob")
    assert "built on" in facts("Feature units") or "did not fit" in facts("Feature units")
    assert ex["Wall cabinets"]["why"].startswith("Wide 900 units") and ex["Wall cabinets"]["source"] == "qwen"
    assert all(ex[i]["why"] for i in ("Layout", "Sink", "Hob", "Tall units", "Water purifier", "Base cabinets"))


def test_brief_asking_for_drawers_fills_runs_with_drawers():
    """'Lots of drawers' in the brief is a stated need: storage runs use drawer units."""
    plain, _ = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="fallback"))
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="fallback", style_notes="Lots of drawers"))
    assert not verify_scene(out), verify_scene(out)

    def full_drawer_share(doc):
        """Full drawer units (no shutter) among the plain storage Base Units."""
        base = [n for n in cabinets(doc["scene"]["nodes"]) if n["placement"]["level"] == "base"
                and n["metadata"]["category"] == "Base Unit"]
        combos = [(n["metadata"].get("cabinetCombination") or "").lower() for n in base]
        return sum("drawer" in c and "shutter" not in c for c in combos), len(base)

    d, n = full_drawer_share(out)
    d0, _ = full_drawer_share(plain)  # quick rules on its own picks the drawer + shutter mix
    assert n and d > d0 and d >= n - 1, (d, n, d0)
    ex = {e["item"]: e for e in report["explanation"]}
    assert ex["Base cabinets"]["source"] == "customer" and "brief" in ex["Base cabinets"]["need"].lower()


def test_requested_feature_units_move_to_another_wall_instead_of_vanishing():
    """The designer asks for more drawer units on the hob wall than fit there: they must
    be built on the next wall (or reported), never silently dropped."""
    from ai_designer import build_shortlist_v2, fallback_proposal
    from catalog_loader import load_filtered_catalog
    from rule_engine import SolverPrefs, solve_layout
    from scene_parser import parse_scene

    cat = load_filtered_catalog()
    g = parse_scene(empty_plan())
    groups = build_shortlist_v2(cat)
    prop = fallback_proposal(g, groups, KitchenPreferences())
    wanted = [it for it in groups["base"] if "drawer" in it.cabinet_combination.lower() and it.cat == "base unit"][:3]
    prop.base_prefs = {prop.hob_wall_id: [it.aid for it in wanted]}
    lay = solve_layout(g, cat, prop, SolverPrefs())
    built = [p.item.aid for p in lay.placements if p.role == "storage"]
    missing = [it for it in wanted if it.aid not in built]
    assert not missing or any("did not fit" in w for w in lay.warnings), (missing, lay.warnings)
    assert len(missing) < len(wanted), "at least the units that fit somewhere must be built"


def test_needed_purifier_tries_hard_then_says_why():
    """L-shaped room: the window wall has no wall space beside the window, so a needed
    purifier goes to the next wall by the corner - or the report says plainly why not."""
    doc = empty_plan(DATA / "l_shaped_room_reversed_wall.json")
    out, report = generate_kitchen(doc, KitchenPreferences(ai_mode="fallback", water_purifier="yes"))
    assert not verify_scene(out, expect_window="window_t6m3s0qnha8vb0d4", expect_door="door_2sqpkgv0swc2anh4")
    built = by_category(out["scene"]["nodes"], "Wall Aquaguard Unit")
    msgs = report["layout"]["warnings"] + report["layout"]["decisions"]
    assert built or any("customer needs one" in m for m in msgs), msgs


@pytest.mark.parametrize("style", ["Slab (Plain)", "Shaker -1"])
def test_knowledge_pack_matches_catalogue(style):
    """Every offered option has a designer note, every catalogue category has a guide entry,
    and the worked examples only use codes that exist in the real menu."""
    import re

    import design_knowledge
    from catalog_loader import load_catalog

    guide = design_knowledge.cabinet_guide()
    assert not sorted({i.category for i in load_catalog().items if i.cat not in guide})
    menu = _v2_menu(style)
    for g in ("sink", "hob", "tall", "base", "wall", "purifier", "display"):
        assert menu[g], f"{style}: empty menu group {g}"
        for code, it in menu[g]:
            assert design_knowledge.note_for(it), f"{style}: no designer note for {code} ({it.category})"
    codes = {c for opts in menu.values() for c, _ in opts}
    used = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d{3,4}(?:-[A-Z0-9]+)*\b", design_knowledge.worked_examples()))
    assert used and used <= codes, sorted(used - codes)


def test_auto_mode_falls_back_without_server(monkeypatch):
    def boom():
        raise ai_designer.AIUnavailable("no server")

    monkeypatch.setattr(ai_designer, "ensure_model", lambda *a, **kw: boom())
    out, report = generate_kitchen(empty_plan(), KitchenPreferences(ai_mode="auto"))
    assert report["ai"]["mode"] == "fallback" and "no server" in report["ai"]["fallback_reason"]
    assert not verify_scene(out)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the kitchen generator verification suite")
    ap.add_argument("--ai-mode", default=AI_MODE, choices=["fallback", "auto", "ollama"])
    args, rest = ap.parse_known_args()
    os.environ["KITCHEN_TEST_AI_MODE"] = args.ai_mode
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider", *rest]))
