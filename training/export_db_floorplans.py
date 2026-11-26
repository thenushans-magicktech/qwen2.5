"""Export the floor plans (rooms only) from the MagickHome MongoDB, read-only.

Designs in the database are not approved, so only the ROOM is kept: walls, doors, windows, slab,
ceiling and the scene's structure nodes. Every placed item (cabinets, appliances, decor, annotations)
is removed, as are any fields outside the scene. Duplicate rooms are dropped, and each room must be
readable by the kitchen engine (scene_parser.parse_scene).

    set MONGO_URI=mongodb+srv://...           (never write it into a file)
    python training/export_db_floorplans.py   -> training/floorplans/db/*.json + index.json

Sources: designs.floorPlan, design_generation_tracks.request.scene and projects.versions[].sceneJson.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

import rulebook  # noqa: E402
from scene_parser import parse_scene  # noqa: E402

KEEP_TYPES = {"site", "building", "level", "wall", "door", "window", "slab", "ceiling"}
OUT = ROOT / "training" / "floorplans" / "db"


def to_doc(scene) -> dict | None:
    if isinstance(scene, str):
        try:
            scene = json.loads(scene)
        except ValueError:
            return None
    if not isinstance(scene, dict):
        return None
    inner = scene.get("scene") if isinstance(scene.get("scene"), dict) else scene
    nodes = inner.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return None
    return {"format": "magickhomes-scene", "version": 1, "scene": {"nodes": nodes, "rootNodeIds": inner.get("rootNodeIds", [])}}


def room_only(doc: dict) -> dict:
    """Keep the room's structure nodes; drop every item and annotation (and their references)."""
    nodes = doc["scene"]["nodes"]
    keep = {nid: n for nid, n in nodes.items() if isinstance(n, dict) and n.get("type") in KEEP_TYPES}
    clean = {}
    for nid, n in keep.items():
        m = dict(n)
        m.pop("metadata", None)
        if isinstance(m.get("children"), list):
            m["children"] = [c for c in m["children"] if c in keep]
        clean[nid] = m
    roots = [r for r in doc["scene"].get("rootNodeIds", []) if r in clean]
    return {"format": "magickhomes-scene", "version": 1, "scene": {"nodes": clean, "rootNodeIds": roots}}


def geometry_key(g) -> str:
    """Same walls, doors and windows (to 10 mm) = the same room."""
    parts = []
    for wid in g.wall_order:
        w = g.walls[wid]
        ops = sorted((o.kind, round(o.center_mm, -1), round(o.width_mm, -1)) for o in w.openings)
        parts.append((round(w.length_mm, -1), round(w.height_m * 100), tuple(ops)))
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]


def shape_name(g) -> str:
    n, closed = len(g.walls), len(g.corners) >= len(g.walls)
    if not closed:
        return f"open ({n} walls)"
    return {4: "rectangle", 5: "cut corner (5 walls)", 6: "L-shaped room (6 walls)"}.get(n, f"{n} walls")


def main() -> int:
    from pymongo import MongoClient

    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("Set MONGO_URI first (the connection string; it is never saved).")
        return 2
    client = MongoClient(uri, serverSelectionTimeoutMS=20000)
    db = client.get_default_database()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()

    def sources():
        for d in db.designs.find({}, {"floorPlan": 1}):
            yield "designs", d.get("floorPlan")
        for d in db.design_generation_tracks.find({}, {"request.scene": 1}):
            yield "tracks", (d.get("request") or {}).get("scene")
        for d in db.projects.find({"versions.0": {"$exists": True}}, {"versions.sceneJson": 1}):
            for v in d.get("versions") or []:
                yield "projects", v.get("sceneJson")

    stats, shapes, index, seen = collections.Counter(), collections.Counter(), [], set()
    for src, raw in sources():
        stats[f"{src}: read"] += 1
        doc = to_doc(raw)
        if doc is None:
            stats[f"{src}: no scene"] += 1
            continue
        doc = room_only(doc)
        try:
            g = parse_scene(doc)
        except Exception:  # noqa: BLE001
            stats[f"{src}: engine cannot read"] += 1
            continue
        if not g.walls:
            stats[f"{src}: no walls"] += 1
            continue
        key = geometry_key(g)
        if key in seen:
            stats[f"{src}: duplicate"] += 1
            continue
        seen.add(key)
        name = f"{src}_{len(index):04d}"
        (OUT / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
        area = rulebook.floor_area_sqft(g)
        band, _ = rulebook.size_band(area)
        shp = shape_name(g)
        shapes[shp] += 1
        index.append({"file": f"{name}.json", "source": src, "shape": shp, "walls": len(g.walls),
                      "doors": len(g.doors), "windows": len(g.windows), "floor_area_sqft": round(area),
                      "size_band": band, "ceiling_mm": round(min(w.height_m for w in g.walls.values()) * 1000)})
        stats["exported"] += 1
    client.close()
    (OUT / "index.json").write_text(json.dumps({"stats": dict(stats), "shapes": dict(shapes), "rooms": index}, indent=1),
                                    encoding="utf-8")
    print(json.dumps({"stats": dict(stats), "shapes": dict(shapes),
                      "size_bands": dict(collections.Counter(r["size_band"] for r in index)),
                      "doors": dict(collections.Counter(r["doors"] for r in index)),
                      "windows": dict(collections.Counter(r["windows"] for r in index))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
