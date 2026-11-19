"""Designer benchmark: does the knowledge pack make Qwen design better?

Runs the same test rooms through each prompt version and scores *Qwen's own
decisions* (before the rule engine corrects anything) against a designer
checklist, plus how much the solver had to correct.

    python designer_benchmark.py                       # live Qwen, v2 then v1, all rooms (~2-4 min per design on CPU)
    python designer_benchmark.py --versions v2 --rooms compact big_U
    python designer_benchmark.py --ai-mode fallback    # quick sanity run of the scorer

Results: benchmark/results_<time>.json and benchmark/REPORT.md. The checklist is a
designer rubric (see CHECKS) - MagickHome designers should review and extend it.
Catalogue: offline snapshot (no network calls); same AIDs as the Assets API.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("MAGICKHOME_ASSETS_API_URL", "")  # offline: no API calls during benchmarking

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from ai_designer import KitchenPreferences, _free_len, _free_segments  # noqa: E402
from catalog_loader import CAT_GRAIN, CAT_WICKER, canon, load_filtered_catalog  # noqa: E402
from main import generate_kitchen  # noqa: E402
from scene_parser import parse_scene, strip_items  # noqa: E402

PRIMARY = ROOT / "magickhomes-scene_2026-11-18_10-38-22.json"
WALL_IDS = ["wall_1oy5ecjq6k11pfoo", "wall_56i56ymve0xf69fm", "wall_49fdugi4g7vj12pf", "wall_7sna3rldnsuwipgy"]
WINDOW_ID, DOOR_ID = "window_a6jgraczhlas4vwp", "door_6a3cugqojq1bwey8"
X0, Z1 = -3.65, -0.5  # front-left corner of the sample room (kept fixed)


# ---------------------------------------------------------------------------
# Test rooms (rectangles built from the production sample; walls 1-4 in order)
# ---------------------------------------------------------------------------

@dataclass
class Room:
    key: str
    title: str
    width_mm: Optional[int] = None  # Wall 2 / Wall 4 length (None = sample as is)
    depth_mm: Optional[int] = None  # Wall 1 / Wall 3 length
    windows: list = field(default_factory=list)  # (wall 1-4, centre mm, width mm)
    doors: list = field(default_factory=list)
    height_m: float = 2.5

    def build(self) -> dict:
        doc = strip_items(json.loads(PRIMARY.read_text(encoding="utf-8")))
        if self.width_mm is None:
            return doc
        nodes = doc["scene"]["nodes"]
        x1, z0 = X0 + self.width_mm / 1000, Z1 - self.depth_mm / 1000
        pts = [(X0, Z1), (X0, z0), (x1, z0), (x1, Z1)]
        for i, wid in enumerate(WALL_IDS):
            nodes[wid]["start"], nodes[wid]["end"] = list(pts[i]), list(pts[(i + 1) % 4])
        for n in nodes.values():
            if n["type"] in ("slab", "ceiling") and isinstance(n.get("polygon"), list):
                n["polygon"] = [list(p) for p in pts]
            if n["type"] == "ceiling":
                n["height"] = self.height_m
        templates = {"window": copy.deepcopy(nodes[WINDOW_ID]), "door": copy.deepcopy(nodes[DOOR_ID])}
        for oid in (WINDOW_ID, DOOR_ID):
            for n in nodes.values():
                if oid in n.get("children", []):
                    n["children"].remove(oid)
            del nodes[oid]
        for kind, specs, first_id in (("window", self.windows, WINDOW_ID), ("door", self.doors, DOOR_ID)):
            for k, (wall_no, centre_mm, width_mm) in enumerate(specs):
                o = copy.deepcopy(templates[kind])
                o["id"] = first_id if k == 0 else f"{kind}_bench{k:012d}"
                wid = WALL_IDS[wall_no - 1]
                o["wallId"] = o["parentId"] = wid
                o["position"][0] = centre_mm / 1000
                o["width"] = width_mm / 1000
                nodes[o["id"]] = o
                nodes[wid]["children"].append(o["id"])
        return doc


ROOMS = [
    Room("reference", "Production sample 4.1 x 4.1 m: window on Wall 2, door on Wall 1"),
    Room("compact", "Small 2.7 x 2.4 m: window on Wall 2, door on Wall 4 near the Wall 1 corner",
         2700, 2400, windows=[(2, 1350, 1000)], doors=[(4, 2050, 900)]),
    Room("window_short_wall", "3.0 x 4.2 m: window on the short Wall 2, door on Wall 4 near the Wall 3 corner",
         3000, 4200, windows=[(2, 1500, 1200)], doors=[(4, 750, 900)]),
    Room("door_by_window_corner", "3.6 x 3.3 m: door on Wall 1 right by the window-wall corner",
         3600, 3300, windows=[(2, 1800, 1200)], doors=[(1, 2650, 900)]),
    Room("no_window", "3.6 x 3.0 m: no window, door on Wall 4",
         3600, 3000, doors=[(4, 1800, 900)]),
    Room("big_U", "Large 4.5 x 3.6 m, 2.7 m high: wide window on Wall 2, door on Wall 4 near the Wall 1 corner",
         4500, 3600, windows=[(2, 2250, 1500)], doors=[(4, 3600, 900)], height_m=2.7),
    Room("window_near_corner", "3.6 x 3.3 m: window close to the Wall 2 / Wall 3 corner",
         3600, 3300, windows=[(2, 2900, 1000)], doors=[(4, 2700, 900)]),
    Room("narrow_galley", "Narrow 4.8 x 2.1 m: long window wall, door in the short Wall 1",
         4800, 2100, windows=[(2, 2400, 1200)], doors=[(1, 1050, 900)]),
]


# ---------------------------------------------------------------------------
# Designer scorecard (Qwen's intent, before the solver corrects it)
# ---------------------------------------------------------------------------

CHECKS = {
    "reads_room": "room reading names the window wall and the door wall correctly",
    "sink_on_window_wall": "sink on the window wall",
    "hob_on_solid_wall": "hob on a solid wall (when a long solid wall is available)",
    "sink_hob_apart": "sink and hob on different walls, or 600+ mm counter between them",
    "hob_clear_of_door": "hob 600+ mm from any door on its wall",
    "hob_position_sane": "requested hob centre leaves 300+ mm counter both sides inside a free run",
    "sink_size_suits_wall": "sink width suits the window wall (1050 needs 3000+, 1200 needs 3600+ mm free)",
    "hob_unit_has_drawers": "drawer hob unit (pans lie flat)",
    "spice_beside_hob": "spice pull-out chosen",
    "tall_well_placed": "tall unit chosen on a cabinet wall, not crowding the hob wall",
    "walls_connected": "cabinet walls form one connected run including the window wall",
    "uses_the_room": "L or U when the window wall has a usable adjacent wall",
    "features_zoned": "drawers on the hob wall, wicker by the sink, grain away from the hob",
    "wall_cabinets_planned": "wall-cabinet sizes chosen, only for cabinet walls",
}
CORRECTION_PATTERNS = (
    "sink moved", "hob moved", "re-chosen", "chosen by solver", "added window wall",
    "dropped", "not a valid", "not in the filtered", "skipped", "unresolved", "removed",
)


def _clauses(text: str) -> list[str]:
    return [c.strip().lower() for c in re.split(r"[.;,\n]| and ", text or "") if c.strip()]


def score_design(geometry, catalog, report: dict, prefs: KitchenPreferences) -> dict:
    prop = report["proposal"]
    g = geometry
    walls = prop["cabinet_walls"]
    name = {w: g.walls[w].name for w in g.wall_order}
    free = {w: _free_len(g, w) for w in g.wall_order}
    usable = {w: free[w] >= config.MIN_CABINET_WALL_MM for w in g.wall_order}
    win = max(g.windows, key=lambda o: o.width_mm) if g.windows else None
    win_wall = win.wall_id if win else None
    item = catalog.get
    sink_w, hob_w = prop["sink"]["wall_id"], prop["hob"]["wall_id"]
    sink_it, hob_it = item(prop["sink"]["aid"]), item(prop["hob"]["aid"])
    hob_c = prop["hob"]["center_mm"]
    adjacent = lambda a: [c.other(a)[1] for c in g.corners_of(a)]  # noqa: E731
    c: dict[str, Optional[bool]] = {}

    # 1. reading the room
    clauses = _clauses(prop.get("notes", ""))
    ok = True
    if win_wall:
        ok &= any("window" in s and name[win_wall].lower() in s for s in clauses)
    for d in g.doors:
        ok &= any("door" in s and name[d.wall_id].lower() in s for s in clauses)
    c["reads_room"] = ok if (win_wall or g.doors) else None

    # 2-3. wet and cooking zones
    c["sink_on_window_wall"] = (sink_w == win_wall) if win_wall and usable[win_wall] else None
    cands = adjacent(win_wall) if win_wall else g.wall_order
    solid = [w for w in cands if not g.walls[w].windows and free[w] >= 2400]
    c["hob_on_solid_wall"] = (not g.walls[hob_w].windows) if solid else None

    # 4. sink-hob separation
    if sink_w != hob_w:
        c["sink_hob_apart"] = True
    elif hob_c is not None and sink_it and hob_it:
        sink_c = win.center_mm if win and win.wall_id == sink_w else None
        c["sink_hob_apart"] = None if sink_c is None else abs(hob_c - sink_c) - (sink_it.width + hob_it.width) / 2 >= 600
    else:
        c["sink_hob_apart"] = None

    # 5-6. hob position
    if hob_c is not None and hob_it:
        lo, hi = hob_c - hob_it.width / 2, hob_c + hob_it.width / 2
        gaps = [max(d.start_mm - hi, lo - d.end_mm) for d in g.walls[hob_w].doors]
        c["hob_clear_of_door"] = all(x >= 600 for x in gaps)
        m = hob_it.width / 2 + 300
        c["hob_position_sane"] = any(s + m <= hob_c <= e - m for s, e in _free_segments(g, hob_w))
    else:
        c["hob_clear_of_door"] = c["hob_position_sane"] = None

    # 7-9. units
    if sink_it:
        f = free[sink_w]
        c["sink_size_suits_wall"] = sink_it.width <= 900 or (sink_it.width <= 1050 and f >= 3000) or f >= 3600
    c["hob_unit_has_drawers"] = bool(hob_it) and "drawer" in canon(hob_it.cabinet_combination)
    c["spice_beside_hob"] = prop.get("spice_aid") is not None

    # 10. tall units
    talls = prop.get("tall_units", [])
    if prefs.include_tall:
        c["tall_well_placed"] = bool(talls) and all(
            t["wall_id"] in walls and (t["wall_id"] != hob_w or free[hob_w] >= 3600) for t in talls
        )

    # 11-12. layout
    connected = len(walls) == 1 or all(any(o in walls for o in adjacent(w)) for w in walls)
    has_win = (win_wall in walls) if win_wall and usable[win_wall] else True
    c["walls_connected"] = connected and has_win
    if win_wall and usable[win_wall]:
        roomy = [a for a in adjacent(win_wall) if free[a] >= 2400]
        c["uses_the_room"] = (len(walls) >= 2) if roomy else None

    # 13. feature zoning
    good = total = 0
    for wid, aids in prop.get("base_prefs", {}).items():
        for aid in aids:
            it = item(aid)
            if it is None:
                continue
            if it.cat == CAT_WICKER:
                total, good = total + 1, good + (wid == sink_w)
            elif it.cat == CAT_GRAIN:
                total, good = total + 1, good + (wid != hob_w)
            elif "drawer" in canon(it.cabinet_combination):
                total, good = total + 1, good + (wid == hob_w)
    c["features_zoned"] = (good / total >= 0.5) if total else None

    # 14. wall cabinets
    wp = prop.get("wall_prefs", {})
    c["wall_cabinets_planned"] = bool(wp) and all(w in walls for w in wp)

    applicable = {k: v for k, v in c.items() if v is not None}
    passed = [k for k, v in applicable.items() if v]

    # solver side
    lay = report["layout"]
    corrections = [m for m in lay["decisions"] + lay["warnings"] if any(p in m for p in CORRECTION_PATTERNS)]
    final_hob = next(
        ((p["start_mm"] + p["end_mm"]) / 2, wid)
        for wid, ps in lay["placements"].items() for p in ps if p["role"] == "hob"
    ) if any(p["role"] == "hob" for ps in lay["placements"].values() for p in ps) else (None, None)
    hob_shift = abs(final_hob[0] - hob_c) if final_hob[0] is not None and hob_c is not None and final_hob[1] == hob_w else None
    extras = {
        "water_purifier": bool(prop.get("purifier_aid")),
        "purifier_built": any(p["role"] == "purifier" for ps in lay["placements"].values() for p in ps),
        "display_unit": bool(prop.get("display_aid")),
        "tall_types": sorted({item(t["aid"]).category for t in talls if item(t["aid"])}),
    }
    return {
        "score": round(100 * len(passed) / max(1, len(applicable)), 1),
        "passed": len(passed),
        "applicable": len(applicable),
        "checks": c,
        "failed": [k for k, v in applicable.items() if not v],
        "corrections": corrections,
        "hob_shift_mm": None if hob_shift is None else round(hob_shift),
        "extras": extras,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(room: Room, version: str, ai_mode: str) -> dict:
    config.AI_PROMPT_VERSION = version
    doc = room.build()
    prefs = KitchenPreferences(ai_mode=ai_mode)
    t0 = time.time()
    try:
        out, report = generate_kitchen(doc, prefs)
    except Exception as exc:  # a failed design is a result too
        return {"room": room.key, "version": version, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 1)}
    geometry = parse_scene(doc)
    catalog = load_filtered_catalog(prefs.door_style, prefs.handle_type)
    from test_pipeline import verify_scene

    expect_win = max(geometry.windows, key=lambda o: o.width_mm).id if geometry.windows else None
    errs = verify_scene(out, expect_window=expect_win, expect_door=DOOR_ID if geometry.doors else None)
    s = score_design(geometry, catalog, report, prefs)
    ai = report["ai"]
    return {
        "room": room.key,
        "version": version,
        "mode": ai.get("mode"),
        **s,
        "verify_errors": errs,
        "seconds": round(time.time() - t0, 1),
        "llm": {k: ai.get(k) for k in ("seconds", "prompt_tokens", "prompt_seconds", "eval_count", "eval_seconds", "attempts")},
        "reading": report["proposal"].get("notes", ""),
        "proposal": report["proposal"],
        "layout_walls": [geometry.walls[w].name for w in report["layout"]["cabinet_walls"]],
    }


def write_report(results: list[dict], path: Path, meta: dict) -> None:
    versions = list(dict.fromkeys(r["version"] for r in results))
    rooms = list(dict.fromkeys(r["room"] for r in results))
    by = {(r["room"], r["version"]): r for r in results}
    L = [
        "# Qwen designer benchmark",
        "",
        f"Model `{meta['model']}`, AI mode `{meta['ai_mode']}`, run {meta['started']}.",
        "Score = share of applicable designer checks that Qwen's own decisions pass (before the rule engine corrects them).",
        "",
        "| Room | " + " | ".join(f"{v} score | {v} fixes | {v} time" for v in versions) + " |",
        "|---|" + "---|---|---|" * len(versions),
    ]
    for room in rooms:
        cells = []
        for v in versions:
            r = by.get((room, v))
            if not r or "error" in r:
                cells += ["error", "-", "-"]
            else:
                cells += [f"{r['score']:.0f}% ({r['passed']}/{r['applicable']})", str(len(r["corrections"])), f"{r['llm'].get('seconds') or r['seconds']:.0f}s"]
        L.append(f"| {room} | " + " | ".join(cells) + " |")
    L.append("")
    for v in versions:
        rs = [r for r in results if r["version"] == v and "error" not in r]
        if rs:
            avg = sum(r["score"] for r in rs) / len(rs)
            fixes = sum(len(r["corrections"]) for r in rs) / len(rs)
            secs = [r["llm"].get("seconds") or r["seconds"] for r in rs]
            ok = sum(1 for r in rs if not r["verify_errors"])
            L.append(f"- **{v}**: average score {avg:.1f}%, {fixes:.1f} solver corrections per design, "
                     f"{sum(secs) / len(secs):.0f}s per design, {ok}/{len(rs)} scenes pass verification.")
    L += ["", "## Failed checks per version", ""]
    for v in versions:
        counts: dict[str, int] = {}
        for r in results:
            if r["version"] == v and "error" not in r:
                for k in r["failed"]:
                    counts[k] = counts.get(k, 0) + 1
        L.append(f"- **{v}**: " + (", ".join(f"{k} x{n}" for k, n in sorted(counts.items(), key=lambda t: -t[1])) or "none"))
    L += ["", "## Checks", ""] + [f"- `{k}`: {d}" for k, d in CHECKS.items()]
    L += ["", "## Qwen's reasoning per room", ""]
    for room in rooms:
        L.append(f"### {room}")
        for v in versions:
            r = by.get((room, v))
            if not r:
                continue
            if "error" in r:
                L.append(f"- **{v}**: {r['error']}")
                continue
            ex = r["extras"]
            L.append(f"- **{v}** ({r['score']:.0f}%, walls {', '.join(r['layout_walls'])}): {r['reading']}")
            L.append(f"  - failed: {', '.join(r['failed']) or 'none'}; purifier {'yes' if ex['water_purifier'] else 'no'}; "
                     f"talls: {', '.join(ex['tall_types']) or 'none'}; corrections: {'; '.join(r['corrections']) or 'none'}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--versions", nargs="+", default=["v2", "v1"], choices=["v1", "v2"])
    ap.add_argument("--rooms", nargs="+", default=["all"])
    ap.add_argument("--ai-mode", default="ollama", choices=["ollama", "fallback", "auto"])
    ap.add_argument("--out", default=str(ROOT / "benchmark"))
    args = ap.parse_args()
    rooms = ROOMS if args.rooms == ["all"] else [r for r in ROOMS if r.key in args.rooms]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    meta = {"model": config.OLLAMA_MODEL, "ai_mode": args.ai_mode, "started": started}
    results = []
    # One version at a time so Ollama can reuse the cached prompt prefix.
    for v in args.versions:
        for room in rooms:
            r = run_one(room, v, args.ai_mode)
            results.append(r)
            msg = r.get("error") or f"score {r['score']:.0f}% ({r['passed']}/{r['applicable']}), fixes {len(r['corrections'])}, failed {r['failed']}"
            print(f"[{v}] {room.key:22s} {r['seconds']:6.0f}s  {msg}", flush=True)
            (out / f"results_{stamp}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
            write_report(results, out / "REPORT.md", meta)
    print(f"report: {out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
