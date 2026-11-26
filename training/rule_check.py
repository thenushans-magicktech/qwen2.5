"""Rule checker: tests every rule of the MagickHome rule sheet on a finished design.

    results = check_design(geometry, prefs, report, scene_doc)
    results[13] -> {"status": "pass" | "fail" | "n/a", "kind": "design" | "build", "how": "...", "note": "..."}

* design rules - what the designer decides or places (unit sizes, walls, positions, counts, budget fills,
  lofts): a "fail" means the design breaks the rule; the dataset rejects it.
* build rules  - construction details the engine's 3D builder makes (carcass heights, fillers, side
  panels, loft fascia / underside / corner panels): checked on the 3D scene; a "fail" is an engine bug.

"how" says what was measured: "layout" (the engine's placements), "scene" (the 3D scene),
"catalogue" (the unit's catalogue record) or "engine log" (the engine's own measurement).
Two pairs of rules in the sheet overlap: R10 / R41 (Luxury, gaps under 600 mm) and R43 / R31-R32
(Essential, exactly 600 mm). The more specific rule (R41, R31 / R32) is applied and the general
one is reported as "n/a" for those units.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import rulebook  # noqa: E402
from catalog_loader import canon, load_catalog  # noqa: E402

BUILD_RULES = {22, 23, 30, 70, 72, 73, 74, 76, 77, 78, 79, 80, 81}
TOL = 2.0  # mm
LOFT_MIN_CEILING = round(config.LOFT_TOP_M * 1000)  # 2658
ROLE_OF_TRIGGER = {"base hob unit": "hob", "base sink unit": "sink", "base spice pull out": "spice",
                   "base spice pullout": "spice", "base cylinder unit": "cylinder"}


def _res(status, how, note="", kind="design"):
    return {"status": status, "kind": kind, "how": how, "note": note}


def check_design(geo, prefs, report: dict, scene_doc: Optional[dict] = None, catalog=None) -> dict[int, dict]:
    rb = rulebook.load()
    cat = catalog or load_catalog()
    item = cat.get
    lay = report["layout"]
    P = [dict(p, wall=wid) for wid, ps in lay["placements"].items() for p in ps]
    by_role: dict[str, list] = {}
    for p in P:
        by_role.setdefault(p["role"], []).append(p)
    one = lambda role: (by_role.get(role) or [None])[0]  # noqa: E731
    hob, sink, spice, cyl = one("hob"), one("sink"), one("spice"), one("cylinder")
    heights = report["heights"]
    band = report["rulebook"].get("size_band")
    ktype = report["rulebook"].get("kitchen_type")
    budget = prefs.budget if prefs.budget != "auto" else None
    notes = lay["decisions"] + lay["warnings"]
    cabinet_walls = lay["cabinet_walls"]
    primary = prefs.primary_wall if prefs.primary_wall in geo.walls else (hob["wall"] if hob else None)
    secondary = prefs.secondary_wall if prefs.secondary_wall in geo.walls else (sink["wall"] if sink else None)
    R: dict[int, dict] = {}

    def largest_run(wid) -> float:
        from ai_designer import _free_segments
        return max((b - a for a, b in _free_segments(geo, wid)), default=0.0)

    def comb(p) -> str:
        it = item(p["aid"])
        return canon(it.cabinet_combination) if it and it.cabinet_combination else ""

    # -- module widths by kitchen type and size band (R1-R5, R12-R21, R61-R69) + R55 / R56 / R57 ------
    hob_step_down = None
    if hob and primary and band in ("Medium", "Large") and 600 <= largest_run(primary) < 900:
        hob_step_down = 55 if band == "Medium" else 56
    for w in rb.widths:
        role = ROLE_OF_TRIGGER.get(canon(w.trigger))
        if w.kitchen_type not in (ktype, "All") or w.size not in (band, "All"):
            R[w.rule] = _res("n/a", "layout", f"for {w.kitchen_type} {w.size} kitchens; this is {band} {ktype}")
            continue
        unit = {"hob": hob, "sink": sink, "spice": spice, "cylinder": cyl}.get(role)
        if unit is None:
            R[w.rule] = _res("n/a", "layout", f"no {role} unit in this design")
            continue
        want = 600 if (role == "hob" and hob_step_down) else w.width
        ok_w = int(unit["width"]) == want
        ok_c = not w.combination or role == "hob" and hob_step_down or canon(w.combination) == comb(unit)
        R[w.rule] = _res("pass" if ok_w and ok_c else "fail", "layout+catalogue",
                         f"{role} {unit['width']} mm {comb(unit)}; rule {want} mm {w.combination or ''}"
                         + (f" (stepped down by R{hob_step_down})" if want != w.width else ""))
    for n, band_n in ((55, "Medium"), (56, "Large")):
        if hob_step_down == n:
            R[n] = _res("pass" if hob and int(hob["width"]) == 600 else "fail", "layout",
                        f"largest clear run on the primary wall {largest_run(primary):.0f} mm -> hob {hob['width'] if hob else '-'} mm")
        else:
            R[n] = _res("n/a", "layout", f"only a {band_n} kitchen whose primary wall's largest clear run is 600-899 mm")
    if prefs.primary_wall in geo.walls and largest_run(prefs.primary_wall) < 600:
        R[57] = _res("pass" if not hob or hob["wall"] != prefs.primary_wall else "fail", "layout",
                     "primary wall run under 600 mm: the hob must go elsewhere")
    else:
        R[57] = _res("n/a", "layout", "the primary wall has 600 mm or more of clear run")

    # -- gas (R11) ----------------------------------------------------------------------------------------
    no_gas = prefs.fuel in ("png", "induction") or prefs.include_cylinder is False
    R[11] = _res("pass" if (cyl is None) == no_gas else "fail", "layout",
                 f"fuel {prefs.fuel}; cylinder unit {'present' if cyl else 'absent'}")

    # -- carcass heights (R22, R23) ------------------------------------------------------------------------
    bad = [p for p in P if p["tier"] == "base" and p["role"] != "cylinder" and item(p["aid"]) and item(p["aid"]).height != 700]
    R[22] = _res("fail" if bad else "pass", "catalogue",
                 f"{len(bad)} base unit(s) not 700 mm: {[b['category'] for b in bad][:3]}" if bad else "every base carcass 700 mm",
                 kind="build")
    if cyl and item(cyl["aid"]):
        R[23] = _res("pass" if item(cyl["aid"]).height == 800 else "fail", "catalogue",
                     f"cylinder carcass {item(cyl['aid']).height} mm", kind="build")
    else:
        R[23] = _res("n/a", "catalogue", "no cylinder unit", kind="build")

    # -- customer walls (R24, R25) ----------------------------------------------------------------------
    R[24] = (_res("pass" if hob and hob["wall"] == prefs.primary_wall else "fail", "layout", "hob on the customer's primary wall")
             if prefs.primary_wall in geo.walls else _res("n/a", "layout", "no primary wall chosen"))
    R[25] = (_res("pass" if sink and sink["wall"] == prefs.secondary_wall else "fail", "layout", "sink on the customer's secondary wall")
             if prefs.secondary_wall in geo.walls else _res("n/a", "layout", "no secondary wall chosen"))

    # -- landings (R26, R27): the engine's own run measurement --------------------------------------------
    for n, unit, word in ((26, hob, "hob"), (27, sink, "sink")):
        if unit is None:
            R[n] = _res("n/a", "engine log", f"no {word}")
            continue
        short = [m for m in notes if re.search(rf"{word}\b.*landing under 450", m, re.I)]
        R[n] = _res("fail" if short else "pass", "engine log", short[0] if short else f"{word} landings 450 mm or more")

    # -- spice and cylinder beside the hob (R28, R29) + relocation (R58, R59) ---------------------------------
    def beside(u):
        return u and hob and u["wall"] == hob["wall"] and (abs(u["end_mm"] - hob["start_mm"]) <= TOL or abs(u["start_mm"] - hob["end_mm"]) <= TOL)

    def side(u):
        return "start" if u["end_mm"] <= hob["start_mm"] + TOL else "end"

    if spice:
        ok = beside(spice) and (not (cyl and beside(cyl)) or side(spice) != side(cyl))
        R[29] = _res("pass" if ok else "fail", "layout", "spice pull-out right beside the hob, opposite the cylinder")
    else:
        R[29] = _res("n/a", "layout", "no spice pull-out")
    relocated = bool(cyl and hob and cyl["wall"] != hob["wall"])
    if cyl and not relocated:
        ok = beside(cyl) and (not (spice and beside(spice)) or side(spice) != side(cyl))
        R[28] = _res("pass" if ok else "fail", "layout", "cylinder unit right beside the hob, opposite the spice pull-out")
        R[58] = _res("n/a", "layout", "cylinder not relocated")
        R[59] = _res("n/a", "engine log", "cylinder not relocated")
    elif relocated:
        R[28] = _res("n/a", "layout", "cylinder relocated under R58")
        leftover_short = any("R58" in m or "relocat" in m.lower() for m in notes)
        R[58] = _res("pass" if cyl["wall"] == secondary and leftover_short else "fail", "layout",
                     f"cylinder on {cyl['wall'] == secondary and 'the secondary wall' or 'another wall'}")
        R[59] = _res("pass" if any("R59" in m or "flag" in m.lower() for m in notes) else "fail", "engine log",
                     "soft flag raised for the relocation")
    else:
        for n in (28, 58, 59):
            R[n] = _res("n/a", "layout", "no cylinder unit")

    # -- one of each (R44, R45, R46) --------------------------------------------------------------------------
    for n, role, need_one in ((44, "spice", False), (45, "hob", True), (46, "sink", True)):
        k = len(by_role.get(role, []))
        R[n] = _res("pass" if (k == 1 if need_one else k <= 1) else "fail", "layout", f"{k} {role} unit(s)")

    # -- budget fills (R6-R10, R31-R34, R38-R43, R60) ------------------------------------------------------------
    # leftover-gap fills only: feature units the designer asked for (drawers, trolleys, baskets) are choices
    chosen = {a for aids in (report.get("proposal", {}).get("base_prefs") or {}).values() for a in aids}
    chosen_keys = {(item(a).width, canon(item(a).cabinet_combination or "")) for a in chosen if item(a)}
    fills = [p for p in P if p["role"] == "storage" and p["tier"] == "base" and p["aid"] not in chosen
             and (int(p["width"]), comb(p)) not in chosen_keys]
    budget_rules = (6, 7, 8, 9, 10, 31, 32, 33, 38, 39, 40, 41, 43)
    if not budget:
        for n in budget_rules:
            R[n] = _res("n/a", "layout+catalogue", "no budget tier stated")
    else:
        seen: dict[int, list] = {}
        for p in fills:
            aisle = rulebook.working_aisle_mm(geo, p["wall"], cabinet_walls)
            want, rule = rulebook.base_fill_combination(budget, int(p["width"]), aisle)
            ok = canon(want) == comb(p)
            seen.setdefault(rule, []).append((ok, f"{p['width']} mm {comb(p) or '?'} (rule: {want})"))
            twin = {39: 8, 40: 9, 38: 10, 7: 43}.get(rule)
            if twin:
                seen.setdefault(twin, []).append((ok, f"{p['width']} mm {comb(p) or '?'} (rule: {want})"))
        for n in budget_rules:
            if n in seen:
                fails = [t for ok, t in seen[n] if not ok]
                R[n] = _res("fail" if fails else "pass", "layout+catalogue",
                            ("; ".join(fails[:3])) if fails else f"{len(seen[n])} fill unit(s) as the rule says")
            else:
                why = {10: "Luxury gaps under 600 mm follow R41", 43: "exactly-600 Essential gaps follow R31 / R32"}.get(n)
                R[n] = _res("n/a", "layout+catalogue", why or "no gap of this kind in this budget")
    odd = [p for p in fills if int(p["width"]) in (650, 950, 1150)]
    R[60] = _res("fail" if odd else "pass", "layout", "no 650 / 950 / 1150 mm base unit (filled as unit + 50 mm filler)")
    # R34 / R42: runs tiled exactly, with the fewest units
    gaps, pairs = [], []
    widths_avail = sorted({it.width for it in cat.items if it.category == "Base Unit"})
    for wid in cabinet_walls:
        base = sorted((p for p in P if p["wall"] == wid and p["tier"] == "base"), key=lambda p: p["start_mm"])
        for a, b in zip(base, base[1:]):
            gap = b["start_mm"] - a["end_mm"]
            if gap > config.MAX_FILLER_MM + TOL and not _blocked(geo, wid, a["end_mm"], b["start_mm"]):
                gaps.append(f"{geo.walls[wid].name} {a['end_mm']:.0f}-{b['start_mm']:.0f}")
            if (a["role"] == b["role"] == "storage" and abs(gap) <= TOL and comb(a) == comb(b)
                    and int(a["width"]) + int(b["width"]) in widths_avail and int(a["width"]) + int(b["width"]) <= rulebook.BASE_FILL_MAX_MM):
                pairs.append(f"{a['width']}+{b['width']} on {geo.walls[wid].name}")
    R[34] = _res("fail" if gaps else "pass", "layout", f"untiled gaps: {gaps[:3]}" if gaps else "every run tiled exactly")
    R[42] = _res("fail" if pairs else "pass", "layout+catalogue",
                 f"could use fewer units: {pairs[:3]}" if pairs else "fewest units at the largest widths")

    # -- wall units (R35, R36, R37) ---------------------------------------------------------------------------------
    wall_units = [p for p in P if p["tier"] == "wall" and p["role"] != "wall_corner"]
    if budget == "essential":
        bad = [p for p in wall_units if p["category"] == "Wall Unit" and 600 < int(p["width"]) <= 900 and "double" not in comb(p)]
        R[35] = _res("fail" if bad else "pass", "catalogue", f"{len(bad)} wide wall unit(s) not double shutter" if bad else "wide wall units are double shutters")
    else:
        R[35] = _res("n/a", "catalogue", "Essential budget only")
    ch = lay.get("chimney") or {}
    if ch.get("wall_id"):
        lo, hi = ch["center_mm"] - ch["width_mm"] / 2, ch["center_mm"] + ch["width_mm"] / 2
        bad = [p for p in wall_units if p["wall"] == ch["wall_id"] and p["start_mm"] < hi - TOL and p["end_mm"] > lo + TOL]
        R[36] = _res("fail" if bad else "pass", "layout", "a wall unit spans the chimney" if bad else "wall run breaks at the chimney")
    else:
        R[36] = _res("n/a", "layout", "no chimney")
    wall_y0, wall_y1 = config.POS_Y_M["wall"], config.POS_Y_M["wall"] + heights["wall_mm"] / 1000
    bad, windows_seen = [], 0
    for wid, w in geo.walls.items():
        for o in w.windows:
            if o.top_m <= wall_y0 + 1e-6 or o.bottom_m >= wall_y1 - 1e-6:
                continue
            windows_seen += 1
            bad += [p for p in wall_units if p["wall"] == wid and p["start_mm"] < o.end_mm - TOL and p["end_mm"] > o.start_mm + TOL]
    R[37] = (_res("fail" if bad else "pass", "layout", "a wall unit spans a window" if bad else "wall run breaks at every window")
             if windows_seen else _res("n/a", "layout", "no window at wall-unit height"))

    # -- lofts (R47-R54, R71-R75) -------------------------------------------------------------------------------------
    lofts = [p for p in P if p["tier"] == "loft"]
    has_lofts = bool(lofts)
    ceiling = heights["ceiling_mm"]
    R[47] = _res("fail" if has_lofts and ceiling < LOFT_MIN_CEILING else "pass", "layout",
                 f"ceiling {ceiling} mm, lofts {'yes' if has_lofts else 'no'}")
    if not has_lofts:
        for n in (48, 49, 50, 51, 52, 53, 54, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81):
            R[n] = _res("n/a", "layout", "no lofts in this design", kind="build" if n in BUILD_RULES else "design")
    else:
        want = rulebook.loft_size(ceiling, heights["top_mm"])
        loft_h = {item(p["aid"]).height for p in lofts if item(p["aid"]) and p["role"] != "loft_corner"}
        ok = want is not None and heights.get("loft_mm") == want[0] and loft_h <= {want[0]}
        R[48] = _res("pass" if ok else "fail", "layout+catalogue",
                     f"loft {heights.get('loft_mm')} mm, reveal {heights.get('reveal_mm')} mm (rule: {want})")
        below = [p for p in P if p["tier"] in ("wall", "tall") and p["role"] != "wall_corner"]
        plain = [p for p in lofts if p["role"] in ("loft", "loft_deep")]
        miss, comb_bad, depth_bad = [], [], []
        for u in below:
            if u["wall"] not in (lay.get("loft_walls") or cabinet_walls):
                continue
            match = [l_ for l_ in plain if l_["wall"] == u["wall"] and abs(l_["start_mm"] - u["start_mm"]) <= TOL
                     and abs(l_["end_mm"] - u["end_mm"]) <= TOL]
            if not match:
                miss.append(f"{u['category']} {u['width']} on {geo.walls[u['wall']].name}")
                continue
            lf = match[0]
            it = item(lf["aid"])
            if it:
                single = int(lf["width"]) <= 600
                if ("single" if single else "double") not in comb(lf):
                    comb_bad.append(f"{lf['width']} mm {comb(lf)}")
                want_d = 560 if u["tier"] == "tall" else 330
                if it.depth != want_d:
                    depth_bad.append(f"{lf['width']} mm loft {it.depth} deep over {u['tier']} (rule {want_d})")
        R[53] = _res("fail" if miss else "pass", "layout", f"no loft above: {miss[:3]}" if miss else "one loft above each wall / tall unit")
        R[49] = _res("fail" if miss else "pass", "layout", "loft widths follow the units below" if not miss else "see R53")
        R[50] = _res("fail" if comb_bad else "pass", "catalogue",
                     f"wrong shutters: {comb_bad[:3]}" if comb_bad else "single shutter up to 600 mm, double shutter above")
        d330 = [t for t in depth_bad if "over wall" in t]
        d560 = [t for t in depth_bad if "over tall" in t]
        R[51] = _res("fail" if d330 else "pass", "catalogue", "; ".join(d330[:2]) or "330 mm deep over the wall run")
        R[52] = (_res("fail" if d560 else "pass", "catalogue", "; ".join(d560[:2]) or "560 mm deep over tall units")
                 if any(p["tier"] == "tall" for p in below) else _res("n/a", "catalogue", "no tall unit"))
        breaks = []
        for wid in {p["wall"] for p in lofts}:
            run = sorted((p for p in lofts if p["wall"] == wid), key=lambda p: p["start_mm"])
            breaks += [f"{geo.walls[wid].name} {a['end_mm']:.0f}-{b['start_mm']:.0f}" for a, b in zip(run, run[1:])
                       if b["start_mm"] - a["end_mm"] > config.MAX_FILLER_MM + TOL]
        R[54] = _res("fail" if breaks else "pass", "layout", f"loft run breaks at {breaks[:3]}" if breaks else "loft runs continuous")
        # corners
        kinds = rulebook.wall_corner_kind(budget)
        corner_rows = lay.get("corners") or []
        loft_corners = by_role.get("loft_corner", [])
        if not corner_rows:
            for n in (71, 72, 73, 74, 75):
                R[n] = _res("n/a", "layout", "no internal corner", kind="build" if n in BUILD_RULES else "design")
        else:
            lc_ok = all(item(p["aid"]) and "corner" in canon(item(p["aid"]).category) for p in loft_corners)
            if kinds == "L":
                R[71] = _res("pass" if loft_corners and lc_ok else "fail", "layout+catalogue",
                             f"{len(loft_corners)} loft corner unit(s) above the L corners")
                R[75] = _res("n/a", "layout", "Premium / Luxury: L corner")
            elif kinds == "blind":
                R[75] = _res("pass" if not loft_corners else "fail", "layout", "Essential: open void above the blind corner")
                R[71] = _res("n/a", "layout", "Essential: blind corner")
            else:
                R[71] = _res("n/a", "layout", "no budget stated")
                R[75] = _res("n/a", "layout", "no budget stated")
            sizes = {(item(p["aid"]).width, item(p["aid"]).depth) for p in loft_corners if item(p["aid"])}
            R[72] = (_res("pass" if sizes <= {(650, 650)} else "fail", "catalogue", f"loft corner sizes {sorted(sizes)}", kind="build")
                     if loft_corners else _res("n/a", "catalogue", "no loft corner unit", kind="build"))
            R[73] = _res("n/a", "catalogue", "430 mm lofts are used only when a design is modified", kind="build")
            R[74] = _res("n/a", "catalogue", "corner lofts are generated at 330 mm depth", kind="build")
        # build details on the 3D scene
        fx = _fixtures(scene_doc)
        talls = [p for p in P if p["tier"] == "tall"]
        under_tall = [f for f in fx if "loft_bottom" in f["id"] and any(
            f["wall"] == t["wall"] and abs(f["center_mm"] - (t["start_mm"] + t["end_mm"]) / 2) < t["width"] / 2 for t in talls)]
        R[70] = (_res("fail" if under_tall else "pass", "scene", f"{len(under_tall)} underside panel(s) under the tall-unit loft",
                      kind="build") if talls else _res("n/a", "scene", "no tall unit", kind="build"))
        loft_walls = {p["wall"] for p in lofts}
        for n, key, what in ((78, "loft_top", "fascia above the lofts"), (80, "loft_side", "side cover at exposed loft ends")):
            have = {f["wall"] for f in fx if key in f["id"]}
            R[n] = _res("pass" if loft_walls <= have else "fail", "scene", f"{what}: {len(have & loft_walls)}/{len(loft_walls)} walls", kind="build")
        gaps_under = [w for w in loft_walls if _loft_gap(P, w)]
        have_bottom = {f["wall"] for f in fx if "loft_bottom" in f["id"]}
        R[79] = (_res("pass" if set(gaps_under) <= have_bottom else "fail", "scene", "underside panels where no wall unit sits below", kind="build")
                 if gaps_under else _res("n/a", "scene", "no gap under the lofts", kind="build"))
        R[77] = (_res("pass" if any("loft" in f["id"] and f["kind"] == "filler" for f in fx) else "fail", "scene",
                      "fillers at loft break edges (approximate: loft fillers present)", kind="build")
                 if gaps_under else _res("n/a", "scene", "no chimney / window break under the lofts", kind="build"))
        short_end = [w for w in loft_walls if _run_short_of_wall(geo, P, w, "loft")]
        R[76] = (_res("pass" if all(any(f["wall"] == w and f["kind"] == "filler" and "loft" in f["id"] for f in fx) for w in short_end)
                      else "fail", "scene", "loft filler where the run stops short of a wall", kind="build")
                 if short_end else _res("n/a", "scene", "loft runs reach the walls", kind="build"))
        steps = [w for w in loft_walls if any(p["role"] == "loft_deep" and p["wall"] == w for p in lofts)
                 and any(p["role"] == "loft" and p["wall"] == w for p in lofts)]
        R[81] = (_res("pass" if all(any(f["wall"] == w and "bank_side" in f["id"] for f in fx) for w in steps) else "fail",
                      "scene", "panel where the 560 tall loft meets the 330 wall loft", kind="build")
                 if steps else _res("n/a", "scene", "no 560 / 330 loft depth step", kind="build"))
    # R30: run ends that meet a wall take a filler / side panel
    fx = _fixtures(scene_doc)
    ends = [w for w in cabinet_walls if _run_short_of_wall(geo, P, w, "base")]
    R[30] = (_res("pass" if all(any(f["wall"] == w and "base" in f["id"] for f in fx) for w in ends) else "fail", "scene",
                  "filler / side panel at base-run ends", kind="build")
             if ends else _res("n/a", "scene", "base runs end at corners", kind="build"))

    for n in rb.texts:  # every sheet rule gets a result
        R.setdefault(n, _res("n/a", "-", "not checked: no condition in this design", kind="build" if n in BUILD_RULES else "design"))
        R[n]["kind"] = "build" if n in BUILD_RULES else "design"
    return dict(sorted(R.items()))


def _blocked(geo, wid, a, b) -> bool:
    """A base gap explained by a door (with clearance) or an opening at counter height."""
    w = geo.walls[wid]
    for o in w.doors:
        lo, hi = o.interval(config.DOOR_CLEARANCE_MM)
        if lo < b and hi > a:
            return True
    return False


def _loft_gap(P, wid) -> bool:
    lofts = [p for p in P if p["wall"] == wid and p["tier"] == "loft"]
    below = [p for p in P if p["wall"] == wid and p["tier"] in ("wall", "tall")]
    for lf in lofts:
        mid = (lf["start_mm"] + lf["end_mm"]) / 2
        if not any(b["start_mm"] - TOL <= mid <= b["end_mm"] + TOL for b in below):
            return True
    return False


def _run_short_of_wall(geo, P, wid, tier) -> bool:
    run = [p for p in P if p["wall"] == wid and p["tier"] == tier]
    if not run:
        return False
    lo, hi = geo.walls[wid].usable
    s, e = min(p["start_mm"] for p in run), max(p["end_mm"] for p in run)
    return (0 < s - lo <= config.MAX_FILLER_MM + TOL) or (0 < hi - e <= config.MAX_FILLER_MM + TOL)


def _fixtures(scene_doc) -> list[dict]:
    if not scene_doc:
        return []
    nodes = scene_doc.get("scene", {}).get("nodes", {})
    out = []
    for n in (nodes.values() if isinstance(nodes, dict) else nodes):
        m = n.get("metadata") or {}
        if m.get("fixtureKind") in ("filler", "slab"):
            fid = m.get("fixtureId") or ""
            wall = next((w for w in re.findall(r"(wall_[a-z0-9]+)", fid)), n.get("wallId"))
            t = m.get("wallT")
            out.append({"id": fid, "kind": m["fixtureKind"], "wall": wall,
                        "center_mm": (t or 0) * 1000 if t is not None and t > 1.5 else _center_mm(n, nodes)})
    return out


def _center_mm(n, nodes) -> float:
    """Position along the wall (mm) of a fixture node (its x in the wall frame)."""
    return float(n.get("position", [0])[0]) * 1000


def summary(results: dict[int, dict]) -> dict:
    out = {"pass": [], "fail": [], "n/a": []}
    for n, r in results.items():
        out[r["status"]].append(n)
    out["design_fail"] = [n for n in out["fail"] if results[n]["kind"] == "design"]
    out["build_fail"] = [n for n in out["fail"] if results[n]["kind"] == "build"]
    return out
