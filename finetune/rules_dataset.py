"""Training examples from the CURRENT RULES alone (no teacher LLM, runs offline in minutes).

For each random room and brief: the rule-based designer proposes, the MagickHome rule book sets the module
sizes, the rule engine lays everything out. The FINAL, rule-checked layout is written back as the answer the
model should give - in the exact answer format - with reasons taken from the playbook and the rule book.
Each answer is then sent through the real pipeline again (as if the model had given it) and kept only if the
engine needs 0 corrections, the layout has 0 issues and the designer scorecard is >= FINETUNE_MIN_SCORE.

    python finetune/rules_dataset.py --n 600            # -> finetune/data/rules.jsonl (resumable)

This teaches the model what the prompt's knowledge pack teaches today (rules, sizes, zones, clearances).
Teacher-LLM data (build_dataset.py) and designer samples add judgement and variety on top later.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter

import common  # noqa: F401  (sets sys.path)
from common import DATA, append_jsonl, catalog_and_groups, judge, make_row, read_jsonl
import config
import finishes
import main as app
from ai_designer import (KitchenPreferences, _design_to_proposal_v2, _free_len, allowed_codes, allowed_lofts,
                         allowed_storage, allowed_wall_styles, base_storage_need, build_menu, build_messages,
                         finish_codes, lofts_possible, wall_labels)
from rooms import random_prefs, random_room
from scene_parser import parse_scene, strip_items

RULES = DATA / "rules.jsonl"
REJECTED = DATA / "rules_rejected.jsonl"


def cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0].rstrip(",;:") + "."


class Answer:
    """Builds one answer (answer-format dict) from a finished report."""

    def __init__(self, doc: dict, prefs: KitchenPreferences, report: dict):
        self.prefs, self.report = prefs, report
        self.g = parse_scene(strip_items(doc))
        self.catalog, self.groups = catalog_and_groups(prefs.door_style, prefs.handle_type)
        self.menu = build_menu(self.groups)
        self.labels = wall_labels(self.g)
        self.L = {wid: label for label, wid in self.labels.items()}
        self.lay = report["layout"]
        self.places = [dict(p, wall=w) for w, ps in self.lay["placements"].items() for p in ps]

    # -- helpers -------------------------------------------------------------
    def role(self, *roles):
        return [p for p in self.places if p["role"] in roles]

    def code(self, group: str, aid) -> str:
        """Menu code for a catalogue aid (same unit, else the nearest allowed width), or NONE."""
        allowed = allowed_codes(self.menu, group, self.prefs) or ["NONE"]
        if aid is None:
            return "NONE" if "NONE" in allowed else allowed[0]
        opts = [(c, it) for c, it in self.menu.get(group, []) if c in allowed]
        for c, it in opts:
            if it.aid == aid:
                return c
        it0 = self.catalog.get(aid)
        if not opts:
            return allowed[0]
        if it0 is None:
            return opts[0][0]
        same = [o for o in opts if o[1].category == it0.category] or opts
        return min(same, key=lambda o: abs(o[1].width - it0.width))[0]

    def corner_of(self, wid: str, end: str) -> str | None:
        for c in self.g.corners_of(wid):
            my_end, other, _ = c.other(wid)
            if my_end == end:
                return self.L[other]
        return None

    def door_text(self, d) -> str:
        w = self.g.walls[d.wall_id]
        near = None
        if d.start_mm < 700:
            near = self.corner_of(d.wall_id, "start")
        elif w.length_mm - d.end_mm < 700:
            near = self.corner_of(d.wall_id, "end")
        return f"door on {self.L[d.wall_id]}" + (f" near the {near} corner" if near else "")

    # -- the answer ------------------------------------------------------------
    def build(self) -> dict:
        g, L, p, rep = self.g, self.L, self.prefs, self.report
        rb = rep.get("rulebook") or {}
        rules = {a["role"]: a for a in rb.get("applied", [])}
        walls = self.lay["cabinet_walls"]
        sink = (self.role("sink") or [None])[0]
        hob = (self.role("hob") or [None])[0]
        win = max(g.windows, key=lambda o: o.width_mm) if g.windows else None
        talls = [t for t in self.role("tall")][:2]
        prop = rep["proposal"]

        # room reading / plan
        solid = [L[w] for w in g.wall_order if not g.walls[w].openings and _free_len(g, w) >= 2400]
        reading = []
        reading.append(f"Window on {L[win.wall_id]} ({win.start_mm:.0f}-{win.end_mm:.0f} mm)" if win else "No window")
        reading += [d for d in (self.door_text(d) for d in g.doors)]
        if solid:
            reading.append(f"long solid walls: {', '.join(solid)}")
        room_reading = cut("; ".join(reading) + ".", 240)
        tall_name = self.catalog.get(talls[0]["aid"]).category.lower() if talls and self.catalog.get(talls[0]["aid"]) else None
        plan = [f"Wet zone: sink on {L[sink['wall']]}" + (" under the window" if win and sink and sink["wall"] == win.wall_id else "") if sink else "",
                f"Cooking: hob on {L[hob['wall']]} with the chimney above" if hob else "",
                f"Storage: {tall_name} on {L[talls[0]['wall']]} at a run end" if talls else ""]
        design_plan = cut(". ".join(x for x in plan if x) + ". Work triangle kept short and clear.", 200)

        # layout
        kt, band, area = rb.get("kitchen_type", ""), rb.get("size_band", ""), rb.get("floor_area_sqft", 0)
        names = " + ".join(L[w] for w in walls)
        why_layout = f"{band} {kt} kitchen ({area:.0f} sqft): {names}"
        if win and win.wall_id in walls:
            why_layout += f" - the window wall {L[win.wall_id]} anchors the run"
        if len(walls) == 3:
            mid = next((w for w in walls if all(g.corner_between(w, o) for o in walls if o != w)), None)
            if mid:
                aisle = g.walls[mid].usable[1] - g.walls[mid].usable[0] - 2 * config.CORNER_DEPTH_MM
                why_layout += f"; aisle {aisle:.0f} mm (needs 1200+)"
        elif len(walls) == 1:
            why_layout += "; the other walls are too short or blocked for a second run"

        # sink
        sfree = _free_len(g, sink["wall"]) if sink else 0
        sink_why = (f"Centred under the {L[sink['wall']]} window for light and air; " if win and sink and sink["wall"] == win.wall_id
                    else f"No usable window wall, so the longest free run ({L[sink['wall']]}); ") if sink else ""
        if "sink" in rules:
            sink_why += f"rule R{rules['sink']['rule']} sets a {rules['sink']['width']} mm sink unit for this kitchen."
        else:
            sink_why += f"{sink['width']} mm suits a {sfree:.0f} mm free run."

        # hob
        hw = g.walls[hob["wall"]] if hob else None
        hob_c = round((hob["start_mm"] + hob["end_mm"]) / 2) if hob else 0
        bits = []
        if hob:
            bits.append(f"{L[hob['wall']]} is " + ("a solid wall" if not hw.windows else "the best free run"))
            if hw.doors:
                bits.append("kept 600+ mm from the door")
            if sink and sink["wall"] == hob["wall"]:
                bits.append("600+ mm of counter from the sink")
            bits.append(f"centred at {hob_c} mm with counter both sides")
        hob_why = ", ".join(bits) + "."
        if "hob" in rules:
            hob_why += f" Rule R{rules['hob']['rule']}: {rules['hob']['width']} mm {rules['hob'].get('combination') or ''} unit."
        else:
            hob_why += " Drawers keep pans and pressure cookers flat."

        # LPG
        cyl = (self.role("cylinder") or [None])[0]
        if p.include_cylinder is False:
            lpg_why = "Piped gas or induction: no cylinder unit is needed."
        elif p.fuel == "lpg" or p.include_cylinder:
            lpg_why = "The customer uses LPG: the cylinder unit sits beside the hob."
        else:
            lpg_why = "Fuel not stated: LPG is assumed (rule R11), so a cylinder unit goes beside the hob."
        if not cyl and p.include_cylinder is not False:
            lpg_why += " There is no room for it beside the hob in this kitchen."

        # tall
        if not p.include_tall:
            tall_why = "The customer does not want tall units."
        elif not talls:
            tall_why = "No free run end for a tall unit without blocking the window or the hob."
        else:
            need = {"microwave": "a built-in microwave", "oven": "a built-in oven",
                    "oven_microwave": "a built-in oven and microwave"}.get(p.built_in)
            where = f"at the end of the {L[talls[0]['wall']]} run"
            tall_why = (f"The customer needs {need}: the tower stands {where}." if need
                        else f"Tall storage {where}, away from the hob and not in front of the window.")
            if len(talls) > 1:
                tall_why += " A second tall unit as needed."

        # purifier / display / open
        pur, disp, opn = self.role("purifier"), self.role("display"), self.role("open")
        purifier_why = ("The customer does not want one." if p.water_purifier == "no" else
                        "The customer needs RO water: Aquaguard unit beside the sink window." if pur and p.water_purifier == "yes" else
                        "Standard in Indian kitchens: RO unit on the wall beside the sink." if pur else
                        "No wall space near the sink for a purifier unit.")
        display_why = ("Customer wants a glass display at a run end." if disp else
                       "Customer does not want one." if p.display_unit == "no" else "Not needed: plain wall units give more storage.")
        open_why = "Open shelf at a run end as the customer wants." if opn else "Not requested; closed units keep dust out."

        # storage + features + wall units
        stated = base_storage_need(p)[0]
        storage = stated if stated != "auto" else (prop.get("base_storage") or "mixed")
        allowed_st = allowed_storage(p)
        storage = storage if storage in allowed_st else allowed_st[0]
        storage_why = ("The customer's choice." if stated != "auto" else
                       "Heavy cooking or a big family: drawers keep many vessels within reach." if storage == "drawers" else
                       "Light cooking: shutter storage is enough." if storage == "shutters" else
                       "Drawer + shutter units: a cutlery drawer on top, shelf space below.")
        feats = []
        for wid, aids in (prop.get("base_prefs") or {}).items():
            if wid not in walls:
                continue
            units = [c for c in (self.code("base", a) for a in aids) if c != "NONE"][:3]
            if units:
                feats.append({"wall": L[wid], "units": units})
        features_why = ("Drawers beside the hob for pans; grain trolley on the sink wall, away from the hob heat."
                        if feats else "Plain base units in the chosen style fill the runs.")
        sizes = []
        for wid in walls:
            widths = Counter(q["aid"] for q in self.places if q["wall"] == wid and q["role"] == "wall")
            units = list(dict.fromkeys(self.code("wall", a) for a, _ in widths.most_common(2)))
            units = [u for u in units if u != "NONE"][:2]
            if units:
                sizes.append({"wall": L[wid], "units": units})
        wall_cabinets_why = "Wide wall units on long runs mean fewer doors; narrower ones fill short gaps between the window and corners."

        # doors, finishes, lofts
        wall_style = rep.get("heights", {}).get("wall_style") or "shutters"
        wall_style = wall_style if wall_style in allowed_wall_styles(p) else allowed_wall_styles(p)[0]
        wall_style_why = "The customer's choice." if p.wall_style != "auto" else "Hinged shutters: the standard, lowest-cost door."
        budget = None if p.budget == "auto" else p.budget
        palette, _, _ = finishes.resolve({"base": p.base_finish, "wall": p.wall_finish}, {}, budget)

        def fin(run):
            allowed = finish_codes(run, p)
            c = finishes.code(palette.get(run)) if palette.get(run) else None
            return c if c in allowed else allowed[0]

        finish_why = cut(finishes.default(budget)[1] or "The standard MagickHome palette for this budget.", 180)
        has_lofts = bool(self.lay.get("loft_walls"))
        lofts = "yes" if has_lofts else "no"
        allowed_l = allowed_lofts(p, lofts_possible(g))
        lofts = lofts if lofts in allowed_l else allowed_l[0]
        ceiling = round(g.ceiling_height_m * 1000) if g.ceiling_height_m else None
        loft_why = ("The customer wants lofts." if p.include_loft else
                    f"The {ceiling} mm ceiling leaves room for lofts: storage for big vessels and seasonal items." if has_lofts else
                    "The customer does not want lofts." if p.include_loft is False else
                    f"The {ceiling} mm ceiling is too low for lofts.")

        return {
            "room_reading": room_reading, "design_plan": design_plan,
            "layout_why": cut(why_layout, 180), "cabinet_walls": [L[w] for w in walls],
            "sink_why": cut(sink_why, 180), "sink_wall": L[sink["wall"]], "sink": self.code("sink", sink["aid"]),
            "hob_why": cut(hob_why, 180), "hob_wall": L[hob["wall"]], "hob": self.code("hob", hob["aid"]),
            "hob_center_mm": int(hob_c),
            "spice": self.code("spice", (self.role("spice") or [{}])[0].get("aid")),
            "lpg_why": cut(lpg_why, 180), "lpg_cylinder": self.code("cylinder", cyl["aid"] if cyl else None),
            "tall_why": cut(tall_why, 180),
            "tall_units": [{"wall": L[t["wall"]], "unit": self.code("tall", t["aid"])} for t in talls] if p.include_tall else [],
            "purifier_why": cut(purifier_why, 180), "water_purifier": self.code("purifier", pur[0]["aid"] if pur else None),
            "display_why": cut(display_why, 180), "display_unit": self.code("display", disp[0]["aid"] if disp else None),
            "open_why": cut(open_why, 180), "open_unit": self.code("open", opn[0]["aid"] if opn else None),
            "storage_why": cut(storage_why, 180), "base_storage": storage,
            "features_why": cut(features_why, 180), "base_feature_units": feats,
            "wall_cabinets_why": cut(wall_cabinets_why, 180), "wall_cabinet_sizes": sizes,
            "wall_style_why": cut(wall_style_why, 180), "wall_style": wall_style,
            "finish_why": finish_why, "base_finish": fin("base"), "wall_finish": fin("wall"),
            "loft_why": cut(loft_why, 180), "lofts": lofts,
        }


def replay(doc: dict, prefs: KitchenPreferences, answer: dict):
    """Run ``answer`` through the real pipeline as if the model had given it; return (report, Model-validated)."""
    g = parse_scene(strip_items(doc))
    _, groups = catalog_and_groups(prefs.door_style, prefs.handle_type)
    _, Model, _ = build_messages(g, groups, prefs, "v2", knowledge=False)
    d = Model.model_validate(answer)
    menu, labels = build_menu(groups), wall_labels(g)
    proposal = _design_to_proposal_v2(d, menu, labels, prefs)
    proposal.source = "gemma"
    original = app.design_kitchen
    app.design_kitchen = lambda *_a, **_k: (proposal, {"mode": "gemma", "model": "rules-teacher", "raw": answer})
    try:
        _, report = app.generate_kitchen(doc, prefs.model_copy(update={"ai_mode": "ollama"}))
    finally:
        app.design_kitchen = original
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    done = {r["id"] for r in read_jsonl(RULES)} | {r["id"] for r in read_jsonl(REJECTED)}
    kept = rej = 0
    reasons = Counter()
    t0 = time.time()
    for i in range(args.n):
        rid = f"rules-s{args.seed}-{i:05d}"
        if rid in done:
            continue
        rng = random.Random(args.seed * 1_000_003 + i)
        room = random_room(rng, rid)
        prefs = random_prefs(rng, ai_mode="fallback")
        doc = room.build()
        try:
            _, rep = app.generate_kitchen(doc, prefs)
            answer = Answer(doc, prefs, rep).build()
            prefs_ai = prefs.model_copy(update={"ai_mode": "auto"})
            rep2 = replay(doc, prefs_ai, answer)
            verdict = judge(doc, prefs_ai, rep2)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            append_jsonl(REJECTED, {"id": rid, "reason": reason, "room": room.title})
            reasons[type(exc).__name__] += 1
            rej += 1
            continue
        if verdict["ok"]:
            append_jsonl(RULES, make_row(rid, "rules", doc, prefs_ai, answer, "rules-teacher", verdict["score"],
                                         {"room": room.title}))
            kept += 1
        else:
            why = (f"score {verdict['score']} failed {verdict['failed']} fixes {verdict['corrections'][:2]} "
                   f"issues {len(verdict['validation_issues'])}")
            append_jsonl(REJECTED, {"id": rid, "reason": why, "room": room.title})
            for f in verdict["failed"]:
                reasons[f] += 1
            if verdict["corrections"]:
                reasons["corrections"] += 1
            rej += 1
        if (kept + rej) % 25 == 0:
            print(f"  {kept + rej}/{args.n}: kept {kept}, rejected {rej} ({time.time() - t0:.0f} s)", flush=True)
    print(f"\nKept {kept}, rejected {rej} in {time.time() - t0:.0f} s. Rejections: {dict(reasons.most_common(8))}")
    print(f"{RULES} has {len(read_jsonl(RULES))} examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
