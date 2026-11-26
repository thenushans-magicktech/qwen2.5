"""Training dataset made by Qwen 2.5 14B on Colab, checked by the MagickHome engine.

For every example:

1. a floor plan  - a real room from the database export (training/floorplans/db, items removed), a
                   variation of one (stretched / shrunk, doors and windows moved), or a generated
                   room (make_dataset.sample_room); door style Slab (Plain) or Shaker 1;
2. a customer    - needs, budget, fuel, brief and chosen walls (make_dataset.sample_prefs);
3. the prompt    - the production prompt in rule-sheet mode: the 81 rules of the rule sheet, the
                   cabinet codes (real AIDs of that door style), the room facts and the customer's
                   needs, plus knowledge/designer_advice.md as ADVICE (never as rules);
4. Qwen designs  - --best-of designs per room (different seeds), streamed through the Colab tunnel;
5. the check     - the real pipeline, twice: what the rule book / engine changed is taken into the
                   answer (with designer repairs when the plan cannot be built), and the design is kept
                   only when the second run needs NO correction and the 3D scene verifies;
6. reasons       - Qwen's own *_why texts (rewritten only where the engine changed the decision) and
                   REASON CODES for every decision, from the rule sheet (RULE_Rn), the customer
                   (CUSTOMER_*), room facts and DESIGNER_CHOICE - nothing else;
7. saved         - structured example + plan file, progress saved after every example (resume).

    # Colab running the notebook (it prints this command with the URL and the password):
    python training/qwen_dataset.py --url https://xxxx.trycloudflare.com --token XXXX --n 1000

    python training/qwen_dataset.py --designer policy --n 10 --out training/try   # no Colab: test run
    python training/qwen_dataset.py --designer mock --n 3 --out training/try      # tests tunnel + proxy

Output in --out (default training/dataset_qwen): scratch/train.jsonl, scratch/val.jsonl, plans/,
scratch/features.csv, scratch/labels.csv, knowledge/, progress.json, run.log.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import make_dataset as M  # noqa: E402  (offline catalogue / textures, the engine check, repairs)
import rule_check as RC  # noqa: E402  (every rule of the rule sheet, checked on the finished design)

from scene_parser import parse_scene  # noqa: E402
from catalog_loader import canon  # noqa: E402

DOOR_STYLES = ("Slab (Plain)", "Shaker 1")
ADVICE_PATH = ROOT / "knowledge" / "designer_advice.md"
ADVICE_HEADER = ("\n\nDESIGNER ADVICE - good practice from an experienced designer, NOT rules. The rule book "
                 "above always wins; follow it and the customer's stated needs first.\n")


class ColabDown(RuntimeError):
    """Colab / the tunnel / Ollama cannot be reached: stop, keep progress, ask for a new link."""


M.FATAL = (ColabDown,)

# ---------------------------------------------------------------------------
# floor plans
# ---------------------------------------------------------------------------


def load_db_rooms(folder: Path) -> list[tuple[str, dict]]:
    rooms = []
    for fp in sorted(folder.glob("*.json")):
        if fp.name == "index.json":
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            parse_scene(doc)
            rooms.append((fp.stem, doc))
        except Exception:  # noqa: BLE001
            continue
    return rooms


def _scale_pt(p, cx, cz, sx, sz):
    return [cx + (p[0] - cx) * sx, cz + (p[1] - cz) * sz] + list(p[2:])


def vary_plan(doc: dict, rng: random.Random) -> Optional[dict]:
    """A variation of a real room: stretched or shrunk on each axis, doors and windows moved along
    their walls (never overlapping), sometimes another ceiling height. None if it is not valid."""
    for _ in range(12):
        d = copy.deepcopy(doc)
        nodes = d["scene"]["nodes"]
        walls = {k: n for k, n in nodes.items() if n.get("type") == "wall" and "start" in n and "end" in n}
        if not walls:
            return None
        xs = [p[0] for n in walls.values() for p in (n["start"], n["end"])]
        zs = [p[1] for n in walls.values() for p in (n["start"], n["end"])]
        cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        sx, sz = rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.25)
        old_len = {k: ((n["end"][0] - n["start"][0]) ** 2 + (n["end"][1] - n["start"][1]) ** 2) ** 0.5 for k, n in walls.items()}
        for n in walls.values():
            n["start"], n["end"] = _scale_pt(n["start"], cx, cz, sx, sz), _scale_pt(n["end"], cx, cz, sx, sz)
        for n in nodes.values():
            if n.get("type") in ("slab", "ceiling") and isinstance(n.get("polygon"), list):
                n["polygon"] = [_scale_pt(p, cx, cz, sx, sz) for p in n["polygon"]]
        if rng.random() < 0.4:
            h = rng.choice([2.4, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2])
            for n in nodes.values():
                if n.get("type") in ("wall", "ceiling") and "height" in n:
                    n["height"] = h
        ok = True
        for wid, w in walls.items():
            new_len = ((w["end"][0] - w["start"][0]) ** 2 + (w["end"][1] - w["start"][1]) ** 2) ** 0.5
            ratio = new_len / old_len[wid] if old_len[wid] else 1
            ops = [nodes[c] for c in w.get("children", []) if c in nodes and nodes[c].get("type") in ("door", "window")]
            spans = []
            for o in ops:
                width = float(o.get("width", 0.9))
                pos = float(o["position"][0]) * ratio + rng.uniform(-0.35, 0.35)
                lo, hi = width / 2 + 0.1, new_len - width / 2 - 0.1
                if hi < lo:
                    ok = False
                    break
                pos = min(max(pos, lo), hi)
                if any(pos - width / 2 < b + 0.15 and pos + width / 2 > a - 0.15 for a, b in spans):
                    pos = float(o["position"][0]) * ratio  # keep it where it was if moving it collides
                o["position"][0] = round(pos, 3)
                spans.append((pos - width / 2, pos + width / 2))
            if not ok:
                break
        if not ok:
            continue
        try:
            g = parse_scene(d)
        except Exception:  # noqa: BLE001
            continue
        if any(M.free(g, w) >= M.MIN_WALL for w in g.wall_order):
            return d
    return None


def pick_plan(i: int, rng: random.Random, rooms: list) -> tuple[str, str, dict, Any]:
    """(source, key, floor plan, Room spec or None) for example i."""
    r = rng.random()
    if rooms and r < 0.35:
        name, doc = rooms[i % len(rooms)]
        return "db", name, copy.deepcopy(doc), None
    if rooms and r < 0.65:
        name, doc = rooms[rng.randrange(len(rooms))]
        v = vary_plan(doc, rng)
        if v is not None:
            return "variant", f"{name}~v{i}", v, None
    room = M.sample_room(rng, i)
    return "generated", room.key, room.build(), room


# ---------------------------------------------------------------------------
# Qwen on Colab
# ---------------------------------------------------------------------------


def qwen_client(url: str, token: str, timeout: float):
    import ollama

    return ollama.Client(host=url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)


def check_colab(client, model: str) -> None:
    try:
        names = {m.model for m in client.list().models}
    except Exception as exc:  # noqa: BLE001
        raise ColabDown(f"Colab not reachable: {exc}") from exc
    if model not in names:
        raise ColabDown(f"{model} is not on Colab (it has: {', '.join(sorted(names)) or 'nothing'}) - run the notebook's pull cell")


def make_qwen_designer(client, model: str, advice: str, seed: int, num_ctx: int):
    """The designer hook for make_dataset: the production messages + advice -> Qwen (streamed) ->
    a schema-valid answer. Qwen's own answer is kept in CTX['qwen_raw']."""
    from pydantic import ValidationError

    def designer(geometry, prefs, Model, messages):
        msgs = [dict(messages[0], content=messages[0]["content"] + ADVICE_HEADER + advice), dict(messages[1])]
        schema = Model.model_json_schema()
        opts = {"temperature": 0.7, "top_p": 0.9, "seed": seed, "num_ctx": num_ctx}
        t0 = time.time()
        for attempt in range(2):
            text = ""
            try:
                for part in client.chat(model=model, messages=msgs, format=schema, options=opts, stream=True,
                                        keep_alive="2h"):
                    msg = part.message if hasattr(part, "message") else part["message"]
                    text += (msg.content if hasattr(msg, "content") else msg["content"]) or ""
            except Exception as exc:  # noqa: BLE001
                if "ResponseError" in type(exc).__name__ and getattr(exc, "status_code", 500) < 500 \
                        and getattr(exc, "status_code", 0) != 401:
                    raise  # a bad request: this example fails, the run goes on
                raise ColabDown(f"Colab stopped answering: {type(exc).__name__}: {str(exc)[:160]}") from exc
            try:
                d = Model.model_validate_json(text)
            except ValidationError as exc:
                msgs += [{"role": "assistant", "content": text or "{}"},
                         {"role": "user", "content": f"Invalid JSON for the schema: {exc.errors()[:3]}. Reply again."}]
                opts["seed"] = seed + 1000 * (attempt + 1)
                continue
            M.CTX["qwen_raw"] = json.loads(d.model_dump_json())
            M.CTX["qwen_seconds"] = round(time.time() - t0, 1)
            return M.CTX["qwen_raw"]
        raise ValueError("Qwen's answer did not fit the answer form twice")

    return designer


def make_policy_designer():
    """Offline stand-in for Qwen (test runs without Colab): the rule-free design policy."""
    def designer(geometry, prefs, Model, messages):
        ans = M.design(geometry, prefs, Model, M.CTX["rng"])
        for k in ans:
            if k.endswith("_why") or k in ("room_reading", "design_plan"):
                ans[k] = "Designer's choice for this room and customer."
        M.CTX["qwen_raw"] = json.loads(json.dumps(ans))
        M.CTX["qwen_seconds"] = 0.0
        return ans
    return designer


# --- a local stand-in Ollama (mock mode): tests streaming through the real password proxy ----------

MOCK = {"next": None}


class _MockOllama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(json.dumps({"models": [{"name": "mock", "model": "mock", "modified_at": "2026-01-01T00:00:00Z",
                                           "size": 1, "digest": "x", "details": {}}]}).encode())

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        text = json.dumps(MOCK["next"])
        parts = [text[i:i + 200] for i in range(0, len(text), 200)]
        lines = [json.dumps({"model": "mock", "created_at": "2026-01-01T00:00:00Z",
                             "message": {"role": "assistant", "content": p}, "done": False}) for p in parts]
        lines.append(json.dumps({"model": "mock", "created_at": "2026-01-01T00:00:00Z",
                                 "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for ln in lines:
            b = (ln + "\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"0\r\n\r\n")


def start_mock() -> tuple[str, str, Any]:
    """Mock Ollama + the real colab/ollama_auth_proxy.py in front of it. Returns (url, token, procs)."""
    import secrets
    import socket

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    token = secrets.token_urlsafe(16)
    proc = subprocess.Popen([sys.executable, str(ROOT / "colab" / "ollama_auth_proxy.py"), str(port)],
                            env=dict(os.environ, OLLAMA_API_KEY=token, OLLAMA_UPSTREAM_PORT=str(srv.server_address[1])),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    return f"http://127.0.0.1:{port}", token, (srv, proc)


def make_mock_designer(client, seed: int):
    qwen = make_qwen_designer(client, "mock", "", seed, 8192)

    def designer(geometry, prefs, Model, messages):
        ans = M.design(geometry, prefs, Model, M.CTX["rng"])
        for k in ans:
            if k.endswith("_why") or k in ("room_reading", "design_plan"):
                ans[k] = "Mock designer's reason."
        MOCK["next"] = ans
        return qwen(geometry, prefs, Model, messages)  # through the proxy, streamed, schema-checked
    return designer


# ---------------------------------------------------------------------------
# reasons: codes from the rule sheet, the customer, the room - and the designer's own choice
# ---------------------------------------------------------------------------

GROUPS = ("layout", "sink", "hob", "lpg", "tall", "purifier", "display", "open", "storage", "features",
          "wall_cabinets", "wall_style", "finish", "lofts")
WHY_KEY = {"layout": "layout_why", "sink": "sink_why", "hob": "hob_why", "lpg": "lpg_why", "tall": "tall_why",
           "purifier": "purifier_why", "display": "display_why", "open": "open_why", "storage": "storage_why",
           "features": "features_why", "wall_cabinets": "wall_cabinets_why", "wall_style": "wall_style_why",
           "finish": "finish_why", "lofts": "loft_why"}
FIELD_GROUP = {"cabinet_walls": "layout", "sink_wall": "sink", "sink": "sink", "hob_wall": "hob", "hob": "hob",
               "spice": "hob", "lpg_cylinder": "lpg", "tall_units": "tall", "water_purifier": "purifier",
               "display_unit": "display", "open_unit": "open", "base_storage": "storage",
               "base_feature_units": "features", "wall_cabinet_sizes": "wall_cabinets", "wall_style": "wall_style",
               "base_finish": "finish", "wall_finish": "finish", "lofts": "lofts"}


def role_group(role: str) -> str:
    r = (role or "").lower()
    for key, g in (("sink", "sink"), ("hob", "hob"), ("spice", "hob"), ("cylinder", "lpg"), ("loft", "lofts"),
                   ("tall", "tall"), ("purifier", "purifier"), ("display", "display"), ("open", "open"),
                   ("wall", "wall_cabinets"), ("base", "storage"), ("corner", "layout")):
        if key in r:
            return g
    return "layout"


def reason_codes(ans: dict, rep: dict, prefs, changed: set) -> dict[str, list[str]]:
    codes: dict[str, list[str]] = {g: [] for g in GROUPS}

    def add(g, c):
        if c not in codes[g]:
            codes[g].append(c)

    for a in rep["rulebook"].get("applied", []):
        if a.get("rule") not in (None, ""):
            add(role_group(a.get("role")), f"RULE_R{a['rule']}")
    for line in rep["layout"]["decisions"]:
        for n in re.findall(r"\bR(\d{1,2})\b", line):
            low = line.lower()
            g = ("lofts" if "loft" in low or "height" in low else "hob" if "hob" in low or "landing" in low else
                 "sink" if "sink" in low else "storage" if "base" in low else "layout")
            add(g, f"RULE_R{n}")
    for n in re.findall(r"\bR(\d{1,2})\b", str(rep["heights"].get("why", ""))):
        add("lofts", f"RULE_R{n}")
    p = prefs
    if p.layout != "auto":
        add("layout", "CUSTOMER_LAYOUT")
    if p.cabinet_walls:
        add("layout", "CUSTOMER_WALLS")
    if p.primary_wall:
        add("hob", "CUSTOMER_COOKING_WALL")
    if p.secondary_wall:
        add("sink", "CUSTOMER_SINK_WALL")
    if p.fuel != "auto":
        add("lpg", f"CUSTOMER_FUEL_{p.fuel.upper()}")
    for field, g in (("built_in", "tall"), ("crockery_unit", "tall"), ("water_purifier", "purifier"),
                     ("display_unit", "display"), ("open_unit", "open"), ("base_storage", "storage"),
                     ("wall_style", "wall_style")):
        if getattr(p, field) != "auto":
            add(g, f"CUSTOMER_{field.upper()}")
    if not p.include_tall:
        add("tall", "CUSTOMER_NO_TALL")
    if p.include_loft is not None:
        add("lofts", "CUSTOMER_LOFTS")
    if p.budget != "auto":
        add("storage", f"CUSTOMER_BUDGET_{p.budget.upper()}")
        add("finish", f"CUSTOMER_BUDGET_{p.budget.upper()}")
    brief = (p.style_notes or "").lower()
    for words, g in ((("bake", "microwave", "crockery"), "tall"), (("drawer",), "storage"), (("vessel",), "lofts"),
                     (("water",), "purifier"), (("crockery",), "display"), (("minimal",), "wall_style"),
                     (("wood", "white", "grey", "dark", "blue", "colour"), "finish")):
        if any(w in brief for w in words):
            add(g, "CUSTOMER_BRIEF")
    h = rep["heights"]
    add("lofts", "ROOM_CEILING_ALLOWS_LOFTS" if h["ceiling_mm"] + 1 >= M.config.LOFT_TOP_M * 1000 else "ROOM_CEILING_TOO_LOW")
    for g in changed:
        add(g, "ENGINE_ADJUSTED")
    for g in GROUPS:
        if not [c for c in codes[g] if not c.startswith("ROOM_")]:
            add(g, "DESIGNER_CHOICE")
    return codes


def fix_reasons(ans: dict, qwen: dict, rep: dict) -> set:
    """Where the engine changed Qwen's decision, Qwen's reason no longer fits: rewrite it from the
    rule that set it, or say the engine fitted it to the room. Returns the changed groups."""
    changed = set()
    for field, g in FIELD_GROUP.items():
        if json.dumps(ans.get(field), sort_keys=True) != json.dumps((qwen or {}).get(field), sort_keys=True):
            changed.add(g)
    by_group = collections.defaultdict(list)
    for a in rep["rulebook"].get("applied", []):
        by_group[role_group(a.get("role"))].append(a)
    for g in changed:
        key = WHY_KEY[g]
        rules = by_group.get(g)
        if rules:
            r = rules[0]
            text = f"Set by the rule book: R{r['rule']} - {r.get('text', '')}"
        else:
            text = "Adjusted by the placement engine to fit this room."
        ans[key] = M.clip(text, 180)
    return changed


# ---------------------------------------------------------------------------
# one example
# ---------------------------------------------------------------------------


def bbox_mm(geo) -> tuple[int, int]:
    """Outer size of the room (its wall bounding box), mm."""
    xs = [p for w in geo.walls.values() for p in (w.start[0], w.end[0])]
    zs = [p for w in geo.walls.values() for p in (w.start[1], w.end[1])]
    return round((max(xs) - min(xs)) * 1000), round((max(zs) - min(zs)) * 1000)


def evaluate(doc, prefs, ans) -> dict:
    """Full pipeline + every check. {"ok": True, out, rep, rules} or {"ok": False, kind, why, rep, ...}."""
    M.CTX["answer"] = ans
    try:
        out, rep = M.run(doc, prefs)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "kind": "error", "why": f"pass 2 {type(exc).__name__}"}
    fixes = M.corrections(rep, ans)
    if fixes:
        return {"ok": False, "kind": "corrected", "why": "engine corrected: " + re.sub(r"[\d.]+", "#", fixes[0])[:60], "rep": rep}
    shift = [abs((p["start_mm"] + p["end_mm"]) / 2 - ans[k]) for p in M.placements(rep) for r, k in
             (("sink", "sink_center_mm"), ("hob", "hob_center_mm")) if p["role"] == r]
    if any(s > 60 for s in shift):
        return {"ok": False, "kind": "corrected", "why": "position shifted", "rep": rep}
    loft_y = (M.config.POS_Y_M["wall"] * 1000 + rep["heights"]["wall_mm"] + M.config.LOFT_BOTTOM_MM) / 1000
    errs = [e for e in M.verify_ai_scene(out, expect_door=None)
            if not (": loft y=" in e and abs(float(e.split("loft y=")[1].split()[0]) - loft_y) < 0.005)]
    if errs:
        return {"ok": False, "kind": "scene", "why": "scene check: " + errs[0][:50], "rep": rep}
    if not M.connected(parse_scene(doc), list(rep["layout"]["cabinet_walls"])):
        return {"ok": False, "kind": "scene", "why": "walls not connected", "rep": rep}
    rules = RC.check_design(parse_scene(doc), M.CTX["prefs"], rep, out)
    fails = RC.summary(rules)["design_fail"]
    if fails:
        return {"ok": False, "kind": "rule", "rule": fails[0], "why": f"breaks R{fails[0]}", "rep": rep, "rules": rules}
    return {"ok": True, "out": out, "rep": rep, "rules": rules}


def readopt(doc, geo, prefs, ans) -> Optional[dict]:
    """Run an adjusted answer and take in what the engine places (positions, widths) - no Qwen call."""
    try:
        M.CTX["answer"] = json.loads(M.CTX["Model"].model_validate(ans).model_dump_json())
        _, rep1 = M.run(doc, prefs, scene=False)
    except Exception:  # noqa: BLE001
        return None
    adopted, bad = M.adopt(json.loads(json.dumps(M.CTX["answer"])), rep1, geo)
    if bad:
        return None
    try:
        return json.loads(M.CTX["Model"].model_validate(adopted).model_dump_json())
    except Exception:  # noqa: BLE001
        return None


def _rule_role(n: int) -> Optional[str]:
    for w in M.rulebook.load().widths:
        if w.rule == n:
            return {"base hob unit": "hob", "base sink unit": "sink", "base spice pull out": "spice",
                    "base cylinder unit": "cylinder"}.get(canon(w.trigger))
    return {26: "hob", 28: "hob", 29: "hob", 55: "hob", 56: "hob", 57: "hob", 27: "sink"}.get(n)


def rule_repair(ans: dict, verdict: dict, geo, tried: set) -> Optional[dict]:
    """The next adjustment that can make a failed design follow the rules, or None.
    Adjustments a designer would make: take in what the engine built; recentre the hob / sink in its run
    (landings, R26 / R27); move tall units off the hob / sink wall so the unit gets the rule's width; no
    spice pull-out when it crowds the hob; the hob / sink on another wall (only when the customer did not
    choose that wall); no lofts when the loft run cannot follow the loft rules (only when lofts are optional)."""
    E = M.schema_enums(M.CTX["Model"])
    labels = wall_labels_of(geo)
    a = json.loads(json.dumps(ans))
    if verdict["kind"] == "corrected" and "took-engine" not in tried:
        tried.add("took-engine")
        adopted, bad = M.adopt(a, verdict["rep"], geo)
        return None if bad else adopted
    if verdict["kind"] != "rule":
        return None
    n = verdict["rule"]
    role = _rule_role(n)

    def mid(lab):
        s, e = max(M._free_segments(geo, labels[lab]), key=lambda t: t[1] - t[0])
        return int(round((s + e) / 2 / 10) * 10)

    def talls_to(target):
        """All tall units onto ``target`` (a cabinet wall) - away from the hob / sink they crowd."""
        if not a["tall_units"] or all(t["wall"] == target for t in a["tall_units"]):
            return False
        a["tall_units"] = [dict(t, wall=target) for t in a["tall_units"]]
        return True

    def one_tall():
        if len(a["tall_units"]) < 2:
            return False
        a["tall_units"] = a["tall_units"][:1]
        return True

    # walls for the tall units: first walls with neither the hob nor the sink, then the rest
    quiet = sorted(a["cabinet_walls"], key=lambda w: (w in (a["hob_wall"], a["sink_wall"]), -M.free(geo, labels[w])))
    tall_steps = [(f"talls-to-{w}", lambda w=w: talls_to(w)) for w in quiet] + [("one-tall", one_tall)]

    if role in ("hob", "spice", "cylinder"):
        steps = ([("hob-mid", lambda: a.update(hob_center_mm=mid(a["hob_wall"])) or True)] + tall_steps
                 + [("no-spice", lambda: "NONE" in E["spice"] and a["spice"] != "NONE" and (a.update(spice="NONE") or True)),
                    ("hob-wall", lambda: _move(a, "hob", E, geo, labels))])
    elif role == "sink":
        steps = ([("sink-mid", lambda: a.update(sink_center_mm=mid(a["sink_wall"])) or True)] + tall_steps
                 + [("sink-wall", lambda: _move(a, "sink", E, geo, labels))])
    elif 47 <= n <= 54 or 70 <= n <= 81:
        steps = [("no-lofts", lambda: "no" in E["lofts"] and a["lofts"] != "no" and (a.update(lofts="no") or True))]
    else:
        steps = []  # budget fills, wall-run breaks...: made by the engine itself, nothing the designer can change
    for key, act in steps:
        if (key, n) in tried:
            continue
        tried.add((key, n))
        if act():
            return a
    return None


def _move(a: dict, unit: str, E: dict, geo, labels) -> bool:
    """Put the hob / sink on another of the cabinet walls (only when the customer did not fix its wall)."""
    key = f"{unit}_wall"
    if len(E[key]) <= 1:
        return False
    other = [w for w in a["cabinet_walls"] if w != a[key] and w in E[key]]
    if not other:
        return False
    best = max(other, key=lambda w: M.free(geo, labels[w]))
    a[key] = best
    s, e = max(M._free_segments(geo, labels[best]), key=lambda t: t[1] - t[0])
    a[f"{unit}_center_mm"] = int(round((s + e) / 2 / 10) * 10)
    return True


def wall_labels_of(geo) -> dict:
    return M.wall_labels(geo)


def candidate(doc, geo, prefs, designer, rng, stats) -> Optional[dict]:
    M.CTX.clear()
    M.CTX["rng"] = rng
    M.CTX["designer"] = designer
    ans, why = M.first_pass(doc, geo, prefs, stats)
    if ans is None and prefs.include_tall and any(w in why for w in ("tall", "tower", "hob")):
        prefs = prefs.model_copy(update={"include_tall": False,
                                         "built_in": "none" if prefs.built_in != "auto" else "auto",
                                         "crockery_unit": "no" if prefs.crockery_unit == "yes" else prefs.crockery_unit})
        M.CTX.clear()
        M.CTX["rng"] = rng
        M.CTX["designer"] = designer
        ans, why = M.first_pass(doc, geo, prefs, stats)
        stats["no-tall variant"] += 1
    if ans is None:
        stats["reject: " + " ".join(why.split(" ")[:3])] += 1
        return None
    qwen = M.CTX.get("qwen_raw")
    # check; when a check fails, ADJUST the design to the rules (seconds, no new Qwen call) and check again
    tried: set = set()
    for _ in range(12):
        verdict = evaluate(doc, prefs, ans)
        if verdict["ok"]:
            break
        fixed = rule_repair(ans, verdict, geo, tried)
        if fixed is None:
            stats["reject: " + verdict["why"]] += 1
            return None
        stats["adjusted to the rules"] += 1
        ans = readopt(doc, geo, prefs, fixed)
        if ans is None:
            stats["reject: adjustment could not be built"] += 1
            return None
    else:
        stats["reject: adjustments exhausted"] += 1
        return None
    out, rep, rules = verdict["out"], verdict["rep"], verdict["rules"]
    for n in RC.summary(rules)["build_fail"]:
        stats[f"engine build issue R{n} (reported, not rejected)"] += 1
    changes = sum(json.dumps(ans.get(f), sort_keys=True) != json.dumps((qwen or {}).get(f), sort_keys=True)
                  for f in FIELD_GROUP)
    return {"ans": ans, "qwen": qwen, "rep": rep, "prefs": M.CTX["prefs"], "messages": M.CTX["messages"],
            "menu": M.CTX["menu"], "Model": M.CTX["Model"], "changes": changes, "rules": rules,
            "warnings": len(rep["layout"]["warnings"]), "seconds": M.CTX.get("qwen_seconds")}


SHAPES = (("single", 1), ("L", 2), ("U", 3))
LOG = print  # main() points this at the run log


def balance_layout(i: int, rng: random.Random, geo, prefs):
    """Even mix of kitchen shapes: example i targets straight / L / U in turn. 70% of customers ask for
    that shape (when the room can carry it); 30% leave the shape to the designer."""
    target, need = SHAPES[i % len(SHAPES)]
    usable = [w for w in geo.wall_order if M.free(geo, w) >= M.MIN_WALL]
    feasible = any(M.connected(geo, list(ws)) for ws in itertools.combinations(usable, need))
    if not feasible or rng.random() >= 0.7:
        return prefs
    upd: dict[str, Any] = {"layout": target}
    if prefs.cabinet_walls and len(prefs.cabinet_walls) != need:
        upd["cabinet_walls"] = None
    if target == "single" and prefs.secondary_wall and prefs.secondary_wall != prefs.primary_wall:
        upd["secondary_wall"] = None
    return prefs.model_copy(update=upd)


def make_example(i: int, args, rooms, designer_for, stats) -> Optional[dict]:
    rng = random.Random(args.seed * 1_000_003 + i)
    for attempt in range(3):
        source, key, doc, room = pick_plan(i, rng, rooms)
        try:
            geo = parse_scene(doc)
        except Exception:  # noqa: BLE001
            stats["reject: plan unreadable"] += 1
            continue
        if not any(M.free(geo, w) >= M.MIN_WALL for w in geo.wall_order):
            stats["reject: no usable wall"] += 1
            continue
        prefs = M.sample_prefs(rng, geo).model_copy(update={"door_style": rng.choice(DOOR_STYLES)})
        prefs = balance_layout(i, rng, geo, prefs)
        LOG(f"example {i}: {source} room ({len(geo.walls)} walls), {prefs.door_style}, shape {prefs.layout} - designing")
        cands = []
        for k in range(args.best_of + 1):  # one more design only when none could be adjusted to the rules
            if cands and k >= args.best_of:
                break
            before = dict(stats)
            t0 = time.time()
            c = candidate(doc, geo, prefs, designer_for(args.seed * 7919 + i * 31 + k), rng, stats)
            if c:
                cands.append(c)
                LOG(f"  design {k + 1}: passed every rule ({time.time() - t0:.0f} s)")
            else:
                why = next((key for key in stats if key.startswith("reject") and stats[key] > before.get(key, 0)), "rejected")
                LOG(f"  design {k + 1}: {why.replace('reject: ', 'rejected - ')} ({time.time() - t0:.0f} s)")
        if not cands:
            continue
        best = min(cands, key=lambda c: (c["changes"], c["warnings"]))
        ans, rep, prefs2 = best["ans"], best["rep"], best["prefs"]
        M.CTX.update(Model=best["Model"], menu=best["menu"], prefs=prefs2)
        changed = fix_reasons(ans, best["qwen"], rep)
        ans = json.loads(best["Model"].model_validate(ans).model_dump_json())
        geo2 = parse_scene(doc)
        catalog = M.catalog_loader.load_filtered_catalog(prefs2.door_style, prefs2.handle_type)
        rep["proposal"]["notes"] = f"{ans['room_reading']} {ans['design_plan']}"
        sc = M.score_design(geo2, catalog, rep, prefs2)
        ex_id = f"q{i:05d}"
        spec = room or SimpleNamespace(width_mm=None, depth_mm=None, height_m=None, windows=None, doors=None)
        spec.key = ex_id
        rec = M.structured_record(spec, geo2, prefs2, ans, rep, sc)
        w, d = bbox_mm(geo2)
        rec["room_spec"] = {"source": source, "from": key, "plan_file": f"plans/{ex_id}.json",
                            "width_mm": w, "depth_mm": d, "walls": len(geo2.walls)}
        rec["reason_codes"] = reason_codes(ans, rep, prefs2, changed)
        rec["checks"] = {"engine_corrections": 0, "scene_verified": True, "walls_connected": True,
                         "candidates": len(cands), "qwen_decisions_changed_by_engine": best["changes"]}
        rs = RC.summary(best["rules"])
        rec["rules_checked"] = {"passed": rs["pass"], "not_applicable": rs["n/a"], "build_issues": rs["build_fail"],
                                "results": {f"R{n}": r for n, r in best["rules"].items()}}
        rec["designer"] = {"name": args.designer, "model": args.model if args.designer == "qwen" else args.designer,
                           "seconds": best["seconds"], "advice": ADVICE_PATH.name}
        rec["door_style"] = prefs2.door_style
        stats["kept"] += 1
        return {"rec": rec, "doc": doc}
    return None


# ---------------------------------------------------------------------------
# tables + knowledge
# ---------------------------------------------------------------------------

WALL_SLOTS = 8


def flat(rec: dict) -> tuple[dict, dict]:
    room, cust, dec = rec["input"]["room"], rec["input"]["customer"], rec["output"]["decisions"]
    f: dict[str, Any] = {"id": rec["id"], "source": rec["room_spec"]["source"], "door_style": rec["door_style"],
                         "walls": len(room["walls"]), "width_mm": rec["room_spec"]["width_mm"],
                         "depth_mm": rec["room_spec"]["depth_mm"], "ceiling_mm": room["ceiling_mm"],
                         "floor_area_sqft": room["floor_area_sqft"], "size_band": room["size_band"],
                         "lofts_possible": int(room["lofts_possible"])}
    for k in range(1, WALL_SLOTS + 1):
        w = room["walls"][k - 1] if k <= len(room["walls"]) else None
        win = max(w["windows"], key=lambda o: o["width_mm"]) if w and w["windows"] else None
        f.update({f"w{k}_length_mm": w["length_mm"] if w else 0, f"w{k}_windows": len(w["windows"]) if w else 0,
                  f"w{k}_doors": len(w["doors"]) if w else 0, f"w{k}_window_width_mm": win["width_mm"] if win else 0,
                  f"w{k}_window_center_mm": win["center_mm"] if win else -1,
                  f"w{k}_longest_free_mm": w["longest_free_mm"] if w else 0,
                  f"w{k}_can_carry": int(w["can_carry_cabinets"]) if w else 0,
                  f"w{k}_corner_start": w["corner_at_start"] or "none" if w else "none",
                  f"w{k}_corner_end": w["corner_at_end"] or "none" if w else "none"})
    for k in ("layout", "budget", "fuel", "water_purifier", "built_in", "display_unit", "base_storage", "wall_style",
              "open_unit", "crockery_unit", "cooking"):
        f[f"cust_{k}"] = cust[k]
    f["cust_household_size"] = cust["household_size"] or 0
    f["cust_include_loft"] = {None: "auto", True: "yes", False: "no"}[cust["include_loft"]]
    f["cust_include_tall"] = int(cust["include_tall"])
    f["cust_cooking_wall"] = cust["cooking_wall"] or "auto"
    f["cust_sink_wall"] = cust["sink_wall"] or "auto"
    for k in range(1, WALL_SLOTS + 1):
        f[f"cust_ticked_w{k}"] = int(f"Wall {k}" in (cust["cabinet_walls"] or []))
    for k, v in cust["brief_flags"].items():
        f[f"brief_{k}"] = int(v)
    f["brief_colour"] = cust["brief_colour"] or "none"
    talls = dec["tall_units"] + [{"wall": "none", "unit": "NONE"}] * 2
    lab: dict[str, Any] = {"id": rec["id"], "shape": {1: "straight", 2: "L", 3: "U"}.get(len(dec["cabinet_walls"]), "other")}
    for k in range(1, WALL_SLOTS + 1):
        lab[f"cabinets_w{k}"] = int(f"Wall {k}" in dec["cabinet_walls"])
    lab.update({"sink_wall": dec["sink_wall"], "sink": dec["sink"], "sink_center_mm": dec["sink_center_mm"],
                "hob_wall": dec["hob_wall"], "hob": dec["hob"], "hob_center_mm": dec["hob_center_mm"],
                "spice": dec["spice"], "lpg_cylinder": dec["lpg_cylinder"],
                "tall1_wall": talls[0]["wall"], "tall1_unit": talls[0]["unit"],
                "tall2_wall": talls[1]["wall"], "tall2_unit": talls[1]["unit"],
                "water_purifier": dec["water_purifier"], "display_unit": dec["display_unit"], "open_unit": dec["open_unit"],
                "base_storage": dec["base_storage"], "wall_style": dec["wall_style"], "base_finish": dec["base_finish"],
                "wall_finish": dec["wall_finish"], "lofts": dec["lofts"]})
    for g, cs in rec["reason_codes"].items():
        lab[f"why_{g}"] = "|".join(cs)
    return f, lab


def write_tables(out: Path) -> int:
    feats, labs = [], []
    for split in ("train", "val"):
        fp = out / "scratch" / f"{split}.jsonl"
        if not fp.exists():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                f, lab = flat(json.loads(line))
                feats.append(dict(f, split=split))
                labs.append(dict(lab, split=split))
    for name, table in (("features.csv", feats), ("labels.csv", labs)):
        if table:
            cols = list(dict.fromkeys(k for row in table for k in row))
            with open(out / "scratch" / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(table)
    return len(feats)


REASON_CODE_DOC = {
    "RULE_Rn": "Rule n of the rule sheet set or limited this decision (text in knowledge/rules.json).",
    "CUSTOMER_LAYOUT / CUSTOMER_WALLS": "The customer chose the kitchen shape / the cabinet walls.",
    "CUSTOMER_COOKING_WALL / CUSTOMER_SINK_WALL": "The customer chose the hob wall (R24) / sink wall (R25).",
    "CUSTOMER_FUEL_x": "The customer's hob fuel (LPG / PNG / induction) - R11.",
    "CUSTOMER_<FIELD>": "A need the customer stated in the form (built-in, crockery, purifier, display, open, storage, wall style).",
    "CUSTOMER_NO_TALL / CUSTOMER_LOFTS": "The customer does not want tall storage / stated lofts yes or no.",
    "CUSTOMER_BUDGET_x": "The budget tier (Essential / Premium / Luxury).",
    "CUSTOMER_BRIEF": "The customer's written brief asked for it (habits, colours, look).",
    "ROOM_CEILING_ALLOWS_LOFTS / ROOM_CEILING_TOO_LOW": "The ceiling height decides whether lofts fit.",
    "ENGINE_ADJUSTED": "The placement engine changed the designer's choice to fit the room or a rule.",
    "DESIGNER_CHOICE": "No rule or customer need decided it: the designer's own judgement (see the written reason).",
}


def write_knowledge(out: Path) -> None:
    kd = out / "knowledge"
    kd.mkdir(parents=True, exist_ok=True)
    M.write_knowledge(out)  # rules.json, finishes.json, codes.json, cabinets.json (default style)
    full = M.catalog_loader.load_catalog()
    keep = ("aid", "description", "width", "depth", "height", "unit_family", "category", "door_type",
            "cabinet_combination", "shutter_count", "drawer_count", "open_box_count", "door_style",
            "opening_direction", "handle_type", "internals", "is_accessory")
    (kd / "cabinets.json").write_text(json.dumps({"source": full.source, "count": len(full),
                                                  "cabinets": [{k: getattr(it, k) for k in keep} for it in full.items]},
                                                 indent=1), encoding="utf-8")
    codes = {}
    for style in DOOR_STYLES:
        cat = M.catalog_loader.load_filtered_catalog(style, M.config.DEFAULT_HANDLE_TYPE)
        menu = M.build_menu(M.build_shortlist_v2(cat))
        codes[style] = {g: [{"code": c, "aid": it.aid, "category": it.category, "width": it.width,
                             "combination": it.cabinet_combination, "opening": it.opening_direction} for c, it in opts]
                        for g, opts in menu.items()}
    (kd / "codes.json").write_text(json.dumps(codes, indent=1), encoding="utf-8")
    (kd / "reason_codes.json").write_text(json.dumps(REASON_CODE_DOC, indent=1), encoding="utf-8")
    shutil.copy(ADVICE_PATH, kd / ADVICE_PATH.name)


# ---------------------------------------------------------------------------
# parallel: N workers, each on its own examples (i = start + k, start + k + N, ...), merged afterwards
# ---------------------------------------------------------------------------


def merge_workers(out: Path) -> int:
    """Fold finished / paused worker folders (out/w0, out/w1, ...) into the dataset. Returns examples added."""
    (out / "scratch").mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    have = set()
    for split in ("train", "val"):
        fp = out / "scratch" / f"{split}.jsonl"
        if fp.exists():
            have |= {json.loads(ln)["id"] for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip()}
    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"next_index": 0, "kept": 0}
    added = 0
    for wdir in sorted(out.glob("w[0-9]*")):
        for split in ("train", "val"):
            fp = wdir / "scratch" / f"{split}.jsonl"
            if not fp.exists():
                continue
            with open(out / "scratch" / f"{split}.jsonl", "a", encoding="utf-8") as dst:
                for ln in fp.read_text(encoding="utf-8").splitlines():
                    if ln.strip() and json.loads(ln)["id"] not in have:
                        dst.write(ln + "\n")
                        have.add(json.loads(ln)["id"])
                        added += 1
        for plan in (wdir / "plans").glob("*.json"):
            shutil.copy(plan, out / "plans" / plan.name)
        wp = wdir / "progress.json"
        if wp.exists():
            prog["next_index"] = max(prog["next_index"], json.loads(wp.read_text())["next_index"])
        log = wdir / "run.log"
        if log.exists():
            with open(out / "run.log", "a", encoding="utf-8") as dst:
                dst.write(log.read_text(encoding="utf-8"))
        shutil.rmtree(wdir, ignore_errors=True)
    prog["kept"] = len(have)
    prog_path.write_text(json.dumps(prog), encoding="utf-8")
    if added:
        write_tables(out)
    return added


def launch(args) -> int:
    """Start N worker processes on separate example numbers, show the combined progress, merge at the end."""
    import math

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merge_workers(out)  # a parallel run that was stopped earlier
    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"next_index": 0, "kept": 0}
    if not (out / "knowledge" / "rules.json").exists():
        write_knowledge(out)
    remaining = args.n - prog["kept"]
    if remaining <= 0:
        print(f"already {prog['kept']} examples - nothing to do")
        return 0
    n = args.workers
    per = math.ceil(remaining / n)
    procs = []
    for k in range(n):
        wdir = out / f"w{k}"
        wdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--designer", args.designer, "--model", args.model,
               "--n", str(per), "--best-of", str(args.best_of), "--seed", str(args.seed), "--val-every", str(args.val_every),
               "--rooms", args.rooms, "--num-ctx", str(args.num_ctx), "--timeout", str(args.timeout),
               "--out", str(wdir), "--workers", str(n), "--worker", str(k), "--start", str(prog["next_index"])]
        if args.url:
            cmd += ["--url", args.url, "--token", args.token]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(wdir / "errors.log", "a")))
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    print(f"{stamp()} {n} workers started - {prog['kept']}/{args.n} kept so far ({remaining} to go)", flush=True)
    t0, last = time.time(), -1
    try:
        while any(pr.poll() is None for pr in procs):
            time.sleep(20)
            done = 0
            for k in range(n):
                wp = out / f"w{k}" / "progress.json"
                if wp.exists():
                    done += json.loads(wp.read_text())["kept"]
            if done != last:
                last = done
                el = time.time() - t0
                left = (el / done * (remaining - done) / 3600) if done else 0
                print(f"{stamp()} {prog['kept'] + done}/{args.n} kept ({n} workers) - "
                      f"{el / max(1, done) / 60:.1f} min per design overall" + (f", ~{left:.1f} h left" if done else ""), flush=True)
    except KeyboardInterrupt:
        print("stopping the workers - their designs are kept")
        for pr in procs:
            pr.terminate()
        for pr in procs:
            pr.wait()
    added = merge_workers(out)
    codes = [pr.returncode for pr in procs]
    print(f"{stamp()} merged {added} new designs - {json.loads(prog_path.read_text())['kept']}/{args.n} in the dataset"
          + (" - PAUSED: Colab stopped answering; send the new link to continue" if 3 in codes else ""), flush=True)
    return 3 if 3 in codes else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen-on-Colab training dataset (checked by the MagickHome engine)")
    ap.add_argument("--url", help="Colab tunnel URL (printed by the notebook)")
    ap.add_argument("--token", help="Colab password (printed by the notebook)")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--designer", choices=("qwen", "policy", "mock"), default="qwen")
    ap.add_argument("--n", type=int, default=1000, help="examples to have in total (a resumed run continues)")
    ap.add_argument("--best-of", type=int, default=1,
                    help="Qwen designs per room (a failed design is adjusted to the rules first; one extra design only if that fails)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val-every", type=int, default=20, help="every n-th kept example goes to validation")
    ap.add_argument("--rooms", default=str(ROOT / "training" / "floorplans" / "db"), help="exported real rooms")
    ap.add_argument("--num-ctx", type=int, default=12288)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=str(ROOT / "training" / "dataset_qwen"))
    ap.add_argument("--workers", type=int, default=1,
                    help="designs in parallel (Colab must run Ollama with OLLAMA_NUM_PARALLEL >= this)")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.workers > 1 and args.worker is None:
        return launch(args)

    out = Path(args.out)
    (out / "scratch").mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a", encoding="utf-8")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    global LOG
    LOG = say
    procs = None
    if args.designer == "qwen":
        if not (args.url and args.token):
            print("Give --url and --token (the notebook prints the full command).")
            return 2
        client = qwen_client(args.url.rstrip("/"), args.token, args.timeout)
        try:
            check_colab(client, args.model)
        except ColabDown as exc:
            say(str(exc))
            return 3
        advice = ADVICE_PATH.read_text(encoding="utf-8")
        designer_for = lambda seed: make_qwen_designer(client, args.model, advice, seed, args.num_ctx)  # noqa: E731
    elif args.designer == "mock":
        url, token, procs = start_mock()
        client = qwen_client(url, token, 60)
        check_colab(client, "mock")
        designer_for = lambda seed: make_mock_designer(client, seed)  # noqa: E731
    else:
        designer_for = lambda seed: make_policy_designer()  # noqa: E731

    rooms = load_db_rooms(Path(args.rooms))
    prog_path = out / "progress.json"
    first = args.start + (args.worker or 0)
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"next_index": first, "kept": 0}
    stride = args.workers if args.worker is not None else 1
    if args.worker is None and not (out / "knowledge" / "rules.json").exists():
        write_knowledge(out)
    say(f"start: designer {args.designer}{' ' + args.model if args.designer == 'qwen' else ''}, "
        f"{len(rooms)} real rooms, {prog['kept']}/{args.n} kept so far, next example {prog['next_index']}")
    stats: collections.Counter = collections.Counter()
    files = {s: open(out / "scratch" / f"{s}.jsonl", "a", encoding="utf-8") for s in ("train", "val")}
    t0, kept0, code = time.time(), prog["kept"], 0
    try:
        while prog["kept"] < args.n:
            i = prog["next_index"]
            try:
                ex = make_example(i, args, rooms, designer_for, stats)
            except ColabDown as exc:
                say(f"PAUSED at example {i}: {exc}")
                say("Restart the Colab notebook (Run all) and run the new command it prints - the run continues here.")
                code = 3
                break
            prog["next_index"] = i + stride
            if ex:
                split = "val" if args.val_every and (prog["kept"] + 1) % args.val_every == 0 else "train"
                rec = dict(ex["rec"], split=split)
                files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                files[split].flush()
                (out / "plans" / f"{rec['id']}.json").write_text(json.dumps(ex["doc"]), encoding="utf-8")
                prog["kept"] += 1
                el = time.time() - t0
                per = el / max(1, prog["kept"] - kept0)
                say(f"{prog['kept']}/{args.n} kept ({rec['room_spec']['source']}, {rec['door_style']}, "
                    f"{len(rec['output']['decisions']['cabinet_walls'])} walls) - {per / 60:.1f} min each, "
                    f"~{per * (args.n - prog['kept']) / 3600:.1f} h left")
            prog_path.write_text(json.dumps(prog), encoding="utf-8")
    except KeyboardInterrupt:
        say("stopped by you - progress saved; run the same command to continue")
        code = 130
    finally:
        for f in files.values():
            f.close()
        n_rows = write_tables(out)
        stats_path = out / "stats.json"
        old = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        merged = collections.Counter(old.get("counts", {}))
        merged.update(stats)
        stats_path.write_text(json.dumps({"examples": n_rows, "counts": dict(merged.most_common())}, indent=1),
                              encoding="utf-8")
        say(f"saved: {n_rows} examples in {out / 'scratch'} (tables rebuilt)")
        if procs:
            procs[0].shutdown()
            procs[1].kill()
        log.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
