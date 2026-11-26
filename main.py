"""MagickHome AI Kitchen Layout Generator - FastAPI service + CLI.

Pipeline (100 % offline):
    empty magickhomes-scene v1 JSON
      -> scene_parser   (walls, openings, corners)
      -> catalog_loader (Assets API or Finalised_Asset_Catalogue_Sheet.xlsx)
      -> ai_designer    (local Ollama Qwen 2.5 Coder, schema-locked; deterministic fallback)
      -> rule_engine    (spatial / ergonomic solver: snapping, packing, clearances)
      -> scene_hydrator (cabinet, appliance and fixture nodes)
    -> hydrated magickhomes-scene v1 JSON for the Three.js renderer

CLI:
    python main.py --input empty_floorplan.json --output generated_kitchen.json
    python main.py --input plan.json --output out.json --layout L --loft --ai fallback --report report.json
Server + web UI:
    python main.py --serve [--host 127.0.0.1 --port 8000]     -> open http://127.0.0.1:8000
    POST /api/generate-kitchen  {"scene": <magickhomes-scene>, "preferences": {...}}   (synchronous)
    POST /api/jobs              same body -> {"id"}; GET /api/jobs/{id}; GET /api/jobs/{id}/scene
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

import config
import finishes
import rulebook
from ai_designer import AIUnavailable, KitchenPreferences, base_storage_need, customer_needs, design_kitchen, ensure_model, list_local_models, ollama_model_for
from explain import explain_design
from catalog_loader import available_door_styles, catalog_status, load_catalog, load_filtered_catalog, refresh_catalog
from rule_engine import SolverError, SolverPrefs, solve_layout, validate_layout
from scene_hydrator import hydrate_scene, scene_seed
from scene_parser import FORMAT, VERSION, SceneError, parse_scene, strip_items

log = logging.getLogger("kitchen")
STATIC_DIR = config.BASE_DIR / "static"


def generate_kitchen(
    doc: dict, prefs: Optional[KitchenPreferences] = None, progress: Optional[Callable[[str], None]] = None
) -> tuple[dict, dict]:
    """Run the full pipeline. Returns ``(hydrated_scene_doc, report)``."""
    prefs = prefs or KitchenPreferences()
    step = progress or (lambda _msg: None)
    t0 = time.time()
    step("Reading the floor plan")
    if prefs.strip_existing_items:
        doc = strip_items(doc)
    geometry = parse_scene(doc)
    step("Loading the cabinet catalogue")
    catalog = load_filtered_catalog(prefs.door_style, prefs.handle_type)

    if prefs.ai_mode == "fallback":
        step("Designing with the rule-based designer")
    elif (prefs.ai_provider or config.AI_PROVIDER) == "gemini":
        step(f"Gemini ({config.GEMINI_MODEL}) is designing the kitchen - usually 1-2 minutes")
    elif (prefs.ai_provider or config.AI_PROVIDER) == "local":
        step(f"Local model ({prefs.local_model or config.LOCAL_MODEL}) is designing the kitchen - {config.LOCAL_EXPECTED_TIME}")
    elif (prefs.ai_provider or config.AI_PROVIDER) == "gemma":
        t = config.GEMMA_EXPECTED_TIME
        step(f"Gemma ({config.GEMMA_MODEL}) is designing the kitchen - {t if t.split()[0] in ('under', 'over', 'about') else 'about ' + t}")
    else:
        t = config.AI_EXPECTED_TIME
        step(f"Qwen ({config.OLLAMA_MODEL}) is designing the kitchen - {t if t.split()[0] in ('under', 'over', 'about') else 'about ' + t}")
    proposal, ai_info = design_kitchen(geometry, catalog, prefs)
    if ai_info.get("mode") == "fallback" and prefs.ai_mode != "fallback":
        step("The AI designer was not available - used the rule-based designer instead")
    if not prefs.include_tall:
        proposal.tall_units = []
    if prefs.include_cylinder is False:
        proposal.cylinder_aid = None
    # R24 / R25: the customer's primary (hob) and secondary (sink) walls are fixed.
    for wid, attr in ((prefs.primary_wall, "hob_wall_id"), (prefs.secondary_wall, "sink_wall_id")):
        if wid and wid in geometry.walls:
            setattr(proposal, attr, wid)
            if wid not in proposal.cabinet_walls:
                proposal.cabinet_walls.append(wid)
    # Rule workbook: module widths by kitchen type / size band, LPG assumed (R11), ...
    step("Applying the MagickHome rule book: module sizes for this kitchen size and shape")
    facts = rulebook.apply_module_rules(proposal, geometry, catalog, prefs)
    # Base cabinet style: the customer's stated style, else the designer's (Qwen's) choice.
    stated_storage = base_storage_need(prefs)[0]
    storage = stated_storage if stated_storage != "auto" else (proposal.base_storage or "auto")
    # Lofts and wall-door style: the customer's answer, else the designer's decision.
    lofts = prefs.include_loft if prefs.include_loft is not None else bool(proposal.lofts)
    wall_style = prefs.wall_style if prefs.wall_style != "auto" else (proposal.wall_style or "shutters")

    step("Applying design rules: window, door, corner and chimney clearances")
    layout = solve_layout(
        geometry,
        catalog,
        proposal,
        SolverPrefs(
            include_loft=lofts,
            layout=prefs.layout,
            cabinet_walls=prefs.cabinet_walls,
            sink_under_window=prefs.sink_under_window,
            base_storage=storage,
            wall_style=wall_style,
            budget=None if prefs.budget == "auto" else prefs.budget,
            primary_wall=prefs.primary_wall if prefs.primary_wall in geometry.walls else None,
            secondary_wall=prefs.secondary_wall if prefs.secondary_wall in geometry.walls else None,
        ),
    )
    # Shutter finishes: the customer's > the designer's (within the budget tier's finish rules) > tier default
    budget = None if prefs.budget == "auto" else prefs.budget
    palette, fin_src, fin_notes = finishes.resolve({"base": prefs.base_finish, "wall": prefs.wall_finish},
                                                   proposal.finishes, budget)
    surfaces = finishes.surface_choice({"countertop": prefs.countertop_finish, "backsplash": prefs.backsplash_finish})
    layout.finishes = {**palette, **surfaces}
    finish_info = {"palette": palette, "source": fin_src, "notes": fin_notes, "budget": budget,
                   "tier": finishes.rules().get("tiers", {}).get(budget or "", {}).get("note", ""),
                   "default_why": finishes.default(budget)[1]}
    step("Building the 3D scene: cabinets, appliances, countertops, panels")
    seed = prefs.seed if prefs.seed is not None else scene_seed(doc, prefs.model_dump_json())
    out, stats = hydrate_scene(doc, layout, seed=seed, include_floor=prefs.include_floor)
    issues = [msg for msg, _ in validate_layout(layout)]
    report = {
        "ai": ai_info,
        "customer_needs": customer_needs(prefs),
        "explanation": explain_design(proposal, layout, prefs, geometry, catalog, facts, finish_info),
        "proposal": proposal.to_dict(),
        "layout": layout.summary(),
        "heights": dict(layout.heights, wall_style=wall_style),
        "rulebook": facts.to_dict(),
        "finishes": _finish_report(out, palette, fin_src, budget, finish_info["tier"], fin_notes, surfaces),
        "nodes_added": stats,
        "validation_issues": issues,
        "catalog": {"source": catalog.source, "filtered_skus": len(catalog), "door_style": prefs.door_style},
        "seconds": round(time.time() - t0, 2),
    }
    return out, report


def _finish_report(scene_doc: dict, palette: dict, src: dict, budget, tier_note: str, notes: list,
                   surfaces: Optional[dict] = None) -> dict:
    """report.finishes: the cabinet runs plus the worktop / backsplash actually in the scene."""
    rep = {run: dict(finishes.summary(aid), source=src.get(run)) for run, aid in palette.items() if run in finishes.RUNS}
    metas = [n.get("metadata") or {} for n in scene_doc["scene"]["nodes"].values()]
    for kind in finishes.SURFACES:
        first = next((m["textures"][0] for m in metas if m.get("fixtureKind") == kind and m.get("textures")), None)
        if first:
            rep[kind] = dict(finishes.summary(first["aid"], finishes.surface_options(kind)),
                             source="customer" if (surfaces or {}).get(kind) else "default")
    rep.update(budget=budget, tier_rule=tier_note, notes=notes)
    return rep


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    scene: dict[str, Any] = Field(..., description="magickhomes-scene v1 document (or its inner `scene` object)")
    preferences: KitchenPreferences = Field(default_factory=KitchenPreferences)
    include_report: bool = Field(False, description="Wrap the response as {scene, report}")
    filename: Optional[str] = Field(None, description="Original upload name (used for the download name)")


def _request_doc(req: GenerateRequest) -> dict:
    doc = req.scene
    if "nodes" in doc and "format" not in doc:
        # Client posted the raw export: {"format", "version", "scene": {...}}.
        extra = req.model_extra or {}
        doc = {"format": extra.get("format", FORMAT), "version": extra.get("version", VERSION),
               "exportedAt": extra.get("exportedAt"), "scene": doc}
    return doc


def _error_payload(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, SceneError):
        return 400, f"invalid scene: {exc}"
    if isinstance(exc, AIUnavailable):
        return 503, str(exc)
    if isinstance(exc, (SolverError, ValueError)):
        return 422, str(exc)
    return 500, f"{type(exc).__name__}: {exc}"


class JobStore:
    """In-memory background jobs for the web UI.

    One worker: Ollama on a CPU laptop serves one generation at a time, so
    further jobs queue instead of competing for the same cores.
    """

    def __init__(self, keep: int = 25):
        self.keep = keep
        self.jobs: "OrderedDict[str, dict]" = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kitchen-job")
        # Quick-rules designs never call Qwen / Gemini and take about a second: their own lane, so
        # they never wait behind a multi-minute AI design.
        self.fast_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kitchen-fast")

    def submit(self, doc: dict, prefs: KitchenPreferences, filename: Optional[str]) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "queued",
            "filename": filename or "floorplan.json",
            "created": time.time(),
            "started": None,
            "finished": None,
            "stage": "Waiting in queue",
            "log": [],
            "preferences": prefs.model_dump(),
            "error": None,
            "error_code": None,
            "report": None,
            "result": None,
            "lane": "fast" if prefs.ai_mode == "fallback" else "ai",
        }
        with self.lock:
            self.jobs[job_id] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)
        (self.fast_pool if job["lane"] == "fast" else self.pool).submit(self._run, job_id, doc, prefs)
        return job

    def _progress(self, job_id: str, msg: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job["stage"] = msg
                job["log"].append({"t": round(time.time() - (job["started"] or job["created"]), 1), "msg": msg})

    def _run(self, job_id: str, doc: dict, prefs: KitchenPreferences) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job["status"], job["started"] = "running", time.time()
        try:
            out, report = generate_kitchen(doc, prefs, progress=lambda m: self._progress(job_id, m))
            with self.lock:
                job.update(status="done", result=out, report=report, stage="Done")
        except Exception as exc:  # surfaced to the UI
            code, msg = _error_payload(exc)
            log.exception("job %s failed", job_id) if code == 500 else log.warning("job %s failed: %s", job_id, msg)
            with self.lock:
                job.update(status="error", error=msg, error_code=code, stage="Failed")
        finally:
            with self.lock:
                job["finished"] = time.time()
            self._save(job)

    @staticmethod
    def _path(job_id: str) -> Path:
        return Path(config.JOBS_DIR) / f"{job_id}.json"

    def _save(self, job: dict) -> None:
        """Keep finished designs on disk so their links survive a server restart."""
        try:
            Path(config.JOBS_DIR).mkdir(parents=True, exist_ok=True)
            with self.lock:
                data = json.dumps(job, ensure_ascii=False)
            self._path(job["id"]).write_text(data, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk full / read-only
            log.warning("could not save design %s: %s", job.get("id"), exc)

    def _load(self, job_id: str) -> Optional[dict]:
        if not job_id.isalnum():
            return None
        path = self._path(job_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def get(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            job = self._load(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="design not found (it may have been deleted)")
            with self.lock:
                self.jobs[job_id] = job
                while len(self.jobs) > self.keep:
                    self.jobs.popitem(last=False)
        return job

    def public(self, job: dict) -> dict:
        with self.lock:
            ahead = [j for j in self.jobs.values() if j["status"] == "queued" and j["created"] < job["created"]
                     and j.get("lane", "ai") == job.get("lane", "ai")]
            now = time.time()
            return {
                "id": job["id"],
                "status": job["status"],
                "stage": job["stage"],
                "log": list(job["log"]),
                "filename": job["filename"],
                "queue_position": len(ahead) if job["status"] == "queued" else 0,
                "elapsed_s": round((job["finished"] or now) - (job["started"] or now), 1) if job["started"] else 0,
                "error": job["error"],
                "error_code": job["error_code"],
                "report": job["report"],
            }


JOBS = JobStore()


def _download_name(filename: str, suffix: str) -> str:
    stem = Path(filename or "floorplan").stem or "floorplan"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[:80]
    return f"{safe}_{suffix}.json"


def _warmup() -> None:
    try:
        # Warm up (and pull if missing) only the default designer's model; the other one is
        # checked when a design asks for it.
        model = ollama_model_for(config.AI_PROVIDER)
        ensure_model(model=model)
        log.info("Ollama model %s ready", model)
    except AIUnavailable as exc:
        log.warning("AI warm-up: %s (requests will use the deterministic fallback in 'auto' mode)", exc)
    try:
        load_catalog()
    except Exception as exc:  # pragma: no cover
        log.error("catalogue load failed: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="MagickHome AI Kitchen Layout Generator",
    version="1.0.0",
    description="Offline kitchen layout generation: local Ollama Qwen 2.5 Coder + deterministic rule solver.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    # no-cache: a normal browser refresh always picks up the latest UI
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})


# UI assets (vendored three.js for the 3D view). Windows can map .js to text/plain from the
# registry, which browsers refuse for ES modules - force the right type.
import mimetypes  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402

mimetypes.add_type("text/javascript", ".js")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _local_models() -> dict:
    """Installed Ollama models for the 'Local model' designer (empty when Ollama is not reachable)."""
    try:
        return {"available": True, "models": list_local_models(), "default": config.LOCAL_MODEL,
                "ai_expected_time": config.LOCAL_EXPECTED_TIME, "knowledge": config.LOCAL_USE_KNOWLEDGE}
    except Exception as exc:
        return {"available": False, "models": [], "status": f"unavailable: {exc}"}


@app.get("/health")
def health() -> dict:
    status = "ready"
    try:
        ensure_model(auto_pull=False)
    except AIUnavailable as exc:
        status = f"unavailable: {exc}"
    gemma_status = "ready"
    try:
        ensure_model(auto_pull=False, model=config.GEMMA_MODEL)
    except AIUnavailable as exc:
        gemma_status = f"unavailable: {exc}"
    return {
        "status": "ok",
        "ollama_model": config.OLLAMA_MODEL,
        "ai_expected_time": config.AI_EXPECTED_TIME,
        "ollama": status,
        # Gemini is offered in the UI only when the server was started with GEMINI_API_KEY set
        "gemini": {"available": bool(os.environ.get("GEMINI_API_KEY", "").strip()), "model": config.GEMINI_MODEL,
                   "knowledge": config.GEMINI_USE_KNOWLEDGE},
        "gemma": {"available": gemma_status == "ready", "status": gemma_status, "model": config.GEMMA_MODEL,
                  "ai_expected_time": config.GEMMA_EXPECTED_TIME},
        "local": _local_models(),
        "catalog": catalog_status(),
    }


@app.get("/api/door-styles")
def door_styles() -> list[str]:
    return available_door_styles()


SAMPLE_SCENE = config.BASE_DIR / "magickhomes-scene_2026-11-18_10-38-22.json"


@app.get("/api/sample-floorplan")
def sample_floorplan() -> dict:
    """The production reference scene with all items stripped (an empty floor plan)."""
    if not SAMPLE_SCENE.exists():
        raise HTTPException(status_code=404, detail="sample scene not found")
    return strip_items(json.loads(SAMPLE_SCENE.read_text(encoding="utf-8")))


@app.get("/api/rulebook")
def rulebook_options() -> dict:
    """The rule workbook's filters (budget, fuel, size band, kitchen type, walls) with the rules each
    one switches on, and the module-width table - the UI builds its rule-book filters from this."""
    return rulebook.options()


@app.get("/api/finishes")
def finish_options() -> dict:
    """Shutter finishes with the budget-tier finish rules: which finishes each tier allows per run
    (base = base + tall units, wall = wall units + lofts), the tier defaults and the match rules."""
    r = finishes.rules()
    budgets = [None, "essential", "premium", "luxury"]
    return {
        "finishes": [finishes.summary(a) for a in finishes.catalogue()],
        "tiers": r.get("tiers", {}),
        "allowed": {b or "auto": {run: finishes.allowed(run, b) for run in finishes.RUNS} for b in budgets},
        "defaults": {b or "auto": dict(zip(("palette", "why"), finishes.default(b))) for b in budgets},
        "match": r.get("match", []),
        "about": r.get("about", ""),
        "surfaces": {k: [finishes.summary(a, finishes.surface_options(k)) for a in finishes.surface_options(k)]
                     for k in finishes.SURFACES},
    }


@app.post("/api/plan-facts")
def plan_facts(req: GenerateRequest) -> dict:
    """Rule-book facts of an uploaded floor plan: floor area and size band, ceiling, lofts, walls, aisles."""
    try:
        geometry = parse_scene(_request_doc(req))
    except SceneError as exc:
        raise HTTPException(status_code=400, detail=f"invalid scene: {exc}")
    return rulebook.plan_facts(geometry, req.preferences.cabinet_walls)


@app.post("/api/catalog/refresh")
def catalog_refresh() -> dict:
    cat = refresh_catalog()
    finishes.refresh()  # re-read the textures API too
    return {"source": cat.source, "skus": len(cat), "finishes": len(finishes.catalogue())}


@app.post("/api/generate-kitchen")
def generate_endpoint(req: GenerateRequest):
    try:
        out, report = generate_kitchen(_request_doc(req), req.preferences)
    except Exception as exc:
        code, msg = _error_payload(exc)
        if code == 500:
            raise
        raise HTTPException(status_code=code, detail=msg)
    if req.include_report:
        return {"scene": out, "report": report}
    return out


@app.post("/api/jobs", status_code=202)
def create_job(req: GenerateRequest) -> dict:
    """Queue a generation (used by the web UI); poll ``GET /api/jobs/{id}``."""
    doc = _request_doc(req)
    try:
        parse_scene(doc)  # reject bad uploads immediately instead of after queueing
    except SceneError as exc:
        raise HTTPException(status_code=400, detail=f"invalid scene: {exc}")
    return JOBS.public(JOBS.submit(doc, req.preferences, req.filename))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return JOBS.public(JOBS.get(job_id))


@app.get("/api/jobs/{job_id}/scene")
def job_scene(job_id: str, download: bool = False):
    job = JOBS.get(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job is {job['status']}")
    body = json.dumps(job["result"], indent=1, ensure_ascii=False)
    headers = {"Content-Disposition": f'attachment; filename="{_download_name(job["filename"], "kitchen")}"'} if download else {}
    return Response(body, media_type="application/json", headers=headers)


class FinishChange(BaseModel):
    """New finishes for a finished design (missing / null = keep the current one)."""
    base_finish: Optional[str] = None
    wall_finish: Optional[str] = None
    countertop_finish: Optional[str] = None
    backsplash_finish: Optional[str] = None


@app.post("/api/jobs/{job_id}/finishes")
def change_finishes(job_id: str, req: FinishChange) -> dict:
    """Re-apply finishes to a finished design without redesigning it: cabinet fronts (with their
    panels and fillers), worktop and backsplash. The budget tier's finish rules still apply (a
    customer's choice outside the tier is kept and flagged). Updates the saved scene and report."""
    job = JOBS.get(job_id)
    if job["status"] != "done" or not job.get("result"):
        raise HTTPException(status_code=409, detail=f"job is {job['status']}")
    rep = job["report"]
    old = rep.get("finishes") or {}
    budget = old.get("budget")
    stated = {run: getattr(req, f"{run}_finish") or (old.get(run) or {}).get("aid") for run in finishes.RUNS}
    palette, src, notes = finishes.resolve(stated, {}, budget)
    surfaces = finishes.surface_choice({k: getattr(req, f"{k}_finish") for k in finishes.SURFACES})
    with JOBS.lock:
        changed = finishes.apply_to_scene(job["result"], {**palette, **surfaces})
        tier = finishes.rules().get("tiers", {}).get(budget or "", {}).get("note", "")
        rep["finishes"] = _finish_report(job["result"], palette, src, budget, tier, notes,
                                         {**{k: True for k in surfaces}, **{k: True for k in finishes.SURFACES
                                          if (old.get(k) or {}).get("source") == "customer"}})
        for e in rep.get("explanation") or []:
            if e.get("item") == "Finishes":
                e["built"] = "; ".join(f"{'Base + tall' if r == 'base' else 'Wall + loft'}: {finishes.describe(a)}"
                                       for r, a in palette.items())
                e["why"], e["source"] = "Changed by the customer after the design.", "customer"
                e["rules"] = ([f"{budget.title()} tier: {tier}"] if budget else ["No budget stated: any finish allowed"]) + notes
    JOBS._save(job)
    return {"changed_nodes": changed, "finishes": rep["finishes"]}


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str, download: bool = False):
    job = JOBS.get(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job is {job['status']}")
    headers = {"Content-Disposition": f'attachment; filename="{_download_name(job["filename"], "design_report")}"'} if download else {}
    return JSONResponse(job["report"], headers=headers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MagickHome AI kitchen layout generator")
    ap.add_argument("--input", type=Path, help="empty magickhomes-scene JSON")
    ap.add_argument("--output", type=Path, help="where to write the generated scene")
    ap.add_argument("--report", type=Path, help="optional JSON design report")
    ap.add_argument("--door-style", default=config.DEFAULT_DOOR_STYLE)
    ap.add_argument("--handle-type", default=config.DEFAULT_HANDLE_TYPE)
    ap.add_argument("--layout", default="auto", choices=["auto", "single", "L", "U"])
    ap.add_argument("--walls", help="comma-separated wall ids that must carry cabinets")
    ap.add_argument("--loft", action="store_true", help="customer wants lofts (when the wall height allows); "
                                                         "without it the designer decides")
    ap.add_argument("--no-tall", action="store_true")
    ap.add_argument("--ai", default="auto", choices=["auto", "ollama", "fallback"])
    ap.add_argument("--local-model", help="Ollama model for --designer local, e.g. qwen3.5:2b")
    ap.add_argument("--designer", choices=["qwen", "gemma", "gemini", "local"],
                    help="which AI designer (default: MAGICKHOME_AI_PROVIDER, else qwen)")
    ap.add_argument("--notes", default="", help="design brief for the AI")
    ap.add_argument("--keep-items", action="store_true", help="do not strip existing item nodes")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--serve", action="store_true", help="run the FastAPI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # catalogue names contain '–' / '×'
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.serve or not args.input:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    doc = json.loads(args.input.read_text(encoding="utf-8"))
    prefs = KitchenPreferences(
        door_style=args.door_style,
        handle_type=args.handle_type,
        layout=args.layout,
        cabinet_walls=[w.strip() for w in args.walls.split(",")] if args.walls else None,
        include_loft=True if args.loft else None,  # not given: the designer decides
        include_tall=not args.no_tall,
        ai_mode=args.ai,
        ai_provider=args.designer,
        local_model=args.local_model,
        style_notes=args.notes,
        strip_existing_items=not args.keep_items,
        seed=args.seed,
    )
    try:
        out, report = generate_kitchen(doc, prefs)
    except (SceneError, SolverError, AIUnavailable, ValueError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 2
    output = args.output or args.input.with_name(args.input.stem + "_kitchen.json")
    output.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lay = report["layout"]
    print(f"AI mode: {report['ai'].get('mode')}  ({report['ai'].get('fallback_reason', report['ai'].get('model', ''))})")
    print(f"Proposal: {report['proposal']['notes']}")
    for wid, items in lay["placements"].items():
        print(f"  {wid}: " + ", ".join(f"{i['category']} {i['width']}" for i in items if i["tier"] in ("base", "tall")))
    for d in lay["decisions"]:
        print(f"  decision: {d}")
    for w in lay["warnings"]:
        print(f"  warning: {w}")
    if report["validation_issues"]:
        print("  VALIDATION ISSUES:", *report["validation_issues"], sep="\n    ")
    print(f"Wrote {output} ({sum(report['nodes_added'].values())} nodes added: {report['nodes_added']}) in {report['seconds']} s")
    return 0 if not report["validation_issues"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
