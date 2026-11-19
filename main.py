"""MagickHome AI Kitchen Layout Generator - FastAPI service + CLI.

Pipeline (100 % offline):
    empty magickhomes-scene v1 JSON
      -> scene_parser   (walls, openings, corners)
      -> catalog_loader (Assets API or Finalised_Asset_Catalogue_Sheet.xlsx)
      -> ai_designer    (local Ollama qwen2.5-coder:3b, schema-locked; deterministic fallback)
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
from ai_designer import AIUnavailable, KitchenPreferences, design_kitchen, ensure_model
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
    else:
        step(f"Qwen ({config.OLLAMA_MODEL}) is designing the kitchen - about 2-3 min on CPU")
    proposal, ai_info = design_kitchen(geometry, catalog, prefs)
    if not prefs.include_tall:
        proposal.tall_units = []
    if prefs.include_cylinder is False:
        proposal.cylinder_aid = None

    step("Applying design rules: window, door, corner and chimney clearances")
    layout = solve_layout(
        geometry,
        catalog,
        proposal,
        SolverPrefs(
            include_loft=prefs.include_loft,
            layout=prefs.layout,
            cabinet_walls=prefs.cabinet_walls,
            sink_under_window=prefs.sink_under_window,
        ),
    )
    step("Building the 3D scene: cabinets, appliances, countertops, panels")
    seed = prefs.seed if prefs.seed is not None else scene_seed(doc, prefs.model_dump_json())
    out, stats = hydrate_scene(doc, layout, seed=seed, include_floor=prefs.include_floor)
    issues = [msg for msg, _ in validate_layout(layout)]
    report = {
        "ai": ai_info,
        "proposal": proposal.to_dict(),
        "layout": layout.summary(),
        "nodes_added": stats,
        "validation_issues": issues,
        "catalog": {"source": catalog.source, "filtered_skus": len(catalog), "door_style": prefs.door_style},
        "seconds": round(time.time() - t0, 2),
    }
    return out, report


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
        }
        with self.lock:
            self.jobs[job_id] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)
        self.pool.submit(self._run, job_id, doc, prefs)
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

    def get(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found (it may have expired)")
            return job

    def public(self, job: dict) -> dict:
        with self.lock:
            ahead = [j for j in self.jobs.values() if j["status"] == "queued" and j["created"] < job["created"]]
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
        ensure_model()
        log.info("Ollama model %s ready", config.OLLAMA_MODEL)
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
    description="Offline kitchen layout generation: Ollama qwen2.5-coder:3b + deterministic rule solver.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health() -> dict:
    status = "ready"
    try:
        ensure_model(auto_pull=False)
    except AIUnavailable as exc:
        status = f"unavailable: {exc}"
    return {"status": "ok", "ollama_model": config.OLLAMA_MODEL, "ollama": status, "catalog": catalog_status()}


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


@app.post("/api/catalog/refresh")
def catalog_refresh() -> dict:
    cat = refresh_catalog()
    return {"source": cat.source, "skus": len(cat)}


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
    ap.add_argument("--loft", action="store_true", help="add loft units when wall height allows")
    ap.add_argument("--no-tall", action="store_true")
    ap.add_argument("--ai", default="auto", choices=["auto", "ollama", "fallback"])
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
        include_loft=args.loft,
        include_tall=not args.no_tall,
        ai_mode=args.ai,
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
