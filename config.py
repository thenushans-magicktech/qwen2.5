"""Central configuration for the MagickHome AI kitchen layout generator.

All distances in this module are millimetres unless the name ends in ``_M``
(metres). Scene JSON positions are always metres; the solver works in mm.
Every value can be overridden with an environment variable of the same name
where noted.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load a local .env file, overriding stale variables inherited from the shell."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[name] = value


_load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Local LLM (Ollama). qwen2.5-coder:7b is Apache-2.0 licensed (commercial use
# allowed); it needs ~5-5.5 GB RAM with an 8k context and takes ~4-5 min per
# design on a CPU-only laptop. qwen2.5-coder:3b is faster (~2 min) but under the
# Qwen RESEARCH licence (non-commercial) - set OLLAMA_MODEL to use it for tests.
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_AUTO_PULL = os.getenv("OLLAMA_AUTO_PULL", "1") not in ("0", "false", "False")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "1200"))  # CPU inference is slow (7B: first design ~8 min)
# Shown to users while Qwen designs (rough CPU-laptop timing for the model above).
AI_EXPECTED_TIME = os.getenv("AI_EXPECTED_TIME", "4-5 minutes" if "7b" in OLLAMA_MODEL else "2-3 minutes")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_OPTIONS = {
    # Creative: every design is a fresh idea for the same plan. The schema still only allows real
    # codes and walls, and the rules still check everything. Set MAGICKHOME_AI_TEMPERATURE=0 (and
    # a request seed) for exactly repeatable designs.
    "temperature": float(os.getenv("MAGICKHOME_AI_TEMPERATURE", "0.7")),
    "top_p": 0.9,
    "num_ctx": 8192,
    "num_predict": 1500,
}
AI_MAX_RETRIES = 1
AI_SHORTLIST_MIN = 35
AI_SHORTLIST_MAX = 45

# Designer knowledge taught to Qwen through the prompt (no GPU / no training).
# v2 = knowledge pack from KNOWLEDGE_DIR (playbook, cabinet guide, worked examples)
#      + short option codes; v1 = the original prompt, kept for A/B comparison.
AI_PROMPT_VERSION = os.getenv("MAGICKHOME_PROMPT_VERSION", "v2")
# Which AI designer: "qwen" (local / Colab Ollama) or "gemini" (Google API, needs GEMINI_API_KEY in the
# environment - never stored in files). A request can override it with preferences.ai_provider.
AI_PROVIDER = os.getenv("MAGICKHOME_AI_PROVIDER", "qwen")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "240"))
# Backup models tried in order when the main one is busy (503), out of its free daily quota (429)
# or times out - each model has its own quota. Comma-separated; "" = no backups.
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.8-flash,gemini-3.7-flash").split(",")]
# Off (default): Gemini designs from its own judgement - it gets the plain room facts, the cabinet
# list, the customer's needs and the answer format, but NOT the MagickHome knowledge pack (playbook,
# worked examples, cabinet notes, rule hints). Set GEMINI_USE_KNOWLEDGE=1 to give it the same prompt as Qwen.
GEMINI_USE_KNOWLEDGE = os.getenv("GEMINI_USE_KNOWLEDGE", "0").strip().lower() not in ("", "0", "false", "no", "off")
# Brief reader: Gemini reads the customer's free-text brief in any language, fills the form settings it
# clearly asks for and writes a clear English brief for the designer (Qwen). "gemini" (default; used only
# when GEMINI_API_KEY is set) or "off" (the designer reads the raw text). BRIEF_MODEL "" = GEMINI_MODEL.
BRIEF_READER = os.getenv("MAGICKHOME_BRIEF_READER", "gemini").strip().lower()
BRIEF_MODEL = os.getenv("MAGICKHOME_BRIEF_MODEL", "").strip()
BRIEF_TIMEOUT_S = float(os.getenv("MAGICKHOME_BRIEF_TIMEOUT_S", "60"))  # per model; busy models are skipped
KNOWLEDGE_DIR = Path(os.getenv("MAGICKHOME_KNOWLEDGE_DIR", str(BASE_DIR / "knowledge")))
# MagickHome Kitchen Rule Engine workbook (module widths by kitchen type / size band, budget
# rules, lofts ...); read by rulebook.py. Rules are cited by their row number (R1, R12, ...).
RULEBOOK_PATH = Path(os.getenv("MAGICKHOME_RULEBOOK_PATH", str(KNOWLEDGE_DIR / "Kitchen_Rule_Engine_V1.xlsx")))
# Which design rules apply:
#   "rulebook" (default) - only the MagickHome rule workbook + the checks that make a kitchen buildable
#              (inside the room, no overlaps, door / window clearance, corner units at corners, chimney
#              zone, real SKUs). The AI designer's knowledge is the workbook's rules, nothing else.
#   "extended" - also the generator's own design rules: the designer playbook and worked examples,
#              wall / tall heights grown to the ceiling, hob landing and hob-away-from-sink preferences.
DESIGN_RULES = os.getenv("MAGICKHOME_DESIGN_RULES", "rulebook").strip().lower()
# Finished web-UI designs are saved here so ?job=<id> links survive a server restart.
JOBS_DIR = Path(os.getenv("MAGICKHOME_JOBS_DIR", str(BASE_DIR / "designs")))

# ---------------------------------------------------------------------------
# Catalogue sources
# ---------------------------------------------------------------------------
# MagickHome Assets API (rule-engine cabinet list): POST ?page=N&limit=M ->
# {"success", "data": {"items": [...], "total", "page", "limit"}}.
# Set MAGICKHOME_ASSETS_API_URL="" to force the offline cache / xlsx.
ASSETS_API_URL = os.getenv(
    "MAGICKHOME_ASSETS_API_URL",
    "https://dev-ai-design.magicktech.com/assetcatalog/api/v1/ruleengine/cabinets",
)
ASSETS_API_METHOD = os.getenv("MAGICKHOME_ASSETS_API_METHOD", "POST").upper()
# Finishes (shutters, countertops, backsplashes, floors, handles ...): read-only GET, cached locally.
TEXTURES_API_URL = os.getenv("MAGICKHOME_TEXTURES_API_URL", ASSETS_API_URL.rsplit("/", 1)[0] + "/textures" if ASSETS_API_URL else "")
TEXTURES_CACHE_PATH = BASE_DIR / "data" / "texture_api_cache.json"
ASSETS_API_PAGE_SIZE = int(os.getenv("MAGICKHOME_ASSETS_API_PAGE_SIZE", "1000"))
ASSETS_API_MAX_PAGES = 50
ASSETS_API_TOKEN = os.getenv("MAGICKHOME_ASSETS_API_TOKEN", "")
ASSETS_API_TIMEOUT_S = float(os.getenv("MAGICKHOME_ASSETS_API_TIMEOUT_S", "20"))

CATALOG_XLSX_PATH = Path(os.getenv("CATALOG_XLSX_PATH", BASE_DIR / "Finalised_Asset_Catalogue_Sheet.xlsx"))
CATALOG_SHEETS = (
    "Kitchen Combinations - Slab",
    "Kitchen Combinations - Shaker 1",
    "Kitchen Combinations - Common",
)
CATALOG_AID_COLUMN = "AID Combined  Asset ID"  # note: two spaces in the production sheet

MATERIAL_PRESETS_PATH = BASE_DIR / "data" / "material_presets.json"
CATALOG_CACHE_PATH = BASE_DIR / "data" / "catalog_api_cache.json"  # last good API fetch (offline fallback)

S3_BASE = "https://aid-assertcatalogue-dev.s3.ap-south-1.amazonaws.com"
CABINET_MODEL_URL = S3_BASE + "/catalogue-models/{aid}.glb"
CABINET_MODEL_KEY = "catalogue-models/{aid}.glb"
CABINET_THUMBNAIL_URL = S3_BASE + "/catalogue-thumbnail/{aid}.png"
CABINET_PLACEHOLDER_THUMBNAIL = S3_BASE + "/catalogue-thumbnail/cabinet_dummythumbnail.png"

# ---------------------------------------------------------------------------
# Appliances (child of host cabinet for sink/hob, wall-mounted for chimney)
# ---------------------------------------------------------------------------
APPLIANCES = {
    "sink": {
        "aid": "APS02",
        "name": "Sink90",
        "category": "Sink",
        "src": S3_BASE + "/appliance-models/APS02.glb",
        "thumbnail": S3_BASE + "/appliance-thumbnail/APS02.jpg",
        "dims_mm": (780, 450, 450),  # width, height, depth
        "local_position": (0.0, 0.55, 0.0),  # relative to host cabinet
        "level": "base",
        "cutout_inset_mm": 20,  # countertop hole = appliance footprint minus inset per side
    },
    "hob": {
        "aid": "APH02",
        "name": "Hob90",
        "category": "Hob",
        "src": S3_BASE + "/appliance-models/APH02.glb",
        "thumbnail": S3_BASE + "/appliance-thumbnail/APH02.jpg",
        "dims_mm": (750, 70, 520),
        "local_position": (0.0, 0.74, 0.0),
        "level": "base",
        "cutout_inset_mm": 20,
    },
    "chimney": {
        "aid": "APC02",
        "name": "Chimney90",
        "category": "Chimney",
        "src": S3_BASE + "/appliance-models/APC02.glb",
        "thumbnail": S3_BASE + "/appliance-thumbnail/APC02.jpg",
        "dims_mm": (900, 460, 540),
        "wall_position_yz": (1.44, 0.32),  # y, z on the wall; x = hob centre
        "level": "wall",
    },
}

# Appliance sizes available as MagickHome models (dimensions measured from the GLBs). The largest
# model that fits is used: sink / hob by the host cabinet's width, chimney by the hob cabinet's
# width. The rule workbook's small modules (600 mm hob and sink units, R1 / R2 / R61 / R63)
# need the 60-series models. Names as in the MagickHome appliances API (ruleengine/appliances).
# (min host cabinet width mm, aid, name, (width, height, depth) mm)
APPLIANCE_VARIANTS = {
    "sink": [(1050, "APS03", "Left Sink105", (950, 450, 450)), (800, "APS02", "Sink90", (780, 450, 450)),
             (0, "APS01", "Sink60", (450, 390, 420))],
    "hob": [(750, "APH02", "Hob90", (750, 70, 520)), (0, "APH01", "Hob60", (550, 68, 520))],
    "chimney": [(750, "APC02", "Chimney90", (900, 460, 540)), (0, "APC01", "Chimney60", (600, 463, 540))],
}
MIN_HOB_UNIT_MM = 600  # narrowest hob cabinet (Hob60 appliance); R57: no hob below 600 mm


def appliance_for(kind: str, host_width_mm: float) -> dict:
    """APPLIANCES[kind] with the model that fits a host cabinet ``host_width_mm`` wide
    (for the chimney: the hob cabinet's width)."""
    base = APPLIANCES[kind]
    for min_w, aid, name, dims in APPLIANCE_VARIANTS[kind]:
        if host_width_mm + 1e-6 >= min_w:
            return dict(base, aid=aid, name=name, dims_mm=dims,
                        src=f"{S3_BASE}/appliance-models/{aid}.glb", thumbnail=f"{S3_BASE}/appliance-thumbnail/{aid}.jpg")
    return dict(base)


def chimney_width_mm(hob_unit_width_mm: float) -> int:
    """Width of the chimney hood above a hob cabinet of this width (600 or 900 mm)."""
    return appliance_for("chimney", hob_unit_width_mm)["dims_mm"][0]

# ---------------------------------------------------------------------------
# Elevations / depths (metres, wall-local frame: x along wall, y up, z into room)
# ---------------------------------------------------------------------------
CABINET_ROTATION = [0, 3.141592653589793, 0]
FIXTURE_ROTATION = [0, 0, 0]

POS_Y_M = {"base": 0.1, "base_floor": 0.0, "tall": 0.1, "wall": 1.44, "loft": 2.058}
POS_Z_M = {"base": 0.33, "tall": 0.33, "wall": 0.215, "wall_corner": 0.375, "loft": 0.215, "loft_deep": 0.33}

# Accepted catalogue heights per tier (mm). Base 800 = floor-standing cylinder units.
# Loft heights 600-850: the loft height follows the ceiling (rule R48, see rule_engine.height_plan).
TIER_HEIGHTS_MM = {"base": (700, 800), "wall": (600,), "tall": (1940,), "loft": (600, 650, 700, 750, 800, 850)}
TIER_DEPTHS_MM = {"base": (560,), "wall": (330, 650), "tall": (560,), "loft": (330, 560, 650)}
FLOOR_STANDING_HEIGHT_MM = 800  # base units this tall sit at y=0 (no plinth)

LOFT_TOP_M = POS_Y_M["loft"] + 0.6  # 2.658
LOFT_BOTTOM_MM = 18  # bottom panel between the wall-cabinet top and the loft carcass
# Height by ceiling: wall cabinets hang from 1440 mm and their top lines up with the tall
# units' top (tall units stand on the 100 mm plinth). With lofts the standard 600 / 1940 pair
# is kept (lofts sit on top). Without lofts the tallest pair whose top stays at least
# CEILING_GAP_MM below the ceiling is used, so storage reaches up instead of leaving a dust shelf.
WALL_TALL_HEIGHTS_MM = ((900, 2240), (700, 2040), (600, 1940))  # tallest first
STANDARD_WALL_TALL_MM = (600, 1940)
CEILING_GAP_MM = 50
DEFAULT_WALL_THICKNESS_M = 0.1
DEFAULT_WALL_HEIGHT_M = 2.5

# ---------------------------------------------------------------------------
# Clearances & ergonomic thresholds (mm)
# ---------------------------------------------------------------------------
DOOR_CLEARANCE_MM = 100          # margin kept clear either side of a door frame
DOOR_SWING_CHECK = True          # also keep hinged-door swing arcs clear on adjacent walls
WINDOW_CLEARANCE_MM = 0          # extra margin beside windows for wall/tall/loft tiers
CORNER_DEPTH_MM = 560            # base-cabinet depth reserved on the perpendicular wall
WALL_CORNER_DEPTH_MM = 650       # Wall L Corner depth reserved on the perpendicular wall
WALL_UNIT_DEPTH_MM = 330
CHIMNEY_WIDTH_MM = 900
CHIMNEY_SIDE_CLEARANCE_MM = 50   # free wall either side of the chimney hood
MIN_LANDING_MM = 300             # counter beside hob / sink
MIN_END_LANDING_MM = 450         # R26 / R27: counter from the hob / sink to the end of its run
MIN_SINK_HOB_GAP_MM = 600        # edge-to-edge when sink and hob share a wall
MIN_UNIT_WIDTH_MM = 300          # narrowest storage cabinet used for filling
MAX_FILLER_MM = 150              # widest acceptable filler strip before a hard failure
MIN_CABINET_WALL_MM = 1200       # usable base length for a wall to carry cabinets
MIN_AISLE_MM = 1200              # clear floor between facing runs (U / galley)
GRID_MM = 50                     # catalogue width module

# Opening-direction convention: catalogue "Left" = hinges on the viewer's left
# (viewer standing in the room facing the wall; wall start is on their left).
HINGE_SIDE_FOR_OPENING = {"Left": "start", "Right": "end"}

# ---------------------------------------------------------------------------
# Fixtures (procedural geometry, metres) - matched to production scenes
# ---------------------------------------------------------------------------
PANEL_T_M = 0.018
TOEKICK = {"y": 0.05, "z": 0.581, "thickness": 0.1, "depth": 0.018, "side_depth": 0.522, "side_z": 0.311}
COUNTERTOP = {"y": 0.82, "z": 0.345, "thickness": 0.04, "depth": 0.59, "edge": {"type": "rounded", "radiusMm": 10}}
BACKSPLASH = {"y": 1.14, "upper_y": 1.74, "z": 0.055, "thickness": 0.6, "depth": 0.01}
BASE_SLAB = {"y": 0.45, "z": 0.339, "thickness": 0.7, "depth": 0.578}
WALL_SLAB = {"y": 1.74, "z": 0.224, "thickness": 0.6, "depth": 0.348}
BASE_FILLER = {"y": 0.45, "z": 0.619, "thickness": 0.7}
WALL_FILLER = {"y": 1.74, "z": 0.389, "thickness": 0.6, "bottom_y": 1.449, "bottom_z": 0.215, "bottom_depth": 0.33}
TALL_FILLER = {"y": 1.07, "z": 0.619, "thickness": 1.94, "front_shift": 0.56}
TALL_SIDE_SLAB_COUNTER = {"y": 1.14, "z": 0.344, "thickness": 0.6, "depth": 0.568}
TALL_SIDE_SLAB_WALL = {"y": 1.74, "z": 0.513, "thickness": 0.6, "depth": 0.23}
TALL_SIDE_SLAB_FULL = {"y": 1.07, "z": 0.339, "thickness": 1.94, "depth": 0.578}
LOFT_BOTTOM = {"y": 2.049, "thickness": 0.018}
LOFT_SIDE = {"y": 2.358, "thickness": 0.6}
FIXTURE_COLORS = {
    "toe-kick": "#2a2a2a",
    "countertop": "#e8e8e8",
    "backsplash": "#e8e8e8",
    "slab": "#e8e8e8",
    "filler": "#e8e8e8",
    "floor": "#e8e8e8",
}

DESIGN_SOURCE = "ai-kitchen-v1"
DESIGN_SOURCE_FIXTURE = DESIGN_SOURCE + "-fixture"
DESIGN_SOURCE_APPLIANCE = DESIGN_SOURCE + "-appliance"

# ---------------------------------------------------------------------------
# Defaults for customer preferences
# ---------------------------------------------------------------------------
DEFAULT_DOOR_STYLE = "Slab (Plain)"
DEFAULT_HANDLE_TYPE = "External Handle"
