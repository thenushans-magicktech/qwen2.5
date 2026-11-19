"""Load and normalise the MagickHome kitchen cabinet catalogue.

Sources (in order):
1. MagickHome Assets API - ``POST {ASSETS_API_URL}?page=N&limit=1000`` returning
   ``{"success", "data": {"items": [...], "total", "page", "limit"}}``. Every
   successful fetch is cached to ``data/catalog_api_cache.json``.
2. The last API cache (when the API is unreachable, e.g. offline).
3. ``Finalised_Asset_Catalogue_Sheet.xlsx`` (fully offline), sheets
   ``Kitchen Combinations - Slab / Shaker 1 / Common``.

All sources are normalised into immutable :class:`CatalogItem` records. String
fields are stripped (the production sheet has trailing whitespace in
``Door Style`` - e.g. ``"Shaker -1 "`` - and in ``Category (Sub Unit)``), and a
canonical key (lower-case, dashes unified, whitespace collapsed) is used for
category matching so ``"Blind corner unit – L"`` and ``"Blind corner unit - L"``
compare equal. Door styles match on letters+digits only, so the API's
``"Shaker 1"`` and the sheet's ``"Shaker -1 "`` are the same style.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Optional

import config

log = logging.getLogger(__name__)

# Canonical category keys used by the solver --------------------------------
CAT_BASE = "base unit"
CAT_SINK = "base sink unit"
CAT_HOB = "base hob unit"
CAT_SPICE = "base spice pull out"
CAT_CYLINDER = "base cylinder unit"
CAT_GRAIN = "base grain trolley"
CAT_WICKER = "base wicker basket unit"
CAT_BCU_L = "blind corner unit - l"
CAT_BCU_R = "blind corner unit - r"
CAT_WALL = "wall unit"
CAT_WALL_CORNER = "wall l corner"
CAT_TALL = "tall unit"
CAT_OVEN = "tall built-in oven"
CAT_LOFT = "loft - modular"
CAT_LOFT_FRAME = "loft - frame with shutter"
CAT_LOFT_CORNER = "loft - l corner - modular"
# Corner-only categories: the only cabinets allowed where two runs meet.
CORNER_CATS = (CAT_BCU_L, CAT_BCU_R, CAT_WALL_CORNER, CAT_LOFT_CORNER)

BASE_STORAGE_CATS = (CAT_BASE, CAT_CYLINDER, CAT_GRAIN, CAT_WICKER)
WALL_STORAGE_CATS = (CAT_WALL,)
TALL_CATS = (CAT_TALL, CAT_OVEN)
LOFT_CATS = (CAT_LOFT, CAT_LOFT_FRAME)

_DASHES = re.compile(r"[‐-―−]")
_WS = re.compile(r"\s+")


def clean(value: Any) -> Optional[str]:
    """Strip a catalogue string; NaN / empty -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = _WS.sub(" ", str(value)).strip()
    return s or None


def canon(value: Any) -> str:
    """Canonical matching key: dashes unified, whitespace collapsed, lower-case."""
    s = clean(value) or ""
    return _DASHES.sub("-", s).lower()


def style_key(value: Any) -> str:
    """Door-style key: letters and digits only ('Shaker 1' == 'Shaker -1 ')."""
    return re.sub(r"[^a-z0-9]", "", canon(value))


def _opt(value: Any) -> Optional[str]:
    """Like :func:`clean` but the API's literal 'None' also means missing."""
    s = clean(value)
    return None if s is None or s.lower() in ("none", "null", "nan", "n/a") else s


def _int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _bool(value: Any, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


_FAMILY_TIER = {
    "base": "base",
    "base corner": "base",
    "wall": "wall",
    "wall corner": "wall",
    "tall": "tall",
    "loft": "loft",
    "loft corner": "loft",
}


@dataclass(frozen=True)
class CatalogItem:
    aid: str
    description: str
    width: int
    depth: int
    height: int
    unit_family: str
    category: str
    door_type: Optional[str]
    cabinet_combination: Optional[str]
    shutter_count: Optional[int]
    drawer_count: Optional[int]
    open_box_count: Optional[int]
    door_style: Optional[str]
    opening_direction: Optional[str]
    handle_type: Optional[str]
    internals: Optional[str]
    is_accessory: bool = False
    is_active: bool = True
    source: str = "xlsx"
    sheet: Optional[str] = None
    # Full Assets-API record (metadata block) when loaded from the API.
    api_record: Optional[dict] = field(default=None, compare=False, hash=False, repr=False)

    @property
    def cat(self) -> str:
        return canon(self.category)

    @property
    def tier(self) -> str:
        return _FAMILY_TIER.get(canon(self.unit_family), canon(self.unit_family))

    @property
    def is_corner(self) -> bool:
        return "corner" in canon(self.unit_family) or "corner" in self.cat

    @property
    def is_single_shutter(self) -> bool:
        return self.opening_direction in ("Left", "Right")

    @property
    def floor_standing(self) -> bool:
        return self.tier == "base" and self.height >= config.FLOOR_STANDING_HEIGHT_MM

    def short(self) -> str:
        """Compact one-line description used in LLM prompts."""
        return (
            f"{self.aid} | {self.category} | {self.width}x{self.depth}x{self.height} | "
            f"{self.cabinet_combination or '-'} | opens {self.opening_direction or '-'}"
        )

    def metadata(self) -> dict:
        """Scene ``metadata`` block (API record when available, else derived)."""
        if self.api_record:
            # API records lack the designer-added keys seen in production scenes.
            meta = dict(self.api_record)
            meta.setdefault("id", meta.get("_id") or self.aid)
            meta.setdefault("name", self.aid)
            meta.setdefault("imageUrl", meta.get("thumbnail") or config.CABINET_THUMBNAIL_URL.format(aid=self.aid))
            meta.setdefault("aidAssetId", self.aid)
            meta.setdefault("subCategory", self.category)
            meta["catalogSource"] = self.source
            return meta
        aid = self.aid
        return {
            "_id": aid,
            "aid": aid,
            "description": self.description,
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "unitFamily": self.unit_family,
            "category": self.category,
            "doorType": self.door_type,
            "cabinetCombination": self.cabinet_combination,
            "shutterCount": self.shutter_count or 0,
            "drawerCount": self.drawer_count or 0,
            "doorStyle": self.door_style,
            "openingDirection": self.opening_direction,
            "handleType": self.handle_type,
            "internals": self.internals,
            "modelUrl": config.CABINET_MODEL_URL.format(aid=aid),
            "modelKey": config.CABINET_MODEL_KEY.format(aid=aid),
            "thumbnail": config.CABINET_THUMBNAIL_URL.format(aid=aid),
            "placeholderThumbnail": config.CABINET_PLACEHOLDER_THUMBNAIL,
            "isActive": self.is_active,
            "id": aid,
            "name": aid,
            "imageUrl": config.CABINET_THUMBNAIL_URL.format(aid=aid),
            "aidAssetId": aid,
            "subCategory": self.category,
            "catalogSource": self.source,
        }


class Catalog:
    """Indexed, filterable collection of catalogue items."""

    def __init__(self, items: Iterable[CatalogItem], source: str = "xlsx"):
        self.items: list[CatalogItem] = list(items)
        self.source = source
        self.by_aid: dict[str, CatalogItem] = {it.aid: it for it in self.items}

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, aid: str) -> bool:
        return aid in self.by_aid

    def get(self, aid: str) -> Optional[CatalogItem]:
        return self.by_aid.get(aid)

    def filter(
        self,
        door_style: Optional[str] = None,
        handle_type: Optional[str] = None,
        active_only: bool = True,
        tier_heights: Optional[dict] = None,
        tier_depths: Optional[dict] = None,
    ) -> "Catalog":
        ds, ht = style_key(door_style), canon(handle_type)
        out = []
        for it in self.items:
            if active_only and not it.is_active:
                continue
            if door_style and style_key(it.door_style) != ds:
                continue
            if handle_type and canon(it.handle_type) != ht:
                continue
            if tier_heights is not None and it.height not in tier_heights.get(it.tier, ()):
                continue
            if tier_depths is not None and it.depth not in tier_depths.get(it.tier, ()):
                continue
            out.append(it)
        return Catalog(out, self.source)

    def find(
        self,
        cat: str | Iterable[str],
        width: Optional[int] = None,
        depth: Optional[int] = None,
        height: Optional[int] = None,
        opening: Optional[str] = None,
        combination: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> list[CatalogItem]:
        cats = {cat} if isinstance(cat, str) else set(cat)
        comb = canon(combination) if combination else None
        res = [
            it
            for it in self.items
            if it.cat in cats
            and (width is None or it.width == width)
            and (depth is None or it.depth == depth)
            and (height is None or it.height == height)
            and (opening is None or it.opening_direction == opening)
            and (comb is None or canon(it.cabinet_combination) == comb)
            and (tier is None or it.tier == tier)
        ]
        return sorted(res, key=_default_rank)

    def widths(self, cat: str | Iterable[str], **kw) -> list[int]:
        return sorted({it.width for it in self.find(cat, **kw)})

    def variant(self, item: CatalogItem, opening: str) -> Optional[CatalogItem]:
        """Same SKU family/size/internals with a different opening direction."""
        if item.opening_direction == opening:
            return item
        for it in self.items:
            if (
                it.cat == item.cat
                and it.width == item.width
                and it.depth == item.depth
                and it.height == item.height
                and canon(it.cabinet_combination) == canon(item.cabinet_combination)
                and canon(it.internals) == canon(item.internals)
                and canon(it.door_type) == canon(item.door_type)
                and it.opening_direction == opening
            ):
                return it
        return None


# Preference order when several SKUs share a size: shelves > no shelf, plain
# shutter combinations first (reference designs use "One Shelf" double shutters).
_COMBO_RANK = {
    "double shutter": 0,
    "single shutter": 1,
    "four drawer": 2,
    "double drawer": 3,
    "triple drawer - a": 0,
    "spice pullout": 0,
    "sink double shutter": 0,
    "sink single shutter": 1,
}
_INTERNAL_RANK = {"one shelf": 0, "no shelf": 1, "two shelf": 2, "four shelf": 0}
_OPEN_RANK = {"Both": 0, "Pull Out": 0, "Left": 1, "Right": 2}


def _default_rank(it: CatalogItem):
    return (
        _COMBO_RANK.get(canon(it.cabinet_combination), 5),
        _INTERNAL_RANK.get(canon(it.internals), 3),
        _OPEN_RANK.get(it.opening_direction or "", 3),
        it.aid,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _items_from_xlsx(path) -> list[CatalogItem]:
    import pandas as pd

    items: list[CatalogItem] = []
    xls = pd.ExcelFile(path)
    for sheet in config.CATALOG_SHEETS:
        if sheet not in xls.sheet_names:
            log.warning("catalogue sheet %r missing from %s", sheet, path)
            continue
        df = pd.read_excel(xls, sheet)
        df.columns = [_WS.sub(" ", str(c)).strip() for c in df.columns]
        aid_col = _WS.sub(" ", config.CATALOG_AID_COLUMN).strip()
        for row in df.to_dict("records"):
            aid = clean(row.get(aid_col))
            width, depth, height = _int(row.get("Width")), _int(row.get("Depth")), _int(row.get("Height"))
            if not aid or not width or not depth or not height:
                continue
            items.append(
                CatalogItem(
                    aid=aid,
                    description=clean(row.get("Combined Product Description")) or aid,
                    width=width,
                    depth=depth,
                    height=height,
                    unit_family=clean(row.get("Unit Family")) or "",
                    category=clean(row.get("Category (Sub Unit)")) or "",
                    door_type=clean(row.get("Door Type")),
                    cabinet_combination=clean(row.get("Cabinet Combination")),
                    shutter_count=_int(row.get("Shutter Count")),
                    drawer_count=_int(row.get("Drawer Count")),
                    open_box_count=_int(row.get("Open Box Count")),
                    door_style=clean(row.get("Door Style")),
                    opening_direction=clean(row.get("Opening Direction")),
                    handle_type=clean(row.get("Handle Type")),
                    internals=clean(row.get("Internals")),
                    is_accessory=_bool(row.get("Applianaces/Accesory"), default=False),
                    is_active=_bool(row.get("isActive", True)),
                    source="xlsx",
                    sheet=sheet,
                )
            )
    return items


def _page_records(payload) -> tuple[list, Optional[int]]:
    """Extract (records, total) from one API page; tolerant of envelope shapes."""
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        raise ValueError("Assets API returned a non-JSON-object payload")
    if payload.get("success") is False:
        raise ValueError(f"Assets API error: {payload.get('message')}")
    body = payload.get("data", payload)
    if isinstance(body, list):
        return body, payload.get("total")
    for key in ("items", "results", "assets", "docs", "data"):
        if isinstance(body.get(key), list):
            return body[key], body.get("total") or payload.get("total")
    raise ValueError("Assets API returned no list of records")


def fetch_api_records(url: str = "") -> list[dict]:
    """All cabinet records from the Assets API (``POST ?page=N&limit=M``, paginated)."""
    import requests

    url = url or config.ASSETS_API_URL
    headers = {"Accept": "application/json", "User-Agent": "magickhome-kitchen-generator"}
    if config.ASSETS_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.ASSETS_API_TOKEN}"
    records: list[dict] = []
    for page in range(1, config.ASSETS_API_MAX_PAGES + 1):
        params = {"page": page, "limit": config.ASSETS_API_PAGE_SIZE}
        if config.ASSETS_API_METHOD == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=config.ASSETS_API_TIMEOUT_S)
        else:
            resp = requests.post(url, params=params, json={}, headers=headers, timeout=config.ASSETS_API_TIMEOUT_S)
        resp.raise_for_status()
        batch, total = _page_records(resp.json())
        records.extend(batch)
        if not batch or len(batch) < config.ASSETS_API_PAGE_SIZE or (total is not None and len(records) >= total):
            break
    if not records:
        raise ValueError("Assets API returned zero records")
    return records


def _items_from_records(records: list[dict], source: str) -> list[CatalogItem]:
    items = []
    for r in records:
        aid = clean(r.get("aid") or r.get("aidAssetId") or r.get("name"))
        width, depth, height = _int(r.get("width")), _int(r.get("depth")), _int(r.get("height"))
        if not aid or not width or not depth or not height:
            continue
        items.append(
            CatalogItem(
                aid=aid,
                description=clean(r.get("description")) or aid,
                width=width,
                depth=depth,
                height=height,
                unit_family=clean(r.get("unitFamily")) or "",
                category=clean(r.get("category") or r.get("subCategory")) or "",
                door_type=clean(r.get("doorType")),
                cabinet_combination=clean(r.get("cabinetCombination")),
                shutter_count=_int(r.get("shutterCount")),
                drawer_count=_int(r.get("drawerCount")),
                open_box_count=_int(r.get("openBoxCount")),
                door_style=clean(r.get("doorStyle")),
                opening_direction=_opt(r.get("openingDirection")),
                handle_type=_opt(r.get("handleType")),
                internals=clean(r.get("internals")),
                is_accessory=_bool(r.get("isApplianceAccessory", r.get("isAccessory")), default=False),
                is_active=_bool(r.get("isActive"), default=True),
                source=source,
                api_record=r,
            )
        )
    if not items:
        raise ValueError("Assets API returned zero usable records")
    return items


def _write_cache(records: list[dict]) -> None:
    try:
        path = config.CATALOG_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fetchedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": config.ASSETS_API_URL, "items": records}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # cache is best-effort
        log.warning("could not write catalogue cache: %s", exc)


def _read_cache() -> Optional[tuple[list[dict], str]]:
    path = config.CATALOG_CACHE_PATH
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("items") or [], payload.get("fetchedAt", "?")


def load_catalog(prefer_api: bool = True) -> Catalog:
    """Load the full (unfiltered) catalogue: Assets API -> last API cache -> xlsx (cached per process)."""
    return _load_catalog(bool(prefer_api))


@lru_cache(maxsize=2)
def _load_catalog(prefer_api: bool) -> Catalog:
    if prefer_api and config.ASSETS_API_URL:
        try:
            records = fetch_api_records()
            items = _items_from_records(records, "api")
            _write_cache(records)
            log.info("catalogue: %d records from Assets API", len(items))
            return Catalog(items, source="api")
        except Exception as exc:  # network down, schema drift, ... -> offline
            log.warning("Assets API unavailable (%s)", exc)
            try:
                cached = _read_cache()
                if cached and cached[0]:
                    items = _items_from_records(cached[0], "api-cache")
                    log.info("catalogue: %d records from API cache (%s)", len(items), cached[1])
                    return Catalog(items, source="api-cache")
            except Exception as cexc:
                log.warning("API cache unreadable (%s)", cexc)
    items = _items_from_xlsx(config.CATALOG_XLSX_PATH)
    log.info("catalogue: %d records from %s", len(items), config.CATALOG_XLSX_PATH.name)
    return Catalog(items, source="xlsx")


def catalog_status() -> dict:
    """Source / size of the in-process catalogue without triggering a load."""
    if _load_catalog.cache_info().currsize == 0:
        return {"loaded": False}
    cat = load_catalog()
    return {"loaded": True, "source": cat.source, "skus": len(cat)}


def refresh_catalog() -> Catalog:
    """Drop the in-process cache and reload (e.g. after catalogue updates)."""
    _load_catalog.cache_clear()
    return load_catalog()


def load_filtered_catalog(
    door_style: str = config.DEFAULT_DOOR_STYLE,
    handle_type: str = config.DEFAULT_HANDLE_TYPE,
    prefer_api: bool = True,
) -> Catalog:
    """Active SKUs for one door style / handle type at the standard tier heights."""
    full = load_catalog(prefer_api)
    filtered = full.filter(
        door_style=door_style,
        handle_type=handle_type,
        active_only=True,
        tier_heights=config.TIER_HEIGHTS_MM,
        tier_depths=config.TIER_DEPTHS_MM,
    )
    if not filtered.items:
        styles = sorted({clean(i.door_style) for i in full.items if i.door_style})
        raise ValueError(f"No catalogue items for door_style={door_style!r}, handle_type={handle_type!r}. Styles: {styles}")
    return filtered


def available_door_styles(kitchen_ready: bool = True) -> list[str]:
    """Door styles in the catalogue; ``kitchen_ready`` keeps only styles with sink and hob units."""
    items = load_catalog().items
    styles = sorted({clean(i.door_style) for i in items if i.door_style})
    if not kitchen_ready:
        return styles
    return [s for s in styles if {CAT_SINK, CAT_HOB} <= {i.cat for i in items if style_key(i.door_style) == style_key(s)}]
