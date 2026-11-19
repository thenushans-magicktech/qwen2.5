"""Harvest material/texture presets from production reference scenes.

The offline catalogue (Finalised_Asset_Catalogue_Sheet.xlsx) carries SKU geometry
but no material data. Production scenes exported by the MagickHome designer do:
every cabinet carries its shutter texture set, every fixture its material slot
textures, and appliances their full asset records. This script copies those
blocks into ``data/material_presets.json`` so generated scenes render with the
same finishes as the reference scenes.

Usage:
    python extract_presets.py magickhomes-scene_2026-11-18_10-38-22.json \
        [magickhomes-scene_2026-11-18_10-24-05.json ...] [--out data/material_presets.json]

The first scene wins whenever two scenes define the same preset.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import config


def _first(store: dict, key: str, value) -> None:
    if value and key not in store:
        store[key] = value


def extract(scene_paths: list[Path]) -> dict:
    shutters: dict[str, list] = {}
    fixtures: dict[str, list] = {}
    colors: dict[str, str] = {}
    appliances: dict[str, dict] = {}

    for path in scene_paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        for node in doc["scene"]["nodes"].values():
            if node.get("type") != "item":
                continue
            asset = node.get("asset") or {}
            meta = node.get("metadata") or {}
            category = asset.get("category")
            if category == "kitchen":
                level = (node.get("placement") or {}).get("level")
                _first(shutters, level, meta.get("textures"))
            elif category == "fixture":
                kind = meta.get("fixtureKind")
                level = meta.get("level")
                key = f"{kind}:{level}" if level else kind
                _first(fixtures, key, meta.get("textures"))
                if meta.get("color"):
                    colors.setdefault(kind, meta["color"])
            elif category == "appliance":
                kind = meta.get("applianceKind")
                if kind and kind not in appliances:
                    m = copy.deepcopy(meta)
                    # Host links are scene specific; the hydrator rewrites them.
                    for k in ("hostCabinetId", "designSource"):
                        m.pop(k, None)
                    appliances[kind] = {"metadata": m, "asset": copy.deepcopy(asset)}

    # Lofts share the wall-unit shutter finish when no loft sample exists.
    if "loft" not in shutters and "wall" in shutters:
        shutters["loft"] = shutters["wall"]
    if "tall" not in shutters and "base" in shutters:
        shutters["tall"] = shutters["base"]

    return {
        "sourceScenes": [Path(p).name for p in scene_paths],
        "shutterTextures": shutters,
        "fixtureTextures": fixtures,
        "fixtureColors": colors,
        "appliances": appliances,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=config.MATERIAL_PRESETS_PATH)
    args = ap.parse_args()
    presets = extract(args.scenes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(presets, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {args.out}: shutter tiers={sorted(presets['shutterTextures'])}, "
        f"fixture slots={sorted(presets['fixtureTextures'])}, appliances={sorted(presets['appliances'])}"
    )


if __name__ == "__main__":
    main()
