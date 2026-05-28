from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import MASK_POSTPROCESS_OUTPUT_DIR, PROJECT_ROOT


_OBJECT_NAME_RE = re.compile(
    r"^(?:object_)?(\d+)_(.+?)"
    r"(?:_(?:mask|points|shape_mesh|remeshed|remeshed_textured|textured|remesh|final_textured_mesh))?$"
)


def load_stuff_keywords() -> list[str]:
    cfg = PROJECT_ROOT / "config" / "stuff_classes.json"
    if not cfg.exists():
        return []
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        keywords = data.get("stuff_keywords", [])
        return [str(k).lower() for k in keywords]
    except Exception as e:
        print(f"stuff_filter: failed to read stuff_classes.json ({e}); proceeding without filter")
        return []


def load_floor_resting_keywords() -> list[str]:
    cfg = PROJECT_ROOT / "config" / "floor_resting_classes.json"
    if not cfg.exists():
        return []
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(k).lower() for k in data.get("floor_resting_keywords", [])]
    except Exception as e:
        print(f"floor_resting: failed to read floor_resting_classes.json ({e})")
        return []


def class_from_filename(stem: str) -> str:
    m = _OBJECT_NAME_RE.match(stem)
    if not m:
        return stem.lower()
    return m.group(2).lower()


def compute_keep_indices(
    mask_dir: Path | None = None,
    stuff_keywords: list[str] | None = None,
    verbose: bool = True,
) -> list[int]:
    """Return indices into the sorted mask file list that should be kept (non-stuff)."""
    mask_dir = mask_dir or MASK_POSTPROCESS_OUTPUT_DIR
    if stuff_keywords is None:
        stuff_keywords = load_stuff_keywords()
    if not stuff_keywords:
        return list(range(len(sorted(Path(mask_dir).glob("object_*_mask.npy")))))

    mask_files = sorted(Path(mask_dir).glob("object_*_mask.npy"))
    kept: list[int] = []
    for i, mp in enumerate(mask_files):
        cname = class_from_filename(mp.stem)
        if any(kw in cname for kw in stuff_keywords):
            if verbose:
                print(f"stuff_filter: drop idx={i} class='{cname}'")
            continue
        kept.append(i)
    return kept
