"""Clean-background inpainting.

Produces a single RGB image where every MOVABLE object has been painted
out by LaMa, leaving the (now-occlusion-free) background. This image is
the texture source used by floor_plane / wall_plane meshes so a moved
object no longer reveals a black hole behind it.

Stuff classes (wall / floor / counter / ceiling …) are NOT painted out —
those *are* the background and we want to keep them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import (
    BACKGROUND_INPAINT_OUTPUT_DIR,
    INSTANCE_SEGMENTATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
)
from src.tool.stuff_filter import class_from_filename, load_stuff_keywords


_LAMA = None
_LAMA_ATTEMPTED = False


def _get_lama():
    global _LAMA, _LAMA_ATTEMPTED
    if _LAMA_ATTEMPTED:
        return _LAMA
    _LAMA_ATTEMPTED = True
    try:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
        print("background_inpaint: loaded simple-lama-inpainting")
    except Exception as e:
        print(f"background_inpaint: LaMa unavailable ({e})")
        _LAMA = None
    return _LAMA


def _movable_union_mask(mask_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    stuff_kws = load_stuff_keywords()
    mask_files = sorted(mask_dir.glob("object_*_mask.npy"))
    if not mask_files:
        raise RuntimeError(f"No masks found in: {mask_dir}")

    union: np.ndarray | None = None
    movable_classes: list[str] = []
    stuff_classes: list[str] = []
    for mp in mask_files:
        cname = class_from_filename(mp.stem)
        is_stuff = any(kw in cname for kw in stuff_kws)
        if is_stuff:
            stuff_classes.append(cname)
            continue
        movable_classes.append(cname)
        m = np.load(mp)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8)
        union = m if union is None else np.maximum(union, m)

    if union is None:
        raise RuntimeError("No movable masks (all classified as stuff). Inpaint skipped.")
    return union, movable_classes, stuff_classes


def _dilate_mask(mask: np.ndarray, frac: float = 0.02) -> np.ndarray:
    h, w = mask.shape[:2]
    k = max(3, int(round(min(h, w) * frac)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def _load_source_rgb() -> np.ndarray:
    results_json = INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"
    with open(results_json, "r", encoding="utf-8") as f:
        infer = json.load(f)
    src_img_path = Path(infer.get("image_path", "")).resolve()
    if not src_img_path.exists():
        raise FileNotFoundError(f"Source image referenced by segmentation results not found: {src_img_path}")
    return np.array(Image.open(src_img_path).convert("RGB"))


def make_clean_background() -> Path:
    BACKGROUND_INPAINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rgb = _load_source_rgb()
    union, movable_classes, stuff_classes = _movable_union_mask(MASK_POSTPROCESS_OUTPUT_DIR)
    inpaint_mask = _dilate_mask(union, frac=0.015)

    print(
        f"background_inpaint: inpainting {len(movable_classes)} movable masks "
        f"(stuff kept: {sorted(set(stuff_classes))})"
    )

    out_path = BACKGROUND_INPAINT_OUTPUT_DIR / "clean_background.png"
    union_viz_path = BACKGROUND_INPAINT_OUTPUT_DIR / "inpaint_mask.png"
    Image.fromarray((inpaint_mask * 255).astype(np.uint8)).save(union_viz_path)

    lama = _get_lama()
    if lama is None:
        print("background_inpaint: LaMa unavailable; writing source RGB unmodified")
        Image.fromarray(source_rgb).save(out_path)
        return out_path

    try:
        h, w = source_rgb.shape[:2]
        # LaMa expects multiples of 8 typically; resize down for memory if huge.
        max_side = 1536
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            src_small = cv2.resize(source_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(inpaint_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            src_small = source_rgb
            mask_small = inpaint_mask
        out_pil = lama(
            Image.fromarray(src_small),
            Image.fromarray((mask_small * 255).astype(np.uint8)),
        )
        out_rgb = np.array(out_pil.convert("RGB"))
        if out_rgb.shape[:2] != (h, w):
            out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        Image.fromarray(out_rgb).save(out_path)
        print(f"background_inpaint: wrote {out_path}")
    except Exception as e:
        print(f"background_inpaint: LaMa inference failed ({e}); writing source RGB unmodified")
        Image.fromarray(source_rgb).save(out_path)

    return out_path
