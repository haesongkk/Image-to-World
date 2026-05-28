"""Amodal completion: heuristic mask + LaMa-inpainted RGB.

Two outputs per object (saved alongside each other in
`AMODAL_COMPLETION_OUTPUT_DIR`):

  - `object_NNN_<class>_amodal_mask.npy`
      Per-pixel amodal mask via convex hull + small dilation. Placeholder
      for a real amodal predictor; useful for bbox extension.

  - `object_NNN_<class>_amodal_rgb.png`
      The source image, with **other objects** occluding this object's
      bbox region painted out by LaMa. Hunyuan3D-2 sees a "clean" crop
      where occluders are replaced with plausible continuation of the
      surroundings — substantially better mesh input than a raw bbox
      that includes occluder pixels.

Architecture so the real amodal-RGB model (pix2gestalt etc.) can later
swap in by replacing only `_inpaint_occluders()` below.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import (
    AMODAL_COMPLETION_OUTPUT_DIR,
    INSTANCE_SEGMENTATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
)


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
        print("amodal_mask: loaded simple-lama-inpainting")
    except Exception as e:
        print(f"amodal_mask: LaMa unavailable ({e}); RGB outputs will skip inpainting")
        _LAMA = None
    return _LAMA


def _visible_to_amodal(visible_mask: np.ndarray, dilate_frac: float = 0.05) -> np.ndarray:
    """Convex-hull + small dilation. Returns uint8 (0/1) mask."""
    binary = (visible_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    all_points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(all_points)
    amodal = np.ascontiguousarray(np.zeros_like(binary, dtype=np.uint8))
    cv2.fillPoly(amodal, [hull.astype(np.int32)], color=1)

    ys, xs = np.where(amodal > 0)
    if ys.size > 0 and dilate_frac > 0:
        bbox_diag = float(np.hypot(ys.ptp() + 1, xs.ptp() + 1))
        k = max(1, int(round(bbox_diag * dilate_frac)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        amodal = cv2.dilate(amodal, kernel, iterations=1)
    return amodal


def _inpaint_occluders(
    source_rgb: np.ndarray,
    object_mask: np.ndarray,
    occluder_mask: np.ndarray,
    bbox_pad_frac: float = 0.10,
) -> np.ndarray:
    """For one object, return an RGB crop where occluders inside its padded bbox
    are inpainted away by LaMa. Falls back to plain bbox crop if LaMa unavailable.
    """
    ys, xs = np.where(object_mask > 0)
    if ys.size == 0:
        return source_rgb.copy()
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    h, w = source_rgb.shape[:2]
    pad_h = max(1, int((y_max - y_min + 1) * bbox_pad_frac))
    pad_w = max(1, int((x_max - x_min + 1) * bbox_pad_frac))
    y0 = max(0, y_min - pad_h)
    y1 = min(h, y_max + 1 + pad_h)
    x0 = max(0, x_min - pad_w)
    x1 = min(w, x_max + 1 + pad_w)

    crop_rgb = source_rgb[y0:y1, x0:x1].copy()
    crop_occ = occluder_mask[y0:y1, x0:x1].copy()
    # Don't inpaint over the target object itself.
    crop_obj = object_mask[y0:y1, x0:x1]
    crop_occ = (crop_occ.astype(np.uint8) > 0).astype(np.uint8)
    crop_occ[crop_obj > 0] = 0

    if crop_occ.sum() == 0:
        return crop_rgb

    lama = _get_lama()
    if lama is None:
        return crop_rgb

    try:
        out = lama(
            Image.fromarray(crop_rgb),
            Image.fromarray((crop_occ * 255).astype(np.uint8)),
        )
        return np.array(out.convert("RGB"))
    except Exception as e:
        print(f"amodal_mask: LaMa inference failed ({e}); returning plain crop")
        return crop_rgb


def make_amodal_masks() -> None:
    src_dir = MASK_POSTPROCESS_OUTPUT_DIR
    out_dir = AMODAL_COMPLETION_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files_all = sorted(src_dir.glob("object_*_mask.npy"))
    if not mask_files_all:
        raise RuntimeError(f"No visible masks found in: {src_dir}")

    # Load source image (the input that was actually segmented).
    import json
    results_json = INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"
    with open(results_json, "r", encoding="utf-8") as f:
        infer = json.load(f)
    src_img_path = Path(infer.get("image_path", ""))
    if not src_img_path.is_absolute():
        src_img_path = src_img_path.resolve()
    source_rgb = np.array(Image.open(src_img_path).convert("RGB"))

    # Keep only masks that match the current input image resolution.
    H, W = source_rgb.shape[:2]
    mask_files: list[Path] = []
    for mp in mask_files_all:
        m0 = np.load(mp)
        if m0.ndim == 3:
            m0 = m0[..., 0]
        if m0.shape[:2] == (H, W):
            mask_files.append(mp)
    if not mask_files:
        raise RuntimeError(
            f"No masks matching source resolution ({H}x{W}) in: {src_dir}"
        )
    if len(mask_files) != len(mask_files_all):
        print(
            "amodal_mask: filtered stale masks by resolution "
            f"({len(mask_files)} / {len(mask_files_all)} kept)"
        )

    # Load all visible masks once so we can compute per-object occluder masks.
    visible_masks: list[np.ndarray] = []
    for mp in mask_files:
        m = np.load(mp)
        if m.ndim == 3:
            m = m[..., 0]
        visible_masks.append((m > 0).astype(np.uint8))

    viz_canvas = np.zeros((*visible_masks[0].shape, 3), dtype=np.uint8)
    color_canvas = np.zeros_like(viz_canvas)

    for idx, mp in enumerate(mask_files):
        visible = visible_masks[idx]
        amodal = _visible_to_amodal(visible)
        amodal_path = out_dir / f"{mp.stem.replace('_mask', '_amodal_mask')}.npy"
        np.save(amodal_path, amodal.astype(np.uint8))

        # Occluder mask = union of all OTHER objects' visible masks.
        occ = np.zeros_like(visible, dtype=np.uint8)
        for j, vm in enumerate(visible_masks):
            if j == idx:
                continue
            occ = np.maximum(occ, vm)
        # Also restrict to the (amodal) extension area — i.e., where this object's
        # amodal mask exists but visible doesn't — that's where occluders could
        # be hiding parts of the object.
        ext_mask = ((amodal > 0) & (visible == 0)).astype(np.uint8)
        # We inpaint anywhere in the padded bbox where occluders are AND
        # the visible mask is not present. Use occ AND not(visible).
        inpaint_mask = occ.copy()
        inpaint_mask[visible > 0] = 0

        amodal_rgb = _inpaint_occluders(source_rgb, visible, inpaint_mask)
        rgb_path = out_dir / f"{mp.stem.replace('_mask', '_amodal_rgb')}.png"
        Image.fromarray(amodal_rgb).save(rgb_path)

        # Update viz canvas
        added = (amodal > 0) & (visible == 0)
        viz_canvas[added] = (60, 180, 220)
        c = (
            int((37 * (idx + 1)) % 256),
            int((97 * (idx + 1)) % 256),
            int((173 * (idx + 1)) % 256),
        )
        color_canvas[visible > 0] = c
        print(f"amodal_mask: idx={idx:02d} cls={mp.stem.split('_', 2)[2].replace('_mask', '')} done")

    composite = np.where(viz_canvas.sum(2, keepdims=True) > 0, viz_canvas, color_canvas)
    cv2.imwrite(str(out_dir / "amodal_viz.png"), composite)
