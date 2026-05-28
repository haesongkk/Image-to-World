"""Heuristic amodal mask completion.

This is the MVP for P0-1 (amodal). It does NOT use a learned model — the goal
is to validate the pipeline architecture so a real model (pix2gestalt /
Amodal3R / SAM-3D) can be swapped in by replacing only `_visible_to_amodal`.

Heuristic:
  1. Take per-object visible mask.
  2. Compute its convex hull. For convex-ish objects (pillows, cups, plants)
     this is a decent approximation when there's mild occlusion — the hull
     fills in concavities.
  3. Apply a small dilation (5% of bbox diagonal) so even fully-visible
     objects gain a tiny margin (helps the crop include the object boundary
     fully when bbox is tight).

Limits:
  - Only useful for mild occlusions on convex objects.
  - Won't predict the back of a heavily occluded chair etc. — that's the
    real-model job.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import AMODAL_COMPLETION_OUTPUT_DIR, MASK_POSTPROCESS_OUTPUT_DIR


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


def make_amodal_masks() -> None:
    src_dir = MASK_POSTPROCESS_OUTPUT_DIR
    out_dir = AMODAL_COMPLETION_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(src_dir.glob("object_*_mask.npy"))
    if not mask_files:
        raise RuntimeError(f"No visible masks found in: {src_dir}")

    viz_canvas = None
    for mp in mask_files:
        visible = np.load(mp)
        if visible.ndim == 3:
            visible = visible[..., 0]
        amodal = _visible_to_amodal(visible)
        out_path = out_dir / f"{mp.stem.replace('_mask', '_amodal_mask')}.npy"
        np.save(out_path, amodal.astype(np.uint8))
        if viz_canvas is None:
            viz_canvas = np.zeros((*amodal.shape, 3), dtype=np.uint8)
        # color the newly added (amodal - visible) area for QA viz
        added = (amodal > 0) & (visible == 0)
        viz_canvas[added] = (60, 180, 220)  # cyan-ish for amodal extension

    # overlay original visible masks for context
    color_canvas = np.zeros_like(viz_canvas)
    for idx, mp in enumerate(mask_files):
        m = np.load(mp)
        if m.ndim == 3:
            m = m[..., 0]
        c = (
            int((37 * (idx + 1)) % 256),
            int((97 * (idx + 1)) % 256),
            int((173 * (idx + 1)) % 256),
        )
        color_canvas[m > 0] = c

    composite = np.where(viz_canvas.sum(2, keepdims=True) > 0, viz_canvas, color_canvas)
    cv2.imwrite(str(out_dir / "amodal_viz.png"), composite)
