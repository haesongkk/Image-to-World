"""Fit a horizontal floor plane from stuff (floor/carpet) pointclouds and
build a simple textured quad mesh to add into the assembled scene.

Approach:
  - Find pointcloud .npy files whose names match "floor" / "carpet" /
    similar stuff keywords already present in `config/stuff_classes.json`.
  - Take the median z of all such points → floor plane height.
  - Compute scene xy extent from ALL object pointclouds (with margin).
  - Build a thin quad mesh at z=floor_z. Color = mean RGB sampled from the
    input image inside the floor visible mask (fallback: neutral grey).
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import trimesh
from PIL import Image

from src.config import (
    INSTANCE_SEGMENTATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
    PROJECT_ROOT,
    SCENE_PRECOMPUTE_OUTPUT_DIR,
)


def _load_floor_keywords() -> list[str]:
    """Subset of stuff_classes that should become floor planes."""
    return ["floor", "carpet", "rug", "ground"]


def _class_from_name(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem.lower()
    return "_".join(parts[2:-2]).lower() if parts[-2] == "mask" else "_".join(parts[2:]).lower()


def find_floor_pointclouds(pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> list[Path]:
    keywords = _load_floor_keywords()
    out: list[Path] = []
    for p in sorted(pc_dir.glob("object_*_points.npy")):
        name = p.stem.lower()
        if any(k in name for k in keywords):
            out.append(p)
    return out


def fit_floor(pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> dict | None:
    floor_paths = find_floor_pointclouds(pc_dir)
    if not floor_paths:
        return None
    all_pts = np.concatenate([np.load(p) for p in floor_paths], axis=0)
    floor_z = float(np.median(all_pts[:, 2]))

    # Scene xy extent from all object pointclouds (not just floor).
    scene_pts = np.concatenate(
        [np.load(p) for p in sorted(pc_dir.glob("object_*_points.npy"))],
        axis=0,
    )
    x_min, x_max = float(scene_pts[:, 0].min()), float(scene_pts[:, 0].max())
    y_min, y_max = float(scene_pts[:, 1].min()), float(scene_pts[:, 1].max())
    margin_x = 0.25 * (x_max - x_min + 1e-6)
    margin_y = 0.25 * (y_max - y_min + 1e-6)

    return {
        "floor_z": floor_z,
        "x_range": [x_min - margin_x, x_max + margin_x],
        "y_range": [y_min - margin_y, y_max + margin_y],
        "source_paths": [str(p) for p in floor_paths],
    }


def _sample_floor_color() -> tuple[int, int, int]:
    """Average input-image RGB inside the floor/carpet visible masks."""
    try:
        results_json = INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"
        with open(results_json, "r", encoding="utf-8") as f:
            infer = json.load(f)
        src_path = Path(infer.get("image_path", "")).resolve()
        if not src_path.exists():
            return (140, 130, 120)
        src = np.array(Image.open(src_path).convert("RGB"))
    except Exception:
        return (140, 130, 120)

    keywords = _load_floor_keywords()
    pixels = []
    for mp in sorted(MASK_POSTPROCESS_OUTPUT_DIR.glob("object_*_mask.npy")):
        name = mp.stem.lower()
        if not any(k in name for k in keywords):
            continue
        m = np.load(mp)
        if m.ndim == 3:
            m = m[..., 0]
        m = m > 0
        if m.sum() == 0:
            continue
        pixels.append(src[m])
    if not pixels:
        return (140, 130, 120)
    all_px = np.concatenate(pixels, axis=0)
    return tuple(int(c) for c in np.mean(all_px, axis=0))


def build_floor_mesh() -> trimesh.Trimesh | None:
    """Return a small thin box mesh sitting on the fitted floor plane in
    pytorch3d/GLB coords (x-right, y-up, -z forward), ready to add to a
    trimesh.Scene exported as GLB.
    """
    fit = fit_floor()
    if fit is None:
        return None

    # Storage frame: x-front (negative=forward, see pointcloud.py).
    # GLB frame conversion via _S_TO_P in scene_glb: x_p=y_s, y_p=z_s, z_p=x_s.
    # So storage in-front objects (x_s < 0) end up at z_p < 0, matching
    # OpenGL "camera looks down -z".
    fz = fit["floor_z"]
    x0, x1 = fit["x_range"]
    y0, y1 = fit["y_range"]

    glb_y = fz                          # storage z (up) → GLB y (up)
    glb_x_min, glb_x_max = y0, y1       # storage y (right) → GLB x (right)
    glb_z_min, glb_z_max = x0, x1       # storage x → GLB z (no negation)

    # Make a flat rectangle (two triangles) at glb_y, spanning [glb_x_min..glb_x_max] x [glb_z_min..glb_z_max].
    verts = np.array(
        [
            [glb_x_min, glb_y, glb_z_min],
            [glb_x_max, glb_y, glb_z_min],
            [glb_x_max, glb_y, glb_z_max],
            [glb_x_min, glb_y, glb_z_max],
        ],
        dtype=np.float32,
    )
    # Reverse winding so the face normal points +y (up) — top of the floor.
    # Add both windings so the floor is visible from above and below if culling
    # is on.
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )

    color = _sample_floor_color()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.face_colors = np.array([[*color, 255]] * len(faces), dtype=np.uint8)
    print(
        f"floor_plane: y={glb_y:.2f}, "
        f"x=[{glb_x_min:.1f}, {glb_x_max:.1f}], "
        f"z=[{glb_z_min:.1f}, {glb_z_max:.1f}], color={color}"
    )
    return mesh
