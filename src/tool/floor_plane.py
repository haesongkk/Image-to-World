"""Floor + wall plane fitting and textured-quad mesh construction.

Two plane meshes are produced from stuff (floor/counter/wall) pointclouds
and added to the assembled scene by `scene_glb.py`:

  - `build_floor_mesh()`: horizontal support plane at median z of
    floor/carpet/counter/tabletop pointclouds.
  - `build_wall_mesh()`: vertical back-wall plane fitted from wall/tile/
    backsplash pointclouds.

When `output/background_inpaint/clean_background.png` exists, both meshes
are textured by projecting their corners through the input camera
intrinsics into the inpainted image — so moving an object away from its
original location reveals plausible background (the spot where the
object used to be has been LaMa-filled).

Frame conventions:
  - storage (npy pointclouds): x=front (negative=forward), y=right, z=up
  - GLB / pytorch3d: x=right, y=up, z=forward (negative=in-front-of-camera)
  - Conversion (storage -> GLB): x_g = y_s, y_g = z_s, z_g = x_s
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from src.config import (
    BACKGROUND_INPAINT_OUTPUT_DIR,
    CAMERA_ESTIMATION_OUTPUT_DIR,
    INSTANCE_SEGMENTATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
    PROJECT_ROOT,
    SCENE_PRECOMPUTE_OUTPUT_DIR,
)


def _load_floor_keywords() -> list[str]:
    return [
        "floor", "carpet", "rug", "ground",
        "counter", "countertop", "counter_top",
        "tabletop", "table_top", "worktop", "kitchen_counter",
    ]


def _load_wall_keywords() -> list[str]:
    return [
        "wall", "tile_wall", "tiled_wall", "tile", "backsplash",
    ]


def _class_from_name(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem.lower()
    return "_".join(parts[2:-2]).lower() if parts[-2] == "mask" else "_".join(parts[2:]).lower()


def _find_pointclouds(keywords: list[str], pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> list[Path]:
    out: list[Path] = []
    for p in sorted(pc_dir.glob("object_*_points.npy")):
        name = p.stem.lower()
        if any(k in name for k in keywords):
            out.append(p)
    return out


def find_floor_pointclouds(pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> list[Path]:
    return _find_pointclouds(_load_floor_keywords(), pc_dir)


def find_wall_pointclouds(pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> list[Path]:
    return _find_pointclouds(_load_wall_keywords(), pc_dir)


def fit_floor(pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> dict | None:
    floor_paths = find_floor_pointclouds(pc_dir)
    if not floor_paths:
        return None
    all_pts = np.concatenate([np.load(p) for p in floor_paths], axis=0)
    floor_z = float(np.median(all_pts[:, 2]))

    scene_pts = np.concatenate(
        [np.load(p) for p in sorted(pc_dir.glob("object_*_points.npy"))],
        axis=0,
    )
    x_min, x_max = float(scene_pts[:, 0].min()), float(scene_pts[:, 0].max())
    y_min, y_max = float(scene_pts[:, 1].min()), float(scene_pts[:, 1].max())
    # Generous margins so the floor extends far enough to cover the camera
    # frustum even when the fitted camera ends up far from objects.
    margin_x = 1.0 * (x_max - x_min + 1e-6)
    margin_y = 1.0 * (y_max - y_min + 1e-6)

    return {
        "floor_z": floor_z,
        "x_range": [x_min - margin_x, x_max + margin_x],
        "y_range": [y_min - margin_y, y_max + margin_y],
        "source_paths": [str(p) for p in floor_paths],
    }


def fit_wall(pc_dir: Path = SCENE_PRECOMPUTE_OUTPUT_DIR) -> dict | None:
    """Fit a vertical (axis-aligned) back wall.

    Storage frame: x=front, smaller x = farther. Preferred source: pointclouds
    labelled wall/tile/backsplash. Fallback (heuristic): if no wall mask was
    detected (RAM often misses surfaces), place a virtual wall behind all
    scene points at min(x) - small_margin. This keeps the backdrop demo
    working on scenes where only objects + floor were detected.
    """
    wall_paths = find_wall_pointclouds(pc_dir)
    if wall_paths:
        all_pts = np.concatenate([np.load(p) for p in wall_paths], axis=0)
        wall_x = float(np.median(all_pts[:, 0]))
    else:
        scene_paths = sorted(pc_dir.glob("object_*_points.npy"))
        if not scene_paths:
            return None
        all_pts = np.concatenate([np.load(p) for p in scene_paths], axis=0)
        # No explicit wall points: place a virtual backdrop clearly behind objects.
        # Storage-x: smaller = farther from camera.
        x_min = float(all_pts[:, 0].min())
        x_max = float(all_pts[:, 0].max())
        x_span = max(1e-6, x_max - x_min)
        wall_x = x_min - 0.6 * x_span
        # Keep the fallback wall away from the object cluster even in shallow scenes.
        wall_x = min(wall_x, -2.0)
        print(
            "fit_wall: no wall mask; placing virtual backdrop behind scene "
            f"(storage_x={wall_x:.3f}, x_min={x_min:.3f}, x_max={x_max:.3f})"
        )

    scene_pts = np.concatenate(
        [np.load(p) for p in sorted(pc_dir.glob("object_*_points.npy"))],
        axis=0,
    )
    y_min, y_max = float(scene_pts[:, 1].min()), float(scene_pts[:, 1].max())
    z_min, z_max = float(scene_pts[:, 2].min()), float(scene_pts[:, 2].max())
    # Wall margins: same generous philosophy as floor — fill camera frame
    # even when camera ends up further than the object cluster's extent.
    margin_y = 1.0 * (y_max - y_min + 1e-6)
    margin_z = 1.0 * (z_max - z_min + 1e-6)

    return {
        "wall_x": wall_x,
        "y_range": [y_min - margin_y, y_max + margin_y],
        "z_range": [z_min - margin_z, z_max + margin_z],
        "source_paths": [str(p) for p in wall_paths],
    }


def _sample_avg_color(keywords: list[str], fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        results_json = INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"
        with open(results_json, "r", encoding="utf-8") as f:
            infer = json.load(f)
        src_path = Path(infer.get("image_path", "")).resolve()
        if not src_path.exists():
            return fallback
        src = np.array(Image.open(src_path).convert("RGB"))
    except Exception:
        return fallback

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
        return fallback
    all_px = np.concatenate(pixels, axis=0)
    return tuple(int(c) for c in np.mean(all_px, axis=0))


def _load_camera_intrinsics() -> tuple[float, int, int] | None:
    """Return (vfov_deg, image_w, image_h) or None if camera info missing."""
    try:
        results_json = INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"
        with open(results_json, "r", encoding="utf-8") as f:
            infer = json.load(f)
        src_path = Path(infer.get("image_path", "")).resolve()
        img = Image.open(src_path)
        w, h = img.size
    except Exception:
        return None

    cam_files = sorted(CAMERA_ESTIMATION_OUTPUT_DIR.glob("*_perspective_fields.json"))
    if not cam_files:
        return None
    try:
        with open(cam_files[0], "r", encoding="utf-8") as f:
            cam = json.load(f)
        vfov = float(cam.get("pred_general_vfov") or cam.get("pred_vfov") or 60.0)
    except Exception:
        return None
    return vfov, w, h


def _project_to_uv(verts_glb: np.ndarray, vfov_deg: float, w: int, h: int) -> np.ndarray:
    """Project GLB-frame points to image-space normalized UV coords.

    GLB convention: camera at origin looking down -z, y up. A point in
    front of the camera has z<0. Returns shape [N, 2] in [0,1] approximately
    (may extend slightly outside for parts beyond the frame).
    """
    aspect = w / float(h)
    half_v = math.radians(vfov_deg) / 2.0
    tan_v = math.tan(half_v)
    tan_h = tan_v * aspect

    out = np.zeros((verts_glb.shape[0], 2), dtype=np.float32)
    for i, v in enumerate(verts_glb):
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        depth = -z
        if depth <= 0.3:
            # Point at or near the camera plane: projection diverges.
            # Clamp to a sane near distance so UV stays bounded.
            depth = 0.3
        u_norm = (x / depth) / tan_h
        v_norm = (y / depth) / tan_v
        u = (u_norm + 1.0) * 0.5
        v_img = (1.0 - v_norm) * 0.5
        # Clamp UVs to [0,1] so vertices outside the camera frustum sample
        # the image edge instead of repeating/tiling the texture.
        u = max(0.0, min(1.0, u))
        v_img = max(0.0, min(1.0, v_img))
        out[i, 0] = u
        out[i, 1] = 1.0 - v_img
    return out


def _load_background_image() -> Path | None:
    p = BACKGROUND_INPAINT_OUTPUT_DIR / "clean_background.png"
    return p if p.exists() else None


def _make_textured_quad(
    verts: np.ndarray,
    faces: np.ndarray,
    fallback_color: tuple[int, int, int],
    label: str,
) -> trimesh.Trimesh:
    """Build a textured trimesh from a quad. If clean background image and
    camera intrinsics are available, use projection-based UVs; otherwise
    fall back to solid face color.
    """
    bg_path = _load_background_image()
    cam = _load_camera_intrinsics()
    if bg_path is None or cam is None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.face_colors = np.array(
            [[*fallback_color, 255]] * len(faces), dtype=np.uint8
        )
        print(f"{label}: untextured (bg={bg_path}, cam={cam})")
        return mesh

    vfov_deg, w, h = cam
    uvs = _project_to_uv(verts, vfov_deg, w, h)

    try:
        tex_img = Image.open(bg_path).convert("RGB")
        # SimpleMaterial defaults its `diffuse` color to ~40% gray, which on
        # GLB export becomes baseColorFactor=(102,102,102) and DARKENS the
        # texture by 40%. Force pure white to display the texture as-is.
        material = trimesh.visual.material.SimpleMaterial(
            image=tex_img,
            diffuse=[255, 255, 255, 255],
            ambient=[255, 255, 255, 255],
        )
        visual = trimesh.visual.TextureVisuals(uv=uvs, material=material, image=tex_img)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
        print(
            f"{label}: textured ({bg_path.name}, vfov={vfov_deg:.1f}, "
            f"uv_range=[{uvs.min():.2f}, {uvs.max():.2f}])"
        )
        return mesh
    except Exception as e:
        print(f"{label}: textured-mesh build failed ({e}); falling back to solid color")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.face_colors = np.array(
            [[*fallback_color, 255]] * len(faces), dtype=np.uint8
        )
        return mesh


def build_floor_mesh() -> trimesh.Trimesh | None:
    fit = fit_floor()
    if fit is None:
        return None

    fz = fit["floor_z"]
    x0, x1 = fit["x_range"]
    y0, y1 = fit["y_range"]

    glb_y = fz
    glb_x_min, glb_x_max = y0, y1
    glb_z_min, glb_z_max = x0, x1
    # Cap the camera-near edge so projection doesn't diverge. Keep a small
    # gap (-0.2 m in GLB z) between the floor's near edge and the camera.
    glb_z_max = min(max(glb_z_max, -0.5), -0.2)
    # Push the far edge well behind the scene so even a high camera sees floor
    # all the way to the back wall.
    glb_z_min = min(glb_z_min, -3.0)

    verts = np.array(
        [
            [glb_x_min, glb_y, glb_z_min],
            [glb_x_max, glb_y, glb_z_min],
            [glb_x_max, glb_y, glb_z_max],
            [glb_x_min, glb_y, glb_z_max],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]],
        dtype=np.int64,
    )
    color = _sample_avg_color(_load_floor_keywords(), (140, 130, 120))
    return _make_textured_quad(verts, faces, color, label="floor_plane")


def build_wall_mesh() -> trimesh.Trimesh | None:
    fit = fit_wall()
    if fit is None:
        return None

    wx_storage = fit["wall_x"]
    y_min, y_max = fit["y_range"]
    z_min, z_max = fit["z_range"]

    # Storage frame -> GLB: x_g = y_s, y_g = z_s, z_g = x_s.
    # Wall: storage_x = wx_storage (constant), so GLB z = wx_storage.
    # That fixes one GLB dim. The wall extends along storage y (->GLB x)
    # and storage z (->GLB y).
    glb_z = wx_storage
    glb_x_min, glb_x_max = y_min, y_max
    glb_y_min, glb_y_max = z_min, z_max
    # Demo tuning: raise wall slightly so the backsplash sits behind countertop
    # rather than cutting through object silhouettes in generic viewers.
    wall_y_offset = 0.18
    glb_y_min += wall_y_offset
    glb_y_max += wall_y_offset

    verts = np.array(
        [
            [glb_x_min, glb_y_min, glb_z],
            [glb_x_max, glb_y_min, glb_z],
            [glb_x_max, glb_y_max, glb_z],
            [glb_x_min, glb_y_max, glb_z],
        ],
        dtype=np.float32,
    )
    # Two windings (visible from both sides).
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
        dtype=np.int64,
    )
    color = _sample_avg_color(_load_wall_keywords(), (220, 220, 220))
    return _make_textured_quad(verts, faces, color, label="wall_plane")
