from __future__ import annotations

from pathlib import Path
import json
import math
import re

import numpy as np
import trimesh
from src.config import (
    MASK_POSTPROCESS_OUTPUT_DIR,
    MESH_GENERATION_OUTPUT_DIR,
    MESH_REMESH_OUTPUT_DIR,
    MESH_TEXTURING_OUTPUT_DIR,
    PROJECT_ROOT,
    SCENE_ASSEMBLY_OUTPUT_DIR,
    SCENE_PRECOMPUTE_OUTPUT_DIR,
)


_OBJECT_NAME_RE = re.compile(r"^(?:object_)?(\d+)_(.+?)(?:_(?:mask|points|shape_mesh|remeshed|remeshed_textured|textured|remesh|final_textured_mesh))?$")


def _load_stuff_keywords() -> list[str]:
    cfg = PROJECT_ROOT / "config" / "stuff_classes.json"
    if not cfg.exists():
        return []
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        keywords = data.get("stuff_keywords", [])
        return [str(k).lower() for k in keywords]
    except Exception as e:
        print(f"scene_glb: failed to read stuff_classes.json ({e}); proceeding without filter")
        return []


def _class_from_filename(stem: str) -> str:
    m = _OBJECT_NAME_RE.match(stem)
    if not m:
        return stem.lower()
    return m.group(2).lower()


def _compute_keep_mask(mask_dir: Path, stuff_keywords: list[str]) -> tuple[list[int], list[str]]:
    """Return (kept_indices, kept_class_names) by scanning per-object mask filenames in order."""
    mask_files = sorted(mask_dir.glob("object_*_mask.npy"))
    kept_indices: list[int] = []
    kept_classes: list[str] = []
    for i, mp in enumerate(mask_files):
        cname = _class_from_filename(mp.stem)
        if any(kw in cname for kw in stuff_keywords):
            print(f"scene_glb: filter stuff idx={i} class='{cname}'")
            continue
        kept_indices.append(i)
        kept_classes.append(cname)
    return kept_indices, kept_classes

_S_TO_P = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _load_mesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry in GLB: {glb_path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"Unsupported GLB load type: {type(loaded)}")


def _euler_xyz_deg_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def _rot_to_euler_xyz_deg(r: np.ndarray) -> tuple[float, float, float]:
    sy = float(-r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rz = 0.0
        rx = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["transforms"] if isinstance(data, dict) and "transforms" in data else data
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        packed = []
        for row in rows:
            t = row.get("translation", {})
            s = row.get("scale", {})
            r = row.get("rotation_deg", row.get("rotation", {}))
            t_s = np.array(
                [float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("z", 0.0))],
                dtype=np.float32,
            )
            s_s = np.array(
                [float(s.get("x", 1.0)), float(s.get("y", 1.0)), float(s.get("z", 1.0))],
                dtype=np.float32,
            )
            r_s = _euler_xyz_deg_to_rot(
                float(r.get("x", 0.0)),
                float(r.get("y", 0.0)),
                float(r.get("z", 0.0)),
            )

            t_p = _S_TO_P @ t_s
            s_p = np.array([s_s[1], s_s[2], s_s[0]], dtype=np.float32)  # (y, z, x)
            r_p = _S_TO_P @ r_s @ _S_TO_P.T
            rx_p, ry_p, rz_p = _rot_to_euler_xyz_deg(r_p)
            packed.append([
                float(t_p[0]), float(t_p[1]), float(t_p[2]),
                float(s_p[0]), float(s_p[1]), float(s_p[2]),
                float(rx_p), float(ry_p), float(rz_p),
            ])
        arr = np.asarray(packed, dtype=np.float32)
    else:
        arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def make_scene_glb(image_path: Path) -> None:
    image_path = Path(image_path).resolve()
    remesh_dir = MESH_REMESH_OUTPUT_DIR
    textured_dir = MESH_TEXTURING_OUTPUT_DIR
    mesh_dir = MESH_GENERATION_OUTPUT_DIR
    # Priority: fitted_transform (refined) > fitted_transform_debug > raw_transform (initial).
    project_root = Path(__file__).resolve().parent.parent.parent
    transform_candidates = [
        project_root / "output" / "fitted_transform" / "fitted_transform.json",
        project_root / "output" / "fitted_transform_debug" / image_path.stem / "fitted_transform.json",
        project_root / "output" / "fitted_transform_debug" / image_path.stem / "simple" / "fitted_transform.json",
        SCENE_PRECOMPUTE_OUTPUT_DIR / "raw_transform.json",
    ]
    transform_path = next((p for p in transform_candidates if p.exists()), transform_candidates[-1])
    print(f"scene_glb: using transforms from {transform_path}")
    output_dir = SCENE_ASSEMBLY_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not transform_path.exists():
        raise RuntimeError(f"Transform file not found: {transform_path}")

    glb_paths = (
        sorted(textured_dir.glob("*_remeshed_textured.glb"))
        or sorted(textured_dir.glob("*_textured.glb"))
        or sorted(remesh_dir.glob("*_remeshed.glb"))
        or sorted(mesh_dir.glob("*_remeshed.glb"))
        or sorted(mesh_dir.glob("*_remesh.glb"))
        or sorted(mesh_dir.glob("*_shape_mesh.glb"))
        or sorted(mesh_dir.glob("*_final_textured_mesh.glb"))
    )
    if not glb_paths:
        raise RuntimeError(
            f"No mesh GLB files found in remesh/Hunyuan outputs: {remesh_dir}, {mesh_dir}"
        )

    transforms = _load_transform_json(transform_path)
    if transforms.shape[1] < 9:
        raise RuntimeError(f"Invalid transform shape (expected [N, >=9]): {transforms.shape}")

    object_count = min(len(glb_paths), transforms.shape[0])
    if object_count == 0:
        raise RuntimeError("No objects to assemble into scene GLB.")

    # Filter stuff classes (e.g. floor, walls, background "living_room") so they
    # don't appear as separate movable meshes. Their Hunyuan3D output is
    # typically degenerate and their visible-AABB scale is wildly oversized.
    stuff_keywords = _load_stuff_keywords()
    if stuff_keywords:
        kept, kept_classes = _compute_keep_mask(MASK_POSTPROCESS_OUTPUT_DIR, stuff_keywords)
        kept = [i for i in kept if i < object_count]
        if not kept:
            print("scene_glb: stuff filter removed all objects; falling back to unfiltered")
            kept = list(range(object_count))
            kept_classes = [None] * object_count
    else:
        kept = list(range(object_count))
        kept_classes = [None] * object_count

    print(f"scene_glb: assembling {len(kept)} / {object_count} objects (stuff filter)")

    scene = trimesh.Scene()
    for slot, obj_idx in enumerate(kept):
        tr = transforms[obj_idx]
        tx, ty, tz = map(float, tr[0:3])
        sx, sy, sz = np.maximum(tr[3:6], 1e-6).astype(np.float32)
        rx, ry, rz = map(float, tr[6:9])

        mesh = _load_mesh(glb_paths[obj_idx]).copy()
        rot = _euler_xyz_deg_to_rot(rx, ry, rz)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        verts_world = (verts * np.array([sx, sy, sz], dtype=np.float32)) @ rot.T + np.array(
            [tx, ty, tz], dtype=np.float32
        )
        mesh.vertices = verts_world
        cname = kept_classes[slot] if slot < len(kept_classes) and kept_classes[slot] else f"object_{obj_idx:03d}"
        node_name = f"object_{obj_idx:03d}_{cname}" if cname and not cname.startswith("object_") else f"object_{obj_idx:03d}"
        scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)

    # Add fitted floor plane (M3 lite) so the scene has ground reference.
    try:
        from src.tool.floor_plane import build_floor_mesh
        floor_mesh = build_floor_mesh()
        if floor_mesh is not None:
            scene.add_geometry(floor_mesh, node_name="background_floor", geom_name="background_floor")
            print("scene_glb: added background_floor plane mesh")
    except Exception as e:
        print(f"scene_glb: floor plane skipped ({e})")

    out_path = output_dir / f"{image_path.stem}_assembled.glb"
    scene.export(out_path)
    print(f"saved {out_path}")
