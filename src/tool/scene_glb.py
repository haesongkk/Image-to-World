from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import trimesh


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


def make_scene_glb(image_path: Path) -> None:
    image_path = Path(image_path).resolve()
    project_root = Path(__file__).resolve().parent.parent.parent

    mesh_dir = project_root / "output" / "Hunyuan3D-2"
    transform_path = project_root / "output" / "fitted_transform" / "fitted_transform.npy"
    output_dir = project_root / "output" / "scene"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_dir.exists():
        raise RuntimeError(f"Hunyuan3D-2 output directory not found: {mesh_dir}")
    if not transform_path.exists():
        raise RuntimeError(f"Fitted transform file not found: {transform_path}")

    glb_paths = sorted(mesh_dir.glob("*_final_textured_mesh.glb"))
    if not glb_paths:
        raise RuntimeError(f"No final textured mesh GLB files found in: {mesh_dir}")

    transforms = np.load(transform_path).astype(np.float32)
    if transforms.ndim == 1:
        transforms = transforms.reshape(1, -1)
    if transforms.shape[1] < 9:
        raise RuntimeError(f"Invalid fitted_transform shape (expected [N, >=9]): {transforms.shape}")

    object_count = min(len(glb_paths), transforms.shape[0])
    if object_count == 0:
        raise RuntimeError("No objects to assemble into scene GLB.")

    scene = trimesh.Scene()
    for obj_idx in range(object_count):
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
        scene.add_geometry(mesh, node_name=f"object_{obj_idx:03d}", geom_name=f"object_{obj_idx:03d}")

    out_path = output_dir / f"{image_path.stem}_assembled.glb"
    scene.export(out_path)
    print(f"saved {out_path}")

