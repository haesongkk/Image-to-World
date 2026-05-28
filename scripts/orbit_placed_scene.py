"""Orbital camera around the mask-placed 3D scene.

Renders the 3 movable Hunyuan-Paint meshes (placed using the same mask-driven
transforms as `place_meshes_to_mask.py`) from a camera that orbits the scene.
Background stays neutral (no LaMa here — LaMa is only valid from the input
camera angle).

Output: output/demo/raw_image0/scene_orbit.gif
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
import imageio.v2 as imageio
import pyrender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tool.stuff_filter import (  # noqa: E402
    class_from_filename,
    load_stuff_keywords,
)


def _build_items(mask_dir: Path, mesh_dir: Path, camera_json: Path,
                 bg_w: int, bg_h: int, target_w: int, target_h: int, depth: float):
    """Replicate place_meshes_to_mask placement to get the same scene."""
    with open(camera_json) as f:
        pf = json.load(f)
    vfov_deg = float(pf.get("pred_general_vfov", pf.get("pred_vfov", 50.0)))
    fy = (target_h * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)
    fx = fy
    cx = target_w * 0.5; cy = target_h * 0.5
    Z = -abs(depth)
    sx = target_w / bg_w
    sy = target_h / bg_h

    stuff_kws = load_stuff_keywords()
    movable = []
    for mp in sorted(mask_dir.glob("object_*_mask.npy")):
        cname = class_from_filename(mp.stem)
        if any(kw in cname for kw in stuff_kws):
            continue
        idx = int(mp.stem.split("_")[1])
        m = np.load(mp)
        if m.ndim == 3: m = m[..., 0]
        m = (m > 0).astype(np.uint8)
        if m.sum() == 0: continue
        mr = np.array(Image.fromarray((m*255).astype(np.uint8)).resize((target_w, target_h), Image.NEAREST))
        mr = (mr > 127).astype(np.uint8)
        ys, xs = np.where(mr > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        movable.append({"idx": idx, "class": cname, "bbox": (x0, y0, x1, y1)})

    mesh_files = sorted(mesh_dir.glob("*_remeshed_textured.glb"))

    def _find(idx):
        for mp in mesh_files:
            if mp.name.startswith(f"{idx:03d}_"):
                return mp
        return None

    items = []
    for m in movable:
        mp = _find(m["idx"])
        if mp is None: continue
        scene = trimesh.load(mp, force="scene")
        mesh = next(iter(scene.geometry.values()))
        bb = mesh.bounds
        c_center = (bb[0] + bb[1]) * 0.5
        c_ex_x = float(bb[1][0] - bb[0][0])
        c_ex_y = float(bb[1][1] - bb[0][1])
        mx0, my0, mx1, my1 = m["bbox"]
        u_c = (mx0 + mx1) * 0.5; v_c = (my0 + my1) * 0.5
        tw = my1 - my0; tw_w = mx1 - mx0
        s_y = (tw * (-Z) / fy) / max(c_ex_y, 1e-6)
        s_x = (tw_w * (-Z) / fx) / max(c_ex_x, 1e-6)
        X = (u_c - cx) * (-Z) / fx
        Y = -(v_c - cy) * (-Z) / fy
        S = np.diag([s_x, s_y, (s_x + s_y) * 0.5, 1.0]).astype(np.float32)
        T_re = np.eye(4, dtype=np.float32); T_re[:3, 3] = -c_center.astype(np.float32)
        T_pl = np.eye(4, dtype=np.float32); T_pl[:3, 3] = np.array([X, Y, Z], dtype=np.float32)
        M = T_pl @ S @ T_re
        items.append((f"object_{m['idx']:03d}_{m['class']}", mesh, M, np.array([X, Y, Z])))
    return items, vfov_deg


def _look_at(eye, target, up):
    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)
    new_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = new_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _render(items, cam_pose, vfov_deg, w, h):
    pscene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0], ambient_light=[0.75, 0.75, 0.75])
    for name, mesh, M, _ in items:
        mm = mesh.copy()
        v = np.asarray(mm.vertices, dtype=np.float32)
        v_h = np.concatenate([v, np.ones((v.shape[0], 1), dtype=np.float32)], axis=1)
        mm.vertices = (M @ v_h.T).T[:, :3]
        pscene.add(pyrender.Mesh.from_trimesh(mm, smooth=True), name=name)
    cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=w / h)
    pscene.add(cam, pose=cam_pose)
    # camera-anchored 3-point light
    look_dir = -cam_pose[:3, 2]; eye = cam_pose[:3, 3]
    up_w = np.array([0, 1, 0])
    for offset, intensity in [
        ((1.0, 1.0, 0.5), 4.0),
        ((-1.0, 0.5, 0.5), 2.5),
        ((0.0, -0.5, -1.0), 1.5),
    ]:
        d = np.array(offset) * 5.0
        L = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        L_pose = _look_at(eye + d, eye + look_dir, up_w)
        pscene.add(L, pose=L_pose)
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color, _ = r.render(pscene)
    finally:
        r.delete()
    return color[..., :3].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw_image0.jpg"))
    ap.add_argument("--background", default=str(PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"))
    ap.add_argument("--mask-dir", default=str(PROJECT_ROOT / "output" / "mask_postprocess"))
    ap.add_argument("--mesh-dir", default=str(PROJECT_ROOT / "output" / "mesh_texturing"))
    ap.add_argument("--camera-json", default=str(PROJECT_ROOT / "output" / "camera_estimation" / "raw_image0_perspective_fields.json"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0" / "scene_orbit.gif"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--depth", type=float, default=1.5)
    ap.add_argument("--orbit-amp-deg", type=float, default=35.0,
                    help="half-amplitude of horizontal swing in degrees")
    args = ap.parse_args()

    bg = Image.open(args.background).convert("RGB")
    bw, bh = bg.size
    tw = args.width
    th = int(round(bh * tw / bw))
    items, vfov = _build_items(
        Path(args.mask_dir), Path(args.mesh_dir), Path(args.camera_json),
        bw, bh, tw, th, args.depth,
    )
    if not items:
        print("no items")
        return
    centers = np.array([it[3] for it in items])
    scene_center = centers.mean(axis=0)
    # Distance from origin (camera) to scene center.
    base_r = float(np.linalg.norm(scene_center))
    print(f"scene center: {scene_center}, distance: {base_r:.3f}")

    print(f"rendering {args.frames} orbital frames {tw}x{th}...")
    frames = []
    for f in range(args.frames):
        a = math.radians(args.orbit_amp_deg) * math.sin(2 * math.pi * f / args.frames)
        # rotate camera around scene_center on horizontal plane
        eye = scene_center.copy().astype(np.float64)
        # Move eye to the original camera position (origin) then rotate around scene_center
        offset = -scene_center  # origin - scene_center
        # rotate offset around world y
        cos_a, sin_a = math.cos(a), math.sin(a)
        rot_offset = np.array([
            offset[0] * cos_a + offset[2] * sin_a,
            offset[1],
            -offset[0] * sin_a + offset[2] * cos_a,
        ])
        eye = scene_center + rot_offset
        cam_pose = _look_at(eye, scene_center, np.array([0.0, 1.0, 0.0]))
        rgb = _render(items, cam_pose, vfov, tw, th)
        frames.append(rgb)

    imageio.mimsave(args.out, frames, duration=0.05, loop=0)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
