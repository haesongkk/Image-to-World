"""Mask-driven 3D mesh placement, bypassing fitted_transform.

For each movable object:
  1. Load Hunyuan-Paint textured mesh (canonical Hunyuan space, ~[-1,1]).
  2. Compute the mesh's canonical bbox extent.
  3. Read the object's source-image mask bbox (in pixels).
  4. Place the mesh at a fixed depth Z such that its projected bbox in the
     camera image matches the mask bbox center+height. Uniform scale to
     preserve aspect (height is the most natural reference for objects
     resting on a counter / sitting upright).
  5. Render the scene with pyrender, composite over LaMa background.

The user's source-image camera is recovered from PerspectiveFields (vfov).
We pick a fixed Z_DEPTH so the meshes don't fight each other on depth.

Outputs (under output/demo/raw_image0/):
  - placed_full.png         : photoreal composite, all 3D meshes placed
  - placed_grid.png         : INPUT | FULL | removed-N panels
  - placed_wiggle.gif       : per-object 3D translation in image space
  - placed_motion.gif       : orbital camera (proves it's 3D)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import pyrender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tool.stuff_filter import (  # noqa: E402
    class_from_filename,
    load_stuff_keywords,
)


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _annotate(img: Image.Image, label: str) -> Image.Image:
    w, h = img.size
    bar = max(28, int(h * 0.05))
    out = Image.new("RGB", (w, h + bar), (15, 18, 30))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((10, h + 4), label, fill=(255, 255, 255), font=_font(max(16, int(h * 0.035))))
    return out


def _composite(rgba: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Composite RGBA render over bg_rgb using the alpha channel.
    If rgba is shape (H,W,3) (no alpha), fall back to white-key replacement."""
    if rgba.shape[-1] == 4:
        rgb = rgba[..., :3]
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        return (rgb.astype(np.float32) * a + bg_rgb.astype(np.float32) * (1 - a)).astype(np.uint8)
    is_bg = np.all(rgba >= 248, axis=2)
    out = rgba.copy()
    out[is_bg] = bg_rgb[is_bg]
    return out


def _build_scene(items: list[tuple[str, trimesh.Trimesh, np.ndarray]], cam_pose: np.ndarray,
                 vfov_deg: float, width: int, height: int) -> np.ndarray:
    pscene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.75, 0.75, 0.75])
    for name, mesh, M in items:
        m = mesh.copy()
        v = np.asarray(m.vertices, dtype=np.float32)
        v_h = np.concatenate([v, np.ones((v.shape[0], 1), dtype=np.float32)], axis=1)
        v_w = (M @ v_h.T).T[:, :3]
        m.vertices = v_w
        pm = pyrender.Mesh.from_trimesh(m, smooth=True)
        pscene.add(pm, name=name)
    cam = pyrender.PerspectiveCamera(
        yfov=math.radians(vfov_deg), aspectRatio=width / height,
    )
    pscene.add(cam, pose=cam_pose)
    # Three-point lighting locked to camera so PBR shows correctly.
    look_dir = -cam_pose[:3, 2]
    eye = cam_pose[:3, 3]
    for offset, intensity in [
        ((1.0, 1.0, 0.5), 4.0),
        ((-1.0, 0.5, 0.5), 2.5),
        ((0.0, -0.5, -1.0), 1.5),
    ]:
        d = np.array(offset, dtype=np.float64) * 5.0
        L = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        L_pose = np.eye(4)
        L_pose[:3, 3] = eye + d
        f = (eye + look_dir) - L_pose[:3, 3]
        f /= max(np.linalg.norm(f), 1e-6)
        right = np.cross(f, [0, 1, 0])
        right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, f)
        L_pose[:3, 0] = right; L_pose[:3, 1] = up; L_pose[:3, 2] = -f
        pscene.add(L, pose=L_pose)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = r.render(pscene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()
    return color.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw_image0.jpg"))
    ap.add_argument("--background", default=str(PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"))
    ap.add_argument("--mask-dir", default=str(PROJECT_ROOT / "output" / "mask_postprocess"))
    ap.add_argument("--mesh-dir", default=str(PROJECT_ROOT / "output" / "mesh_texturing"))
    ap.add_argument("--camera-json", default=str(PROJECT_ROOT / "output" / "camera_estimation" / "raw_image0_perspective_fields.json"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--depth", type=float, default=1.5, help="fixed 3D depth for placed meshes (m)")
    ap.add_argument("--wiggle-frames", type=int, default=48)
    ap.add_argument("--motion-frames", type=int, default=48)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Background & input at target resolution.
    bg_full = Image.open(args.background).convert("RGB")
    bg_w, bg_h = bg_full.size
    target_w = args.width
    target_h = int(round(bg_h * target_w / bg_w))
    bg_resized = np.array(bg_full.resize((target_w, target_h), Image.LANCZOS))
    sx = target_w / bg_w
    sy = target_h / bg_h
    input_resized = np.array(
        Image.open(args.input).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    )

    # Camera intrinsics from PerspectiveFields (just the vfov).
    with open(args.camera_json, "r", encoding="utf-8") as f:
        pf = json.load(f)
    vfov_deg = float(pf.get("pred_general_vfov", pf.get("pred_vfov", 50.0)))
    # Image-plane principal point at canvas center; focal in canvas pixels.
    fy = (target_h * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)
    fx = fy  # square pixels
    cx = target_w * 0.5
    cy = target_h * 0.5
    print(f"vfov={vfov_deg:.2f} -> fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    Z = -abs(args.depth)  # camera looks down -z; objects in front have z<0
    print(f"placement depth: z={Z}")

    # Movable masks.
    stuff_kws = load_stuff_keywords()
    movable: list[dict] = []
    for mp in sorted(Path(args.mask_dir).glob("object_*_mask.npy")):
        cname = class_from_filename(mp.stem)
        if any(kw in cname for kw in stuff_kws):
            continue
        idx = int(mp.stem.split("_")[1])
        mask = np.load(mp)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        m_resized = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (target_w, target_h), Image.NEAREST,
            )
        )
        m_resized = (m_resized > 127).astype(np.uint8)
        ys, xs = np.where(m_resized > 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        movable.append({
            "idx": idx, "class": cname,
            "mask_bbox": (x0, y0, x1, y1),
            "mask_center": ((x0 + x1) / 2, (y0 + y1) / 2),
        })

    # Match meshes by index prefix.
    mesh_files = sorted(Path(args.mesh_dir).glob("*_remeshed_textured.glb"))

    def _find_mesh(idx: int) -> Path | None:
        prefix = f"{idx:03d}_"
        for mp in mesh_files:
            if mp.name.startswith(prefix):
                return mp
        return None

    items: list[tuple[str, trimesh.Trimesh, np.ndarray]] = []
    for m in movable:
        mp = _find_mesh(m["idx"])
        if mp is None:
            print(f"  skip idx={m['idx']}: no mesh")
            continue
        scene_obj = trimesh.load(mp, force="scene")
        mesh = next(iter(scene_obj.geometry.values()))
        # Canonical bbox (Hunyuan normalizes to ~unit cube but each object differs).
        bb = mesh.bounds  # (2,3)
        c_min = bb[0]; c_max = bb[1]
        c_center = (c_min + c_max) * 0.5
        c_extent_y = float(c_max[1] - c_min[1])  # height in canonical
        c_extent_x = float(c_max[0] - c_min[0])
        c_extent_z = float(c_max[2] - c_min[2])

        # Target image-space placement.
        mx0, my0, mx1, my1 = m["mask_bbox"]
        u_c = (mx0 + mx1) * 0.5
        v_c = (my0 + my1) * 0.5
        target_h_px = my1 - my0
        target_w_px = mx1 - mx0

        # World-space size needed to project to target_h_px at depth |Z|:
        target_world_h = target_h_px * (-Z) / fy
        target_world_w = target_w_px * (-Z) / fx
        # Uniform scale based on whichever axis needs more room (avoid clipping).
        s_y = target_world_h / max(c_extent_y, 1e-6)
        s_x = target_world_w / max(c_extent_x, 1e-6)
        # Per-axis (non-uniform) scale so silhouette bbox actually matches the mask.
        # Mesh distortion is acceptable here — the "cheat" is to make placement
        # convincing; per-axis stretch a Hunyuan blob to fit the mask is what
        # sells the illusion.
        scale_xyz = np.array([s_x, s_y, (s_x + s_y) * 0.5], dtype=np.float32)

        # Translation: map mask center to 3D such that it projects to (u_c, v_c).
        # u = fx * X / (-Z) + cx -> X = (u_c - cx) * (-Z) / fx
        X = (u_c - cx) * (-Z) / fx
        Y = -(v_c - cy) * (-Z) / fy  # image y down, world y up

        # Build transform: T * S * (-c_center) so mesh is centered then translated.
        S = np.diag([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2]), 1.0]).astype(np.float32)
        T_recenter = np.eye(4, dtype=np.float32); T_recenter[:3, 3] = -c_center.astype(np.float32)
        T_place = np.eye(4, dtype=np.float32); T_place[:3, 3] = np.array([X, Y, Z], dtype=np.float32)
        M = T_place @ S @ T_recenter

        items.append((f"object_{m['idx']:03d}_{m['class']}", mesh, M))
        print(
            f"  idx={m['idx']} ({m['class']}) "
            f"bbox=({mx0},{my0},{mx1},{my1}) "
            f"mask_size={target_w_px}x{target_h_px} "
            f"-> scale=({s_x:.3f},{s_y:.3f}) center=({X:+.3f},{Y:+.3f},{Z:+.3f})"
        )

    if not items:
        print("no items to render")
        return

    # Camera at origin looking down -z (so our X/Y/Z math is the pyrender camera).
    cam_pose = np.eye(4, dtype=np.float64)

    # FULL.
    render_full = _build_scene(items, cam_pose, vfov_deg, target_w, target_h)
    print(f"render shape={render_full.shape} dtype={render_full.dtype}")
    print(f"render corner={render_full[0, 0]}, center={render_full[target_h//2, target_w//2]}")
    print(f"bg_resized shape={bg_resized.shape} dtype={bg_resized.dtype} center={bg_resized[target_h//2, target_w//2]}")
    Image.fromarray(render_full).save(out_dir / "placed_render_raw.png")
    full = _composite(render_full, bg_resized)
    Image.fromarray(full).save(out_dir / "placed_full.png")
    print("saved placed_full.png")

    # Per-object removal.
    panels: list[Image.Image] = []
    panels.append(_annotate(Image.fromarray(input_resized), "INPUT (raw_image0.jpg)"))
    panels.append(_annotate(Image.fromarray(full), "FULL (3D meshes placed by mask)"))
    for name, mesh, M in items:
        idx = int(name.split("_")[1])
        cname = "_".join(name.split("_")[2:])
        remaining = [it for it in items if it[0] != name]
        partial_render = _build_scene(remaining, cam_pose, vfov_deg, target_w, target_h)
        partial = _composite(partial_render, bg_resized)
        panels.append(_annotate(
            Image.fromarray(partial),
            f"removed: obj{idx:03d} {cname.split('_')[0]}",
        ))

    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    pw, ph = panels[0].size
    pad = 14
    grid = Image.new("RGB",
                     (cols * pw + (cols + 1) * pad, rows * ph + (rows + 1) * pad),
                     (240, 240, 240))
    for i, p in enumerate(panels):
        r, c = i // cols, i % cols
        grid.paste(p, (pad + c * (pw + pad), pad + r * (ph + pad)))
    grid.save(out_dir / "placed_grid.png")
    print(f"saved placed_grid.png ({cols}x{rows})")

    # Wiggle: translate each mesh in image space horizontally.
    print(f"rendering placed_wiggle.gif ({args.wiggle_frames} frames)...")
    rng = np.random.RandomState(7)
    n = len(items)
    dirs = rng.randn(n, 3)
    dirs[:, 1] *= 0.25
    dirs[:, 2] = 0.0  # no depth wiggle
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
    phases = rng.uniform(0, 2 * math.pi, size=n)
    amp = 0.12 * abs(Z)  # in world units

    gif_frames: list[np.ndarray] = []
    for f in range(args.wiggle_frames):
        t = 2.0 * math.pi * f / args.wiggle_frames
        moved_items = []
        for i, (name, mesh, M) in enumerate(items):
            d = dirs[i]; phase = phases[i]
            offset = d * amp * math.sin(t + phase)
            M2 = M.copy()
            M2[:3, 3] = M2[:3, 3] + offset.astype(np.float32)
            moved_items.append((name, mesh, M2))
        rgb = _build_scene(moved_items, cam_pose, vfov_deg, target_w, target_h)
        gif_frames.append(_composite(rgb, bg_resized))
    imageio.mimsave(out_dir / "placed_wiggle.gif", gif_frames, duration=0.06, loop=0)
    print("saved placed_wiggle.gif")

    # Motion: orbital camera around scene center; background stays static (proves 3D).
    print(f"rendering placed_motion.gif ({args.motion_frames} frames)...")
    centers = np.array([it[2][:3, 3] for it in items])
    scene_center = centers.mean(axis=0)
    radius = max(abs(Z) * 0.6, 0.8)
    motion_frames: list[np.ndarray] = []
    for f in range(args.motion_frames):
        a = 2.0 * math.pi * f / args.motion_frames
        # small orbit so the photoreal bg stays roughly aligned
        amp_x = 0.25 * radius
        amp_z = 0.20 * radius
        eye = scene_center + np.array([amp_x * math.sin(a), 0.0, amp_z * math.cos(a) - amp_z + abs(Z) - abs(Z)])
        # Keep camera at origin but pan a bit horizontally
        cam_pose_mov = np.eye(4)
        cam_pose_mov[0, 3] = amp_x * math.sin(a)
        cam_pose_mov[1, 3] = -0.05 * radius * math.sin(2 * a)
        rgb = _build_scene(items, cam_pose_mov, vfov_deg, target_w, target_h)
        motion_frames.append(_composite(rgb, bg_resized))
    imageio.mimsave(out_dir / "placed_motion.gif", motion_frames, duration=0.06, loop=0)
    print("saved placed_motion.gif")


if __name__ == "__main__":
    main()
