"""Photoreal composite that ignores fitted_transform's camera fit.

For each movable object:
  1. Render the textured Hunyuan-Paint mesh ALONE on white from a fixed
     3/4 angle.
  2. Read the original visible mask from `output/mask_postprocess/`.
  3. Resize the rendered RGBA crop so its non-white bbox matches the mask's
     bbox in the source image (preserving the rendered object's own aspect).
  4. Paste it onto the LaMa-inpainted background at the mask position.

This bypasses fitted_transform's IoU 0.018 problem: instead of trying to
make the 3D scene project to the same silhouettes as the input, we just
PLACE each 3D object exactly where its source-image silhouette was, sized
to match.

Outputs:
  - composite_full.png      : background + all 3D objects placed
  - composite_remove_NN.png : same but with object N omitted
  - composite_wiggle.gif    : objects oscillate independently in image space
  - composite_grid.png      : INPUT + FULL + per-object-removed strip

These are the "what user sees" deliverables. Use this script after the
pipeline produces masks + clean_background + mesh_texturing GLBs.
"""

from __future__ import annotations

import argparse
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


def _render_object_alone(
    mesh: trimesh.Trimesh,
    width: int,
    height: int,
    vfov_deg: float = 38.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb HxWx3, alpha HxW) where alpha=255 inside object outline."""
    bb = mesh.bounds
    center = (bb[0] + bb[1]) / 2
    extent = float(np.linalg.norm(bb[1] - bb[0]))
    eye = center + np.array([0.0, 0.0, extent * 1.3])

    pscene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0), ambient_light=[0.85, 0.85, 0.85])
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    pscene.add(pm)

    forward = (center - eye); forward /= np.linalg.norm(forward)
    up_w = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_w); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4); pose[:3, 0] = right; pose[:3, 1] = up
    pose[:3, 2] = -forward; pose[:3,3] = eye

    cam = pyrender.PerspectiveCamera(
        yfov=math.radians(vfov_deg), aspectRatio=width / height,
    )
    pscene.add(cam, pose=pose)

    # Three-point lighting so PBR-baked textures stay readable.
    for direction, intensity in [
        ((1, 1, 1), 4.0),
        ((-1, -1, 0.5), 2.5),
        ((0, -0.5, -1), 1.5),
    ]:
        d = np.array(direction, dtype=np.float64) * extent
        L = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        L_pose = np.eye(4)
        L_pose[:3, 3] = center + d
        f = (center - (center + d)); f /= np.linalg.norm(f)
        r2 = np.cross(f, up_w); r2 /= np.linalg.norm(r2)
        u2 = np.cross(r2, f)
        L_pose[:3, 0] = r2; L_pose[:3, 1] = u2; L_pose[:3, 2] = -f
        pscene.add(L, pose=L_pose)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = r.render(pscene)
    finally:
        r.delete()
    rgb = color[..., :3].copy()
    is_bg = np.all(rgb >= 248, axis=2)
    alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    return rgb, alpha


def _bbox_of_alpha(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _annotate(img: Image.Image, label: str) -> Image.Image:
    w, h = img.size
    bar = max(28, int(h * 0.05))
    out = Image.new("RGB", (w, h + bar), (15, 18, 30))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((10, h + 4), label, fill=(255, 255, 255), font=_font(max(16, int(h * 0.035))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw_image0.jpg"))
    ap.add_argument("--background", default=str(PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"))
    ap.add_argument("--mask-dir", default=str(PROJECT_ROOT / "output" / "mask_postprocess"))
    ap.add_argument("--mesh-dir", default=str(PROJECT_ROOT / "output" / "mesh_texturing"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0"))
    ap.add_argument("--width", type=int, default=1280, help="output composite width")
    ap.add_argument("--wiggle-frames", type=int, default=36)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Background.
    bg_pil = Image.open(args.background).convert("RGB")
    bg_w, bg_h = bg_pil.size
    target_w = args.width
    target_h = int(bg_h * target_w / bg_w)
    bg_resized = bg_pil.resize((target_w, target_h), Image.LANCZOS)
    bg_np = np.array(bg_resized)
    print(f"background {bg_w}x{bg_h} -> {target_w}x{target_h}")
    sx = target_w / bg_w
    sy = target_h / bg_h

    # Movable masks (skip stuff).
    stuff_kws = load_stuff_keywords()
    mask_paths_all = sorted(Path(args.mask_dir).glob("object_*_mask.npy"))
    movable: list[tuple[int, Path, str]] = []
    for mp in mask_paths_all:
        cname = class_from_filename(mp.stem)
        if any(kw in cname for kw in stuff_kws):
            continue
        # idx in filename "object_NNN_class"
        idx = int(mp.stem.split("_")[1])
        movable.append((idx, mp, cname))
    print(f"movable objects: {[(i, c) for i, _, c in movable]}")

    # Match each movable to a Hunyuan-Paint textured GLB by index prefix.
    mesh_files = sorted(Path(args.mesh_dir).glob("*_remeshed_textured.glb"))

    def _find_mesh(idx: int) -> Path | None:
        prefix = f"{idx:03d}_"
        for mp in mesh_files:
            if mp.name.startswith(prefix):
                return mp
        return None

    # Render each movable object alone and resize/paste to its mask bbox.
    per_object: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], str]] = {}
    for idx, mask_path, cname in movable:
        mesh_path = _find_mesh(idx)
        if mesh_path is None:
            print(f"  skip idx={idx} ({cname}): no mesh found")
            continue
        scene = trimesh.load(mesh_path, force="scene")
        mesh = next(iter(scene.geometry.values()))

        # Render at a generous resolution and crop tightly.
        rgb, alpha = _render_object_alone(mesh, 1024, 1024)
        ob = _bbox_of_alpha(alpha)
        if ob is None:
            print(f"  skip idx={idx} ({cname}): render produced no pixels")
            continue
        ox0, oy0, ox1, oy1 = ob
        rgb_crop = rgb[oy0:oy1, ox0:ox1]
        alpha_crop = alpha[oy0:oy1, ox0:ox1]

        # Mask bbox in source-image coords -> target-canvas coords.
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mb = _bbox_of_mask(mask)
        if mb is None:
            print(f"  skip idx={idx} ({cname}): empty mask")
            continue
        mx0, my0, mx1, my1 = mb
        tx0 = int(round(mx0 * sx))
        ty0 = int(round(my0 * sy))
        tx1 = int(round(mx1 * sx))
        ty1 = int(round(my1 * sy))
        tw = max(1, tx1 - tx0)
        th = max(1, ty1 - ty0)
        # Resize render crop to the mask bbox (forced aspect to fit).
        rgb_resized = np.array(
            Image.fromarray(rgb_crop).resize((tw, th), Image.LANCZOS)
        )
        alpha_resized = np.array(
            Image.fromarray(alpha_crop).resize((tw, th), Image.LANCZOS)
        )
        per_object[idx] = (rgb_resized, alpha_resized, (tx0, ty0, tx1, ty1), cname)
        print(f"  idx={idx} ({cname}): mask bbox {mb} -> canvas {(tx0, ty0, tx1, ty1)}")

    def _paste_objects(canvas: np.ndarray, omit: set[int]) -> np.ndarray:
        out = canvas.copy()
        # depth-sort: smaller mask bbox bottom y -> further back -> draw first.
        order = sorted(per_object.keys(), key=lambda i: per_object[i][2][3])
        for idx in order:
            if idx in omit:
                continue
            rgb, alpha, (tx0, ty0, tx1, ty1), _ = per_object[idx]
            a = alpha.astype(np.float32) / 255.0
            patch = out[ty0:ty1, tx0:tx1]
            out[ty0:ty1, tx0:tx1] = (rgb * a[..., None] + patch * (1 - a[..., None])).astype(np.uint8)
        return out

    # FULL composite.
    full = _paste_objects(bg_np, omit=set())
    Image.fromarray(full).save(out_dir / "composite_full.png")
    print(f"saved composite_full.png")

    # Per-object removal.
    panels: list[Image.Image] = []
    input_pil = Image.open(args.input).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    panels.append(_annotate(input_pil, "INPUT (reference)"))
    panels.append(_annotate(Image.fromarray(full), "FULL (3D objects in source positions)"))
    for idx, _, cname in movable:
        if idx not in per_object:
            continue
        partial = _paste_objects(bg_np, omit={idx})
        panels.append(_annotate(Image.fromarray(partial), f"removed: obj{idx:03d} {cname.split('_')[0]}"))

    # Composite grid.
    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    pw, ph = panels[0].size
    pad = 14
    grid = Image.new(
        "RGB",
        (cols * pw + (cols + 1) * pad, rows * ph + (rows + 1) * pad),
        (240, 240, 240),
    )
    for i, p in enumerate(panels):
        r, c = i // cols, i % cols
        grid.paste(p, (pad + c * (pw + pad), pad + r * (ph + pad)))
    grid.save(out_dir / "composite_grid.png")
    print(f"saved composite_grid.png ({cols}x{rows})")

    # Wiggle gif: per-object 2D translation oscillation.
    print(f"rendering composite_wiggle.gif ({args.wiggle_frames} frames)...")
    rng = np.random.RandomState(42)
    movable_ids = list(per_object.keys())
    dirs = rng.randn(len(movable_ids), 2)
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
    phases = rng.uniform(0, 2 * math.pi, size=len(movable_ids))
    id_to_dir = dict(zip(movable_ids, zip(dirs, phases)))

    # Amplitude: 15% of the smallest movable's bbox short side.
    short_sides = []
    for idx in movable_ids:
        _, _, (tx0, ty0, tx1, ty1), _ = per_object[idx]
        short_sides.append(min(tx1 - tx0, ty1 - ty0))
    amp = 0.35 * (min(short_sides) if short_sides else 80)

    gif_frames: list[np.ndarray] = []
    for f in range(args.wiggle_frames):
        t = 2.0 * math.pi * f / args.wiggle_frames
        out = bg_np.copy()
        order = sorted(movable_ids, key=lambda i: per_object[i][2][3])
        for idx in order:
            rgb, alpha, (tx0, ty0, tx1, ty1), _ = per_object[idx]
            d, phase = id_to_dir[idx]
            dx = int(round(d[0] * amp * math.sin(t + phase)))
            dy = int(round(d[1] * amp * math.sin(t + phase) * 0.4))
            nx0 = max(0, tx0 + dx); nx1 = min(target_w, tx1 + dx)
            ny0 = max(0, ty0 + dy); ny1 = min(target_h, ty1 + dy)
            ow = nx1 - nx0; oh = ny1 - ny0
            if ow <= 0 or oh <= 0:
                continue
            src_x0 = nx0 - (tx0 + dx); src_y0 = ny0 - (ty0 + dy)
            rgb_clip = rgb[src_y0:src_y0 + oh, src_x0:src_x0 + ow]
            a_clip = alpha[src_y0:src_y0 + oh, src_x0:src_x0 + ow].astype(np.float32) / 255.0
            patch = out[ny0:ny1, nx0:nx1]
            out[ny0:ny1, nx0:nx1] = (rgb_clip * a_clip[..., None] + patch * (1 - a_clip[..., None])).astype(np.uint8)
        gif_frames.append(out)
    imageio.mimsave(out_dir / "composite_wiggle.gif", gif_frames, duration=0.06, loop=0)
    print(f"saved composite_wiggle.gif")


if __name__ == "__main__":
    main()
