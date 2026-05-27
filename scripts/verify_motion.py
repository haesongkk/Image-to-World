"""Visual verification demo for object-level scene decomposition.

Generates two artifacts proving "각 객체가 따로 움직인다":

  - motion.gif: orbital camera around the assembled scene (full GLB).
  - grid.png : original camera viewpoint, one panel per object removed.

Usage:
  python scripts/verify_motion.py
  python scripts/verify_motion.py --glb output/scene_assembly/raw_image_assembled.glb
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
import imageio.v2 as imageio

import pyrender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _default_glb() -> Path:
    return PROJECT_ROOT / "output" / "scene_assembly" / "raw_image_assembled.glb"


def _default_camera_json() -> Path:
    return PROJECT_ROOT / "output" / "camera_estimation" / "raw_image_perspective_fields.json"


def _default_image() -> Path:
    return PROJECT_ROOT / "data" / "raw_image.jpg"


def _default_out_dir() -> Path:
    return PROJECT_ROOT / "output" / "demo"


def input_camera_pose(roll_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Cam-to-world pose matching the pipeline's input camera.

    Scene GLB uses OpenGL/PyTorch3D convention: +x right, +y up, -z forward (camera at origin).
    PerspectiveFields roll/pitch tilt the camera around its local axes.
    """
    pose = np.eye(4, dtype=np.float64)
    if abs(roll_deg) < 1e-6 and abs(pitch_deg) < 1e-6:
        return pose
    rr = math.radians(roll_deg)
    pr = math.radians(pitch_deg)
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    pose[:3, :3] = R_pitch @ R_roll
    return pose


def look_at(eye: np.ndarray, target: np.ndarray, up_world: np.ndarray | None = None) -> np.ndarray:
    """OpenGL-style cam-to-world pose. Camera looks from eye toward target, up=+world_z by default."""
    if up_world is None:
        up_world = np.array([0.0, 0.0, 1.0])
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up_world = np.asarray(up_world, dtype=np.float64).reshape(3)
    forward = target - eye
    n = float(np.linalg.norm(forward))
    if n < 1e-12:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / n
    if abs(float(np.dot(forward, up_world))) > 0.999:
        up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_world)
    right = right / max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def load_objects(glb_path: Path) -> list[tuple[str, trimesh.Trimesh]]:
    scene = trimesh.load(glb_path, force="scene")
    items: list[tuple[str, trimesh.Trimesh]] = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            items.append((name, geom))
    items.sort(key=lambda x: x[0])
    return items


def build_pyrender_scene(
    items: list[tuple[str, trimesh.Trimesh]],
    exclude: set[str] | None = None,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pyrender.Scene:
    pscene = pyrender.Scene(bg_color=bg, ambient_light=[0.45, 0.45, 0.45])
    for name, geom in items:
        if exclude is not None and name in exclude:
            continue
        try:
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        except Exception:
            continue
        pscene.add(mesh, name=name)
    return pscene


def add_lights(pscene: pyrender.Scene, centroid: np.ndarray, extent: float) -> None:
    L = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    pscene.add(L, pose=look_at(centroid + np.array([extent, extent, extent]), centroid))
    L2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    pscene.add(L2, pose=look_at(centroid + np.array([-extent, -extent, 0.5 * extent]), centroid))


def render(pscene: pyrender.Scene, width: int, height: int) -> np.ndarray:
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = r.render(pscene)
    finally:
        r.delete()
    return color[..., :3].copy()


def annotate(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    pim = Image.fromarray(img)
    draw = ImageDraw.Draw(pim)
    bar_h = 22
    draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))
    draw.text((6, h - bar_h + 4), label, fill=(255, 255, 255))
    return np.array(pim)


def make_grid(images: list[np.ndarray], cols: int, pad: int = 6) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    H = rows * h + (rows + 1) * pad
    W = cols * w + (cols + 1) * pad
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = img
    return canvas


def render_grid(
    items: list[tuple[str, trimesh.Trimesh]],
    centroid: np.ndarray,
    extent: float,
    vfov_deg: float,
    roll_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    cols: int,
    input_image: np.ndarray | None,
) -> np.ndarray:
    # GLB world matches OpenGL camera convention: camera at origin looks -z.
    cam_pose = input_camera_pose(roll_deg=roll_deg, pitch_deg=pitch_deg)
    aspect = width / height

    panels: list[np.ndarray] = []

    if input_image is not None:
        ref = _fit_panel(input_image, width, height)
        panels.append(annotate(ref, "INPUT (reference)"))

    pscene = build_pyrender_scene(items)
    cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
    pscene.add(cam, pose=cam_pose)
    add_lights(pscene, centroid, extent)
    panels.append(annotate(render(pscene, width, height), "FULL (reproj)"))

    for name, _ in items:
        pscene = build_pyrender_scene(items, exclude={name})
        cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
        pscene.add(cam, pose=cam_pose)
        add_lights(pscene, centroid, extent)
        panels.append(annotate(render(pscene, width, height), f"- {name}"))

    return make_grid(panels, cols=cols)


def _fit_panel(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to fit a panel of (width, height), preserving aspect with white padding."""
    pim = Image.fromarray(img).convert("RGB")
    iw, ih = pim.size
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    pim_r = pim.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(pim_r, ((width - nw) // 2, (height - nh) // 2))
    return np.array(canvas)


def render_orbit_gif(
    items: list[tuple[str, trimesh.Trimesh]],
    centroid: np.ndarray,
    extent: float,
    vfov_deg: float,
    width: int,
    height: int,
    frames: int,
) -> list[np.ndarray]:
    aspect = width / height
    radius = max(extent * 0.8, 1.0)
    eye_z = centroid[2] + 0.25 * extent

    frames_rgb: list[np.ndarray] = []
    for i in range(frames):
        theta = 2.0 * math.pi * i / frames
        eye = np.array(
            [
                centroid[0] + radius * math.cos(theta),
                centroid[1] + radius * math.sin(theta),
                eye_z,
            ]
        )
        cam_pose = look_at(eye, centroid)
        pscene = build_pyrender_scene(items)
        cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
        pscene.add(cam, pose=cam_pose)
        add_lights(pscene, centroid, extent)
        frames_rgb.append(render(pscene, width, height))
    return frames_rgb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", default=str(_default_glb()))
    ap.add_argument("--camera-json", default=str(_default_camera_json()))
    ap.add_argument("--input-image", default=str(_default_image()))
    ap.add_argument("--out-dir", default=str(_default_out_dir()))
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=None,
                    help="If omitted, derived from input image aspect ratio.")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--grid-cols", type=int, default=5)
    args = ap.parse_args()

    glb_path = Path(args.glb).resolve()
    cam_json_path = Path(args.camera_json).resolve()
    image_path = Path(args.input_image).resolve()
    if not glb_path.exists():
        raise SystemExit(f"GLB not found: {glb_path}")

    stem = glb_path.stem.replace("_assembled", "")
    out_dir = Path(args.out_dir).resolve() / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_objects(glb_path)
    if not items:
        raise SystemExit(f"No mesh geometries found in: {glb_path}")
    print(f"loaded {len(items)} objects from {glb_path.name}")

    all_verts = np.concatenate([g.vertices for _, g in items], axis=0)
    centroid = all_verts.mean(axis=0)
    bb_min = all_verts.min(axis=0)
    bb_max = all_verts.max(axis=0)
    extent = float(np.linalg.norm(bb_max - bb_min))
    print(f"centroid={centroid.tolist()}, extent={extent:.3f}")

    input_image: np.ndarray | None = None
    if image_path.exists():
        input_image = np.array(Image.open(image_path).convert("RGB"))

    width = args.width
    if args.height is None:
        if input_image is not None:
            ih, iw = input_image.shape[:2]
            height = max(1, int(round(width * ih / iw)))
        else:
            height = int(round(width * 9 / 16))
    else:
        height = args.height

    vfov_deg = 60.0
    roll_deg = 0.0
    pitch_deg = 0.0
    if cam_json_path.exists():
        with open(cam_json_path, "r", encoding="utf-8") as f:
            pf = json.load(f)
        vfov_deg = float(pf.get("pred_general_vfov", pf.get("pred_vfov", vfov_deg)))
        roll_deg = float(pf.get("pred_roll", 0.0))
        pitch_deg = float(pf.get("pred_pitch", 0.0))
    else:
        print(f"warning: camera json not found, using default vfov={vfov_deg}")

    print(f"vfov={vfov_deg:.2f}, roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, render={width}x{height}")

    print("rendering grid (input-camera view, each object removed once)...")
    grid = render_grid(
        items, centroid, extent, vfov_deg, roll_deg, pitch_deg,
        width, height, args.grid_cols, input_image,
    )
    grid_path = out_dir / "grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"saved {grid_path}")

    print(f"rendering motion.gif ({args.frames} orbital frames)...")
    frames = render_orbit_gif(items, centroid, extent, vfov_deg, width, height, args.frames)
    gif_path = out_dir / "motion.gif"
    imageio.mimsave(gif_path, frames, duration=0.06, loop=0)
    print(f"saved {gif_path}")


if __name__ == "__main__":
    main()
