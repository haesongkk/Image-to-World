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


def _default_fitted_camera_json(image_stem: str = "raw_image") -> Path:
    return PROJECT_ROOT / "output" / "fitted_transform_debug" / image_stem / "simple" / "render_meta.json"


def _default_image() -> Path:
    return PROJECT_ROOT / "data" / "raw_image.jpg"


def _default_out_dir() -> Path:
    return PROJECT_ROOT / "output" / "demo"


def input_camera_pose(
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    dist: float = 0.0,
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
) -> np.ndarray:
    """Cam-to-world pose matching the pipeline's input camera.

    Two regimes:
      - `dist == 0`: camera sits at world origin looking -z, optionally tilted
        by PerspectiveFields roll/pitch. Used when fitted_transform stored its
        transforms with the camera at origin.
      - `dist > 0`: pytorch3d-style `look_at_view_transform(dist, elev, azim)`
        camera around the world origin. Used when fitted_transform optimized a
        free camera (render_meta.json `camera.dist`).
    """
    if dist > 1e-6:
        elev_r = math.radians(elev_deg)
        azim_r = math.radians(azim_deg)
        eye = np.array([
            dist * math.cos(elev_r) * math.sin(azim_r),
            dist * math.sin(elev_r),
            dist * math.cos(elev_r) * math.cos(azim_r),
        ], dtype=np.float64)
        # GLB / pytorch3d world up is +y, NOT +z (look_at default).
        return look_at(eye, np.zeros(3), up_world=np.array([0.0, 1.0, 0.0]))

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


def composite_over_background(
    render_rgb: np.ndarray,
    bg_path: Path | None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Replace the white background of a pyrender output with a real image.

    pyrender renders movable objects against `bg_color` (white). Any pixel
    that is exactly bg_color is treated as "no object" and replaced with the
    corresponding pixel from the resized background image.
    """
    if bg_path is None or not bg_path.exists():
        return render_rgb
    bg = Image.open(bg_path).convert("RGB")
    h, w = render_rgb.shape[:2]
    bg_resized = np.array(bg.resize((w, h), Image.BILINEAR))
    is_bg = np.all(render_rgb >= 250, axis=2)
    out = render_rgb.copy()
    out[is_bg] = bg_resized[is_bg]
    return out


def build_pyrender_scene(
    items: list[tuple[str, trimesh.Trimesh]],
    exclude: set[str] | None = None,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ambient: float = 0.70,
    skip_background_planes: bool = False,
) -> pyrender.Scene:
    # Higher ambient so Hunyuan-Paint textures stay readable on dark objects
    # (the air fryer is nearly black, so weak ambient + side-lighting alone
    # makes it render as a featureless silhouette).
    pscene = pyrender.Scene(bg_color=bg, ambient_light=[ambient, ambient, ambient])
    for name, geom in items:
        if exclude is not None and name in exclude:
            continue
        if skip_background_planes and ("background_" in name):
            continue
        try:
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        except Exception:
            continue
        pscene.add(mesh, name=name)
    return pscene


def add_lights(pscene: pyrender.Scene, centroid: np.ndarray, extent: float) -> None:
    # Front-key + back-fill + rim, all with `centroid` as look-at target so any
    # rendering scale gets even illumination on the visible side.
    L = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    pscene.add(L, pose=look_at(centroid + np.array([extent, extent, extent]), centroid))
    L2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    pscene.add(L2, pose=look_at(centroid + np.array([-extent, -extent, 0.5 * extent]), centroid))
    L3 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    pscene.add(L3, pose=look_at(centroid + np.array([0.0, -0.5 * extent, -extent]), centroid))


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
    # Font size grows with panel height so 4K-style panels get readable labels.
    font_size = max(14, int(round(h * 0.06)))
    bar_h = font_size + 12
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = None
    draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))
    draw.text((8, h - bar_h + 5), label, fill=(255, 255, 255), font=font)
    return np.array(pim)


def _shorten_label(name: str) -> str:
    """object_002_appliance_home_appliance -> 'appliance'."""
    s = name
    for prefix in ("object_",):
        if s.startswith(prefix):
            rest = s[len(prefix):]
            # rest = "002_appliance_home_appliance"
            parts = rest.split("_", 1)
            if len(parts) == 2:
                cls = parts[1]
                # collapse duplicate-like suffix "home_appliance" -> keep first word
                cls = cls.split("_")[0]
                return f"obj{parts[0]} {cls}"
    if s.startswith("background_"):
        return s[len("background_"):]
    return s


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
    cam_dist: float = 0.0,
    cam_elev: float = 0.0,
    cam_azim: float = 0.0,
    bg_path: Path | None = None,
) -> np.ndarray:
    cam_pose = input_camera_pose(
        roll_deg=roll_deg, pitch_deg=pitch_deg,
        dist=cam_dist, elev_deg=cam_elev, azim_deg=cam_azim,
    )
    aspect = width / height

    # When a photoreal background is provided, skip the procedural floor/wall
    # planes so we don't double-render the kitchen — the inpainted background
    # already contains those.
    skip_planes = bg_path is not None and bg_path.exists()

    def _composite_label(raw: np.ndarray) -> np.ndarray:
        return composite_over_background(raw, bg_path) if skip_planes else raw

    panels: list[np.ndarray] = []

    if input_image is not None:
        ref = _fit_panel(input_image, width, height)
        panels.append(annotate(ref, "INPUT (reference)"))

    pscene = build_pyrender_scene(items, skip_background_planes=skip_planes)
    cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
    pscene.add(cam, pose=cam_pose)
    add_lights(pscene, centroid, extent)
    panels.append(annotate(_composite_label(render(pscene, width, height)), "FULL (3D objects on photoreal bg)"))

    for name, _ in items:
        if skip_planes and "background_" in name:
            continue  # nothing useful to show by removing a plane we didn't draw
        pscene = build_pyrender_scene(items, exclude={name}, skip_background_planes=skip_planes)
        cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
        pscene.add(cam, pose=cam_pose)
        add_lights(pscene, centroid, extent)
        panels.append(annotate(_composite_label(render(pscene, width, height)), f"removed: {_shorten_label(name)}"))

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


def render_wiggle_gif(
    items: list[tuple[str, trimesh.Trimesh]],
    centroid: np.ndarray,
    extent: float,
    vfov_deg: float,
    roll_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    frames: int,
    cam_dist: float = 0.0,
    cam_elev: float = 0.0,
    cam_azim: float = 0.0,
    bg_path: Path | None = None,
) -> list[np.ndarray]:
    """Each non-background object oscillates independently along a random
    direction to visually prove object-level rigid-body decomposition.
    Camera stays at the input pose so the viewer can compare against the
    reference image.
    """
    cam_pose = input_camera_pose(
        roll_deg=roll_deg, pitch_deg=pitch_deg,
        dist=cam_dist, elev_deg=cam_elev, azim_deg=cam_azim,
    )
    aspect = width / height
    rng = np.random.RandomState(42)

    movable_names = [n for n, _ in items if "background" not in n]
    dirs = rng.randn(len(movable_names), 3)
    dirs[:, 1] = 0.0  # keep wiggle horizontal so objects don't sink through floor
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-6)
    phases = rng.uniform(0, 2 * math.pi, size=len(movable_names))
    name_to_dir = dict(zip(movable_names, zip(dirs, phases)))

    # Wiggle amplitude scaled to typical object size, not whole-scene extent.
    # extent includes the wide floor/wall planes, so 0.18*extent was ~3x bigger
    # than the largest movable. Pick a fraction of the smallest non-background
    # object so even small objects show clearly without giant ones flying.
    movable_extents = []
    for n, g in items:
        if "background" in n:
            continue
        bb = g.vertices.max(0) - g.vertices.min(0)
        movable_extents.append(float(np.linalg.norm(bb)))
    base = min(movable_extents) if movable_extents else extent
    amp = 0.6 * base

    skip_planes = bg_path is not None and bg_path.exists()

    frames_rgb: list[np.ndarray] = []
    for f in range(frames):
        t = 2.0 * math.pi * f / frames
        pscene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0), ambient_light=[0.70, 0.70, 0.70])
        for name, geom in items:
            if skip_planes and "background_" in name:
                continue
            if name not in name_to_dir:
                pscene.add(pyrender.Mesh.from_trimesh(geom, smooth=False), name=name)
                continue
            d, phase = name_to_dir[name]
            offset = d * (amp * math.sin(t + phase))
            geom_off = geom.copy()
            geom_off.vertices = geom.vertices + offset[None, :]
            pscene.add(pyrender.Mesh.from_trimesh(geom_off, smooth=False), name=name)
        cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
        pscene.add(cam, pose=cam_pose)
        add_lights(pscene, centroid, extent)
        raw = render(pscene, width, height)
        frames_rgb.append(composite_over_background(raw, bg_path) if skip_planes else raw)
    return frames_rgb


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
    ap.add_argument("--fitted-camera-json", default=None,
                    help="render_meta.json from fitted_transform; if present uses the optimized look_at camera instead of camera-at-origin.")
    ap.add_argument("--background", default=str(PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"),
                    help="Photoreal LaMa-inpainted background to composite under the rendered 3D objects. Pass empty string to disable.")
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

    cam_dist = 0.0
    cam_elev = 0.0
    cam_azim = 0.0
    fitted_cam_path = (
        Path(args.fitted_camera_json).resolve() if args.fitted_camera_json
        else _default_fitted_camera_json(stem)
    )
    if fitted_cam_path.exists():
        with open(fitted_cam_path, "r", encoding="utf-8") as f:
            rm = json.load(f)
        fc = rm.get("camera", {})
        cam_dist = float(fc.get("dist", 0.0))
        cam_elev = float(fc.get("elev", 0.0))
        cam_azim = float(fc.get("azim", 0.0))
        cam_fov = fc.get("fov")
        if cam_fov is not None:
            vfov_deg = float(cam_fov)
            roll_deg = 0.0
            pitch_deg = 0.0
        print(f"using fitted camera: dist={cam_dist:.3f}, elev={cam_elev:.2f}, azim={cam_azim:.2f}, fov={vfov_deg:.2f}")

    print(f"vfov={vfov_deg:.2f}, roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, render={width}x{height}")

    bg_path = Path(args.background).resolve() if args.background else None
    if bg_path is not None and not bg_path.exists():
        print(f"warning: background image not found at {bg_path}; using procedural floor/wall planes")
        bg_path = None
    elif bg_path is not None:
        print(f"using photoreal background: {bg_path.name}")

    print("rendering grid (input-camera view, each object removed once)...")
    grid = render_grid(
        items, centroid, extent, vfov_deg, roll_deg, pitch_deg,
        width, height, args.grid_cols, input_image,
        cam_dist=cam_dist, cam_elev=cam_elev, cam_azim=cam_azim,
        bg_path=bg_path,
    )
    grid_path = out_dir / "grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"saved {grid_path}")

    print(f"rendering motion.gif ({args.frames} orbital frames)...")
    frames = render_orbit_gif(items, centroid, extent, vfov_deg, width, height, args.frames)
    gif_path = out_dir / "motion.gif"
    imageio.mimsave(gif_path, frames, duration=0.06, loop=0)
    print(f"saved {gif_path}")

    print(f"rendering wiggle.gif ({args.frames} frames, per-object independent motion)...")
    wig_frames = render_wiggle_gif(
        items, centroid, extent, vfov_deg, roll_deg, pitch_deg,
        width, height, args.frames,
        cam_dist=cam_dist, cam_elev=cam_elev, cam_azim=cam_azim,
        bg_path=bg_path,
    )
    wig_path = out_dir / "wiggle.gif"
    imageio.mimsave(wig_path, wig_frames, duration=0.06, loop=0)
    print(f"saved {wig_path}")


if __name__ == "__main__":
    main()
