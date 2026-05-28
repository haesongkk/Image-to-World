"""Evaluation runner (E1 + E2 + E6 MVP).

Computes scene-level reprojection metrics by rendering the assembled GLB
from the estimated input camera and comparing to the original input image.

Outputs a per-run record appended to `output/eval/manifest.jsonl`.

Usage:
  python scripts/run_eval.py
  python scripts/run_eval.py --image data/raw_image.jpg --glb output/scene_assembly/raw_image_assembled.glb
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
import pyrender
from skimage.metrics import structural_similarity as ssim_metric

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse rendering helpers from verify_motion to keep one source of truth.
from scripts.verify_motion import (  # noqa: E402
    build_pyrender_scene,
    add_lights,
    input_camera_pose,
    load_objects,
    render as render_pyscene,
    _default_fitted_camera_json,
)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _file_sha256(path: Path, max_bytes: int = 8 * 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            chunk = f.read(max_bytes)
            h.update(chunk)
            extra = f.read(64)
            if extra:
                h.update(extra)
        return h.hexdigest()[:16]
    except Exception:
        return None


def render_input_view(
    items: list[tuple[str, trimesh.Trimesh]],
    width: int,
    height: int,
    vfov_deg: float,
    roll_deg: float,
    pitch_deg: float,
    cam_dist: float = 0.0,
    cam_elev: float = 0.0,
    cam_azim: float = 0.0,
    skip_background_planes: bool = False,
) -> np.ndarray:
    all_verts = np.concatenate([g.vertices for _, g in items], axis=0)
    centroid = all_verts.mean(axis=0)
    bb_min = all_verts.min(axis=0)
    bb_max = all_verts.max(axis=0)
    extent = float(np.linalg.norm(bb_max - bb_min))

    cam_pose = input_camera_pose(
        roll_deg=roll_deg, pitch_deg=pitch_deg,
        dist=cam_dist, elev_deg=cam_elev, azim_deg=cam_azim,
    )
    aspect = width / height
    pscene = build_pyrender_scene(items, skip_background_planes=skip_background_planes)
    cam = pyrender.PerspectiveCamera(yfov=math.radians(vfov_deg), aspectRatio=aspect)
    pscene.add(cam, pose=cam_pose)
    add_lights(pscene, centroid, extent)
    return render_pyscene(pscene, width, height)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        ssim_metric(
            a,
            b,
            channel_axis=2,
            data_range=255,
            win_size=7,
        )
    )


def compute_metrics(
    input_rgb: np.ndarray,
    render_rgb: np.ndarray,
) -> dict:
    if input_rgb.shape != render_rgb.shape:
        h, w = render_rgb.shape[:2]
        input_resized = np.array(
            Image.fromarray(input_rgb).resize((w, h), Image.BILINEAR)
        )
    else:
        input_resized = input_rgb

    metrics = {
        "psnr_full": psnr(input_resized, render_rgb),
        "ssim_full": ssim_rgb(input_resized, render_rgb),
    }

    # Foreground mask: pixels where the render has non-background color (anything not pure white).
    bg_mask = np.all(render_rgb >= 250, axis=2)
    fg_mask = ~bg_mask
    if fg_mask.sum() > 100:
        fg_pixels_input = input_resized[fg_mask]
        fg_pixels_render = render_rgb[fg_mask]
        mse_fg = float(np.mean((fg_pixels_input.astype(np.float64) / 255.0
                                 - fg_pixels_render.astype(np.float64) / 255.0) ** 2))
        metrics["psnr_fg"] = 99.0 if mse_fg <= 0 else float(10.0 * np.log10(1.0 / mse_fg))
        metrics["fg_coverage"] = float(fg_mask.sum() / fg_mask.size)
    else:
        metrics["psnr_fg"] = float("nan")
        metrics["fg_coverage"] = float(fg_mask.sum() / fg_mask.size)

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(PROJECT_ROOT / "data" / "raw_image.jpg"))
    ap.add_argument("--glb", default=str(PROJECT_ROOT / "output" / "scene_assembly" / "raw_image_assembled.glb"))
    ap.add_argument("--camera-json", default=str(PROJECT_ROOT / "output" / "camera_estimation" / "raw_image_perspective_fields.json"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "eval"))
    ap.add_argument("--max-width", type=int, default=1024,
                    help="Cap render width to keep eval fast. Input is downsampled to match.")
    ap.add_argument("--tag", default="",
                    help="Free-form label written into the manifest record (e.g. 'after-amodal').")
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    glb_path = Path(args.glb).resolve()
    cam_json_path = Path(args.camera_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")
    if not glb_path.exists():
        raise SystemExit(f"GLB not found: {glb_path}")

    items = load_objects(glb_path)
    if not items:
        raise SystemExit(f"No mesh geometries in: {glb_path}")

    vfov_deg, roll_deg, pitch_deg = 60.0, 0.0, 0.0
    if cam_json_path.exists():
        with open(cam_json_path, "r", encoding="utf-8") as f:
            pf = json.load(f)
        vfov_deg = float(pf.get("pred_general_vfov", pf.get("pred_vfov", vfov_deg)))
        roll_deg = float(pf.get("pred_roll", 0.0))
        pitch_deg = float(pf.get("pred_pitch", 0.0))

    # Prefer the camera that fitted_transform actually used (its render_meta).
    cam_dist, cam_elev, cam_azim = 0.0, 0.0, 0.0
    fitted_cam_path = _default_fitted_camera_json(image_path.stem)
    if fitted_cam_path.exists():
        with open(fitted_cam_path, "r", encoding="utf-8") as f:
            rm = json.load(f)
        fc = rm.get("camera", {})
        cam_dist = float(fc.get("dist", 0.0))
        cam_elev = float(fc.get("elev", 0.0))
        cam_azim = float(fc.get("azim", 0.0))
        if "fov" in fc:
            vfov_deg = float(fc["fov"])
            roll_deg = 0.0
            pitch_deg = 0.0
        print(f"using fitted camera: dist={cam_dist:.3f} elev={cam_elev:.2f} azim={cam_azim:.2f} fov={vfov_deg:.2f}")

    input_rgb = np.array(Image.open(image_path).convert("RGB"))
    ih, iw = input_rgb.shape[:2]

    w = min(args.max_width, iw)
    h = max(1, int(round(w * ih / iw)))
    print(f"render res {w}x{h} (input {iw}x{ih})")
    print(f"vfov={vfov_deg:.2f}, roll={roll_deg:.2f}, pitch={pitch_deg:.2f}")

    bg_path = PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"
    use_bg = bg_path.exists()
    render_rgb = render_input_view(
        items, w, h, vfov_deg, roll_deg, pitch_deg,
        cam_dist=cam_dist, cam_elev=cam_elev, cam_azim=cam_azim,
        skip_background_planes=use_bg,
    )
    if use_bg:
        from scripts.verify_motion import composite_over_background  # noqa: E402
        render_rgb = composite_over_background(render_rgb, bg_path)

    image_stem = image_path.stem
    run_out_dir = out_dir / image_stem
    run_out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_rgb).save(run_out_dir / "reproj.png")

    # Side-by-side input | reproj
    input_resized = np.array(Image.fromarray(input_rgb).resize((w, h), Image.BILINEAR))
    side_by_side = np.concatenate([input_resized, render_rgb], axis=1)
    Image.fromarray(side_by_side).save(run_out_dir / "reproj_compare.png")

    metrics = compute_metrics(input_rgb, render_rgb)
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "tag": args.tag,
        "image": str(image_path),
        "image_sha16": _file_sha256(image_path),
        "glb": str(glb_path),
        "glb_sha16": _file_sha256(glb_path),
        "render_resolution": [w, h],
        "input_resolution": [iw, ih],
        "camera": {
            "vfov_deg": vfov_deg,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
        },
        "object_count": len(items),
        "metrics": metrics,
        "git_commit": _git_commit(),
    }

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"appended record to {manifest_path}")
    print(f"saved reproj.png + reproj_compare.png to {run_out_dir}")


if __name__ == "__main__":
    main()
