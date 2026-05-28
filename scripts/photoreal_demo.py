"""Photoreal demo for presentation: pixel cutouts on LaMa background.

Strategy: 3D mesh quality (Hunyuan-Paint output) is too rough for a polished
"reproj matches input" panel. Instead we showcase the SYSTEM (segmentation +
amodal mask + LaMa background inpaint + scene composition) by:

  - Cutting each object out of the original image using its visible mask.
  - Placing those cutouts over the LaMa-inpainted background.
  - For demos (removed / wiggle), translate or omit individual cutouts.

The 3D meshes live in a separate panel (`isolated_strip.png`, produced by
`render_isolated.py`) so the audience still sees that geometry was extracted.

Outputs:
  - photoreal_full.png     — recovered input (cutouts in source positions).
  - photoreal_grid.png     — INPUT + FULL + removed-X panels.
  - photoreal_wiggle.gif   — each cutout oscillates while LaMa bg stays still.
  - photoreal_explode.png  — cutouts pulled apart radially so each is clearly
    separable (sells the editability story in one frame).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tool.stuff_filter import class_from_filename, load_stuff_keywords  # noqa: E402


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


def _feather_alpha(mask: np.ndarray, feather_px: int = 3) -> np.ndarray:
    """Soft edge for natural compositing."""
    from PIL import ImageFilter
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return np.array(blurred).astype(np.uint8)


def _paste_rgba(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite (rgb, alpha) onto canvas at (x, y). Clips to canvas."""
    H, W = canvas.shape[:2]
    h, w = rgb.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return canvas
    sx0 = x0 - x; sy0 = y0 - y
    sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
    rgb_clip = rgb[sy0:sy1, sx0:sx1]
    a_clip = alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    patch = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = (
        rgb_clip * a_clip[..., None] + patch * (1 - a_clip[..., None])
    ).astype(np.uint8)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw_image0.jpg"))
    ap.add_argument("--background", default=str(PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"))
    ap.add_argument("--mask-dir", default=str(PROJECT_ROOT / "output" / "mask_postprocess"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--wiggle-frames", type=int, default=48)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_pil = Image.open(args.input).convert("RGB")
    bg_pil = Image.open(args.background).convert("RGB")
    iw, ih = input_pil.size
    target_w = args.width
    target_h = int(round(ih * target_w / iw))
    input_resized = input_pil.resize((target_w, target_h), Image.LANCZOS)
    bg_resized = bg_pil.resize((target_w, target_h), Image.LANCZOS)
    input_np = np.array(input_resized)
    bg_np = np.array(bg_resized)
    sx = target_w / iw
    sy = target_h / ih
    print(f"canvas {target_w}x{target_h}")

    # Movable masks.
    stuff_kws = load_stuff_keywords()
    cutouts: list[dict] = []
    for mp in sorted(Path(args.mask_dir).glob("object_*_mask.npy")):
        cname = class_from_filename(mp.stem)
        if any(kw in cname for kw in stuff_kws):
            continue
        idx = int(mp.stem.split("_")[1])
        mask_full = np.load(mp)
        if mask_full.ndim == 3:
            mask_full = mask_full[..., 0]
        mask_full = (mask_full > 0).astype(np.uint8)
        if mask_full.sum() == 0:
            continue
        # Resize mask to canvas resolution.
        m_pil = Image.fromarray((mask_full * 255).astype(np.uint8))
        m_resized = np.array(m_pil.resize((target_w, target_h), Image.NEAREST))
        m_resized = (m_resized > 127).astype(np.uint8)
        ys, xs = np.where(m_resized > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        rgb_crop = input_np[y0:y1, x0:x1].copy()
        alpha_crop = _feather_alpha(m_resized[y0:y1, x0:x1], feather_px=2)
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        cutouts.append({
            "idx": idx,
            "class": cname,
            "rgb": rgb_crop,
            "alpha": alpha_crop,
            "bbox": (x0, y0, x1, y1),
            "center": center,
            "size": (x1 - x0, y1 - y0),
        })
        print(f"  idx={idx} ({cname}) bbox={x0,y0,x1,y1} size={x1-x0}x{y1-y0}")

    if not cutouts:
        print("no movable cutouts found")
        return

    # Depth-order so larger / lower objects draw first (behind).
    # Use mask-bbox bottom y as depth proxy: higher y (closer to bottom) is in front.
    cutouts_sorted = sorted(cutouts, key=lambda c: c["bbox"][3])

    def _compose(omit: set[int], translations: dict[int, tuple[int, int]] | None = None) -> np.ndarray:
        out = bg_np.copy()
        for c in cutouts_sorted:
            if c["idx"] in omit:
                continue
            x0 = c["bbox"][0]; y0 = c["bbox"][1]
            if translations and c["idx"] in translations:
                dx, dy = translations[c["idx"]]
                x0 = x0 + dx; y0 = y0 + dy
            _paste_rgba(out, c["rgb"], c["alpha"], x0, y0)
        return out

    # FULL.
    full = _compose(omit=set())
    Image.fromarray(full).save(out_dir / "photoreal_full.png")
    print("saved photoreal_full.png")

    # Per-object removal grid.
    panels: list[Image.Image] = []
    panels.append(_annotate(Image.fromarray(input_np), "INPUT (raw_image0.jpg)"))
    panels.append(_annotate(Image.fromarray(full), "FULL (cutouts on LaMa background)"))
    for c in cutouts:
        partial = _compose(omit={c["idx"]})
        panels.append(_annotate(
            Image.fromarray(partial),
            f"removed: obj{c['idx']:03d} {c['class'].split('_')[0]}  (LaMa fills the hole)",
        ))

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
    grid.save(out_dir / "photoreal_grid.png")
    print(f"saved photoreal_grid.png ({cols}x{rows})")

    # Exploded view: pull cutouts radially outward from scene centroid.
    scene_cx = np.mean([c["center"][0] for c in cutouts])
    scene_cy = np.mean([c["center"][1] for c in cutouts])
    explode_translations: dict[int, tuple[int, int]] = {}
    explode_distance_px = int(0.10 * target_w)
    for c in cutouts:
        cx, cy = c["center"]
        dx = cx - scene_cx
        dy = cy - scene_cy
        n = max(1.0, math.hypot(dx, dy))
        explode_translations[c["idx"]] = (
            int(round(dx / n * explode_distance_px)),
            int(round(dy / n * explode_distance_px)),
        )
    explode = _compose(omit=set(), translations=explode_translations)
    Image.fromarray(explode).save(out_dir / "photoreal_explode.png")
    print("saved photoreal_explode.png")

    # Wiggle: each object oscillates along its own random horizontal direction.
    rng = np.random.RandomState(7)
    movable_ids = [c["idx"] for c in cutouts]
    dirs = rng.randn(len(movable_ids), 2)
    # bias to horizontal (objects on counter shouldn't levitate)
    dirs[:, 1] *= 0.25
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
    phases = rng.uniform(0, 2 * math.pi, size=len(movable_ids))
    id_to_motion = dict(zip(movable_ids, zip(dirs, phases)))

    # Amplitude: ~12% of canvas width.
    amp = 0.12 * target_w

    print(f"rendering photoreal_wiggle.gif ({args.wiggle_frames} frames)...")
    gif_frames: list[np.ndarray] = []
    for f in range(args.wiggle_frames):
        t = 2.0 * math.pi * f / args.wiggle_frames
        trans = {}
        for idx in movable_ids:
            d, phase = id_to_motion[idx]
            s = math.sin(t + phase)
            trans[idx] = (int(round(d[0] * amp * s)), int(round(d[1] * amp * s)))
        frame = _compose(omit=set(), translations=trans)
        gif_frames.append(frame)
    imageio.mimsave(out_dir / "photoreal_wiggle.gif", gif_frames, duration=0.06, loop=0)
    print("saved photoreal_wiggle.gif")


if __name__ == "__main__":
    main()
