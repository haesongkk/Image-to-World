"""Honest presentation comparison.

Real improvements over 6126f0c baseline (which already had 3-object detection,
Hunyuan-Paint textures, and object-level editability):

  1. Object PLACEMENT — meshes now sit at correct image positions.
  2. Occluded-region BACKGROUND recovery via LaMa inpainting.

Everything else (detection count, texture, per-object editing capability)
was already present in 6126f0c.

Layout:
  HEADER
  ROW 1  INPUT photo            |  NEW placement (placed_full.png)
  ROW 2  BASELINE 6126f0c        |  NEW background recovery (clean_background.png)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _font(size: int):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _wrap_text(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _resize_letterbox(img: Image.Image, w: int, h: int, bg=(255, 255, 255)) -> Image.Image:
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _panel(img_path: Path, w: int, h: int, title: str, subtitle: str = "") -> Image.Image:
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
    else:
        img = Image.new("RGB", (w, h), (210, 210, 210))
        d0 = ImageDraw.Draw(img)
        d0.text((10, 10), f"(missing) {img_path.name}", fill=(80, 80, 80), font=_font(16))
    img = _resize_letterbox(img, w, h, bg=(20, 24, 36))
    title_h = 48
    sub_h = 38 if subtitle else 0
    out = Image.new("RGB", (w, h + title_h + sub_h), (20, 24, 36))
    out.paste(img, (0, title_h + sub_h))
    d = ImageDraw.Draw(out)
    d.text((14, 10), title, fill=(255, 255, 255), font=_font(24))
    if subtitle:
        lines = _wrap_text(d, subtitle, _font(15), w - 28)
        for i, line in enumerate(lines[:2]):
            d.text((14, title_h + i * 19), line, fill=(180, 200, 255), font=_font(15))
    return out


def _header(width: int) -> Image.Image:
    h = 130
    img = Image.new("RGB", (width, h), (12, 18, 38))
    d = ImageDraw.Draw(img)
    d.text((24, 14), "Image-to-World — overnight progress", fill=(255, 255, 255), font=_font(36))
    d.text((24, 62), "Improvements vs 6126f0c artifacts (same source image):", fill=(190, 210, 255), font=_font(17))
    d.text((24, 86),
           "1) all 3 objects visible at the source-camera angle (was: olive oil hidden)",
           fill=(140, 220, 200), font=_font(16))
    d.text((24, 106),
           "2) occluded-region background recovery via LaMa inpainting (new)",
           fill=(140, 220, 200), font=_font(16))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = PROJECT_ROOT / "output" / "demo" / "raw_image0"
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw_image0.jpg"))
    ap.add_argument("--scene", default=str(base / "placed_full.png"))
    ap.add_argument("--baseline", default=str(PROJECT_ROOT / "doc" / "260505" / "screenshot.png"))
    ap.add_argument("--lama-bg", default=str(PROJECT_ROOT / "output" / "background_inpaint" / "clean_background.png"))
    ap.add_argument("--out", default=str(base / "presentation.png"))
    ap.add_argument("--col-w", type=int, default=920)
    ap.add_argument("--col-h", type=int, default=620)
    args = ap.parse_args()

    pad = 18
    cw, ch = args.col_w, args.col_h
    canvas_w = 2 * cw + 3 * pad

    header = _header(canvas_w)

    p_input = _panel(
        Path(args.input), cw, ch,
        "INPUT",
        "raw_image0.jpg — same source photo used by both runs.",
    )
    p_scene = _panel(
        Path(args.scene), cw, ch,
        "NOW — source-camera view",
        "Air fryer + olive oil + wine bottle all visible at the source-camera angle (mask-driven mesh placement).",
    )
    p_base = _panel(
        Path(args.baseline), cw, ch,
        "BEFORE — 6126f0c artifact",
        "3 meshes were extracted but at the source-camera angle the olive oil was hidden behind the air fryer.",
    )
    p_lama = _panel(
        Path(args.lama_bg), cw, ch,
        "NEW — background recovery (LaMa)",
        "Movable-object pixels removed and inpainted. Reused under every per-object edit so removals don't leave holes.",
    )

    row_h = p_input.height
    total_h = header.height + 2 * row_h + 3 * pad
    canvas = Image.new("RGB", (canvas_w, total_h), (235, 238, 246))
    y = 0
    canvas.paste(header, (0, y)); y += header.height + pad
    canvas.paste(p_input, (pad, y))
    canvas.paste(p_scene, (2 * pad + cw, y))
    y += row_h + pad
    canvas.paste(p_base, (pad, y))
    canvas.paste(p_lama, (2 * pad + cw, y))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"saved {out_path}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
