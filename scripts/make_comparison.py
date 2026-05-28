"""Side-by-side comparison: baseline (commit 6126f0c) vs current night-run.

Produces a single PNG that tells the improvement story at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _resize_letterbox(img: Image.Image, w: int, h: int, bg=(245, 245, 245)) -> Image.Image:
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _annotate(img: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    w, h = img.size
    title_h = 40
    sub_h = 28 if subtitle else 0
    bar = title_h + sub_h + 4
    out = Image.new("RGB", (w, h + bar), (0, 0, 0))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((12, 8), title, fill=(255, 255, 255), font=_font(20))
    if subtitle:
        d.text((12, 8 + title_h), subtitle, fill=(180, 180, 180), font=_font(15))
    return out


def _header_strip(width: int, title: str, subtitle: str) -> Image.Image:
    h = 90
    img = Image.new("RGB", (width, h), (25, 30, 50))
    d = ImageDraw.Draw(img)
    d.text((20, 16), title, fill=(255, 255, 255), font=_font(28))
    d.text((20, 54), subtitle, fill=(200, 210, 240), font=_font(16))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw_image0.jpg"))
    ap.add_argument("--baseline", default=str(PROJECT_ROOT / "doc" / "260505" / "screenshot.png"))
    ap.add_argument("--reproj", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0" / "placed_full.png"))
    ap.add_argument("--grid", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0" / "placed_grid.png"))
    ap.add_argument("--isolated", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0" / "isolated_strip.png"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0" / "comparison.png"))
    ap.add_argument("--panel-w", type=int, default=720)
    ap.add_argument("--panel-h", type=int, default=540)
    args = ap.parse_args()

    panel_specs = [
        ("INPUT", "raw_image0.jpg (the source photo)", Path(args.input)),
        ("NEW 3D scene", "3 textured Hunyuan-Paint meshes placed on LaMa-inpainted background", Path(args.reproj)),
        ("BASELINE 6126f0c", "2 untextured meshes in Blender, no scene context", Path(args.baseline)),
        ("Per-object editability", "remove any object → LaMa-inpainted background appears behind it", Path(args.grid)),
    ]
    panels = []
    for title, subtitle, p in panel_specs:
        if p.exists():
            img = Image.open(p).convert("RGB")
        else:
            img = Image.new("RGB", (args.panel_w, args.panel_h), (200, 200, 200))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"(missing) {p.name}", fill=(0, 0, 0), font=_font(18))
        img = _resize_letterbox(img, args.panel_w, args.panel_h)
        panels.append(_annotate(img, title, subtitle))

    pw, ph = panels[0].size
    pad = 14

    # Optional bottom strip: per-object isolated turntable cards.
    isolated_path = Path(args.isolated)
    isolated_img = None
    if isolated_path.exists():
        raw = Image.open(isolated_path).convert("RGB")
        target_w = 2 * pw + pad
        scale = target_w / raw.width
        target_h = int(raw.height * scale)
        isolated_img = raw.resize((target_w, target_h), Image.BILINEAR)
        isolated_img = _annotate(
            isolated_img,
            "Extracted objects (canonical Hunyuan-Paint output)",
            "Each is a separate textured GLB node — move, rotate, scale independently in Blender/Unity/Three.js.",
        )

    header = _header_strip(
        2 * pw + 3 * pad,
        "Image-to-World — overnight run",
        "raw_image0.jpg  ·  3 objects extracted (vs 2 in baseline)  ·  per-object Hunyuan3D mesh + Paint texture  ·  LaMa background inpaint  ·  editable scene composition",
    )

    total_h = header.height + 2 * ph + 3 * pad + (isolated_img.height + pad if isolated_img else 0)
    canvas = Image.new(
        "RGB",
        (2 * pw + 3 * pad, total_h),
        (240, 240, 240),
    )
    canvas.paste(header, (0, 0))
    base_y = header.height + pad
    for idx, panel in enumerate(panels):
        r, c = idx // 2, idx % 2
        x = pad + c * (pw + pad)
        y = base_y + r * (ph + pad)
        canvas.paste(panel, (x, y))
    if isolated_img is not None:
        canvas.paste(isolated_img, (pad, base_y + 2 * (ph + pad)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
