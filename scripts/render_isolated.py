"""Per-object isolated turntable renders.

For each non-background node in the assembled GLB, render the object alone
on a clean background from a fixed 3/4 angle. Produces one PNG per object
plus a horizontal strip combining them — useful as a "we extracted these
3 objects" cards on the final deliverable image.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import pyrender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_motion import (  # noqa: E402
    add_lights,
    build_pyrender_scene,
    look_at,
    _shorten_label,
)


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_isolated(
    items: list[tuple[str, trimesh.Trimesh]],
    width: int,
    height: int,
    vfov_deg: float = 35.0,
) -> dict[str, np.ndarray]:
    """Render each movable object alone, centered, on white bg."""
    results: dict[str, np.ndarray] = {}
    for name, geom in items:
        if "background" in name:
            continue
        bb = geom.bounds
        center = (bb[0] + bb[1]) / 2
        extent = float(np.linalg.norm(bb[1] - bb[0]))
        # 3/4 angle view from upper-front
        view = np.array([0.5, 0.4, 1.0])
        view = view / np.linalg.norm(view)
        eye = center + view * extent * 1.4

        pscene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0), ambient_light=[0.75, 0.75, 0.75])
        pm = pyrender.Mesh.from_trimesh(geom, smooth=True)
        pscene.add(pm, name=name)
        cam = pyrender.PerspectiveCamera(
            yfov=math.radians(vfov_deg), aspectRatio=width / height,
        )
        pose = look_at(eye, center, up_world=np.array([0.0, 1.0, 0.0]))
        pscene.add(cam, pose=pose)
        add_lights(pscene, center, extent)

        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            color, _ = r.render(pscene)
        finally:
            r.delete()
        results[name] = color[..., :3].copy()
    return results


def make_strip(panels: dict[str, np.ndarray], pad: int = 16) -> Image.Image:
    if not panels:
        raise ValueError("no panels")
    items = list(panels.items())
    h, w = items[0][1].shape[:2]
    bar = max(28, int(h * 0.06))
    n = len(items)
    canvas = Image.new("RGB", (n * w + (n + 1) * pad, h + bar + 2 * pad), (245, 245, 245))
    for idx, (name, img) in enumerate(items):
        x = pad + idx * (w + pad)
        canvas.paste(Image.fromarray(img), (x, pad))
        d = ImageDraw.Draw(canvas)
        label = _shorten_label(name)
        font = _font(max(16, int(h * 0.04)))
        d.rectangle(
            [x, pad + h, x + w, pad + h + bar],
            fill=(20, 25, 40),
        )
        d.text((x + 8, pad + h + max(4, int(bar * 0.15))), label, fill=(255, 255, 255), font=font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", default=str(PROJECT_ROOT / "output" / "scene_assembly" / "raw_image0_assembled.glb"))
    ap.add_argument("--source-dir", default=str(PROJECT_ROOT / "output" / "mesh_texturing"),
                    help="Prefer canonical Hunyuan-Paint meshes (unscaled, original proportions) over the assembled GLB (whose vertices were scaled by fitted_transform — depth axis often gets squashed).")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "demo" / "raw_image0"))
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=520)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, trimesh.Trimesh]] = []
    source_dir = Path(args.source_dir).resolve()
    canonical_files = sorted(source_dir.glob("*_remeshed_textured.glb"))
    if canonical_files:
        try:
            from src.tool.stuff_filter import load_stuff_keywords
            stuff_kws = load_stuff_keywords()
        except Exception:
            stuff_kws = []
        for p in canonical_files:
            # filename like "000_bottle_0.988_remeshed_textured.glb"
            stem = p.stem.replace("_remeshed_textured", "")
            parts = stem.split("_", 1)
            if len(parts) == 2:
                idx_str, rest = parts
                # drop trailing "_<score>"
                cls = "_".join(rest.split("_")[:-1])
                node = f"object_{int(idx_str):03d}_{cls}"
            else:
                node = stem
            cname_low = node.lower()
            if any(kw in cname_low for kw in stuff_kws):
                continue
            g = trimesh.load(p, force="scene")
            for _, mesh in g.geometry.items():
                items.append((node, mesh))
        print(f"loaded {len(items)} canonical (Hunyuan-Paint) meshes from {source_dir.name}")
    else:
        glb_path = Path(args.glb).resolve()
        scene = trimesh.load(glb_path, force="scene")
        items = [(n, g) for n, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh)]
        print(f"loaded {len(items)} items from assembled {glb_path.name}")

    panels = render_isolated(items, args.width, args.height)
    for name, img in panels.items():
        Image.fromarray(img).save(out_dir / f"isolated_{name}.png")

    strip = make_strip(panels)
    strip_path = out_dir / "isolated_strip.png"
    strip.save(strip_path)
    print(f"saved isolated strip: {strip_path}")


if __name__ == "__main__":
    main()
