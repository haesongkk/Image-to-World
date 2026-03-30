from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from image_to_world.common import save_json


def deterministic_color(idx: int) -> tuple[float, float, float]:
    palette = [
        (0.90, 0.25, 0.25), (0.25, 0.55, 0.95), (0.20, 0.75, 0.35), (0.95, 0.70, 0.20),
        (0.65, 0.35, 0.90), (0.20, 0.80, 0.80), (0.95, 0.45, 0.70), (0.60, 0.60, 0.20),
        (0.90, 0.55, 0.15), (0.45, 0.75, 0.95),
    ]
    return palette[idx % len(palette)]


def render_layout_visualization(*, raw_image_path: Path, placements: list[dict[str, Any]], png_path: Path, summary_path: Path) -> None:
    image = Image.open(raw_image_path).convert("RGB") if raw_image_path.exists() else None

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_front = fig.add_subplot(gs[1, 1])

    if image is not None:
        ax_img.imshow(image)
    else:
        ax_img.set_facecolor((0.08, 0.08, 0.08))
        ax_img.text(0.5, 0.5, "raw_image.jpg not found", ha="center", va="center", color="white", transform=ax_img.transAxes)

    xs, zs, ys = [], [], []
    for i, obj in enumerate(placements):
        color = deterministic_color(i)
        bbox = obj.get("bbox_xyxy")
        pos = obj.get("pseudo_world", {}).get("position_xyz", [0, 0, 0])
        scale = obj.get("pseudo_world", {}).get("scale_xyz", [1, 1, 1])
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
            ax_img.text(x1, max(0, y1 - 6), f"[{obj.get('id', i)}] {obj.get('class_name')}\nz={pos[2]:.2f}", fontsize=8, color="white", bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, pad=2.0))
        xs.append(float(pos[0]))
        zs.append(float(pos[2]))
        ys.append(float(pos[1]))
        ax_top.scatter([pos[0]], [pos[2]], s=max(20.0, float(scale[0]) * 300.0), c=[color], alpha=0.85, edgecolors="black", linewidths=0.8)
        ax_front.scatter([pos[0]], [pos[1]], s=max(20.0, float(scale[1]) * 300.0), c=[color], alpha=0.85, edgecolors="black", linewidths=0.8)

    ax_img.set_title("Image Overlay (bbox + label + pseudo z)")
    ax_img.axis("off")
    ax_top.set_title("Top View (X-Z)")
    ax_top.set_xlabel("pseudo world X")
    ax_top.set_ylabel("pseudo world Z")
    ax_top.grid(True, alpha=0.25)
    ax_front.set_title("Front View (X-Y)")
    ax_front.set_xlabel("pseudo world X")
    ax_front.set_ylabel("pseudo world Y")
    ax_front.grid(True, alpha=0.25)

    fig.suptitle("Place Result Visualization", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    save_json(summary_path, {
        "raw_image_path": str(raw_image_path) if raw_image_path.exists() else None,
        "num_placements": len(placements),
        "output_png": str(png_path),
        "notes": [
            "Left: source image with bbox, labels, and pseudo depth.",
            "Top-right: pseudo X-Z top view.",
            "Bottom-right: pseudo X-Y front view.",
        ],
    })
