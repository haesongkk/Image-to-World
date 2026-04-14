from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from image_to_world.common import save_json


PALETTE = [
    (0.90, 0.25, 0.25),
    (0.25, 0.55, 0.95),
    (0.20, 0.75, 0.35),
    (0.95, 0.70, 0.20),
    (0.65, 0.35, 0.90),
    (0.20, 0.80, 0.80),
    (0.95, 0.45, 0.70),
    (0.60, 0.60, 0.20),
    (0.90, 0.55, 0.15),
    (0.45, 0.75, 0.95),
]


def deterministic_color(idx: int) -> tuple[float, float, float]:
    return PALETTE[idx % len(PALETTE)]


def set_equal_limits_3d(ax, points: np.ndarray, pad_ratio: float = 0.08) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float((maxs - mins).max()), 1e-3)
    half = 0.5 * span * (1.0 + pad_ratio)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render_depth_object_pointcloud(
    *,
    object_pointclouds: list[dict[str, Any]],
    png_path: Path,
    summary_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    view_specs = [
        ("Perspective", 22, -58),
        ("Top", 90, -90),
        ("Front", 0, -90),
        ("Side", 0, 0),
    ]

    all_points = []
    object_summaries = []
    for idx, obj in enumerate(object_pointclouds):
        pointcloud_path_raw = obj.get("pointcloud_path")
        if not pointcloud_path_raw:
            continue
        pointcloud_path = Path(str(pointcloud_path_raw))
        if not pointcloud_path.exists():
            continue
        points = np.load(pointcloud_path)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            continue
        color = deterministic_color(idx)
        for ax in axes:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3.5, color=[color], alpha=0.75, depthshade=False)
        center = points.mean(axis=0)
        for ax_idx, ax in enumerate(axes):
            if ax_idx in (0, 1):
                ax.text(center[0], center[1], center[2], f"[{obj.get('id', idx)}]", fontsize=7, color="black")
        all_points.append(points)
        object_summaries.append({
            "id": obj.get("id", idx),
            "class_name": obj.get("class_name"),
            "num_points": int(points.shape[0]),
            "color_rgb": list(color),
            "center_xyz": center.tolist(),
            "pointcloud_path": str(pointcloud_path),
        })

    all_concat = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float64)
    for ax, (title, elev, azim) in zip(axes, view_specs):
        ax.set_title(f"{title} View")
        ax.set_xlabel("world X")
        ax.set_ylabel("world Z")
        ax.set_zlabel("world Y")
        ax.view_init(elev=elev, azim=azim)
        if all_points:
            set_equal_limits_3d(ax, all_concat)

    fig.suptitle("Depth Object Point Cloud", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    save_json(summary_path, {
        "output_png": str(png_path),
        "num_objects": len(object_summaries),
        "objects": object_summaries,
        "notes": [
            "Point coordinates are precomputed in estimate_depth and loaded here for rendering only.",
        ],
    })
