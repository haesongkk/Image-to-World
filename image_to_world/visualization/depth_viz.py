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


def sample_mask_points(mask: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask == 1)
    if xs.size == 0:
        return xs, ys
    if xs.size <= max_points:
        return xs.astype(np.float64), ys.astype(np.float64)
    select = np.linspace(0, xs.size - 1, num=max_points, dtype=np.int32)
    return xs[select].astype(np.float64), ys[select].astype(np.float64)


def depth_to_pseudo_points(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    depth_values: np.ndarray,
    image_w: int,
    image_h: int,
    global_depth_min: float,
    global_depth_max: float,
    xy_scale: float,
    z_scale: float,
) -> np.ndarray:
    if global_depth_max - global_depth_min < 1e-8:
        z_norm = np.full_like(depth_values, 0.5, dtype=np.float64)
    else:
        z_norm = np.clip((depth_values - global_depth_min) / (global_depth_max - global_depth_min), 0.0, 1.0)
    cx = image_w * 0.5
    cy = image_h * 0.5
    xw = ((xs - cx) / max(float(image_w), 1.0)) * xy_scale
    yw = -((ys - cy) / max(float(image_h), 1.0)) * xy_scale * (float(image_h) / max(float(image_w), 1.0))
    zw = (1.0 - z_norm) * z_scale
    return np.stack([xw, zw, yw], axis=1)


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
    depth: np.ndarray,
    annotations: list[dict[str, Any]],
    mask_loader,
    png_path: Path,
    summary_path: Path,
    max_points_per_object: int = 1200,
) -> None:
    image_h, image_w = depth.shape[:2]
    global_depth_min = float(np.nanmin(depth))
    global_depth_max = float(np.nanmax(depth))
    xy_scale = 2.0
    z_scale = xy_scale * 1.35
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
    for idx, ann in enumerate(annotations):
        mask = mask_loader(ann.get("mask_path"), (image_h, image_w)) if ann.get("mask_path") else None
        if mask is None:
            continue
        xs, ys = sample_mask_points(mask, max_points_per_object)
        if xs.size == 0:
            continue
        depth_values = depth[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float64)
        valid = np.isfinite(depth_values)
        if not valid.any():
            continue
        xs = xs[valid]
        ys = ys[valid]
        depth_values = depth_values[valid]
        points = depth_to_pseudo_points(
            xs=xs,
            ys=ys,
            depth_values=depth_values,
            image_w=image_w,
            image_h=image_h,
            global_depth_min=global_depth_min,
            global_depth_max=global_depth_max,
            xy_scale=xy_scale,
            z_scale=z_scale,
        )
        color = deterministic_color(idx)
        for ax in axes:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3.5, color=[color], alpha=0.75, depthshade=False)
        center = points.mean(axis=0)
        for ax_idx, ax in enumerate(axes):
            if ax_idx in (0, 1):
                ax.text(center[0], center[1], center[2], f"[{ann.get('id', idx)}]", fontsize=7, color="black")
        all_points.append(points)
        object_summaries.append({
            "id": ann.get("id", idx),
            "class_name": ann.get("class_name"),
            "num_points": int(points.shape[0]),
            "color_rgb": list(color),
            "center_xyz": center.tolist(),
        })

    all_concat = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float64)
    for ax, (title, elev, azim) in zip(axes, view_specs):
        ax.set_title(f"{title} View")
        ax.set_xlabel("image plane X")
        ax.set_ylabel("depth")
        ax.set_zlabel("image plane Y")
        ax.view_init(elev=elev, azim=azim)
        if all_points:
            set_equal_limits_3d(ax, all_concat)

    fig.suptitle("Depth Object Point Cloud Multi-View", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    save_json(summary_path, {
        "image_size_wh": [image_w, image_h],
        "output_png": str(png_path),
        "num_objects": len(object_summaries),
        "objects": object_summaries,
        "notes": [
            "Each object mask is rendered as a differently colored 3D point cloud in a shared image/depth coordinate frame.",
            "X and Y stay close to the image plane, while depth extrudes the scene outward.",
            "Views are perspective, top, front, and side.",
        ],
    })
