from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from image_to_world.common import save_json
from image_to_world.geometry import rotation_matrix_xyz_deg
from image_to_world.visualization.camera_viz import deterministic_color, set_equal_limits_3d


def cuboid_corners(size_xyz: list[float], position_xyz: list[float], rotation_xyz_deg: list[float]) -> np.ndarray:
    sx, sy, sz = [max(1e-4, float(v)) for v in size_xyz]
    px, py, pz = [float(v) for v in position_xyz]
    local = np.array([
        [-0.5 * sx, -0.5 * sy, -0.5 * sz],
        [0.5 * sx, -0.5 * sy, -0.5 * sz],
        [0.5 * sx, 0.5 * sy, -0.5 * sz],
        [-0.5 * sx, 0.5 * sy, -0.5 * sz],
        [-0.5 * sx, -0.5 * sy, 0.5 * sz],
        [0.5 * sx, -0.5 * sy, 0.5 * sz],
        [0.5 * sx, 0.5 * sy, 0.5 * sz],
        [-0.5 * sx, 0.5 * sy, 0.5 * sz],
    ], dtype=np.float64)
    rotated = (rotation_matrix_xyz_deg(rotation_xyz_deg) @ local.T).T
    return rotated + np.array([px, py, pz], dtype=np.float64)


def cuboid_edges() -> list[tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def to_visual_xyz(points_xyz: np.ndarray) -> np.ndarray:
    return np.stack([points_xyz[:, 0], points_xyz[:, 2], points_xyz[:, 1]], axis=1)


def load_pointcloud(pointcloud_path: str | None) -> np.ndarray | None:
    if not pointcloud_path:
        return None
    path = Path(pointcloud_path)
    if not path.exists():
        return None
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    return points.astype(np.float64)


def draw_cuboid_wireframe(ax: plt.Axes, corners_visual: np.ndarray, color: tuple[float, float, float]) -> None:
    for start_idx, end_idx in cuboid_edges():
        segment = corners_visual[[start_idx, end_idx]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=1.4, alpha=0.95)


def render_layout_visualization(*, raw_image_path: Path, placements: list[dict[str, Any]], png_path: Path, summary_path: Path) -> None:
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

    all_points: list[np.ndarray] = []
    object_summaries: list[dict[str, Any]] = []
    for idx, obj in enumerate(placements):
        color = deterministic_color(idx)
        pointcloud = load_pointcloud(obj.get("source_paths", {}).get("pointcloud_path"))

        pos = obj.get("pseudo_world", {}).get("position_xyz", [0, 0, 0])
        rot = obj.get("pseudo_world", {}).get("rotation_euler_xyz_deg", [0, 0, 0])
        scale = obj.get("pseudo_world", {}).get("scale_xyz", [1, 1, 1])
        corners_world = cuboid_corners(scale, pos, rot)
        corners_visual = to_visual_xyz(corners_world)
        center = corners_visual.mean(axis=0)

        for ax in axes:
            draw_cuboid_wireframe(ax, corners_visual, color)

        for ax_idx, ax in enumerate(axes):
            if ax_idx in (0, 1):
                ax.text(center[0], center[1], center[2], f"[{obj.get('id', idx)}]", fontsize=7, color="black")

        all_points.append(corners_visual)
        object_summaries.append({
            "id": obj.get("id", idx),
            "class_name": obj.get("class_name"),
            "num_points": int(pointcloud.shape[0]) if pointcloud is not None else 0,
            "color_rgb": list(color),
            "center_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
            "yaw_deg": float(rot[1]) if len(rot) > 1 else 0.0,
            "size_xyz": [float(scale[0]), float(scale[1]), float(scale[2])],
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

    fig.suptitle("Layout OBB Validation", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    save_json(summary_path, {
        "raw_image_path": str(raw_image_path) if raw_image_path.exists() else None,
        "num_placements": len(placements),
        "output_png": str(png_path),
        "objects": object_summaries,
        "notes": [
            "Each panel matches the camera_viz point-cloud validation format: perspective, top, front, and side views.",
            "Only yaw-only OBB wireframes are rendered in the validation views.",
            "Object point clouds are still used internally for fitting but are not drawn here.",
        ],
    })
