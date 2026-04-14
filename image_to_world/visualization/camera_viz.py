from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

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


def set_equal_limits_3d(ax: plt.Axes, points: np.ndarray, pad_ratio: float = 0.08) -> None:
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


def rotate_image(image_np: np.ndarray, roll_deg: float) -> np.ndarray:
    h, w = image_np.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, -roll_deg, 1.0)
    return cv2.warpAffine(image_np, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=np.float32)


def rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def build_world_to_camera_rotation(*, roll_deg: float, pitch_deg: float) -> np.ndarray:
    # Perspective Fields pitch sign is opposite of the simple y-up world used here.
    pitch_vis_rad = math.radians(-pitch_deg)
    roll_rad = math.radians(roll_deg)
    return rotation_z(roll_rad) @ rotation_x(pitch_vis_rad)


def project_world_points(
    points_world: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = (rotation_world_to_camera @ points_world.T).T
    z = points_camera[:, 2]
    valid = z > 1e-4
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        cam_valid = points_camera[valid]
        uv_valid = np.empty((len(cam_valid), 2), dtype=np.float32)
        uv_valid[:, 0] = fx * (cam_valid[:, 0] / cam_valid[:, 2]) + cx
        uv_valid[:, 1] = cy - fy * (cam_valid[:, 1] / cam_valid[:, 2])
        uv[valid] = uv_valid
    return uv, valid


def draw_projected_polyline(
    ax: plt.Axes,
    points_world: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation_world_to_camera: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    uv, valid = project_world_points(
        points_world,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_world_to_camera=rotation_world_to_camera,
    )
    if not np.all(valid):
        return
    ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def draw_projected_segment(
    ax: plt.Axes,
    *,
    start_world: np.ndarray,
    end_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation_world_to_camera: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    points = np.stack([start_world, end_world], axis=0).astype(np.float32)
    uv, valid = project_world_points(
        points,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_world_to_camera=rotation_world_to_camera,
    )
    if not np.all(valid):
        return
    ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def line_segment_through_box(
    *,
    point_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    width: float,
    height: float,
) -> np.ndarray | None:
    px, py = point_xy
    dx, dy = direction_xy
    eps = 1e-6
    candidates: list[tuple[float, float]] = []

    if abs(dx) > eps:
        for x in (0.0, width):
            t = (x - px) / dx
            y = py + t * dy
            if 0.0 <= y <= height:
                candidates.append((x, y))
    if abs(dy) > eps:
        for y in (0.0, height):
            t = (y - py) / dy
            x = px + t * dx
            if 0.0 <= x <= width:
                candidates.append((x, y))

    unique: list[tuple[float, float]] = []
    for candidate in candidates:
        if not any(abs(candidate[0] - existing[0]) < 1e-3 and abs(candidate[1] - existing[1]) < 1e-3 for existing in unique):
            unique.append(candidate)
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        max_dist = -1.0
        best_pair = None
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (unique[i], unique[j])
        if best_pair is None:
            return None
        unique = [best_pair[0], best_pair[1]]
    return np.asarray(unique, dtype=np.float32)


def cube_world_corners(
    *,
    center_world: np.ndarray,
    right_world: np.ndarray,
    up_world: np.ndarray,
    forward_world: np.ndarray,
    size_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    sx, sy, sz = size_xyz
    hx = sx * 0.5
    hy = sy * 0.5
    hz = sz * 0.5
    corners: list[np.ndarray] = []
    for y_sign in (-1.0, 1.0):
        for z_sign in (-1.0, 1.0):
            for x_sign in (-1.0, 1.0):
                corners.append(
                    center_world
                    + right_world * (x_sign * hx)
                    + up_world * (y_sign * hy)
                    + forward_world * (z_sign * hz)
                )
    corners_np = np.asarray(corners, dtype=np.float32)
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return corners_np, edges


def draw_reference_volume_overlay(
    ax: plt.Axes,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    horizon_y: float,
    rotation_world_to_camera: np.ndarray,
    roll_deg: float,
) -> None:
    rotation_camera_to_world = rotation_world_to_camera.T
    right_world = rotation_camera_to_world[:, 0].astype(np.float32)
    up_world = rotation_camera_to_world[:, 1].astype(np.float32)
    forward_world = rotation_camera_to_world[:, 2].astype(np.float32)

    depth_values = np.linspace(2.5, 6.5, 5, dtype=np.float32)
    grid_half_width = 1.8
    grid_half_height = 1.2
    subdivision = 4
    x_values = np.linspace(-grid_half_width, grid_half_width, subdivision + 1, dtype=np.float32)
    y_values = np.linspace(-grid_half_height, grid_half_height, subdivision + 1, dtype=np.float32)

    for depth in depth_values:
        plane_center = forward_world * depth
        # Depth slice rectangle.
        rect = np.array([
            plane_center + right_world * (-grid_half_width) + up_world * (-grid_half_height),
            plane_center + right_world * (grid_half_width) + up_world * (-grid_half_height),
            plane_center + right_world * (grid_half_width) + up_world * (grid_half_height),
            plane_center + right_world * (-grid_half_width) + up_world * (grid_half_height),
            plane_center + right_world * (-grid_half_width) + up_world * (-grid_half_height),
        ], dtype=np.float32)
        draw_projected_polyline(
            ax,
            rect,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            rotation_world_to_camera=rotation_world_to_camera,
            color="#8ce99a",
            linewidth=1.0,
            alpha=0.38,
        )
        for x_offset in x_values:
            start = plane_center + right_world * x_offset + up_world * (-grid_half_height)
            end = plane_center + right_world * x_offset + up_world * grid_half_height
            draw_projected_segment(
                ax,
                start_world=start,
                end_world=end,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rotation_world_to_camera=rotation_world_to_camera,
                color="#8ce99a",
                linewidth=0.8,
                alpha=0.22,
            )
        for y_offset in y_values:
            start = plane_center + up_world * y_offset + right_world * (-grid_half_width)
            end = plane_center + up_world * y_offset + right_world * grid_half_width
            draw_projected_segment(
                ax,
                start_world=start,
                end_world=end,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rotation_world_to_camera=rotation_world_to_camera,
                color="#8ce99a",
                linewidth=0.8,
                alpha=0.22,
            )

    front_center = forward_world * depth_values[0]
    back_center = forward_world * depth_values[-1]
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            start = front_center + right_world * (x_sign * grid_half_width) + up_world * (y_sign * grid_half_height)
            end = back_center + right_world * (x_sign * grid_half_width) + up_world * (y_sign * grid_half_height)
            draw_projected_segment(
                ax,
                start_world=start,
                end_world=end,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rotation_world_to_camera=rotation_world_to_camera,
                color="#63e6be",
                linewidth=1.1,
                alpha=0.45,
            )

    cube_center = forward_world * 4.2 + right_world * 0.1 + up_world * (-0.15)
    corners, edges = cube_world_corners(
        center_world=cube_center,
        right_world=right_world,
        up_world=up_world,
        forward_world=forward_world,
        size_xyz=(1.0, 1.0, 1.0),
    )
    uv, valid = project_world_points(
        corners,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_world_to_camera=rotation_world_to_camera,
    )
    if np.all(valid):
        for start_idx, end_idx in edges:
            segment = uv[[start_idx, end_idx]]
            ax.plot(segment[:, 0], segment[:, 1], color="#ffd43b", linewidth=2.0, alpha=0.95)
        cube_anchor = uv[7]
        ax.text(
            cube_anchor[0] + 8,
            cube_anchor[1] - 10,
            "1m cube",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="#ffd43b", edgecolor="none", alpha=0.9, pad=2.0),
        )

    axis_length = 1.0
    axis_origin = forward_world * 4.2
    axes = {
        "X": ("#ff922b", axis_origin + right_world * axis_length),
        "Y": ("#51cf66", axis_origin + up_world * axis_length),
        "Z": ("#4dabf7", axis_origin + forward_world * axis_length),
    }
    origin_uv, origin_valid = project_world_points(
        axis_origin.reshape(1, 3),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_world_to_camera=rotation_world_to_camera,
    )
    if bool(origin_valid[0]):
        for label, (color, endpoint_world) in axes.items():
            end_uv, end_valid = project_world_points(
                endpoint_world.reshape(1, 3),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rotation_world_to_camera=rotation_world_to_camera,
            )
            if not bool(end_valid[0]):
                continue
            ax.annotate("", xy=(end_uv[0, 0], end_uv[0, 1]), xytext=(origin_uv[0, 0], origin_uv[0, 1]), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
            ax.text(
                end_uv[0, 0] + 6,
                end_uv[0, 1] - 6,
                label,
                color="white",
                fontsize=9,
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.9, pad=2.0),
            )

    ax.axhline(horizon_y, color="#ff6b6b", linewidth=2.0, linestyle="--")
    ax.scatter([cx], [cy], s=70, color="#4dabf7", edgecolors="black", linewidths=0.8, zorder=5)
    ax.plot([cx, cx], [cy, horizon_y], color="#ffd43b", linewidth=2.0, linestyle=":")

    ax.text(cx + 14, cy + 12, "principal point", color="white", fontsize=9, bbox=dict(facecolor="#4dabf7", edgecolor="none", alpha=0.85, pad=2.0))
    ax.text(24, horizon_y - 8, "estimated horizon", color="#ff6b6b", fontsize=10, va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2.0))
    ax.text(
        cx + 10,
        (cy + horizon_y) * 0.5,
        f"pitch offset = {abs(cy - horizon_y):.1f}px",
        color="#fff3bf",
        fontsize=9,
        va="center",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=2.0),
    )
    ax.text(
        cx + 14,
        horizon_y + 20,
        "camera optical axis passes here",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=2.0),
    )


def render_camera_calibrated_pointcloud(
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

    all_points: list[np.ndarray] = []
    object_summaries: list[dict[str, Any]] = []
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

    fig.suptitle("Camera-Calibrated Depth Object Point Cloud", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    save_json(summary_path, {
        "output_png": str(png_path),
        "num_objects": len(object_summaries),
        "objects": object_summaries,
        "notes": [
            "Point coordinates are precomputed in estimate_camera and loaded here for rendering only.",
        ],
    })


def render_camera_estimate_visualization(
    *,
    raw_image_path: Path,
    camera_payload: dict[str, Any],
    png_path: Path,
    summary_path: Path,
) -> None:
    image = Image.open(raw_image_path).convert("RGB") if raw_image_path.exists() else None
    image_w, image_h = camera_payload["image_size_wh"]
    fx = float(camera_payload["intrinsics"]["fx"])
    fy = float(camera_payload["intrinsics"]["fy"])
    cx = float(camera_payload["intrinsics"]["cx"])
    cy = float(camera_payload["intrinsics"]["cy"])
    horizon_y = float(camera_payload["orientation"]["horizon_y"])
    roll_deg = float(camera_payload["orientation"]["roll_deg"])
    pitch_deg = float(camera_payload["orientation"]["pitch_deg"])
    rotation_world_to_camera = build_world_to_camera_rotation(roll_deg=roll_deg, pitch_deg=pitch_deg)

    fig, ax_overlay = plt.subplots(1, 1, figsize=(11, 7))
    if image is not None:
        image_np = np.array(image)
        ax_overlay.imshow(image_np)
    else:
        ax_overlay.set_xlim(0, image_w)
        ax_overlay.set_ylim(image_h, 0)
        ax_overlay.set_facecolor((0.08, 0.08, 0.08))

    draw_reference_volume_overlay(
        ax_overlay,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        horizon_y=horizon_y,
        rotation_world_to_camera=rotation_world_to_camera,
        roll_deg=roll_deg,
    )

    info = [
        f"fx={fx:.1f}",
        f"fy={fy:.1f}",
        f"cx={cx:.1f}",
        f"cy={cy:.1f}",
        f"roll={roll_deg:.2f} deg",
        f"pitch={pitch_deg:.2f} deg",
        f"vfov={camera_payload['intrinsics']['vfov_deg']:.2f} deg",
        f"horizon_y={horizon_y:.1f}",
        f"confidence={camera_payload['confidence']:.2f}",
    ]
    ax_overlay.text(
        24,
        28,
        "\n".join(info),
        color="white",
        fontsize=10,
        va="top",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=6.0),
    )
    ax_overlay.text(
        24,
        image_h - 28,
        "green: 3D reference grid  |  yellow: 1m cube  |  orange/green/blue: camera X/Y/Z axes",
        color="white",
        fontsize=10,
        va="bottom",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=4.0),
    )

    ax_overlay.set_title("Camera Reprojection Overlay")
    ax_overlay.set_xlim(0, image_w)
    ax_overlay.set_ylim(image_h, 0)
    ax_overlay.set_aspect("equal", adjustable="box")
    ax_overlay.axis("off")

    fig.suptitle("Estimated Camera Validation Preview", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    save_json(summary_path, {
        "output_png": str(png_path),
        "raw_image_path": str(raw_image_path) if raw_image_path.exists() else None,
        "notes": [
            "Original image with a camera-centered 3D reference grid, 1m cube, principal point, and horizon.",
        ],
    })
