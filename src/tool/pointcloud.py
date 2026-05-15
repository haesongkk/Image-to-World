import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
from src.config import (
    CAMERA_ESTIMATION_OUTPUT_DIR,
    DEPTH_ESTIMATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
    SCENE_PRECOMPUTE_OUTPUT_DIR,
)

def make_pointcloud(image_path: Path):
    object_masks_dir = MASK_POSTPROCESS_OUTPUT_DIR
    depth_map_path = DEPTH_ESTIMATION_OUTPUT_DIR / f"{image_path.stem}.npz"
    camera_intrinsics_path = CAMERA_ESTIMATION_OUTPUT_DIR / f"{image_path.stem}_perspective_fields.json"
    output_dir  = SCENE_PRECOMPUTE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    depth_npz = np.load(depth_map_path)
    depth = depth_npz["depth"]
    h, w = depth.shape

    with open(camera_intrinsics_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)

    rel_focal = float(intrinsics["pred_rel_focal"])
    rel_cx = float(intrinsics["pred_rel_cx"])
    rel_cy = float(intrinsics["pred_rel_cy"])
    roll_deg = float(intrinsics["pred_roll"])
    pitch_deg = float(intrinsics["pred_pitch"])
    vfov_deg = float(intrinsics.get("pred_general_vfov", intrinsics["pred_vfov"]))

    fy = (h * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)
    fx = fy * rel_focal
    cx = w * ( 0.5 + rel_cx )
    cy = h * ( 0.5 + rel_cy )

    pr = math.radians(-pitch_deg)
    rr = math.radians(-roll_deg)    
    cp, sp = math.cos(pr), math.sin(pr)
    cr, sr = math.cos(rr), math.sin(rr)

    R_pitch = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp],
    ], dtype=np.float32)

    R_roll = np.array([
        [cr, -sr, 0],
        [sr,  cr, 0],
        [0,   0,  1],
    ], dtype=np.float32)

    R = R_roll @ R_pitch

    all_points = []

    max_mask_points = 4000

    for obj_idx, npy_path in enumerate(sorted(object_masks_dir.glob("*.npy"))):
        mask = np.load(npy_path)
        y, x = np.nonzero(mask > 0)
        z = depth[y, x]

        if z.size > max_mask_points:
            select = np.linspace(0, z.size - 1, num=max_mask_points, dtype=np.int32)
            y, x, z = y[select], x[select], z[select]

        # Camera space coordinates (right-handed, x-right, y-down, z-forward)
        x_cam = (x.astype(np.float32) - cx) * z / fx
        y_cam = - (y.astype(np.float32) - cy) * z / fy
        z_cam = z.astype(np.float32)
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # World space coordinates (right-handed, x-front, y-right, z-up)
        pts_aligned = (R @ pts_cam.T).T

        # Reorder axes and flip y to convert to x-front, y-right, z-up
        x_front = -pts_aligned[:, 2]
        y_right = pts_aligned[:, 0]
        z_up = pts_aligned[:, 1]
        pts_world = np.stack([x_front, y_right, z_up], axis=1).astype(np.float32)

        color = np.array(plt.get_cmap("tab20")(obj_idx % 20)[:3], dtype=np.float32)  # 0~1 RGB
        colors = np.tile(color, (pts_world.shape[0], 1))  # (N,3)
        pts_rgb = np.concatenate([pts_world, colors], axis=1)
        all_points.append(pts_rgb)

        out_path = output_dir / f"{npy_path.stem}_points.npy"
        np.save(out_path, pts_world)
        print(f"saved {out_path.name}: {pts_world.shape}")


    pts = np.concatenate(all_points, axis=0)

    fig = plt.figure(figsize=(16, 12))
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    view_specs = [
        ("Perspective", 30, 45),
        ("TOP", 90, 0),
        ("FRONT", 0, 0),
        ("SIDE", 0, 90),
    ]

    for ax, (title, elev, azim) in zip(axes, view_specs):
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        C = pts[:, 3:6]
        ax.scatter(X, Y, Z, s=10, c=C)
        ax.set_xlabel("X (front)")
        ax.set_ylabel("Y (right)")
        ax.set_zlabel("Z (up)")
        m = np.array([X.mean(), Y.mean(), Z.mean()])
        r = 0.5 * max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min())
        ax.set_xlim(m[0]-r, m[0]+r); ax.set_ylim(m[1]-r, m[1]+r); ax.set_zlim(m[2]-r, m[2]+r)
        ax.set_box_aspect((1, 1, 1))

    fig.tight_layout()
    viz_path = output_dir / "pointcloud_4views.png"
    fig.savefig(viz_path, dpi=180)
    plt.close(fig)
    print(f"saved {viz_path}")
