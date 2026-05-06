from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import math

def make_raw_transform(image_path: Path):
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir  = project_root / "output" / "raw_transform"
    output_dir.mkdir(parents=True, exist_ok=True)

    pointcloud_dir = project_root / "output" / "pointcloud"
    transform = []
    for pointcloud_path in sorted(pointcloud_dir.glob("*.npy")):
        pointcloud = np.load(pointcloud_path)
        x = np.mean(pointcloud[:, 0])
        y = np.mean(pointcloud[:, 1])
        z = np.mean(pointcloud[:, 2])
        depth = np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0])
        width = np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1])
        height = np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])
        rot_deg = 0.0
        transform.append([x, y, z, depth, width, height, rot_deg, rot_deg, rot_deg])

    transform = np.stack(transform, axis=0)
    payload = {
        "transforms": [
            {
                "translation": {"x": float(tr[0]), "y": float(tr[1]), "z": float(tr[2])},
                "scale": {"x": float(tr[3]), "y": float(tr[4]), "z": float(tr[5])},
                "rotation_deg": {"x": float(tr[6]), "y": float(tr[7]), "z": float(tr[8])},
            }
            for tr in transform
        ]
    }
    with open(output_dir / "raw_transform.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

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
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    colors = plt.get_cmap("tab20")

    all_corners = []
    for i, tr in enumerate(transform):
        center = tr[0:3].astype(np.float32)
        size = np.maximum(tr[3:6].astype(np.float32), 1e-6)
        rot = tr[6:9].astype(np.float32)

        rx = math.radians(float(rot[0]))
        ry = math.radians(float(rot[1]))
        rz = math.radians(float(rot[2]))
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        r = rz_m @ ry_m @ rx_m

        dx, dy, dz = size * 0.5
        local = np.array(
            [
                [-dx, -dy, -dz],
                [ dx, -dy, -dz],
                [ dx,  dy, -dz],
                [-dx,  dy, -dz],
                [-dx, -dy,  dz],
                [ dx, -dy,  dz],
                [ dx,  dy,  dz],
                [-dx,  dy,  dz],
            ],
            dtype=np.float32,
        )
        corners = (local @ r.T) + center[None, :]
        all_corners.append(corners)

        # local axes (x:red, y:green, z:blue)
        axis_len = 0.5 * float(np.max(size))
        local_axes = r @ np.eye(3, dtype=np.float32)
        all_corners.append(center[None, :])
        all_corners.append((center[None, :] + local_axes.T * axis_len))

    for ax, (title, elev, azim) in zip(axes, view_specs):
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        for i, tr in enumerate(transform):
            center = tr[0:3]
            size = np.maximum(tr[3:6], 1e-6)
            rot = tr[6:9]
            c = colors(i % 20)

            rx = math.radians(float(rot[0]))
            ry = math.radians(float(rot[1]))
            rz = math.radians(float(rot[2]))
            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)
            rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            r = rz_m @ ry_m @ rx_m

            dx, dy, dz = size * 0.5
            local = np.array(
                [
                    [-dx, -dy, -dz],
                    [ dx, -dy, -dz],
                    [ dx,  dy, -dz],
                    [-dx,  dy, -dz],
                    [-dx, -dy,  dz],
                    [ dx, -dy,  dz],
                    [ dx,  dy,  dz],
                    [-dx,  dy,  dz],
                ],
                dtype=np.float32,
            )
            corners = (local @ r.T) + center[None, :]

            for s, e in box_edges:
                p0, p1 = corners[s], corners[e]
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=c,
                    linewidth=2,
                )

            # center point + local axis
            ax.scatter([center[0]], [center[1]], [center[2]], color=[c], s=20)
            axis_len = 0.5 * float(np.max(size))
            local_axes = r @ np.eye(3, dtype=np.float32)
            ax.quiver(center[0], center[1], center[2], *(local_axes[:, 0] * axis_len), color="r", linewidth=1.5)
            ax.quiver(center[0], center[1], center[2], *(local_axes[:, 1] * axis_len), color="g", linewidth=1.5)
            ax.quiver(center[0], center[1], center[2], *(local_axes[:, 2] * axis_len), color="b", linewidth=1.5)

        ax.set_xlabel("X (front)")
        ax.set_ylabel("Y (right)")
        ax.set_zlabel("Z (up)")

    all_pts = np.concatenate([c.reshape(-1, 3) for c in all_corners], axis=0) if all_corners else np.zeros((1, 3), dtype=np.float32)
    x_all, y_all, z_all = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    m = np.array([x_all.mean(), y_all.mean(), z_all.mean()], dtype=np.float32)
    r = 0.5 * max(
        float(x_all.max() - x_all.min()),
        float(y_all.max() - y_all.min()),
        float(z_all.max() - z_all.min()),
        1e-3,
    )
    for ax in axes:
        ax.set_xlim(m[0] - r, m[0] + r)
        ax.set_ylim(m[1] - r, m[1] + r)
        ax.set_zlim(m[2] - r, m[2] + r)
        ax.set_box_aspect((1, 1, 1))

    fig.tight_layout()
    viz_path = output_dir / "raw_transform_4views.png"
    fig.savefig(viz_path, dpi=180)
    plt.close(fig)
    print(f"saved {viz_path}")
