from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def deg2rad(value: float) -> float:
    return value * math.pi / 180.0


def rotation_matrix_xyz_deg(euler_xyz_deg: Sequence[float]) -> np.ndarray:
    rx, ry, rz = [deg2rad(float(v)) for v in euler_xyz_deg]

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    ry_m = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rz_m = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz_m @ ry_m @ rx_m


def compose_transform(
    scale_xyz: Sequence[float],
    rot_xyz_deg: Sequence[float],
    translate_xyz: Sequence[float],
) -> np.ndarray:
    scale = np.diag([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])])
    rotation = rotation_matrix_xyz_deg(rot_xyz_deg)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation @ scale
    transform[:3, 3] = np.array(translate_xyz, dtype=np.float64)
    return transform


def apply_transform(vertices: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([vertices.astype(np.float64), ones], axis=1)
    transformed = (transform_4x4 @ homo.T).T
    return transformed[:, :3]
