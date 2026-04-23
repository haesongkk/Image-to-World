from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
from sklearn.linear_model import RANSACRegressor, LinearRegression

from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import RuntimeConfig, SceneLayoutConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import MeshArtifact, Placement, StageResult
from image_to_world.stages.base import Stage
from image_to_world.visualization.layout_viz import render_layout_visualization


class ComposeLayoutStage(Stage):
    stage_name = "compose_layout"
    MIN_POINT_COUNT = 16
    BOUNDS_LOW_Q = 5.0
    BOUNDS_HIGH_Q = 95.0
    REGISTRATION_SAMPLE_COUNT = 2048
    REGISTRATION_ITERS = 8
    REGISTRATION_SUB_ITERS = 2
    REG_ACCEPT_MIN_EXTENT_RATIO = 0.35
    REG_ACCEPT_MAX_EXTENT_RATIO = 2.5
    REG_ACCEPT_MAX_CENTER_SHIFT_RATIO = 0.35
    REG_ACCEPT_MAX_NRMSE = 0.25
    CROP_SIZE = 512
    CROP_FOV_DEG = 49.1

    def __init__(self, *, config: SceneLayoutConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime

    @staticmethod
    def bbox_center_and_size(bbox_xyxy: list[float]) -> dict[str, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return {"cx": float(x1 + x2) * 0.5, "cy": float(y1 + y2) * 0.5, "w": max(1.0, float(x2 - x1)), "h": max(1.0, float(y2 - y1))}

    @staticmethod
    def build_index_by_id(items: list[dict]) -> dict[int, dict]:
        return {int(item["id"]): item for item in items if item.get("id") is not None}

    @staticmethod
    def angle_from_axis(axis_xz: np.ndarray) -> float:
        return float(math.degrees(math.atan2(float(axis_xz[0]), float(axis_xz[1]))))

    @staticmethod
    def yaw_rotation_matrix(yaw_deg: float) -> np.ndarray:
        yaw_rad = math.radians(float(yaw_deg))
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def matrix_to_euler_xyz_deg(rot: np.ndarray) -> list[float]:
        # XYZ Tait-Bryan angles from rotation matrix.
        sy = math.sqrt(float(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
            y = math.atan2(float(-rot[2, 0]), sy)
            z = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        else:
            x = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
            y = math.atan2(float(-rot[2, 0]), sy)
            z = 0.0
        return [float(math.degrees(x)), float(math.degrees(y)), float(math.degrees(z))]

    @staticmethod
    def percentile_span(values: np.ndarray, low_q: float, high_q: float) -> tuple[float, float]:
        low, high = np.percentile(values, [low_q, high_q])
        return float(low), float(high)

    @staticmethod
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

    @staticmethod
    def normalize_point_order(points_xyz: np.ndarray, point_order: str | None) -> np.ndarray:
        # Backward compatibility: historical artifacts stored points in XZY order.
        order = str(point_order or "xzy").lower()
        if order == "xyz":
            return points_xyz
        if order == "xzy":
            return points_xyz[:, [0, 2, 1]]
        return points_xyz

    @staticmethod
    def load_mask(mask_path: str | None, image_hw: tuple[int, int]) -> np.ndarray | None:
        if not mask_path:
            return None
        p = Path(mask_path)
        if not p.exists():
            return None
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        h, w = image_hw
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask > 127

    @staticmethod
    def depth_to_points_map(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
        h, w = depth_map.shape[:2]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        z = depth_map.astype(np.float64)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        x = ((xx - cx) / max(fx, 1e-8)) * z
        y = -((yy - cy) / max(fy, 1e-8)) * z
        pts = np.stack([x, y, z], axis=-1)
        return pts

    @staticmethod
    def create_triangles(h: int, w: int, mask: np.ndarray | None = None) -> np.ndarray:
        x, y = np.meshgrid(range(w - 1), range(h - 1))
        tl = y * w + x
        tr = y * w + x + 1
        bl = (y + 1) * w + x
        br = (y + 1) * w + x + 1
        triangles = np.array([tl, bl, tr, br, tr, bl])
        triangles = np.transpose(triangles, (1, 2, 0)).reshape(((w - 1) * (h - 1) * 2, 3))
        if mask is not None:
            m = mask.reshape(-1)
            triangles = triangles[m[triangles].all(1)]
        return triangles

    @staticmethod
    def rotation_matrix_to_align_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        a = np.asarray(v1, dtype=np.float64)
        b = np.asarray(v2, dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-12)
        b = b / max(np.linalg.norm(b), 1e-12)
        axis = np.cross(a, b)
        if np.linalg.norm(axis) < 1e-10:
            return np.eye(4, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
        kx, ky, kz = axis.tolist()
        c = math.cos(angle)
        s = math.sin(angle)
        v = 1.0 - c
        rot = np.array(
            [
                [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
                [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
                [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c],
            ],
            dtype=np.float64,
        )
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = rot
        return out

    @staticmethod
    def get_crop_calibration(crop_size: int, fov_deg: float = 49.1) -> np.ndarray:
        focal = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        return np.array(
            [
                [focal * crop_size / 2.0, 0.0, crop_size / 2.0],
                [0.0, focal * crop_size / 2.0, crop_size / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def intersect_rays_mesh(mesh: trimesh.Trimesh, K_def: np.ndarray, c2w_def: np.ndarray, crop_size: int):
        tx = np.linspace(0, crop_size - 1, crop_size)
        ty = np.linspace(0, crop_size - 1, crop_size)
        pixels_x, pixels_y = np.meshgrid(tx, ty)
        p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1)
        p = np.einsum("ij,mnj->mni", np.linalg.inv(K_def), p)
        rays_v = p / np.maximum(np.linalg.norm(p, ord=2, axis=-1, keepdims=True), 1e-12)
        rays_v = np.einsum("ij,mnj->mni", c2w_def[:3, :3], rays_v)
        rays_o = np.broadcast_to(c2w_def[:3, -1], rays_v.shape)
        ro = rays_o.reshape(-1, 3)
        rv = rays_v.reshape(-1, 3)
        try:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            loc, ray_idx, _ = intersector.intersects_location(ro, rv, multiple_hits=False)
        except Exception:
            loc = np.zeros((0, 3), dtype=np.float64)
            ray_idx = np.zeros((0,), dtype=np.int64)
        return loc, ray_idx

    def build_depth_reprojection(self, depth_map: np.ndarray, mask: np.ndarray, K_img: np.ndarray):
        pts3d = self.depth_to_points_map(depth_map, K_img)
        valid_depth = np.isfinite(depth_map) & (depth_map > 1e-6)
        mask_valid = mask & valid_depth
        if int(mask_valid.sum()) < 64:
            return None
        h, w = depth_map.shape[:2]
        triangles = self.create_triangles(h, w, mask=mask_valid)
        if triangles.shape[0] < 64:
            return None
        mesh = trimesh.Trimesh(vertices=pts3d.reshape(-1, 3), faces=triangles, process=False)
        if mesh.vertices.shape[0] < 64:
            return None

        normalization_mat = np.eye(4, dtype=np.float64)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        mid_point = (v.min(axis=0) + v.max(axis=0)) / 2.0
        z_min = float(np.min(pts3d[mask_valid, 2]))
        if abs(float(mid_point[2])) < 1e-8:
            return None
        mid_point = mid_point * (z_min / float(mid_point[2]))
        mesh.apply_translation(-mid_point)
        normalization_mat[:3, -1] -= mid_point

        rotation = self.rotation_matrix_to_align_vectors(mid_point, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        mesh.apply_transform(rotation)
        normalization_mat = rotation @ normalization_mat

        v2 = np.asarray(mesh.vertices, dtype=np.float64)
        if v2.shape[0] < 64:
            return None
        z_q90 = np.quantile(v2[:, -1], 0.9)
        z_q05 = np.quantile(v2[:, -1], 0.05)
        keep = (v2[:, -1] < z_q90) & (v2[:, -1] > z_q05)
        vv = v2[keep] if int(np.sum(keep)) > 32 else v2
        xy_extent = np.max((vv.max(axis=0) - vv.min(axis=0))[:-1])
        if float(xy_extent) < 1e-8:
            return None
        scale = 1.0 / float(xy_extent)
        mesh.apply_scale(scale)
        normalization_mat[:3] *= scale

        v3 = np.asarray(mesh.vertices, dtype=np.float64)
        if v3.shape[0] < 64:
            return None
        translation = -(v3.min(axis=0) + v3.max(axis=0)) / 2.0
        translation[-1] = -(v3[:, -1].min() + 0.5)
        mesh.apply_translation(translation)
        normalization_mat[:3, -1] += translation

        K_crop = self.get_crop_calibration(self.CROP_SIZE, self.CROP_FOV_DEG)
        c2w_crop = np.eye(4, dtype=np.float64)
        c2w_crop[:3, -1] = [0.0, 0.0, -2.0]
        out_depth = self.intersect_rays_mesh(mesh, K_crop, c2w_crop, self.CROP_SIZE)
        if out_depth[0].shape[0] < 64:
            return None
        return {
            "normalization_mat": normalization_mat,
            "out_depth": out_depth,
            "c2w_crop": c2w_crop,
            "K_crop": K_crop,
        }

    def align_to_depth_rep(
        self,
        obj_mesh: trimesh.Trimesh,
        normalization_mat: np.ndarray,
        out_depth,
        c2w_crop: np.ndarray,
        K_crop: np.ndarray,
        crop_size: int,
    ) -> np.ndarray | None:
        crop_shape = (crop_size, crop_size)
        w2c_crop = np.linalg.inv(c2w_crop)
        out_rec = self.intersect_rays_mesh(obj_mesh, K_crop, c2w_crop, crop_size)
        if out_rec[0].shape[0] < 32:
            return None

        depth_mask = np.zeros(crop_shape, dtype=bool)
        depth_mask[np.unravel_index(out_depth[1], crop_shape)] = True
        rec_mask = np.zeros(crop_shape, dtype=bool)
        rec_mask[np.unravel_index(out_rec[1], crop_shape)] = True
        mask_both = depth_mask & rec_mask

        rec_keep = mask_both[np.unravel_index(out_rec[1], crop_shape)]
        dep_keep = mask_both[np.unravel_index(out_depth[1], crop_shape)]
        if int(np.sum(rec_keep)) < 16 or int(np.sum(dep_keep)) < 16:
            return None
        pc_rec = out_rec[0][rec_keep]
        pc_depth = out_depth[0][dep_keep]
        n = min(pc_rec.shape[0], pc_depth.shape[0])
        if n < 16:
            return None
        pc_rec = pc_rec[:n]
        pc_depth = pc_depth[:n]

        # Match Gen3DSR logic: scale correction on camera-space z only.
        h_rec = np.concatenate([pc_rec, np.ones((pc_rec.shape[0], 1), dtype=np.float64)], axis=1)
        h_dep = np.concatenate([pc_depth, np.ones((pc_depth.shape[0], 1), dtype=np.float64)], axis=1)
        rec_cam = (w2c_crop @ h_rec.T).T[:, :3]
        dep_cam = (w2c_crop @ h_dep.T).T[:, :3]

        regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=0.2)
        regressor.fit(rec_cam[:, -1:].reshape(-1, 1), dep_cam[:, -1:].reshape(-1, 1))
        z_scale = float(regressor.estimator_.coef_[0, 0])
        if not np.isfinite(z_scale) or z_scale <= 1e-6:
            return None
        transform = w2c_crop.copy()
        transform[:3] *= z_scale
        transform = np.linalg.inv(normalization_mat) @ c2w_crop @ transform
        return transform

    @staticmethod
    def load_mesh_points(mesh_path: str, sample_count: int) -> tuple[np.ndarray | None, str | None]:
        try:
            loaded = trimesh.load(str(mesh_path), force="scene")
            meshes: list[trimesh.Trimesh] = []
            if isinstance(loaded, trimesh.Scene):
                for node_name in loaded.graph.nodes_geometry:
                    geom_name = loaded.graph[node_name][1]
                    geom = loaded.geometry.get(geom_name)
                    if isinstance(geom, trimesh.Trimesh) and geom.vertices.shape[0] > 0 and geom.faces.shape[0] > 0:
                        meshes.append(geom.copy())
            elif isinstance(loaded, trimesh.Trimesh):
                meshes = [loaded.copy()]
            if not meshes:
                return None, "mesh_has_no_valid_trimesh_geometry"
            face_counts = np.array([max(1, int(m.faces.shape[0])) for m in meshes], dtype=np.int64)
            face_sum = int(face_counts.sum())
            sampled_parts = []
            for m, fcount in zip(meshes, face_counts):
                c = max(64, int(round(sample_count * (float(fcount) / float(face_sum)))))
                p, _ = trimesh.sample.sample_surface(m, c)
                sampled_parts.append(np.asarray(p, dtype=np.float64))
            sampled = np.concatenate(sampled_parts, axis=0)
            if sampled.shape[0] < 64:
                return None, "mesh_sampling_too_few_points"
            return sampled, None
        except Exception as exc:
            return None, f"mesh_load_failed: {exc}"

    @staticmethod
    def normalize_mesh_points(points_xyz: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        pmin = points_xyz.min(axis=0)
        pmax = points_xyz.max(axis=0)
        center = 0.5 * (pmin + pmax)
        ext = pmax - pmin
        scale = float(max(np.max(ext), 1e-6))
        normalized = (points_xyz - center[None, :]) / scale
        return normalized, {
            "mesh_center_xyz": center.tolist(),
            "mesh_norm_scale": scale,
            "mesh_bounds_min_xyz": pmin.tolist(),
            "mesh_bounds_max_xyz": pmax.tolist(),
        }

    @staticmethod
    def estimate_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        # Solve dst ~= s * R * src + t
        if src.shape[0] < 4 or dst.shape[0] < 4:
            return np.eye(3, dtype=np.float64), 1.0, np.zeros(3, dtype=np.float64)
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean[None, :]
        dst_c = dst - dst_mean[None, :]
        cov = (dst_c.T @ src_c) / float(src.shape[0])
        u, singular_vals, vt = np.linalg.svd(cov)
        sign_fix = np.eye(3, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0.0:
            sign_fix[2, 2] = -1.0
        rot = u @ sign_fix @ vt
        src_var = float(np.sum(src_c * src_c) / float(src.shape[0]))
        if src_var < 1e-12:
            scale = 1.0
        else:
            scale = float(np.sum(singular_vals * np.diag(sign_fix)) / src_var)
        trans = dst_mean - scale * (rot @ src_mean)
        return rot, scale, trans

    @staticmethod
    def nearest_neighbors(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # For each src point, find nearest dst point (brute force, small N regime).
        # Returns (matched_dst, dists, dst_idx)
        diff = src[:, None, :] - dst[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        nn_idx = np.argmin(d2, axis=1)
        d = np.sqrt(np.maximum(d2[np.arange(src.shape[0]), nn_idx], 0.0))
        return dst[nn_idx], d, nn_idx.astype(np.int32)

    @staticmethod
    def mutual_correspondences(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns indices (src_idx, dst_idx, dists) for mutual nearest-neighbor pairs.
        diff = src[:, None, :] - dst[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        src_to_dst = np.argmin(d2, axis=1)
        dst_to_src = np.argmin(d2, axis=0)
        src_idx = np.arange(src.shape[0], dtype=np.int32)
        mutual_mask = dst_to_src[src_to_dst] == src_idx
        src_idx = src_idx[mutual_mask]
        dst_idx = src_to_dst[mutual_mask]
        if src_idx.size == 0:
            return src_idx, dst_idx, np.zeros((0,), dtype=np.float64)
        d = np.sqrt(np.maximum(d2[src_idx, dst_idx], 0.0))
        return src_idx, dst_idx, d

    def run_icp_similarity(
        self, source_local: np.ndarray, target_world: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        # Returns 4x4 transform (world <- local), plus diagnostics.
        if source_local.shape[0] < 64 or target_world.shape[0] < 64:
            return None, {"reason": "insufficient_points_for_registration"}

        src = source_local
        dst = target_world
        # Init transform from centroid/scale.
        src_std = float(np.sqrt(np.mean(np.sum((src - src.mean(axis=0)[None, :]) ** 2, axis=1))))
        dst_std = float(np.sqrt(np.mean(np.sum((dst - dst.mean(axis=0)[None, :]) ** 2, axis=1))))
        s = float(max(dst_std / max(src_std, 1e-6), 1e-6))
        r = np.eye(3, dtype=np.float64)
        t = dst.mean(axis=0) - s * (r @ src.mean(axis=0))

        transformed = s * (src @ r.T) + t[None, :]
        for _ in range(self.REGISTRATION_ITERS):
            src_idx, dst_idx, d = self.mutual_correspondences(transformed, dst)
            if src_idx.size < 16:
                matched, d_one, dst_idx_one = self.nearest_neighbors(transformed, dst)
                src_idx = np.arange(matched.shape[0], dtype=np.int32)
                dst_idx = dst_idx_one
                d = d_one
            if src_idx.size < 16:
                break
            keep = d <= np.percentile(d, 85.0)
            if int(np.sum(keep)) < 16:
                keep = np.ones_like(d, dtype=bool)
            r_next, s_next, t_next = self.estimate_similarity_umeyama(src[src_idx[keep]], dst[dst_idx[keep]])
            for _sub in range(self.REGISTRATION_SUB_ITERS):
                transformed_sub = s_next * (src @ r_next.T) + t_next[None, :]
                sub_src_idx, sub_dst_idx, d_sub = self.mutual_correspondences(transformed_sub, dst)
                if sub_src_idx.size < 16:
                    break
                keep_sub = d_sub <= np.percentile(d_sub, 85.0)
                if int(np.sum(keep_sub)) < 16:
                    break
                r_next, s_next, t_next = self.estimate_similarity_umeyama(
                    src[sub_src_idx[keep_sub]], dst[sub_dst_idx[keep_sub]]
                )
            r, s, t = r_next, float(max(s_next, 1e-6)), t_next
            transformed = s * (src @ r.T) + t[None, :]

        final_matched, final_d, _ = self.nearest_neighbors(transformed, dst)
        rmse = float(np.sqrt(np.mean(final_d * final_d)))
        inlier_thr = float(np.percentile(final_d, 70.0))
        inlier_ratio = float(np.mean(final_d <= inlier_thr))
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = s * r
        tf[:3, 3] = t
        return tf, {
            "registration_status": "ok",
            "registration_rmse": rmse,
            "registration_inlier_ratio": inlier_ratio,
            "registration_inlier_threshold": inlier_thr,
        }

    def fit_gen3dsr_style_box(self, points_xyz: np.ndarray) -> dict[str, object] | None:
        if points_xyz.shape[0] < self.MIN_POINT_COUNT:
            return None
        valid = np.isfinite(points_xyz).all(axis=1)
        points_xyz = points_xyz[valid]
        if points_xyz.shape[0] < self.MIN_POINT_COUNT:
            return None

        points_fit = points_xyz
        center = np.median(points_fit, axis=0)
        centered = points_fit - center[None, :]

        # Gen3DSR-style quick placement proxy:
        # estimate object yaw from XZ principal direction, then fit local extents.
        xz = centered[:, [0, 2]]
        cov = np.cov(xz.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        if float(principal[1]) < 0.0:
            principal = -principal
        yaw_deg = self.angle_from_axis(principal)

        rot_world_from_local = self.yaw_rotation_matrix(yaw_deg)
        local_points = (rot_world_from_local.T @ centered.T).T

        min_x, max_x = self.percentile_span(local_points[:, 0], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_y, max_y = self.percentile_span(local_points[:, 1], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_z, max_z = self.percentile_span(local_points[:, 2], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)

        width_world = max(1e-3, max_x - min_x)
        height_world = max(1e-3, max_y - min_y)
        depth_world = max(1e-3, max_z - min_z)
        center_x = float(center[0])
        center_y = float(center[1])
        center_z = float(center[2])

        return {
            "position_xyz": [float(center_x), float(center_y), float(center_z)],
            "rotation_euler_xyz_deg": [0.0, float(yaw_deg), 0.0],
            "scale_xyz": [width_world, height_world, depth_world],
            "estimated_world_size_wh": [width_world, height_world],
            "estimated_world_size_xyz": [width_world, height_world, depth_world],
            "support_type": "unknown",
            "analysis": {
                "point_count": int(points_xyz.shape[0]),
                "obb_source": "gen3dsr_style_depth_alignment_proxy",
                "yaw_deg": float(yaw_deg),
                "local_bounds_x": [min_x, max_x],
                "local_bounds_y": [min_y, max_y],
                "local_bounds_z": [min_z, max_z],
                "center_xyz_median": [center_x, center_y, center_z],
            },
        }

    def register_mesh_to_pointcloud(
        self,
        *,
        mesh_path: str | None,
        points_xyz: np.ndarray,
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        if not mesh_path:
            return None, {"reason": "mesh_path_missing"}
        path = Path(mesh_path)
        if not path.exists():
            return None, {"reason": f"mesh_not_found: {mesh_path}"}
        mesh_points, mesh_load_error = self.load_mesh_points(str(path), self.REGISTRATION_SAMPLE_COUNT)
        if mesh_points is None:
            return None, {"reason": mesh_load_error or "mesh_load_failed"}

        local_points, local_meta = self.normalize_mesh_points(mesh_points)
        tf, reg = self.run_icp_similarity(local_points, points_xyz)
        if tf is None:
            return None, {**local_meta, **reg}

        transformed = (tf[:3, :3] @ local_points.T).T + tf[:3, 3][None, :]
        min_x, max_x = self.percentile_span(transformed[:, 0], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_y, max_y = self.percentile_span(transformed[:, 1], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_z, max_z = self.percentile_span(transformed[:, 2], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        center = np.median(transformed, axis=0)
        lin = tf[:3, :3]
        scale_uniform = float(np.linalg.norm(lin[:, 0]))
        rot = lin / max(scale_uniform, 1e-8)
        euler = self.matrix_to_euler_xyz_deg(rot)
        reg_rmse = float(reg.get("registration_rmse", 1e9))
        pc_center = np.median(points_xyz, axis=0)
        pc_pmin = points_xyz.min(axis=0)
        pc_pmax = points_xyz.max(axis=0)
        pc_extent = np.maximum(pc_pmax - pc_pmin, 1e-8)
        pc_diag = float(np.linalg.norm(pc_extent))
        fit_extent = np.array(
            [
                max(1e-3, float(max_x - min_x)),
                max(1e-3, float(max_y - min_y)),
                max(1e-3, float(max_z - min_z)),
            ],
            dtype=np.float64,
        )
        extent_ratio = fit_extent / pc_extent
        center_shift = float(np.linalg.norm(center - pc_center))
        center_shift_ratio = center_shift / max(pc_diag, 1e-8)
        nrmse = reg_rmse / max(pc_diag, 1e-8)
        extent_ok = bool(
            np.all(extent_ratio >= self.REG_ACCEPT_MIN_EXTENT_RATIO)
            and np.all(extent_ratio <= self.REG_ACCEPT_MAX_EXTENT_RATIO)
        )
        shift_ok = bool(center_shift_ratio <= self.REG_ACCEPT_MAX_CENTER_SHIFT_RATIO)
        rmse_ok = bool(nrmse <= self.REG_ACCEPT_MAX_NRMSE)
        quality_ok = bool(extent_ok and shift_ok and rmse_ok)
        fit = {
            "position_xyz": [float(center[0]), float(center[1]), float(center[2])],
            "rotation_euler_xyz_deg": [float(euler[0]), float(euler[1]), float(euler[2])],
            "scale_xyz": [
                max(1e-3, float(max_x - min_x)),
                max(1e-3, float(max_y - min_y)),
                max(1e-3, float(max_z - min_z)),
            ],
            "estimated_world_size_wh": [
                max(1e-3, float(max_x - min_x)),
                max(1e-3, float(max_y - min_y)),
            ],
            "estimated_world_size_xyz": [
                max(1e-3, float(max_x - min_x)),
                max(1e-3, float(max_y - min_y)),
                max(1e-3, float(max_z - min_z)),
            ],
            "support_type": "unknown",
            "analysis": {
                "point_count": int(points_xyz.shape[0]),
                "obb_source": "gen3dsr_style_mesh_to_depth_registration",
                **local_meta,
                **reg,
                "similarity_scale": scale_uniform,
                "quality_gate": {
                    "quality_ok": quality_ok,
                    "extent_ratio_xyz": extent_ratio.tolist(),
                    "center_shift_ratio": center_shift_ratio,
                    "nrmse": nrmse,
                    "thresholds": {
                        "min_extent_ratio": self.REG_ACCEPT_MIN_EXTENT_RATIO,
                        "max_extent_ratio": self.REG_ACCEPT_MAX_EXTENT_RATIO,
                        "max_center_shift_ratio": self.REG_ACCEPT_MAX_CENTER_SHIFT_RATIO,
                        "max_nrmse": self.REG_ACCEPT_MAX_NRMSE,
                    },
                },
            },
        }
        if not quality_ok:
            return None, {"reason": "registration_quality_gate_failed", "analysis": fit["analysis"]}
        return fit, {"status": "ok", "analysis": fit["analysis"]}

    def register_mesh_gen3dsr_style(
        self,
        *,
        mesh_path: str | None,
        mask_path: str | None,
        depth_map: np.ndarray | None,
        K_img: np.ndarray | None,
        points_xyz: np.ndarray,
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        if not mesh_path:
            return None, {"reason": "mesh_path_missing"}
        if depth_map is None or K_img is None:
            return None, {"reason": "depth_or_intrinsics_missing"}
        h, w = depth_map.shape[:2]
        mask = self.load_mask(mask_path, (h, w))
        if mask is None:
            return None, {"reason": "mask_missing_or_invalid"}
        reproj = self.build_depth_reprojection(depth_map, mask, K_img)
        if reproj is None:
            return None, {"reason": "depth_reprojection_failed"}

        mesh_points, mesh_load_error = self.load_mesh_points(str(mesh_path), self.REGISTRATION_SAMPLE_COUNT)
        if mesh_points is None:
            return None, {"reason": mesh_load_error or "mesh_load_failed"}
        # Gen3DSR assumes object mesh in normalized object space.
        # We normalize loaded mesh to unit box to match that assumption.
        local_points, local_meta = self.normalize_mesh_points(mesh_points)
        obj_mesh: trimesh.Trimesh
        # Keep original topology for ray intersection by reloading as mesh.
        loaded = trimesh.load(str(mesh_path), force="scene")
        if isinstance(loaded, trimesh.Scene):
            geom_list = []
            for n in loaded.graph.nodes_geometry:
                t_local, g_name = loaded.graph.get(n)
                g = loaded.geometry[g_name].copy()
                g.apply_transform(t_local)
                geom_list.append(g)
            if not geom_list:
                return None, {"reason": "mesh_geometry_missing"}
            obj_mesh = trimesh.util.concatenate(geom_list)
        elif isinstance(loaded, trimesh.Trimesh):
            obj_mesh = loaded.copy()
        else:
            return None, {"reason": "mesh_geometry_missing"}
        # Normalize mesh similarly to sampled points so transform maps canonical local->world.
        v = np.asarray(obj_mesh.vertices, dtype=np.float64)
        v = (v - np.array(local_meta["mesh_center_xyz"], dtype=np.float64)[None, :]) / float(local_meta["mesh_norm_scale"])
        obj_mesh.vertices = v

        tf = self.align_to_depth_rep(
            obj_mesh=obj_mesh,
            normalization_mat=reproj["normalization_mat"],
            out_depth=reproj["out_depth"],
            c2w_crop=reproj["c2w_crop"],
            K_crop=reproj["K_crop"],
            crop_size=self.CROP_SIZE,
        )
        if tf is None:
            return None, {"reason": "align_to_depth_rep_failed"}

        transformed = (tf[:3, :3] @ v.T).T + tf[:3, 3][None, :]
        min_x, max_x = self.percentile_span(transformed[:, 0], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_y, max_y = self.percentile_span(transformed[:, 1], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_z, max_z = self.percentile_span(transformed[:, 2], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        center = np.median(transformed, axis=0)

        # Evaluate alignment quality against depth pointcloud target.
        _, d_eval, _ = self.nearest_neighbors(transformed, points_xyz)
        rmse = float(np.sqrt(np.mean(d_eval * d_eval)))
        pc_pmin = points_xyz.min(axis=0)
        pc_pmax = points_xyz.max(axis=0)
        pc_extent = np.maximum(pc_pmax - pc_pmin, 1e-8)
        pc_diag = float(np.linalg.norm(pc_extent))
        fit_extent = np.array([max_x - min_x, max_y - min_y, max_z - min_z], dtype=np.float64)
        extent_ratio = fit_extent / pc_extent
        nrmse = rmse / max(pc_diag, 1e-8)
        quality_ok = bool(
            np.all(extent_ratio >= self.REG_ACCEPT_MIN_EXTENT_RATIO)
            and np.all(extent_ratio <= self.REG_ACCEPT_MAX_EXTENT_RATIO)
            and (nrmse <= self.REG_ACCEPT_MAX_NRMSE)
        )

        lin = tf[:3, :3]
        scale_uniform = float(np.linalg.norm(lin[:, 0]))
        rot = lin / max(scale_uniform, 1e-8)
        euler = self.matrix_to_euler_xyz_deg(rot)
        fit = {
            "position_xyz": [float(center[0]), float(center[1]), float(center[2])],
            "rotation_euler_xyz_deg": [float(euler[0]), float(euler[1]), float(euler[2])],
            "scale_xyz": [max(1e-3, float(max_x - min_x)), max(1e-3, float(max_y - min_y)), max(1e-3, float(max_z - min_z))],
            "estimated_world_size_wh": [max(1e-3, float(max_x - min_x)), max(1e-3, float(max_y - min_y))],
            "estimated_world_size_xyz": [max(1e-3, float(max_x - min_x)), max(1e-3, float(max_y - min_y)), max(1e-3, float(max_z - min_z))],
            "support_type": "unknown",
            "analysis": {
                "point_count": int(points_xyz.shape[0]),
                "obb_source": "gen3dsr_align_to_depth_rep",
                "registration_status": "ok",
                "registration_rmse": rmse,
                "quality_gate": {
                    "quality_ok": quality_ok,
                    "extent_ratio_xyz": extent_ratio.tolist(),
                    "nrmse": nrmse,
                },
                **local_meta,
                "similarity_scale": scale_uniform,
            },
        }
        if not quality_ok:
            return None, {"reason": "registration_quality_gate_failed", "analysis": fit["analysis"]}
        return fit, {"status": "ok", "analysis": fit["analysis"]}

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "scene_layout.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        camera_data = load_json(self.config.camera_json_path) if self.config.camera_json_path.exists() else {}
        mask_data = load_json(self.config.mask_json_path) if self.config.mask_json_path.exists() else {}
        depth_data = load_json(self.config.depth_json_path) if self.config.depth_json_path.exists() else {}
        mesh_data = load_json(self.config.gen3d_json_path) if self.config.gen3d_json_path.exists() else {}
        mesh_by_id = self.build_index_by_id(mesh_data.get("results", []))
        mask_by_id = self.build_index_by_id(mask_data.get("annotations", []))
        object_pointclouds = camera_data.get("object_pointclouds", [])
        image_size_wh = camera_data.get("image_size_wh")
        if not object_pointclouds:
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": 0}, skipped=True), self.config_to_cache_payload(self.config))
        image_w, image_h = image_size_wh if isinstance(image_size_wh, list) and len(image_size_wh) == 2 else [0, 0]
        depth_map = None
        depth_npy = depth_data.get("depth_raw_npy_path")
        if depth_npy and Path(depth_npy).exists():
            depth_map = np.load(depth_npy).astype(np.float64)
        intr = camera_data.get("intrinsics", {})
        if all(k in intr for k in ("fx", "fy", "cx", "cy")):
            K_img = np.array(
                [
                    [float(intr["fx"]), 0.0, float(intr["cx"])],
                    [0.0, float(intr["fy"]), float(intr["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        else:
            K_img = None

        placements = []
        skipped = []
        for object_ann in object_pointclouds:
            obj_id = int(object_ann["id"])
            label = object_ann.get("class_name", "unknown")
            bbox_xyxy = object_ann.get("bbox_xyxy")
            pointcloud_path = object_ann.get("pointcloud_path")
            mesh_ann = mesh_by_id.get(obj_id)
            mask_ann = mask_by_id.get(obj_id, {})
            points_xyz = self.load_pointcloud(pointcloud_path)
            if points_xyz is None:
                skipped.append({"id": obj_id, "class_name": label, "reason": f"pointcloud missing or invalid: {pointcloud_path}"})
                continue
            points_xyz = self.normalize_point_order(points_xyz, object_ann.get("point_order"))
            fallback_world = self.fit_gen3dsr_style_box(points_xyz)
            if fallback_world is None:
                skipped.append({"id": obj_id, "class_name": label, "reason": "insufficient valid pointcloud samples"})
                continue
            placement_world, reg_meta = self.register_mesh_gen3dsr_style(
                mesh_path=mesh_ann.get("mesh_path") if mesh_ann else None,
                mask_path=mask_ann.get("mask_path") or object_ann.get("source_paths", {}).get("mask_path"),
                depth_map=depth_map,
                K_img=K_img,
                points_xyz=points_xyz,
            )
            if placement_world is None:
                placement_world, reg_meta = self.register_mesh_to_pointcloud(
                    mesh_path=mesh_ann.get("mesh_path") if mesh_ann else None,
                    points_xyz=points_xyz,
                )
            if placement_world is None:
                placement_world = fallback_world
                placement_world["analysis"] = {
                    **placement_world.get("analysis", {}),
                    "registration_status": "fallback",
                    "registration_failure_reason": reg_meta.get("reason"),
                    "registration_failure_analysis": reg_meta.get("analysis"),
                }
            bbox_info = self.bbox_center_and_size(bbox_xyxy) if bbox_xyxy is not None else {"cx": 0.0, "cy": 0.0, "w": 1.0, "h": 1.0}
            placements.append(Placement(
                id=obj_id,
                class_name=label,
                score=object_ann.get("score"),
                bbox_xyxy=bbox_xyxy,
                bbox_center_xy=[bbox_info["cx"], bbox_info["cy"]],
                bbox_size_wh=[bbox_info["w"], bbox_info["h"]],
                depth_source="camera_object_pointcloud",
                depth_value=float(placement_world["position_xyz"][2]),
                pseudo_world=placement_world,
                mesh=MeshArtifact(
                    mesh_path=mesh_ann.get("mesh_path") if mesh_ann else None,
                    mesh_format=mesh_ann.get("mesh_format") if mesh_ann else None,
                ),
                source_paths={
                    **object_ann.get("source_paths", {}),
                    "pointcloud_path": pointcloud_path,
                },
            ).to_dict())
            placements[-1]["primitive"] = {
                "type": "cuboid",
                "size_xyz": list(placement_world["scale_xyz"]),
            }

        placements = sorted(placements, key=lambda item: item["pseudo_world"]["position_xyz"][2])
        save_json(output_path, {
            "camera_json_path": str(self.config.camera_json_path) if self.config.camera_json_path.exists() else None,
            "mesh_json_path": str(self.config.gen3d_json_path) if self.config.gen3d_json_path.exists() else None,
            "image_size_wh": [int(image_w), int(image_h)],
            "coordinate_frame": camera_data.get("coordinate_frame"),
            "layout_method": {
                "type": "gen3dsr_style_mesh_to_depth_registration",
                "notes": [
                    "compose_layout consumes calibrated object point clouds from estimate_camera.",
                    "Primary path ports Gen3DSR align_to_depth_rep-style depth reprojection alignment.",
                    "Secondary fallback uses similarity registration between mesh surface points and object depth pointcloud.",
                    "If both fail, stage falls back to depth-only yaw proxy placement.",
                ],
            },
            "num_requested_objects": len(object_pointclouds),
            "num_placed_objects": len(placements),
            "num_skipped_objects": len(skipped),
            "placements": placements,
            "skipped": skipped,
        })

        if self.config.save_visualization:
            render_layout_visualization(
                raw_image_path=self.config.raw_image_path,
                placements=placements,
                png_path=self.config.output_dir / "layout_preview.png",
                summary_path=self.config.output_dir / "layout_preview_summary.json",
            )

        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": len(placements), "skipped": len(skipped)})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the compose_layout stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = ComposeLayoutStage(config=SceneLayoutConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
