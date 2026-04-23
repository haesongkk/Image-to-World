from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_to_world.adapters.perspective_fields import PerspectiveFieldsAdapter
from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import CameraEstimationConfig, RuntimeConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage
from image_to_world.visualization.camera_viz import (
    render_camera_calibrated_pointcloud,
    render_camera_estimate_visualization,
)


class EstimateCameraStage(Stage):
    stage_name = "estimate_camera"
    MAX_MASK_POINTS = 4000

    def __init__(self, *, config: CameraEstimationConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime

    @staticmethod
    def load_mask(mask_path: str | None, target_hw: tuple[int, int]) -> np.ndarray | None:
        if not mask_path:
            return None
        path = Path(mask_path)
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        target_h, target_w = target_hw
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8)

    @staticmethod
    def sample_indices(count: int, max_count: int) -> np.ndarray:
        if count <= max_count:
            return np.arange(count, dtype=np.int32)
        return np.linspace(0, count - 1, num=max_count, dtype=np.int32)

    @staticmethod
    def rotate_points(
        points: np.ndarray,
        *,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> np.ndarray:
        pitch_rad = math.radians(float(pitch_deg))
        roll_rad = math.radians(float(roll_deg))
        yaw_rad = math.radians(float(yaw_deg))

        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ], dtype=np.float64)
        rot_z = np.array([
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        rot_y = np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ], dtype=np.float64)
        rot = rot_y @ rot_z @ rot_x
        return (rot @ points.T).T

    @staticmethod
    def depth_to_camera_points(
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        depth_values: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
    ) -> np.ndarray:
        z_camera = np.maximum(depth_values.astype(np.float64) * float(depth_scale), 1e-3)
        x_camera = ((xs.astype(np.float64) - float(cx)) / max(float(fx), 1e-6)) * z_camera
        y_camera = -((ys.astype(np.float64) - float(cy)) / max(float(fy), 1e-6)) * z_camera
        return np.stack([x_camera, y_camera, z_camera], axis=1)

    def build_object_pointclouds(
        self,
        *,
        mask_annotations: list[dict],
        depth: np.ndarray,
        image_hw: tuple[int, int],
        fx: float,
        fy: float,
        cx0: float,
        cy0: float,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float,
        depth_scale: float,
    ) -> list[dict]:
        image_h, image_w = image_hw
        output_dir = self.config.output_dir / "object_pointclouds"
        output_dir.mkdir(parents=True, exist_ok=True)
        pointclouds = []

        for ann in mask_annotations:
            obj_id = ann.get("id")
            mask = self.load_mask(ann.get("mask_path"), (image_h, image_w))
            if obj_id is None or mask is None:
                continue
            ys, xs = np.nonzero(mask == 1)
            if xs.size < 16:
                continue
            select = self.sample_indices(xs.size, self.MAX_MASK_POINTS)
            xs = xs[select]
            ys = ys[select]
            depth_values = depth[ys, xs].astype(np.float64)
            valid = np.isfinite(depth_values) & (depth_values > 1e-6)
            if not valid.any():
                continue
            xs = xs[valid]
            ys = ys[valid]
            depth_values = depth_values[valid]
            points_camera = self.depth_to_camera_points(
                xs=xs,
                ys=ys,
                depth_values=depth_values,
                fx=fx,
                fy=fy,
                cx=cx0,
                cy=cy0,
                depth_scale=depth_scale,
            )
            # Test mode: apply inverse rotation using available Euler angles.
            points_pitch_inv = self.rotate_points(
                points_camera,
                pitch_deg=-float(pitch_deg),
                roll_deg=-float(roll_deg),
                yaw_deg=-float(yaw_deg),
            )
            # Canonical world coordinate order is XYZ.
            points_xyz = np.stack([points_pitch_inv[:, 0], points_pitch_inv[:, 1], points_pitch_inv[:, 2]], axis=1)
            pointcloud_path = output_dir / f"object_{int(obj_id):03d}.npy"
            np.save(pointcloud_path, points_xyz.astype(np.float32))
            pointclouds.append({
                "id": int(obj_id),
                "class_name": ann.get("class_name", "unknown"),
                "score": ann.get("score"),
                "bbox_xyxy": ann.get("bbox_xyxy"),
                "pointcloud_path": str(pointcloud_path),
                "point_order": "xyz",
                "point_count": int(points_xyz.shape[0]),
                "bounds_min_xyz": points_xyz.min(axis=0).tolist(),
                "bounds_max_xyz": points_xyz.max(axis=0).tolist(),
                "centroid_xyz": points_xyz.mean(axis=0).tolist(),
                "source_paths": {
                    "mask_path": ann.get("mask_path"),
                    "crop_rgb_path": ann.get("crop_rgb_path"),
                    "crop_rgba_path": ann.get("crop_rgba_path"),
                },
            })
        return pointclouds

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        adapter = PerspectiveFieldsAdapter(
            version=self.config.model_version,
            weights_path=self.config.weights_path,
            config_file=self.config.config_file,
            device=self.runtime.device,
        )
        pred = adapter.predict_camera(self.config.image_path)
        image_w, image_h = Image.open(self.config.image_path).convert("RGB").size
        confidence = 0.92
        gravity = pred.pop("pred_gravity", None)
        latitude = pred.pop("pred_latitude", None)
        fx = float(pred["fx"])
        fy = float(pred["fy"])
        cx0 = float(pred["cx"])
        cy0 = float(pred["cy"])
        pitch_deg = float(pred["pitch_deg"])
        roll_deg = float(pred["roll_deg"])
        yaw_deg = 0.0

        preview_path = self.config.output_dir / "camera_preview.png"
        preview_summary_path = self.config.output_dir / "camera_preview_summary.json"
        pointcloud_path = self.config.output_dir / "camera_depth_pointcloud.png"
        pointcloud_summary_path = self.config.output_dir / "camera_depth_pointcloud_summary.json"
        payload = {
            "image_path": str(self.config.image_path),
            "mask_result_json_path": str(self.config.mask_result_json_path) if self.config.mask_result_json_path.exists() else None,
            "depth_result_json_path": str(self.config.depth_result_json_path) if self.config.depth_result_json_path.exists() else None,
            "image_size_wh": [image_w, image_h],
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx0,
                "cy": cy0,
                "vfov_deg": float(pred["vfov_deg"]),
            },
            "orientation": {
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "yaw_deg": yaw_deg,
                "horizon_y": float(pred["horizon_y"]),
                "vanishing_point_xy": [cx0, float(pred["horizon_y"])],
            },
            "priors": {
                "source": "PerspectiveFields",
                "model_version": self.config.model_version,
                "weights_path": str(self.config.weights_path),
                "raw_prediction_keys": pred["raw_prediction_keys"],
            },
            "fields": {
                "gravity_shape": list(gravity.shape) if gravity is not None else None,
                "latitude_shape": list(latitude.shape) if latitude is not None else None,
            },
            "coordinate_frame": {
                "name": "camera_origin_gravity_aligned",
                "origin": "camera_center",
                "axes": {
                    "x": "image-right after roll correction",
                    "y": "up after roll/pitch correction",
                    "z": "forward from the camera using absolute depth",
                },
                "pointcloud_axis_order": "xyz",
                "notes": [
                    "Single-image reconstruction in this project uses the camera-origin frame as the scene/world frame.",
                    "compose_layout consumes object point clouds already expressed in this frame.",
                ],
            },
            "depth_context": None,
            "object_pointclouds": [],
            "confidence": confidence,
            "camera_preview_path": str(preview_path),
            "camera_preview_summary_path": str(preview_summary_path),
            "camera_depth_pointcloud_path": str(pointcloud_path) if self.config.depth_result_json_path.exists() else None,
            "camera_depth_pointcloud_summary_path": str(pointcloud_summary_path) if self.config.depth_result_json_path.exists() else None,
        }
        if self.config.depth_result_json_path.exists() and self.config.mask_result_json_path.exists():
            depth_result = load_json(self.config.depth_result_json_path)
            mask_result = load_json(self.config.mask_result_json_path)
            payload["depth_context"] = {
                "depth_type": depth_result.get("depth_type"),
                "depth_unit": depth_result.get("depth_unit"),
                "global_depth_stats": depth_result.get("global_depth_stats"),
                "adapter_metadata": depth_result.get("adapter_metadata"),
                "absolute_depth_scale_multiplier": 1.0,
            }
            depth_npy_path = depth_result.get("depth_raw_npy_path")
            if depth_npy_path:
                depth = np.load(depth_npy_path)
                payload["object_pointclouds"] = self.build_object_pointclouds(
                    mask_annotations=mask_result.get("annotations", []),
                    depth=depth,
                    image_hw=(image_h, image_w),
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    yaw_deg=yaw_deg,
                    depth_scale=float(payload["depth_context"].get("absolute_depth_scale_multiplier", 1.0)),
                )
                render_camera_calibrated_pointcloud(
                    object_pointclouds=payload["object_pointclouds"],
                    png_path=pointcloud_path,
                    summary_path=pointcloud_summary_path,
                )
        save_json(output_path, payload)
        render_camera_estimate_visualization(
            raw_image_path=self.config.image_path,
            camera_payload={
                **payload,
                "fields": {
                    "gravity": gravity.tolist() if gravity is not None else None,
                    "latitude": latitude.tolist() if latitude is not None else None,
                },
            },
            png_path=preview_path,
            summary_path=preview_summary_path,
        )
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"confidence": confidence, "object_pointclouds": len(payload["object_pointclouds"])})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the estimate_camera stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = EstimateCameraStage(config=CameraEstimationConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
