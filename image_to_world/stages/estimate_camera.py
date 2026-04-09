from __future__ import annotations

import argparse

from image_to_world.adapters.perspective_fields import PerspectiveFieldsAdapter
from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import CameraEstimationConfig, RuntimeConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage
from image_to_world.visualization.camera_viz import render_camera_calibrated_pointcloud, render_camera_estimate_visualization
from PIL import Image
import numpy as np


class EstimateCameraStage(Stage):
    stage_name = "estimate_camera"

    def __init__(self, *, config: CameraEstimationConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime

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
                "fx": float(pred["fx"]),
                "fy": float(pred["fy"]),
                "cx": float(pred["cx"]),
                "cy": float(pred["cy"]),
                "vfov_deg": float(pred["vfov_deg"]),
            },
            "orientation": {
                "pitch_deg": float(pred["pitch_deg"]),
                "roll_deg": float(pred["roll_deg"]),
                "yaw_deg": 0.0,
                "horizon_y": float(pred["horizon_y"]),
                "vanishing_point_xy": [float(pred["cx"]), float(pred["horizon_y"])],
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
            "depth_context": None,
            "confidence": confidence,
            "camera_preview_path": str(preview_path),
            "camera_preview_summary_path": str(preview_summary_path),
            "camera_depth_pointcloud_path": str(pointcloud_path) if self.config.depth_result_json_path.exists() else None,
            "camera_depth_pointcloud_summary_path": str(pointcloud_summary_path) if self.config.depth_result_json_path.exists() else None,
        }
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
        if self.config.depth_result_json_path.exists() and self.config.mask_result_json_path.exists():
            depth_result = load_json(self.config.depth_result_json_path)
            depth_npy_path = depth_result.get("depth_raw_npy_path")
            annotations = depth_result.get("annotations", [])
            if depth_npy_path:
                depth = np.load(depth_npy_path)
                render_camera_calibrated_pointcloud(
                    depth=depth,
                    annotations=annotations,
                    camera_payload=payload,
                    png_path=pointcloud_path,
                    summary_path=pointcloud_summary_path,
                )
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"confidence": confidence})
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
