from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from image_to_world.adapters.defaults import DepthAnythingEstimator
from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import DepthEstimationConfig, RuntimeConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import DepthAnnotation, StageResult
from image_to_world.stages.base import Stage
from image_to_world.visualization.depth_viz import render_depth_object_pointcloud


class EstimateDepthStage(Stage):
    stage_name = "estimate_depth"

    def __init__(self, *, config: DepthEstimationConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime
        self.adapter = DepthAnythingEstimator(config, runtime.device)

    @staticmethod
    def normalize_to_uint(depth: np.ndarray, scale: int, dtype) -> np.ndarray:
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max - depth_min < 1e-8:
            return np.zeros_like(depth, dtype=dtype)
        norm = (depth - depth_min) / (depth_max - depth_min)
        return (norm * scale).clip(0, scale).astype(dtype)

    @staticmethod
    def load_mask(mask_path: str, target_hw: tuple[int, int]) -> np.ndarray | None:
        path = Path(mask_path)
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        th, tw = target_hw
        if mask.shape[:2] != (th, tw):
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8)

    @staticmethod
    def robust_mask_stats(depth: np.ndarray, mask: np.ndarray | None) -> dict[str, Any] | None:
        if mask is None:
            return None
        values = depth[mask == 1]
        if values.size == 0:
            return None
        return {
            "depth_min": float(values.min()),
            "depth_max": float(values.max()),
            "depth_mean": float(values.mean()),
            "depth_median": float(np.median(values)),
            "depth_std": float(values.std()),
            "num_pixels": int(values.size),
        }

    @staticmethod
    def get_center_depth(depth: np.ndarray, bbox_xyxy: list[float]) -> float:
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        cx = max(0, min(int(round((x1 + x2) * 0.5)), w - 1))
        cy = max(0, min(int(round((y1 + y2) * 0.5)), h - 1))
        return float(depth[cy, cx])

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        image = Image.open(self.config.image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        depth = self.adapter.estimate(image)
        depth_npy_path = self.config.output_dir / "depth_raw.npy"
        np.save(depth_npy_path, depth)
        depth_u8 = self.normalize_to_uint(depth, 255, np.uint8)
        depth_u16 = self.normalize_to_uint(depth, 65535, np.uint16)
        depth_gray_path = self.config.output_dir / "depth_gray_8bit.png"
        depth_gray16_path = self.config.output_dir / "depth_gray_16bit.png"
        depth_color_path = self.config.output_dir / "depth_color.png"
        cv2.imwrite(str(depth_gray_path), depth_u8)
        cv2.imwrite(str(depth_gray16_path), depth_u16)
        cv2.imwrite(str(depth_color_path), cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO))

        annotations_out = []
        if self.config.mask_result_json_path.exists():
            mask_result = load_json(self.config.mask_result_json_path)
            for ann in mask_result.get("annotations", []):
                mask = self.load_mask(ann.get("mask_path"), (h, w)) if ann.get("mask_path") else None
                annotations_out.append(DepthAnnotation(
                    id=ann.get("id"),
                    class_name=ann.get("class_name"),
                    mask_path=ann.get("mask_path"),
                    bbox_xyxy=ann.get("bbox_xyxy"),
                    bbox_center_depth=self.get_center_depth(depth, ann["bbox_xyxy"]) if ann.get("bbox_xyxy") else None,
                    mask_depth_stats=self.robust_mask_stats(depth, mask),
                ).to_dict())

        pointcloud_png_path = self.config.output_dir / "depth_objects_pointcloud.png"
        pointcloud_summary_path = self.config.output_dir / "depth_objects_pointcloud_summary.json"
        if annotations_out:
            render_depth_object_pointcloud(
                depth=depth,
                annotations=annotations_out,
                mask_loader=self.load_mask,
                png_path=pointcloud_png_path,
                summary_path=pointcloud_summary_path,
            )

        save_json(output_path, {
            "image_path": str(self.config.image_path),
            "mask_result_json_path": str(self.config.mask_result_json_path) if self.config.mask_result_json_path.exists() else None,
            "model_id": self.config.model_id,
            "device": self.runtime.device,
            "depth_type": "relative",
            "image_size_wh": [w, h],
            "depth_raw_npy_path": str(depth_npy_path),
            "depth_gray_8bit_path": str(depth_gray_path),
            "depth_gray_16bit_path": str(depth_gray16_path),
            "depth_color_path": str(depth_color_path),
            "depth_object_pointcloud_path": str(pointcloud_png_path) if annotations_out else None,
            "depth_object_pointcloud_summary_path": str(pointcloud_summary_path) if annotations_out else None,
            "global_depth_stats": {
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "depth_mean": float(depth.mean()),
                "depth_median": float(np.median(depth)),
                "depth_std": float(depth.std()),
            },
            "annotations": annotations_out,
        })
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": len(annotations_out)})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the estimate_depth stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = EstimateDepthStage(config=DepthEstimationConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
