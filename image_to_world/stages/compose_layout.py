from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

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

    def fit_axis_aligned_box(self, points_xyz: np.ndarray) -> dict[str, object] | None:
        if points_xyz.shape[0] < self.MIN_POINT_COUNT:
            return None
        valid = np.isfinite(points_xyz).all(axis=1)
        points_xyz = points_xyz[valid]
        if points_xyz.shape[0] < self.MIN_POINT_COUNT:
            return None

        points_fit = points_xyz
        min_x, max_x = self.percentile_span(points_fit[:, 0], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_y, max_y = self.percentile_span(points_fit[:, 1], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)
        min_z, max_z = self.percentile_span(points_fit[:, 2], self.BOUNDS_LOW_Q, self.BOUNDS_HIGH_Q)

        width_world = max(1e-3, max_x - min_x)
        height_world = max(1e-3, max_y - min_y)
        depth_world = max(1e-3, max_z - min_z)
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        center_z = 0.5 * (min_z + max_z)

        return {
            "position_xyz": [float(center_x), float(center_y), float(center_z)],
            "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
            "scale_xyz": [width_world, height_world, depth_world],
            "estimated_world_size_wh": [width_world, height_world],
            "estimated_world_size_xyz": [width_world, height_world, depth_world],
            "support_type": "unknown",
            "analysis": {
                "point_count": int(points_xyz.shape[0]),
                "obb_source": "axis_aligned_percentile_bounds_no_rotation",
                "bounds_x": [min_x, max_x],
                "bounds_y": [min_y, max_y],
                "bounds_z": [min_z, max_z],
            },
        }

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "scene_layout.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        camera_data = load_json(self.config.camera_json_path) if self.config.camera_json_path.exists() else {}
        mesh_data = load_json(self.config.gen3d_json_path) if self.config.gen3d_json_path.exists() else {}
        mesh_by_id = self.build_index_by_id(mesh_data.get("results", []))
        object_pointclouds = camera_data.get("object_pointclouds", [])
        image_size_wh = camera_data.get("image_size_wh")
        if not object_pointclouds:
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": 0}, skipped=True), self.config_to_cache_payload(self.config))
        image_w, image_h = image_size_wh if isinstance(image_size_wh, list) and len(image_size_wh) == 2 else [0, 0]

        placements = []
        skipped = []
        for object_ann in object_pointclouds:
            obj_id = int(object_ann["id"])
            label = object_ann.get("class_name", "unknown")
            bbox_xyxy = object_ann.get("bbox_xyxy")
            pointcloud_path = object_ann.get("pointcloud_path")
            mesh_ann = mesh_by_id.get(obj_id)
            points_xyz = self.load_pointcloud(pointcloud_path)
            if points_xyz is None:
                skipped.append({"id": obj_id, "class_name": label, "reason": f"pointcloud missing or invalid: {pointcloud_path}"})
                continue
            cube_world = self.fit_axis_aligned_box(points_xyz)
            if cube_world is None:
                skipped.append({"id": obj_id, "class_name": label, "reason": "insufficient valid pointcloud samples"})
                continue
            bbox_info = self.bbox_center_and_size(bbox_xyxy) if bbox_xyxy is not None else {"cx": 0.0, "cy": 0.0, "w": 1.0, "h": 1.0}
            placements.append(Placement(
                id=obj_id,
                class_name=label,
                score=object_ann.get("score"),
                bbox_xyxy=bbox_xyxy,
                bbox_center_xy=[bbox_info["cx"], bbox_info["cy"]],
                bbox_size_wh=[bbox_info["w"], bbox_info["h"]],
                depth_source="camera_object_pointcloud",
                depth_value=float(cube_world["position_xyz"][2]),
                pseudo_world=cube_world,
                mesh=MeshArtifact(
                    obj_path=mesh_ann.get("obj_path") if mesh_ann else None,
                    ply_path=mesh_ann.get("ply_path") if mesh_ann else None,
                    meta_path=str(Path(mesh_ann.get("obj_path", "")).with_name("meta.json")) if mesh_ann and mesh_ann.get("obj_path") else None,
                ),
                source_paths={
                    **object_ann.get("source_paths", {}),
                    "pointcloud_path": pointcloud_path,
                },
            ).to_dict())
            placements[-1]["primitive"] = {
                "type": "cuboid",
                "size_xyz": list(cube_world["scale_xyz"]),
            }

        placements = sorted(placements, key=lambda item: item["pseudo_world"]["position_xyz"][2])
        save_json(output_path, {
            "camera_json_path": str(self.config.camera_json_path) if self.config.camera_json_path.exists() else None,
            "mesh_json_path": str(self.config.gen3d_json_path) if self.config.gen3d_json_path.exists() else None,
            "image_size_wh": [int(image_w), int(image_h)],
            "coordinate_frame": camera_data.get("coordinate_frame"),
            "layout_method": {
                "type": "yaw_only_obb_from_object_pointclouds",
                "notes": [
                    "compose_layout consumes calibrated object point clouds from estimate_camera.",
                    "Each object is summarized as a yaw-only oriented bounding box in the shared camera-origin frame.",
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
