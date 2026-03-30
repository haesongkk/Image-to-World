from __future__ import annotations

import argparse
import math
from pathlib import Path

from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import RuntimeConfig, SceneLayoutConfig
from image_to_world.geometry import clamp
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import MeshArtifact, Placement, StageResult
from image_to_world.stages.base import Stage
from image_to_world.visualization.layout_viz import render_layout_visualization


class ComposeLayoutStage(Stage):
    stage_name = "compose_layout"

    def __init__(self, *, config: SceneLayoutConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime

    @staticmethod
    def bbox_center_and_size(bbox_xyxy: list[float]) -> dict[str, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return {"cx": float(x1 + x2) * 0.5, "cy": float(y1 + y2) * 0.5, "w": max(1.0, float(x2 - x1)), "h": max(1.0, float(y2 - y1))}

    @staticmethod
    def choose_object_depth(depth_ann: dict) -> float | None:
        mask_stats = depth_ann.get("mask_depth_stats")
        if isinstance(mask_stats, dict) and mask_stats.get("depth_median") is not None:
            return float(mask_stats["depth_median"])
        if depth_ann.get("bbox_center_depth") is not None:
            return float(depth_ann["bbox_center_depth"])
        return None

    @staticmethod
    def build_index_by_id(items: list[dict]) -> dict[int, dict]:
        return {int(item["id"]): item for item in items if item.get("id") is not None}

    def normalize_depth_value(self, depth_value: float, dmin: float, dmax: float) -> float:
        if dmax - dmin < 1e-8:
            return 0.5
        return clamp((depth_value - dmin) / (dmax - dmin), 0.0, 1.0)

    def relative_depth_to_pseudo_z(self, depth_value: float, dmin: float, dmax: float) -> float:
        depth_norm = self.normalize_depth_value(depth_value, dmin, dmax)
        closeness = depth_norm if self.config.close_is_larger else (1.0 - depth_norm)
        return float(self.config.z_far - closeness * (self.config.z_far - self.config.z_near))

    def estimate_world_position(self, cx_px: float, cy_px: float, z: float, image_w: float, image_h: float, fx: float, fy: float) -> dict[str, float]:
        cx0 = image_w * 0.5
        cy0 = image_h * 0.5
        return {"x": float(((cx_px - cx0) / fx) * z), "y": float(-((cy_px - cy0) / fy) * z), "z": float(z)}

    def estimate_world_size(self, bbox_w_px: float, bbox_h_px: float, z: float, fx: float, fy: float) -> dict[str, float]:
        world_w = max(1e-6, (bbox_w_px / fx) * z * self.config.global_scale_multiplier)
        world_h = max(1e-6, (bbox_h_px / fy) * z * self.config.global_scale_multiplier)
        return {"world_w": float(world_w), "world_h": float(world_h), "suggested_uniform_scale": float(math.sqrt(world_w * world_h))}

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "scene_layout.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        mask_data = load_json(self.config.mask_json_path)
        depth_data = load_json(self.config.depth_json_path)
        mesh_data = load_json(self.config.gen3d_json_path)
        mask_annotations = mask_data.get("annotations", [])
        depth_by_id = self.build_index_by_id(depth_data.get("annotations", []))
        mesh_by_id = self.build_index_by_id(mesh_data.get("results", []))
        if not mask_annotations:
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": 0}, skipped=True), self.config_to_cache_payload(self.config))

        image_w, image_h = map(float, depth_data["image_size_wh"])
        fx = image_w * self.config.focal_scale_x
        fy = image_h * self.config.focal_scale_y
        global_stats = depth_data.get("global_depth_stats", {})
        dmin = float(global_stats.get("depth_min", 0.0))
        dmax = float(global_stats.get("depth_max", 1.0))

        placements = []
        skipped = []
        for mask_ann in mask_annotations:
            obj_id = int(mask_ann["id"])
            label = mask_ann.get("class_name", "unknown")
            bbox_xyxy = mask_ann.get("bbox_xyxy")
            depth_ann = depth_by_id.get(obj_id)
            mesh_ann = mesh_by_id.get(obj_id)
            if bbox_xyxy is None or depth_ann is None or mesh_ann is None:
                skipped.append({"id": obj_id, "class_name": label, "reason": "missing bbox, depth annotation, or mesh artifact"})
                continue
            depth_value = self.choose_object_depth(depth_ann)
            if depth_value is None:
                skipped.append({"id": obj_id, "class_name": label, "reason": "usable depth not found"})
                continue
            bbox_info = self.bbox_center_and_size(bbox_xyxy)
            pseudo_z = self.relative_depth_to_pseudo_z(depth_value, dmin, dmax)
            pos = self.estimate_world_position(bbox_info["cx"], bbox_info["cy"], pseudo_z, image_w, image_h, fx, fy)
            size = self.estimate_world_size(bbox_info["w"], bbox_info["h"], pseudo_z, fx, fy)
            placements.append(Placement(
                id=obj_id,
                class_name=label,
                score=mask_ann.get("score"),
                bbox_xyxy=bbox_xyxy,
                bbox_center_xy=[bbox_info["cx"], bbox_info["cy"]],
                bbox_size_wh=[bbox_info["w"], bbox_info["h"]],
                depth_source="mask_depth_stats.depth_median" if depth_ann.get("mask_depth_stats") else "bbox_center_depth",
                relative_depth_value=float(depth_value),
                pseudo_world={
                    "position_xyz": [pos["x"], pos["y"], pos["z"]],
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "scale_xyz": [size["suggested_uniform_scale"]] * 3,
                    "estimated_world_size_wh": [size["world_w"], size["world_h"]],
                },
                mesh=MeshArtifact(
                    obj_path=mesh_ann.get("obj_path"),
                    ply_path=mesh_ann.get("ply_path"),
                    meta_path=str(Path(mesh_ann.get("obj_path", "")).with_name("meta.json")) if mesh_ann.get("obj_path") else None,
                ),
                source_paths={
                    "mask_path": mask_ann.get("mask_path"),
                    "crop_rgb_path": mask_ann.get("crop_rgb_path"),
                    "crop_rgba_path": mask_ann.get("crop_rgba_path"),
                },
            ).to_dict())

        placements = sorted(placements, key=lambda item: item["pseudo_world"]["position_xyz"][2])
        save_json(output_path, {
            "mask_json_path": str(self.config.mask_json_path),
            "depth_json_path": str(self.config.depth_json_path),
            "mesh_json_path": str(self.config.gen3d_json_path),
            "image_size_wh": [int(image_w), int(image_h)],
            "camera_assumption": {
                "type": "pseudo_pinhole_camera",
                "fx": fx,
                "fy": fy,
                "cx": image_w * 0.5,
                "cy": image_h * 0.5,
                "close_is_larger": self.config.close_is_larger,
                "z_near": self.config.z_near,
                "z_far": self.config.z_far,
                "global_scale_multiplier": self.config.global_scale_multiplier,
                "notes": [
                    "Relative depth is converted into a pseudo scene depth range.",
                    "These placements are initialization values and still need later calibration.",
                ],
            },
            "num_requested_objects": len(mask_annotations),
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
