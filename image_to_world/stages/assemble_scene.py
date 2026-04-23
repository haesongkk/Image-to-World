from __future__ import annotations

import argparse
from pathlib import Path

from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import RuntimeConfig, SceneAssemblyConfig
from image_to_world.io.external_glb_assembler import assemble_scene_glb_external
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage


class AssembleSceneStage(Stage):
    stage_name = "assemble_scene"

    def __init__(self, *, config: SceneAssemblyConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "assembly_result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))

        self.ensure_output_dir(self.config.output_dir)
        scene_layout = load_json(self.config.input_layout_json_path)
        placements = scene_layout.get("placements", [])

        assembly_records: list[dict] = []
        skipped: list[dict] = []
        for idx, placement in enumerate(placements):
            primitive_type = placement.get("primitive", {}).get("type")
            mesh_info = placement.get("mesh", {})
            mesh_path = mesh_info.get("mesh_path")
            mesh_format = (mesh_info.get("mesh_format") or Path(mesh_path).suffix.lstrip(".") if mesh_path else "").lower()
            if mesh_path and not Path(mesh_path).exists():
                mesh_path = None
                mesh_format = ""
            pseudo = placement.get("pseudo_world", {})
            # Temporary workaround requested by user:
            # invert Z translation sign during assembly.
            pos_xyz = list(pseudo.get("position_xyz", [0.0, 0.0, 0.0]))
            if len(pos_xyz) < 3:
                pos_xyz = [0.0, 0.0, 0.0]
            pos_xyz[2] = -float(pos_xyz[2])
            if mesh_path and mesh_format:
                assembly_records.append({
                    "kind": "mesh",
                    "mesh_path": mesh_path,
                    "mesh_format": mesh_format,
                    "scale_xyz": pseudo.get("scale_xyz", [1.0, 1.0, 1.0]),
                    "rotation_euler_xyz_deg": pseudo.get("rotation_euler_xyz_deg", [0.0, 0.0, 0.0]),
                    "position_xyz": pos_xyz,
                    "global_pre_rot_euler_deg": self.config.global_pre_rot_euler_deg,
                    "normalize_mesh_to_unit_box": self.config.normalize_mesh_to_unit_box,
                    "placement_id": placement.get("id"),
                    "class_name": placement.get("class_name"),
                })
            elif primitive_type in {"cube", "cuboid"}:
                assembly_records.append({
                    "kind": "cube",
                    "scale_xyz": pseudo.get("scale_xyz", [1.0, 1.0, 1.0]),
                    "rotation_euler_xyz_deg": pseudo.get("rotation_euler_xyz_deg", [0.0, 0.0, 0.0]),
                    "position_xyz": pos_xyz,
                    "global_pre_rot_euler_deg": self.config.global_pre_rot_euler_deg,
                    "normalize_mesh_to_unit_box": False,
                    "placement_id": placement.get("id"),
                    "class_name": placement.get("class_name"),
                })
            else:
                skipped.append({
                    "id": placement.get("id", idx),
                    "class_name": placement.get("class_name"),
                    "reason": f"mesh path missing or invalid: {mesh_info.get('mesh_path')}",
                })

        merged_glb_path = self.config.output_dir / "assembled_scene.glb"
        external_summary_path = self.config.output_dir / "assembled_scene_external_summary.json"
        if assembly_records:
            assembly_payload = assemble_scene_glb_external(
                records=assembly_records,
                output_glb_path=merged_glb_path,
                python_path=self.config.mesh_converter_python,
                repo_dir=self.config.mesh_converter_repo_dir,
            )
        else:
            assembly_payload = {
                "output_glb_path": None,
                "num_assembled_objects": 0,
                "num_skipped_objects": 0,
                "scene_bounds": {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [0.0, 0.0, 0.0], "center": [0.0, 0.0, 0.0], "size_xyz": [0.0, 0.0, 0.0]},
            }

        preview_csv_path = self.config.output_dir / "scene_preview_points.csv"
        with preview_csv_path.open("w", encoding="utf-8") as handle:
            handle.write("placement_id,class_name,center_x,center_y,center_z\n")
            for rec in assembly_records:
                center = rec.get("position_xyz", [0.0, 0.0, 0.0])
                handle.write(f"{rec.get('placement_id')},{rec.get('class_name')},{float(center[0]):.8f},{float(center[1]):.8f},{float(center[2]):.8f}\n")

        preview_summary_path = self.config.output_dir / "scene_assembly_preview_summary.json"
        if self.config.save_visualization:
            save_json(preview_summary_path, {
                "type": "glb_assembly_preview",
                "note": "GLB-native assembly mode does not generate a 2D preview image in this stage.",
                "output_glb_path": str(merged_glb_path) if merged_glb_path.exists() else None,
            })

        save_json(output_path, {
            "input_layout_json_path": str(self.config.input_layout_json_path),
            "output_glb_path": str(merged_glb_path) if merged_glb_path.exists() else None,
            "external_summary_path": str(external_summary_path) if external_summary_path.exists() else None,
            "preview_csv_path": str(preview_csv_path),
            "preview_png_path": None,
            "preview_summary_path": str(preview_summary_path) if self.config.save_visualization else None,
            "num_input_placements": len(placements),
            "num_assembled_objects": int(assembly_payload.get("num_assembled_objects", 0)),
            "num_skipped_objects": int(assembly_payload.get("num_skipped_objects", 0)) + len(skipped),
            "scene_bounds": assembly_payload.get("scene_bounds", {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [0.0, 0.0, 0.0], "center": [0.0, 0.0, 0.0], "size_xyz": [0.0, 0.0, 0.0]}),
            "settings": {
                "assembly_output_format": "glb",
                "add_axis_helper": self.config.add_axis_helper,
                "axis_length": self.config.axis_length,
                "normalize_mesh_to_unit_box": self.config.normalize_mesh_to_unit_box,
                "global_pre_rot_euler_deg": self.config.global_pre_rot_euler_deg,
                "save_visualization": self.config.save_visualization,
                "visualization_backend": self.config.visualization_backend,
                "visualization_device": self.config.visualization_device,
                "visualization_image_size": self.config.visualization_image_size,
                "visualization_point_size": self.config.visualization_point_size,
                "visualization_max_faces_per_object": self.config.visualization_max_faces_per_object,
                "mesh_converter_repo_dir": str(self.config.mesh_converter_repo_dir),
                "mesh_converter_python": str(self.config.mesh_converter_python),
                "object_color_palette": [list(c) for c in self.config.object_color_palette],
            },
            "assembled_objects": assembly_records,
            "external_assembly": assembly_payload,
            "skipped": skipped,
        })
        result = StageResult(
            stage_name=self.stage_name,
            output_path=output_path,
            metadata={
                "count": int(assembly_payload.get("num_assembled_objects", 0)),
                "skipped": int(assembly_payload.get("num_skipped_objects", 0)) + len(skipped),
            },
        )
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the assemble_scene stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = AssembleSceneStage(config=SceneAssemblyConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
