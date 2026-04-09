from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import RuntimeConfig, SceneAssemblyConfig
from image_to_world.geometry import apply_transform, compose_transform, rotation_matrix_xyz_deg
from image_to_world.io.obj_utils import load_obj_basic
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage
from image_to_world.visualization.scene_assembly_viz_torch import render_assembled_scene_visualization_torch


AXIS_MATERIALS = {
    "axis_x": (1.0, 0.15, 0.15),
    "axis_y": (0.15, 0.85, 0.15),
    "axis_z": (0.15, 0.35, 1.0),
}


class AssembleSceneStage(Stage):
    stage_name = "assemble_scene"
    UNIT_CUBE_VERTICES = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ], dtype=np.float64)
    UNIT_CUBE_FACES = [
        (0, 1, 2), (0, 2, 3),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]

    def __init__(self, *, config: SceneAssemblyConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime

    @staticmethod
    def normalize_vertices_to_centered_unit_box(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        center = (vmin + vmax) * 0.5
        size = np.maximum(vmax - vmin, 1e-8)
        max_extent = float(size.max())
        return (vertices - center) / max_extent, {
            "orig_bbox_min": vmin.tolist(),
            "orig_bbox_max": vmax.tolist(),
            "orig_center": center.tolist(),
            "orig_size_xyz": size.tolist(),
            "normalization_divisor": max_extent,
        }

    def build_transformed_mesh_record(self, obj_path: str, placement: dict, object_index: int) -> dict:
        mesh = load_obj_basic(obj_path)
        vertices = mesh["vertices"]
        normalization_info = None
        if self.config.normalize_mesh_to_unit_box:
            vertices, normalization_info = self.normalize_vertices_to_centered_unit_box(vertices)
        pseudo = placement["pseudo_world"]
        scale_xyz = pseudo.get("scale_xyz", [1.0, 1.0, 1.0])
        rot_xyz_deg = pseudo.get("rotation_euler_xyz_deg", [0.0, 0.0, 0.0])
        pos_xyz = pseudo.get("position_xyz", [0.0, 0.0, 0.0])
        pos_xyz = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]
        pre_rot = rotation_matrix_xyz_deg(self.config.global_pre_rot_euler_deg)
        vertices = (pre_rot @ vertices.T).T
        transform = compose_transform(scale_xyz=scale_xyz, rot_xyz_deg=rot_xyz_deg, translate_xyz=pos_xyz)
        vertices_world = apply_transform(vertices, transform)
        class_name = placement.get("class_name") or f"object_{object_index:02d}"
        return {
            "source_obj_path": obj_path,
            "vertices_world": vertices_world,
            "faces_v": mesh["faces_v"],
            "normalization_info": normalization_info,
            "applied_scale_xyz": scale_xyz,
            "applied_rotation_euler_xyz_deg": rot_xyz_deg,
            "applied_translation_xyz": pos_xyz,
            "object_name": f"object_{object_index:02d}_{class_name}".replace(" ", "_"),
            "material_name": f"mat_{object_index:02d}_{class_name}".replace(" ", "_"),
            "material_rgb": list(self.config.object_color_palette[object_index % len(self.config.object_color_palette)]),
            "placement_id": placement.get("id"),
            "class_name": class_name,
        }

    def build_cube_record(self, placement: dict, object_index: int) -> dict:
        pseudo = placement["pseudo_world"]
        scale_xyz = pseudo.get("scale_xyz", [1.0, 1.0, 1.0])
        rot_xyz_deg = pseudo.get("rotation_euler_xyz_deg", [0.0, 0.0, 0.0])
        pos_xyz = pseudo.get("position_xyz", [0.0, 0.0, 0.0])
        pos_xyz = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])]
        pre_rot = rotation_matrix_xyz_deg(self.config.global_pre_rot_euler_deg)
        vertices = (pre_rot @ self.UNIT_CUBE_VERTICES.T).T
        transform = compose_transform(scale_xyz=scale_xyz, rot_xyz_deg=rot_xyz_deg, translate_xyz=pos_xyz)
        vertices_world = apply_transform(vertices, transform)
        class_name = placement.get("class_name") or f"object_{object_index:02d}"
        return {
            "source_obj_path": None,
            "source_primitive": "cube",
            "vertices_world": vertices_world,
            "faces_v": list(self.UNIT_CUBE_FACES),
            "normalization_info": {"source": "unit_cube_proxy"},
            "applied_scale_xyz": scale_xyz,
            "applied_rotation_euler_xyz_deg": rot_xyz_deg,
            "applied_translation_xyz": pos_xyz,
            "object_name": f"object_{object_index:02d}_{class_name}".replace(" ", "_"),
            "material_name": f"mat_{object_index:02d}_{class_name}".replace(" ", "_"),
            "material_rgb": list(self.config.object_color_palette[object_index % len(self.config.object_color_palette)]),
            "placement_id": placement.get("id"),
            "class_name": class_name,
        }

    @staticmethod
    def world_to_export_vertices(vertices_world: np.ndarray) -> np.ndarray:
        # Compatibility note:
        # We mirror X at OBJ export ([x, y, z] -> [-x, y, z]) to match current Blender-facing output.
        # It compensates an upstream axis-handedness mismatch and should be treated as a temporary compatibility fix.

        return np.stack([-vertices_world[:, 0], vertices_world[:, 1], vertices_world[:, 2]], axis=1)

    @staticmethod
    def write_axis_helper_obj_lines(handle, start_vertex_index_1based: int, length: float) -> None:
        verts_world = np.array(
            [(0.0, 0.0, 0.0), (length, 0.0, 0.0), (0.0, length, 0.0), (0.0, 0.0, length)],
            dtype=np.float64,
        )
        verts = AssembleSceneStage.world_to_export_vertices(verts_world)
        for v in verts:
            handle.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        base = start_vertex_index_1based
        handle.write("o axis_helper_x\nusemtl axis_x\n")
        handle.write(f"l {base} {base + 1}\n\n")
        handle.write("o axis_helper_y\nusemtl axis_y\n")
        handle.write(f"l {base} {base + 2}\n\n")
        handle.write("o axis_helper_z\nusemtl axis_z\n")
        handle.write(f"l {base} {base + 3}\n\n")

    def write_mtl(self, path: Path, transformed_meshes: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# assembled scene materials\n\n")
            for rec in transformed_meshes:
                r, g, b = rec["material_rgb"]
                handle.write(f"newmtl {rec['material_name']}\n")
                handle.write(f"Ka {0.15 * r:.6f} {0.15 * g:.6f} {0.15 * b:.6f}\n")
                handle.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
                handle.write("Ks 0.050000 0.050000 0.050000\nNs 10.000000\nillum 2\n\n")
            for mat_name, (r, g, b) in AXIS_MATERIALS.items():
                handle.write(f"newmtl {mat_name}\n")
                handle.write(f"Ka {0.15 * r:.6f} {0.15 * g:.6f} {0.15 * b:.6f}\n")
                handle.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\nKs 0.000000 0.000000 0.000000\nNs 1.000000\nillum 1\n\n")

    def write_merged_obj(self, obj_path: Path, mtl_filename: str, transformed_meshes: list[dict]) -> None:
        with obj_path.open("w", encoding="utf-8") as handle:
            handle.write("# assembled scene OBJ\n")
            handle.write(f"# num_objects = {len(transformed_meshes)}\n")
            handle.write(f"mtllib {mtl_filename}\n\n")
            global_vertex_offset = 0
            for rec in transformed_meshes:
                handle.write(f"o {rec['object_name']}\n")
                handle.write(f"# source_obj_path = {rec['source_obj_path']}\n")
                handle.write(f"usemtl {rec['material_name']}\n")
                export_vertices = self.world_to_export_vertices(rec["vertices_world"])
                for vertex in export_vertices:
                    handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
                for face in rec["faces_v"]:
                    a = global_vertex_offset + face[0] + 1
                    b = global_vertex_offset + face[1] + 1
                    c = global_vertex_offset + face[2] + 1
                    handle.write(f"f {a} {b} {c}\n")
                handle.write("\n")
                global_vertex_offset += len(rec["vertices_world"])
            if self.config.add_axis_helper:
                self.write_axis_helper_obj_lines(handle, global_vertex_offset + 1, self.config.axis_length)

    @staticmethod
    def compute_scene_bounds(transformed_meshes: list[dict]) -> dict:
        if not transformed_meshes:
            return {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [0.0, 0.0, 0.0], "center": [0.0, 0.0, 0.0], "size_xyz": [0.0, 0.0, 0.0]}
        all_v = np.concatenate([rec["vertices_world"] for rec in transformed_meshes], axis=0)
        vmin = all_v.min(axis=0)
        vmax = all_v.max(axis=0)
        center = (vmin + vmax) * 0.5
        return {"bbox_min": vmin.tolist(), "bbox_max": vmax.tolist(), "center": center.tolist(), "size_xyz": (vmax - vmin).tolist()}

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "assembly_result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        scene_layout = load_json(self.config.input_layout_json_path)
        placements = scene_layout.get("placements", [])
        transformed_meshes = []
        skipped = []
        for idx, placement in enumerate(placements):
            primitive_type = placement.get("primitive", {}).get("type")
            obj_path = placement.get("mesh", {}).get("obj_path")
            try:
                if obj_path and Path(obj_path).exists():
                    transformed_meshes.append(self.build_transformed_mesh_record(obj_path, placement, idx))
                elif primitive_type in {"cube", "cuboid"}:
                    transformed_meshes.append(self.build_cube_record(placement, idx))
                else:
                    skipped.append({"id": placement.get("id", idx), "class_name": placement.get("class_name"), "reason": f"obj_path missing or not found: {obj_path}"})
            except Exception as exc:
                skipped.append({"id": placement.get("id", idx), "class_name": placement.get("class_name"), "reason": str(exc)})

        merged_obj_path = self.config.output_dir / "assembled_scene.obj"
        merged_mtl_path = self.config.output_dir / "assembled_scene.mtl"
        self.write_mtl(merged_mtl_path, transformed_meshes)
        self.write_merged_obj(merged_obj_path, merged_mtl_path.name, transformed_meshes)

        preview_csv_path = self.config.output_dir / "scene_preview_points.csv"
        with preview_csv_path.open("w", encoding="utf-8") as handle:
            handle.write("object_name,material_name,r,g,b,center_x,center_y,center_z\n")
            for rec in transformed_meshes:
                center = rec["vertices_world"].mean(axis=0)
                r, g, b = rec["material_rgb"]
                handle.write(f"{rec['object_name']},{rec['material_name']},{r:.6f},{g:.6f},{b:.6f},{center[0]:.8f},{center[1]:.8f},{center[2]:.8f}\n")

        preview_png_path = self.config.output_dir / "scene_assembly_preview.png"
        preview_summary_path = self.config.output_dir / "scene_assembly_preview_summary.json"
        if self.config.save_visualization:
            backend = (self.config.visualization_backend or "auto").strip().lower()
            if backend in {"auto", "torch", "cuda"}:
                render_assembled_scene_visualization_torch(
                    transformed_meshes=transformed_meshes,
                    png_path=preview_png_path,
                    summary_path=preview_summary_path,
                    device=self.config.visualization_device,
                    image_size=self.config.visualization_image_size,
                    point_size=max(1, int(self.config.visualization_point_size)),
                )
            else:
                raise ValueError(f"Unsupported scene assembly visualization backend: {backend}")

        save_json(output_path, {
            "input_layout_json_path": str(self.config.input_layout_json_path),
            "output_obj_path": str(merged_obj_path),
            "output_mtl_path": str(merged_mtl_path),
            "preview_csv_path": str(preview_csv_path),
            "preview_png_path": str(preview_png_path) if self.config.save_visualization else None,
            "preview_summary_path": str(preview_summary_path) if self.config.save_visualization else None,
            "num_input_placements": len(placements),
            "num_assembled_objects": len(transformed_meshes),
            "num_skipped_objects": len(skipped),
            "scene_bounds": self.compute_scene_bounds(transformed_meshes),
            "settings": {
                "add_axis_helper": self.config.add_axis_helper,
                "axis_length": self.config.axis_length,
                "normalize_mesh_to_unit_box": self.config.normalize_mesh_to_unit_box,
                "global_pre_rot_euler_deg": self.config.global_pre_rot_euler_deg,
                "obj_export_axis_mapping": "blender_mirror_x_[-x,y,z]",
                "save_visualization": self.config.save_visualization,
                "visualization_backend": self.config.visualization_backend,
                "visualization_device": self.config.visualization_device,
                "visualization_image_size": self.config.visualization_image_size,
                "visualization_point_size": self.config.visualization_point_size,
                "visualization_max_faces_per_object": self.config.visualization_max_faces_per_object,
                "object_color_palette": [list(c) for c in self.config.object_color_palette],
            },
            "assembled_objects": [{
                "placement_id": rec["placement_id"],
                "class_name": rec["class_name"],
                "source_obj_path": rec["source_obj_path"],
                "object_name": rec["object_name"],
                "material_name": rec["material_name"],
                "material_rgb": rec["material_rgb"],
                "num_vertices": int(rec["vertices_world"].shape[0]),
                "num_faces": int(len(rec["faces_v"])),
                "normalization_info": rec["normalization_info"],
                "applied_scale_xyz": rec["applied_scale_xyz"],
                "applied_rotation_euler_xyz_deg": rec["applied_rotation_euler_xyz_deg"],
                "applied_translation_xyz": rec["applied_translation_xyz"],
            } for rec in transformed_meshes],
            "skipped": skipped,
        })
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": len(transformed_meshes), "skipped": len(skipped)})
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

