from __future__ import annotations

import argparse
from pathlib import Path

from image_to_world.adapters.hunyuan_external import HunyuanExternalGenerator
from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import MeshGenerationConfig, RuntimeConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage


class GenerateMeshesStage(Stage):
    stage_name = "generate_meshes"

    def __init__(self, *, config: MeshGenerationConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime
        self.adapter = HunyuanExternalGenerator(config)

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "gen3d_result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        data = load_json(self.config.input_json_path)
        annotations = data.get("annotations", [])
        if not annotations:
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": 0}, skipped=True), self.config_to_cache_payload(self.config))

        all_results = []
        for ann in annotations:
            obj_id = ann["id"]
            image_path = Path(ann["amodal_rgb_path"])
            if not image_path.exists():
                continue
            obj_dir = self.config.output_dir / f"object_{obj_id:02d}"
            obj_dir.mkdir(parents=True, exist_ok=True)
            shape_output_path = obj_dir / f"object_{obj_id:02d}_shape.{self.config.hunyuan_output_format.lstrip('.')}"
            texture_output_path = obj_dir / f"object_{obj_id:02d}_textured.{self.config.hunyuan_texture_output_format.lstrip('.')}"
            mesh_output = self.adapter.generate_mesh(
                image_path=image_path,
                shape_output_path=shape_output_path,
                texture_output_path=texture_output_path,
            )
            meta = {
                "id": obj_id,
                "class_name": ann["class_name"],
                "input_image_path": str(image_path),
                "mesh_path": mesh_output["mesh_path"],
                "mesh_format": mesh_output["mesh_format"],
                "shape_mesh_path": mesh_output.get("shape_mesh_path"),
                "shape_mesh_format": mesh_output.get("shape_mesh_format"),
                "textured_mesh_path": mesh_output.get("textured_mesh_path"),
                "textured_mesh_format": mesh_output.get("textured_mesh_format"),
                "device": self.runtime.device,
                "model_backend": "hunyuan3d_2",
                "model_id": self.config.hunyuan_model_id,
                "texture_enabled": self.config.hunyuan_enable_texture,
                "texture_model_id": self.config.hunyuan_texture_model_id if self.config.hunyuan_enable_texture else None,
                "hunyuan_repo_dir": str(self.config.hunyuan_repo_dir),
                "hunyuan_venv_python": str(self.config.hunyuan_venv_python),
                "background_removal": self.config.hunyuan_use_background_removal,
            }
            meta_path = obj_dir / "meta.json"
            save_json(meta_path, meta)
            all_results.append(meta)

        save_json(output_path, {
            "input_json_path": str(self.config.input_json_path),
            "device": self.runtime.device,
            "num_objects": len(all_results),
            "results": all_results,
        })
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": len(all_results)})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the generate_meshes stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-texture", action="store_true")
    parser.add_argument("--texture-model-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    config = MeshGenerationConfig()
    if args.enable_texture:
        config.hunyuan_enable_texture = True
    if args.texture_model_id:
        config.hunyuan_texture_model_id = args.texture_model_id
    stage = GenerateMeshesStage(config=config, runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
