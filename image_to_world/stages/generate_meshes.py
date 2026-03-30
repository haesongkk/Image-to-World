from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from image_to_world.adapters.defaults import ShapEGenerator
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
        self.adapter = ShapEGenerator(config, runtime.device)

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
            mesh = self.adapter.generate_mesh(Image.open(image_path).convert("RGB"))
            obj_path = obj_dir / f"object_{obj_id:02d}.obj"
            ply_path = obj_dir / f"object_{obj_id:02d}.ply"
            with obj_path.open("w", encoding="utf-8") as handle:
                mesh.write_obj(handle)
            with ply_path.open("wb") as handle:
                mesh.write_ply(handle)
            meta = {
                "id": obj_id,
                "class_name": ann["class_name"],
                "input_image_path": str(image_path),
                "obj_path": str(obj_path),
                "ply_path": str(ply_path),
                "guidance_scale": self.config.guidance_scale,
                "karras_steps": self.config.karras_steps,
                "device": self.runtime.device,
                "model_name": self.config.image_model_name,
                "transmitter_name": self.config.transmitter_name,
                "asset_pre_rotation_euler_xyz_deg": self.config.asset_pre_rot_euler_xyz_deg,
            }
            save_json(obj_dir / "meta.json", meta)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = GenerateMeshesStage(config=MeshGenerationConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
