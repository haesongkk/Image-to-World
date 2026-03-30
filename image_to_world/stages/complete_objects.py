from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_world.adapters.defaults import SdxlInpainter
from image_to_world.cache import CacheStore
from image_to_world.common import load_json, save_json
from image_to_world.config import ObjectCompletionConfig, RuntimeConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage


class CompleteObjectsStage(Stage):
    stage_name = "complete_objects"

    def __init__(self, *, config: ObjectCompletionConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime
        self.adapter = SdxlInpainter(config, runtime.device)

    @staticmethod
    def build_inpaint_inputs(rgba_img: Image.Image, canvas_scale: float) -> tuple[Image.Image, Image.Image]:
        w, h = rgba_img.size
        canvas_w = max(64, int(w * canvas_scale))
        canvas_h = max(64, int(h * canvas_scale))
        rgba_canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
        rgba_canvas.paste(rgba_img, ((canvas_w - w) // 2, (canvas_h - h) // 2), rgba_img)
        rgba_np = np.array(rgba_canvas)
        rgb_np = rgba_np[:, :, :3].copy()
        alpha_np = rgba_np[:, :, 3].copy()
        rgb_np[alpha_np == 0] = 255
        mask_np = np.where(alpha_np > 0, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb_np).convert("RGB"), Image.fromarray(mask_np).convert("L")

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "amodal_result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        data = load_json(self.config.input_json_path)
        annotations = data.get("annotations", [])
        if not annotations:
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": 0}, skipped=True), self.config_to_cache_payload(self.config))

        out_annotations = []
        for ann in annotations:
            obj_id = ann["id"]
            crop_rgba_path = Path(ann["crop_rgba_path"])
            if not crop_rgba_path.exists():
                continue
            prompt = f"a complete {ann['class_name'].strip().lower()}"
            rgba_img = Image.open(crop_rgba_path).convert("RGBA")
            input_image, mask_image = self.build_inpaint_inputs(rgba_img, self.config.canvas_scale)
            input_path = self.config.output_dir / f"inpaint_input_{obj_id:02d}.png"
            mask_path = self.config.output_dir / f"inpaint_mask_{obj_id:02d}.png"
            input_image.save(input_path)
            mask_image.save(mask_path)
            result_image = self.adapter.inpaint(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.strength,
            )
            output_image_path = self.config.output_dir / f"amodal_rgb_{obj_id:02d}.png"
            result_image.save(output_image_path)
            out_annotations.append({
                "id": obj_id,
                "class_name": ann["class_name"],
                "prompt": prompt,
                "crop_rgba_path": str(crop_rgba_path),
                "inpaint_input_path": str(input_path),
                "inpaint_mask_path": str(mask_path),
                "amodal_rgb_path": str(output_image_path),
            })

        save_json(output_path, {
            "input_json_path": str(self.config.input_json_path),
            "model_id": self.config.model_id,
            "device": self.runtime.device,
            "canvas_scale": self.config.canvas_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "strength": self.config.strength,
            "annotations": out_annotations,
        })
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": len(out_annotations)})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete_objects stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = CompleteObjectsStage(config=ObjectCompletionConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
