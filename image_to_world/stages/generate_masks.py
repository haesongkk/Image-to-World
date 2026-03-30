from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_to_world.adapters.defaults import GroundedSamAdapter
from image_to_world.cache import CacheStore
from image_to_world.common import load_json, require_file, save_json
from image_to_world.config import MaskGenerationConfig, RuntimeConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import Annotation, StageResult
from image_to_world.stages.base import Stage


class GenerateMasksStage(Stage):
    stage_name = "generate_masks"

    def __init__(self, *, config: MaskGenerationConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime
        self.adapter = GroundedSamAdapter(config, runtime.device)

    @staticmethod
    def random_color(idx: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(seed=idx)
        return tuple(int(x) for x in rng.integers(50, 255, size=3))

    @staticmethod
    def clamp_box_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return x1, y1, x2, y2

    def save_object_crops(self, image_bgr: np.ndarray, mask: np.ndarray, box: np.ndarray, obj_id: int) -> dict[str, object]:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = self.clamp_box_xyxy(box, w, h)
        crop_bgr = image_bgr[y1:y2, x1:x2].copy()
        crop_rgb_path = self.config.output_dir / f"crop_rgb_{obj_id:02d}.png"
        crop_rgba_path = self.config.output_dir / f"crop_rgba_{obj_id:02d}.png"
        cv2.imwrite(str(crop_rgb_path), crop_bgr)
        crop_mask = mask[y1:y2, x1:x2].copy().astype(np.uint8)
        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        crop_bgra[:, :, 3] = crop_mask * 255
        cv2.imwrite(str(crop_rgba_path), crop_bgra)
        return {
            "crop_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "crop_rgb_path": str(crop_rgb_path),
            "crop_rgba_path": str(crop_rgba_path),
        }

    def run(self) -> StageResult:
        output_path = self.config.output_dir / "result.json"
        if self.should_skip(output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=output_path, skipped=True), self.config_to_cache_payload(self.config))
        self.ensure_output_dir(self.config.output_dir)
        require_file(self.config.image_path, "Image file")
        require_file(self.config.prompt_path, "Prompt JSON")
        require_file(self.config.sam2_config, "SAM2 config")
        require_file(self.config.sam2_checkpoint, "SAM2 checkpoint")

        tags = load_json(self.config.prompt_path)
        text_prompt = " . ".join(tags)
        image_pil = Image.open(self.config.image_path).convert("RGB")
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detections = self.adapter.detect(image_pil, text_prompt)
        boxes = detections["boxes"].cpu().numpy() if len(detections["boxes"]) > 0 else np.empty((0, 4))
        labels = [str(x) for x in detections["labels"]]
        scores = detections["scores"].cpu().numpy() if len(detections["scores"]) > 0 else np.array([])
        masks = self.adapter.segment(image_np, boxes) if len(boxes) > 0 else []

        overlay = image_bgr.copy()
        annotations: list[dict[str, object]] = []
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            color = self.random_color(i)
            mask_path = self.config.output_dir / f"mask_{i:02d}.png"
            cv2.imwrite(str(mask_path), mask * 255)
            crop_info = self.save_object_crops(image_bgr, mask, box, i)
            colored = np.zeros_like(overlay, dtype=np.uint8)
            colored[:, :] = color
            overlay = np.where(mask[:, :, None] == 1, overlay * 0.45 + colored * 0.55, overlay).astype(np.uint8)
            x1, y1, x2, y2 = self.clamp_box_xyxy(box, image_bgr.shape[1], image_bgr.shape[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{labels[i]} {float(scores[i]):.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            annotations.append(Annotation(
                id=i,
                class_name=labels[i],
                score=float(scores[i]),
                bbox_xyxy=[float(v) for v in box.tolist()],
                mask_path=str(mask_path),
                crop_bbox_xyxy=crop_info["crop_bbox_xyxy"],
                crop_rgb_path=crop_info["crop_rgb_path"],
                crop_rgba_path=crop_info["crop_rgba_path"],
            ).to_dict())

        overlay_path = self.config.output_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        save_json(output_path, {
            "image_path": str(self.config.image_path),
            "prompt_path": str(self.config.prompt_path),
            "text_prompt": text_prompt,
            "annotations": annotations,
            "overlay_path": str(overlay_path),
        })
        result = StageResult(stage_name=self.stage_name, output_path=output_path, metadata={"count": len(annotations), "overlay_path": str(overlay_path)})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the generate_masks stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = GenerateMasksStage(config=MaskGenerationConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
