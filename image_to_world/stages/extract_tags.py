from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from image_to_world.adapters.defaults import RamTagger
from image_to_world.cache import CacheStore
from image_to_world.common import require_file, save_json
from image_to_world.config import RuntimeConfig, TagExtractionConfig
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult
from image_to_world.stages.base import Stage


class ExtractTagsStage(Stage):
    stage_name = "extract_tags"

    def __init__(self, *, config: TagExtractionConfig, runtime: RuntimeConfig, manifest: ManifestStore, cache: CacheStore) -> None:
        super().__init__(manifest=manifest, cache=cache)
        self.config = config
        self.runtime = runtime
        self.adapter = RamTagger(config, runtime.device)

    def load_image(self, image_path: Path) -> Image.Image:
        require_file(image_path, "Image file")
        return Image.open(image_path).convert("RGB")

    def process_image(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.runtime.device)

    def filter_tags(self, tags: list[str]) -> list[str]:
        filtered: list[str] = []
        seen = set()
        for tag in tags:
            normalized = tag.strip().lower()
            if not normalized or normalized in self.config.background_like_tags or normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(normalized)
        return filtered

    def run(self) -> StageResult:
        if self.should_skip(self.config.output_path):
            return self.finalize(StageResult(stage_name=self.stage_name, output_path=self.config.output_path, skipped=True), self.config_to_cache_payload(self.config))
        image = self.load_image(self.config.image_path)
        tensor = self.process_image(image)
        raw_tags = self.adapter.predict_tags(tensor)
        tags = self.filter_tags(raw_tags)
        save_json(self.config.output_path, tags)
        result = StageResult(stage_name=self.stage_name, output_path=self.config.output_path, metadata={"tags": tags})
        return self.finalize(result, self.config_to_cache_payload(self.config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the extract_tags stage.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = RuntimeConfig(device=args.device or RuntimeConfig().device, skip_existing=args.skip_existing, overwrite=args.overwrite)
    stage = ExtractTagsStage(config=TagExtractionConfig(), runtime=runtime, manifest=ManifestStore(), cache=CacheStore())
    stage.run()


if __name__ == "__main__":
    main()
