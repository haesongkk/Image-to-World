from __future__ import annotations

import argparse

from image_to_world.cache import CacheStore
from image_to_world.config import PipelineConfig
from image_to_world.logging_utils import configure_logging, get_logger
from image_to_world.manifest import ManifestStore
from image_to_world.stages.assemble_scene import AssembleSceneStage
from image_to_world.stages.complete_objects import CompleteObjectsStage
from image_to_world.stages.compose_layout import ComposeLayoutStage
from image_to_world.stages.estimate_depth import EstimateDepthStage
from image_to_world.stages.extract_tags import ExtractTagsStage
from image_to_world.stages.generate_masks import GenerateMasksStage
from image_to_world.stages.generate_meshes import GenerateMeshesStage

STAGE_ORDER = [
    "extract_tags",
    "generate_masks",
    "complete_objects",
    "generate_meshes",
    "estimate_depth",
    "compose_layout",
    "assemble_scene",
]


def build_stage_map(config: PipelineConfig, manifest: ManifestStore, cache: CacheStore):
    runtime = config.runtime
    return {
        "extract_tags": ExtractTagsStage(config=config.extract_tags, runtime=runtime, manifest=manifest, cache=cache),
        "generate_masks": GenerateMasksStage(config=config.generate_masks, runtime=runtime, manifest=manifest, cache=cache),
        "complete_objects": CompleteObjectsStage(config=config.complete_objects, runtime=runtime, manifest=manifest, cache=cache),
        "generate_meshes": GenerateMeshesStage(config=config.generate_meshes, runtime=runtime, manifest=manifest, cache=cache),
        "estimate_depth": EstimateDepthStage(config=config.estimate_depth, runtime=runtime, manifest=manifest, cache=cache),
        "compose_layout": ComposeLayoutStage(config=config.compose_layout, runtime=runtime, manifest=manifest, cache=cache),
        "assemble_scene": AssembleSceneStage(config=config.assemble_scene, runtime=runtime, manifest=manifest, cache=cache),
    }


def select_stages(stage_from: str | None, stage_to: str | None) -> list[str]:
    start_idx = STAGE_ORDER.index(stage_from) if stage_from else 0
    end_idx = STAGE_ORDER.index(stage_to) if stage_to else len(STAGE_ORDER) - 1
    return STAGE_ORDER[start_idx:end_idx + 1]


def run_pipeline(config: PipelineConfig, *, stage_from: str | None = None, stage_to: str | None = None) -> list[dict]:
    configure_logging()
    logger = get_logger("pipeline")
    manifest = ManifestStore()
    cache = CacheStore()
    stages = build_stage_map(config, manifest, cache)
    selected = select_stages(stage_from, stage_to)
    logger.info("Running stages: %s", ", ".join(selected))
    results = []
    for stage_name in selected:
        logger.info("Starting stage: %s", stage_name)
        result = stages[stage_name].run()
        results.append(result.to_dict())
        logger.info("Finished stage: %s -> %s", stage_name, result.output_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Image-to-World pipeline.")
    parser.add_argument("--from", dest="stage_from", choices=STAGE_ORDER, default=None)
    parser.add_argument("--to", dest="stage_to", choices=STAGE_ORDER, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig()
    if args.device:
        config.runtime.device = args.device
    config.runtime.skip_existing = args.skip_existing
    config.runtime.overwrite = args.overwrite
    run_pipeline(config, stage_from=args.stage_from, stage_to=args.stage_to)


if __name__ == "__main__":
    main()
