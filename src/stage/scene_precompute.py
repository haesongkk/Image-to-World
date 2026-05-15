from __future__ import annotations

from pathlib import Path

from src.config import SCENE_PRECOMPUTE_OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.pointcloud import make_pointcloud
from src.tool.raw_transform import make_raw_transform


def run_scene_precompute(image_path: Path) -> StageResult:
    outputs = []
    make_pointcloud(image_path)
    print("Point cloud generation finished successfully.")
    outputs.append(SCENE_PRECOMPUTE_OUTPUT_DIR / "pointcloud_4views.png")

    make_raw_transform(image_path)
    print("Raw transform preparation finished successfully.")
    outputs.append(SCENE_PRECOMPUTE_OUTPUT_DIR / "raw_transform.json")
    return StageResult(stage="scene_precompute", outputs=outputs)
