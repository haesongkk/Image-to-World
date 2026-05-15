from __future__ import annotations

from pathlib import Path

from src.config import SCENE_ASSEMBLY_OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.scene_glb import make_scene_glb


def run_scene_assembly(image_path: Path) -> StageResult:
    make_scene_glb(image_path)
    print("Scene GLB assembly finished successfully.")
    return StageResult(
        stage="scene_assembly",
        outputs=[SCENE_ASSEMBLY_OUTPUT_DIR / f"{image_path.stem}_assembled.glb"],
    )
