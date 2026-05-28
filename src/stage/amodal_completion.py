from __future__ import annotations

from pathlib import Path

from src.config import AMODAL_COMPLETION_OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.amodal_mask import make_amodal_masks


def run_amodal_completion(image_path: Path) -> StageResult:
    make_amodal_masks()
    print("Amodal completion finished successfully.")
    return StageResult(
        stage="amodal_completion",
        outputs=[AMODAL_COMPLETION_OUTPUT_DIR / "amodal_viz.png"],
    )
