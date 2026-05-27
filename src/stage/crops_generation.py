from __future__ import annotations

from src.config import CROPS_GENERATION_OUTPUT_DIR
from src.external.groundedsam2 import run_groundedsam2_crop_generation
from src.pipeline_types import StageResult


def run_crops_generation() -> StageResult:
    run_groundedsam2_crop_generation()
    print("Crop generation finished successfully.")
    return StageResult(
        stage="crops_generation",
        outputs=[CROPS_GENERATION_OUTPUT_DIR / "crops"],
    )
