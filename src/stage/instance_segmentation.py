from __future__ import annotations

from pathlib import Path

from src.config import INSTANCE_SEGMENTATION_OUTPUT_DIR
from src.external.groundedsam2 import run_groundedsam2_inference
from src.pipeline_types import StageResult


def run_instance_segmentation(image_path: Path) -> StageResult:
    run_groundedsam2_inference(image_path)
    print("Grounded-SAM-2 finished successfully.")
    return StageResult(
        stage="instance_segmentation",
        outputs=[INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"],
    )
