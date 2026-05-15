from __future__ import annotations

from pathlib import Path

from src.config import MASK_POSTPROCESS_OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.mask import make_mask


def run_mask_postprocess(image_path: Path) -> StageResult:
    make_mask(image_path)
    print("Mask generation finished successfully.")
    return StageResult(
        stage="mask_postprocess",
        outputs=[MASK_POSTPROCESS_OUTPUT_DIR / "mask_viz.png"],
    )
