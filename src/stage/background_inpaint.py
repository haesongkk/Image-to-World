from __future__ import annotations

from pathlib import Path

from src.config import BACKGROUND_INPAINT_OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.background_inpaint import make_clean_background


def run_background_inpaint(image_path: Path) -> StageResult:
    out = make_clean_background()
    print("Background inpaint finished successfully.")
    return StageResult(
        stage="background_inpaint",
        outputs=[BACKGROUND_INPAINT_OUTPUT_DIR / "clean_background.png", out],
    )
