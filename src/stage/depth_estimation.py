from __future__ import annotations

from pathlib import Path

from src.config import DEPTH_ESTIMATION_OUTPUT_DIR
from src.external.mldepthpro import run_mldepthpro
from src.pipeline_types import StageResult


def run_depth_estimation(image_path: Path) -> StageResult:
    run_mldepthpro(image_path)
    print("ML Depth Pro finished successfully.")
    return StageResult(
        stage="depth_estimation",
        outputs=[DEPTH_ESTIMATION_OUTPUT_DIR / f"{image_path.stem}.npz"],
    )
