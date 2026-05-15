from __future__ import annotations

from pathlib import Path

from src.config import CAMERA_ESTIMATION_OUTPUT_DIR
from src.external.perspectivefields import run_perspectivefields
from src.pipeline_types import StageResult


def run_camera_estimation(image_path: Path) -> StageResult:
    run_perspectivefields(image_path)
    print("PerspectiveFields finished successfully.")
    return StageResult(
        stage="camera_estimation",
        outputs=[CAMERA_ESTIMATION_OUTPUT_DIR / f"{image_path.stem}_perspective_fields.json"],
    )
