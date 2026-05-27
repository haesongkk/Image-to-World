from __future__ import annotations

from pathlib import Path

from src.config import OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.fitted_transform import make_fitted_transform


def run_fitted_transform(image_path: Path) -> StageResult:
    make_fitted_transform(image_path)
    print("Fitted transform finished successfully.")
    return StageResult(
        stage="fitted_transform",
        outputs=[OUTPUT_DIR / "fitted_transform" / "fitted_transform.json"],
    )
