from __future__ import annotations

from pathlib import Path

from src.config import PROMPTING_OUTPUT_DIR
from src.external.recognizeanything import run_recognizeanything
from src.pipeline_types import StageResult


def run_prompting(image_path: Path) -> StageResult:
    run_recognizeanything(image_path)
    print("Recognize-Anything finished successfully.")
    return StageResult(
        stage="prompting",
        outputs=[PROMPTING_OUTPUT_DIR / "text_prompt.txt"],
    )
