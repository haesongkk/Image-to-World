import os
from pathlib import Path

from src.config import OUTPUT_DIR, THIRD_PARTY_DIR
from src.external.runner import run_external_command


def run_perspectivefields(image_path: Path):
    repo_root = THIRD_PARTY_DIR / "PerspectiveFields"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script =  Path(__file__).resolve().parent / "inference_script" / "inference_perspectivefields.py"

    output_dir = OUTPUT_DIR / "PerspectiveFields"
    os.makedirs(output_dir, exist_ok=True)

    run_external_command(
        name="perspectivefields",
        command=[
            str(venv_python),
            str(inference_script),
            "--image-path",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        log_dir=output_dir,
    )
