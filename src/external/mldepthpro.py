import os
from pathlib import Path

from src.config import DEPTH_ESTIMATION_OUTPUT_DIR, THIRD_PARTY_DIR
from src.external.runner import run_external_command


def run_mldepthpro(image_path: Path):
    repo_root = THIRD_PARTY_DIR / "ml-depth-pro"
    cmd = repo_root / ".venv" / "Scripts" / "depth-pro-run.exe"

    output_dir = DEPTH_ESTIMATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    run_external_command(
        name="mldepthpro",
        command=[
            str(cmd),
            "-i",
            str(image_path),
            "-o",
            str(output_dir),
            "--skip-display",
        ],
        cwd=repo_root,
        log_dir=output_dir,
    )
    
