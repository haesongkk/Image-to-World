import os
import subprocess
from pathlib import Path

def run_mldepthpro(image_path: Path):
    project_root = Path(__file__).resolve().parent.parent.parent

    repo_root = project_root / "third_party" / "ml-depth-pro"
    cmd = repo_root / ".venv" / "Scripts" / "depth-pro-run.exe"

    output_dir  = project_root / "output" / "ml-depth-pro" 
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        [
            str(cmd),
            "-i", str(image_path),
            "-o", str(output_dir),
            "--skip-display",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo_root),
    )

    if(result.returncode != 0):
        raise RuntimeError("ML Depth Pro inference failed..\n" + result.stderr)
    
