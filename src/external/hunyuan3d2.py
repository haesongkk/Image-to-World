import os
import subprocess
from pathlib import Path

def run_hunyuan3d2(image_path: Path):
    project_root = Path(__file__).resolve().parent.parent.parent

    repo_root = project_root / "third_party" / "Hunyuan3D-2"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script =  Path(__file__).resolve().parent / "inference_script" / "inference_hunyuan3d2.py"

    output_dir  = project_root / "output" / "Hunyuan3D-2"
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        [
            str(venv_python),
            str(inference_script),
            "--image-path", str(image_path),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo_root),
    )

    if(result.returncode != 0):
        raise RuntimeError("Hunyuan3D-2 inference failed..\n" + result.stderr)