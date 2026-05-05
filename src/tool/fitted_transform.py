from __future__ import annotations

from pathlib import Path
import json
import subprocess
import shutil


def _windows_to_wsl_path(path: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    drive = p.drive.rstrip(":").lower()
    posix = p.as_posix()
    tail = posix.split(":", 1)[1] if ":" in posix else posix
    return f"/mnt/{drive}{tail}"


def make_fitted_transform(image_path: Path):
    image_path = Path(image_path).resolve()
    project_root = Path(__file__).resolve().parent.parent.parent
    output_root = project_root / "output" / "fitted_transform_debug" / image_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    wsl_python = project_root / "third_party" / "pytorch3d" / ".venv" / "bin" / "python"
    wsl_script = project_root / "src" / "tool" / "fitted_transform_wsl.py"
    if not wsl_script.exists():
        raise FileNotFoundError(f"WSL render script not found: {wsl_script}")

    wsl_project_root = _windows_to_wsl_path(project_root)
    wsl_image_path = _windows_to_wsl_path(image_path)
    wsl_python_path = _windows_to_wsl_path(wsl_python)
    wsl_script_path = _windows_to_wsl_path(wsl_script)

    run_name = "simple"
    cmd = [
        "wsl",
        "-e",
        "bash",
        "-lc",
        (
            f"cd '{wsl_project_root}' && "
            f"'{wsl_python_path}' '{wsl_script_path}' "
            f"--project-root '{wsl_project_root}' "
            f"--image-path '{wsl_image_path}' "
            f"--device cuda --use-fitted-init --run-name '{run_name}' "
            "--steps 400 --image-size 256 --save-every 50 "
            "--faces-pp 24 --final-faces-pp 64 "
            "--prefer-fit-mesh --auto-create-fit-mesh --fit-face-count 5000"
        ),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        raise RuntimeError(f"WSL differentiable rendering failed (code={result.returncode}): {stderr[-2000:]}")

    run_dir = output_root / run_name
    trial_meta_path = run_dir / "render_meta.json"
    if trial_meta_path.exists():
        with open(trial_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = json.loads(stdout.splitlines()[-1]) if stdout else {"stdout": stdout, "stderr": stderr}

    for name in ["render_ref.png", "render_ref.npy", "target_ref.png", "fitted_transform.npy", "render_meta.json"]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, output_root / name)

    return meta
