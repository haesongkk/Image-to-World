from __future__ import annotations

import os
from pathlib import Path

from src.config import OUTPUT_DIR, THIRD_PARTY_DIR
from src.external.runner import run_external_command


def _latest_snapshot_dir(hub_model_dir: Path) -> Path | None:
    snapshots = hub_model_dir / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        return None
    return dirs[-1]


def _pick_paint_model() -> tuple[str, str]:
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    hub_21 = hf_hub / "models--tencent--Hunyuan3D-2.1"
    hub_20 = hf_hub / "models--tencent--Hunyuan3D-2"

    snap_21 = _latest_snapshot_dir(hub_21)
    if snap_21 and (snap_21 / "hunyuan3d-paintpbr-v2-1").exists() and (snap_21 / "hunyuan3d-delight-v2-0").exists():
        return str(snap_21), "hunyuan3d-paintpbr-v2-1"

    snap_20 = _latest_snapshot_dir(hub_20)
    if snap_20 and (snap_20 / "hunyuan3d-paint-v2-0-turbo").exists() and (snap_20 / "hunyuan3d-delight-v2-0").exists():
        return str(snap_20), "hunyuan3d-paint-v2-0-turbo"

    # Fallback to repo ids; may require network if local cache is missing.
    if hub_21.exists():
        return "tencent/Hunyuan3D-2.1", "hunyuan3d-paintpbr-v2-1"
    return "tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo"


def run_hunyuan3d_paint(image_path: Path, mesh_path: Path, output_path: Path):
    repo_root = THIRD_PARTY_DIR / "Hunyuan3D-2"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script = Path(__file__).resolve().parent / "inference_script" / "inference_hunyuan3d_paint.py"
    paint_model, paint_subfolder = _pick_paint_model()

    output_dir = OUTPUT_DIR / "Hunyuan3D-2"
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    local_hf_home = output_dir / ".hf_home"
    local_hf_modules = output_dir / ".hf_modules"
    local_hf_home.mkdir(parents=True, exist_ok=True)
    local_hf_modules.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(local_hf_home)
    env["HF_MODULES_CACHE"] = str(local_hf_modules)

    run_external_command(
        name=f"hunyuan3d_paint_{image_path.stem}",
        command=[
            str(venv_python),
            str(inference_script),
            "--image-path",
            str(image_path),
            "--mesh-path",
            str(mesh_path),
            "--output-path",
            str(output_path),
            "--paint-model",
            paint_model,
            "--paint-subfolder",
            paint_subfolder,
        ],
        cwd=repo_root,
        log_dir=output_dir,
        env=env,
    )
