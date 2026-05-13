from __future__ import annotations

from pathlib import Path

from src.config import THIRD_PARTY_DIR


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Preflight failed: {label} not found: {path}")


def run_preflight(stage: str, input_image: Path) -> None:
    require_exists(input_image, "input image")

    if stage in ("all", "segmentation"):
        require_exists(THIRD_PARTY_DIR / "recognize-anything" / ".venv" / "Scripts" / "python.exe", "recognize-anything python")
        require_exists(THIRD_PARTY_DIR / "Grounded-SAM-2" / ".venv" / "Scripts" / "python.exe", "Grounded-SAM-2 python")

    if stage in ("all", "generation"):
        require_exists(THIRD_PARTY_DIR / "Hunyuan3D-2" / ".venv" / "Scripts" / "python.exe", "Hunyuan3D-2 python")

    if stage in ("all", "placement"):
        require_exists(THIRD_PARTY_DIR / "ml-depth-pro" / ".venv" / "Scripts" / "depth-pro-run.exe", "ml-depth-pro executable")
        require_exists(THIRD_PARTY_DIR / "PerspectiveFields" / ".venv" / "Scripts" / "python.exe", "PerspectiveFields python")
