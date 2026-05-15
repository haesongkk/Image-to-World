from __future__ import annotations

from pathlib import Path

from src.config import THIRD_PARTY_DIR


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Preflight failed: {label} not found: {path}")


def run_preflight(stage: str, input_image: Path) -> None:
    require_exists(input_image, "input image")

    segmentation_stages = ("prompting", "instance_segmentation", "mask_postprocess")
    generation_stages = ("mesh_generation", "mesh_remesh", "mesh_texturing")
    placement_stages = ("depth_estimation", "camera_estimation", "scene_precompute", "scene_assembly")

    if stage == "all" or stage in segmentation_stages:
        require_exists(THIRD_PARTY_DIR / "recognize-anything" / ".venv" / "Scripts" / "python.exe", "recognize-anything python")
        require_exists(THIRD_PARTY_DIR / "Grounded-SAM-2" / ".venv" / "Scripts" / "python.exe", "Grounded-SAM-2 python")

    if stage == "all" or stage in generation_stages:
        require_exists(THIRD_PARTY_DIR / "Hunyuan3D-2" / ".venv" / "Scripts" / "python.exe", "Hunyuan3D-2 python")

    if stage == "all" or stage in placement_stages:
        require_exists(THIRD_PARTY_DIR / "ml-depth-pro" / ".venv" / "Scripts" / "depth-pro-run.exe", "ml-depth-pro executable")
        require_exists(THIRD_PARTY_DIR / "PerspectiveFields" / ".venv" / "Scripts" / "python.exe", "PerspectiveFields python")
