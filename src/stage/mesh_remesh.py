from __future__ import annotations

from src.config import INSTANCE_SEGMENTATION_OUTPUT_DIR, MESH_REMESH_OUTPUT_DIR
from src.pipeline_types import StageResult
from src.tool.remesh import make_remesh_glb


def _list_crop_images():
    crop_image_dir = INSTANCE_SEGMENTATION_OUTPUT_DIR / "crops"
    if not crop_image_dir.exists():
        raise RuntimeError(f"Crop image directory not found: {crop_image_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    crop_images = sorted(
        [p for p in crop_image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    )
    if not crop_images:
        raise RuntimeError(f"No crop images found in: {crop_image_dir}")
    return crop_images


def run_mesh_remesh() -> StageResult:
    outputs = []
    for crop_image_path in _list_crop_images():
        make_remesh_glb(crop_image_path.stem)
        print(f"Remeshing finished: {crop_image_path.name}")
        outputs.append(MESH_REMESH_OUTPUT_DIR / f"{crop_image_path.stem}_remeshed.glb")
    return StageResult(stage="mesh_remesh", outputs=outputs)
