from __future__ import annotations

from src.config import INSTANCE_SEGMENTATION_OUTPUT_DIR, MESH_GENERATION_OUTPUT_DIR
from src.external.hunyuan3d2 import run_hunyuan3d2
from src.pipeline_types import StageResult


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


def run_mesh_generation() -> StageResult:
    outputs = []
    for crop_image_path in _list_crop_images():
        run_hunyuan3d2(crop_image_path)
        print(f"Hunyuan3D-2 finished: {crop_image_path.name}")
        outputs.append(MESH_GENERATION_OUTPUT_DIR / f"{crop_image_path.stem}_shape_mesh.glb")
    return StageResult(stage="mesh_generation", outputs=outputs)
