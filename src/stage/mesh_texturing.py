from __future__ import annotations

from src.config import CROPS_GENERATION_OUTPUT_DIR, MESH_REMESH_OUTPUT_DIR, MESH_TEXTURING_OUTPUT_DIR
from src.external.hunyuan3d_paint import run_hunyuan3d_paint
from src.pipeline_types import StageResult


def _list_crop_images():
    crop_image_dir = CROPS_GENERATION_OUTPUT_DIR / "crops"
    if not crop_image_dir.exists():
        raise RuntimeError(f"Crop image directory not found: {crop_image_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    crop_images = sorted(
        [p for p in crop_image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    )
    if not crop_images:
        raise RuntimeError(f"No crop images found in: {crop_image_dir}")
    return crop_images


def run_mesh_texturing() -> StageResult:
    outputs = []
    for crop_image_path in _list_crop_images():
        remeshed_path = MESH_REMESH_OUTPUT_DIR / f"{crop_image_path.stem}_remeshed.glb"
        textured_path = MESH_TEXTURING_OUTPUT_DIR / f"{crop_image_path.stem}_remeshed_textured.glb"
        run_hunyuan3d_paint(crop_image_path, remeshed_path, textured_path)
        print(f"Texturing finished: {crop_image_path.name}")
        outputs.append(textured_path)
    return StageResult(stage="mesh_texturing", outputs=outputs)
