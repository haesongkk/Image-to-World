
from src.config import OUTPUT_DIR
from src.external.hunyuan3d_paint import run_hunyuan3d_paint
from src.external.hunyuan3d2 import run_hunyuan3d2
from src.pipeline_types import StageResult
from src.tool.remesh import make_remesh_glb


def run_generation() -> StageResult:
    crop_image_dir = OUTPUT_DIR / "Grounded-SAM-2" / "crops"
    if not crop_image_dir.exists():
        raise RuntimeError(f"Crop image directory not found: {crop_image_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    crop_images = sorted(
        [
            p for p in crop_image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts
        ]
    )
    if not crop_images:
        raise RuntimeError(f"No crop images found in: {crop_image_dir}")

    outputs = []
    for crop_image_path in crop_images:
        run_hunyuan3d2(crop_image_path)
        print(f"Hunyuan3D-2 finished: {crop_image_path.name}")
        outputs.append(OUTPUT_DIR / "Hunyuan3D-2" / f"{crop_image_path.stem}_shape_mesh.glb")

        make_remesh_glb(crop_image_path.stem)
        print(f"Remeshing finished: {crop_image_path.name}")
        outputs.append(OUTPUT_DIR / "remesh" / f"{crop_image_path.stem}_remeshed.glb")

        remeshed_path = OUTPUT_DIR / "remesh" / f"{crop_image_path.stem}_remeshed.glb"
        textured_path = OUTPUT_DIR / "remesh" / f"{crop_image_path.stem}_remeshed_textured.glb"
        run_hunyuan3d_paint(crop_image_path, remeshed_path, textured_path)
        print(f"Texturing finished: {crop_image_path.name}")
        outputs.append(textured_path)
    return StageResult(stage="generation", outputs=outputs)
