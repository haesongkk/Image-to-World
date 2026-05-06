
from pathlib import Path

from src.external.hunyuan3d2 import run_hunyuan3d2
from src.tool.bake import make_bake_maps
from src.tool.remesh import make_remesh_glb


def run_generation():
    project_root = Path(__file__).resolve().parent.parent.parent
    crop_image_dir = project_root / "output" / "Grounded-SAM-2" / "crops"
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

    for crop_image_path in crop_images:
        run_hunyuan3d2(crop_image_path)
        print(f"Hunyuan3D-2 finished: {crop_image_path.name}")

        make_remesh_glb(crop_image_path.stem)
        print(f"Remeshing finished: {crop_image_path.name}")

        # make_bake_maps(crop_image_path.stem)
        # print(f"Baking finished: {crop_image_path.name}")
