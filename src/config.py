from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

PROMPTING_OUTPUT_DIR = OUTPUT_DIR / "prompting"
INSTANCE_SEGMENTATION_OUTPUT_DIR = OUTPUT_DIR / "instance_segmentation"
CROPS_GENERATION_OUTPUT_DIR = OUTPUT_DIR / "crops_generation"
MASK_POSTPROCESS_OUTPUT_DIR = OUTPUT_DIR / "mask_postprocess"
AMODAL_COMPLETION_OUTPUT_DIR = OUTPUT_DIR / "amodal_completion"
MESH_GENERATION_OUTPUT_DIR = OUTPUT_DIR / "mesh_generation"
MESH_REMESH_OUTPUT_DIR = OUTPUT_DIR / "mesh_remesh"
MESH_TEXTURING_OUTPUT_DIR = OUTPUT_DIR / "mesh_texturing"
DEPTH_ESTIMATION_OUTPUT_DIR = OUTPUT_DIR / "depth_estimation"
CAMERA_ESTIMATION_OUTPUT_DIR = OUTPUT_DIR / "camera_estimation"
SCENE_PRECOMPUTE_OUTPUT_DIR = OUTPUT_DIR / "scene_precompute"
FITTED_TRANSFORM_OUTPUT_DIR = OUTPUT_DIR / "fitted_transform"
SCENE_ASSEMBLY_OUTPUT_DIR = OUTPUT_DIR / "scene_assembly"
BACKGROUND_INPAINT_OUTPUT_DIR = OUTPUT_DIR / "background_inpaint"

RAW_IMAGE_NAME = "raw_image.jpg"


def raw_image_path(image_path: str | Path | None = None) -> Path:
    if image_path is None:
        return DATA_DIR / RAW_IMAGE_NAME
    return Path(image_path).expanduser().resolve()
