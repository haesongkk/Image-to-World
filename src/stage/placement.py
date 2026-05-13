
from pathlib import Path

from src.external.mldepthpro import run_mldepthpro
from src.external.perspectivefields import run_perspectivefields
from src.config import OUTPUT_DIR
from src.pipeline_types import StageResult

from src.tool.pointcloud import make_pointcloud
from src.tool.raw_transform import make_raw_transform
from src.tool.fitted_transform import make_fitted_transform
from src.tool.scene_glb import make_scene_glb


def run_placement(image_path: Path) -> StageResult:
    outputs = []

    run_mldepthpro(image_path)
    print("ML Depth Pro finished successfully.")
    outputs.append(OUTPUT_DIR / "ml-depth-pro" / f"{image_path.stem}.npz")

    run_perspectivefields(image_path)
    print("PerspectiveFields finished successfully.")
    outputs.append(OUTPUT_DIR / "PerspectiveFields" / f"{image_path.stem}_perspective_fields.json")

    make_pointcloud(image_path)
    print("Point cloud generation finished successfully.")
    outputs.append(OUTPUT_DIR / "pointcloud" / "pointcloud_4views.png")

    make_raw_transform(image_path)
    print("Raw transform preparation finished successfully.")
    outputs.append(OUTPUT_DIR / "raw_transform" / "raw_transform.json")

    # make_fitted_transform(image_path)
    # print("Fitted transform preparation finished successfully.")

    make_scene_glb(image_path)
    print("Scene GLB assembly finished successfully.")
    outputs.append(OUTPUT_DIR / "scene" / f"{image_path.stem}_assembled.glb")
    return StageResult(stage="placement", outputs=outputs)
