
from pathlib import Path

from src.external.mldepthpro import run_mldepthpro
from src.external.perspectivefields import run_perspectivefields

from src.tool.pointcloud import make_pointcloud
from src.tool.raw_transform import make_raw_transform
from src.tool.fitted_transform import make_fitted_transform
from src.tool.scene_glb import make_scene_glb


def run_placement():
    project_root = Path(__file__).resolve().parent.parent.parent
    image_path = project_root / "data" / "raw_image.jpg"

    # run_mldepthpro(image_path)
    # print("ML Depth Pro finished successfully.")

    # run_perspectivefields(image_path)
    # print("PerspectiveFields finished successfully.")

    # make_pointcloud(image_path)
    # print("Point cloud generation finished successfully.")

    # make_raw_transform(image_path)
    # print("Raw transform preparation finished successfully.")

    # make_fitted_transform(image_path)
    # print("Fitted transform preparation finished successfully.")

    make_scene_glb(image_path)
    print("Scene GLB assembly finished successfully.")
