
from pathlib import Path

from src.external.mldepthpro import run_mldepthpro
from src.external.perspectivefields import run_perspectivefields

from src.tool.pointcloud import make_pointcloud
from src.tool.mask import make_mask

def run_placement():
    project_root = Path(__file__).resolve().parent.parent.parent
    image_path = project_root / "data" / "raw_image.jpg"

    run_mldepthpro(image_path)
    print("ML Depth Pro finished successfully.")

    run_perspectivefields(image_path)
    print("PerspectiveFields finished successfully.")

    make_pointcloud(image_path)
    print("Point cloud generation finished successfully.")

