
from pathlib import Path

from src.external.recognizeanything import run_recognizeanything
from src.external.groundedsam2 import run_groundedsam2

from src.tool.mask import make_mask

def run_segmentation():
    project_root = Path(__file__).resolve().parent.parent.parent
    image_path = project_root / "data" / "raw_image.jpg"

    run_recognizeanything(image_path)
    print("Recognize-Anything finished successfully.")
    
    run_groundedsam2(image_path)
    print("Grounded-SAM-2 finished successfully.")

    make_mask(image_path)
    print("Mask generation finished successfully.")
