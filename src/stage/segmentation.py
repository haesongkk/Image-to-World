from pathlib import Path

from src.config import OUTPUT_DIR
from src.external.birefnet import run_birefnet
from src.external.recognizeanything import run_recognizeanything
from src.external.groundedsam2 import run_groundedsam2
from src.pipeline_types import StageResult

from src.tool.mask import make_mask

def run_segmentation(image_path: Path) -> StageResult:
    outputs: list = []

    run_birefnet(image_path)
    print("BiRefNet finished successfully.")
    outputs.append(OUTPUT_DIR / "BirefNet" / f"{image_path.stem}_birefnet.png")

    run_recognizeanything(image_path)
    print("Recognize-Anything finished successfully.")
    outputs.append(OUTPUT_DIR / "recognize-anything" / "text_prompt.txt")
    
    run_groundedsam2(image_path)
    print("Grounded-SAM-2 finished successfully.")
    outputs.append(OUTPUT_DIR / "Grounded-SAM-2" / "grounded_sam2_hf_model_demo_results.json")

    make_mask(image_path)
    print("Mask generation finished successfully.")
    outputs.append(OUTPUT_DIR / "mask" / "mask_viz.png")
    return StageResult(stage="segmentation", outputs=outputs)
