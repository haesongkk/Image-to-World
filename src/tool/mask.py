from pathlib import Path
import os
import numpy as np
import json
import re
import cv2
import pycocotools.mask as mask_util

def make_mask(image_path: Path):
    project_root = Path(__file__).resolve().parent.parent.parent

    grounded_sam_2_results_path = project_root / "output" / "Grounded-SAM-2" / "grounded_sam2_hf_model_demo_results.json"
    output_dir = project_root / "output" / "mask"
    os.makedirs(output_dir, exist_ok=True)
    input_image_path = project_root / "output" / "BirefNet" / f"{image_path.stem}_birefnet.png"
    image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)

    with open(grounded_sam_2_results_path, "r", encoding="utf-8") as f:
        infer_results = json.load(f)

    annotations = infer_results.get("annotations", [])
    h, w = image.shape[:2]
    color_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, ann in enumerate(annotations):
        class_name = str(ann.get("class_name"))
        class_name = re.sub(r"[^0-9A-Za-z_-]+", "_", class_name).strip("_") 
        seg = ann.get("segmentation")
        mask = mask_util.decode({"size": seg["size"], "counts": seg["counts"]})

        base_name = f"object_{idx:03d}_{class_name}"
        mask_npy_path = output_dir / f"{base_name}_mask.npy"

        np.save(mask_npy_path, mask)

        color = (
            int((37 * (idx + 1)) % 256),
            int((97 * (idx + 1)) % 256),
            int((173 * (idx + 1)) % 256),
        )
        color_canvas[mask == 1] = color

    output_viz_path = output_dir / "mask_viz.png"
    cv2.imwrite(str(output_viz_path), color_canvas)

