import os
import subprocess
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

def run_groundedsam2(image_path: Path):
    project_root = Path(__file__).resolve().parent.parent.parent

    repo_root = project_root / "third_party" / "Grounded-SAM-2"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script = repo_root / "grounded_sam2_hf_model_demo.py"

    text_prompt_path = project_root / "output" / "recognize-anything" / "text_prompt.txt"
    if not text_prompt_path.exists():
        raise RuntimeError(f"Text prompt file not found: {text_prompt_path}")
    with open(text_prompt_path, "r", encoding="utf-8") as f:
        text_prompt = f.read().strip()

    output_dir = project_root / "output" / "Grounded-SAM-2"
    os.makedirs(output_dir , exist_ok=True)
    
    result = subprocess.run(
        [
            str(venv_python),
            str(inference_script),
            "--text-prompt", str(text_prompt),
            "--img-path", str(image_path),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo_root),
    )

    if(result.returncode != 0):
        raise RuntimeError("Grounded-SAM-2 inference failed..\n" + result.stderr)

    results_json_path = output_dir / "grounded_sam2_hf_model_demo_results.json"
    if not results_json_path.exists():
        raise RuntimeError(f"Results json file not found: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        infer_results = json.load(f)

    source_image_path = Path(infer_results.get("image_path", str(image_path)))
    if not source_image_path.exists():
        raise RuntimeError(f"Source image file not found: {source_image_path}")

    source_image = Image.open(source_image_path).convert("RGBA")
    source_arr = np.array(source_image)

    crops_dir = output_dir / "crops"
    masks_dir = output_dir / "masks"
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    annotations = infer_results.get("annotations", [])
    for idx, ann in enumerate(annotations):
        class_name = str(ann.get("class_name", "object")).strip() or "object"
        safe_class_name = re.sub(r"[^0-9A-Za-z_-]+", "_", class_name).strip("_") or "object"

        seg = ann.get("segmentation")
        if not seg or "counts" not in seg or "size" not in seg:
            continue

        mask = mask_util.decode({"size": seg["size"], "counts": seg["counts"]})
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = (mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        cutout = source_arr.copy()
        cutout[:, :, 3] = mask * 255
        crop = cutout[y_min : y_max + 1, x_min : x_max + 1]

        score_val = ann.get("score", 0.0)
        if isinstance(score_val, list):
            score_val = score_val[0] if score_val else 0.0
        try:
            score_str = f"{float(score_val):.3f}"
        except (TypeError, ValueError):
            score_str = "0.000"

        out_name = f"{idx:03d}_{safe_class_name}_{score_str}.png"
        out_path = crops_dir / out_name
        Image.fromarray(crop, mode="RGBA").save(out_path)

        mask_out_name = f"{idx:03d}_{safe_class_name}_{score_str}_mask.png"
        mask_out_path = masks_dir / mask_out_name
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_out_path)
