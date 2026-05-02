import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PF_ROOT = BASE_DIR / "third_party" / "PerspectiveFields"
sys.path.insert(0, str(PF_ROOT))

import argparse
import cv2
import json

from perspective2d import PerspectiveFields

parser = argparse.ArgumentParser(description="Inference script for PerspectiveFields")
parser.add_argument('--image-path', required=True)
parser.add_argument('--output-dir', required=True)

args = parser.parse_args()
image_path = Path(args.image_path)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# specify model version
version = 'Paramnet-360Cities-edina-centered'
# load model
pf_model = PerspectiveFields(version).eval().cuda()
# load image
img_bgr = cv2.imread(str(image_path))
# inference
predictions = pf_model.inference(img_bgr=img_bgr)

output = {
    "pred_rel_focal": predictions["pred_rel_focal"].item(),
    "pred_rel_cx" : predictions["pred_rel_cx"].item(),
    "pred_rel_cy" : predictions["pred_rel_cy"].item(),
    "pred_roll" : predictions["pred_roll"].item(),
    "pred_pitch" : predictions["pred_pitch"].item(),
    "pred_general_vfov": predictions["pred_general_vfov"].item(),
    "pred_vfov": predictions["pred_vfov"].item(),
}

# save results
output_json_path = output_dir / f"{image_path.stem}_perspective_fields.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)