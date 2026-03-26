import os
import json
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


BASE_DIR = os.path.join(os.path.dirname(__file__), "tmp")

IMAGE_PATH = os.path.join(BASE_DIR, "raw_image.jpg")
MASK_RESULT_JSON_PATH = os.path.join(BASE_DIR, "02_mask", "result.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_depth")

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())

    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth, dtype=np.uint8)

    norm = (depth - depth_min) / (depth_max - depth_min)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def normalize_to_uint16(depth: np.ndarray) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())

    if depth_max - depth_min < 1e-8:
        return np.zeros_like(depth, dtype=np.uint16)

    norm = (depth - depth_min) / (depth_max - depth_min)
    return (norm * 65535.0).clip(0, 65535).astype(np.uint16)


def load_mask(mask_path: str, target_hw: tuple[int, int]) -> Optional[np.ndarray]:
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    th, tw = target_hw
    if mask.shape[0] != th or mask.shape[1] != tw:
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    return (mask > 127).astype(np.uint8)


def robust_mask_stats(depth: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    values = depth[mask == 1]
    if values.size == 0:
        return None

    return {
        "depth_min": float(values.min()),
        "depth_max": float(values.max()),
        "depth_mean": float(values.mean()),
        "depth_median": float(np.median(values)),
        "depth_std": float(values.std()),
        "num_pixels": int(values.size),
    }


def get_center_depth(depth: np.ndarray, bbox_xyxy: List[float]) -> float:
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    return float(depth[cy, cx])


def build_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


def infer_depth(image: Image.Image, processor, model) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )
    predicted_depth = post_processed[0]["predicted_depth"]
    return predicted_depth.detach().cpu().numpy().astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    print("IMAGE_PATH =", IMAGE_PATH)
    print("MASK_RESULT_JSON_PATH =", MASK_RESULT_JSON_PATH)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("MODEL_ID =", MODEL_ID)
    print("DEVICE =", DEVICE)
    print("IMAGE_SIZE =", (w, h))

    processor, model = build_model()
    depth = infer_depth(image, processor, model)

    depth_npy_path = os.path.join(OUTPUT_DIR, "depth_raw.npy")
    np.save(depth_npy_path, depth)

    depth_u8 = normalize_to_uint8(depth)
    depth_u16 = normalize_to_uint16(depth)

    depth_gray_path = os.path.join(OUTPUT_DIR, "depth_gray_8bit.png")
    depth_gray16_path = os.path.join(OUTPUT_DIR, "depth_gray_16bit.png")
    depth_color_path = os.path.join(OUTPUT_DIR, "depth_color.png")

    cv2.imwrite(depth_gray_path, depth_u8)
    cv2.imwrite(depth_gray16_path, depth_u16)

    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(depth_color_path, depth_color)

    annotations_out: List[Dict[str, Any]] = []

    if os.path.exists(MASK_RESULT_JSON_PATH):
        mask_result = json.load(open(MASK_RESULT_JSON_PATH, "r", encoding="utf-8"))
        annotations = mask_result.get("annotations", [])

        for ann in annotations:
            obj_id = ann.get("id")
            label = ann.get("class_name")
            mask_path = ann.get("mask_path")
            bbox_xyxy = ann.get("bbox_xyxy")

            mask = load_mask(mask_path, (h, w)) if mask_path else None
            mask_stats = robust_mask_stats(depth, mask) if mask is not None else None
            center_depth = get_center_depth(depth, bbox_xyxy) if bbox_xyxy is not None else None

            ann_out = {
                "id": obj_id,
                "class_name": label,
                "mask_path": mask_path,
                "bbox_xyxy": bbox_xyxy,
                "bbox_center_depth": center_depth,
                "mask_depth_stats": mask_stats,
            }
            annotations_out.append(ann_out)

            print(f"[{obj_id}] {label}")
            print(f"    center_depth = {center_depth}")
            if mask_stats is not None:
                print(f"    median_depth = {mask_stats['depth_median']}")
                print(f"    mean_depth   = {mask_stats['depth_mean']}")
                print(f"    pixels       = {mask_stats['num_pixels']}")
            else:
                print("    mask depth stats = None")
    else:
        print("[INFO] grounded_sam result.json이 없어서 전체 depth만 저장합니다.")

    result = {
        "image_path": IMAGE_PATH,
        "mask_result_json_path": MASK_RESULT_JSON_PATH if os.path.exists(MASK_RESULT_JSON_PATH) else None,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "depth_type": "relative",
        "image_size_wh": [w, h],
        "depth_raw_npy_path": depth_npy_path,
        "depth_gray_8bit_path": depth_gray_path,
        "depth_gray_16bit_path": depth_gray16_path,
        "depth_color_path": depth_color_path,
        "global_depth_stats": {
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "depth_mean": float(depth.mean()),
            "depth_median": float(np.median(depth)),
            "depth_std": float(depth.std()),
        },
        "annotations": annotations_out,
    }

    result_json_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[SAVE]", depth_npy_path)
    print("[SAVE]", depth_gray_path)
    print("[SAVE]", depth_gray16_path)
    print("[SAVE]", depth_color_path)
    print("[SAVE]", result_json_path)


if __name__ == "__main__":
    main()
