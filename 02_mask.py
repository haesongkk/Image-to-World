import os
import json
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_PATH = os.path.join(os.path.dirname(__file__), "tmp", "raw_image.jpg")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "tmp", "01_tag", "ram_result.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tmp", "02_mask")

SAM2_CONFIG = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
SAM2_CHECKPOINT = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2", "checkpoints", "sam2.1_hiera_large.pt")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def random_color(idx: int):
    rng = np.random.default_rng(seed=idx)
    return tuple(int(x) for x in rng.integers(50, 255, size=3))


def clamp_box_xyxy(box, width: int, height: int):
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def save_object_crops(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    output_dir: str,
    obj_id: int
):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box_xyxy(box, w, h)

    crop_bgr = image_bgr[y1:y2, x1:x2].copy()
    crop_rgb_path = os.path.join(output_dir, f"crop_rgb_{obj_id:02d}.png")
    cv2.imwrite(crop_rgb_path, crop_bgr)

    crop_mask = mask[y1:y2, x1:x2].copy().astype(np.uint8)
    crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_bgra[:, :, 3] = crop_mask * 255

    crop_rgba_path = os.path.join(output_dir, f"crop_rgba_{obj_id:02d}.png")
    cv2.imwrite(crop_rgba_path, crop_bgra)

    return {
        "crop_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "crop_rgb_path": crop_rgb_path,
        "crop_rgba_path": crop_rgba_path
    }

def run_gdino(image: Image.Image, text_prompt: str) -> List[Dict[str, Any]]:
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(DEVICE)
    gdino.eval()

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = gdino(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {IMAGE_PATH}")
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"프롬프트 파일이 없습니다: {PROMPT_PATH}")
    if not os.path.exists(SAM2_CONFIG):
        raise FileNotFoundError(f"SAM2 config 파일이 없습니다: {SAM2_CONFIG}")
    if not os.path.exists(SAM2_CHECKPOINT):
        raise FileNotFoundError(f"SAM2 체크포인트 파일이 없습니다: {SAM2_CHECKPOINT}")

    tags = load_json(PROMPT_PATH)
    text_prompt = " . ".join(tags)
    print("IMAGE_PATH =", IMAGE_PATH)
    print("PROMPT_PATH =", PROMPT_PATH)
    print("TEXT_PROMPT =", text_prompt)
    print("DEVICE =", DEVICE)

    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = run_gdino(image_pil, text_prompt)

    boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.empty((0, 4))
    labels = [str(x) for x in results["labels"]]
    scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([])

    print(f"[INFO] detected boxes = {len(boxes)}")

    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_np)

    overlay = image_bgr.copy()
    annotations = []

    for i, box in enumerate(boxes):
        masks, mask_scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=False
        )

        mask = masks[0].astype(np.uint8)
        color = random_color(i)

        mask_path = os.path.join(OUTPUT_DIR, f"mask_{i:02d}.png")
        cv2.imwrite(mask_path, mask * 255)

        crop_info = save_object_crops(
            image_bgr=image_bgr,
            mask=mask,
            box=box,
            output_dir=OUTPUT_DIR,
            obj_id=i
        )

        colored = np.zeros_like(overlay, dtype=np.uint8)
        colored[:, :] = color
        overlay = np.where(
            mask[:, :, None] == 1,
            overlay * 0.45 + colored * 0.55,
            overlay
        ).astype(np.uint8)

        x1, y1, x2, y2 = clamp_box_xyxy(box, image_bgr.shape[1], image_bgr.shape[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = labels[i]
        score = float(scores[i])

        cv2.putText(
            overlay,
            f"{label} {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        annotations.append({
            "id": i,
            "class_name": label,
            "score": score,
            "bbox_xyxy": [float(v) for v in box.tolist()],
            "mask_path": mask_path,
            "crop_bbox_xyxy": crop_info["crop_bbox_xyxy"],
            "crop_rgb_path": crop_info["crop_rgb_path"],
            "crop_rgba_path": crop_info["crop_rgba_path"]
        })

        print(f"[{i}] {label} {score:.3f}")
        print(f"    mask  -> {mask_path}")
        print(f"    rgb   -> {crop_info['crop_rgb_path']}")
        print(f"    rgba  -> {crop_info['crop_rgba_path']}")

    overlay_path = os.path.join(OUTPUT_DIR, "overlay.png")
    cv2.imwrite(overlay_path, overlay)

    result_json = {
        "image_path": IMAGE_PATH,
        "prompt_path": PROMPT_PATH,
        "text_prompt": text_prompt,
        "annotations": annotations,
        "overlay_path": overlay_path
    }

    json_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print("[SAVE]", overlay_path)
    print("[SAVE]", json_path)


if __name__ == "__main__":
    main()