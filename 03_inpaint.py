import os
import json

from typing import Dict, Any, List

import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting


INPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "tmp", "02_mask", "result.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tmp", "03_inpaint")

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

CANVAS_SCALE = 1.5

NUM_INFERENCE_STEPS = 12
GUIDANCE_SCALE = 5.0
STRENGTH = 0.95

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def clean_label(label: str) -> str:
    return label.strip().lower()


def make_prompt(label: str) -> str:
    label = clean_label(label)
    return f"a complete {label}"


def load_rgba_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def build_inpaint_inputs(rgba_img: Image.Image, canvas_scale: float = 2.0):
    """
    입력 RGBA crop을 더 큰 캔버스 중앙에 배치하고,
    객체가 있는 부분은 유지(검정), 나머지 확장 영역은 생성(흰색)하도록 mask 생성
    """
    w, h = rgba_img.size
    canvas_w = max(64, int(w * canvas_scale))
    canvas_h = max(64, int(h * canvas_scale))

    rgba_canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))

    offset_x = (canvas_w - w) // 2
    offset_y = (canvas_h - h) // 2

    rgba_canvas.paste(rgba_img, (offset_x, offset_y), rgba_img)

    rgba_np = np.array(rgba_canvas)
    rgb_np = rgba_np[:, :, :3].copy()
    alpha_np = rgba_np[:, :, 3].copy()

    bg_mask = alpha_np == 0
    rgb_np[bg_mask] = 255

    mask_np = np.where(alpha_np > 0, 0, 255).astype(np.uint8)

    input_image = Image.fromarray(rgb_np).convert("RGB")
    mask_image = Image.fromarray(mask_np).convert("L")

    return input_image, mask_image


def build_pipeline():
    if DEVICE == "cuda":
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        ).to("cpu")

    return pipe


def save_debug_images(
    input_image: Image.Image,
    mask_image: Image.Image,
    output_dir: str,
    obj_id: int
):
    input_path = os.path.join(output_dir, f"inpaint_input_{obj_id:02d}.png")
    mask_path = os.path.join(output_dir, f"inpaint_mask_{obj_id:02d}.png")

    input_image.save(input_path)
    mask_image.save(mask_path)

    return input_path, mask_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"입력 JSON이 없습니다: {INPUT_JSON_PATH}")

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations: List[Dict[str, Any]] = data.get("annotations", [])

    if len(annotations) == 0:
        print("[INFO] annotations가 없습니다.")
        return

    print("INPUT_JSON_PATH =", INPUT_JSON_PATH)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("DEVICE =", DEVICE)
    print("NUM_OBJECTS =", len(annotations))

    pipe = build_pipeline()

    amodal_annotations = []

    for ann in annotations:
        obj_id = ann["id"]
        label = ann["class_name"]
        crop_rgba_path = ann["crop_rgba_path"]

        if not os.path.exists(crop_rgba_path):
            print(f"[SKIP] crop_rgba 파일 없음: {crop_rgba_path}")
            continue

        prompt = make_prompt(label)

        rgba_img = load_rgba_image(crop_rgba_path)
        input_image, mask_image = build_inpaint_inputs(rgba_img, canvas_scale=CANVAS_SCALE)

        debug_input_path, debug_mask_path = save_debug_images(
            input_image=input_image,
            mask_image=mask_image,
            output_dir=OUTPUT_DIR,
            obj_id=obj_id
        )

        print(f"[{obj_id}] label={label}")
        print(f"    prompt = {prompt}")

        result = pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            strength=STRENGTH
        ).images[0]

        amodal_rgb_path = os.path.join(OUTPUT_DIR, f"amodal_rgb_{obj_id:02d}.png")
        result.save(amodal_rgb_path)

        amodal_annotations.append({
            "id": obj_id,
            "class_name": label,
            "prompt": prompt,
            "crop_rgba_path": crop_rgba_path,
            "inpaint_input_path": debug_input_path,
            "inpaint_mask_path": debug_mask_path,
            "amodal_rgb_path": amodal_rgb_path
        })

        print(f"    saved -> {amodal_rgb_path}")

    out_json = {
        "input_json_path": INPUT_JSON_PATH,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "canvas_scale": CANVAS_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "strength": STRENGTH,
        "annotations": amodal_annotations
    }

    out_json_path = os.path.join(OUTPUT_DIR, "amodal_result.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print("[SAVE]", out_json_path)


if __name__ == "__main__":
    main()