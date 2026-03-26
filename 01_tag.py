import os
import json
from typing import Dict, Any

import torch
from PIL import Image
from torchvision import transforms
from typing import List

from ram.models import ram_plus
from ram import inference_ram as inference

BACKGROUND_LIKE_TAGS = ["room", "floor", "wood floor", "wall", "wood wall", "ceiling", "background", "indoor", "interior", "furniture", "home", "house", "bedroom", "hotel room"]

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "ram_plus_swin_large_14m.pth")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "tmp", "raw_image.jpg")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "tmp", "01_tag", "ram_result.json")

IMAGE_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
    return Image.open(image_path).convert("RGB")


def process_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def run_ram(tensor: torch.Tensor) -> List[str]:
    model = ram_plus(
        pretrained=CHECKPOINT_PATH,
        image_size=IMAGE_SIZE,
        vit="swin_l"
    )
    model.eval()
    model = model.to(DEVICE)

    with torch.no_grad():
        result = inference(tensor, model)

    english_tags = result[0]
    tags = [t.strip().lower() for t in english_tags.split("|") if t.strip()]

    return tags

# 배경 태그 필터링
def filter_tags(tags: List[str]) -> List[str]:
    filtered = []
    seen = set()

    for tag in tags:
        t = tag.strip().lower()
        if not t:
            continue
        if t in BACKGROUND_LIKE_TAGS:
            continue
        if t in seen:
            continue
        seen.add(t)
        filtered.append(t)

    return filtered

def save_json(path: str, data: Dict[str, Any]):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


image = load_image(IMAGE_PATH)
tensor = process_image(image)

raw_tags = run_ram(tensor)
tags = filter_tags(raw_tags)

save_json(OUTPUT_PATH, tags)

print("=== RAM++ TAGS ===")
print(tags)
print(f"[SAVE] {OUTPUT_PATH}")