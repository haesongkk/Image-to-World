from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from src.config import OUTPUT_DIR


def run_birefnet(input_image_path: Path) -> Path:
    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    output_dir = OUTPUT_DIR / "BirefNet"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_image_path.stem}_birefnet.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(input_image_path).convert("RGB")

    input_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    alpha = preds[0].squeeze()
    alpha_mask = transforms.ToPILImage()(alpha).resize(image.size)

    # image_rgba = image.copy()
    # image_rgba.putalpha(alpha_mask)
    # image_rgba.save(output_path)
    
    # Save a 3-channel result to avoid downstream channel-format issues.
    image_rgb = Image.new("RGB", image.size, (255, 255, 255))
    image_rgb.paste(image, mask=alpha_mask)
    image_rgb.save(output_path)

    return output_path
