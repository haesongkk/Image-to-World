from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

RAW_IMAGE_NAME = "raw_image.jpg"


def raw_image_path(image_path: str | Path | None = None) -> Path:
    if image_path is None:
        return DATA_DIR / RAW_IMAGE_NAME
    return Path(image_path).expanduser().resolve()
