from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TaggerAdapter(Protocol):
    def predict_tags(self, image) -> list[str]: ...


class DetectorSegmenterAdapter(Protocol):
    def detect(self, image, text_prompt: str) -> dict[str, Any]: ...
    def segment(self, image_np, boxes): ...


class InpainterAdapter(Protocol):
    def inpaint(self, *, prompt: str, image, mask_image, **kwargs): ...


class AssetGeneratorAdapter(Protocol):
    def generate_mesh(self, image, **kwargs): ...


class DepthEstimatorAdapter(Protocol):
    def estimate(self, image): ...
