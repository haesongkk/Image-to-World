from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


@dataclass
class StageResult:
    stage_name: str
    output_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class Annotation:
    id: int
    class_name: str
    score: Optional[float] = None
    bbox_xyxy: Optional[list[float]] = None
    mask_path: Optional[str] = None
    crop_bbox_xyxy: Optional[list[int]] = None
    crop_rgb_path: Optional[str] = None
    crop_rgba_path: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra")
        payload.update(extra)
        return _serialize(payload)


@dataclass
class DepthAnnotation:
    id: int
    class_name: str
    mask_path: Optional[str]
    bbox_xyxy: Optional[list[float]]
    bbox_center_depth: Optional[float]
    mask_depth_stats: Optional[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class MeshArtifact:
    obj_path: Optional[str]
    ply_path: Optional[str]
    meta_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class Placement:
    id: int
    class_name: str
    score: Optional[float]
    bbox_xyxy: list[float]
    bbox_center_xy: list[float]
    bbox_size_wh: list[float]
    depth_source: str
    relative_depth_value: float
    pseudo_world: dict[str, Any]
    mesh: MeshArtifact
    source_paths: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mesh"] = self.mesh.to_dict()
        return _serialize(payload)
