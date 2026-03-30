from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from image_to_world.common import artifact_path, ensure_dir, save_json


class CacheStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = ensure_dir(root or artifact_path("cache"))

    def build_key(self, stage_name: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"{stage_name}-{digest}"

    def marker_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.marker_path(key).exists()

    def record(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.marker_path(key)
        save_json(path, payload)
        return path
