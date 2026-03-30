from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from image_to_world.cache import CacheStore
from image_to_world.common import ensure_dir
from image_to_world.logging_utils import get_logger
from image_to_world.manifest import ManifestStore
from image_to_world.schemas import StageResult


class Stage:
    stage_name = "base"

    def __init__(self, *, manifest: ManifestStore, cache: CacheStore) -> None:
        self.manifest = manifest
        self.cache = cache
        self.logger = get_logger(f"stage.{self.stage_name}")

    def run(self, *args, **kwargs) -> StageResult:
        raise NotImplementedError

    def finalize(self, result: StageResult, cache_payload: dict[str, Any] | None = None) -> StageResult:
        self.manifest.record(result)
        if cache_payload is not None:
            key = self.cache.build_key(result.stage_name, cache_payload)
            self.cache.record(key, result.to_dict())
        return result

    def should_skip(self, output_path: Path) -> bool:
        return self.cache is not None and output_path.exists() and getattr(self, "runtime", None) is not None and self.runtime.skip_existing and not self.runtime.overwrite

    @staticmethod
    def config_to_cache_payload(config: Any) -> dict[str, Any]:
        if is_dataclass(config):
            payload = asdict(config)
        else:
            payload = dict(config)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}

    @staticmethod
    def ensure_output_dir(path: Path) -> Path:
        return ensure_dir(path)
