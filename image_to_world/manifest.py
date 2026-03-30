from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_to_world.common import artifact_path, load_json, save_json
from image_to_world.schemas import StageResult


@dataclass
class ManifestStore:
    path: Path = artifact_path("manifest.json")

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"stages": {}}
        return load_json(self.path)

    def record(self, result: StageResult) -> None:
        manifest = self.load()
        stages = manifest.setdefault("stages", {})
        stages[result.stage_name] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        save_json(self.path, manifest)

    def get_stage(self, stage_name: str) -> dict[str, Any] | None:
        return self.load().get("stages", {}).get(stage_name)
