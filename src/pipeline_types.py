from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageResult:
    stage: str
    skipped: bool = False
    duration_sec: float = 0.0
    outputs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
