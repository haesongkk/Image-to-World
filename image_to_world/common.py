from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def artifact_path(*parts: str) -> Path:
    return ARTIFACTS_DIR.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_json(path: Path) -> Any:
    require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> Path:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
