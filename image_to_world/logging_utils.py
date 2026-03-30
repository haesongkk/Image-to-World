from __future__ import annotations

import logging
from pathlib import Path

from image_to_world.common import artifact_path, ensure_parent_dir

DEFAULT_LOG_PATH = artifact_path("logs", "pipeline.log")


def configure_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    target = ensure_parent_dir(log_path or DEFAULT_LOG_PATH)
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
