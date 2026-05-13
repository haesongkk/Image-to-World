from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

from src.pipeline_types import StageResult


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_run_manifest(
    output_path: Path,
    input_image: Path,
    stage_results: list[StageResult],
    stage: str,
    resume: bool,
    project_root: Path,
    started_at: str,
    failed_stage: str | None = None,
    error_message: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "requested_stage": stage,
        "resume": resume,
        "git_commit": git_commit(project_root),
        "input_image_path": str(input_image),
        "input_image_sha256": file_sha256(input_image) if input_image.exists() else None,
        "failed_stage": failed_stage,
        "error_message": error_message,
        "stages": [
            {
                "stage": r.stage,
                "skipped": r.skipped,
                "duration_sec": round(r.duration_sec, 3),
                "outputs": [str(p) for p in r.outputs],
                "warnings": r.warnings,
            }
            for r in stage_results
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
