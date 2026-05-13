from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Mapping


def run_external_command(
    name: str,
    command: list[str],
    cwd: Path,
    log_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_sec: int = 3600,
) -> subprocess.CompletedProcess[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    if not command:
        raise ValueError(f"{name}: command is empty")
    executable = Path(command[0])
    if executable.suffix.lower() in {".exe", ".bat", ".cmd", ".ps1", ""} and not executable.exists():
        raise RuntimeError(f"{name}: executable not found: {executable}")
    if not cwd.exists():
        raise RuntimeError(f"{name}: cwd not found: {cwd}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        timeout=timeout_sec,
    )

    (log_dir / f"{name}.stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr or "", encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"{name} inference failed with exit code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )
    return result
