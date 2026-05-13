from __future__ import annotations

from pathlib import Path


def stage_deps_ready(stage: str, deps: dict[str, list[Path]]) -> tuple[bool, list[Path]]:
    missing = [p for p in deps.get(stage, []) if not p.exists()]
    return len(missing) == 0, missing


def stage_output_ready(stage: str, outputs: dict[str, list[Path]]) -> bool:
    candidates = outputs.get(stage, [])
    if stage == "generation":
        if len(candidates) < 2:
            return False
        remesh_dir, mesh_dir = candidates[0], candidates[1]
        has_remesh = remesh_dir.exists() and any(remesh_dir.glob("*_remeshed.glb"))
        has_hunyuan = mesh_dir.exists() and any(mesh_dir.glob("*_shape_mesh.glb"))
        return has_remesh or has_hunyuan
    return all(p.exists() for p in candidates)
