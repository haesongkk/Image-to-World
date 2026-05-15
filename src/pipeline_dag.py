from __future__ import annotations

from pathlib import Path


def stage_deps_ready(stage: str, deps: dict[str, list[Path]]) -> tuple[bool, list[Path]]:
    missing = [p for p in deps.get(stage, []) if not p.exists()]
    return len(missing) == 0, missing


def stage_output_ready(stage: str, outputs: dict[str, list[Path]]) -> bool:
    candidates = outputs.get(stage, [])
    if stage == "mesh_generation":
        if not candidates:
            return False
        mesh_dir = candidates[0]
        return mesh_dir.exists() and any(mesh_dir.glob("*_shape_mesh.glb"))
    if stage == "mesh_remesh":
        if not candidates:
            return False
        remesh_dir = candidates[0]
        return remesh_dir.exists() and any(remesh_dir.glob("*_remeshed.glb"))
    if stage == "mesh_texturing":
        if not candidates:
            return False
        remesh_dir = candidates[0]
        return remesh_dir.exists() and any(remesh_dir.glob("*_remeshed_textured.glb"))
    return all(p.exists() for p in candidates)
