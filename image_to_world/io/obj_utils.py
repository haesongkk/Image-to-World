from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def try_parse_face_triplet(token: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    parts = token.split("/")
    v_idx = int(parts[0]) if len(parts) >= 1 and parts[0] else None
    vt_idx = int(parts[1]) if len(parts) >= 2 and parts[1] else None
    vn_idx = int(parts[2]) if len(parts) >= 3 and parts[2] else None
    return v_idx, vt_idx, vn_idx


def triangulate_face_tokens(face_tokens: list[str]) -> list[list[str]]:
    if len(face_tokens) < 3:
        return []
    if len(face_tokens) == 3:
        return [face_tokens]
    return [[face_tokens[0], face_tokens[i], face_tokens[i + 1]] for i in range(1, len(face_tokens) - 1)]


def load_obj_basic(path: str | Path) -> dict[str, Any]:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    faces_v: list[list[int]] = []
    faces_vt: list[list[Optional[int]]] = []
    faces_vn: list[list[Optional[int]]] = []

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                for tri in triangulate_face_tokens(line.split()[1:]):
                    tri_v, tri_vt, tri_vn = [], [], []
                    for tok in tri:
                        v_idx, vt_idx, vn_idx = try_parse_face_triplet(tok)
                        if v_idx is None or v_idx < 1:
                            raise ValueError(f"OBJ relative/invalid index unsupported: {tok} in {path}")
                        tri_v.append(v_idx - 1)
                        tri_vt.append(None if vt_idx is None else vt_idx - 1)
                        tri_vn.append(None if vn_idx is None else vn_idx - 1)
                    faces_v.append(tri_v)
                    faces_vt.append(tri_vt)
                    faces_vn.append(tri_vn)

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")

    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "texcoords": np.asarray(texcoords, dtype=np.float64) if texcoords else np.zeros((0, 2), dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64) if normals else np.zeros((0, 3), dtype=np.float64),
        "faces_v": faces_v,
        "faces_vt": faces_vt,
        "faces_vn": faces_vn,
    }
