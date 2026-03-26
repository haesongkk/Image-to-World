import os
import json

import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "tmp")
INPUT_LAYOUT_JSON_PATH = os.path.join(BASE_DIR, "06_transform", "scene_layout.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "07_assemble")

ADD_AXIS_HELPER = True
AXIS_LENGTH = 0.6
NORMALIZE_MESH_TO_UNIT_BOX = True
GLOBAL_PRE_ROT_EULER_DEG = [0.0, 0.0, 0.0]

OBJECT_COLOR_PALETTE = [
    (0.90, 0.25, 0.25),
    (0.25, 0.55, 0.95),
    (0.20, 0.75, 0.35),
    (0.95, 0.70, 0.20),
    (0.65, 0.35, 0.90),
    (0.20, 0.80, 0.80),
    (0.95, 0.45, 0.70),
    (0.60, 0.60, 0.20),
    (0.90, 0.55, 0.15),
    (0.45, 0.75, 0.95),
]

AXIS_MATERIALS = {
    "axis_x": (1.0, 0.15, 0.15),
    "axis_y": (0.15, 0.85, 0.15),
    "axis_z": (0.15, 0.35, 1.0),
}


def deg2rad(v: float) -> float:
    return v * math.pi / 180.0


def rotation_matrix_xyz_deg(euler_xyz_deg: List[float]) -> np.ndarray:
    rx, ry, rz = [deg2rad(float(v)) for v in euler_xyz_deg]

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=np.float64)
    ry_m = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float64)
    rz_m = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return rz_m @ ry_m @ rx_m


def compose_transform(scale_xyz: List[float], rot_xyz_deg: List[float], translate_xyz: List[float]) -> np.ndarray:
    s = np.diag([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])])
    r = rotation_matrix_xyz_deg(rot_xyz_deg)
    rs = r @ s

    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rs
    t[:3, 3] = np.array(translate_xyz, dtype=np.float64)
    return t


def apply_transform(vertices: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([vertices.astype(np.float64), ones], axis=1)
    out = (transform_4x4 @ homo.T).T
    return out[:, :3]


def try_parse_face_triplet(token: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    parts = token.split("/")
    v_idx = int(parts[0]) if len(parts) >= 1 and parts[0] != "" else None
    vt_idx = int(parts[1]) if len(parts) >= 2 and parts[1] != "" else None
    vn_idx = int(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
    return v_idx, vt_idx, vn_idx


def triangulate_face_tokens(face_tokens: List[str]) -> List[List[str]]:
    if len(face_tokens) < 3:
        return []
    if len(face_tokens) == 3:
        return [face_tokens]

    triangles = []
    for i in range(1, len(face_tokens) - 1):
        triangles.append([face_tokens[0], face_tokens[i], face_tokens[i + 1]])
    return triangles


def load_obj_basic(path: str) -> Dict[str, Any]:
    vertices: List[List[float]] = []
    texcoords: List[List[float]] = []
    normals: List[List[float]] = []
    faces_v: List[List[int]] = []
    faces_vt: List[List[Optional[int]]] = []
    faces_vn: List[List[Optional[int]]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                tri_faces = triangulate_face_tokens(parts)
                for tri in tri_faces:
                    tri_v = []
                    tri_vt = []
                    tri_vn = []
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

    if len(vertices) == 0:
        raise ValueError(f"정점이 없는 OBJ입니다: {path}")

    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "texcoords": np.asarray(texcoords, dtype=np.float64) if texcoords else np.zeros((0, 2), dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64) if normals else np.zeros((0, 3), dtype=np.float64),
        "faces_v": faces_v,
        "faces_vt": faces_vt,
        "faces_vn": faces_vn,
    }


def normalize_vertices_to_centered_unit_box(vertices: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    size = np.maximum(vmax - vmin, 1e-8)
    max_extent = float(size.max())
    normalized = (vertices - center) / max_extent
    return normalized, {
        "orig_bbox_min": vmin.tolist(),
        "orig_bbox_max": vmax.tolist(),
        "orig_center": center.tolist(),
        "orig_size_xyz": size.tolist(),
        "normalization_divisor": max_extent,
    }


def object_color_rgb(idx: int) -> Tuple[float, float, float]:
    return OBJECT_COLOR_PALETTE[idx % len(OBJECT_COLOR_PALETTE)]


def build_transformed_mesh_record(obj_path: str, placement: Dict[str, Any], object_index: int) -> Dict[str, Any]:
    mesh = load_obj_basic(obj_path)
    vertices = mesh["vertices"]

    normalization_info = None
    if NORMALIZE_MESH_TO_UNIT_BOX:
        vertices, normalization_info = normalize_vertices_to_centered_unit_box(vertices)

    pseudo = placement["pseudo_world"]
    scale_xyz = pseudo.get("scale_xyz", [1.0, 1.0, 1.0])
    rot_xyz_deg = pseudo.get("rotation_euler_xyz_deg", [0.0, 0.0, 0.0])
    pos_xyz = pseudo.get("position_xyz", [0.0, 0.0, 0.0])

    # 좌표 변환 결과를 assemble export 좌표계에 맞게 Z만 반전
    pos_xyz = [float(pos_xyz[0]), float(pos_xyz[1]), -float(pos_xyz[2])]    

    pre_rot = rotation_matrix_xyz_deg(GLOBAL_PRE_ROT_EULER_DEG)
    vertices = (pre_rot @ vertices.T).T

    transform = compose_transform(scale_xyz=scale_xyz, rot_xyz_deg=rot_xyz_deg, translate_xyz=pos_xyz)
    vertices_world = apply_transform(vertices, transform)

    class_name = placement.get("class_name") or f"object_{object_index:02d}"
    material_name = f"mat_{object_index:02d}_{class_name}".replace(" ", "_")
    color_rgb = object_color_rgb(object_index)

    return {
        "source_obj_path": obj_path,
        "vertices_world": vertices_world,
        "faces_v": mesh["faces_v"],
        "faces_vt": mesh["faces_vt"],
        "faces_vn": mesh["faces_vn"],
        "texcoords": mesh["texcoords"],
        "normals": mesh["normals"],
        "normalization_info": normalization_info,
        "applied_scale_xyz": scale_xyz,
        "applied_rotation_euler_xyz_deg": rot_xyz_deg,
        "applied_translation_xyz": pos_xyz,
        "object_name": f"object_{object_index:02d}_{class_name}".replace(" ", "_"),
        "material_name": material_name,
        "material_rgb": list(color_rgb),
        "placement_id": placement.get("id"),
        "class_name": class_name,
    }


def write_mtl(path: str, transformed_meshes: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# assembled scene materials\n\n")

        for rec in transformed_meshes:
            r, g, b = rec["material_rgb"]
            f.write(f"newmtl {rec['material_name']}\n")
            f.write(f"Ka {0.15*r:.6f} {0.15*g:.6f} {0.15*b:.6f}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write("Ks 0.050000 0.050000 0.050000\n")
            f.write("Ns 10.000000\n")
            f.write("illum 2\n\n")

        for mat_name, (r, g, b) in AXIS_MATERIALS.items():
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Ka {0.15*r:.6f} {0.15*g:.6f} {0.15*b:.6f}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("Ns 1.000000\n")
            f.write("illum 1\n\n")


def write_axis_helper_obj_lines(f, start_vertex_index_1based: int, length: float):
    verts = [
        (0.0, 0.0, 0.0),
        (length, 0.0, 0.0),
        (0.0, length, 0.0),
        (0.0, 0.0, length),
    ]
    for v in verts:
        f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

    base = start_vertex_index_1based
    f.write("o axis_helper_x\n")
    f.write("usemtl axis_x\n")
    f.write(f"l {base} {base + 1}\n\n")

    f.write("o axis_helper_y\n")
    f.write("usemtl axis_y\n")
    f.write(f"l {base} {base + 2}\n\n")

    f.write("o axis_helper_z\n")
    f.write("usemtl axis_z\n")
    f.write(f"l {base} {base + 3}\n\n")


def write_merged_obj(obj_path: str, mtl_filename: str, transformed_meshes: List[Dict[str, Any]], add_axis_helper: bool = True):
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# assembled scene OBJ\n")
        f.write(f"# num_objects = {len(transformed_meshes)}\n")
        f.write(f"mtllib {mtl_filename}\n\n")

        global_vertex_offset = 0

        for rec in transformed_meshes:
            object_name = rec["object_name"]
            material_name = rec["material_name"]
            vertices_world = rec["vertices_world"]
            faces_v = rec["faces_v"]

            f.write(f"o {object_name}\n")
            f.write(f"# source_obj_path = {rec['source_obj_path']}\n")
            f.write(f"usemtl {material_name}\n")

            for v in vertices_world:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

            for face in faces_v:
                a = global_vertex_offset + face[0] + 1
                b = global_vertex_offset + face[1] + 1
                c = global_vertex_offset + face[2] + 1
                f.write(f"f {a} {b} {c}\n")

            f.write("\n")
            global_vertex_offset += len(vertices_world)

        if add_axis_helper:
            write_axis_helper_obj_lines(f, start_vertex_index_1based=global_vertex_offset + 1, length=AXIS_LENGTH)


def compute_scene_bounds(transformed_meshes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not transformed_meshes:
        return {
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
            "size_xyz": [0.0, 0.0, 0.0],
        }

    all_v = np.concatenate([rec["vertices_world"] for rec in transformed_meshes], axis=0)
    vmin = all_v.min(axis=0)
    vmax = all_v.max(axis=0)
    center = (vmin + vmax) * 0.5
    size = vmax - vmin
    return {
        "bbox_min": vmin.tolist(),
        "bbox_max": vmax.tolist(),
        "center": center.tolist(),
        "size_xyz": size.tolist(),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_LAYOUT_JSON_PATH):
        raise FileNotFoundError(f"scene_layout.json이 없습니다: {INPUT_LAYOUT_JSON_PATH}")

    scene_layout = json.load(open(INPUT_LAYOUT_JSON_PATH, "r", encoding="utf-8"))
    placements = scene_layout.get("placements", [])

    transformed_meshes = []
    skipped = []

    for idx, placement in enumerate(placements):
        mesh_info = placement.get("mesh", {})
        obj_path = mesh_info.get("obj_path")
        if not obj_path or not os.path.exists(obj_path):
            skipped.append({
                "id": placement.get("id", idx),
                "class_name": placement.get("class_name"),
                "reason": f"obj_path missing or not found: {obj_path}",
            })
            print(f"[SKIP] id={placement.get('id', idx)} class={placement.get('class_name')} : {obj_path}")
            continue

        try:
            rec = build_transformed_mesh_record(obj_path, placement, idx)
            transformed_meshes.append(rec)
            print(f"[OK] {placement.get('id', idx)} {placement.get('class_name')} -> {obj_path}")
        except Exception as e:
            skipped.append({
                "id": placement.get("id", idx),
                "class_name": placement.get("class_name"),
                "reason": str(e),
            })
            print(f"[SKIP] id={placement.get('id', idx)} class={placement.get('class_name')} : {e}")

    merged_obj_path = os.path.join(OUTPUT_DIR, "assembled_scene.obj")
    merged_mtl_path = os.path.join(OUTPUT_DIR, "assembled_scene.mtl")
    write_mtl(merged_mtl_path, transformed_meshes)
    write_merged_obj(merged_obj_path, os.path.join(OUTPUT_DIR, "assembled_scene.mtl"), transformed_meshes, add_axis_helper=ADD_AXIS_HELPER)

    bounds = compute_scene_bounds(transformed_meshes)

    result = {
        "input_layout_json_path": INPUT_LAYOUT_JSON_PATH,
        "output_obj_path": merged_obj_path,
        "output_mtl_path": merged_mtl_path,
        "num_input_placements": len(placements),
        "num_assembled_objects": len(transformed_meshes),
        "num_skipped_objects": len(skipped),
        "scene_bounds": bounds,
        "settings": {
            "add_axis_helper": ADD_AXIS_HELPER,
            "axis_length": AXIS_LENGTH,
            "normalize_mesh_to_unit_box": NORMALIZE_MESH_TO_UNIT_BOX,
            "global_pre_rot_euler_deg": GLOBAL_PRE_ROT_EULER_DEG,
            "object_color_palette": [list(c) for c in OBJECT_COLOR_PALETTE],
        },
        "assembled_objects": [
            {
                "placement_id": rec["placement_id"],
                "class_name": rec["class_name"],
                "source_obj_path": rec["source_obj_path"],
                "object_name": rec["object_name"],
                "material_name": rec["material_name"],
                "material_rgb": rec["material_rgb"],
                "num_vertices": int(rec["vertices_world"].shape[0]),
                "num_faces": int(len(rec["faces_v"])),
                "normalization_info": rec["normalization_info"],
                "applied_scale_xyz": rec["applied_scale_xyz"],
                "applied_rotation_euler_xyz_deg": rec["applied_rotation_euler_xyz_deg"],
                "applied_translation_xyz": rec["applied_translation_xyz"],
            }
            for rec in transformed_meshes
        ],
        "skipped": skipped,
    }

    result_json_path = os.path.join(OUTPUT_DIR, "assembly_result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    preview_csv_path = os.path.join(OUTPUT_DIR, "scene_preview_points.csv")
    with open(preview_csv_path, "w", encoding="utf-8") as f:
        f.write("object_name,material_name,r,g,b,center_x,center_y,center_z\n")
        for rec in transformed_meshes:
            v = rec["vertices_world"]
            center = v.mean(axis=0)
            r, g, b = rec["material_rgb"]
            f.write(
                f"{rec['object_name']},{rec['material_name']},{r:.6f},{g:.6f},{b:.6f},"
                f"{center[0]:.8f},{center[1]:.8f},{center[2]:.8f}\n"
            )

    print("\n[SAVE]", merged_obj_path)
    print("[SAVE]", merged_mtl_path)
    print("[SAVE]", result_json_path)
    print("[SAVE]", preview_csv_path)


if __name__ == "__main__":
    main()
