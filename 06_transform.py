import os
import json

import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


try:
    from PIL import Image
except ImportError:
    Image = None


BASE_DIR = os.path.join(os.path.dirname(__file__), "tmp")

MASK_JSON_PATH = os.path.join(BASE_DIR, "02_mask", "result.json")
DEPTH_JSON_PATH = os.path.join(BASE_DIR, "05_depth", "result.json")
GEN3D_JSON_PATH = os.path.join(BASE_DIR, "04_asset", "gen3d_result.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "06_transform")
RAW_IMAGE_PATH = os.path.join(BASE_DIR, "raw_image.jpg")

CLOSE_IS_LARGER = True

Z_NEAR = 1.0
Z_FAR = 6.0

FOCAL_SCALE_X = 1.2
FOCAL_SCALE_Y = 1.2

GLOBAL_SCALE_MULTIPLIER = 1.0

SAVE_VISUALIZATION = True
VIS_PNG_PATH = os.path.join(OUTPUT_DIR, "place_viz.png")
VIS_SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "place_viz_summary.json")




def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))



def bbox_center_and_size(bbox_xyxy: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = float(x1 + x2) * 0.5
    cy = float(y1 + y2) * 0.5
    return {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
    }



def normalize_depth_value(depth_value: float, dmin: float, dmax: float) -> float:
    if dmax - dmin < 1e-8:
        return 0.5
    return clamp((depth_value - dmin) / (dmax - dmin), 0.0, 1.0)



def relative_depth_to_pseudo_z(depth_value: float, dmin: float, dmax: float, close_is_larger: bool) -> float:
    depth_norm = normalize_depth_value(depth_value, dmin, dmax)

    closeness = depth_norm if close_is_larger else (1.0 - depth_norm)

    z = Z_FAR - closeness * (Z_FAR - Z_NEAR)
    return float(z)



def estimate_world_position(
    cx_px: float,
    cy_px: float,
    z: float,
    image_w: float,
    image_h: float,
    fx: float,
    fy: float,
) -> Dict[str, float]:
    cx0 = image_w * 0.5
    cy0 = image_h * 0.5

    x = ((cx_px - cx0) / fx) * z
    y = -((cy_px - cy0) / fy) * z

    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }



def estimate_world_size(
    bbox_w_px: float,
    bbox_h_px: float,
    z: float,
    fx: float,
    fy: float,
    global_scale_multiplier: float,
) -> Dict[str, float]:
    world_w = max(1e-6, (bbox_w_px / fx) * z * global_scale_multiplier)
    world_h = max(1e-6, (bbox_h_px / fy) * z * global_scale_multiplier)

    uniform_scale = math.sqrt(world_w * world_h)

    return {
        "world_w": float(world_w),
        "world_h": float(world_h),
        "suggested_uniform_scale": float(uniform_scale),
    }



def choose_object_depth(depth_ann: Dict[str, Any]) -> Optional[float]:
    mask_stats = depth_ann.get("mask_depth_stats")
    if isinstance(mask_stats, dict):
        median_depth = mask_stats.get("depth_median")
        if median_depth is not None:
            return float(median_depth)

    center_depth = depth_ann.get("bbox_center_depth")
    if center_depth is not None:
        return float(center_depth)

    return None



def build_index_by_id(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in items:
        obj_id = item.get("id")
        if obj_id is None:
            continue
        out[int(obj_id)] = item
    return out



def load_image(path: str):
    if not os.path.exists(path) or Image is None:
        return None
    return Image.open(path).convert("RGB")



def deterministic_color(idx: int) -> Tuple[float, float, float]:
    palette = [
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
    return palette[idx % len(palette)]



def draw_image_overlay(ax, image, placements: List[Dict[str, Any]]):
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor((0.08, 0.08, 0.08))
        ax.text(0.5, 0.5, "raw_image.jpg 없음", ha="center", va="center", color="white", transform=ax.transAxes)

    for i, obj in enumerate(placements):
        color = deterministic_color(i)
        bbox = obj.get("bbox_xyxy")
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        rect = Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)

        center = obj.get("bbox_center_xy")
        if center is not None and len(center) == 2:
            ax.scatter([center[0]], [center[1]], s=18, c=[color])

        label = obj.get("class_name", f"obj_{i}")
        pos = obj.get("pseudo_world", {}).get("position_xyz", [0, 0, 0])
        txt = f"[{obj.get('id', i)}] {label}\nz={pos[2]:.2f}"
        ax.text(
            x1,
            max(0, y1 - 6),
            txt,
            fontsize=8,
            color="white",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, pad=2.0),
        )

    ax.set_title("Image Overlay (bbox + label + pseudo z)")
    ax.axis("off")



def draw_top_view(ax, placements: List[Dict[str, Any]]):
    xs, zs, labels, sizes, colors = [], [], [], [], []

    for i, obj in enumerate(placements):
        pos = obj.get("pseudo_world", {}).get("position_xyz", [0, 0, 0])
        scale = obj.get("pseudo_world", {}).get("scale_xyz", [1, 1, 1])
        color = deterministic_color(i)

        x, _, z = pos
        s = max(20.0, float(scale[0]) * 300.0)

        xs.append(float(x))
        zs.append(float(z))
        labels.append(f"[{obj.get('id', i)}] {obj.get('class_name', 'unknown')}")
        sizes.append(s)
        colors.append(color)

    if len(xs) == 0:
        ax.text(0.5, 0.5, "배치된 객체 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Top View (X-Z)")
        return

    ax.scatter(xs, zs, s=sizes, c=colors, alpha=0.85, edgecolors="black", linewidths=0.8)
    for x, z, lab in zip(xs, zs, labels):
        ax.text(x, z, lab, fontsize=8, ha="left", va="bottom")

    ax.scatter([0], [0], s=100, c="black", marker="^")
    ax.text(0, 0, "camera", fontsize=9, ha="left", va="bottom", color="black")

    ax.set_xlabel("pseudo world X")
    ax.set_ylabel("pseudo world Z")
    ax.set_title("Top View (X-Z)")
    ax.grid(True, alpha=0.25)

    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    dx = max(0.5, (xmax - xmin) * 0.15)
    dz = max(0.5, (zmax - zmin) * 0.15)
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(min(0, zmin - dz), zmax + dz)



def draw_front_view(ax, placements: List[Dict[str, Any]]):
    xs, ys, labels, sizes, colors = [], [], [], [], []

    for i, obj in enumerate(placements):
        pos = obj.get("pseudo_world", {}).get("position_xyz", [0, 0, 0])
        scale = obj.get("pseudo_world", {}).get("scale_xyz", [1, 1, 1])
        color = deterministic_color(i)

        x, y, _ = pos
        s = max(20.0, float(scale[1]) * 300.0)

        xs.append(float(x))
        ys.append(float(y))
        labels.append(f"[{obj.get('id', i)}] {obj.get('class_name', 'unknown')}")
        sizes.append(s)
        colors.append(color)

    if len(xs) == 0:
        ax.text(0.5, 0.5, "배치된 객체 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Front View (X-Y)")
        return

    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.85, edgecolors="black", linewidths=0.8)
    for x, y, lab in zip(xs, ys, labels):
        ax.text(x, y, lab, fontsize=8, ha="left", va="bottom")

    ax.axhline(0, color="gray", linewidth=1.0, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("pseudo world X")
    ax.set_ylabel("pseudo world Y")
    ax.set_title("Front View (X-Y)")
    ax.grid(True, alpha=0.25)



def save_visualization(result: Dict[str, Any]):
    placements = result.get("placements", [])
    image = load_image(RAW_IMAGE_PATH)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_front = fig.add_subplot(gs[1, 1])

    draw_image_overlay(ax_img, image, placements)
    draw_top_view(ax_top, placements)
    draw_front_view(ax_front, placements)

    fig.suptitle("Place Result Visualization", fontsize=16)
    fig.tight_layout()
    fig.savefig(VIS_PNG_PATH, dpi=180)
    plt.close(fig)

    vis_summary = {
        "scene_layout_json": os.path.join(OUTPUT_DIR, "scene_layout.json"),
        "raw_image_path": RAW_IMAGE_PATH if os.path.exists(RAW_IMAGE_PATH) else None,
        "num_placements": len(placements),
        "output_png": VIS_PNG_PATH,
        "notes": [
            "왼쪽: 원본 이미지 위 bbox/label/pseudo z",
            "오른쪽 위: X-Z top view",
            "오른쪽 아래: X-Y front view",
            "좌표는 metric world가 아니라 place.py의 pseudo world 좌표임",
        ],
    }
    with open(VIS_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(vis_summary, f, ensure_ascii=False, indent=2)
    print("[SAVE]", VIS_PNG_PATH)
    print("[SAVE]", VIS_SUMMARY_JSON_PATH)



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MASK_JSON_PATH):
        raise FileNotFoundError(f"mask result.json이 없습니다: {MASK_JSON_PATH}")
    if not os.path.exists(DEPTH_JSON_PATH):
        raise FileNotFoundError(f"depth result.json이 없습니다: {DEPTH_JSON_PATH}")
    if not os.path.exists(GEN3D_JSON_PATH):
        raise FileNotFoundError(f"gen3d_result.json이 없습니다: {GEN3D_JSON_PATH}")

    mask_data = json.load(open(MASK_JSON_PATH, "r", encoding="utf-8"))
    depth_data = json.load(open(DEPTH_JSON_PATH, "r", encoding="utf-8"))
    gen3d_data = json.load(open(GEN3D_JSON_PATH, "r", encoding="utf-8"))

    mask_annotations = mask_data.get("annotations", [])
    depth_annotations = depth_data.get("annotations", [])
    gen3d_results = gen3d_data.get("results", [])

    if len(mask_annotations) == 0:
        print("[INFO] mask annotations가 없습니다.")
        return

    image_size_wh = depth_data.get("image_size_wh")
    if not image_size_wh or len(image_size_wh) != 2:
        raise ValueError("depth result.json에 image_size_wh가 없습니다.")

    image_w = float(image_size_wh[0])
    image_h = float(image_size_wh[1])
    fx = image_w * FOCAL_SCALE_X
    fy = image_h * FOCAL_SCALE_Y

    global_stats = depth_data.get("global_depth_stats", {})
    dmin = float(global_stats.get("depth_min", 0.0))
    dmax = float(global_stats.get("depth_max", 1.0))

    depth_by_id = build_index_by_id(depth_annotations)
    gen3d_by_id = build_index_by_id(gen3d_results)

    placements: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    print("MASK_JSON_PATH  =", MASK_JSON_PATH)
    print("DEPTH_JSON_PATH =", DEPTH_JSON_PATH)
    print("GEN3D_JSON_PATH =", GEN3D_JSON_PATH)
    print("OUTPUT_DIR      =", OUTPUT_DIR)
    print("IMAGE_SIZE      =", (int(image_w), int(image_h)))
    print("FX, FY          =", (fx, fy))
    print("DEPTH_RANGE     =", (dmin, dmax))
    print("CLOSE_IS_LARGER =", CLOSE_IS_LARGER)

    for mask_ann in mask_annotations:
        obj_id = int(mask_ann["id"])
        label = mask_ann.get("class_name", "unknown")
        bbox_xyxy = mask_ann.get("bbox_xyxy")

        if bbox_xyxy is None:
            skipped.append({
                "id": obj_id,
                "class_name": label,
                "reason": "bbox_xyxy 없음",
            })
            continue

        depth_ann = depth_by_id.get(obj_id)
        gen3d_ann = gen3d_by_id.get(obj_id)

        if depth_ann is None:
            skipped.append({
                "id": obj_id,
                "class_name": label,
                "reason": "depth annotation 없음",
            })
            continue

        if gen3d_ann is None:
            skipped.append({
                "id": obj_id,
                "class_name": label,
                "reason": "3D asset result 없음",
            })
            continue

        depth_value = choose_object_depth(depth_ann)
        if depth_value is None:
            skipped.append({
                "id": obj_id,
                "class_name": label,
                "reason": "사용 가능한 depth 값 없음",
            })
            continue

        bbox_info = bbox_center_and_size(bbox_xyxy)
        pseudo_z = relative_depth_to_pseudo_z(
            depth_value=depth_value,
            dmin=dmin,
            dmax=dmax,
            close_is_larger=CLOSE_IS_LARGER,
        )

        pos = estimate_world_position(
            cx_px=bbox_info["cx"],
            cy_px=bbox_info["cy"],
            z=pseudo_z,
            image_w=image_w,
            image_h=image_h,
            fx=fx,
            fy=fy,
        )

        size = estimate_world_size(
            bbox_w_px=bbox_info["w"],
            bbox_h_px=bbox_info["h"],
            z=pseudo_z,
            fx=fx,
            fy=fy,
            global_scale_multiplier=GLOBAL_SCALE_MULTIPLIER,
        )

        placement = {
            "id": obj_id,
            "class_name": label,
            "score": mask_ann.get("score"),
            "bbox_xyxy": bbox_xyxy,
            "bbox_center_xy": [bbox_info["cx"], bbox_info["cy"]],
            "bbox_size_wh": [bbox_info["w"], bbox_info["h"]],
            "depth_source": "mask_depth_stats.depth_median" if depth_ann.get("mask_depth_stats") else "bbox_center_depth",
            "relative_depth_value": float(depth_value),
            "pseudo_world": {
                "position_xyz": [pos["x"], pos["y"], pos["z"]],
                "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                "scale_xyz": [
                    size["suggested_uniform_scale"],
                    size["suggested_uniform_scale"],
                    size["suggested_uniform_scale"],
                ],
                "estimated_world_size_wh": [size["world_w"], size["world_h"]],
            },
            "mesh": {
                "obj_path": gen3d_ann.get("obj_path"),
                "ply_path": gen3d_ann.get("ply_path"),
                "meta_path": os.path.join(os.path.dirname(gen3d_ann.get("obj_path", "")), "meta.json") if gen3d_ann.get("obj_path") else None,
            },
            "source_paths": {
                "mask_path": mask_ann.get("mask_path"),
                "crop_rgb_path": mask_ann.get("crop_rgb_path"),
                "crop_rgba_path": mask_ann.get("crop_rgba_path"),
            },
        }

        placements.append(placement)

        print(f"[{obj_id}] {label}")
        print(f"    depth = {depth_value:.6f} -> pseudo_z = {pseudo_z:.6f}")
        print(f"    pos   = ({pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f})")
        print(f"    size  = world_w={size['world_w']:.4f}, world_h={size['world_h']:.4f}, uniform={size['suggested_uniform_scale']:.4f}")

    placements_sorted = sorted(placements, key=lambda x: x["pseudo_world"]["position_xyz"][2])

    result = {
        "mask_json_path": MASK_JSON_PATH,
        "depth_json_path": DEPTH_JSON_PATH,
        "gen3d_json_path": GEN3D_JSON_PATH,
        "image_size_wh": [int(image_w), int(image_h)],
        "camera_assumption": {
            "type": "pseudo_pinhole_camera",
            "fx": fx,
            "fy": fy,
            "cx": image_w * 0.5,
            "cy": image_h * 0.5,
            "close_is_larger": CLOSE_IS_LARGER,
            "z_near": Z_NEAR,
            "z_far": Z_FAR,
            "global_scale_multiplier": GLOBAL_SCALE_MULTIPLIER,
            "notes": [
                "Depth Anything V2 relative depth를 metric depth로 쓰지 않고 pseudo depth로 사용함",
                "좌표와 스케일은 최소 데모용 초기값이며, 이후 ground plane / camera recovery / mesh normalization 단계에서 보정 필요",
            ],
        },
        "num_requested_objects": len(mask_annotations),
        "num_placed_objects": len(placements_sorted),
        "num_skipped_objects": len(skipped),
        "placements": placements_sorted,
        "skipped": skipped,
    }

    out_json_path = os.path.join(OUTPUT_DIR, "scene_layout.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n[SAVE]", out_json_path)

    if SAVE_VISUALIZATION:
        save_visualization(result)


if __name__ == "__main__":
    main()
