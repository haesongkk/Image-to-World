import os
import json
import math

from typing import Dict, Any, List

import numpy as np
import torch
from PIL import Image

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh

# 축 회전 설정 (모델이 누워있음)
ASSET_PRE_ROT_EULER_XYZ_DEG = [-90.0, 0.0, 0.0]

INPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "tmp", "03_inpaint", "amodal_result.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tmp", "04_asset")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
GUIDANCE_SCALE = 3.0
USE_KARRAS = True
KARRAS_STEPS = 64
SIGMA_MIN = 1e-3
SIGMA_MAX = 160
S_CHURN = 0

TRANSMITTER_NAME = "transmitter"
IMAGE_MODEL_NAME = "image300M"

def rotation_matrix_xyz_deg(euler_xyz_deg: List[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(v)) for v in euler_xyz_deg]

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], dtype=np.float64)

    ry_m = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float64)

    rz_m = np.array([
        [cz, -sz, 0.0],
        [sz,  cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return rz_m @ ry_m @ rx_m


def rotate_mesh_vertices(mesh, euler_xyz_deg: List[float]):
    verts = np.asarray(mesh.verts, dtype=np.float64)
    rot = rotation_matrix_xyz_deg(euler_xyz_deg)
    rotated = (rot @ verts.T).T
    mesh.verts = rotated
    return mesh

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"입력 JSON이 없습니다: {INPUT_JSON_PATH}")

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations: List[Dict[str, Any]] = data.get("annotations", [])

    if len(annotations) == 0:
        print("[INFO] amodal annotations가 없습니다.")
        return

    print("INPUT_JSON_PATH =", INPUT_JSON_PATH)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("DEVICE =", DEVICE)
    print("NUM_OBJECTS =", len(annotations))

    print("[LOAD] transmitter")
    xm = load_model(TRANSMITTER_NAME, device=DEVICE)

    print("[LOAD] image-conditioned model")
    model = load_model(IMAGE_MODEL_NAME, device=DEVICE)

    print("[LOAD] diffusion config")
    diffusion = diffusion_from_config(load_config("diffusion"))

    all_results = []

    for ann in annotations:
        obj_id = ann["id"]
        label = ann["class_name"]
        image_path = ann["amodal_rgb_path"]

        if not os.path.exists(image_path):
            print(f"[SKIP] 이미지 파일 없음: {image_path}")
            continue

        obj_dir = os.path.join(OUTPUT_DIR, f"object_{obj_id:02d}")
        os.makedirs(obj_dir, exist_ok=True)

        print(f"\n[OBJECT {obj_id:02d}] label={label}")
        print(f"  image={image_path}")

        image = Image.open(image_path).convert("RGB")

        latents = sample_latents(
            batch_size=BATCH_SIZE,
            model=model,
            diffusion=diffusion,
            guidance_scale=GUIDANCE_SCALE,
            model_kwargs=dict(images=[image] * BATCH_SIZE),
            progress=True,
            clip_denoised=True,
            use_fp16=(DEVICE.type == "cuda"),
            use_karras=USE_KARRAS,
            karras_steps=KARRAS_STEPS,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            s_churn=S_CHURN,
        )

        latent = latents[0]

        mesh = decode_latent_mesh(xm, latent).tri_mesh()
        mesh = rotate_mesh_vertices(mesh, ASSET_PRE_ROT_EULER_XYZ_DEG)

        obj_path = os.path.join(obj_dir, f"object_{obj_id:02d}.obj")
        ply_path = os.path.join(obj_dir, f"object_{obj_id:02d}.ply")

        with open(obj_path, "w", encoding="utf-8") as f:
            mesh.write_obj(f)

        with open(ply_path, "wb") as f:
            mesh.write_ply(f)

        meta = {
            "id": obj_id,
            "class_name": label,
            "input_image_path": image_path,
            "obj_path": obj_path,
            "ply_path": ply_path,
            "guidance_scale": GUIDANCE_SCALE,
            "karras_steps": KARRAS_STEPS,
            "device": str(DEVICE),
            "model_name": IMAGE_MODEL_NAME,
            "transmitter_name": TRANSMITTER_NAME,
            "asset_pre_rotation_euler_xyz_deg": ASSET_PRE_ROT_EULER_XYZ_DEG,
        }

        meta_path = os.path.join(obj_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  saved obj -> {obj_path}")
        print(f"  saved ply -> {ply_path}")
        print(f"  saved meta -> {meta_path}")

        all_results.append(meta)

    out_json_path = os.path.join(OUTPUT_DIR, "gen3d_result.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "input_json_path": INPUT_JSON_PATH,
            "device": str(DEVICE),
            "num_objects": len(all_results),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    print("\n[SAVE]", out_json_path)


if __name__ == "__main__":
    main()