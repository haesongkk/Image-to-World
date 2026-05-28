import os
import json
import re
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import pycocotools.mask as mask_util
from src.config import (
    AMODAL_COMPLETION_OUTPUT_DIR,
    CROPS_GENERATION_OUTPUT_DIR,
    INSTANCE_SEGMENTATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
    PROMPTING_OUTPUT_DIR,
    THIRD_PARTY_DIR,
)
from src.external.runner import run_external_command


def _clean_object_mask(mask: np.ndarray) -> np.ndarray:
    """Drop tiny islands and shrink boundary noise to reduce background leakage."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        m = (labels == largest).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size > 0:
        diag = float(np.hypot(ys.ptp() + 1, xs.ptp() + 1))
        k = max(1, int(round(diag * 0.01)))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.erode(m, kernel, iterations=1)
    return m


def _resolve_grounding_dino_local_path() -> Path:
    hf_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--IDEA-Research--grounding-dino-tiny"
    snapshots_dir = hf_root / "snapshots"
    if not snapshots_dir.exists():
        raise RuntimeError(f"Local grounding-dino cache not found: {snapshots_dir}")
    snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not snapshots:
        raise RuntimeError(f"No grounding-dino snapshots found: {snapshots_dir}")
    return snapshots[-1]


def run_groundedsam2_inference(image_path: Path):
    repo_root = THIRD_PARTY_DIR / "Grounded-SAM-2"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script = repo_root / "grounded_sam2_hf_model_demo.py"

    input_image_path = image_path
    text_prompt_path = PROMPTING_OUTPUT_DIR / "text_prompt.txt"
    if not text_prompt_path.exists():
        raise RuntimeError(f"Text prompt file not found: {text_prompt_path}")
    with open(text_prompt_path, "r", encoding="utf-8") as f:
        text_prompt = f.read().strip()

    output_dir = INSTANCE_SEGMENTATION_OUTPUT_DIR
    os.makedirs(output_dir , exist_ok=True)
    local_grounding_model = _resolve_grounding_dino_local_path()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    run_external_command(
        name="groundedsam2",
        command=[
            str(venv_python),
            str(inference_script),
            "--text-prompt",
            str(text_prompt),
            "--grounding-model",
            str(local_grounding_model),
            "--img-path",
            str(input_image_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        log_dir=output_dir,
        env=env,
    )

    results_json_path = output_dir / "grounded_sam2_hf_model_demo_results.json"
    if not results_json_path.exists():
        raise RuntimeError(f"Results json file not found: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        infer_results = json.load(f)

    source_image_path = Path(infer_results.get("image_path", str(image_path)))
    if not source_image_path.is_absolute():
        source_image_path = (repo_root / source_image_path).resolve()
    if not source_image_path.exists():
        raise RuntimeError(f"Source image file not found: {source_image_path}")

    return source_image_path, infer_results


def run_groundedsam2_crop_generation():
    results_json_path = INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"
    if not results_json_path.exists():
        raise RuntimeError(f"Results json file not found: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        infer_results = json.load(f)

    source_image_path = Path(infer_results.get("image_path", ""))
    if not source_image_path.is_absolute():
        source_image_path = source_image_path.resolve()
    if not source_image_path.exists():
        raise RuntimeError(f"Source image file not found: {source_image_path}")

    source_image = Image.open(source_image_path).convert("RGB")
    source_arr = np.array(source_image)

    output_dir = CROPS_GENERATION_OUTPUT_DIR
    crops_dir = output_dir / "crops"
    os.makedirs(crops_dir, exist_ok=True)

    annotations = infer_results.get("annotations", [])
    # Default to visible-mask crops (6126f0c behavior) to avoid background
    # leakage into object meshes. Enable amodal only when explicitly requested.
    use_amodal = os.environ.get("CROPS_USE_AMODAL", "").strip().lower() in {"1", "true", "yes", "on"}
    amodal_rgb_available = False
    amodal_mask_available = False
    if use_amodal:
        amodal_rgb_available = AMODAL_COMPLETION_OUTPUT_DIR.exists() and any(
            AMODAL_COMPLETION_OUTPUT_DIR.glob("object_*_amodal_rgb.png")
        )
        amodal_mask_available = AMODAL_COMPLETION_OUTPUT_DIR.exists() and any(
            AMODAL_COMPLETION_OUTPUT_DIR.glob("object_*_amodal_mask.npy")
        )
        if amodal_rgb_available:
            print("crops_generation: using amodal RGB crops (CROPS_USE_AMODAL=1)")
        elif amodal_mask_available:
            print("crops_generation: using amodal mask bboxes (CROPS_USE_AMODAL=1)")
        else:
            print("crops_generation: CROPS_USE_AMODAL=1 but amodal outputs missing; using visible masks")
    else:
        print("crops_generation: using visible-mask crops (default)")
    for idx, ann in enumerate(annotations):
        class_name = str(ann.get("class_name", "object")).strip() or "object"
        safe_class_name = re.sub(r"[^0-9A-Za-z_-]+", "_", class_name).strip("_") or "object"

        score_val = ann.get("score", 0.0)
        if isinstance(score_val, list):
            score_val = score_val[0] if score_val else 0.0
        try:
            score_str = f"{float(score_val):.3f}"
        except (TypeError, ValueError):
            score_str = "0.000"

        # PNG with RGBA alpha=mask so Hunyuan3D treats outside-object pixels as
        # truly transparent (cleaner foreground separation than white-fill).
        out_name = f"{idx:03d}_{safe_class_name}_{score_str}.png"
        out_path = crops_dir / out_name

        def _save_with_alpha(rgb_arr: np.ndarray, alpha_arr: np.ndarray, path: Path) -> None:
            """Save 4-channel PNG. alpha_arr same H,W as rgb_arr, uint8 in [0,255]."""
            rgba = np.concatenate([rgb_arr, alpha_arr[..., None]], axis=2)
            Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(path)

        # Skip stuff classes (floor/wall/counter/...) entirely: they're filtered
        # at scene_assembly anyway, and feeding them to Hunyuan3D wastes
        # ~5 min/object on geometry that will be discarded.
        try:
            from src.tool.stuff_filter import load_stuff_keywords, class_from_filename
            _stuff_kws = load_stuff_keywords()
            _cname_test = safe_class_name.lower()
            if any(kw in _cname_test for kw in _stuff_kws):
                print(f"crops_generation: skip stuff idx={idx} class='{_cname_test}'")
                continue
        except Exception:
            pass

        # Resolve the amodal mask (full-image coords). Used to:
        #   (a) zero out wall/counter/etc. background pixels in the crop, so
        #       Hunyuan3D doesn't bake them into the mesh as extra geometry;
        #   (b) provide a sensible bbox for Path 2/3 when amodal_rgb missing.
        amodal_mask = None
        visible_pattern = f"object_{idx:03d}_*_mask.npy"
        matching = sorted(MASK_POSTPROCESS_OUTPUT_DIR.glob(visible_pattern))
        if matching and amodal_mask_available:
            amodal_path = AMODAL_COMPLETION_OUTPUT_DIR / matching[0].name.replace(
                "_mask.npy", "_amodal_mask.npy"
            )
            if amodal_path.exists():
                amodal_mask = np.load(amodal_path)
                if amodal_mask.ndim == 3:
                    amodal_mask = amodal_mask[..., 0]
                amodal_mask = (amodal_mask > 0).astype(np.uint8)

        # Path 1: Use LaMa-inpainted amodal RGB if available.
        if amodal_rgb_available and matching:
            amodal_rgb_path = AMODAL_COMPLETION_OUTPUT_DIR / matching[0].name.replace(
                "_mask.npy", "_amodal_rgb.png"
            )
            if amodal_rgb_path.exists():
                rgb_pil = Image.open(amodal_rgb_path).convert("RGB")
                if amodal_mask is None:
                    rgb_pil.save(out_path)
                    continue
                # amodal_rgb is a PADDED BBOX CROP of source_arr at the
                # bbox of the visible mask + pad. We need the matching
                # crop of amodal_mask. Re-derive the bbox the same way
                # _inpaint_occluders did (visible mask bbox + 10% pad).
                visible_mask = np.load(matching[0])
                if visible_mask.ndim == 3:
                    visible_mask = visible_mask[..., 0]
                vys, vxs = np.where(visible_mask > 0)
                if vys.size == 0:
                    rgb_pil.save(out_path)
                    continue
                yv0, yv1 = int(vys.min()), int(vys.max())
                xv0, xv1 = int(vxs.min()), int(vxs.max())
                H, W = visible_mask.shape[:2]
                pad_h = max(1, int((yv1 - yv0 + 1) * 0.10))
                pad_w = max(1, int((xv1 - xv0 + 1) * 0.10))
                yc0 = max(0, yv0 - pad_h)
                yc1 = min(H, yv1 + 1 + pad_h)
                xc0 = max(0, xv0 - pad_w)
                xc1 = min(W, xv1 + 1 + pad_w)
                rgb_arr = np.asarray(rgb_pil, dtype=np.uint8)
                # Match the crop dims of amodal_mask to rgb_arr size.
                amodal_crop = amodal_mask[yc0:yc1, xc0:xc1]
                if amodal_crop.shape[:2] != rgb_arr.shape[:2]:
                    amodal_crop = np.array(
                        Image.fromarray(amodal_crop * 255).resize(
                            (rgb_arr.shape[1], rgb_arr.shape[0]),
                            Image.NEAREST,
                        )
                    )
                    amodal_crop = (amodal_crop > 0).astype(np.uint8)
                # White-fill the RGB outside the mask (defense in depth ??some
                # consumers ignore alpha), and write alpha=mask*255.
                white_bg = np.full_like(rgb_arr, 255)
                composed = np.where(amodal_crop[..., None] > 0, rgb_arr, white_bg)
                alpha = (amodal_crop * 255).astype(np.uint8)
                _save_with_alpha(composed, alpha, out_path)
                continue

        # Path 2/3: bbox crop on source RGB, mask non-object pixels to white.
        mask = amodal_mask
        if mask is None:
            seg = ann.get("segmentation")
            if not seg or "counts" not in seg or "size" not in seg:
                continue
            mask = mask_util.decode({"size": seg["size"], "counts": seg["counts"]})
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]

        mask = _clean_object_mask(mask)
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        crop = source_arr[y_min : y_max + 1, x_min : x_max + 1].copy()
        mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
        crop[mask_crop == 0] = 255
        alpha = (mask_crop * 255).astype(np.uint8)
        _save_with_alpha(crop, alpha, out_path)

