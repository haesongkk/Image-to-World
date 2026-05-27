from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from src.config import (
    CAMERA_ESTIMATION_OUTPUT_DIR,
    CROPS_GENERATION_OUTPUT_DIR,
    DEPTH_ESTIMATION_OUTPUT_DIR,
    FITTED_TRANSFORM_OUTPUT_DIR,
    INSTANCE_SEGMENTATION_OUTPUT_DIR,
    MASK_POSTPROCESS_OUTPUT_DIR,
    MESH_GENERATION_OUTPUT_DIR,
    MESH_REMESH_OUTPUT_DIR,
    MESH_TEXTURING_OUTPUT_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    PROMPTING_OUTPUT_DIR,
    SCENE_ASSEMBLY_OUTPUT_DIR,
    SCENE_PRECOMPUTE_OUTPUT_DIR,
    raw_image_path,
)
from src.manifest import write_run_manifest
from src.pipeline_dag import stage_deps_ready, stage_output_ready
from src.pipeline_types import StageResult
from src.preflight import run_preflight
from src.stage.camera_estimation import run_camera_estimation
from src.stage.crops_generation import run_crops_generation
from src.stage.depth_estimation import run_depth_estimation
from src.stage.fitted_transform import run_fitted_transform
from src.stage.instance_segmentation import run_instance_segmentation
from src.stage.mask_postprocess import run_mask_postprocess
from src.stage.mesh_generation import run_mesh_generation
from src.stage.mesh_remesh import run_mesh_remesh
from src.stage.mesh_texturing import run_mesh_texturing
from src.stage.prompting import run_prompting
from src.stage.scene_assembly import run_scene_assembly
from src.stage.scene_precompute import run_scene_precompute

STAGES = (
    "prompting",
    "instance_segmentation",
    "crops_generation",
    "mask_postprocess",
    "mesh_generation",
    "mesh_remesh",
    "mesh_texturing",
    "depth_estimation",
    "camera_estimation",
    "scene_precompute",
    "fitted_transform",
    "scene_assembly",
)

def _stage_dependencies(input_image: Path) -> dict[str, list[Path]]:
    return {
        "prompting": [input_image],
        "instance_segmentation": [input_image, PROMPTING_OUTPUT_DIR / "text_prompt.txt"],
        "crops_generation": [INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"],
        "mask_postprocess": [input_image, INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"],
        "mesh_generation": [CROPS_GENERATION_OUTPUT_DIR / "crops"],
        "mesh_remesh": [CROPS_GENERATION_OUTPUT_DIR / "crops"],
        "mesh_texturing": [CROPS_GENERATION_OUTPUT_DIR / "crops", MESH_REMESH_OUTPUT_DIR],
        "depth_estimation": [input_image],
        "camera_estimation": [input_image],
        "scene_precompute": [
            input_image,
            MASK_POSTPROCESS_OUTPUT_DIR / "mask_viz.png",
            DEPTH_ESTIMATION_OUTPUT_DIR / f"{input_image.stem}.npz",
            CAMERA_ESTIMATION_OUTPUT_DIR / f"{input_image.stem}_perspective_fields.json",
        ],
        "fitted_transform": [
            input_image,
            MESH_GENERATION_OUTPUT_DIR,
            MASK_POSTPROCESS_OUTPUT_DIR / "mask_viz.png",
            SCENE_PRECOMPUTE_OUTPUT_DIR / "raw_transform.json",
            CAMERA_ESTIMATION_OUTPUT_DIR / f"{input_image.stem}_perspective_fields.json",
        ],
        "scene_assembly": [
            input_image,
            MESH_REMESH_OUTPUT_DIR,
            SCENE_PRECOMPUTE_OUTPUT_DIR / "raw_transform.json",
        ],
    }


def _stage_output_candidates(input_image: Path) -> dict[str, list[Path]]:
    return {
        "prompting": [PROMPTING_OUTPUT_DIR / "text_prompt.txt"],
        "instance_segmentation": [INSTANCE_SEGMENTATION_OUTPUT_DIR / "grounded_sam2_hf_model_demo_results.json"],
        "crops_generation": [CROPS_GENERATION_OUTPUT_DIR / "crops"],
        "mask_postprocess": [MASK_POSTPROCESS_OUTPUT_DIR / "mask_viz.png"],
        "mesh_generation": [MESH_GENERATION_OUTPUT_DIR],
        "mesh_remesh": [MESH_REMESH_OUTPUT_DIR],
        "mesh_texturing": [MESH_TEXTURING_OUTPUT_DIR],
        "depth_estimation": [DEPTH_ESTIMATION_OUTPUT_DIR / f"{input_image.stem}.npz"],
        "camera_estimation": [CAMERA_ESTIMATION_OUTPUT_DIR / f"{input_image.stem}_perspective_fields.json"],
        "scene_precompute": [SCENE_PRECOMPUTE_OUTPUT_DIR / "raw_transform.json"],
        "fitted_transform": [FITTED_TRANSFORM_OUTPUT_DIR / "fitted_transform.json"],
        "scene_assembly": [SCENE_ASSEMBLY_OUTPUT_DIR / f"{input_image.stem}_assembled.glb"],
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stage(stage: str, input_image: Path) -> StageResult:
    start = time.perf_counter()
    if stage == "prompting":
        result = run_prompting(input_image)
    elif stage == "instance_segmentation":
        result = run_instance_segmentation(input_image)
    elif stage == "crops_generation":
        result = run_crops_generation()
    elif stage == "mask_postprocess":
        result = run_mask_postprocess(input_image)
    elif stage == "mesh_generation":
        result = run_mesh_generation()
    elif stage == "mesh_remesh":
        result = run_mesh_remesh()
    elif stage == "mesh_texturing":
        result = run_mesh_texturing()
    elif stage == "depth_estimation":
        result = run_depth_estimation(input_image)
    elif stage == "camera_estimation":
        result = run_camera_estimation(input_image)
    elif stage == "scene_precompute":
        result = run_scene_precompute(input_image)
    elif stage == "fitted_transform":
        result = run_fitted_transform(input_image)
    elif stage == "scene_assembly":
        result = run_scene_assembly(input_image)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    result.duration_sec = time.perf_counter() - start
    print(f"{stage.capitalize()} stage finished successfully in {result.duration_sec:.2f}s.")
    return result


def run_pipeline(stage: str = "all", resume: bool = False, image_path: str | Path | None = None) -> list[StageResult]:
    input_image = raw_image_path(image_path)
    run_preflight(stage, input_image)
    stage_dependencies = _stage_dependencies(input_image)
    stage_outputs = _stage_output_candidates(input_image)
    run_started = _utc_now_iso()
    stage_results: list[StageResult] = []

    if stage == "all":
        target_stages = STAGES
    elif stage in STAGES:
        target_stages = (stage,)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    current_stage = None
    try:
        for stage_name in target_stages:
            current_stage = stage_name
            deps_ready, missing = stage_deps_ready(stage_name, stage_dependencies)
            if not deps_ready:
                raise RuntimeError(f"Dependencies not ready for {stage_name}: {missing}")

            if resume and stage_output_ready(stage_name, stage_outputs):
                print(f"Skipping {stage_name}: outputs already exist.")
                stage_results.append(StageResult(stage=stage_name, skipped=True))
                continue

            stage_results.append(_run_stage(stage_name, input_image))
    except Exception as e:
        write_run_manifest(
            output_path=OUTPUT_DIR / "run_manifest.json",
            input_image=input_image,
            stage_results=stage_results,
            stage=stage,
            resume=resume,
            project_root=PROJECT_ROOT,
            started_at=run_started,
            failed_stage=current_stage,
            error_message=str(e),
        )
        raise

    write_run_manifest(
        output_path=OUTPUT_DIR / "run_manifest.json",
        input_image=input_image,
        stage_results=stage_results,
        stage=stage,
        resume=resume,
        project_root=PROJECT_ROOT,
        started_at=run_started,
    )
    return stage_results
