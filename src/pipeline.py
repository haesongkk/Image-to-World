from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from src.config import OUTPUT_DIR, PROJECT_ROOT, raw_image_path
from src.manifest import write_run_manifest
from src.pipeline_dag import stage_deps_ready, stage_output_ready
from src.pipeline_types import StageResult
from src.preflight import run_preflight
from src.stage.generation import run_generation
from src.stage.placement import run_placement
from src.stage.segmentation import run_segmentation

STAGES = ("segmentation", "generation", "placement")

def _stage_dependencies(input_image: Path) -> dict[str, list[Path]]:
    return {
        "segmentation": [input_image],
        "generation": [OUTPUT_DIR / "Grounded-SAM-2" / "crops"],
        "placement": [
            input_image,
            OUTPUT_DIR / "mask" / "mask_viz.png",
            OUTPUT_DIR / "Grounded-SAM-2" / "crops",
        ],
    }


def _stage_output_candidates(input_image: Path) -> dict[str, list[Path]]:
    return {
        "segmentation": [OUTPUT_DIR / "mask" / "mask_viz.png"],
        "generation": [
            OUTPUT_DIR / "remesh",
            OUTPUT_DIR / "Hunyuan3D-2",
        ],
        "placement": [OUTPUT_DIR / "scene" / f"{input_image.stem}_assembled.glb"],
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stage(stage: str, input_image: Path) -> StageResult:
    start = time.perf_counter()
    if stage == "segmentation":
        result = run_segmentation(input_image)
    elif stage == "generation":
        result = run_generation()
    elif stage == "placement":
        result = run_placement(input_image)
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
