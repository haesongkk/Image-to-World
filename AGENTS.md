# AGENTS.md

## Project Structure
- Entry point: `run.py`
- Core source: `src/`
  - Pipeline orchestration: `src/pipeline.py`
  - Stage modules: `src/stage/`
    - `src/stage/prompting.py`
    - `src/stage/instance_segmentation.py`
    - `src/stage/mask_postprocess.py`
    - `src/stage/mesh_generation.py`
    - `src/stage/mesh_remesh.py`
    - `src/stage/mesh_texturing.py`
    - `src/stage/depth_estimation.py`
    - `src/stage/camera_estimation.py`
    - `src/stage/scene_precompute.py`
    - `src/stage/scene_assembly.py`
  - External integrations: `src/external/`
    - Command runner: `src/external/runner.py`
    - Tool wrappers: model/tool-specific modules (e.g. GroundedSAM2, BiRefNet, Hunyuan3D, DepthPro)
  - Geometry/utility tools: `src/tool/`
  - Shared config: `src/config.py`
  - Preflight checks: `src/preflight.py`
  - Pipeline DAG/helpers: `src/pipeline_dag.py`
  - Shared types: `src/pipeline_types.py`
  - Manifest writer: `src/manifest.py`
- Third-party submodules: `third_party/`
- Runtime/artifact directories:
  - Input/reference assets: `data/`, `references/`
  - Outputs: `output/`, `output0/`, `output1/`
  - Docs and debug snapshots: `doc/`

## Runtime Commands
- Full run: `python run.py`
- Single stage:
  - `python run.py --stage prompting`
  - `python run.py --stage instance_segmentation`
  - `python run.py --stage mask_postprocess`
  - `python run.py --stage mesh_generation`
  - `python run.py --stage mesh_remesh`
  - `python run.py --stage mesh_texturing`
  - `python run.py --stage depth_estimation`
  - `python run.py --stage camera_estimation`
  - `python run.py --stage scene_precompute`
  - `python run.py --stage scene_assembly`
- Resume mode: `python run.py --resume`

## Coding Rules
- Stage functions should return `StageResult`.
- For external tools, use `run_external_command` instead of direct `subprocess.run`.
- Use path constants from `src/config.py` (avoid repeated hardcoded path construction).
- Keep pipeline orchestration in `src/pipeline.py` and dependency/output checks in `src/pipeline_dag.py`.
- Avoid modifying repositories under `third_party/*` unless absolutely necessary.

## Change Scope
- Before implementation, identify the target stage(s) and the expected impact scope in related files.
- Keep changes minimal and focused; avoid unrelated refactors in the same change.

## Logging and Artifacts
- External process stdout/stderr logs are saved under each module output directory.
- Run metadata is saved to `output/run_manifest.json`.
- New stages should report key outputs via `StageResult.outputs`.
- During implementation or modification, leave useful debug logs and intermediate visualization images so failures can be diagnosed quickly.

## Failure Handling
- If a run fails, record the executed command, key input conditions, and the failing stage.
- Keep references to relevant stdout/stderr log files for quick triage.
- Capture reproducible context (parameters, input assets, and environment differences when relevant).

## Output Contract
- Treat stage outputs as contracts for downstream stages; do not rename or remove artifacts casually.
- When output schema/paths must change, update all dependent code in the same change.
- Validate that required outputs exist and are readable before marking a stage as successful.

## Config and Secrets
- Keep shared runtime paths and constants centralized in `src/config.py`.
- Do not hardcode secrets or credentials in source files.
- Use environment variables for machine-specific or sensitive settings.

## Test Rules
- Minimum check after edits:
  - Run the related stage and the stages immediately before and after it (pre-modified-post sequence).
  - Verify the pipeline runs without crashes.
  - Verify outputs are produced correctly and have no obvious issues.

## PR Validation Checklist
- Confirm target/adjacent stage execution checks were completed (pre-modified-post sequence).
- Confirm expected artifacts were produced and visually/structurally validated.
- Confirm useful debug logs or intermediate visualizations are available for troubleshooting.

## Execution Gate (Hard Requirement)
- After any code change, the agent MUST run validation before responding.
- Minimum required validation is the pre-modified-post sequence for the affected stage.
  - Example: if `mesh_remesh` changed, run `mesh_generation -> mesh_remesh -> mesh_texturing`.
- The agent MUST NOT send a completion/final response until all required validations finish.

## Reporting Contract (Hard Requirement)
- Every completion response MUST include:
  1. Executed commands (exact)
  2. Pass/fail status per command
  3. Produced artifact paths
  4. Relevant stdout/stderr log file paths
- If any validation is skipped or fails, the response MUST start with:
  `VALIDATION INCOMPLETE` or `VALIDATION FAILED`
  and include the reason.
- Any response that lacks this Reporting Contract is considered non-compliant.
- Non-compliant responses must be corrected in the next turn before new work.
