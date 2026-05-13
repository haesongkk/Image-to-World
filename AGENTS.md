# AGENTS.md

## Goal
- Shared working rules for contributors/agents in this repository.

## Submodule Policy (Important)
- `third_party/*` is managed by Git submodules.
- Do not `git clone` third-party repos manually into `third_party`.
- Always sync/init with:
  - `git submodule sync --recursive`
  - `git submodule update --init --recursive`
- To refresh all third-party repos cleanly:
  - `git submodule deinit -f --all`
  - `git submodule update --init --recursive --force`

## Project Structure
- Entry: `run.py`
- Pipeline: `src/pipeline.py`
- Stages:
  - `src/stage/segmentation.py`
  - `src/stage/generation.py`
  - `src/stage/placement.py`
- Shared config: `src/config.py`
- External process runner: `src/external/runner.py`
- Preflight checks: `src/preflight.py`
- Manifest writer: `src/manifest.py`
- DAG helpers: `src/pipeline_dag.py`
- Tests: `tests/`

## Runtime Commands
- Full run: `python run.py`
- Single stage:
  - `python run.py --stage segmentation`
  - `python run.py --stage generation`
  - `python run.py --stage placement`
- Resume mode: `python run.py --resume`

## Coding Rules
- Stage functions should return `StageResult`.
- For external tools, use `run_external_command` instead of direct `subprocess.run`.
- Use path constants from `src/config.py` (avoid repeated hardcoded path construction).
- Keep pipeline orchestration in `src/pipeline.py` and dependency/output checks in `src/pipeline_dag.py`.

## Logging and Artifacts
- External process stdout/stderr logs are saved under each module output directory.
- Run metadata is saved to `output/run_manifest.json`.
- New stages should report key outputs via `StageResult.outputs`.

## Test Rules
- Minimum check after edits:
  - `python -m unittest discover -s tests -p "test_*.py"`
- For new behavior, add at least one of:
  - unit test (logic)
  - lightweight integration test (mocked external runner)
  - regression test (artifact schema/output existence)

## Cautions
- Do not change third-party CLI arguments lightly.
- Do not rename output artifacts without updating dependent code.
- Keep heavyweight model runs out of default test paths.
