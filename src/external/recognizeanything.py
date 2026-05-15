import os
from pathlib import Path

from src.config import PROMPTING_OUTPUT_DIR, THIRD_PARTY_DIR
from src.external.runner import run_external_command


def run_recognizeanything(image_path: Path):
    repo_root = THIRD_PARTY_DIR / "recognize-anything"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script = repo_root / "inference_ram_plus.py"

    input_image_path = image_path
    output_dir = PROMPTING_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "stdout.txt"
    text_prompt_path = output_dir / "text_prompt.txt"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    result = run_external_command(
        name="recognizeanything",
        command=[
            str(venv_python),
            str(inference_script),
            "--image",
            str(input_image_path),
        ],
        cwd=repo_root,
        log_dir=output_dir,
        env=env,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    tags = []
    for line in result.stdout.splitlines():
        if line.strip().startswith("Image Tags:"):
            _, _, raw_tags = line.partition(":")
            tags = [tag.strip() for tag in raw_tags.split("|") if tag.strip()]
            break
    text_prompt = ". ".join(tags)
    with open(text_prompt_path, "w", encoding="utf-8") as f:
        f.write(text_prompt)
    
