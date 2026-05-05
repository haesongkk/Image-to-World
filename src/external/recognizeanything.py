import os
import subprocess
from pathlib import Path

def run_recognizeanything(image_path: Path):
    project_root = Path(__file__).resolve().parent.parent.parent

    repo_root = project_root / "third_party" / "recognize-anything"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    inference_script = repo_root / "inference_ram_plus.py"

    input_image_path = project_root / "output" / "BirefNet" / f"{image_path.stem}_birefnet.png"
    output_dir  = project_root / "output" / "recognize-anything" 
    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "stdout.txt"
    text_prompt_path = output_dir / "text_prompt.txt"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            str(venv_python),
            str(inference_script),
            "--image", str(input_image_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(repo_root),
        env=env,
    )

    if(result.returncode != 0):
        raise RuntimeError("RAM inference failed..\n" + result.stderr)
        
    
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
    
