from __future__ import annotations

import json
import subprocess
from pathlib import Path

from image_to_world.config import MeshGenerationConfig


class HunyuanExternalGenerator:
    def __init__(self, config: MeshGenerationConfig) -> None:
        self.config = config

    def _build_inline_script(self) -> str:
        # Run directly from the Hunyuan3D-2 repo in its own virtual environment.
        # The generated mesh is saved as-is with no geometry postprocessing.
        return (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from PIL import Image\n"
            "repo = Path(sys.argv[1])\n"
            "image_path = Path(sys.argv[2])\n"
            "shape_output_path = Path(sys.argv[3])\n"
            "shape_model_id = sys.argv[4]\n"
            "use_rembg = sys.argv[5] == '1'\n"
            "enable_texture = sys.argv[6] == '1'\n"
            "texture_model_id = sys.argv[7]\n"
            "texture_output_path = Path(sys.argv[8])\n"
            "sys.path.insert(0, str(repo))\n"
            "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline\n"
            "image = Image.open(image_path).convert('RGBA')\n"
            "if use_rembg:\n"
            "    from hy3dgen.rembg import BackgroundRemover\n"
            "    image = BackgroundRemover()(image)\n"
            "pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(shape_model_id)\n"
            "mesh = pipeline(image=image)[0]\n"
            "shape_output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "mesh.export(str(shape_output_path))\n"
            "result = {\n"
            "    'shape_mesh_path': str(shape_output_path),\n"
            "    'shape_mesh_format': shape_output_path.suffix.lstrip('.').lower(),\n"
            "    'mesh_path': str(shape_output_path),\n"
            "    'mesh_format': shape_output_path.suffix.lstrip('.').lower(),\n"
            "}\n"
            "if enable_texture:\n"
            "    from hy3dgen.texgen import Hunyuan3DPaintPipeline\n"
            "    paint = Hunyuan3DPaintPipeline.from_pretrained(texture_model_id)\n"
            "    textured_mesh = paint(mesh, image=image)\n"
            "    texture_output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    textured_mesh.export(str(texture_output_path))\n"
            "    result['textured_mesh_path'] = str(texture_output_path)\n"
            "    result['textured_mesh_format'] = texture_output_path.suffix.lstrip('.').lower()\n"
            "    result['mesh_path'] = str(texture_output_path)\n"
            "    result['mesh_format'] = texture_output_path.suffix.lstrip('.').lower()\n"
            "print(json.dumps(result))\n"
        )

    def generate_mesh(self, *, image_path: Path, shape_output_path: Path, texture_output_path: Path) -> dict[str, str]:
        repo_dir = self.config.hunyuan_repo_dir
        python_path = self.config.hunyuan_venv_python
        if not python_path.exists():
            raise FileNotFoundError(f"Hunyuan venv python not found: {python_path}")
        if not repo_dir.exists():
            raise FileNotFoundError(f"Hunyuan repo dir not found: {repo_dir}")
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        command = [
            str(python_path),
            "-c",
            self._build_inline_script(),
            str(repo_dir),
            str(image_path),
            str(shape_output_path),
            self.config.hunyuan_model_id,
            "1" if self.config.hunyuan_use_background_removal else "0",
            "1" if self.config.hunyuan_enable_texture else "0",
            self.config.hunyuan_texture_model_id,
            str(texture_output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(repo_dir),
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            raise RuntimeError(f"Hunyuan execution failed. stdout={stdout!r} stderr={stderr!r}") from exc
        stdout = (completed.stdout or "").strip()
        if not stdout:
            raise RuntimeError("Hunyuan command did not return output metadata")
        last_line = stdout.splitlines()[-1]
        payload = json.loads(last_line)
        mesh_path = payload.get("mesh_path")
        mesh_format = payload.get("mesh_format")
        if not mesh_path or not mesh_format:
            raise RuntimeError(f"Invalid Hunyuan output payload: {payload}")
        return payload
