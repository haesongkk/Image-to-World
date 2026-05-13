import argparse
import sys
from pathlib import Path

import trimesh
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
HY3D_ROOT = BASE_DIR / "third_party" / "Hunyuan3D-2"
sys.path.insert(0, str(HY3D_ROOT))

from hy3dgen.texgen import Hunyuan3DPaintPipeline


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No mesh geometry found: {mesh_path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise RuntimeError(f"Unsupported mesh type from {mesh_path}: {type(loaded)}")


parser = argparse.ArgumentParser(description="Texture remeshed mesh with Hunyuan3D-Paint")
parser.add_argument("--image-path", required=True)
parser.add_argument("--mesh-path", required=True)
parser.add_argument("--output-path", required=True)
parser.add_argument("--paint-model", default="tencent/Hunyuan3D-2.1")
parser.add_argument("--paint-subfolder", default="hunyuan3d-paintpbr-v2-1")
args = parser.parse_args()

image_path = Path(args.image_path)
mesh_path = Path(args.mesh_path)
output_path = Path(args.output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

image = Image.open(image_path).convert("RGBA")
mesh = load_mesh(mesh_path)

paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    args.paint_model,
    subfolder=args.paint_subfolder,
)
textured_mesh = paint_pipeline(mesh, image=image)
textured_mesh.export(output_path)
