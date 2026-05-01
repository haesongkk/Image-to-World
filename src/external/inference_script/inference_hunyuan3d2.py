import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
HY3D_ROOT = BASE_DIR / "third_party" / "Hunyuan3D-2"
sys.path.insert(0, str(HY3D_ROOT))

import torch
from PIL import Image
import argparse

# from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

parser = argparse.ArgumentParser(description="Inference script for Hunyuan3D-2")
parser.add_argument('--image-path', required=True)
parser.add_argument('--output-dir', required=True)

args = parser.parse_args()
image_path = Path(args.image_path)
output_dir = Path(args.output_dir)
output_name = image_path.stem
output_dir.mkdir(parents=True, exist_ok=True)

shape_mesh_path = output_dir / f"{output_name}_shape_mesh.glb"
paint_mesh_path = output_dir / f"{output_name}_final_textured_mesh.glb"

image = Image.open(image_path).convert("RGBA")
# rembg = BackgroundRemover()
# image = rembg(image)

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini',
    variant='fp16'
)
paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

mesh = shape_pipeline(
    image=image,
    num_inference_steps=50,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]
mesh.export(shape_mesh_path)

mesh = paint_pipeline(mesh, image=image)
mesh.export(paint_mesh_path)
