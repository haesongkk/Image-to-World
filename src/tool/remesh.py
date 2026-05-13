from pathlib import Path

import pymeshlab
import trimesh

def make_remesh_glb(output_name: str):
    project_root = Path(__file__).resolve().parent.parent.parent
    mesh_input_dir = project_root / "output" / "Hunyuan3D-2"
    remesh_output_dir = project_root / "output" / "remesh"
    remesh_output_dir.mkdir(parents=True, exist_ok=True)

    src_mesh_path = mesh_input_dir / f"{output_name}_shape_mesh.glb"
    dst_mesh_path = remesh_output_dir / f"{output_name}_remeshed.glb"

    src_obj_path = remesh_output_dir / f"{output_name}_source.obj"
    temp_obj_path = remesh_output_dir / f"{output_name}_remeshed.obj"

    src_scene = trimesh.load(str(src_mesh_path), force="scene")
    src_scene.export(str(src_obj_path))

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src_obj_path))

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=5000,
        preservenormal=True,
        preservetopology=True,
        qualitythr=1.0,
    )
    ms.save_current_mesh(str(temp_obj_path), save_textures=False)

    remeshed_scene = trimesh.load(str(temp_obj_path), force="scene")
    remeshed_scene.export(str(dst_mesh_path))

    print(f"[DONE] {dst_mesh_path}")
