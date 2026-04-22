from __future__ import annotations

import json
import subprocess
from pathlib import Path


def assemble_scene_glb_external(
    *,
    records: list[dict],
    output_glb_path: Path,
    python_path: Path,
    repo_dir: Path,
) -> dict:
    if not python_path.exists():
        raise FileNotFoundError(f"GLB assembler python not found: {python_path}")
    if not repo_dir.exists():
        raise FileNotFoundError(f"GLB assembler repo dir not found: {repo_dir}")

    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = output_glb_path.with_name("assembled_scene_inputs.json")
    summary_path = output_glb_path.with_name("assembled_scene_external_summary.json")
    records_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    script = (
        "import json, math, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import trimesh\n"
        "records = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "output_glb = Path(sys.argv[2])\n"
        "summary_path = Path(sys.argv[3])\n"
        "def rot_xyz_deg(e):\n"
        "  rx, ry, rz = [float(v) * math.pi / 180.0 for v in e]\n"
        "  cx, sx = math.cos(rx), math.sin(rx)\n"
        "  cy, sy = math.cos(ry), math.sin(ry)\n"
        "  cz, sz = math.cos(rz), math.sin(rz)\n"
        "  rx_m = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)\n"
        "  ry_m = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64)\n"
        "  rz_m = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)\n"
        "  return rz_m @ ry_m @ rx_m\n"
        "def compose_xform(pre_rot, scale_xyz, rot_xyz, pos_xyz):\n"
        "  pre = rot_xyz_deg(pre_rot)\n"
        "  rot = rot_xyz_deg(rot_xyz)\n"
        "  m = np.eye(4, dtype=np.float64)\n"
        "  scale = np.diag([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])])\n"
        "  m[:3,:3] = (rot @ scale) @ pre\n"
        "  m[:3,3] = np.array(pos_xyz, dtype=np.float64)\n"
        "  return m\n"
        "def normalize_matrix_from_points(points):\n"
        "  vmin = points.min(axis=0)\n"
        "  vmax = points.max(axis=0)\n"
        "  center = (vmin + vmax) * 0.5\n"
        "  size = np.maximum(vmax - vmin, 1e-8)\n"
        "  max_extent = float(size.max())\n"
        "  n = np.eye(4, dtype=np.float64)\n"
        "  n[:3,:3] *= (1.0 / max_extent)\n"
        "  n[:3,3] = -center / max_extent\n"
        "  return n\n"
        "scene_out = trimesh.Scene()\n"
        "root_node = 'assembled_root'\n"
        "scene_out.graph.update(frame_from=scene_out.graph.base_frame, frame_to=root_node, matrix=np.eye(4, dtype=np.float64))\n"
        "placed, skipped = 0, 0\n"
        "vertex_sum, face_sum = 0, 0\n"
        "for rec in records:\n"
        "  obj_idx = int(rec.get('placement_id') if rec.get('placement_id') is not None else placed)\n"
        "  class_name = str(rec.get('class_name') or f'object_{obj_idx:02d}').replace(' ', '_')\n"
        "  object_node = f'object_{obj_idx:02d}_{class_name}'\n"
        "  scene_out.graph.update(frame_from=root_node, frame_to=object_node, matrix=np.eye(4, dtype=np.float64))\n"
        "  pre_rot = rec.get('global_pre_rot_euler_deg', [0.0,0.0,0.0])\n"
        "  scale_xyz = rec.get('scale_xyz', [1.0,1.0,1.0])\n"
        "  rot_xyz = rec.get('rotation_euler_xyz_deg', [0.0,0.0,0.0])\n"
        "  pos_xyz = rec.get('position_xyz', [0.0,0.0,0.0])\n"
        "  normalize = bool(rec.get('normalize_mesh_to_unit_box', False))\n"
        "  kind = rec.get('kind')\n"
        "  object_xform = compose_xform(pre_rot, scale_xyz, rot_xyz, pos_xyz)\n"
        "  object_norm = np.eye(4, dtype=np.float64)\n"
        "  if kind == 'mesh':\n"
        "    p = Path(rec.get('mesh_path', ''))\n"
        "    if not p.exists():\n"
        "      skipped += 1\n"
        "      continue\n"
        "    loaded = trimesh.load(str(p), force='scene')\n"
        "    if isinstance(loaded, trimesh.Scene):\n"
        "      node_geoms = list(loaded.graph.nodes_geometry)\n"
        "      if not node_geoms:\n"
        "        skipped += 1\n"
        "        continue\n"
        "      if normalize:\n"
        "        pts = []\n"
        "        for n in node_geoms:\n"
        "          t_local, g_name = loaded.graph.get(n)\n"
        "          g = loaded.geometry[g_name]\n"
        "          v = np.asarray(g.vertices, dtype=np.float64)\n"
        "          if v.size == 0:\n"
        "            continue\n"
        "          h = np.concatenate([v, np.ones((v.shape[0],1), dtype=np.float64)], axis=1)\n"
        "          pts.append((t_local @ h.T).T[:, :3])\n"
        "        if pts:\n"
        "          object_norm = normalize_matrix_from_points(np.concatenate(pts, axis=0))\n"
        "      part_count = 0\n"
        "      for n in node_geoms:\n"
        "        t_local, g_name = loaded.graph.get(n)\n"
        "        geom = loaded.geometry[g_name].copy()\n"
        "        v = np.asarray(geom.vertices, dtype=np.float64)\n"
        "        if v.size == 0:\n"
        "          continue\n"
        "        node_xform = object_xform @ object_norm @ t_local\n"
        "        node_name = f'{object_node}_part_{part_count:03d}'\n"
        "        geom_name = f'{object_node}_geom_{part_count:03d}'\n"
        "        scene_out.add_geometry(geom, node_name=node_name, geom_name=geom_name, parent_node_name=object_node, transform=node_xform)\n"
        "        vertex_sum += int(len(geom.vertices))\n"
        "        face_sum += int(len(geom.faces))\n"
        "        part_count += 1\n"
        "      if part_count == 0:\n"
        "        skipped += 1\n"
        "        continue\n"
        "      placed += 1\n"
        "      continue\n"
        "    mesh = loaded.copy()\n"
        "    v = np.asarray(mesh.vertices, dtype=np.float64)\n"
        "    if v.size == 0:\n"
        "      skipped += 1\n"
        "      continue\n"
        "    if normalize:\n"
        "      object_norm = normalize_matrix_from_points(v)\n"
        "    node_xform = object_xform @ object_norm\n"
        "    scene_out.add_geometry(mesh, node_name=f'{object_node}_part_000', geom_name=f'{object_node}_geom_000', parent_node_name=object_node, transform=node_xform)\n"
        "    vertex_sum += int(len(mesh.vertices))\n"
        "    face_sum += int(len(mesh.faces))\n"
        "    placed += 1\n"
        "    continue\n"
        "  elif kind == 'cube':\n"
        "    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])\n"
        "    scene_out.add_geometry(cube, node_name=f'{object_node}_part_000', geom_name=f'{object_node}_geom_000', parent_node_name=object_node, transform=object_xform)\n"
        "    vertex_sum += int(len(cube.vertices))\n"
        "    face_sum += int(len(cube.faces))\n"
        "    placed += 1\n"
        "    continue\n"
        "  else:\n"
        "    skipped += 1\n"
        "    continue\n"
        "if placed == 0:\n"
        "  raise RuntimeError('No meshes to assemble')\n"
        "output_glb.parent.mkdir(parents=True, exist_ok=True)\n"
        "scene_out.export(str(output_glb))\n"
        "bmin, bmax = scene_out.bounds\n"
        "center = ((bmin + bmax) * 0.5).tolist()\n"
        "summary = {\n"
        "  'output_glb_path': str(output_glb),\n"
        "  'num_assembled_objects': int(placed),\n"
        "  'num_skipped_objects': int(skipped),\n"
        "  'num_vertices': int(vertex_sum),\n"
        "  'num_faces': int(face_sum),\n"
        "  'scene_root_node': root_node,\n"
        "  'scene_bounds': {\n"
        "    'bbox_min': np.asarray(bmin, dtype=np.float64).tolist(),\n"
        "    'bbox_max': np.asarray(bmax, dtype=np.float64).tolist(),\n"
        "    'center': center,\n"
        "    'size_xyz': (np.asarray(bmax, dtype=np.float64) - np.asarray(bmin, dtype=np.float64)).tolist(),\n"
        "  },\n"
        "}\n"
        "summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding='utf-8')\n"
        "print(json.dumps(summary, ensure_ascii=False))\n"
    )
    command = [
        str(python_path),
        "-c",
        script,
        str(records_path),
        str(output_glb_path),
        str(summary_path),
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
        raise RuntimeError(f"External GLB assembly failed. stdout={stdout!r} stderr={stderr!r}") from exc
    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise RuntimeError("External GLB assembly returned no metadata")
    payload = json.loads(stdout.splitlines()[-1])
    if not output_glb_path.exists():
        raise RuntimeError(f"Assembled GLB not found: {output_glb_path}")
    return payload
