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
        "def apply_xform(v, pre_rot, scale_xyz, rot_xyz, pos_xyz):\n"
        "  pre = rot_xyz_deg(pre_rot)\n"
        "  v = (pre @ v.T).T\n"
        "  rot = rot_xyz_deg(rot_xyz)\n"
        "  scale = np.diag([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])])\n"
        "  m = np.eye(4, dtype=np.float64)\n"
        "  m[:3,:3] = rot @ scale\n"
        "  m[:3,3] = np.array(pos_xyz, dtype=np.float64)\n"
        "  homo = np.concatenate([v.astype(np.float64), np.ones((v.shape[0],1), dtype=np.float64)], axis=1)\n"
        "  return (m @ homo.T).T[:, :3]\n"
        "meshes = []\n"
        "placed, skipped = 0, 0\n"
        "for rec in records:\n"
        "  pre_rot = rec.get('global_pre_rot_euler_deg', [0.0,0.0,0.0])\n"
        "  scale_xyz = rec.get('scale_xyz', [1.0,1.0,1.0])\n"
        "  rot_xyz = rec.get('rotation_euler_xyz_deg', [0.0,0.0,0.0])\n"
        "  pos_xyz = rec.get('position_xyz', [0.0,0.0,0.0])\n"
        "  normalize = bool(rec.get('normalize_mesh_to_unit_box', False))\n"
        "  kind = rec.get('kind')\n"
        "  mesh = None\n"
        "  if kind == 'mesh':\n"
        "    p = Path(rec.get('mesh_path', ''))\n"
        "    if not p.exists():\n"
        "      skipped += 1\n"
        "      continue\n"
        "    scene_or_mesh = trimesh.load(str(p), force='scene')\n"
        "    if isinstance(scene_or_mesh, trimesh.Scene):\n"
        "      mesh = scene_or_mesh.dump(concatenate=True)\n"
        "    else:\n"
        "      mesh = scene_or_mesh\n"
        "  elif kind == 'cube':\n"
        "    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])\n"
        "  else:\n"
        "    skipped += 1\n"
        "    continue\n"
        "  v = np.asarray(mesh.vertices, dtype=np.float64)\n"
        "  if v.size == 0:\n"
        "    skipped += 1\n"
        "    continue\n"
        "  if normalize and kind == 'mesh':\n"
        "    vmin = v.min(axis=0)\n"
        "    vmax = v.max(axis=0)\n"
        "    center = (vmin + vmax) * 0.5\n"
        "    size = np.maximum(vmax - vmin, 1e-8)\n"
        "    max_extent = float(size.max())\n"
        "    v = (v - center) / max_extent\n"
        "  v = apply_xform(v, pre_rot, scale_xyz, rot_xyz, pos_xyz)\n"
        "  mesh.vertices = v\n"
        "  meshes.append(mesh)\n"
        "  placed += 1\n"
        "if not meshes:\n"
        "  raise RuntimeError('No meshes to assemble')\n"
        "merged = trimesh.util.concatenate(meshes)\n"
        "output_glb.parent.mkdir(parents=True, exist_ok=True)\n"
        "merged.export(str(output_glb))\n"
        "bmin, bmax = merged.bounds\n"
        "center = ((bmin + bmax) * 0.5).tolist()\n"
        "summary = {\n"
        "  'output_glb_path': str(output_glb),\n"
        "  'num_assembled_objects': int(placed),\n"
        "  'num_skipped_objects': int(skipped),\n"
        "  'num_vertices': int(len(merged.vertices)),\n"
        "  'num_faces': int(len(merged.faces)),\n"
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
