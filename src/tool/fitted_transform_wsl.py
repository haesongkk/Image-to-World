from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh

from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene


def _decimate_mesh(mesh: trimesh.Trimesh, target_face_count: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_face_count:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_face_count)
    except TypeError:
        return mesh.simplify_quadric_decimation(target_face_count)
    except Exception as e:
        raise RuntimeError(
            f"Mesh decimation failed. Install fast-simplification or open3d in the WSL venv. Original error: {e}"
        )


def _ensure_fit_meshes(mesh_dir: Path, target_face_count: int) -> None:
    for shape_path in sorted(mesh_dir.glob("*_shape_mesh.glb")):
        fit_path = shape_path.with_name(shape_path.name.replace("_shape_mesh.glb", "_fit_mesh.glb"))
        if fit_path.exists():
            continue
        loaded = trimesh.load(shape_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No mesh geometry in GLB: {shape_path}")
            merged = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            merged = loaded
        else:
            raise ValueError(f"Unsupported GLB load type: {type(loaded)}")
        _decimate_mesh(merged, target_face_count=target_face_count).export(fit_path)


def _resolve_glb_paths(mesh_dir: Path, prefer_fit_mesh: bool) -> list[Path]:
    if prefer_fit_mesh:
        return sorted(mesh_dir.glob("*_fit_mesh.glb")) or sorted(mesh_dir.glob("*_shape_mesh.glb")) or sorted(mesh_dir.glob("*.glb"))
    return sorted(mesh_dir.glob("*_shape_mesh.glb")) or sorted(mesh_dir.glob("*.glb"))


def _load_mesh(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry in GLB: {glb_path}")
        merged = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        merged = loaded
    else:
        raise ValueError(f"Unsupported GLB load type: {type(loaded)}")
    return np.asarray(merged.vertices, np.float32), np.asarray(merged.faces, np.int64)


def _euler_xyz_deg(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = torch.deg2rad(rx), torch.deg2rad(ry), torch.deg2rad(rz)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    rxm = torch.stack([torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]), torch.stack([torch.zeros_like(cx), cx, -sx]), torch.stack([torch.zeros_like(cx), sx, cx])])
    rym = torch.stack([torch.stack([cy, torch.zeros_like(cy), sy]), torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]), torch.stack([-sy, torch.zeros_like(cy), cy])])
    rzm = torch.stack([torch.stack([cz, -sz, torch.zeros_like(cz)]), torch.stack([sz, cz, torch.zeros_like(cz)]), torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)])])
    return rzm @ rym @ rxm


def _save_gray(path: Path, alpha: np.ndarray) -> None:
    Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1.0 - (2.0 * inter + eps) / den


def _iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else float(inter / union)


def run(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root)
    image_path = Path(args.image_path)
    mesh_dir = project_root / "output" / "Hunyuan3D-2"
    mask_dir = project_root / "output" / "mask"
    raw_transform_path = project_root / "output" / "raw_transform" / "raw_transform.npy"
    fitted_init_path = project_root / "output" / "fitted_transform" / "fitted_transform.npy"
    output_dir = project_root / "output" / "fitted_transform_debug" / image_path.stem / (args.run_name or "simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_dir = output_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    if args.auto_create_fit_mesh:
        _ensure_fit_meshes(mesh_dir, target_face_count=args.fit_face_count)
    glb_paths = _resolve_glb_paths(mesh_dir, prefer_fit_mesh=args.prefer_fit_mesh)
    mask_paths = sorted(mask_dir.glob("object_*_mask.npy"))
    raw = np.load(raw_transform_path).astype(np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    n = min(len(glb_paths), len(mask_paths), raw.shape[0])
    if n == 0:
        raise ValueError("No objects to optimize.")

    transforms = raw[:n, :9].copy()
    if args.use_fitted_init and fitted_init_path.exists():
        fi = np.load(fitted_init_path).astype(np.float32)
        if fi.ndim == 2 and fi.shape[0] >= n and fi.shape[1] >= 9:
            transforms[:] = fi[:n, :9]

    verts, faces = [], []
    for i in range(n):
        v, f = _load_mesh(glb_paths[i])
        verts.append(torch.tensor(v, dtype=torch.float32, device=device))
        faces.append(torch.tensor(f, dtype=torch.int64, device=device))

    image_size = args.image_size
    targets = []
    for i in range(n):
        m = torch.from_numpy(np.load(mask_paths[i]).astype(np.float32)).to(device)
        m = F.interpolate(m[None, None], size=(image_size, image_size), mode="nearest")[0, 0]
        targets.append(m)
    target_union = torch.clamp(torch.stack(targets).sum(0), 0.0, 1.0)

    raw_t = torch.tensor(transforms[:, 0:3], dtype=torch.float32, device=device)
    raw_s = torch.tensor(np.maximum(transforms[:, 3:6], 1e-6), dtype=torch.float32, device=device)
    raw_r = torch.tensor(transforms[:, 6:9], dtype=torch.float32, device=device)

    dt = torch.nn.Parameter(torch.zeros((n, 3), dtype=torch.float32, device=device))
    dls = torch.nn.Parameter(torch.zeros((n, 3), dtype=torch.float32, device=device))
    dr = torch.nn.Parameter(torch.zeros((n, 3), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([{"params": [dt], "lr": args.lr_t}, {"params": [dls], "lr": args.lr_s}, {"params": [dr], "lr": args.lr_r}])

    R, T = look_at_view_transform(dist=float(args.camera_dist), elev=float(args.camera_elev), azim=float(args.camera_azim), device=device)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(args.fov))

    def make_renderer(sigma: float, gamma: float, fpp: int) -> MeshRenderer:
        bp = BlendParams(sigma=sigma, gamma=gamma)
        rs = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * bp.sigma,
            faces_per_pixel=fpp,
            bin_size=16,
            max_faces_per_bin=200000,
        )
        return MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=rs), SoftSilhouetteShader(blend_params=bp))

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        p = float(step) / max(float(args.steps - 1), 1.0)
        renderer = make_renderer((1e-2 * (1.0 - p)) + (2e-4 * p), (1e-2 * (1.0 - p)) + (2e-4 * p), args.faces_pp)

        meshes, losses = [], []
        for i in range(n):
            s = raw_s[i] * torch.exp(torch.clamp(dls[i], -2.0, 2.0))
            r = raw_r[i] + torch.clamp(dr[i], -90.0, 90.0)
            t = raw_t[i] + dt[i]
            rot = _euler_xyz_deg(r[0], r[1], r[2]).to(device=device, dtype=torch.float32)
            vw = (verts[i] * s.unsqueeze(0)) @ rot.T + t.unsqueeze(0)
            mesh_i = Meshes(verts=[vw], faces=[faces[i]])
            meshes.append(mesh_i)
            alpha = torch.clamp(renderer(mesh_i)[0, ..., 3], 0.0, 1.0)
            losses.append(F.binary_cross_entropy(alpha, targets[i]) + _dice_loss(alpha, targets[i]))

        alpha_scene = torch.clamp(renderer(join_meshes_as_scene(meshes))[0, ..., 3], 0.0, 1.0)
        reg = 0.05 * (dt ** 2).mean() + 0.01 * (dls ** 2).mean() + 0.0005 * (dr ** 2).mean()
        loss = torch.stack(losses).mean() + 0.1 * (F.binary_cross_entropy(alpha_scene, target_union) + _dice_loss(alpha_scene, target_union)) + reg
        loss.backward()
        opt.step()

        if (step % args.save_every == 0) or (step == args.steps - 1):
            alpha_np = alpha_scene.detach().cpu().numpy().astype(np.float32)
            target_np = target_union.detach().cpu().numpy().astype(np.float32)
            _save_gray(step_dir / f"render_step_{step:04d}.png", alpha_np)
            _save_gray(step_dir / f"target_union_step_{step:04d}.png", target_np)

    renderer_final = make_renderer(2e-4, 2e-4, args.final_faces_pp)
    fitted_rows, obj_metrics, meshes = [], [], []
    with torch.no_grad():
        for i in range(n):
            s = raw_s[i] * torch.exp(torch.clamp(dls[i], -2.0, 2.0))
            r = raw_r[i] + torch.clamp(dr[i], -90.0, 90.0)
            t = raw_t[i] + dt[i]
            rot = _euler_xyz_deg(r[0], r[1], r[2]).to(device=device, dtype=torch.float32)
            vw = (verts[i] * s.unsqueeze(0)) @ rot.T + t.unsqueeze(0)
            mesh_i = Meshes(verts=[vw], faces=[faces[i]])
            meshes.append(mesh_i)
            alpha = torch.clamp(renderer_final(mesh_i)[0, ..., 3], 0.0, 1.0)
            iou_i = _iou(alpha.detach().cpu().numpy() >= 0.5, targets[i].detach().cpu().numpy() >= 0.5)
            obj_metrics.append({"index": i, "iou": iou_i})
            fitted_rows.append([float(t[0]), float(t[1]), float(t[2]), float(s[0]), float(s[1]), float(s[2]), float(r[0]), float(r[1]), float(r[2])])

        alpha_scene = torch.clamp(renderer_final(join_meshes_as_scene(meshes))[0, ..., 3], 0.0, 1.0)
        alpha_np = alpha_scene.detach().cpu().numpy().astype(np.float32)
        target_np = target_union.detach().cpu().numpy().astype(np.float32)

    _save_gray(output_dir / "render_ref.png", alpha_np)
    np.save(output_dir / "render_ref.npy", alpha_np)
    _save_gray(output_dir / "target_ref.png", target_np)
    np.save(output_dir / "fitted_transform.npy", np.asarray(fitted_rows, dtype=np.float32))

    meta = {
        "image_path": str(image_path),
        "device": str(device),
        "object_count_used": n,
        "mean_object_iou": float(np.mean([m["iou"] for m in obj_metrics])) if obj_metrics else 0.0,
        "union_iou": _iou(alpha_np >= 0.5, target_np >= 0.5),
        "objects": obj_metrics,
        "saved_fitted_transform_npy": str(output_dir / "fitted_transform.npy"),
    }
    with open(output_dir / "render_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, ensure_ascii=False))
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=str, required=True)
    p.add_argument("--image-path", type=str, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--camera-dist", type=float, default=3.7)
    p.add_argument("--camera-elev", type=float, default=0.9)
    p.add_argument("--camera-azim", type=float, default=1.4)
    p.add_argument("--lr-t", type=float, default=4e-3)
    p.add_argument("--lr-s", type=float, default=3e-3)
    p.add_argument("--lr-r", type=float, default=8e-3)
    p.add_argument("--lr-cam", type=float, default=5e-3)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--learn-camera", action="store_true")
    p.add_argument("--use-fitted-init", action="store_true")
    p.add_argument("--faces-pp", type=int, default=24)
    p.add_argument("--final-faces-pp", type=int, default=64)
    p.add_argument("--disable-rescue", action="store_true")
    p.add_argument("--union-only-steps", type=int, default=0)
    p.add_argument("--prefer-fit-mesh", action="store_true")
    p.add_argument("--auto-create-fit-mesh", action="store_true")
    p.add_argument("--fit-face-count", type=int, default=5000)
    p.add_argument("--run-name", type=str, default="")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
