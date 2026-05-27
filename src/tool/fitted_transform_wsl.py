from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
from src.config import CAMERA_ESTIMATION_OUTPUT_DIR, MASK_POSTPROCESS_OUTPUT_DIR, MESH_GENERATION_OUTPUT_DIR, SCENE_PRECOMPUTE_OUTPUT_DIR

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

DEFAULTS = {
    # main_steps=0 → 4-way heading search only, no silhouette refinement.
    # Current raw_transform has visible-AABB scale issues that make silhouette
    # optimization diverge or actively hurt. Enable >0 once P0-1 (amodal) lands
    # and raw scales become realistic.
    "steps": 0,
    "save_every": 20,
    "image_size": 160,
    "fov": 60.0,
    "camera_dist": 3.7,
    "camera_elev": 0.9,
    "camera_azim": 1.4,
    "lr_t": 1.5e-3,
    "lr_s": 1.0e-3,
    "lr_r": 3e-3,
    "device": "cuda",  # "cuda" or "cpu"
    "use_fitted_init": True,
    "faces_pp": 16,
    "final_faces_pp": 32,
    "sil_ema_beta": 0.9,
    "early_stop_warmup_steps": 40,
    "early_stop_total_loss": 0.08,
    "early_stop_patience": 15,
    "camera_prefit_steps": 30,
    "camera_prefit_lr": 0.04,
    "camera_prefit_w_dist": 0.01,
    "camera_prefit_w_angle": 0.002,
    "camera_prefit_w_fov": 0.002,
}


def _resolve_glb_paths(mesh_dir: Path) -> list[Path]:
    return (
        sorted(mesh_dir.glob("*_remesh.glb"))
        or sorted(mesh_dir.glob("*_shape_mesh.glb"))
        or sorted(mesh_dir.glob("*.glb"))
    )


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

_S_TO_P = np.array(
    [
        [0.0, 1.0, 0.0],   # x_p = -y_s
        [0.0, 0.0, 1.0],    # y_p =  z_s
        [1.0, 0.0, 0.0],   # z_p = -x_s
    ],
    dtype=np.float32,
)


def _euler_xyz_deg_to_rot_np(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def _rot_to_euler_xyz_deg_np(r: np.ndarray) -> tuple[float, float, float]:
    sy = float(-r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rz = 0.0
        rx = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


# storage:  x=front, y=right, z=up
# pytorch3d: x=left,  y=up,    z=in(back)
def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["transforms"] if isinstance(data, dict) and "transforms" in data else data
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        packed = []
        for row in rows:
            t = row.get("translation", {})
            s = row.get("scale", {})
            r = row.get("rotation_deg", row.get("rotation", {}))
            t_s = np.array(
                [float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("z", 0.0))],
                dtype=np.float32,
            )
            s_s = np.array(
                [float(s.get("x", 1.0)), float(s.get("y", 1.0)), float(s.get("z", 1.0))],
                dtype=np.float32,
            )
            r_s = _euler_xyz_deg_to_rot_np(
                float(r.get("x", 0.0)),
                float(r.get("y", 0.0)),
                float(r.get("z", 0.0)),
            )

            t_p = _S_TO_P @ t_s
            s_p = np.array([s_s[1], s_s[2], s_s[0]], dtype=np.float32)  # (y,z,x)
            r_p = _S_TO_P @ r_s @ _S_TO_P.T
            rx_p, ry_p, rz_p = _rot_to_euler_xyz_deg_np(r_p)
            packed.append([
                float(t_p[0]), float(t_p[1]), float(t_p[2]),
                float(s_p[0]), float(s_p[1]), float(s_p[2]),
                float(rx_p), float(ry_p), float(rz_p),
            ])
        arr = np.asarray(packed, dtype=np.float32)
    else:
        arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


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


def _safe_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def run(project_root: str, image_path: str, run_name: str = "simple") -> dict:
    project_root = Path(project_root)
    image_path = Path(image_path)
    mesh_dir = MESH_GENERATION_OUTPUT_DIR
    mask_dir = MASK_POSTPROCESS_OUTPUT_DIR
    raw_transform_path = SCENE_PRECOMPUTE_OUTPUT_DIR / "raw_transform.json"
    perspective_path = CAMERA_ESTIMATION_OUTPUT_DIR / f"{image_path.stem}_perspective_fields.json"
    fitted_init_path = project_root / "output" / "fitted_transform" / "fitted_transform.json"
    output_dir = project_root / "output" / "fitted_transform_debug" / image_path.stem / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    step_dir = output_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)

    device_mode = str(DEFAULTS["device"]).lower()
    device = torch.device("cuda" if device_mode != "cpu" and torch.cuda.is_available() else "cpu")

    glb_paths = _resolve_glb_paths(mesh_dir)
    mask_paths = sorted(mask_dir.glob("object_*_mask.npy"))
    raw = _load_transform_json(raw_transform_path)

    n = min(len(glb_paths), len(mask_paths), raw.shape[0])
    if n == 0:
        raise ValueError("No objects to optimize.")

    transforms = raw[:n, :9].copy()

    verts, faces = [], []
    for i in range(n):
        v, f = _load_mesh(glb_paths[i])
        verts.append(torch.tensor(v, dtype=torch.float32, device=device))
        faces.append(torch.tensor(f, dtype=torch.int64, device=device))

    image_size = int(DEFAULTS["image_size"])
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
    opt = torch.optim.Adam(
        [
            {"params": [dt], "lr": float(DEFAULTS["lr_t"])},
            {"params": [dls], "lr": float(DEFAULTS["lr_s"])},
            {"params": [dr], "lr": float(DEFAULTS["lr_r"])},
        ]
    )

    init_dist = float(DEFAULTS["camera_dist"])
    init_elev = float(DEFAULTS["camera_elev"])
    init_azim = float(DEFAULTS["camera_azim"])
    init_fov = float(DEFAULTS["fov"])
    if perspective_path.exists():
        with open(perspective_path, "r", encoding="utf-8") as f:
            pf = json.load(f)
        pf_vfov = _safe_float(pf.get("pred_vfov"), init_fov)
        pf_pitch = _safe_float(pf.get("pred_pitch"), 0.0)
        pf_roll = _safe_float(pf.get("pred_roll"), 0.0)
        pf_rel_focal = max(0.25, min(2.5, _safe_float(pf.get("pred_rel_focal"), 1.0)))
        init_fov = max(35.0, min(85.0, pf_vfov))
        init_elev = max(-35.0, min(35.0, -0.55 * pf_pitch))
        init_azim = float(DEFAULTS["camera_azim"]) + (0.2 * pf_roll)
        init_dist = max(2.0, min(7.0, float(DEFAULTS["camera_dist"]) / pf_rel_focal))

    def _make_camera(dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor, fov: torch.Tensor) -> FoVPerspectiveCameras:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        return FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    with torch.no_grad():
        d0 = torch.tensor([init_dist], dtype=torch.float32, device=device)
        e0 = torch.tensor([init_elev], dtype=torch.float32, device=device)
        a0 = torch.tensor([init_azim], dtype=torch.float32, device=device)
        f0 = torch.tensor([init_fov], dtype=torch.float32, device=device)
        cam = _make_camera(d0, e0, a0, f0)

    def make_renderer(camera: FoVPerspectiveCameras, sigma: float, gamma: float, fpp: int) -> MeshRenderer:
        bp = BlendParams(sigma=sigma, gamma=gamma)
        rs = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * bp.sigma,
            faces_per_pixel=fpp,
            bin_size=16,
            max_faces_per_bin=200000,
        )
        sil_raster = MeshRasterizer(cameras=camera, raster_settings=rs)
        return MeshRenderer(sil_raster, SoftSilhouetteShader(blend_params=bp))

    def make_vis_renderer(camera: FoVPerspectiveCameras) -> MeshRenderer:
        # Monitoring-only renderer: crisper silhouette for easier visual inspection.
        bp = BlendParams(sigma=1e-6, gamma=1e-6)
        rs = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=16,
            max_faces_per_bin=200000,
        )
        raster = MeshRasterizer(cameras=camera, raster_settings=rs)
        return MeshRenderer(raster, SoftSilhouetteShader(blend_params=bp))
    # Camera pre-fit: keep raw transforms fixed and optimize only camera intrinsics/extrinsics
    log_dist = torch.nn.Parameter(torch.tensor([math.log(max(init_dist, 1e-3))], dtype=torch.float32, device=device))
    elev_p = torch.nn.Parameter(torch.tensor([init_elev], dtype=torch.float32, device=device))
    azim_p = torch.nn.Parameter(torch.tensor([init_azim], dtype=torch.float32, device=device))
    fov_p = torch.nn.Parameter(torch.tensor([init_fov], dtype=torch.float32, device=device))
    cam_opt = torch.optim.Adam([log_dist, elev_p, azim_p, fov_p], lr=float(DEFAULTS["camera_prefit_lr"]))
    for _ in range(max(int(DEFAULTS["camera_prefit_steps"]), 0)):
        cam_opt.zero_grad(set_to_none=True)
        dist_cur = torch.exp(log_dist).clamp(1.5, 8.0)
        elev_cur = elev_p.clamp(-45.0, 45.0)
        azim_cur = azim_p.clamp(-180.0, 180.0)
        fov_cur = fov_p.clamp(30.0, 90.0)
        cam_cur = _make_camera(dist_cur, elev_cur, azim_cur, fov_cur)
        renderer_cur = make_renderer(cam_cur, 2e-3, 2e-3, int(DEFAULTS["faces_pp"]))
        pre_meshes = []
        for i in range(n):
            rot0 = _euler_xyz_deg(raw_r[i, 0], raw_r[i, 1], raw_r[i, 2]).to(device=device, dtype=torch.float32)
            vw0 = (verts[i] * raw_s[i].unsqueeze(0)) @ rot0.T + raw_t[i].unsqueeze(0)
            pre_meshes.append(Meshes(verts=[vw0], faces=[faces[i]]))
        alpha_prefit = torch.clamp(renderer_cur(join_meshes_as_scene(pre_meshes))[0, ..., 3], 0.0, 1.0)
        loss_prefit = (
            F.binary_cross_entropy(alpha_prefit, target_union) + _dice_loss(alpha_prefit, target_union)
            + float(DEFAULTS["camera_prefit_w_dist"]) * ((dist_cur - init_dist) ** 2).mean()
            + float(DEFAULTS["camera_prefit_w_angle"]) * (((elev_cur - init_elev) ** 2).mean() + ((azim_cur - init_azim) ** 2).mean())
            + float(DEFAULTS["camera_prefit_w_fov"]) * ((fov_cur - init_fov) ** 2).mean()
        )
        loss_prefit.backward()
        cam_opt.step()

    with torch.no_grad():
        dist_cur = torch.exp(log_dist).clamp(1.5, 8.0)
        elev_cur = elev_p.clamp(-45.0, 45.0)
        azim_cur = azim_p.clamp(-180.0, 180.0)
        fov_cur = fov_p.clamp(30.0, 90.0)
        cam = _make_camera(dist_cur, elev_cur, azim_cur, fov_cur)

    vis_renderer = make_vis_renderer(cam)

    # 4-way heading seed (P0-2): pick the best gravity-axis rotation per object.
    # In pytorch3d frame (after S_TO_P) the up axis is +y, so heading lives in raw_r[:, 1].
    # fitted_transform's main loop can only correct ±90°, so this resolves 180° flips first.
    heading_offsets_deg = [0.0, 90.0, 180.0, 270.0]
    heading_log: list[dict] = []
    with torch.no_grad():
        heading_renderer = make_renderer(cam, 2e-3, 2e-3, int(DEFAULTS["faces_pp"]))
        best_offsets = torch.zeros(n, dtype=torch.float32, device=device)
        for i in range(n):
            ious = []
            for off in heading_offsets_deg:
                r_test = raw_r[i].clone()
                r_test[1] = r_test[1] + float(off)
                rot = _euler_xyz_deg(r_test[0], r_test[1], r_test[2]).to(device=device, dtype=torch.float32)
                vw = (verts[i] * raw_s[i].unsqueeze(0)) @ rot.T + raw_t[i].unsqueeze(0)
                mesh_i = Meshes(verts=[vw], faces=[faces[i]])
                alpha = torch.clamp(heading_renderer(mesh_i)[0, ..., 3], 0.0, 1.0)
                iou_val = _iou(
                    alpha.detach().cpu().numpy() >= 0.5,
                    targets[i].detach().cpu().numpy() >= 0.5,
                )
                ious.append(iou_val)
            best_idx = int(np.argmax(ious))
            best_offsets[i] = float(heading_offsets_deg[best_idx])
            heading_log.append({
                "index": i,
                "ious": [float(x) for x in ious],
                "best_offset_deg": float(heading_offsets_deg[best_idx]),
                "best_iou": float(ious[best_idx]),
            })
            print(
                f"[heading] obj {i:02d}: "
                f"ious={[f'{x:.3f}' for x in ious]} "
                f"-> {heading_offsets_deg[best_idx]:6.1f}° (iou={ious[best_idx]:.3f})"
            )
        raw_r = raw_r.clone()
        raw_r[:, 1] = raw_r[:, 1] + best_offsets

    with open(output_dir / "heading_search.json", "w", encoding="utf-8") as f:
        json.dump({"offsets_tried_deg": heading_offsets_deg, "per_object": heading_log}, f, indent=2)

    total_steps = int(DEFAULTS["steps"])
    total_loss_ema: float | None = None
    early_stop_warmup = max(int(DEFAULTS["early_stop_warmup_steps"]), 0)
    early_stop_patience = max(int(DEFAULTS["early_stop_patience"]), 1)
    early_stop_count = 0
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        p = float(step) / max(float(total_steps - 1), 1.0)
        renderer = make_renderer(
            cam,
            (1e-2 * (1.0 - p)) + (2e-4 * p),
            (1e-2 * (1.0 - p)) + (2e-4 * p),
            int(DEFAULTS["faces_pp"]),
        )

        meshes, losses = [], []
        for i in range(n):
            s = raw_s[i] * torch.exp(torch.clamp(dls[i], -2.0, 2.0))
            r = raw_r[i] + torch.clamp(dr[i], -90.0, 90.0)
            t = raw_t[i] + torch.clamp(dt[i], -5.0, 5.0)
            rot = _euler_xyz_deg(r[0], r[1], r[2]).to(device=device, dtype=torch.float32)
            vw = (verts[i] * s.unsqueeze(0)) @ rot.T + t.unsqueeze(0)
            mesh_i = Meshes(verts=[vw], faces=[faces[i]])
            meshes.append(mesh_i)
            alpha = torch.clamp(renderer(mesh_i)[0, ..., 3], 0.0, 1.0)
            obj_loss = F.binary_cross_entropy(alpha, targets[i]) + _dice_loss(alpha, targets[i])
            if torch.isfinite(obj_loss):
                losses.append(obj_loss)

        if not losses:
            print(f"[skip] step={step} all per-object losses non-finite")
            continue

        sil_loss = torch.stack(losses).mean()

        scene_mesh = join_meshes_as_scene(meshes)
        alpha_scene = torch.clamp(renderer(scene_mesh)[0, ..., 3], 0.0, 1.0)
        reg = 0.05 * (dt ** 2).mean() + 0.01 * (dls ** 2).mean() + 0.0005 * (dr ** 2).mean()
        loss = sil_loss + reg

        if not torch.isfinite(loss):
            print(f"[skip] step={step} total loss non-finite ({float(loss.detach().item())})")
            opt.zero_grad(set_to_none=True)
            continue

        total_loss_value = float(loss.detach().item())
        if total_loss_ema is None:
            total_loss_ema = total_loss_value
        else:
            beta = float(DEFAULTS["sil_ema_beta"])
            total_loss_ema = (beta * total_loss_ema) + ((1.0 - beta) * total_loss_value)

        if step >= early_stop_warmup and total_loss_ema <= float(DEFAULTS["early_stop_total_loss"]):
            early_stop_count += 1
        else:
            early_stop_count = 0
        if early_stop_count >= early_stop_patience:
            print(
                f"[early-stop] step={step} total_loss_ema={total_loss_ema:.6f} "
                f"threshold={float(DEFAULTS['early_stop_total_loss']):.6f}"
            )
            break

        if (step % int(DEFAULTS["save_every"]) == 0):
            alpha_vis = torch.clamp(vis_renderer(scene_mesh)[0, ..., 3], 0.0, 1.0)
            alpha_np = alpha_vis.detach().cpu().numpy().astype(np.float32)
            target_np = target_union.detach().cpu().numpy().astype(np.float32)
            _save_gray(step_dir / f"render_step_{step:04d}_loss_{total_loss_value:.6f}.png", alpha_np)


        loss.backward()
        # Gradient clipping to suppress occasional spikes when an object's mesh
        # leaves the frustum.
        torch.nn.utils.clip_grad_norm_([dt, dls, dr], max_norm=2.0)
        opt.step()

    renderer_final = make_renderer(cam, 2e-4, 2e-4, int(DEFAULTS["final_faces_pp"]))
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

        scene_mesh = join_meshes_as_scene(meshes)
        alpha_scene = torch.clamp(renderer_final(scene_mesh)[0, ..., 3], 0.0, 1.0)
        alpha_vis = torch.clamp(vis_renderer(scene_mesh)[0, ..., 3], 0.0, 1.0)
        alpha_np = alpha_vis.detach().cpu().numpy().astype(np.float32)
        target_np = target_union.detach().cpu().numpy().astype(np.float32)

    _save_gray(output_dir / "render_ref.png", alpha_np)
    np.save(output_dir / "render_ref.npy", alpha_np)
    _save_gray(output_dir / "target_ref.png", target_np)
    fitted_json_path = output_dir / "fitted_transform.json"
    fitted_payload_rows = []
    for tr in np.asarray(fitted_rows, dtype=np.float32):
        t_p = np.array([float(tr[0]), float(tr[1]), float(tr[2])], dtype=np.float32)
        s_p = np.array([float(tr[3]), float(tr[4]), float(tr[5])], dtype=np.float32)
        r_p = _euler_xyz_deg_to_rot_np(float(tr[6]), float(tr[7]), float(tr[8]))

        t_s = _S_TO_P.T @ t_p
        s_s = np.array([s_p[2], s_p[0], s_p[1]], dtype=np.float32)  # inverse of (y,z,x)
        r_s = _S_TO_P.T @ r_p @ _S_TO_P
        rx_s, ry_s, rz_s = _rot_to_euler_xyz_deg_np(r_s)

        fitted_payload_rows.append(
            {
                "translation": {"x": float(t_s[0]), "y": float(t_s[1]), "z": float(t_s[2])},
                "scale": {"x": float(s_s[0]), "y": float(s_s[1]), "z": float(s_s[2])},
                "rotation_deg": {"x": float(rx_s), "y": float(ry_s), "z": float(rz_s)},
            }
        )

    fitted_payload = {"transforms": fitted_payload_rows}
    with open(fitted_json_path, "w", encoding="utf-8") as f:
        json.dump(fitted_payload, f, ensure_ascii=False, indent=2)

    meta = {
        "image_path": str(image_path),
        "device": str(device),
        "object_count_used": n,
        "mean_object_iou": float(np.mean([m["iou"] for m in obj_metrics])) if obj_metrics else 0.0,
        "union_iou": _iou(alpha_np >= 0.5, target_np >= 0.5),
        "objects": obj_metrics,
        "saved_fitted_transform_json": str(fitted_json_path),
        "camera": {
            "dist": float(torch.exp(log_dist).detach().item()),
            "elev": float(elev_p.detach().item()),
            "azim": float(azim_p.detach().item()),
            "fov": float(fov_p.detach().item()),
            "perspective_path_used": str(perspective_path) if perspective_path.exists() else None,
        },
    }
    with open(output_dir / "render_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit("Usage: fitted_transform_wsl.py <project_root> <image_path> [run_name]")
    run_name_arg = argv[2] if len(argv) >= 3 else "simple"
    run(argv[0], argv[1], run_name_arg)
