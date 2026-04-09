from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from image_to_world.common import save_json


def to_visual_xyz(points_xyz: np.ndarray) -> np.ndarray:
    return np.stack([points_xyz[:, 0], points_xyz[:, 2], points_xyz[:, 1]], axis=1)


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)


def _build_camera_basis(center: torch.Tensor, elev_deg: float, azim_deg: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    elev = torch.deg2rad(torch.tensor(float(elev_deg), dtype=torch.float32, device=center.device))
    azim = torch.deg2rad(torch.tensor(float(azim_deg), dtype=torch.float32, device=center.device))
    view_dir = torch.stack([
        torch.cos(elev) * torch.cos(azim),
        torch.cos(elev) * torch.sin(azim),
        torch.sin(elev),
    ])
    radius = 3.0
    eye = center + radius * view_dir
    forward = _normalize(center - eye)
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=center.device)
    if torch.abs(torch.dot(forward, world_up)) > 0.98:
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=center.device)
    right = _normalize(torch.cross(forward, world_up, dim=0))
    up = _normalize(torch.cross(right, forward, dim=0))
    return right, up, forward


def _project_points(
    points: torch.Tensor,
    center: torch.Tensor,
    right: torch.Tensor,
    up: torch.Tensor,
    forward: torch.Tensor,
    scale: torch.Tensor,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = points - center.unsqueeze(0)
    x = p @ right
    y = p @ up
    z = p @ forward

    u = ((x / scale) * 0.5 + 0.5) * float(image_size - 1)
    v = (1.0 - ((y / scale) * 0.5 + 0.5)) * float(image_size - 1)
    u = torch.clamp(u, 0, image_size - 1)
    v = torch.clamp(v, 0, image_size - 1)
    return u, v, z


def _render_one_view_mesh(
    objects: list[dict[str, Any]],
    center: torch.Tensor,
    *,
    elev_deg: float,
    azim_deg: float,
    image_size: int,
    mesh_alpha: int,
    max_faces_per_object: int,
) -> np.ndarray:
    all_points = torch.cat([obj["verts"] for obj in objects], dim=0)
    right, up, forward = _build_camera_basis(center, elev_deg=elev_deg, azim_deg=azim_deg)
    rel = all_points - center.unsqueeze(0)
    span_x = torch.max(torch.abs(rel @ right))
    span_y = torch.max(torch.abs(rel @ up))
    scale = torch.maximum(span_x, span_y) * 1.12 + 1e-6

    bg = Image.new("RGBA", (image_size, image_size), (228, 231, 236, 255))
    draw_bg = ImageDraw.Draw(bg)
    for y in range(image_size):
        t = y / max(1, image_size - 1)
        r = int(137 * (1.0 - t) + 116 * t)
        g = int(141 * (1.0 - t) + 121 * t)
        b = int(147 * (1.0 - t) + 129 * t)
        draw_bg.line((0, y, image_size, y), fill=(r, g, b, 255), width=1)

    faces_to_draw: list[tuple[float, tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[int, int, int, int]]] = []
    for obj in objects:
        verts = obj["verts"]
        faces = obj["faces"]
        color = obj["color"]
        u, v, z = _project_points(verts, center, right, up, forward, scale, image_size=image_size)
        if faces.shape[0] > max_faces_per_object:
            sel = np.linspace(0, faces.shape[0] - 1, num=max_faces_per_object, dtype=np.int64)
            faces = faces[sel]
        f_t = torch.from_numpy(faces).to(verts.device, dtype=torch.int64)
        uvz = torch.stack([u, v, z], dim=1)
        tri = uvz[f_t]  # [F,3,3]
        tri_uv = tri[:, :, :2].detach().cpu().numpy()
        tri_z = tri[:, :, 2].mean(dim=1).detach().cpu().numpy()
        base_r = int(max(0, min(255, color[0] * 255)))
        base_g = int(max(0, min(255, color[1] * 255)))
        base_b = int(max(0, min(255, color[2] * 255)))
        zmin = float(tri_z.min()) if tri_z.size > 0 else 0.0
        zmax = float(tri_z.max()) if tri_z.size > 0 else 1.0
        denom = max(1e-6, zmax - zmin)
        for i in range(tri_uv.shape[0]):
            zz = float(tri_z[i])
            shade = 0.70 + 0.30 * ((zz - zmin) / denom)
            rgba = (
                int(max(0, min(255, base_r * shade))),
                int(max(0, min(255, base_g * shade))),
                int(max(0, min(255, base_b * shade))),
                int(mesh_alpha),
            )
            p0 = (float(tri_uv[i, 0, 0]), float(tri_uv[i, 0, 1]))
            p1 = (float(tri_uv[i, 1, 0]), float(tri_uv[i, 1, 1]))
            p2 = (float(tri_uv[i, 2, 0]), float(tri_uv[i, 2, 1]))
            faces_to_draw.append((zz, (p0, p1, p2), rgba))

    faces_to_draw.sort(key=lambda x: x[0])  # far -> near painter's algorithm
    layer = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(layer, "RGBA")
    for _, pts, rgba in faces_to_draw:
        draw_layer.polygon(pts, fill=rgba)
    out = Image.alpha_composite(bg, layer)

    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, image_size - 1, image_size - 1), outline=(110, 120, 138, 255), width=2)
    return np.asarray(out.convert("RGB"), dtype=np.uint8)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ])
    else:
        candidates.extend([
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_assembled_scene_visualization_torch(
    *,
    transformed_meshes: list[dict[str, Any]],
    png_path: Path,
    summary_path: Path,
    device: str = "cuda",
    image_size: int = 1400,
    point_size: int = 2,
) -> None:
    if not transformed_meshes:
        blank = Image.new("RGB", (image_size, image_size), (250, 250, 250))
        blank.save(png_path)
        save_json(summary_path, {
            "output_png": str(png_path),
            "num_objects": 0,
            "objects": [],
            "backend": "torch",
            "notes": ["No transformed meshes to render."],
        })
        return

    render_device = torch.device(device)
    if render_device.type == "cuda" and not torch.cuda.is_available():
        render_device = torch.device("cpu")

    objects: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    for idx, rec in enumerate(transformed_meshes):
        v_np = to_visual_xyz(rec["vertices_world"]).astype(np.float32, copy=False)
        verts = torch.from_numpy(v_np).to(render_device)
        faces = np.asarray(rec.get("faces_v", []), dtype=np.int64)
        color = tuple(float(c) for c in rec["material_rgb"])
        objects.append({"verts": verts, "faces": faces, "color": color})
        center = v_np.mean(axis=0).tolist()
        object_summaries.append({
            "placement_id": rec.get("placement_id", idx),
            "class_name": rec.get("class_name"),
            "object_name": rec.get("object_name"),
            "material_rgb": [float(color[0]), float(color[1]), float(color[2])],
            "num_vertices_rendered": int(verts.shape[0]),
            "num_faces_rendered": int(faces.shape[0]),
            "center_visual_xyz": center,
        })

    all_points = torch.cat([o["verts"] for o in objects], dim=0)
    center = torch.mean(all_points, dim=0)
    # Render at 2x and downsample for anti-aliasing.
    sub = max(360, int(image_size * 1.2))
    views = [
        ("Perspective View", 22.0, -58.0),
        ("Top View", 90.0, -90.0),
        ("Front View", 0.0, -90.0),
        ("Side View", 0.0, 0.0),
    ]
    max_faces_per_object = 80000
    mesh_alpha = 255
    rendered = [
        _render_one_view_mesh(
            objects,
            center,
            elev_deg=elev,
            azim_deg=azim,
            image_size=sub,
            mesh_alpha=mesh_alpha,
            max_faces_per_object=max_faces_per_object,
        )
        for title, elev, azim in views
    ]

    # Match the overall aspect ratio of the original Matplotlib output (4:3).
    final_h = max(1080, int(image_size * 1.54))
    final_h = max(3, (final_h // 3) * 3)
    final_w = int(final_h * 4 / 3)
    outer_pad = max(28, final_h // 40)
    header_h = max(88, final_h // 13)
    panel_gap = max(24, final_h // 72)
    row_gap = max(30, final_h // 62)
    panel_title_h = max(28, final_h // 58)

    usable_w = final_w - outer_pad * 2 - panel_gap
    usable_h = final_h - outer_pad * 2 - header_h - row_gap - panel_title_h * 2
    panel_size = max(240, min(usable_w // 2, usable_h // 2))
    panel_w = panel_size
    panel_h = panel_size
    grid_w = panel_w * 2 + panel_gap
    inner_w = final_w - outer_pad * 2
    x0 = outer_pad + max(0, (inner_w - grid_w) // 2)

    final = Image.new("RGB", (final_w, final_h), (244, 247, 252))
    draw = ImageDraw.Draw(final)
    draw.rectangle((0, 0, final_w - 1, final_h - 1), outline=(116, 126, 142), width=2)

    title_font = _load_font(max(22, final_h // 38), bold=True)
    subtitle_font = _load_font(max(14, final_h // 80), bold=False)
    panel_font = _load_font(max(16, final_h // 72), bold=True)

    title_text = "Assembled Scene Validation"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = max(outer_pad, (final_w - title_w) // 2)
    draw.text((title_x, outer_pad), title_text, fill=(28, 34, 44), font=title_font)
    subtitle_text = f"GPU Preview | Objects: {len(transformed_meshes)} | Device: {render_device}"
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
    subtitle_w = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = max(outer_pad, (final_w - subtitle_w) // 2)
    draw.text((subtitle_x, outer_pad + max(36, final_h // 30)), subtitle_text, fill=(72, 82, 96), font=subtitle_font)

    panel_titles = ["Perspective View", "Top View", "Front View", "Side View"]
    top_title_y = outer_pad + header_h
    top_img_y = top_title_y + panel_title_h
    bottom_title_y = top_img_y + panel_h + row_gap
    bottom_img_y = bottom_title_y + panel_title_h

    panel_positions = [
        (x0, top_img_y),
        (x0 + panel_w + panel_gap, top_img_y),
        (x0, bottom_img_y),
        (x0 + panel_w + panel_gap, bottom_img_y),
    ]
    panel_title_positions = [
        (x0, top_title_y),
        (x0 + panel_w + panel_gap, top_title_y),
        (x0, bottom_title_y),
        (x0 + panel_w + panel_gap, bottom_title_y),
    ]

    for idx, (px, py) in enumerate(panel_positions):
        tx, ty = panel_title_positions[idx]
        draw.text((tx, ty), panel_titles[idx], fill=(40, 48, 60), font=panel_font)
        draw.rectangle((px - 1, py - 1, px + panel_w + 1, py + panel_h + 1), outline=(142, 154, 172), width=2)
        view_img = Image.fromarray(rendered[idx], mode="RGB").resize((panel_w, panel_h), resample=Image.Resampling.LANCZOS)
        final.paste(view_img, (px, py))

    final.save(png_path)

    save_json(summary_path, {
        "output_png": str(png_path),
        "num_objects": len(transformed_meshes),
        "objects": object_summaries,
        "backend": "torch",
        "render_device": str(render_device),
        "image_size_per_view": int(sub),
        "output_size": [int(final_w), int(final_h)],
        "output_aspect_ratio": "4:3",
        "render_mode": "mesh_opaque",
        "mesh_alpha": int(mesh_alpha),
        "mesh_faces_per_object_cap": int(max_faces_per_object),
        "point_size": int(point_size),
        "notes": [
            "GPU mesh renderer for fast scene assembly previews.",
            "Each panel is rendered with an orthographic camera from perspective, top, front, and side viewpoints.",
            "Rendered geometry uses transformed mesh faces with per-object flat colors.",
        ],
    })
