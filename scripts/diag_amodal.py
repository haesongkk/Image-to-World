"""Diagnostic: visible vs amodal mask comparison."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
vis_dir = ROOT / "output" / "mask_postprocess"
amo_dir = ROOT / "output" / "amodal_completion"

vis_masks = sorted(vis_dir.glob("object_*_mask.npy"))
header = f"{'idx':>3} {'class':<25} {'vis_area':>10} {'amo_area':>10} {'growth%':>8} {'vis_diag':>10} {'amo_diag':>10}"
print(header)
print("-" * len(header))
for mp in vis_masks:
    name = mp.stem
    visible = np.load(mp)
    if visible.ndim == 3:
        visible = visible[..., 0]
    visible = (visible > 0)
    amo_path = amo_dir / f"{name.replace('_mask', '_amodal_mask')}.npy"
    amodal = np.load(amo_path)
    if amodal.ndim == 3:
        amodal = amodal[..., 0]
    amodal = (amodal > 0)
    v_area = int(visible.sum())
    a_area = int(amodal.sum())
    growth = 100.0 * (a_area - v_area) / max(v_area, 1)
    vy, vx = np.where(visible)
    ay, ax = np.where(amodal)
    v_diag = float(np.hypot((vy.ptp() + 1) if vy.size else 0, (vx.ptp() + 1) if vx.size else 0))
    a_diag = float(np.hypot((ay.ptp() + 1) if ay.size else 0, (ax.ptp() + 1) if ax.size else 0))
    parts = name.split("_")
    cls = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]
    idx = parts[1]
    print(f"{idx:>3} {cls:<25} {v_area:>10} {a_area:>10} {growth:>7.1f}% {v_diag:>10.0f} {a_diag:>10.0f}")
