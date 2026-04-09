from __future__ import annotations

from importlib import resources
from pathlib import Path

import cv2
import numpy as np
import torch

from perspective2d.config import get_perspective2d_cfg_defaults
from perspective2d.perspectivefields import LowLevelEncoder, ResizeTransform
from perspective2d.modeling.backbone import build_backbone
from perspective2d.modeling.param_network import build_param_net
from perspective2d.modeling.persformer_heads import build_persformer_heads


class PerspectiveFieldsAdapter(torch.nn.Module):
    def __init__(self, *, version: str, weights_path: Path, config_file: str, device: str) -> None:
        super().__init__()
        self.version = version
        self.weights_path = Path(weights_path)
        self.device_name = device

        cfg = get_perspective2d_cfg_defaults()
        with resources.path("perspective2d.config", config_file) as config_path:
            cfg.merge_from_file(str(config_path))
        cfg.freeze()
        self.cfg = cfg

        self.backbone = build_backbone(cfg)
        self.ll_enc = LowLevelEncoder()
        self.persformer_heads = build_persformer_heads(cfg, self.backbone.output_shape())
        self.param_net = build_param_net(cfg) if cfg.MODEL.RECOVER_RPF or cfg.MODEL.RECOVER_PP else None
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), persistent=False)
        self.input_format = cfg.INPUT.FORMAT
        self.aug = ResizeTransform(cfg.DATALOADER.RESIZE[0], cfg.DATALOADER.RESIZE[1])
        self._load_weights()
        self.eval()
        self.to(torch.device(device))

    def _load_weights(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"PerspectiveFields weights not found: {self.weights_path}")
        state_dict = torch.load(str(self.weights_path), map_location="cpu")
        model_state = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
        self.load_state_dict(model_state, strict=False)

    @torch.no_grad()
    def inference(self, image_bgr: np.ndarray) -> dict:
        original_image = image_bgr.copy()
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = self.forward([inputs])[0]
        return predictions

    def forward(self, batched_inputs) -> list[dict]:
        images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = torch.stack(images)
        hl_features = self.backbone(images)
        ll_features = self.ll_enc(images)
        features = {"hl": hl_features, "ll": ll_features}

        results = self.persformer_heads.inference(features)
        processed_results = self.persformer_heads.postprocess(results, batched_inputs, images)

        if self.param_net is not None:
            param = self.param_net(results, batched_inputs)
            if "pred_general_vfov" not in param:
                param["pred_general_vfov"] = param["pred_vfov"]
            if "pred_rel_cx" not in param:
                param["pred_rel_cx"] = torch.zeros_like(param["pred_general_vfov"])
            if "pred_rel_cy" not in param:
                param["pred_rel_cy"] = torch.zeros_like(param["pred_general_vfov"])
            for i in range(len(processed_results)):
                processed_results[i].update({k: v[i] for k, v in param.items()})
        return processed_results

    @staticmethod
    def tensor_to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def predict_camera(self, image_path: Path) -> dict:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image for PerspectiveFields inference: {image_path}")
        pred = self.inference(image_bgr)
        h, w = image_bgr.shape[:2]
        vfov_deg = self.tensor_to_float(pred["pred_general_vfov"])
        roll_deg = self.tensor_to_float(pred["pred_roll"])
        pitch_deg = self.tensor_to_float(pred["pred_pitch"])
        rel_cx = self.tensor_to_float(pred["pred_rel_cx"])
        rel_cy = self.tensor_to_float(pred["pred_rel_cy"])

        vfov_rad = np.deg2rad(max(vfov_deg, 1e-4))
        fy = float((h * 0.5) / np.tan(vfov_rad * 0.5))
        fx = fy
        cx = float(w * (0.5 + rel_cx))
        cy = float(h * (0.5 + rel_cy))
        horizon_y = float(cy - fy * np.tan(np.deg2rad(pitch_deg)))
        gravity = pred.get("pred_gravity")
        latitude = pred.get("pred_latitude")
        return {
            "vfov_deg": vfov_deg,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "rel_cx": rel_cx,
            "rel_cy": rel_cy,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "horizon_y": horizon_y,
            "raw_prediction_keys": sorted(pred.keys()),
            "pred_gravity": gravity.detach().cpu().numpy() if gravity is not None else None,
            "pred_latitude": latitude.detach().cpu().numpy() if latitude is not None else None,
        }
