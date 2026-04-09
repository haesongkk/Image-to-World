from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForZeroShotObjectDetection, AutoProcessor

from image_to_world.config import DepthEstimationConfig, MaskGenerationConfig, MeshGenerationConfig, ObjectCompletionConfig, TagExtractionConfig
from image_to_world.geometry import rotation_matrix_xyz_deg
from ram import inference_ram as inference
from ram.models import ram_plus
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_config, load_model
from shap_e.util.notebooks import decode_latent_mesh


class RamTagger:
    def __init__(self, config: TagExtractionConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.model = None

    def _load(self):
        if self.model is None:
            model = ram_plus(pretrained=str(self.config.checkpoint_path), image_size=self.config.image_size, vit="swin_l")
            model.eval()
            self.model = model.to(self.device)
        return self.model

    def predict_tags(self, tensor: torch.Tensor) -> list[str]:
        model = self._load()
        with torch.no_grad():
            result = inference(tensor, model)
        return [tag.strip().lower() for tag in result[0].split("|") if tag.strip()]


class GroundedSamAdapter:
    def __init__(self, config: MaskGenerationConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.processor = None
        self.detector = None
        self.predictor = None

    def _load_detector(self):
        if self.processor is None or self.detector is None:
            self.processor = AutoProcessor.from_pretrained(self.config.grounding_model_id)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.grounding_model_id).to(self.device)
            self.detector.eval()
        return self.processor, self.detector

    def _load_predictor(self):
        if self.predictor is None:
            sam2_model = build_sam2(str(self.config.sam2_config), str(self.config.sam2_checkpoint), device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
        return self.predictor

    def detect(self, image: Image.Image, text_prompt: str) -> dict[str, Any]:
        processor, detector = self._load_detector()
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = detector(**inputs)
        return processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

    def segment(self, image_np: np.ndarray, boxes: np.ndarray):
        predictor = self._load_predictor()
        predictor.set_image(image_np)
        masks = []
        for box in boxes:
            pred_masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
            masks.append(pred_masks[0].astype(np.uint8))
        return masks


class SdxlInpainter:
    def __init__(self, config: ObjectCompletionConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.pipeline = None

    def _load(self):
        if self.pipeline is None:
            kwargs = {"torch_dtype": torch.float16, "variant": "fp16"} if self.device == "cuda" else {"torch_dtype": torch.float32}
            self.pipeline = AutoPipelineForInpainting.from_pretrained(self.config.model_id, **kwargs).to(self.device)
        return self.pipeline

    def inpaint(self, *, prompt: str, image, mask_image, **kwargs):
        pipe = self._load()
        return pipe(prompt=prompt, image=image, mask_image=mask_image, **kwargs).images[0]


class ShapEGenerator:
    def __init__(self, config: MeshGenerationConfig, device: str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._transmitter = None
        self._model = None
        self._diffusion = None

    def _load(self):
        if self._transmitter is None:
            self._transmitter = load_model(self.config.transmitter_name, device=self.device)
            self._model = load_model(self.config.image_model_name, device=self.device)
            self._diffusion = diffusion_from_config(load_config("diffusion"))
        return self._transmitter, self._model, self._diffusion

    def generate_mesh(self, image, **kwargs):
        transmitter, model, diffusion = self._load()
        latents = sample_latents(
            batch_size=self.config.batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=self.config.guidance_scale,
            model_kwargs=dict(images=[image] * self.config.batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=(self.device.type == "cuda"),
            use_karras=self.config.use_karras,
            karras_steps=self.config.karras_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            s_churn=self.config.s_churn,
        )
        mesh = decode_latent_mesh(transmitter, latents[0]).tri_mesh()
        rot = rotation_matrix_xyz_deg(self.config.asset_pre_rot_euler_xyz_deg)
        verts = np.asarray(mesh.verts, dtype=np.float64)
        mesh.verts = (rot @ verts.T).T
        return mesh


class ConfigurableDepthEstimator:
    def __init__(self, config: DepthEstimationConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.processor = None
        self.model = None

    def _load(self):
        if self.processor is None or self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.config.model_id).to(self.device)
            self.model.eval()
        return self.processor, self.model

    def estimate(self, image: Image.Image):
        processor, model = self._load()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        post_processed = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        predicted = post_processed[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        metadata = {
            "model_family": self.config.model_family,
        }
        if "field_of_view" in post_processed[0]:
            fov_value = post_processed[0]["field_of_view"]
            if hasattr(fov_value, "detach"):
                fov_value = fov_value.detach().cpu().numpy()
            metadata["field_of_view"] = np.asarray(fov_value).tolist()
        return {
            "depth": predicted,
            "depth_type": "absolute",
            "metadata": metadata,
        }


DepthAnythingEstimator = ConfigurableDepthEstimator
