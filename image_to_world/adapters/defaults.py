from __future__ import annotations

from typing import Any

import numpy as np
import torch
import transformers
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForZeroShotObjectDetection, AutoProcessor

from image_to_world.config import DepthEstimationConfig, MaskGenerationConfig, ObjectCompletionConfig, TagExtractionConfig


class RamTagger:
    def __init__(self, config: TagExtractionConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.model = None

    def _load(self):
        if self.model is None:
            # RAM may rely on legacy symbol location from transformers.modeling_utils.
            # Keep compatibility when using newer transformers for SAM2.
            try:
                import transformers.modeling_utils as modeling_utils
                if not hasattr(modeling_utils, "apply_chunking_to_forward"):
                    from transformers.pytorch_utils import apply_chunking_to_forward
                    modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
            except Exception:
                pass
            from ram.models import ram_plus
            model = ram_plus(pretrained=str(self.config.checkpoint_path), image_size=self.config.image_size, vit="swin_l")
            model.eval()
            self.model = model.to(self.device)
        return self.model

    def predict_tags(self, tensor: torch.Tensor) -> list[str]:
        model = self._load()
        from ram import inference_ram as inference
        with torch.no_grad():
            result = inference(tensor, model)
        return [tag.strip().lower() for tag in result[0].split("|") if tag.strip()]


class GroundedSamAdapter:
    def __init__(self, config: MaskGenerationConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.processor = None
        self.detector = None
        self.sam2_processor = None
        self.sam2_model = None

    def _load_detector(self):
        if self.processor is None or self.detector is None:
            self.processor = AutoProcessor.from_pretrained(self.config.grounding_model_id)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.config.grounding_model_id).to(self.device)
            self.detector.eval()
        return self.processor, self.detector

    def _load_segmenter(self):
        if self.sam2_processor is None or self.sam2_model is None:
            if not hasattr(transformers, "Sam2Model") or not hasattr(transformers, "Sam2Processor"):
                raise RuntimeError(
                    "Transformers SAM2 backend requires transformers>=4.57. "
                    f"Current version: {transformers.__version__}"
                )
            Sam2Processor = getattr(transformers, "Sam2Processor")
            Sam2Model = getattr(transformers, "Sam2Model")
            self.sam2_processor = Sam2Processor.from_pretrained(self.config.sam2_model_id)
            self.sam2_model = Sam2Model.from_pretrained(self.config.sam2_model_id).to(self.device)
            self.sam2_model.eval()
        return self.sam2_processor, self.sam2_model

    def detect(self, image: Image.Image, text_prompt: str) -> dict[str, Any]:
        processor, detector = self._load_detector()
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = detector(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        # Future-proof against Grounding DINO label output type changes.
        if "text_labels" in result and result["text_labels"] is not None:
            result["labels"] = result["text_labels"]
        return result

    def segment(self, image_np: np.ndarray, boxes: np.ndarray):
        if len(boxes) == 0:
            return []
        processor, model = self._load_segmenter()
        image = Image.fromarray(image_np)
        input_boxes = [[box.astype(float).tolist() for box in boxes]]
        inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        pred_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        pred_masks = pred_masks.detach().cpu().numpy() if hasattr(pred_masks, "detach") else np.asarray(pred_masks)
        if pred_masks.ndim == 4:
            pred_masks = pred_masks[:, 0, :, :]
        elif pred_masks.ndim == 2:
            pred_masks = pred_masks[None, :, :]
        binary_masks = (pred_masks > 0).astype(np.uint8)
        return [mask for mask in binary_masks]


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
