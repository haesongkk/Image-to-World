from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from image_to_world.common import artifact_path, project_path, resolve_device


@dataclass
class RuntimeConfig:
    device: str = field(default_factory=resolve_device)
    overwrite: bool = False
    skip_existing: bool = False


@dataclass
class TagExtractionConfig:
    checkpoint_path: Path = field(default_factory=lambda: project_path("pretrained", "ram_plus_swin_large_14m.pth"))
    image_path: Path = field(default_factory=lambda: artifact_path("raw_image.jpg"))
    output_path: Path = field(default_factory=lambda: artifact_path("extract_tags", "ram_result.json"))
    image_size: int = 384
    background_like_tags: list[str] = field(default_factory=lambda: [
        "room", "floor", "wood floor", "wall", "wood wall", "ceiling", "background",
        "indoor", "interior", "furniture", "home", "house", "bedroom", "hotel room",
    ])


@dataclass
class MaskGenerationConfig:
    image_path: Path = field(default_factory=lambda: artifact_path("raw_image.jpg"))
    prompt_path: Path = field(default_factory=lambda: artifact_path("extract_tags", "ram_result.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("generate_masks"))
    sam2_config: Path = field(default_factory=lambda: project_path("Grounded-SAM-2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"))
    sam2_checkpoint: Path = field(default_factory=lambda: project_path("Grounded-SAM-2", "checkpoints", "sam2.1_hiera_large.pt"))
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"


@dataclass
class ObjectCompletionConfig:
    input_json_path: Path = field(default_factory=lambda: artifact_path("generate_masks", "result.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("complete_objects"))
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    canvas_scale: float = 1.5
    num_inference_steps: int = 12
    guidance_scale: float = 5.0
    strength: float = 0.95


@dataclass
class MeshGenerationConfig:
    input_json_path: Path = field(default_factory=lambda: artifact_path("complete_objects", "amodal_result.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("generate_meshes"))
    asset_pre_rot_euler_xyz_deg: list[float] = field(default_factory=lambda: [-90.0, 0.0, 0.0])
    batch_size: int = 1
    guidance_scale: float = 3.0
    use_karras: bool = True
    karras_steps: int = 64
    sigma_min: float = 1e-3
    sigma_max: float = 160.0
    s_churn: int = 0
    transmitter_name: str = "transmitter"
    image_model_name: str = "image300M"


@dataclass
class DepthEstimationConfig:
    image_path: Path = field(default_factory=lambda: artifact_path("raw_image.jpg"))
    mask_result_json_path: Path = field(default_factory=lambda: artifact_path("generate_masks", "result.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("estimate_depth"))
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass
class CameraEstimationConfig:
    image_path: Path = field(default_factory=lambda: artifact_path("raw_image.jpg"))
    mask_result_json_path: Path = field(default_factory=lambda: artifact_path("generate_masks", "result.json"))
    depth_result_json_path: Path = field(default_factory=lambda: artifact_path("estimate_depth", "result.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("estimate_camera"))
    model_version: str = "Paramnet-360Cities-edina-uncentered"
    weights_path: Path = field(default_factory=lambda: project_path("external", "PerspectiveFields", "models", "paramnet_360cities_edina_rpfpp.pth"))
    config_file: str = "paramnet_360cities_edina_rpfpp.yaml"


@dataclass
class SceneLayoutConfig:
    mask_json_path: Path = field(default_factory=lambda: artifact_path("generate_masks", "result.json"))
    depth_json_path: Path = field(default_factory=lambda: artifact_path("estimate_depth", "result.json"))
    camera_json_path: Path = field(default_factory=lambda: artifact_path("estimate_camera", "result.json"))
    gen3d_json_path: Path = field(default_factory=lambda: artifact_path("generate_meshes", "gen3d_result.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("compose_layout"))
    raw_image_path: Path = field(default_factory=lambda: artifact_path("raw_image.jpg"))
    close_is_larger: bool = True
    z_near: float = 1.0
    z_far: float = 6.0
    focal_scale_x: float = 1.2
    focal_scale_y: float = 1.2
    global_scale_multiplier: float = 1.0
    save_visualization: bool = True


@dataclass
class SceneAssemblyConfig:
    input_layout_json_path: Path = field(default_factory=lambda: artifact_path("compose_layout", "scene_layout.json"))
    output_dir: Path = field(default_factory=lambda: artifact_path("assemble_scene"))
    add_axis_helper: bool = True
    axis_length: float = 0.6
    normalize_mesh_to_unit_box: bool = True
    global_pre_rot_euler_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    object_color_palette: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (0.90, 0.25, 0.25), (0.25, 0.55, 0.95), (0.20, 0.75, 0.35), (0.95, 0.70, 0.20),
        (0.65, 0.35, 0.90), (0.20, 0.80, 0.80), (0.95, 0.45, 0.70), (0.60, 0.60, 0.20),
        (0.90, 0.55, 0.15), (0.45, 0.75, 0.95),
    ])


@dataclass
class PipelineConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extract_tags: TagExtractionConfig = field(default_factory=TagExtractionConfig)
    generate_masks: MaskGenerationConfig = field(default_factory=MaskGenerationConfig)
    complete_objects: ObjectCompletionConfig = field(default_factory=ObjectCompletionConfig)
    generate_meshes: MeshGenerationConfig = field(default_factory=MeshGenerationConfig)
    estimate_depth: DepthEstimationConfig = field(default_factory=DepthEstimationConfig)
    estimate_camera: CameraEstimationConfig = field(default_factory=CameraEstimationConfig)
    compose_layout: SceneLayoutConfig = field(default_factory=SceneLayoutConfig)
    assemble_scene: SceneAssemblyConfig = field(default_factory=SceneAssemblyConfig)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))
