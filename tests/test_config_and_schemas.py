import unittest

from image_to_world.config import PipelineConfig
from image_to_world.schemas import MeshArtifact, Placement


class ConfigAndSchemaTests(unittest.TestCase):
    def test_pipeline_config_serializes_paths(self):
        config = PipelineConfig()
        payload = config.to_dict()
        self.assertIsInstance(payload["extract_tags"]["image_path"], str)

    def test_placement_to_dict(self):
        placement = Placement(
            id=1,
            class_name="chair",
            score=0.9,
            bbox_xyxy=[0, 0, 10, 10],
            bbox_center_xy=[5, 5],
            bbox_size_wh=[10, 10],
            depth_source="bbox_center_depth",
            relative_depth_value=0.5,
            pseudo_world={"position_xyz": [0, 0, 1], "scale_xyz": [1, 1, 1]},
            mesh=MeshArtifact(obj_path="a.obj", ply_path="a.ply"),
            source_paths={},
        )
        payload = placement.to_dict()
        self.assertEqual(payload["mesh"]["obj_path"], "a.obj")


if __name__ == "__main__":
    unittest.main()
