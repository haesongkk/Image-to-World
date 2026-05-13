from pathlib import Path
import tempfile
import unittest

from src.pipeline_dag import stage_deps_ready, stage_output_ready


class TestPipelineDag(unittest.TestCase):
    def test_stage_deps_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.txt"
            b = root / "b.txt"
            a.write_text("ok", encoding="utf-8")

            ready, missing = stage_deps_ready("segmentation", {"segmentation": [a, b]})
            self.assertFalse(ready)
            self.assertEqual(missing, [b])

    def test_generation_output_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            remesh = root / "remesh"
            mesh = root / "mesh"
            remesh.mkdir()
            mesh.mkdir()

            self.assertFalse(stage_output_ready("generation", {"generation": [remesh, mesh]}))
            (remesh / "foo_remeshed.glb").write_text("x", encoding="utf-8")
            self.assertTrue(stage_output_ready("generation", {"generation": [remesh, mesh]}))


if __name__ == "__main__":
    unittest.main()
