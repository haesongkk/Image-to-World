from pathlib import Path
import json
import tempfile
import unittest

from src.manifest import write_run_manifest
from src.pipeline_types import StageResult


class TestManifestRegression(unittest.TestCase):
    def test_manifest_created_with_stage_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_image = root / "raw_image.jpg"
            input_image.write_bytes(b"fake-image")
            manifest_path = root / "output" / "run_manifest.json"

            results = [
                StageResult(stage="segmentation", skipped=False, duration_sec=1.2, outputs=[root / "a.txt"]),
                StageResult(stage="generation", skipped=True, duration_sec=0.0, outputs=[]),
            ]

            write_run_manifest(
                output_path=manifest_path,
                input_image=input_image,
                stage_results=results,
                stage="all",
                resume=True,
                project_root=root,
                started_at="2026-05-13T00:00:00+00:00",
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["requested_stage"], "all")
            self.assertTrue(payload["resume"])
            self.assertEqual(payload["stages"][0]["stage"], "segmentation")
            self.assertIn("input_image_sha256", payload)


if __name__ == "__main__":
    unittest.main()
