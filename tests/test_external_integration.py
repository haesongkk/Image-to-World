from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from src.external.recognizeanything import run_recognizeanything


class TestExternalIntegration(unittest.TestCase):
    def test_recognizeanything_uses_runner(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_biref = root / "output" / "BirefNet"
            output_biref.mkdir(parents=True)
            (output_biref / "raw_image_birefnet.png").write_text("x", encoding="utf-8")

            output_ra = root / "output" / "recognize-anything"
            output_ra.mkdir(parents=True)

            fake_result = types.SimpleNamespace(
                stdout="Image Tags: bottle | blender",
                stderr="",
                returncode=0,
            )

            with patch("src.external.recognizeanything.OUTPUT_DIR", root / "output"), patch(
                "src.external.recognizeanything.THIRD_PARTY_DIR", root / "third_party"
            ), patch("src.external.recognizeanything.run_external_command", return_value=fake_result) as mock_run:
                run_recognizeanything(Path("data/raw_image.jpg"))
                self.assertTrue(mock_run.called)
                self.assertTrue((output_ra / "text_prompt.txt").exists())


if __name__ == "__main__":
    unittest.main()
