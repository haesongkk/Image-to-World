import tempfile
import unittest
from pathlib import Path

from image_to_world.io.obj_utils import load_obj_basic


class ObjUtilsTests(unittest.TestCase):
    def test_load_obj_basic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "simple.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            data = load_obj_basic(path)
            self.assertEqual(data["vertices"].shape[0], 3)
            self.assertEqual(len(data["faces_v"]), 1)


if __name__ == "__main__":
    unittest.main()
