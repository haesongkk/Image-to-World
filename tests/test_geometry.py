import unittest

import numpy as np

from image_to_world.geometry import apply_transform, clamp, compose_transform, rotation_matrix_xyz_deg


class GeometryTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 3), 3)
        self.assertEqual(clamp(-1, 0, 3), 0)

    def test_rotation_matrix_shape(self):
        rot = rotation_matrix_xyz_deg([0, 0, 90])
        self.assertEqual(rot.shape, (3, 3))

    def test_compose_transform_and_apply(self):
        transform = compose_transform([2, 2, 2], [0, 0, 0], [1, 2, 3])
        vertices = np.array([[1.0, 1.0, 1.0]])
        result = apply_transform(vertices, transform)
        np.testing.assert_allclose(result[0], [3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
