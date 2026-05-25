import unittest
from unittest import mock

import numpy as np
import trimesh

from camera_control.Method import io as io_module
from camera_control.Method.io import _sanitize_mesh, postProcessMesh


class TestSanitizeMeshFixWinding(unittest.TestCase):
    def _make_mesh_with_one_zero_area_face(self, n_good_faces: int) -> trimesh.Trimesh:
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        good_faces = np.column_stack(
            [
                np.zeros(n_good_faces, dtype=np.int64),
                np.ones(n_good_faces, dtype=np.int64),
                np.full(n_good_faces, 2, dtype=np.int64),
            ]
        )
        zero_area_face = np.array([[0, 0, 0]], dtype=np.int64)
        faces = np.vstack([good_faces, zero_area_face])
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    @mock.patch('trimesh.repair.fix_winding')
    def test_small_mesh_still_calls_fix_winding(self, mock_fix_winding):
        mesh = self._make_mesh_with_one_zero_area_face(n_good_faces=10)
        result = _sanitize_mesh(mesh, print_progress=False)
        self.assertIsNotNone(result)
        mock_fix_winding.assert_called_once()

    @mock.patch('trimesh.repair.fix_winding')
    def test_large_mesh_skips_fix_winding_by_default(self, mock_fix_winding):
        mesh = self._make_mesh_with_one_zero_area_face(
            n_good_faces=io_module._SANITIZE_FIX_WINDING_MAX_FACES
        )
        result = _sanitize_mesh(mesh, print_progress=False)
        self.assertIsNotNone(result)
        mock_fix_winding.assert_not_called()
        self.assertEqual(result.faces.shape[0], io_module._SANITIZE_FIX_WINDING_MAX_FACES)

    @mock.patch('trimesh.repair.fix_winding')
    def test_large_mesh_can_force_fix_winding(self, mock_fix_winding):
        mesh = self._make_mesh_with_one_zero_area_face(
            n_good_faces=io_module._SANITIZE_FIX_WINDING_MAX_FACES
        )
        result = _sanitize_mesh(mesh, print_progress=False, fix_winding=True)
        self.assertIsNotNone(result)
        mock_fix_winding.assert_called_once()

    @mock.patch('trimesh.repair.fix_winding')
    def test_post_process_mesh_forwards_fix_winding_override(self, mock_fix_winding):
        mesh = self._make_mesh_with_one_zero_area_face(n_good_faces=10)
        result = postProcessMesh(mesh, print_progress=False, fix_winding=False)
        self.assertIsNotNone(result)
        mock_fix_winding.assert_not_called()


if __name__ == '__main__':
    unittest.main()
