"""Tests for the refactored `VolumeMarker.markVisible`.

重构后的算法：
  1. 候选占据体素（points 优先，否则多相机 CCM 并集）；
  2. 候选体素并集的外露立方体面 mesh + triangle->voxel 映射；
  3. 用 nvdiffrast 渲染稳定遮挡 depth，命中的 face id 映射回 voxel -> VALID，
     被完全遮挡的候选保持 UNKNOWN；
  4. 非候选体素才做 FREE / UNKNOWN 判定。

纯逻辑（体素化 / 边界 mesh / points 归一化 / FREE 证据）在 CPU 即可验证；
需要真正渲染遮挡关系的端到端用例由 CUDA + nvdiffrast 守护。
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from camera_control.Module.volume_marker import VolumeMarker
from camera_control.Module.camera import Camera

try:
    import nvdiffrast.torch  # noqa: F401
    _NVDR_AVAILABLE = True
except Exception:  # pragma: no cover - 环境无 nvdiffrast 时直接跳过
    _NVDR_AVAILABLE = False

_RENDER_AVAILABLE = _NVDR_AVAILABLE and torch.cuda.is_available()


def _voxel_center(idx, R):
    """voxel (i, j, k) 在 [-0.5, 0.5]^3 中的中心世界坐标。"""
    return (np.asarray(idx, dtype=np.float64) + 0.5) / R - 0.5


class TestCandidateVoxelization(unittest.TestCase):
    """候选点收集 / 体素化 / points 归一化（纯逻辑，CPU）。"""

    def test_voxelize_points_maps_to_expected_voxel(self):
        R = 8
        # voxel (2, 3, 5) 的中心点。
        center = torch.from_numpy(
            np.stack([_voxel_center((2, 3, 5), R)]).astype(np.float32),
        )
        occ = VolumeMarker._voxelizePoints(center, R, 'cpu')
        self.assertEqual(occ.shape, (R, R, R))
        self.assertTrue(bool(occ[2, 3, 5]))
        self.assertEqual(int(occ.sum().item()), 1)

    def test_voxelize_points_filters_out_of_bounds_and_nan(self):
        R = 4
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],        # 中心 -> voxel (2,2,2)
                [10.0, 0.0, 0.0],       # 越界
                [float('nan'), 0.0, 0.0],  # 非有限
            ],
            dtype=torch.float32,
        )
        occ = VolumeMarker._voxelizePoints(pts, R, 'cpu')
        self.assertEqual(int(occ.sum().item()), 1)
        self.assertTrue(bool(occ[2, 2, 2]))

    def test_voxelize_empty_points(self):
        R = 4
        occ = VolumeMarker._voxelizePoints(
            torch.zeros((0, 3), dtype=torch.float32), R, 'cpu',
        )
        self.assertEqual(int(occ.sum().item()), 0)

    def test_normalize_points_from_numpy(self):
        pts = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32)
        out = VolumeMarker._normalizePoints(pts, torch.float32, 'cpu')
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertTrue(torch.allclose(out[0], torch.tensor([0.1, 0.2, 0.3])))

    def test_normalize_points_truncates_extra_columns(self):
        # (N, 6) 形如 xyz + rgb，只取前 3 列。
        pts = np.array([[0.1, 0.2, 0.3, 1.0, 1.0, 1.0]], dtype=np.float32)
        out = VolumeMarker._normalizePoints(pts, torch.float32, 'cpu')
        self.assertEqual(tuple(out.shape), (1, 3))

    def test_normalize_points_from_npy_file(self):
        pts = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'pts.npy')
            np.save(path, pts)
            out = VolumeMarker._normalizePoints(path, torch.float32, 'cpu')
        self.assertEqual(tuple(out.shape), (1, 3))
        self.assertTrue(torch.allclose(out[0], torch.tensor([0.1, 0.2, 0.3])))

    def test_normalize_points_from_npz_file(self):
        pts = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'pts.npz')
            np.savez(path, points=pts)
            out = VolumeMarker._normalizePoints(path, torch.float32, 'cpu')
        self.assertEqual(tuple(out.shape), (1, 3))
        self.assertTrue(torch.allclose(out[0], torch.tensor([0.4, 0.5, 0.6])))

    def test_normalize_points_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            VolumeMarker._normalizePoints(
                '/nonexistent/path/pts.npy', torch.float32, 'cpu',
            )


class TestBoundaryMesh(unittest.TestCase):
    """候选体素并集边界 mesh 与 triangle->voxel 映射（纯逻辑，CPU）。"""

    def test_single_voxel_has_six_faces_twelve_triangles(self):
        R = 4
        candidate = torch.zeros((R, R, R), dtype=torch.bool)
        candidate[1, 1, 1] = True

        verts, faces, tri2vox = VolumeMarker._buildVoxelBoundaryMesh(candidate, R)

        # 单个孤立体素：6 个外露面 -> 12 个三角。
        self.assertEqual(faces.shape[0], 12)
        self.assertEqual(tri2vox.shape[0], 12)

        # 全部三角都映射到同一个 voxel flat index。
        expected_flat = (1 * R + 1) * R + 1
        self.assertTrue(torch.all(tri2vox == expected_flat))

        # 顶点应落在 voxel (1,1,1) 的立方体角点范围内。
        lo = -0.5 + 1.0 / R
        hi = -0.5 + 2.0 / R
        self.assertTrue(torch.all(verts >= lo - 1e-6))
        self.assertTrue(torch.all(verts <= hi + 1e-6))

    def test_internal_faces_between_neighbors_are_skipped(self):
        R = 4
        candidate = torch.zeros((R, R, R), dtype=torch.bool)
        candidate[1, 1, 1] = True
        candidate[2, 1, 1] = True  # 沿 +x 相邻

        verts, faces, tri2vox = VolumeMarker._buildVoxelBoundaryMesh(candidate, R)

        # 2 个相邻体素：共 12 个面，但相邻处的 2 个内部面被跳过 -> 10 面 -> 20 三角。
        self.assertEqual(faces.shape[0], 20)
        self.assertEqual(tri2vox.shape[0], 20)

        flat_a = (1 * R + 1) * R + 1
        flat_b = (2 * R + 1) * R + 1
        unique_voxels = set(tri2vox.tolist())
        self.assertEqual(unique_voxels, {flat_a, flat_b})

    def test_empty_candidate_returns_empty_mesh(self):
        R = 4
        candidate = torch.zeros((R, R, R), dtype=torch.bool)
        verts, faces, tri2vox = VolumeMarker._buildVoxelBoundaryMesh(candidate, R)
        self.assertEqual(verts.shape[0], 0)
        self.assertEqual(faces.shape[0], 0)
        self.assertEqual(tri2vox.shape[0], 0)

    def test_face_indices_in_bounds(self):
        R = 4
        candidate = torch.zeros((R, R, R), dtype=torch.bool)
        candidate[0, 0, 0] = True
        candidate[3, 3, 3] = True

        verts, faces, _ = VolumeMarker._buildVoxelBoundaryMesh(candidate, R)
        self.assertTrue(int(faces.max().item()) < verts.shape[0])
        self.assertTrue(int(faces.min().item()) >= 0)


class TestFreeEvidence(unittest.TestCase):
    """非候选体素的 FREE 判定（需要构造带 depth 的相机，CPU 即可）。"""

    def _make_camera_seeing_plane(self, plane_depth: float) -> Camera:
        """相机位于 (0,0,2) 看向 -Z，整幅 depth 都是恒定 plane_depth。

        plane_depth 是相机前方正距离；plane 在世界 z = 2 - plane_depth。
        """
        camera = Camera(
            width=16,
            height=16,
            fovx_degree=60.0,
            pos=[0.0, 0.0, 2.0],
            look_at=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            device='cpu',
        )
        depth = torch.full((16, 16), float(plane_depth), dtype=torch.float32)
        camera.loadDepth(depth)
        return camera

    def test_voxels_in_front_of_surface_are_free(self):
        R = 8
        # 表面在相机前方 2.4（世界 z = -0.4），整个 [-0.5,0.5]^3 体素都在表面前方。
        camera = self._make_camera_seeing_plane(plane_depth=2.4)

        free_any, observed_any = VolumeMarker._computeFreeEvidence(
            [camera], R, 'cpu',
        )

        # 至少存在被观测到且判为 FREE 的体素（投影落在画面内的中心区域）。
        self.assertTrue(bool(observed_any.any()))
        self.assertTrue(bool(free_any.any()))
        # FREE 的体素必然先 observed。
        self.assertTrue(bool((free_any & observed_any).any()))
        self.assertFalse(bool((free_any & (~observed_any)).any()))

    def test_background_rays_count_as_free(self):
        R = 8
        # 全背景 depth=0：画面内的体素 8 角都是背景射线 -> FREE。
        camera = self._make_camera_seeing_plane(plane_depth=0.0)

        free_any, observed_any = VolumeMarker._computeFreeEvidence(
            [camera], R, 'cpu',
        )
        self.assertTrue(bool(observed_any.any()))
        self.assertTrue(bool(free_any.any()))


class TestMarkVisibleNoCamera(unittest.TestCase):
    def test_empty_camera_list_returns_all_free(self):
        R = 5
        labels = VolumeMarker.markVisible([], R)
        self.assertEqual(tuple(labels.shape), (R, R, R))
        self.assertTrue(torch.all(labels == VolumeMarker.FREE))
        self.assertEqual(labels.dtype, torch.int64)


@unittest.skipUnless(
    _RENDER_AVAILABLE,
    'markVisible 渲染路径仅在 CUDA + nvdiffrast 可用时有效',
)
class TestMarkVisibleRender(unittest.TestCase):
    """需要真正渲染遮挡关系的端到端用例。"""

    def _make_front_camera(self, R: int = 16) -> Camera:
        camera = Camera(
            width=128,
            height=128,
            fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0],
            look_at=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            device='cuda:0',
        )
        depth = torch.full((128, 128), 0.0, dtype=torch.float32, device='cuda:0')
        camera.loadDepth(depth)
        return camera

    def test_occluded_candidate_stays_unknown(self):
        """相机 +Z 看向 -Z：前面体素遮挡后面体素，后面应为 UNKNOWN 而非 VALID。"""
        R = 16
        device = 'cuda:0'

        # 沿视线方向（z 轴）排列两个体素：靠近相机的 (8,8,12)，被遮挡的 (8,8,3)。
        front_idx = (8, 8, 12)
        back_idx = (8, 8, 3)

        pts = torch.from_numpy(
            np.stack([
                _voxel_center(front_idx, R),
                _voxel_center(back_idx, R),
            ]).astype(np.float32),
        ).to(device)

        camera = self._make_front_camera(R)
        labels = VolumeMarker.markVisible([camera], R, points=pts)

        self.assertEqual(int(labels[front_idx].item()), VolumeMarker.VALID)
        # 后面体素被前面完全遮挡，保持 UNKNOWN（不会被改成 VALID 或 FREE）。
        self.assertEqual(int(labels[back_idx].item()), VolumeMarker.UNKNOWN)

    def test_multi_view_union_of_visible_candidates(self):
        """两个相反方向的相机分别看到前/后体素，VALID 取并集。"""
        R = 16
        device = 'cuda:0'

        front_idx = (8, 8, 12)
        back_idx = (8, 8, 3)
        pts = torch.from_numpy(
            np.stack([
                _voxel_center(front_idx, R),
                _voxel_center(back_idx, R),
            ]).astype(np.float32),
        ).to(device)

        cam_front = self._make_front_camera(R)

        cam_back = Camera(
            width=128, height=128, fovx_degree=60.0,
            pos=[0.0, 0.0, -3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device=device,
        )
        cam_back.loadDepth(torch.zeros((128, 128), dtype=torch.float32, device=device))

        labels = VolumeMarker.markVisible([cam_front, cam_back], R, points=pts)

        self.assertEqual(int(labels[front_idx].item()), VolumeMarker.VALID)
        self.assertEqual(int(labels[back_idx].item()), VolumeMarker.VALID)

    def test_free_does_not_overwrite_candidate(self):
        """单个可见候选 voxel 被标 VALID 后，FREE 逻辑不得把它降级。"""
        R = 16
        device = 'cuda:0'
        idx = (8, 8, 12)
        pts = torch.from_numpy(
            np.stack([_voxel_center(idx, R)]).astype(np.float32),
        ).to(device)

        camera = self._make_front_camera(R)
        labels = VolumeMarker.markVisible([camera], R, points=pts)

        self.assertEqual(int(labels[idx].item()), VolumeMarker.VALID)


def test():
    unittest.main()


if __name__ == '__main__':
    unittest.main()
