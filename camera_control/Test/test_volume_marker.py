"""Tests for the refactored `VolumeMarker.markVisible`.

重构后的算法：
  1. 候选占据体素（points 优先，否则多相机 CCM 并集）；
  2. 候选体素并集的外露立方体面 mesh + triangle->voxel 映射；
  3. 用 nvdiffrast 渲染稳定遮挡 depth，命中的 face id 映射回 voxel -> VALID，
     被完全遮挡的候选保持 UNKNOWN；
  4. 非候选体素才做 FREE / UNKNOWN 判定；FREE voxel 全量编码为
     FREE_KN = -K（K = 到最近非 FREE voxel（VALID 或 UNKNOWN）的 L1 距离），
     全 FREE 时为 -(3R)。

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


class TestMeshVoxelization(unittest.TestCase):
    """三角网格按分辨率求占用（open3d 表面体采样 + 体素化，需 open3d）。"""

    def test_triangle_overlap_single_voxel(self):
        # 一个完全位于 voxel (2,2,2) 内部的小三角形，仅占据该体素。
        R = 4
        c = _voxel_center((2, 2, 2), R)  # voxel 中心
        d = 0.5 / R * 0.5  # 远小于半个体素
        verts = torch.tensor(
            [
                [c[0] - d, c[1] - d, c[2]],
                [c[0] + d, c[1] - d, c[2]],
                [c[0], c[1] + d, c[2]],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        self.assertTrue(bool(occ[2, 2, 2]))
        self.assertEqual(int(occ.sum().item()), 1)

    def test_triangle_spanning_multiple_voxels(self):
        # 一个横跨多个体素的大三角形：占据的体素数应明显多于 1，且为表面占用。
        R = 8
        verts = torch.tensor(
            [
                [-0.4, -0.4, 0.0],
                [0.4, -0.4, 0.0],
                [0.0, 0.4, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        self.assertGreater(int(occ.sum().item()), 1)
        # 三角形位于 z=0 平面，只应占据 z 方向跨 0 的那一层体素（k=R/2-1 或 R/2）。
        occupied_k = torch.nonzero(occ).unique(dim=0)[:, 2].unique()
        self.assertTrue(set(occupied_k.tolist()).issubset({R // 2 - 1, R // 2}))

    def test_mesh_voxelization_is_surface_not_filled(self):
        # 用一个轴对齐 box mesh：表面体素化应是空心壳，内部体素不被占据。
        import trimesh as _tm

        R = 16
        box = _tm.creation.box(extents=(0.5, 0.5, 0.5))  # 居中 [-0.25,0.25]^3
        verts = torch.as_tensor(np.asarray(box.vertices), dtype=torch.float32)
        faces = torch.as_tensor(np.asarray(box.faces), dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')

        # 几何中心 voxel 必为空（表面占用而非实心填充）。
        center_idx = R // 2
        self.assertFalse(bool(occ[center_idx, center_idx, center_idx]))
        self.assertGreater(int(occ.sum().item()), 0)

    def test_geometry_dispatch_trimesh_mesh_uses_mesh_voxelization(self):
        import trimesh as _tm

        R = 8
        box = _tm.creation.box(extents=(0.4, 0.4, 0.4))
        occ = VolumeMarker._geometryToOccupancy(box, R, torch.float32, 'cpu')
        self.assertEqual(tuple(occ.shape), (R, R, R))
        self.assertGreater(int(occ.sum().item()), 0)
        # mesh 为空心壳：中心体素应为空。
        self.assertFalse(bool(occ[R // 2, R // 2, R // 2]))

    def test_geometry_dispatch_pointcloud_uses_point_voxelization(self):
        import trimesh as _tm

        R = 8
        pc = _tm.points.PointCloud(
            np.stack([_voxel_center((3, 4, 5), R)]).astype(np.float64),
        )
        occ = VolumeMarker._geometryToOccupancy(pc, R, torch.float32, 'cpu')
        self.assertTrue(bool(occ[3, 4, 5]))
        self.assertEqual(int(occ.sum().item()), 1)

    def test_geometry_dispatch_numpy_points(self):
        R = 8
        pts = np.stack([_voxel_center((1, 2, 6), R)]).astype(np.float32)
        occ = VolumeMarker._geometryToOccupancy(pts, R, torch.float32, 'cpu')
        self.assertTrue(bool(occ[1, 2, 6]))
        self.assertEqual(int(occ.sum().item()), 1)

    def test_is_triangle_mesh_classification(self):
        import trimesh as _tm

        box = _tm.creation.box(extents=(0.4, 0.4, 0.4))
        self.assertTrue(VolumeMarker._isTriangleMesh(box))
        pc = _tm.points.PointCloud(np.zeros((3, 3)))
        self.assertFalse(VolumeMarker._isTriangleMesh(pc))
        self.assertFalse(VolumeMarker._isTriangleMesh(np.zeros((3, 3))))

    def test_sample_count_grows_with_resolution_and_area(self):
        # 采样点数应随分辨率与面积单调不减（密度保证的基础）。
        n_lo = VolumeMarker._meshSampleCount(area=1.0, R=8, K=3)
        n_hi = VolumeMarker._meshSampleCount(area=1.0, R=32, K=3)
        self.assertGreaterEqual(n_hi, n_lo)
        n_small = VolumeMarker._meshSampleCount(area=0.1, R=16, K=3)
        n_large = VolumeMarker._meshSampleCount(area=10.0, R=16, K=3)
        self.assertGreaterEqual(n_large, n_small)
        # 上限生效。
        capped = VolumeMarker._meshSampleCount(area=1e12, R=64, K=3)
        self.assertLessEqual(capped, VolumeMarker.MESH_SAMPLE_MAX_POINTS)

    def test_axis_aligned_quad_covers_all_crossed_voxels(self):
        """完整覆盖回归：一个 z=const 的平面四边形（两三角形）应激活其 xy
        投影范围内、该 z 所在层的全部体素，不漏任何被网格穿过的体素。
        """
        R = 16
        # 平面位于某个体素层中心 z，避免落在体素边界产生歧义。
        k = 6
        z = (k + 0.5) / R - 0.5
        # xy 覆盖 [lo, hi]，恰好对应体素 i,j ∈ [i_lo, i_hi]。
        i_lo, i_hi = 3, 11
        lo = (i_lo) / R - 0.5 + 1e-3
        hi = (i_hi + 1) / R - 0.5 - 1e-3
        verts = torch.tensor(
            [
                [lo, lo, z],
                [hi, lo, z],
                [hi, hi, z],
                [lo, hi, z],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')

        # 该 z 层 [i_lo, i_hi] x [i_lo, i_hi] 的每个体素都应被激活。
        for i in range(i_lo, i_hi + 1):
            for j in range(i_lo, i_hi + 1):
                self.assertTrue(
                    bool(occ[i, j, k]),
                    f'voxel ({i},{j},{k}) 被网格穿过却未被采样覆盖',
                )
        # 不应越出到相邻 z 层（平面 z 固定）。
        occupied_k = torch.nonzero(occ)[:, 2].unique().tolist()
        self.assertEqual(occupied_k, [k])


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


class TestVoxelCenters(unittest.TestCase):
    """voxel 中心坐标（纯逻辑，CPU）。"""

    def test_centers_match_voxelization_convention(self):
        R = 8
        centers = VolumeMarker.createVoxelCenters(R, torch.float32, 'cpu')
        self.assertEqual(tuple(centers.shape), (R, R, R, 3))

        # voxel (2, 3, 5) 中心反体素化应回到自身。
        c = centers[2, 3, 5]
        occ = VolumeMarker._voxelizePoints(c.reshape(1, 3), R, 'cpu')
        self.assertTrue(bool(occ[2, 3, 5]))
        self.assertEqual(int(occ.sum().item()), 1)

    def test_centers_in_unit_cube(self):
        R = 4
        centers = VolumeMarker.createVoxelCenters(R, torch.float32, 'cpu')
        self.assertTrue(torch.all(centers >= -0.5))
        self.assertTrue(torch.all(centers <= 0.5))


class TestSampleRenderedDepth(unittest.TestCase):
    """渲染深度采样 + 前/后/背景分类（纯逻辑，CPU）。

    这是新 FREE 判定的核心：voxel 中心相对候选 mesh 渲染表面深度的位置，
    决定该视角是 FREE（更近 / 背景）还是无信息（被遮挡）。
    """

    def _make_depth(self, H, W, surface_depth, hit_value=True):
        rendered = torch.full((H, W), float(surface_depth), dtype=torch.float32)
        hit = torch.full((H, W), bool(hit_value), dtype=torch.bool)
        return rendered, hit

    def test_voxel_in_front_of_surface_is_free(self):
        H = W = 8
        rendered, hit = self._make_depth(H, W, surface_depth=5.0)
        # 一个落在画面中心的 voxel UV，深度 2.0 < 表面 5.0 -> 更近。
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        sampled, surface_hit, background = VolumeMarker._sampleRenderedDepth(
            rendered, hit, uv, H, W,
        )
        self.assertTrue(bool(surface_hit[0]))
        self.assertFalse(bool(background[0]))
        self.assertAlmostEqual(float(sampled[0]), 5.0, places=5)
        # in_front 判定（与 _renderClassify 内一致）：point_depth < sampled - eps
        self.assertTrue(2.0 < float(sampled[0]))

    def test_voxel_behind_surface_not_free(self):
        H = W = 8
        rendered, hit = self._make_depth(H, W, surface_depth=2.0)
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        sampled, surface_hit, background = VolumeMarker._sampleRenderedDepth(
            rendered, hit, uv, H, W,
        )
        self.assertTrue(bool(surface_hit[0]))
        # voxel 深度 5.0 > 表面 2.0 -> 被遮挡，不应判 FREE。
        self.assertFalse(5.0 < float(sampled[0]))

    def test_background_pixel_is_free(self):
        H = W = 8
        rendered, hit = self._make_depth(H, W, surface_depth=0.0, hit_value=False)
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        sampled, surface_hit, background = VolumeMarker._sampleRenderedDepth(
            rendered, hit, uv, H, W,
        )
        self.assertFalse(bool(surface_hit[0]))
        self.assertTrue(bool(background[0]))

    def test_out_of_frame_is_neither(self):
        H = W = 8
        rendered, hit = self._make_depth(H, W, surface_depth=2.0)
        uv = torch.tensor(
            [[1.5, 0.5], [float('nan'), 0.5]], dtype=torch.float32,
        )
        sampled, surface_hit, background = VolumeMarker._sampleRenderedDepth(
            rendered, hit, uv, H, W,
        )
        self.assertFalse(bool(surface_hit[0]))
        self.assertFalse(bool(background[0]))
        self.assertFalse(bool(surface_hit[1]))
        self.assertFalse(bool(background[1]))


class TestVoxelHelpers(unittest.TestCase):
    """重构抽出的体素/网格原子函数（纯逻辑，CPU）。"""

    def test_require_positive_resolution_ok(self):
        self.assertEqual(VolumeMarker._requirePositiveResolution(7, 'x'), 7)
        # 浮点也应被 int 化。
        self.assertEqual(VolumeMarker._requirePositiveResolution(4.0, 'x'), 4)

    def test_require_positive_resolution_raises(self):
        for bad in (0, -1, -5):
            with self.assertRaises(ValueError):
                VolumeMarker._requirePositiveResolution(bad, 'x')

    def test_empty_labels_value_dtype_device(self):
        R = 3
        free_inf = VolumeMarker._emptyLabels(R, VolumeMarker.freeLabelInf(R))
        unknown = VolumeMarker._emptyLabels(R, VolumeMarker.UNKNOWN, 'cpu')
        self.assertEqual(tuple(free_inf.shape), (R, R, R))
        self.assertEqual(free_inf.dtype, torch.int64)
        self.assertTrue(torch.all(free_inf == VolumeMarker.freeLabelInf(R)))
        self.assertTrue(torch.all(unknown == VolumeMarker.UNKNOWN))

    def test_voxel_indices_to_flat_matches_manual_formula(self):
        R = 8
        idx = torch.tensor([[2, 3, 5], [0, 0, 0], [7, 7, 7]])
        flat = VolumeMarker._voxelIndicesToFlat(idx, R)
        expected = (idx[:, 0] * R + idx[:, 1]) * R + idx[:, 2]
        self.assertTrue(torch.equal(flat, expected))

    def test_points_to_voxel_indices_clamps_into_range(self):
        R = 4
        # +0.5 应落在最后一个体素（clamp 到 R-1），-0.5 落在第 0 个。
        pts = torch.tensor(
            [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]], dtype=torch.float32,
        )
        idx = VolumeMarker._pointsToVoxelIndices(pts, R)
        self.assertTrue(torch.equal(idx[0], torch.tensor([R - 1, R - 1, R - 1])))
        self.assertTrue(torch.equal(idx[1], torch.tensor([0, 0, 0])))

    def test_finite_in_bounds_includes_boundaries(self):
        pts = torch.tensor(
            [
                [0.5, 0.5, 0.5],    # 上边界 -> 内
                [-0.5, -0.5, -0.5],  # 下边界 -> 内
                [0.51, 0.0, 0.0],   # 越界
                [float('nan'), 0.0, 0.0],  # 非有限
            ],
            dtype=torch.float32,
        )
        mask = VolumeMarker._finiteInBounds(pts)
        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))
        self.assertFalse(bool(mask[2]))
        self.assertFalse(bool(mask[3]))

    def test_make_ijk_grid_shape_and_ordering(self):
        coords = torch.tensor([0.0, 1.0, 2.0])
        grid = VolumeMarker._makeIjkGrid(coords)
        self.assertEqual(tuple(grid.shape), (3, 3, 3, 3))
        # (x, y, z) ij 顺序：grid[i, j, k] == (coords[i], coords[j], coords[k])。
        self.assertTrue(torch.equal(grid[1, 2, 0], torch.tensor([1.0, 2.0, 0.0])))


class TestVoxelizeBoundary(unittest.TestCase):
    """体素化在 ±0.5 边界上的稳定性（纯逻辑，CPU）。"""

    def test_boundary_points_clamp_into_volume(self):
        R = 4
        pts = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ],
            dtype=torch.float32,
        )
        occ = VolumeMarker._voxelizePoints(pts, R, 'cpu')
        self.assertEqual(int(occ.sum().item()), 2)
        self.assertTrue(bool(occ[R - 1, R - 1, R - 1]))
        self.assertTrue(bool(occ[0, 0, 0]))


class TestFreeNeighborLevels(unittest.TestCase):
    """L1 距离场与 FREE_KN 标签赋值（纯逻辑，CPU）。"""

    def test_compute_l1_distance_to_mask(self):
        R = 5
        source = torch.zeros((R, R, R), dtype=torch.bool)
        source[2, 2, 2] = True
        dist = VolumeMarker.computeL1DistanceToMask(source)
        self.assertEqual(dist.dtype, torch.int64)
        self.assertEqual(int(dist[2, 2, 2].item()), 0)
        self.assertEqual(int(dist[3, 2, 2].item()), 1)
        self.assertEqual(int(dist[2, 1, 2].item()), 1)
        self.assertEqual(int(dist[4, 2, 2].item()), 2)
        self.assertEqual(int(dist[3, 3, 2].item()), 2)
        self.assertEqual(int(dist[2, 2, 0].item()), 2)
        # 全量精确 L1 距离：角点 (0,0,0) 到 (2,2,2) 为 6。
        self.assertEqual(int(dist[0, 0, 0].item()), 6)
        self.assertEqual(int(dist[4, 4, 4].item()), 6)
        self.assertEqual(int(dist[4, 0, 2].item()), 4)

    def test_compute_l1_distance_exact_against_bruteforce(self):
        # 随机 source 与暴力逐点最小 L1 距离对比，验证可分离扫描的精确性。
        torch.manual_seed(0)
        R = 7
        source = torch.rand((R, R, R)) < 0.05
        source[1, 2, 3] = True  # 保证非空
        dist = VolumeMarker.computeL1DistanceToMask(source)

        src_idx = torch.nonzero(source).to(torch.int64)  # (S, 3)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(R), torch.arange(R), torch.arange(R),
                indexing='ij',
            ),
            dim=-1,
        ).reshape(-1, 3)  # (R^3, 3)
        brute = (
            (grid[:, None, :] - src_idx[None, :, :]).abs().sum(dim=-1).min(dim=1).values
        ).reshape(R, R, R)
        self.assertTrue(torch.equal(dist, brute))

    def test_compute_l1_distance_empty_source_is_all_inf(self):
        R = 4
        source = torch.zeros((R, R, R), dtype=torch.bool)
        dist = VolumeMarker.computeL1DistanceToMask(source)
        self.assertTrue(torch.all(dist == 3 * R))

    def test_assign_free_labels_basic(self):
        R = 5
        labels = torch.full((R, R, R), VolumeMarker.UNKNOWN, dtype=torch.int64)
        labels[2, 2, 2] = VolumeMarker.VALID

        free_mask = torch.ones((R, R, R), dtype=torch.bool)
        free_mask[2, 2, 2] = False  # VALID
        free_mask[2, 2, 3] = False  # 保持 UNKNOWN

        out = VolumeMarker.assignFreeLabels(labels, free_mask)

        # VALID / UNKNOWN 不动。
        self.assertEqual(int(out[2, 2, 2].item()), VolumeMarker.VALID)
        self.assertEqual(int(out[2, 2, 3].item()), VolumeMarker.UNKNOWN)
        # 距离 1 的 FREE -> -1。
        self.assertEqual(int(out[3, 2, 2].item()), VolumeMarker.freeLabelKN(1))
        self.assertEqual(int(out[2, 1, 2].item()), VolumeMarker.freeLabelKN(1))
        # 距离 2 的 FREE -> -2。
        self.assertEqual(int(out[4, 2, 2].item()), VolumeMarker.freeLabelKN(2))
        # UNKNOWN 也是距离源：(2,2,4) 紧贴 UNKNOWN (2,2,3) -> -1
        # （而非到 VALID (2,2,2) 的距离 2）。
        self.assertEqual(int(out[2, 2, 4].item()), VolumeMarker.freeLabelKN(1))
        # 任意距离全量编码：角点 (0,0,0) 到最近源 (2,2,2) 距离 6 -> -6。
        self.assertEqual(int(out[0, 0, 0].item()), VolumeMarker.freeLabelKN(6))
        # 输入不被原地修改。
        self.assertEqual(int(labels[3, 2, 2].item()), VolumeMarker.UNKNOWN)

    def test_assign_free_labels_unknown_is_distance_source(self):
        # 没有任何 VALID，只有一个 UNKNOWN：FREE 的 K 按到该 UNKNOWN 的
        # L1 距离编码（回归：紧贴 UNKNOWN 的 FREE 的 K 不应过大）。
        R = 3
        labels = torch.full((R, R, R), VolumeMarker.UNKNOWN, dtype=torch.int64)
        free_mask = torch.ones((R, R, R), dtype=torch.bool)
        free_mask[0, 0, 0] = False

        out = VolumeMarker.assignFreeLabels(labels, free_mask)

        self.assertEqual(int(out[0, 0, 0].item()), VolumeMarker.UNKNOWN)
        self.assertEqual(int(out[1, 0, 0].item()), VolumeMarker.freeLabelKN(1))
        self.assertEqual(int(out[1, 1, 0].item()), VolumeMarker.freeLabelKN(2))
        self.assertEqual(int(out[2, 2, 2].item()), VolumeMarker.freeLabelKN(6))

    def test_assign_free_labels_all_free_is_inf_sentinel(self):
        # 整个空间都是 FREE（无任何 VALID/UNKNOWN）-> 全部哨兵 -(3R)。
        R = 3
        labels = torch.full((R, R, R), VolumeMarker.UNKNOWN, dtype=torch.int64)
        free_mask = torch.ones((R, R, R), dtype=torch.bool)

        out = VolumeMarker.assignFreeLabels(labels, free_mask)

        self.assertTrue(torch.all(out == VolumeMarker.freeLabelInf(R)))

    def test_free_label_kn_encoding(self):
        self.assertEqual(VolumeMarker.freeLabelKN(1), -1)
        self.assertEqual(VolumeMarker.freeLabelKN(2), -2)
        self.assertEqual(VolumeMarker.freeLabelKN(3), -3)
        with self.assertRaises(ValueError):
            VolumeMarker.freeLabelKN(0)
        self.assertEqual(VolumeMarker.freeLabelInf(16), -48)
        with self.assertRaises(ValueError):
            VolumeMarker.freeLabelInf(0)


class TestMarkVisibleNoCamera(unittest.TestCase):
    def test_empty_camera_list_returns_all_free_inf(self):
        R = 5
        labels = VolumeMarker.markVisible([], R)
        self.assertEqual(tuple(labels.shape), (R, R, R))
        self.assertTrue(torch.all(labels == VolumeMarker.freeLabelInf(R)))
        self.assertEqual(labels.dtype, torch.int64)

    def test_non_positive_resolution_raises(self):
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                VolumeMarker.markVisible([], bad)


class TestMarkVisibleCpuCandidate(unittest.TestCase):
    """无需渲染即可在 CPU 验证的候选/深度路径。"""

    def _make_cpu_camera(self, depth_value: float) -> Camera:
        camera = Camera(
            width=8, height=8, fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device='cpu',
        )
        depth = torch.full((8, 8), float(depth_value), dtype=torch.float32)
        camera.loadDepth(depth)
        return camera

    def test_missing_depth_raises(self):
        camera = Camera(
            width=8, height=8, fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device='cpu',
        )
        with self.assertRaises(ValueError):
            VolumeMarker.markVisible([camera], 4)

    def test_points_none_with_no_valid_depth_is_all_free(self):
        # depth 全 0 -> valid_depth_mask 全 False -> CCM 候选为空
        # -> 全 FREE（无 VALID -> 哨兵 -(3R)）。
        R = 4
        camera = self._make_cpu_camera(depth_value=0.0)
        labels = VolumeMarker.markVisible([camera], R)
        self.assertEqual(tuple(labels.shape), (R, R, R))
        self.assertTrue(torch.all(labels == VolumeMarker.freeLabelInf(R)))

    def test_empty_points_is_all_free(self):
        R = 4
        camera = self._make_cpu_camera(depth_value=0.0)
        empty_pts = torch.zeros((0, 3), dtype=torch.float32)
        labels = VolumeMarker.markVisible([camera], R, geometry=empty_pts)
        self.assertTrue(torch.all(labels == VolumeMarker.freeLabelInf(R)))


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
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)

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

        labels = VolumeMarker.markVisible([cam_front, cam_back], R, geometry=pts)

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
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)

        self.assertEqual(int(labels[idx].item()), VolumeMarker.VALID)

    def test_voxels_closer_than_valid_are_free_not_unknown(self):
        """核心回归：单视角下，比 VALID voxel 更靠近相机的 voxel 必为 FREE。

        相机在 +Z 看向 -Z，世界 z 越大越靠近相机。把唯一的候选体素放在远处
        （小 z 索引），那么沿同一条视线、位于它与相机之间（大 z 索引）的所有
        非候选 voxel 都应是 FREE，绝不能是 UNKNOWN。
        """
        R = 16
        device = 'cuda:0'

        valid_idx = (8, 8, 3)  # 远处候选体素
        pts = torch.from_numpy(
            np.stack([_voxel_center(valid_idx, R)]).astype(np.float32),
        ).to(device)

        camera = self._make_front_camera(R)
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)

        self.assertEqual(int(labels[valid_idx].item()), VolumeMarker.VALID)

        # 同一条视线上、比 VALID 更靠近相机的非候选 voxel（z 索引更大）必为
        # FREE_KN（负值），且 K 即到 VALID 的 L1 距离（同一条线 -> k - 3）。
        for k in range(valid_idx[2] + 1, R):
            label = int(labels[valid_idx[0], valid_idx[1], k].item())
            self.assertEqual(
                label, VolumeMarker.freeLabelKN(k - valid_idx[2]),
                f'voxel (8,8,{k}) 比 VALID (8,8,3) 更靠近相机，应为 FREE_KN，'
                f'实际 label={label}',
            )

        # 不应在整个体积中出现「比某个 VALID 更近却 UNKNOWN」的矛盾：
        # 这里直接断言不存在 UNKNOWN 落在 VALID 前方的视线段上。
        front_segment = labels[valid_idx[0], valid_idx[1], valid_idx[2] + 1:]
        self.assertFalse(
            bool((front_segment == VolumeMarker.UNKNOWN).any().item()),
            'VALID 前方视线段不应出现 UNKNOWN',
        )

    def test_no_candidate_means_all_free(self):
        """无候选体素（points 为空）时，整个空间应为 FREE（无 VALID -> 哨兵）。"""
        R = 8
        device = 'cuda:0'
        camera = self._make_front_camera(R)
        empty_pts = torch.zeros((0, 3), dtype=torch.float32, device=device)
        labels = VolumeMarker.markVisible([camera], R, geometry=empty_pts)
        self.assertTrue(torch.all(labels == VolumeMarker.freeLabelInf(R)))


def test():
    unittest.main()


if __name__ == '__main__':
    unittest.main()
