"""Tests for the refactored `VolumeMarker.markVisible`.

严格可信可见性算法（用户目标五阶段）：
  a. 整个空间初始化为 UNKNOWN；
  b. 候选占据体素（geometry 优先，否则多相机可信观测域内 CCM 并集，mask 外
     的深度点不进入候选）；
  c. 候选体素并集的外露立方体面 mesh，对每个相机渲染一次得到 mask/depth（不开
     抗锯齿），命中 face（受相机 mask 观测域门控）-> 真正可见 VALID（固定不变）；
  d. 同一次候选立方体渲染：对每个非候选体素取「中心 + 8 角点」共 9 点，要求 9 点
     全部落在画面内、且每个点相机前方深度严格小于其遮挡终点深度 -> FREE。遮挡终点
     深度逐点取：UV 命中候选渲染 -> 候选 depth；UV 落在 camera.mask 外（mask==0）
     -> +inf（相机在该方向未观测到物体，整条射线空到无穷远）；
  e. 被遮挡、视锥外、相机后方、mask 内但未命中候选、稀疏孔洞 -> UNKNOWN；
     FREE voxel 全量编码为 FREE_KN = -K（K = 到最近非 FREE 锚点（VALID∪UNKNOWN）
     的 L1 距离；全 FREE 时为哨兵 -(3R)）。

纯逻辑（体素化 / 边界 mesh / 9 点证据 / 观测域 / FREE_KN）在 CPU 即可验证；
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


class TestSampleCandidateAndCameraMaskAtUV(unittest.TestCase):
    """候选渲染 + 相机 mask 的 9 点双通道 UV 采样（纯逻辑，CPU）。

    这是 9 点 FREE 证明的核心采样原子，对每个点 UV 返回逐点遮挡终点深度
    effective_depth：
      - UV 命中候选渲染 -> effective = 候选 depth（有限遮挡终点）；
      - UV 落在 camera.mask 外（mask==0）-> effective = +inf（射线空到无穷远）；
      - 画面内但 mask 内又未命中候选 -> effective = -inf（无可信终点，不可释放）。
    """

    def _make_camera(self, W=8, H=8):
        return Camera(
            width=W, height=H, fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device='cpu',
        )

    def _make_render(self, H, W, surface_depth, hit_value=True):
        rendered = torch.full((H, W), float(surface_depth), dtype=torch.float32)
        hit = torch.full((H, W), bool(hit_value), dtype=torch.bool)
        return rendered, hit

    def test_candidate_hit_no_mask_uses_render_depth(self):
        camera = self._make_camera()
        rendered, hit = self._make_render(8, 8, surface_depth=5.0)
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        in_frame, cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        self.assertTrue(bool(in_frame[0]))
        self.assertTrue(bool(cand_hit[0]))
        self.assertFalse(bool(mask_out[0]))
        self.assertAlmostEqual(float(eff[0]), 5.0, places=5)

    def test_background_no_mask_is_neg_inf(self):
        camera = self._make_camera()
        rendered, hit = self._make_render(
            8, 8, surface_depth=0.0, hit_value=False,
        )
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        in_frame, cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        # 无 mask、候选未命中 -> 无可信遮挡终点 -> -inf（不可释放）。
        self.assertTrue(bool(in_frame[0]))
        self.assertFalse(bool(cand_hit[0]))
        self.assertFalse(bool(mask_out[0]))
        self.assertEqual(float(eff[0]), float('-inf'))

    def test_mask_outside_is_pos_inf(self):
        # camera.mask 全 False -> 画面内任意点都在 mask 外 -> effective +inf，
        # 即使该像素命中了候选渲染（mask 外通道优先）。
        camera = self._make_camera()
        camera.loadMask(torch.zeros((8, 8), dtype=torch.bool))
        rendered, hit = self._make_render(8, 8, surface_depth=5.0, hit_value=True)
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        in_frame, cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        self.assertTrue(bool(mask_out[0]))
        self.assertEqual(float(eff[0]), float('inf'))

    def test_mask_inside_hit_uses_render_depth(self):
        # mask 全 True -> 都在 mask 内；命中候选 -> effective = 候选 depth。
        camera = self._make_camera()
        camera.loadMask(torch.ones((8, 8), dtype=torch.bool))
        rendered, hit = self._make_render(8, 8, surface_depth=4.0, hit_value=True)
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        in_frame, cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        self.assertFalse(bool(mask_out[0]))
        self.assertTrue(bool(cand_hit[0]))
        self.assertAlmostEqual(float(eff[0]), 4.0, places=5)

    def test_mask_inside_miss_is_neg_inf(self):
        # mask 全 True 但候选未命中 -> 无可信遮挡终点 -> effective -inf。
        camera = self._make_camera()
        camera.loadMask(torch.ones((8, 8), dtype=torch.bool))
        rendered, hit = self._make_render(
            8, 8, surface_depth=0.0, hit_value=False,
        )
        uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        in_frame, cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        self.assertFalse(bool(mask_out[0]))
        self.assertFalse(bool(cand_hit[0]))
        self.assertEqual(float(eff[0]), float('-inf'))

    def test_out_of_frame_is_neg_inf(self):
        camera = self._make_camera()
        camera.loadMask(torch.zeros((8, 8), dtype=torch.bool))  # 即使 mask 外
        rendered, hit = self._make_render(8, 8, surface_depth=5.0, hit_value=True)
        uv = torch.tensor(
            [[1.5, 0.5], [float('nan'), 0.5]], dtype=torch.float32,
        )
        in_frame, cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        for i in (0, 1):
            self.assertFalse(bool(in_frame[i]))
            self.assertFalse(bool(mask_out[i]))
            self.assertFalse(bool(cand_hit[i]))
            self.assertEqual(float(eff[i]), float('-inf'))

    def test_out_of_frame_is_not_mask_outside(self):
        # 核心不变量：画面外 != camera.mask 外。即使 camera.mask 全 False，
        # 画面外点也不能走「+inf 空射线」释放通道：mask_outside 恒 False，
        # effective 恒 -inf。对照同一相机下画面内的点确实走 +inf 通道。
        camera = self._make_camera()
        camera.loadMask(torch.zeros((8, 8), dtype=torch.bool))
        rendered, hit = self._make_render(8, 8, surface_depth=5.0, hit_value=True)
        uv = torch.tensor(
            [[0.5, 0.5], [1.5, 0.5]], dtype=torch.float32,
        )  # 点 0 画面内（mask 外），点 1 画面外
        in_frame, _cand_hit, mask_out, eff = \
            VolumeMarker._sampleCandidateAndCameraMaskAtUV(
                camera, rendered, hit, uv, 8, 8,
            )
        # 画面内、mask 外 -> +inf 空射线。
        self.assertTrue(bool(in_frame[0]))
        self.assertTrue(bool(mask_out[0]))
        self.assertEqual(float(eff[0]), float('inf'))
        # 画面外 -> 不是 mask 外，-inf（无权释放）。
        self.assertFalse(bool(in_frame[1]))
        self.assertFalse(bool(mask_out[1]))
        self.assertEqual(float(eff[1]), float('-inf'))


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

        # 距离源 = ~free_mask = 非 FREE 锚点（VALID∪UNKNOWN）。此处保留
        # 一个 VALID 锚点 (2,2,2) 与一个 UNKNOWN 锚点 (2,2,3)，其余皆 FREE。
        free_mask = torch.ones((R, R, R), dtype=torch.bool)
        free_mask[2, 2, 2] = False  # VALID 锚点
        free_mask[2, 2, 3] = False  # UNKNOWN 锚点

        out = VolumeMarker.assignFreeLabels(labels, free_mask)

        # VALID / UNKNOWN 锚点不动。
        self.assertEqual(int(out[2, 2, 2].item()), VolumeMarker.VALID)
        self.assertEqual(int(out[2, 2, 3].item()), VolumeMarker.UNKNOWN)
        # 距最近锚点 1 -> -1。
        self.assertEqual(int(out[3, 2, 2].item()), VolumeMarker.freeLabelKN(1))
        self.assertEqual(int(out[2, 1, 2].item()), VolumeMarker.freeLabelKN(1))
        # 距最近锚点 2 -> -2。
        self.assertEqual(int(out[4, 2, 2].item()), VolumeMarker.freeLabelKN(2))
        # UNKNOWN 也是距离源：(2,2,4) 紧贴 UNKNOWN 锚点 (2,2,3) -> -1。
        self.assertEqual(int(out[2, 2, 4].item()), VolumeMarker.freeLabelKN(1))
        # 任意距离全量编码：角点 (0,0,0) 到最近锚点 (2,2,2) 距离 6 -> -6。
        self.assertEqual(int(out[0, 0, 0].item()), VolumeMarker.freeLabelKN(6))
        # 输入不被原地修改。
        self.assertEqual(int(labels[3, 2, 2].item()), VolumeMarker.UNKNOWN)

    def test_assign_free_labels_unknown_is_distance_source(self):
        # FREE 从 VALID∪UNKNOWN 的并集向外扩散：UNKNOWN 同样是距离源，
        # 紧贴 UNKNOWN 的 FREE 会被正确压成 FREE_1N。
        R = 4
        labels = torch.full((R, R, R), VolumeMarker.UNKNOWN, dtype=torch.int64)
        labels[0, 0, 0] = VolumeMarker.VALID

        free_mask = torch.ones((R, R, R), dtype=torch.bool)
        free_mask[0, 0, 0] = False  # VALID 锚点
        free_mask[0, 0, 3] = False  # UNKNOWN 锚点

        out = VolumeMarker.assignFreeLabels(labels, free_mask)

        # 贴近 VALID 锚点的层。
        self.assertEqual(int(out[1, 0, 0].item()), VolumeMarker.freeLabelKN(1))
        self.assertEqual(int(out[1, 1, 0].item()), VolumeMarker.freeLabelKN(2))
        # (0,0,2) 紧贴 UNKNOWN 锚点 (0,0,3) -> UNKNOWN 也作距离源 -> -1。
        self.assertEqual(int(out[0, 0, 2].item()), VolumeMarker.freeLabelKN(1))

    def test_assign_free_labels_all_free_is_inf_sentinel(self):
        # free_mask 覆盖整个空间（无任何非 FREE 锚点）-> FREE 全部哨兵 -(3R)。
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
    def test_empty_camera_list_default_is_all_unknown(self):
        # 严格语义：无相机 -> 无可信证据 -> 全 UNKNOWN（不再哨兵 FREE）。
        R = 5
        labels = VolumeMarker.markVisible([], R)
        self.assertEqual(tuple(labels.shape), (R, R, R))
        self.assertTrue(torch.all(labels == VolumeMarker.UNKNOWN))
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

    def test_points_none_with_no_valid_depth_is_all_unknown(self):
        # depth 全 0 -> valid_depth_mask 全 False -> CCM 候选为空 -> 无候选
        # -> 无可信证据 -> 严格默认全 UNKNOWN。
        R = 4
        camera = self._make_cpu_camera(depth_value=0.0)
        labels = VolumeMarker.markVisible([camera], R)
        self.assertEqual(tuple(labels.shape), (R, R, R))
        self.assertTrue(torch.all(labels == VolumeMarker.UNKNOWN))

    def test_empty_points_is_all_unknown(self):
        R = 4
        camera = self._make_cpu_camera(depth_value=0.0)
        empty_pts = torch.zeros((0, 3), dtype=torch.float32)
        labels = VolumeMarker.markVisible([camera], R, geometry=empty_pts)
        self.assertTrue(torch.all(labels == VolumeMarker.UNKNOWN))


class TestResolutionValidation(unittest.TestCase):
    """严格 resolution 校验（纯逻辑，CPU）。"""

    def test_integer_value_float_ok(self):
        self.assertEqual(VolumeMarker._requirePositiveResolution(8.0, 'x'), 8)

    def test_non_integer_float_raises(self):
        with self.assertRaises(ValueError):
            VolumeMarker._requirePositiveResolution(4.9, 'x')

    def test_bool_raises(self):
        with self.assertRaises(TypeError):
            VolumeMarker._requirePositiveResolution(True, 'x')

    def test_non_positive_raises(self):
        for bad in (0, -1, -5):
            with self.assertRaises(ValueError):
                VolumeMarker._requirePositiveResolution(bad, 'x')


class TestDomainStats(unittest.TestCase):
    """越界 geometry 诊断统计（纯逻辑，CPU）。"""

    def test_domain_stats_counts_inside_and_finite(self):
        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],            # inside
                [0.4, -0.4, 0.49],         # inside
                [2.0, 0.0, 0.0],           # outside
                [float('nan'), 0.0, 0.0],  # non-finite
            ],
            dtype=torch.float32,
        )
        finite_count, inside_count = VolumeMarker._domainStats(pts)
        self.assertEqual(finite_count, 3)
        self.assertEqual(inside_count, 2)

    def test_domain_stats_empty(self):
        finite_count, inside_count = VolumeMarker._domainStats(
            torch.zeros((0, 3), dtype=torch.float32),
        )
        self.assertEqual((finite_count, inside_count), (0, 0))


class TestGeometryDepthDecoupling(unittest.TestCase):
    """显式 geometry 路径不应要求相机 depth（纯逻辑，CPU）。

    候选完全由 geometry 决定时，只需相机位姿/内参；depth 仅在
    geometry=None 的 CCM 候选路径才必需。这里只验证候选构建不再因
    缺 depth 而抛错（渲染分类走 CUDA，端到端由渲染用例守护）。
    """

    def _make_cpu_camera_no_depth(self) -> Camera:
        return Camera(
            width=8, height=8, fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device='cpu',
        )

    def test_build_candidate_with_geometry_no_depth_ok(self):
        R = 8
        camera = self._make_cpu_camera_no_depth()
        self.assertIsNone(camera.depth)
        pts = np.stack([_voxel_center((3, 3, 3), R)]).astype(np.float32)
        # 不应抛错（不依赖 depth）。
        candidate = VolumeMarker._buildCandidateOccupancy(
            [camera], pts, R, torch.float32, 'cpu',
        )
        self.assertTrue(bool(candidate[3, 3, 3]))
        self.assertEqual(int(candidate.sum().item()), 1)

    def test_build_candidate_without_geometry_requires_depth(self):
        R = 8
        camera = self._make_cpu_camera_no_depth()
        with self.assertRaises(ValueError):
            VolumeMarker._buildCandidateOccupancy(
                [camera], None, R, torch.float32, 'cpu',
            )


class TestExactMeshVoxelizer(unittest.TestCase):
    """确定性精确 mesh 表面体素化（纯逻辑，CPU，无需 open3d）。"""

    def test_single_small_triangle_one_voxel(self):
        R = 4
        c = _voxel_center((2, 2, 2), R)
        d = 0.5 / R * 0.4
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

    def test_axis_aligned_quad_covers_all_crossed_voxels(self):
        R = 16
        k = 6
        z = (k + 0.5) / R - 0.5
        i_lo, i_hi = 3, 11
        lo = (i_lo) / R - 0.5 + 1e-3
        hi = (i_hi + 1) / R - 0.5 - 1e-3
        verts = torch.tensor(
            [[lo, lo, z], [hi, lo, z], [hi, hi, z], [lo, hi, z]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        for i in range(i_lo, i_hi + 1):
            for j in range(i_lo, i_hi + 1):
                self.assertTrue(bool(occ[i, j, k]), (i, j, k))
        self.assertEqual(torch.nonzero(occ)[:, 2].unique().tolist(), [k])

    def test_box_surface_is_shell_not_filled(self):
        R = 16
        bmin, bmax = -0.25, 0.25
        corners = np.array(
            [[x, y, z] for x in (bmin, bmax)
             for y in (bmin, bmax) for z in (bmin, bmax)],
            dtype=np.float64,
        )
        quads = [
            (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
            (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
        ]
        tri_list = []
        for a, b, c2, d2 in quads:
            tri_list.append([corners[a], corners[b], corners[c2]])
            tri_list.append([corners[a], corners[c2], corners[d2]])
        verts = torch.tensor(
            np.array(tri_list).reshape(-1, 3), dtype=torch.float32,
        )
        faces = torch.arange(verts.shape[0], dtype=torch.int64).reshape(-1, 3)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        mid = R // 2
        self.assertFalse(bool(occ[mid, mid, mid]))
        self.assertGreater(int(occ.sum().item()), 0)

    def test_thin_sliver_triangle_is_not_missed(self):
        # 极薄长三角形：采样近似可能漏，精确算法必须覆盖至少一个体素。
        R = 8
        verts = torch.tensor(
            [[-0.4, 0.0, 0.0], [0.4, 1e-4, 0.0], [0.4, -1e-4, 0.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        self.assertGreaterEqual(int(occ.sum().item()), 1)

    def test_deterministic(self):
        R = 12
        verts = torch.tensor(
            [[-0.3, -0.3, 0.0], [0.35, -0.2, 0.1], [0.0, 0.4, -0.05]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        occ_a = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        occ_b = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        self.assertTrue(torch.equal(occ_a, occ_b))

    def test_degenerate_zero_area_triangle_falls_back_to_vertices(self):
        # 退化（共线/零面积）三角形：退回顶点点体素化。
        R = 8
        c = _voxel_center((4, 4, 4), R)
        verts = torch.tensor([c, c, c], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        self.assertTrue(bool(occ[4, 4, 4]))
        self.assertEqual(int(occ.sum().item()), 1)

    def test_subdivision_large_triangle_bounded_memory(self):
        # 跨越整个域的大三角形会触发细分，仍应正确覆盖且不报错。
        R = 32
        verts = torch.tensor(
            [[-0.49, -0.49, 0.0], [0.49, -0.49, 0.0], [0.0, 0.49, 0.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        occ = VolumeMarker._voxelizeMesh(verts, faces, R, 'cpu')
        self.assertGreater(int(occ.sum().item()), R)  # 至少覆盖一整条带


class TestL1DistanceRobustness(unittest.TestCase):
    """L1 距离场鲁棒性：非连续输入、CUDA、小分辨率（纯逻辑）。"""

    def _brute(self, source):
        R = source.shape[0]
        src_idx = torch.nonzero(source).to(torch.int64)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(R), torch.arange(R), torch.arange(R),
                indexing='ij',
            ),
            dim=-1,
        ).reshape(-1, 3).to(source.device)
        if src_idx.numel() == 0:
            return torch.full((R, R, R), 3 * R, dtype=torch.int64,
                              device=source.device)
        src_idx = src_idx.to(source.device)
        return (
            (grid[:, None, :] - src_idx[None, :, :]).abs().sum(dim=-1)
            .min(dim=1).values
        ).reshape(R, R, R)

    def test_non_contiguous_input(self):
        torch.manual_seed(1)
        src = (torch.rand((6, 6, 6)) < 0.12).transpose(0, 2)
        dist = VolumeMarker.computeL1DistanceToMask(src)
        self.assertTrue(torch.equal(dist, self._brute(src)))

    def test_small_resolutions(self):
        for R in (1, 2, 3):
            src = torch.zeros((R, R, R), dtype=torch.bool)
            src[0, 0, 0] = True
            dist = VolumeMarker.computeL1DistanceToMask(src)
            self.assertTrue(torch.equal(dist, self._brute(src)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_matches_bruteforce(self):
        torch.manual_seed(2)
        R = 7
        src = (torch.rand((R, R, R)) < 0.1).cuda()
        src[1, 2, 3] = True
        dist = VolumeMarker.computeL1DistanceToMask(src)
        self.assertTrue(torch.equal(dist.cpu(), self._brute(src).cpu()))


class TestVoxelCorners(unittest.TestCase):
    """体素 8 顶点坐标与网格顶点 / 中心约定一致（纯逻辑，CPU）。"""

    def test_corners_shape(self):
        R = 4
        corners = VolumeMarker._createVoxelCorners(R, torch.float32, 'cpu')
        self.assertEqual(tuple(corners.shape), (R, R, R, 8, 3))

    def test_corners_match_grid_vertices(self):
        # voxel (i,j,k) 的最小角 = grid[i,j,k]，最大角 = grid[i+1,j+1,k+1]。
        R = 8
        corners = VolumeMarker._createVoxelCorners(R, torch.float32, 'cpu')
        grid = VolumeMarker.createGridVertices(R, torch.float32, 'cpu')
        i, j, k = 2, 3, 5
        cube = corners[i, j, k]  # (8, 3)
        cmin = cube.min(dim=0).values
        cmax = cube.max(dim=0).values
        self.assertTrue(torch.allclose(cmin, grid[i, j, k]))
        self.assertTrue(torch.allclose(cmax, grid[i + 1, j + 1, k + 1]))

    def test_corners_centroid_matches_voxel_center(self):
        R = 6
        corners = VolumeMarker._createVoxelCorners(R, torch.float32, 'cpu')
        centers = VolumeMarker.createVoxelCenters(R, torch.float32, 'cpu')
        i, j, k = 1, 4, 2
        centroid = corners[i, j, k].mean(dim=0)
        self.assertTrue(torch.allclose(centroid, centers[i, j, k], atol=1e-6))

    def test_corners_within_unit_cube(self):
        R = 4
        corners = VolumeMarker._createVoxelCorners(R, torch.float32, 'cpu')
        self.assertTrue(torch.all(corners >= -0.5 - 1e-6))
        self.assertTrue(torch.all(corners <= 0.5 + 1e-6))


class TestVoxelQueryPoints(unittest.TestCase):
    """体素 9 点查询（中心 + 8 角点）几何一致性（纯逻辑，CPU）。"""

    def test_query_points_shape(self):
        R = 4
        q = VolumeMarker._createVoxelQueryPoints(R, torch.float32, 'cpu')
        self.assertEqual(tuple(q.shape), (R * R * R, 9, 3))

    def test_first_point_is_center_rest_are_corners(self):
        R = 5
        q = VolumeMarker._createVoxelQueryPoints(R, torch.float32, 'cpu')
        centers = VolumeMarker.createVoxelCenters(
            R, torch.float32, 'cpu',
        ).reshape(-1, 3)
        corners = VolumeMarker._createVoxelCorners(
            R, torch.float32, 'cpu',
        ).reshape(-1, 8, 3)
        # 点 0 = 体素中心；点 1..8 = 8 角点。
        self.assertTrue(torch.allclose(q[:, 0, :], centers))
        self.assertTrue(torch.allclose(q[:, 1:, :], corners))

    def test_center_is_centroid_of_corners(self):
        R = 6
        q = VolumeMarker._createVoxelQueryPoints(R, torch.float32, 'cpu')
        centroid = q[:, 1:, :].mean(dim=1)
        self.assertTrue(torch.allclose(q[:, 0, :], centroid, atol=1e-6))


class TestVoxelQueryBoundaryExemptMask(unittest.TestCase):
    """单位立方体外边界 corner 豁免 mask（纯逻辑，CPU）。"""

    def _flat_index(self, R, i, j, k):
        return (i * R + j) * R + k

    def test_shape_and_dtype(self):
        R = 4
        mask = VolumeMarker._createVoxelQueryBoundaryExemptMask(R, 'cpu')
        self.assertEqual(tuple(mask.shape), (R * R * R, 9))
        self.assertEqual(mask.dtype, torch.bool)

    def test_center_never_exempt(self):
        R = 4
        mask = VolumeMarker._createVoxelQueryBoundaryExemptMask(R, 'cpu')
        # 第 0 列（体素中心）对所有 voxel 都不豁免。
        self.assertFalse(bool(mask[:, 0].any()))

    def test_interior_voxel_has_no_exempt_corner(self):
        # 内部 voxel（不接触单位立方体任何外边界面）的 8 角点都不在 ±0.5 上。
        R = 4
        mask = VolumeMarker._createVoxelQueryBoundaryExemptMask(R, 'cpu')
        flat = self._flat_index(R, 1, 1, 1)  # 远离 i/j/k=0 与 R-1 的内部体素
        self.assertFalse(bool(mask[flat].any()))

    def test_boundary_voxel_only_exempts_outer_face_corners(self):
        # 角 voxel (0,0,0)：最小角 corner 落在 (-0.5,-0.5,-0.5)，被豁免；
        # 而其 (1,1,1) 角点位于体素内侧 (-0.5+1/R)，不在外边界 -> 不豁免。
        R = 4
        mask = VolumeMarker._createVoxelQueryBoundaryExemptMask(R, 'cpu')
        corners = VolumeMarker._createVoxelCorners(
            R, torch.float32, 'cpu',
        ).reshape(-1, 8, 3)
        flat = self._flat_index(R, 0, 0, 0)

        # 列 1..8 对应 8 角点；用几何重新判定，确认 mask 与「任一坐标命中 ±0.5」一致。
        expected = (corners[flat].abs() >= (0.5 - 1e-6)).any(dim=-1)
        self.assertTrue(torch.equal(mask[flat, 1:], expected))
        # 至少存在一个被豁免（最小角）与一个不豁免（内侧角）。
        self.assertTrue(bool(mask[flat, 1:].any()))
        self.assertFalse(bool(mask[flat, 1:].all()))

    def test_matches_query_points_boundary_condition(self):
        # mask 的 corner 列与 _createVoxelQueryPoints 的 corner 列（1..8）一一对应：
        # corner 任一坐标命中 ±0.5 <=> 豁免。
        R = 5
        mask = VolumeMarker._createVoxelQueryBoundaryExemptMask(R, 'cpu')
        q = VolumeMarker._createVoxelQueryPoints(R, torch.float32, 'cpu')
        corner_pts = q[:, 1:, :]  # (R^3, 8, 3)
        expected = (corner_pts.abs() >= (0.5 - 1e-6)).any(dim=-1)
        self.assertTrue(torch.equal(mask[:, 1:], expected))


class TestNinePointFreeEvidence(unittest.TestCase):
    """9 点严格 FREE 证明（纯逻辑，CPU）。

    回归核心：唯有 9 点（中心 + 8 角点）同时「画面内 + 严格在遮挡终点前方」才
    给出 FREE 证据；任一点越界 / 不在前方 / 无可信终点都不成立。遮挡终点既可是
    候选渲染深度（有限），也可是 camera.mask 外的 +inf。
    """

    def _all_pass(self):
        # 有限候选遮挡终点 5.0，9 点深度 2.0 均严格在前方。
        point_depth9 = torch.full((1, 9), 2.0)
        in_frame9 = torch.ones((1, 9), dtype=torch.bool)
        effective9 = torch.full((1, 9), 5.0)
        return point_depth9, in_frame9, effective9

    def test_all_nine_in_front_is_free(self):
        pd, infr, eff = self._all_pass()
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertTrue(bool(free[0]))

    def test_one_corner_behind_breaks_proof(self):
        pd, infr, eff = self._all_pass()
        pd[0, 4] = 9.0  # 某角点在遮挡终点后方
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertFalse(bool(free[0]))

    def test_center_point_behind_breaks_proof(self):
        # 中心点（索引 0）在终点后方 -> 整个体素不成立（中心点是额外保证）。
        pd, infr, eff = self._all_pass()
        pd[0, 0] = 9.0
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertFalse(bool(free[0]))

    def test_one_point_out_of_frame_breaks_proof(self):
        pd, infr, eff = self._all_pass()
        infr[0, 5] = False  # 某点投影出画面
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertFalse(bool(free[0]))

    def test_one_point_without_terminus_breaks_proof(self):
        # effective = -inf（画面内但 mask 内又未命中候选）-> 否决该点 -> 非 FREE。
        pd, infr, eff = self._all_pass()
        eff[0, 2] = float('-inf')
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertFalse(bool(free[0]))

    def test_equal_depth_not_strictly_in_front(self):
        # 点深度等于遮挡终点（紧贴表面）-> 不算严格前方 -> 非 FREE。
        pd, infr, eff = self._all_pass()
        eff = pd.clone()
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertFalse(bool(free[0]))

    def test_mixed_channels_all_in_front_is_free(self):
        # 混合通道：部分点终点为有限候选 depth、部分点终点为 mask 外 +inf，
        # 只要 9 点全部严格在各自终点前方即 FREE。
        pd = torch.full((1, 9), 2.0)
        infr = torch.ones((1, 9), dtype=torch.bool)
        eff = torch.full((1, 9), 5.0)
        eff[0, 1] = float('inf')  # mask 外通道
        eff[0, 4] = float('inf')
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertTrue(bool(free[0]))

    def test_mask_outside_channel_frees_far_point(self):
        # mask 外 +inf 终点：即使点深度很大（远体素），仍严格在 +inf 前方 -> FREE。
        pd = torch.full((1, 9), 1e6)
        infr = torch.ones((1, 9), dtype=torch.bool)
        eff = torch.full((1, 9), float('inf'))
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertTrue(bool(free[0]))

    def test_exempt_corner_out_of_frame_does_not_break_proof(self):
        # 外边界 corner 即使画面外 / 无遮挡终点，被豁免后不否决 FREE。
        pd, infr, eff = self._all_pass()
        infr[0, 5] = False          # 该角点投影出画面
        eff[0, 5] = float('-inf')   # 且无可信遮挡终点
        exempt = torch.zeros((1, 9), dtype=torch.bool)
        exempt[0, 5] = True         # 但它是外边界 corner -> 豁免
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff, exempt)
        self.assertTrue(bool(free[0]))

    def test_exempt_does_not_rescue_non_boundary_failure(self):
        # 仅豁免被标记的外边界 corner；另一个非豁免点失败仍否决 FREE。
        pd, infr, eff = self._all_pass()
        infr[0, 5] = False          # 外边界 corner，画面外
        infr[0, 3] = False          # 另一个普通必需 corner 也画面外
        exempt = torch.zeros((1, 9), dtype=torch.bool)
        exempt[0, 5] = True         # 只豁免点 5
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff, exempt)
        self.assertFalse(bool(free[0]))

    def test_center_never_exempt_even_if_marked(self):
        # 中心点（列 0）即使被错误标为豁免也不应救活失败的中心；但更重要的是
        # 调用方约定中心永不豁免：这里直接验证中心失败 + 中心未豁免 -> 否决。
        pd, infr, eff = self._all_pass()
        pd[0, 0] = 9.0              # 中心点落在遮挡终点后方
        exempt = torch.zeros((1, 9), dtype=torch.bool)  # 中心列 False（不豁免）
        free = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff, exempt)
        self.assertFalse(bool(free[0]))

    def test_none_exempt_mask_equivalent_to_no_exemption(self):
        # exempt_mask9=None 等价于全 False：行为与旧三参调用一致。
        pd, infr, eff = self._all_pass()
        infr[0, 5] = False
        free_none = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff, None)
        free_default = VolumeMarker._freeEvidenceNinePointsInFront(pd, infr, eff)
        self.assertFalse(bool(free_none[0]))
        self.assertEqual(bool(free_none[0]), bool(free_default[0]))


class TestStrictFreeMask(unittest.TestCase):
    """严格 FREE = 非候选 且 9 点证明（纯逻辑，CPU）。"""

    def test_only_non_candidate_with_evidence_is_free(self):
        R = 3
        candidate = torch.zeros((R, R, R), dtype=torch.bool)
        candidate[0, 0, 0] = True
        free_evidence = torch.zeros((R, R, R), dtype=torch.bool)
        free_evidence[0, 0, 0] = True  # 候选即使有证据也不释放
        free_evidence[1, 1, 1] = True  # 非候选 + 证据 -> FREE
        free_evidence[2, 2, 2] = False  # 非候选无证据 -> 非 FREE

        mask = VolumeMarker._deriveStrictFreeMask(candidate, free_evidence)
        self.assertFalse(bool(mask[0, 0, 0]))
        self.assertTrue(bool(mask[1, 1, 1]))
        self.assertFalse(bool(mask[2, 2, 2]))

    def test_no_evidence_means_no_free(self):
        R = 3
        candidate = torch.zeros((R, R, R), dtype=torch.bool)
        candidate[1, 1, 1] = True
        free_evidence = torch.zeros((R, R, R), dtype=torch.bool)
        mask = VolumeMarker._deriveStrictFreeMask(candidate, free_evidence)
        self.assertFalse(bool(mask.any()))


class TestMergeLabels(unittest.TestCase):
    """VALID / FREE_KN / UNKNOWN 合并（纯逻辑，CPU）。"""

    def test_valid_fixed_and_free_kn_distance_to_anchor(self):
        # FREE_KN 度量到最近非 FREE 锚点（VALID∪UNKNOWN，即 ~free_mask）的
        # L1 距离。VALID 固定不被改写。
        R = 5
        visible = torch.zeros((R, R, R), dtype=torch.bool)
        visible[0, 0, 0] = True  # 远离 FREE 块的 VALID 锚点
        # 居中 3x3x3 FREE 块（索引 1..3），其余体素都是 UNKNOWN 锚点。
        free_mask = torch.zeros((R, R, R), dtype=torch.bool)
        free_mask[1:4, 1:4, 1:4] = True

        labels = VolumeMarker._mergeLabels(R, visible, free_mask, 'cpu')

        # VALID 固定为 1，绝不被 FREE 改写。
        self.assertEqual(int(labels[0, 0, 0].item()), VolumeMarker.VALID)
        # FREE 块表面体素紧贴 UNKNOWN 锚点 -> -1。
        self.assertEqual(int(labels[1, 2, 2].item()), VolumeMarker.freeLabelKN(1))
        # FREE 块中心被 FREE 包裹，到最近锚点 L1 距离 2 -> -2。
        self.assertEqual(int(labels[2, 2, 2].item()), VolumeMarker.freeLabelKN(2))
        # 非 FREE、非 VALID 的体素保持 UNKNOWN。
        self.assertEqual(int(labels[4, 4, 4].item()), VolumeMarker.UNKNOWN)

    def test_free_kn_layers_increase_toward_interior(self):
        # 厚 FREE 区域内部 K 随到最近非 FREE 边界的距离递增（1N/2N/3N 分层）。
        # 新语义下 UNKNOWN 与 VALID 同为锚点，FREE 从该并集向内逐层加深。
        R = 7
        visible = torch.zeros((R, R, R), dtype=torch.bool)  # 无 VALID
        free_mask = torch.zeros((R, R, R), dtype=torch.bool)
        free_mask[1:6, 1:6, 1:6] = True  # 5x5x5 FREE 块，外围全是 UNKNOWN 锚点

        labels = VolumeMarker._mergeLabels(R, visible, free_mask, 'cpu')

        self.assertEqual(int(labels[1, 3, 3].item()), VolumeMarker.freeLabelKN(1))
        self.assertEqual(int(labels[2, 3, 3].item()), VolumeMarker.freeLabelKN(2))
        self.assertEqual(int(labels[3, 3, 3].item()), VolumeMarker.freeLabelKN(3))


class TestObservationDomain(unittest.TestCase):
    """相机 mask 观测域门控（纯逻辑，CPU）。"""

    def _make_camera(self, width=8, height=6):
        return Camera(
            width=width, height=height, fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device='cpu',
        )

    def test_pixel_observation_mask_no_mask_is_all_true(self):
        camera = self._make_camera()
        obs = VolumeMarker._cameraPixelObservationMask(camera, 6, 8)
        self.assertEqual(tuple(obs.shape), (6, 8))
        self.assertTrue(bool(obs.all()))

    def test_pixel_observation_mask_matches_sample_with_size(self):
        camera = self._make_camera()
        mask = torch.zeros((6, 8), dtype=torch.bool)
        mask[1:4, 2:6] = True
        camera.loadMask(mask)
        obs = VolumeMarker._cameraPixelObservationMask(camera, 6, 8)
        expected = camera.sampleMaskWithSize(8, 6).to(torch.bool)
        self.assertTrue(torch.equal(obs, expected))
        # mask 外像素必须为 False（不产生证据）。
        self.assertFalse(bool(obs[0, 0]))
        self.assertTrue(bool(obs[2, 3]))

    def test_depth_observation_domain_no_mask_is_valid_depth(self):
        camera = self._make_camera()
        camera.loadDepth(torch.ones((6, 8), dtype=torch.float32))
        domain = VolumeMarker._depthObservationDomain(camera)
        self.assertTrue(torch.equal(domain, camera.valid_depth_mask))

    def test_depth_observation_domain_intersects_mask(self):
        camera = self._make_camera()
        camera.loadDepth(torch.ones((6, 8), dtype=torch.float32))
        mask = torch.zeros((6, 8), dtype=torch.bool)
        mask[1:4, 2:6] = True
        camera.loadMask(mask)
        domain = VolumeMarker._depthObservationDomain(camera)
        # 观测域 = valid_depth_mask 且 mask 内；mask 外应被排除。
        self.assertTrue(bool(domain.any()))
        self.assertFalse(bool(domain[0, 0]))
        self.assertLessEqual(
            int(domain.sum().item()), int(camera.valid_depth_mask.sum().item()),
        )


class TestVisibleVoxelsFromRaster(unittest.TestCase):
    """命中 face -> voxel 的观测域门控映射（纯逻辑，CPU）。"""

    def _make_raster(self, H, W):
        rast = torch.zeros((H, W, 4), dtype=torch.float32)
        # 第 4 通道存 1 起的三角 id：像素 (0,0)->tri1, (1,1)->tri2。
        rast[0, 0, 3] = 1.0
        rast[1, 1, 3] = 2.0
        return rast

    def test_hits_inside_domain_map_to_voxels(self):
        R = 2
        H = W = 4
        rast = self._make_raster(H, W)
        # tri 局部 0 -> voxel flat 3 = (0,1,1); tri 局部 1 -> voxel flat 5 = (1,0,1)。
        tri2voxel = torch.tensor([3, 5], dtype=torch.int64)
        obs = torch.ones((H, W), dtype=torch.bool)
        visible = VolumeMarker._visibleVoxelsFromRaster(
            rast, tri2voxel, R, 'cpu', obs,
        )
        self.assertTrue(bool(visible[0, 1, 1]))
        self.assertTrue(bool(visible[1, 0, 1]))
        self.assertEqual(int(visible.sum().item()), 2)

    def test_hits_outside_domain_are_dropped(self):
        R = 2
        H = W = 4
        rast = self._make_raster(H, W)
        tri2voxel = torch.tensor([3, 5], dtype=torch.int64)
        obs = torch.ones((H, W), dtype=torch.bool)
        obs[1, 1] = False  # tri2 命中像素落在 mask 外 -> 不算 VALID
        visible = VolumeMarker._visibleVoxelsFromRaster(
            rast, tri2voxel, R, 'cpu', obs,
        )
        self.assertTrue(bool(visible[0, 1, 1]))
        self.assertFalse(bool(visible[1, 0, 1]))
        self.assertEqual(int(visible.sum().item()), 1)


class TestCameraPrincipalPointDefault(unittest.TestCase):
    """相机默认主点与非正方图像投影一致性（纯逻辑，CPU）。"""

    def test_default_cy_uses_height(self):
        camera = Camera(width=128, height=64, fovx_degree=60.0, device='cpu')
        self.assertAlmostEqual(camera.cx, 64.0, places=5)
        self.assertAlmostEqual(camera.cy, 32.0, places=5)

    def test_center_point_projects_to_image_center(self):
        # 非正方图像：相机正前方中心点应投影到画面中心。
        # 主点在 (width/2, height/2)，UV = (pixel + 0.5) / N，故 u/v 各带
        # 半像素偏移 0.5/width、0.5/height（非正方时两轴偏移不同）。
        width, height = 128, 64
        camera = Camera(
            width=width, height=height, fovx_degree=60.0,
            pos=[0.0, 0.0, 2.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device='cpu',
        )
        center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        uv = camera.project_points_to_uv(center)
        self.assertAlmostEqual(float(uv[0, 0]), 0.5 + 0.5 / width, places=4)
        self.assertAlmostEqual(float(uv[0, 1]), 0.5 + 0.5 / height, places=4)


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

    def test_voxels_closer_than_valid_are_free_by_nine_point_proof(self):
        """核心回归（严格 9 点）：相机与可见 VALID 之间、被 VALID 表面完整
        包住的非候选 voxel 才是 FREE。

        相机在 +Z 看向 -Z，世界 z 越大越靠近相机。用一块远处的候选板（占据
        若干 i,j、固定小 z），渲染后其 silhouette 较宽；位于其前方（更大 z）
        且 9 点（中心+8角点）都投影在该 silhouette 内的非候选 voxel 应为
        FREE_KN（负值）。紧贴 VALID 前表面的一层（共享平面，点深度等于渲染深度）
        按严格判据不算严格前方，可能保持 UNKNOWN，因此只断言有明确间隙的体素。
        """
        R = 16
        device = 'cuda:0'

        # 远处一块 3x3 候选板（z 索引 3），扩大 silhouette 让前方体素 9 点
        # 稳定落在其投影轮廓内部。
        valid_k = 3
        slab = [
            _voxel_center((i, j, valid_k), R)
            for i in (7, 8, 9) for j in (7, 8, 9)
        ]
        pts = torch.from_numpy(np.stack(slab).astype(np.float32)).to(device)

        camera = self._make_front_camera(R)
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)

        self.assertEqual(int(labels[8, 8, valid_k].item()), VolumeMarker.VALID)

        # 中心列上、与 VALID 板有明确间隙（z 索引 >= valid_k + 2）的非候选体素，
        # 其 9 点完整落在板的 silhouette 内且严格在前方 -> FREE_KN(<0)。
        for k in range(valid_k + 2, R):
            label = int(labels[8, 8, k].item())
            self.assertLess(
                label, 0,
                f'voxel (8,8,{k}) 被可见 VALID 板完整包住且在其前方，应为 '
                f'FREE_KN，实际 label={label}',
            )

        # VALID 板后方（更小 z）不应出现 FREE：被遮挡，无可信空证据。
        for k in range(0, valid_k):
            self.assertGreaterEqual(
                int(labels[8, 8, k].item()), 0,
                f'voxel (8,8,{k}) 在 VALID 板后方，不应为 FREE',
            )

    def test_sparse_column_without_blocker_stays_unknown(self):
        """严格回归：没有任何可见 VALID 作遮挡终点的方向全部 UNKNOWN。

        只有一根远处候选柱（中心列），其它列没有任何候选/VALID。那些列上的
        体素没有可信遮挡终点，沿该方向不应产生任何 FREE，应保持 UNKNOWN。
        """
        R = 16
        device = 'cuda:0'
        valid_idx = (8, 8, 3)
        pts = torch.from_numpy(
            np.stack([_voxel_center(valid_idx, R)]).astype(np.float32),
        ).to(device)

        camera = self._make_front_camera(R)
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)

        # 远离候选柱的列（角落）没有 VALID 阻挡 -> 全列 UNKNOWN（无 FREE）。
        corner_col = labels[0, 0, :]
        self.assertTrue(
            bool((corner_col == VolumeMarker.UNKNOWN).all().item()),
            '无可见 VALID 阻挡的方向不应出现 FREE',
        )

    def test_camera_mask_all_false_frees_observed_frustum(self):
        """相机 mask 全 False：候选不被标 VALID；视锥内非候选体素被释放为 FREE。

        新语义下 mask 外表示「相机在该方向未观测到物体 -> 整条射线空到无穷远」。
        整幅 mask 全 False 时，该相机不提供任何 VALID 命中（候选保持非 VALID），
        但其视锥内、9 点都落在 mask 外的非候选体素遮挡终点取 +inf -> FREE。
        候选体素自身被排除，保持 UNKNOWN，并与其余 UNKNOWN 一起作为 FREE_KN
        的距离锚点，因此 FREE 体素取到最近锚点的有限负距离（而非全 FREE 哨兵）。
        """
        R = 16
        device = 'cuda:0'
        idx = (8, 8, 3)  # 候选体素
        pts = torch.from_numpy(
            np.stack([_voxel_center(idx, R)]).astype(np.float32),
        ).to(device)

        camera = self._make_front_camera(R)
        camera.loadMask(torch.zeros((128, 128), dtype=torch.bool, device=device))
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)

        # mask 外不产生命中证据 -> 没有任何 VALID。
        self.assertEqual(int((labels == VolumeMarker.VALID).sum().item()), 0)
        # 候选体素被排除，不会变 FREE，保持 UNKNOWN（也是 FREE_KN 距离锚点）。
        self.assertEqual(int(labels[idx].item()), VolumeMarker.UNKNOWN)
        # 视锥中心、靠近相机的非候选体素 9 点都在 mask 外 -> FREE（label < 0）。
        self.assertLess(int(labels[8, 8, 8].item()), 0)
        # 仍存在 UNKNOWN 锚点 -> FREE_KN 不再退化为全 FREE 哨兵 -(3R)。
        self.assertNotEqual(
            int(labels[8, 8, 8].item()), VolumeMarker.freeLabelInf(R),
        )
        self.assertGreater(
            int((labels < 0).sum().item()), 0,
            'mask 全 False 时视锥内非候选体素应被释放为 FREE',
        )

    def test_no_candidate_means_all_unknown(self):
        """无候选体素（points 为空）时，严格默认整个空间为 UNKNOWN。"""
        R = 8
        device = 'cuda:0'
        camera = self._make_front_camera(R)
        empty_pts = torch.zeros((0, 3), dtype=torch.float32, device=device)
        labels = VolumeMarker.markVisible([camera], R, geometry=empty_pts)
        self.assertTrue(torch.all(labels == VolumeMarker.UNKNOWN))

    def test_far_camera_beyond_default_clip_still_hits(self):
        """相机距离远超默认 far=100 时，动态 near/far 仍能命中候选 -> VALID。

        体素域固定在 [-0.5, 0.5]^3，但相机可被放到 ~300 单位外（超出默认
        far=100）。若不动态设裁剪面，候选 mesh 会被裁掉、无 VALID 命中。
        """
        R = 16
        device = 'cuda:0'
        idx = (8, 8, 8)
        pts = torch.from_numpy(
            np.stack([_voxel_center(idx, R)]).astype(np.float32),
        ).to(device)

        camera = Camera(
            width=128, height=128, fovx_degree=60.0,
            pos=[0.0, 0.0, 300.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device=device,
        )
        camera.loadDepth(
            torch.zeros((128, 128), dtype=torch.float32, device=device),
        )
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)
        self.assertEqual(int(labels[idx].item()), VolumeMarker.VALID)

    def test_geometry_path_without_depth_runs(self):
        """显式 geometry 时即使相机未加载 depth 也能完成渲染分类。"""
        R = 16
        device = 'cuda:0'
        idx = (8, 8, 8)
        pts = torch.from_numpy(
            np.stack([_voxel_center(idx, R)]).astype(np.float32),
        ).to(device)
        camera = Camera(
            width=128, height=128, fovx_degree=60.0,
            pos=[0.0, 0.0, 3.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            device=device,
        )
        self.assertIsNone(camera.depth)
        labels = VolumeMarker.markVisible([camera], R, geometry=pts)
        self.assertEqual(int(labels[idx].item()), VolumeMarker.VALID)

    def _upper_hemisphere_cameras(self, look_at, device='cuda:0', n_az=4):
        """正上方俯视 + 上半球若干斜视角（全部在 +Z 半球，看不到底部）。"""
        import math

        cams = [Camera(
            width=256, height=256, fovx_degree=50.0,
            pos=[0.0, 0.0, 3.0], look_at=look_at, up=[0.0, 1.0, 0.0],
            device=device,
        )]
        elev = math.radians(55.0)
        rh = 3.0 * math.cos(elev)
        z = 3.0 * math.sin(elev)
        for i in range(n_az):
            az = 2.0 * math.pi * i / n_az
            cams.append(Camera(
                width=256, height=256, fovx_degree=50.0,
                pos=[rh * math.cos(az), rh * math.sin(az), z],
                look_at=look_at, up=[0.0, 1.0, 0.0], device=device,
            ))
        return cams

    @staticmethod
    def _disk_base_shell_points(r=0.35, zlo=-0.45, zhi=-0.15,
                                n_ring=72, n_rad=18, n_side=20):
        """采样圆盘底座的「表面壳」点云（顶/底圆面 + 侧壁），内部不含点。

        模拟真实扫描/深度点云：只有表面被采到，底座内部是非候选体素，正是
        从上半球观察时易被错误标 FREE 的区域。
        """
        import math

        pts = []
        for z in (zlo, zhi):
            for rr in np.linspace(0.0, r, n_rad):
                m = max(4, int(n_ring * (rr / r)) + 4)
                for j in range(m):
                    th = 2.0 * math.pi * j / m
                    pts.append([rr * math.cos(th), rr * math.sin(th), z])
        for z in np.linspace(zlo, zhi, n_side):
            for j in range(n_ring):
                th = 2.0 * math.pi * j / n_ring
                pts.append([r * math.cos(th), r * math.sin(th), float(z)])
        return np.asarray(pts, dtype=np.float32)

    def test_upper_hemisphere_disk_base_interior_stays_unknown(self):
        """上半球俯视带圆盘底座点云：底座内部被顶面遮挡，应 UNKNOWN 而非 FREE。

        复现 bug：从坐标原点上半球各角度观察点云时，底座内部（非候选、被顶面
        遮挡）与底部未观测空间此前会因 background / 未观测被错误标 FREE。严格
        9 点重构后，FREE 只来自「整个 voxel 被夹在相机与遮挡终点之间」，被顶面
        遮挡的内部体素在表面后方（不在前方）-> 不满足证明 -> UNKNOWN。
        这些相机未加载 mask，FREE 只能走候选渲染命中通道（无 mask 外通道）。
        """
        R = 16
        device = 'cuda:0'
        pts = torch.from_numpy(self._disk_base_shell_points()).to(device)
        cams = self._upper_hemisphere_cameras(
            look_at=[0.0, 0.0, -0.3], device=device,
        )

        labels = VolumeMarker.markVisible(cams, R, geometry=pts)

        # 顶面被看到 -> 至少存在 VALID（保证几何真的渲染了，测试不退化）。
        self.assertTrue(bool((labels == VolumeMarker.VALID).any().item()))

        # 底座内部中轴体素（被顶面遮挡的非候选）必须 UNKNOWN，不得为 FREE(<0)。
        for k in (2, 3):
            label = int(labels[8, 8, k].item())
            self.assertEqual(
                label, VolumeMarker.UNKNOWN,
                f'底座内部 voxel (8,8,{k}) 被顶面遮挡，应 UNKNOWN，'
                f'实际 label={label}',
            )

        # 雕刻仍有效：底座顶面上方、相机与顶面之间的空间应为 FREE(<0)。
        self.assertLess(int(labels[8, 8, 10].item()), 0)


def test():
    unittest.main()


if __name__ == '__main__':
    unittest.main()
