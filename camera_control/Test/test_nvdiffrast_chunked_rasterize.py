"""Regression tests for `NVDiffRastRenderer._rasterize_faces_chunked`.

第一性原理：分块光栅必须与完整 `dr.rasterize` 在可见性 / 三角 ID 上等价。
重叠两个三角时，靠近相机的三角必须始终覆盖远处三角，与 chunk 切分顺序、
chunk 大小无关。本测试通过强制 `chunk_size=1` 把每个三角放在独立 chunk 中，
并刻意把远处三角排在前、近处三角排在后，让旧的 camera-space Z 合并逻辑
（z 越小越近）必然误判。
"""

import os
import unittest

import numpy as np
import torch
import trimesh

try:
    import nvdiffrast.torch  # noqa: F401
    _NVDR_AVAILABLE = True
except Exception:  # pragma: no cover - 环境无 nvdiffrast 时直接跳过
    _NVDR_AVAILABLE = False


@unittest.skipUnless(
    _NVDR_AVAILABLE and torch.cuda.is_available(),
    'chunked rasterize 仅在 CUDA + nvdiffrast 可用时有效'
)
class TestChunkedRasterizeDepth(unittest.TestCase):
    """覆盖 chunk merge 的深度选择是否与完整 rasterize 一致。"""

    def setUp(self) -> None:
        # 0 字节阈值 = 任意 mesh 都走 chunked 分支，用来强制覆盖分块路径
        os.environ['NVDR_CHUNK_MEM_THRESHOLD_BYTES'] = '0'
        os.environ['NVDR_CHUNK_SIZE'] = '1'

        import importlib

        from camera_control.Module import nvdiffrast_renderer as nvr_module

        importlib.reload(nvr_module)
        self._nvr_module = nvr_module
        self.NVDiffRastRenderer = nvr_module.NVDiffRastRenderer

        from camera_control.Module.camera import Camera

        self.camera = Camera(
            width=64,
            height=64,
            fovx_degree=60.0,
            pos=[0.0, 0.0, 2.0],
            look_at=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            device='cuda:0',
        )

    def tearDown(self) -> None:
        for key in ('NVDR_CHUNK_MEM_THRESHOLD_BYTES', 'NVDR_CHUNK_SIZE'):
            os.environ.pop(key, None)

    def _make_two_triangle_mesh(self) -> trimesh.Trimesh:
        """两个屏幕投影完全重合、深度不同的三角形。

        相机位于 ``(0, 0, 2)`` 看向 ``-Z``：
          - far_tri 位于 ``z = -0.5`` 平面（相对相机更远）；
          - near_tri 位于 ``z = +0.5`` 平面（相对相机更近）。

        故意把 far_tri 排在 face index 0，near_tri 排在 face index 1，
        让 chunk 顺序为 [far, near]；完整 dr.rasterize 仍然会让 near 覆盖
        far，正确的 chunk merge 也必须如此。
        """
        far_z = -0.5
        near_z = 0.5

        vertices = np.array(
            [
                [-0.6, -0.6, far_z],
                [0.6, -0.6, far_z],
                [0.0, 0.6, far_z],
                [-0.6, -0.6, near_z],
                [0.6, -0.6, near_z],
                [0.0, 0.6, near_z],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=np.int64,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def test_near_triangle_wins_in_chunked_merge(self) -> None:
        mesh = self._make_two_triangle_mesh()

        rast_dict = self.NVDiffRastRenderer.rasterize(mesh, self.camera)
        rast_out = rast_dict['rast_out']

        triangle_id = rast_out[0, :, :, 3].long()
        hit_mask = triangle_id > 0
        self.assertTrue(
            bool(hit_mask.any().item()),
            'chunked rasterize 没有命中任何像素，分块逻辑被破坏',
        )

        cy, cx = self.camera.height // 2, self.camera.width // 2
        center_tri = int(triangle_id[cy, cx].item())
        self.assertEqual(
            center_tri,
            2,
            f'chunk merge 应让近处三角(face_id=1, rast 通道=2)覆盖远处三角，'
            f'实际中心像素 triangle_id={center_tri}',
        )

        hit_ids = torch.unique(triangle_id[hit_mask]).tolist()
        self.assertNotIn(
            1,
            hit_ids,
            f'近处三角完全遮挡远处三角，rast 中不应再出现远处 triangle_id=1，'
            f'实际命中集合={hit_ids}',
        )

    def test_chunked_matches_full_rasterize(self) -> None:
        """chunk 路径与单次完整 rasterize 在三角 ID 图上应该完全一致。"""
        mesh = self._make_two_triangle_mesh()

        rast_dict_chunked = self.NVDiffRastRenderer.rasterize(mesh, self.camera)

        # 超大阈值 = 永远不走 chunked，强制完整 rasterize 作为基线对比
        os.environ['NVDR_CHUNK_MEM_THRESHOLD_BYTES'] = str(1 << 62)
        import importlib

        importlib.reload(self._nvr_module)
        full_renderer = self._nvr_module.NVDiffRastRenderer
        rast_dict_full = full_renderer.rasterize(mesh, self.camera)

        os.environ['NVDR_CHUNK_MEM_THRESHOLD_BYTES'] = '0'
        importlib.reload(self._nvr_module)
        self.NVDiffRastRenderer = self._nvr_module.NVDiffRastRenderer

        tri_chunked = rast_dict_chunked['rast_out'][0, :, :, 3].long().cpu()
        tri_full = rast_dict_full['rast_out'][0, :, :, 3].long().cpu()

        self.assertTrue(
            torch.equal(tri_chunked, tri_full),
            'chunked rasterize 与完整 rasterize 在 triangle_id 图上出现不一致，'
            '说明 chunk 合并仍然丢面或选错了深度',
        )


def test():
    unittest.main()
