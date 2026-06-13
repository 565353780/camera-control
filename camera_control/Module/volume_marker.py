import os
import torch
import trimesh
import numpy as np
import open3d as o3d

from typing import List, Optional, Tuple, Union

from camera_control.Config.visible import (
    VISIBLE_LABEL_UNKNOWN,
    VISIBLE_LABEL_VALID,
    visibleLabelFreeKN,
    visibleLabelFreeInf,
)
from camera_control.Module.camera import Camera


class VolumeMarker(object):
    UNKNOWN: int = VISIBLE_LABEL_UNKNOWN
    VALID: int = VISIBLE_LABEL_VALID

    def __init__(self) -> None:
        return

    @staticmethod
    def freeLabelKN(k: int) -> int:
        """FREE_KN 标签编码（-K，K >= 1），转发 Config.visible.visibleLabelFreeKN。"""
        return visibleLabelFreeKN(k)

    @staticmethod
    def freeLabelInf(resolution: int) -> int:
        """「无限远 FREE」哨兵编码（-(3R)），转发 Config.visible.visibleLabelFreeInf。"""
        return visibleLabelFreeInf(resolution)

    # ------------------------------------------------------------------
    # 体素 / 网格基础原子函数
    # ------------------------------------------------------------------
    @staticmethod
    def _requirePositiveResolution(volume_resolution: int, caller: str) -> int:
        """校验体素分辨率为正整数（或整数值浮点），返回 int 化后的 R。

        第一性原理：分辨率是离散网格的边长格数，必须是正整数。允许 4.0
        这类整数值浮点（常见于配置反序列化），但拒绝 4.9 这类会被静默
        截断成 4 的非整数浮点，避免「以为是 5 实际跑成 4」的隐性错误。
        """
        if isinstance(volume_resolution, bool):
            raise TypeError(
                f'[ERROR][VolumeMarker::{caller}] '
                f'volume_resolution must be an integer, got bool {volume_resolution}'
            )

        if isinstance(volume_resolution, float):
            if not volume_resolution.is_integer():
                raise ValueError(
                    f'[ERROR][VolumeMarker::{caller}] '
                    f'volume_resolution must be an integer value, '
                    f'got non-integer float {volume_resolution}'
                )
            volume_resolution = int(volume_resolution)

        if not isinstance(volume_resolution, int):
            raise TypeError(
                f'[ERROR][VolumeMarker::{caller}] '
                f'volume_resolution must be an int, got {type(volume_resolution)}'
            )

        if volume_resolution <= 0:
            raise ValueError(
                f'[ERROR][VolumeMarker::{caller}] '
                f'volume_resolution must be positive, got {volume_resolution}'
            )
        return int(volume_resolution)

    # 触发越界 geometry 告警的越界点占比阈值：超过该比例的候选点落在
    # [-0.5, 0.5]^3 之外时，几乎可以确定 geometry 未归一化到体素域。
    OUT_OF_DOMAIN_WARN_RATIO: float = 0.5

    @staticmethod
    def _domainStats(points: torch.Tensor) -> Tuple[int, int]:
        """返回 (有限点数, 落在 [-0.5, 0.5]^3 域内的点数)。

        用于诊断 geometry 是否归一化到体素域：大量越界点意味着上游坐标
        系/尺度与体素网格不一致，候选体素会大面积丢失。
        """
        if points.numel() == 0:
            return 0, 0
        finite = torch.isfinite(points).all(dim=-1)
        finite_count = int(finite.sum().item())
        if finite_count == 0:
            return 0, 0
        inside = VolumeMarker._finiteInBounds(points)
        return finite_count, int(inside.sum().item())

    @staticmethod
    def _warnIfOutOfDomain(points: torch.Tensor, caller: str) -> None:
        """候选点大面积越出体素域 [-0.5, 0.5]^3 时打印告警（不抛错）。

        体素域是硬约束，越界点会在体素化阶段被静默过滤。若绝大多数点
        越界，结果会退化成「无候选 -> 全 FREE」，这通常是 geometry 未
        归一化的征兆，及早告警以免静默产出错误标签。
        """
        finite_count, inside_count = VolumeMarker._domainStats(points)
        if finite_count == 0:
            return
        out_ratio = 1.0 - inside_count / finite_count
        if out_ratio > VolumeMarker.OUT_OF_DOMAIN_WARN_RATIO:
            print(
                f'[WARN][VolumeMarker::{caller}] '
                f'{out_ratio * 100.0:.1f}% of finite candidate points fall '
                f'outside the voxel domain [-0.5, 0.5]^3 '
                f'({inside_count}/{finite_count} inside); geometry may not be '
                f'normalized to the unit cube.'
            )

    @staticmethod
    def _emptyLabels(
        R: int,
        value: int,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """创建 (R, R, R) 的 int64 标签张量，统一全 FREE/UNKNOWN 初始化。"""
        if device is None:
            return torch.full((R, R, R), value, dtype=torch.int64)
        return torch.full((R, R, R), value, dtype=torch.int64, device=device)

    @staticmethod
    def _finiteInBounds(
        points: torch.Tensor,
        lower: float = -0.5,
        upper: float = 0.5,
    ) -> torch.Tensor:
        """返回 (N,) bool：点的前三维有限且落在 [lower, upper] 立方体内。"""
        finite = torch.isfinite(points).all(dim=-1)
        return (
            finite
            & (points[:, 0] >= lower) & (points[:, 0] <= upper)
            & (points[:, 1] >= lower) & (points[:, 1] <= upper)
            & (points[:, 2] >= lower) & (points[:, 2] <= upper)
        )

    @staticmethod
    def _pointsToVoxelIndices(points: torch.Tensor, R: int) -> torch.Tensor:
        """把点体素化为 (N, 3) 的 long 体素索引（已 clamp 到 [0, R-1]）。

        坐标约定：体素覆盖 [-0.5, 0.5]^3，
        voxel index = floor((point + 0.5) * R)，(x, y, z) 对应 (i, j, k)。
        """
        return torch.floor((points + 0.5) * R).long().clamp(0, R - 1)

    @staticmethod
    def _voxelIndicesToFlat(indices: torch.Tensor, R: int) -> torch.Tensor:
        """把 (..., 3) 的 (i, j, k) 体素索引压成 flat = (i * R + j) * R + k。"""
        return (indices[..., 0] * R + indices[..., 1]) * R + indices[..., 2]

    @staticmethod
    def _makeIjkGrid(coords: torch.Tensor) -> torch.Tensor:
        """对一维坐标做 (x, y, z) 的 ij meshgrid 并 stack 成 (L, L, L, 3)。"""
        xs, ys, zs = torch.meshgrid(coords, coords, coords, indexing='ij')
        return torch.stack([xs, ys, zs], dim=-1)

    # ------------------------------------------------------------------
    # 候选占据体素
    # ------------------------------------------------------------------
    @staticmethod
    def _loadPointsFile(points_file_path: str) -> Optional[np.ndarray]:
        """从文件加载候选点云，返回 (N, 3) 的 numpy 数组，失败返回 None。

        支持 .npy / .npz（取第一个含 3 列的数组），以及 trimesh 能解析的
        .ply / .xyz / .obj 等几何文件（取顶点）。
        """
        if not os.path.exists(points_file_path):
            raise FileNotFoundError(
                '[ERROR][VolumeMarker::_loadPointsFile] '
                f'points file not exist: {points_file_path}'
            )

        ext = os.path.splitext(points_file_path)[1].lower()

        if ext == '.npy':
            arr = np.load(points_file_path)
            return np.asarray(arr)

        if ext == '.npz':
            data = np.load(points_file_path)
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim >= 2 and arr.shape[-1] >= 3:
                    return arr
            raise ValueError(
                '[ERROR][VolumeMarker::_loadPointsFile] '
                f'no (..., >=3) array found in npz: {points_file_path}'
            )

        if ext in ('.txt', '.xyz', '.csv'):
            arr = np.loadtxt(points_file_path)
            return np.asarray(arr)

        # 其余交给 trimesh：点云或网格都取顶点。
        loaded = trimesh.load(points_file_path, process=False)
        if isinstance(loaded, trimesh.points.PointCloud):
            return np.asarray(loaded.vertices)
        if isinstance(loaded, trimesh.Trimesh):
            return np.asarray(loaded.vertices)
        if isinstance(loaded, trimesh.Scene):
            verts = []
            for geom in loaded.geometry.values():
                if hasattr(geom, 'vertices'):
                    verts.append(np.asarray(geom.vertices))
            if len(verts) == 0:
                raise ValueError(
                    '[ERROR][VolumeMarker::_loadPointsFile] '
                    f'no vertices found in scene: {points_file_path}'
                )
            return np.concatenate(verts, axis=0)

        raise ValueError(
            '[ERROR][VolumeMarker::_loadPointsFile] '
            f'unsupported points file: {points_file_path}'
        )

    @staticmethod
    def _normalizePoints(
        points: Union[np.ndarray, torch.Tensor, str],
        dtype,
        device: str,
    ) -> torch.Tensor:
        """把多种 points 输入统一成 (N, 3) 的 torch.Tensor（位于目标 device）。"""
        if isinstance(points, str):
            points = VolumeMarker._loadPointsFile(points)

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(np.ascontiguousarray(points))

        if not isinstance(points, torch.Tensor):
            raise TypeError(
                '[ERROR][VolumeMarker::_normalizePoints] '
                f'unsupported points type: {type(points)}'
            )

        points = points.to(dtype=dtype, device=device)

        if points.ndim == 1:
            points = points.unsqueeze(0)

        points = points.reshape(-1, points.shape[-1])
        if points.shape[-1] < 3:
            raise ValueError(
                '[ERROR][VolumeMarker::_normalizePoints] '
                f'points last dim must be >= 3, got {points.shape}'
            )
        return points[:, :3].contiguous()

    # ------------------------------------------------------------------
    # geometry 输入归一化：自动判别 点云 / 三角网格 / numpy / tensor / 文件
    # ------------------------------------------------------------------
    @staticmethod
    def _isTriangleMesh(geometry) -> bool:
        """判断输入是否为「带三角面」的网格（open3d / trimesh）。

        只有真正含有三角面拓扑的对象才走 mesh 体素化路径；裸点云
        （PointCloud / numpy / tensor）一律走点体素化路径。
        """
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            return len(geometry.triangles) > 0
        if isinstance(geometry, trimesh.Trimesh):
            return geometry.faces is not None and len(geometry.faces) > 0
        return False

    @staticmethod
    def _meshVerticesFaces(
        geometry,
        dtype,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 open3d / trimesh 三角网格取出 (V,3) 顶点与 (F,3) 面索引张量。"""
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            vertices_np = np.asarray(geometry.vertices)
            faces_np = np.asarray(geometry.triangles)
        elif isinstance(geometry, trimesh.Trimesh):
            vertices_np = np.asarray(geometry.vertices)
            faces_np = np.asarray(geometry.faces)
        else:
            raise TypeError(
                '[ERROR][VolumeMarker::_meshVerticesFaces] '
                f'unsupported mesh type: {type(geometry)}'
            )

        vertices = torch.as_tensor(
            np.ascontiguousarray(vertices_np), dtype=dtype, device=device,
        )
        faces = torch.as_tensor(
            np.ascontiguousarray(faces_np), dtype=torch.int64, device=device,
        )
        return vertices, faces

    # 体采样相对体素的过采样倍率（仅用于可选的采样 fallback 路径）。
    MESH_SAMPLE_OVERSAMPLE_K: int = 3
    # 采样点数上限，避免巨大网格 + 高分辨率时点数爆炸（仅 fallback 路径）。
    MESH_SAMPLE_MAX_POINTS: int = 8_000_000

    # 精确体素化时，单个三角形允许覆盖的最大 voxel-AABB 体积；超过则把三角形
    # 二分细分，使每块的候选 voxel 数有界，从而 (三角形 x 候选 voxel) 配对
    # 的内存峰值可控。值偏大牺牲内存换更少细分，偏小相反。
    MESH_VOXELIZE_MAX_VOXELS_PER_TRI: int = 4096
    # 三角形细分的最大递归深度，防御退化输入导致的无界细分。
    MESH_VOXELIZE_MAX_SUBDIV_DEPTH: int = 24

    @staticmethod
    def _meshSurfaceArea(vertices: torch.Tensor, faces: torch.Tensor) -> float:
        """三角网格总表面积（用于按目标点间距估算采样点数）。"""
        tris = vertices[faces]  # (F, 3, 3)
        e0 = tris[:, 1] - tris[:, 0]
        e1 = tris[:, 2] - tris[:, 0]
        cross = torch.cross(e0, e1, dim=-1)
        return float(0.5 * cross.norm(dim=-1).sum().item())

    @staticmethod
    def _meshSampleCount(area: float, R: int, K: int) -> int:
        """按目标点间距 d = 1/(K*R) 估算面积均匀采样所需点数（fallback 路径）。"""
        d = 1.0 / (K * R)
        n = int(np.ceil(area / (d * d) * 2.0))
        n = max(n, (K * R) ** 2)
        return int(min(n, VolumeMarker.MESH_SAMPLE_MAX_POINTS))

    @staticmethod
    def _triBoxOverlapBatch(
        tris: torch.Tensor,
        box_centers: torch.Tensor,
        half: float,
    ) -> torch.Tensor:
        """批量三角形 vs 轴对齐立方体（voxel）相交测试（SAT，Akenine-Möller）。

        第一性原理：凸体相交可由分离轴定理判定——两凸体不相交当且仅当存在
        一条轴上它们的投影不重叠。三角形 vs AABB 需检验 13 条轴：
          1) 3 条 box 面法线（坐标轴 x/y/z）；
          2) 1 条三角形面法线；
          3) 9 条 box 边 (x/y/z) 与三角形三边的叉积。

        Args:
            tris: (M, 3, 3) 已平移到「以对应 box 中心为原点」的三角形顶点，
                即 tris[m] = triangle_m_vertices - box_centers[m]。
            box_centers: (M, 3) 仅用于校验形状一致（计算已在中心坐标系完成）。
            half: voxel 半边长（= 0.5 / R）。

        Returns:
            (M,) bool，三角形是否与对应 voxel 相交（含边界接触）。
        """
        M = tris.shape[0]
        device = tris.device
        if M == 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)

        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]

        # 数值容差：相对 half 的小量，吸收浮点误差，保证「接触即命中」的保守性。
        eps = half * 1e-5

        # --- 轴 1：3 条坐标轴（box 面法线）---
        # 三角形在各坐标轴上的投影区间需与 [-half, half] 重叠。
        tri_min = torch.minimum(torch.minimum(v0, v1), v2)  # (M, 3)
        tri_max = torch.maximum(torch.maximum(v0, v1), v2)  # (M, 3)
        axis_ok = ((tri_min <= half + eps) & (tri_max >= -half - eps)).all(dim=1)

        # --- 轴 2：三角形面法线（平面 vs box）---
        e0 = v1 - v0
        e1 = v2 - v1
        normal = torch.cross(e0, e1, dim=1)  # (M, 3)
        # box 在法线方向的半投影半径 r = half * (|nx| + |ny| + |nz|)。
        r = half * normal.abs().sum(dim=1) + eps  # (M,)
        # 平面到 box 中心（原点）的有符号距离 = -dot(normal, v0)。
        d = -(normal * v0).sum(dim=1)  # (M,)
        plane_ok = d.abs() <= r

        # --- 轴 3：9 条 边叉积轴 ---
        edges = torch.stack([e0, e1, v0 - v2], dim=1)  # (M, 3, 3) 三条边
        verts = torch.stack([v0, v1, v2], dim=1)        # (M, 3, 3) 三个顶点

        cross_ok = torch.ones((M,), dtype=torch.bool, device=device)
        # 坐标基轴，逐个 (边 e_i x 基轴 a_j) 检验。
        for a in range(3):
            base = torch.zeros(3, dtype=tris.dtype, device=device)
            base[a] = 1.0
            for ei in range(3):
                edge = edges[:, ei, :]  # (M, 3)
                axis = torch.cross(
                    edge, base.expand_as(edge), dim=1,
                )  # (M, 3)
                # 三顶点在该轴上的投影。
                p = torch.matmul(verts, axis.unsqueeze(-1)).squeeze(-1)  # (M, 3)
                p_min = p.min(dim=1).values
                p_max = p.max(dim=1).values
                # box 在该轴上的投影半径。
                rad = half * axis.abs().sum(dim=1) + eps
                this_ok = (p_min <= rad) & (p_max >= -rad)
                cross_ok = cross_ok & this_ok

        return axis_ok & plane_ok & cross_ok

    @staticmethod
    def _voxelizeTrianglesExact(
        tris: torch.Tensor,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """对 (M, 3, 3) 三角形做确定性精确表面体素化，返回 (R,R,R) bool。

        第一性原理：一个 voxel 被网格穿过 <=> 至少一个三角形与该 voxel 的
        AABB 相交（SAT 判定）。逐三角形枚举其 voxel-AABB 内的候选 voxel，
        做精确 tri-box 相交，结果是确定性的（与采样无关），且不漏任何被
        网格穿过的 voxel。

        为控制 (三角形 x 候选 voxel) 配对的内存峰值，voxel-AABB 过大的
        三角形会沿最长边二分细分，直到每块候选 voxel 数有界。

        坐标约定：体素覆盖 [-0.5, 0.5]^3，voxel index = floor((p + 0.5) * R)。
        """
        occupancy = torch.zeros((R, R, R), dtype=torch.bool, device=device)
        if tris.shape[0] == 0:
            return occupancy

        occ_flat = occupancy.reshape(-1)
        voxel_size = 1.0 / R
        half = 0.5 * voxel_size
        # voxel (i,j,k) 中心世界坐标 = -0.5 + (idx + 0.5) * voxel_size。
        max_voxels = VolumeMarker.MESH_VOXELIZE_MAX_VOXELS_PER_TRI

        # 用栈做迭代式细分，避免递归深度限制与 Python 递归开销。
        stack: List[Tuple[torch.Tensor, int]] = [(tris, 0)]
        while stack:
            batch, depth = stack.pop()
            if batch.shape[0] == 0:
                continue

            # 每个三角形覆盖的 voxel index AABB（clamp 进网格）。
            tri_min = batch.min(dim=1).values  # (B, 3)
            tri_max = batch.max(dim=1).values  # (B, 3)
            idx_lo = torch.floor((tri_min + 0.5) * R).long().clamp(0, R - 1)
            idx_hi = torch.floor((tri_max + 0.5) * R).long().clamp(0, R - 1)
            span = (idx_hi - idx_lo + 1)  # (B, 3)
            voxel_count = span[:, 0] * span[:, 1] * span[:, 2]  # (B,)

            too_big = voxel_count > max_voxels
            if depth < VolumeMarker.MESH_VOXELIZE_MAX_SUBDIV_DEPTH and bool(too_big.any()):
                big = batch[too_big]
                small = batch[~too_big]
                stack.extend(VolumeMarker._subdivideTriangles(big, depth))
                if small.shape[0] == 0:
                    continue
                batch = small
                idx_lo = idx_lo[~too_big]
                idx_hi = idx_hi[~too_big]
                span = span[~too_big]
                voxel_count = voxel_count[~too_big]

            VolumeMarker._markBatchExact(
                batch, idx_lo, span, voxel_count, R, half, occ_flat,
            )

        return occ_flat.reshape(R, R, R)

    @staticmethod
    def _subdivideTriangles(
        tris: torch.Tensor,
        depth: int,
    ) -> List[Tuple[torch.Tensor, int]]:
        """沿最长边中点把每个三角形二分为两个子三角形（保持覆盖等价）。"""
        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]
        len01 = (v1 - v0).norm(dim=1)
        len12 = (v2 - v1).norm(dim=1)
        len20 = (v0 - v2).norm(dim=1)
        edge_lens = torch.stack([len01, len12, len20], dim=1)  # (B, 3)
        longest = edge_lens.argmax(dim=1)  # (B,)

        # 对三种「最长边」分别构造二分；用 mask 合并，保持全向量化。
        children: List[torch.Tensor] = []
        # 边 0 = (v0, v1)，对边顶点 v2；中点 m=(v0+v1)/2 -> (v0,m,v2),(m,v1,v2)
        # 边 1 = (v1, v2)，对边顶点 v0
        # 边 2 = (v2, v0)，对边顶点 v1
        configs = [
            (0, 1, 2),  # split edge v0-v1, apex v2
            (1, 2, 0),  # split edge v1-v2, apex v0
            (2, 0, 1),  # split edge v2-v0, apex v1
        ]
        verts = [v0, v1, v2]
        for cfg_i, (a, b, apex) in enumerate(configs):
            sel = longest == cfg_i
            if not bool(sel.any()):
                continue
            va, vb, vap = verts[a][sel], verts[b][sel], verts[apex][sel]
            mid = 0.5 * (va + vb)
            child0 = torch.stack([va, mid, vap], dim=1)
            child1 = torch.stack([mid, vb, vap], dim=1)
            children.append(child0)
            children.append(child1)

        if len(children) == 0:
            return []
        merged = torch.cat(children, dim=0)
        return [(merged, depth + 1)]

    @staticmethod
    def _markBatchExact(
        batch: torch.Tensor,
        idx_lo: torch.Tensor,
        span: torch.Tensor,
        voxel_count: torch.Tensor,
        R: int,
        half: float,
        occ_flat: torch.Tensor,
    ) -> None:
        """对一批 voxel-AABB 有界的三角形做精确 tri-box 相交并写入 occ_flat。"""
        B = batch.shape[0]
        if B == 0:
            return

        # 把每个三角形展开成其 AABB 内的候选 voxel：用 repeat_interleave 把
        # 三角形重复 voxel_count[m] 次，再按局部线性 id 还原 (di, dj, dk)。
        total = int(voxel_count.sum().item())
        if total == 0:
            return

        tri_id = torch.repeat_interleave(
            torch.arange(B, device=batch.device), voxel_count,
        )  # (T,)

        # 每个三角形局部 voxel 线性 id：0 .. voxel_count[m]-1。
        offsets = torch.cumsum(voxel_count, dim=0) - voxel_count  # (B,)
        local = (
            torch.arange(total, device=batch.device) - offsets[tri_id]
        )  # (T,)

        span_sel = span[tri_id]  # (T, 3)
        sx, sy, sz = span_sel[:, 0], span_sel[:, 1], span_sel[:, 2]
        di = local // (sy * sz)
        rem = local % (sy * sz)
        dj = rem // sz
        dk = rem % sz

        lo_sel = idx_lo[tri_id]  # (T, 3)
        vi = lo_sel[:, 0] + di
        vj = lo_sel[:, 1] + dj
        vk = lo_sel[:, 2] + dk

        # 候选 voxel 中心世界坐标。
        centers = torch.stack([
            -0.5 + (vi.to(batch.dtype) + 0.5) / R,
            -0.5 + (vj.to(batch.dtype) + 0.5) / R,
            -0.5 + (vk.to(batch.dtype) + 0.5) / R,
        ], dim=1)  # (T, 3)

        tris_sel = batch[tri_id]  # (T, 3, 3)
        tris_local = tris_sel - centers.unsqueeze(1)  # 平移到 box 中心坐标系

        hit = VolumeMarker._triBoxOverlapBatch(tris_local, centers, half)
        if not bool(hit.any()):
            return

        flat = (vi[hit] * R + vj[hit]) * R + vk[hit]
        occ_flat[flat.unique()] = True

    @staticmethod
    def _voxelizeMeshSampled(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """采样近似 mesh 体素化（fallback）：open3d 表面均匀采样后点体素化。

        保留作为「无 open3d / 需要更快近似」时的可选路径。对极薄三角形、
        退化面、采样上限截断等情况可能漏体素，因此默认走精确路径
        _voxelizeTrianglesExact。
        """
        K = VolumeMarker.MESH_SAMPLE_OVERSAMPLE_K
        vertices_np = vertices.detach().cpu().numpy().astype(np.float64)
        faces_np = faces.detach().cpu().numpy().astype(np.int32)

        area = VolumeMarker._meshSurfaceArea(vertices.to(torch.float64), faces)
        if area <= 0.0:
            return VolumeMarker._voxelizePoints(
                vertices.to(torch.float32), R, device,
            )

        num_points = VolumeMarker._meshSampleCount(area, R, K)
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices_np),
            o3d.utility.Vector3iVector(faces_np),
        )
        sampled = o3d_mesh.sample_points_uniformly(number_of_points=num_points)
        points = torch.from_numpy(
            np.ascontiguousarray(np.asarray(sampled.points)),
        ).to(dtype=torch.float32, device=device)
        return VolumeMarker._voxelizePoints(points, R, device)

    @staticmethod
    def _voxelizeMesh(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """把三角网格按体素分辨率 R 求表面占用，返回 (R,R,R) bool 占据 mask。

        默认实现是确定性精确算法（_voxelizeTrianglesExact）：逐三角形对其
        voxel-AABB 内的候选 voxel 做 SAT tri-box 相交，覆盖网格穿过的每个
        voxel，不依赖随机采样，对薄三角形 / 退化面 / 体素边界都稳定。

        退化（零面积）三角形不构成面，SAT 法线轴会消失；这类三角形退回到
        「顶点点体素化」（顶点仍是几何上被占据的位置），与原行为一致。

        坐标约定与体素网格一致：体素覆盖 [-0.5, 0.5]^3，
        voxel index = floor((point + 0.5) * R)。
        """
        if faces.numel() == 0 or vertices.numel() == 0:
            return torch.zeros((R, R, R), dtype=torch.bool, device=device)

        vertices = vertices.to(device=device)
        faces = faces.to(device=device)
        tris = vertices[faces].to(torch.float32)  # (F, 3, 3)

        # 退化（零面积）三角形：无有效面法线，用顶点点体素化兜底。
        e0 = tris[:, 1] - tris[:, 0]
        e1 = tris[:, 2] - tris[:, 0]
        normal_norm = torch.cross(e0, e1, dim=1).norm(dim=1)
        degenerate = normal_norm <= 0.0

        occupancy = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        good_tris = tris[~degenerate]
        if good_tris.shape[0] > 0:
            occupancy |= VolumeMarker._voxelizeTrianglesExact(
                good_tris, R, device,
            )

        if bool(degenerate.any()):
            deg_pts = tris[degenerate].reshape(-1, 3)
            occupancy |= VolumeMarker._voxelizePoints(deg_pts, R, device)

        return occupancy

    @staticmethod
    def _geometryToOccupancy(
        geometry,
        R: int,
        dtype,
        device: str,
    ) -> torch.Tensor:
        """把任意支持的 geometry 自动转成 (R,R,R) bool 候选占据体素。

        类型分发（第一性原理：先看是否含三角面拓扑）：
          - open3d.TriangleMesh / trimesh.Trimesh（含面）-> 网格体素化：用
            open3d 在表面按 K*R 分辨率做面积均匀采样再体素化，覆盖网格穿过的
            所有体素，不走「直接取顶点当点云」的稀疏近似；
          - open3d.PointCloud / trimesh.PointCloud / numpy / tensor / 文件路径
            -> 点体素化（点落在哪个体素）；
          - 字符串文件路径：用 trimesh.load 加载后再按上面规则判别；含面则当
            mesh，否则取顶点当点云。
        """
        if isinstance(geometry, str):
            geometry = VolumeMarker._loadGeometryFile(geometry)

        if VolumeMarker._isTriangleMesh(geometry):
            vertices, faces = VolumeMarker._meshVerticesFaces(
                geometry, dtype, device,
            )
            VolumeMarker._warnIfOutOfDomain(vertices, '_geometryToOccupancy')
            return VolumeMarker._voxelizeMesh(vertices, faces, R, device)

        points = VolumeMarker._geometryToPoints(geometry, dtype, device)
        VolumeMarker._warnIfOutOfDomain(points, '_geometryToOccupancy')
        return VolumeMarker._voxelizePoints(points, R, device)

    @staticmethod
    def _geometryToPoints(
        geometry,
        dtype,
        device: str,
    ) -> torch.Tensor:
        """把「点云类」geometry 统一成 (N,3) 点张量。

        支持 open3d.PointCloud / trimesh.PointCloud / numpy / tensor，其余
        交给 _normalizePoints（含越界裁剪与列截断）。
        """
        if isinstance(geometry, o3d.geometry.PointCloud):
            geometry = np.asarray(geometry.points)
        elif isinstance(geometry, trimesh.points.PointCloud):
            geometry = np.asarray(geometry.vertices)
        return VolumeMarker._normalizePoints(geometry, dtype, device)

    @staticmethod
    def _loadGeometryFile(geometry_file_path: str):
        """加载几何文件，保留 mesh / 点云拓扑信息以便后续类型分发。

        含三角面则返回 trimesh.Trimesh（走 mesh 体素化），否则返回顶点 numpy
        数组（走点体素化）。.npy/.npz/.txt 等纯数组沿用 _loadPointsFile。
        """
        ext = os.path.splitext(geometry_file_path)[1].lower()
        if ext in ('.npy', '.npz', '.txt', '.xyz', '.csv'):
            return VolumeMarker._loadPointsFile(geometry_file_path)

        if not os.path.exists(geometry_file_path):
            raise FileNotFoundError(
                '[ERROR][VolumeMarker::_loadGeometryFile] '
                f'geometry file not exist: {geometry_file_path}'
            )

        loaded = trimesh.load(geometry_file_path, process=False)
        if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
            return loaded
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                g for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
            ]
            if len(meshes) > 0:
                return trimesh.util.concatenate(meshes)
        # 退回到点云顶点。
        return VolumeMarker._loadPointsFile(geometry_file_path)

    @staticmethod
    def _requireDepthLoaded(camera_list: List[Camera], caller: str) -> None:
        """确保每个相机都已加载 depth 与 valid_depth_mask。"""
        for camera in camera_list:
            if camera.depth is None or camera.valid_depth_mask is None:
                raise ValueError(
                    f'[ERROR][VolumeMarker::{caller}] '
                    'camera missing depth or valid_depth_mask; call loadDepth first.'
                )

    @staticmethod
    def _voxelizePoints(
        points: torch.Tensor,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """把世界坐标系中的点体素化为 (R, R, R) 的 bool 占据 mask。

        坐标约定与体素网格一致：体素覆盖 [-0.5, 0.5]^3，
        voxel index = floor((point + 0.5) * R)，(x, y, z) 对应 (i, j, k)。
        """
        occupancy = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        if points.numel() == 0:
            return occupancy

        points = points.to(device)
        inside = VolumeMarker._finiteInBounds(points)
        if not bool(inside.any()):
            return occupancy

        idx = VolumeMarker._pointsToVoxelIndices(points[inside], R)
        flat_idx = VolumeMarker._voxelIndicesToFlat(idx, R)

        occ_flat = occupancy.reshape(-1)
        occ_flat[flat_idx.unique()] = True
        return occ_flat.reshape(R, R, R)

    @staticmethod
    def _collectCandidatePoints(
        camera_list: List[Camera],
        dtype,
        device: str,
    ) -> torch.Tensor:
        """合并所有相机 toCCM(use_mask=False) 的真实表面点，返回 (N, 3)。"""
        VolumeMarker._requireDepthLoaded(camera_list, '_collectCandidatePoints')

        all_points: List[torch.Tensor] = []
        for camera in camera_list:
            ccm = camera.toCCM(use_mask=False)  # (H, W, 3)
            points = ccm[camera.valid_depth_mask]  # (P, 3)
            if points.numel() > 0:
                all_points.append(points.to(dtype=dtype, device=device))

        if len(all_points) == 0:
            return torch.zeros((0, 3), dtype=dtype, device=device)

        return torch.cat(all_points, dim=0)

    @staticmethod
    def _buildCandidateOccupancy(
        camera_list: List[Camera],
        geometry,
        R: int,
        dtype,
        device: str,
    ) -> torch.Tensor:
        """构建 (R, R, R) bool 候选占据体素，按输入语义选择 depth 依赖。

        第一性原理：候选占据的来源决定了对相机 depth 的依赖。
          - 显式 geometry：候选完全由几何先验决定，只需要相机位姿/内参
            来渲染遮挡，不需要任何 depth；因此这里不校验 depth。
          - geometry 为 None：候选来自多相机 depth 反投影的真实表面点，
            此时每个相机必须已加载 depth 与 valid_depth_mask。
        """
        if geometry is not None:
            return VolumeMarker._geometryToOccupancy(geometry, R, dtype, device)

        candidate_points = VolumeMarker._collectCandidatePoints(
            camera_list, dtype, device,
        )
        VolumeMarker._warnIfOutOfDomain(
            candidate_points, '_buildCandidateOccupancy',
        )
        return VolumeMarker._voxelizePoints(candidate_points, R, device)

    # ------------------------------------------------------------------
    # 候选体素并集的边界 mesh
    # ------------------------------------------------------------------
    # 6 个面方向：邻居偏移 + 该面 4 个角点（单位立方体局部坐标 0/1）。
    # 角点顺序保证三角化后外法线朝向 candidate 外侧（这里只用于遮挡，
    # nvdiffrast 不做背面剔除，朝向不影响命中判定，仅保持几何闭合）。
    _FACE_DEFS = (
        # +x 面
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        # -x 面
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        # +y 面
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        # -y 面
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        # +z 面
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        # -z 面
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    )

    @staticmethod
    def _emptyBoundaryMesh(
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """空候选时返回统一的空 (vertices, faces, triangle_to_voxel)。"""
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.int64, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )

    @staticmethod
    def _exposedFacesForDirection(
        candidate: torch.Tensor,
        occ_idx: torch.Tensor,
        base: torch.Tensor,
        voxel_flat: torch.Tensor,
        offset: Tuple[int, int, int],
        corners,
        R: int,
        voxel_size: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """为单个面方向取出所有外露 quad 及其对应 voxel flat index。

        外露面 = 该方向邻居越界或为空体素。返回 (quad_verts, quad_voxel)，
        若该方向无外露面返回 None。
        """
        device = candidate.device
        i, j, k = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]
        dx, dy, dz = offset

        ni, nj, nk = i + dx, j + dy, k + dz
        out_of_grid = (
            (ni < 0) | (ni >= R)
            | (nj < 0) | (nj >= R)
            | (nk < 0) | (nk >= R)
        )
        neighbor_occupied = candidate[
            ni.clamp(0, R - 1), nj.clamp(0, R - 1), nk.clamp(0, R - 1),
        ] & (~out_of_grid)

        exposed = ~neighbor_occupied  # (M,)
        if not bool(exposed.any()):
            return None

        sel_base = base[exposed]  # (E, 3)
        sel_voxel = voxel_flat[exposed]  # (E,)

        corner_offsets = torch.tensor(
            corners, dtype=torch.float32, device=device,
        ) * voxel_size  # (4, 3)

        quad = sel_base.unsqueeze(1) + corner_offsets.unsqueeze(0)  # (E, 4, 3)
        return quad, sel_voxel

    @staticmethod
    def _quadsToTriangles(
        Q: int,
        device,
    ) -> torch.Tensor:
        """为 Q 个 quad 生成 (Q*2, 3) 三角面索引（顶点按每 quad 4 个连续排布）。"""
        quad_offset = torch.arange(Q, device=device, dtype=torch.int64) * 4
        tri0 = torch.stack(
            [quad_offset + 0, quad_offset + 1, quad_offset + 2], dim=-1,
        )
        tri1 = torch.stack(
            [quad_offset + 0, quad_offset + 2, quad_offset + 3], dim=-1,
        )
        return torch.stack([tri0, tri1], dim=1).reshape(-1, 3)  # (Q*2, 3)

    @staticmethod
    def _buildVoxelBoundaryMesh(
        candidate: torch.Tensor,
        R: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为候选体素并集构建只含外露面的立方体表面 mesh。

        只生成与空体素（或网格外）相邻的外表面，跳过两个相邻候选体素之间
        的内部面，从而把面数从 6*candidate_count 降到外表面数量。

        Args:
            candidate: (R, R, R) bool，候选占据体素。
            R: 体素分辨率。

        Returns:
            vertices: (V, 3) float32，位于 [-0.5, 0.5]^3 世界坐标。
            faces: (F, 3) int64，三角面顶点索引。
            triangle_to_voxel: (F,) int64，每个三角对应的 voxel flat index
                （flat = (i * R + j) * R + k）。
        """
        device = candidate.device
        occ_idx = torch.nonzero(candidate, as_tuple=False)  # (M, 3) -> (i, j, k)
        if occ_idx.shape[0] == 0:
            return VolumeMarker._emptyBoundaryMesh(device)

        voxel_size = 1.0 / R
        base = (occ_idx.to(torch.float32) - 0.5 * R) * voxel_size  # voxel 最小角世界坐标
        voxel_flat = VolumeMarker._voxelIndicesToFlat(occ_idx, R)  # (M,)

        all_quad_verts: List[torch.Tensor] = []
        all_quad_voxel: List[torch.Tensor] = []

        for offset, corners in VolumeMarker._FACE_DEFS:
            face = VolumeMarker._exposedFacesForDirection(
                candidate, occ_idx, base, voxel_flat,
                offset, corners, R, voxel_size,
            )
            if face is None:
                continue
            quad, sel_voxel = face
            all_quad_verts.append(quad)
            all_quad_voxel.append(sel_voxel)

        if len(all_quad_verts) == 0:
            return VolumeMarker._emptyBoundaryMesh(device)

        quad_verts = torch.cat(all_quad_verts, dim=0)  # (Q, 4, 3)
        quad_voxel = torch.cat(all_quad_voxel, dim=0)  # (Q,)
        Q = quad_verts.shape[0]

        vertices = quad_verts.reshape(-1, 3)  # (Q*4, 3)
        faces = VolumeMarker._quadsToTriangles(Q, device)  # (Q*2, 3)
        triangle_to_voxel = quad_voxel.repeat_interleave(2)  # (Q*2,)

        return vertices, faces, triangle_to_voxel

    # ------------------------------------------------------------------
    # 用 nvdiffrast 渲染候选 mesh，统一产出 VALID 与 FREE 证据
    # ------------------------------------------------------------------
    @staticmethod
    def _sampleRenderedDepth(
        rendered_depth: torch.Tensor,
        hit_mask: torch.Tensor,
        uv: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在候选 mesh 渲染出的 depth 图上，按 voxel 中心 UV 做最近邻采样。

        Args:
            rendered_depth: (H, W) 渲染 mesh 的相机前方正距离，背景为 0。
            hit_mask: (H, W) bool，像素是否命中 mesh。
            uv: (N, 2) voxel 中心归一化 UV [0, 1]，与 camera UV 约定一致。
            height/width: 渲染图分辨率。

        Returns:
            sampled_depth: (N,) 采样到的渲染表面深度（背景/越界处为 0）。
            surface_hit:   (N,) bool，UV 在画面内且该像素命中 mesh 表面。
            background:    (N,) bool，UV 在画面内但该像素是背景（无 mesh）。
        """
        in_frame = (
            torch.isfinite(uv).all(dim=-1)
            & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        )

        uv_safe = torch.where(
            in_frame.unsqueeze(-1), uv, torch.zeros_like(uv),
        )

        u_nearest = (uv_safe[:, 0] * width).long().clamp(0, width - 1)
        v_nearest = ((1.0 - uv_safe[:, 1]) * height).long().clamp(0, height - 1)

        raw_depth = rendered_depth[v_nearest, u_nearest]
        pix_hit = hit_mask[v_nearest, u_nearest]

        surface_hit = in_frame & pix_hit
        background = in_frame & (~pix_hit)

        sampled_depth = torch.where(
            surface_hit, raw_depth, torch.zeros_like(raw_depth),
        )
        return sampled_depth, surface_hit, background

    @staticmethod
    def _projectCentersForCamera(
        camera: Camera,
        centers_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 voxel 中心投影到某相机，返回 (uv, point_depth)。

        Args:
            centers_flat: (N, 3) voxel 中心世界坐标（与相机同 dtype/device）。

        Returns:
            uv:          (N, 2) 归一化 UV [0, 1]，与 camera UV 约定一致。
            point_depth: (N,) voxel 中心在相机坐标系下的前方正深度。
        """
        uv = camera.project_points_to_uv(centers_flat)  # (N, 2)

        centers_homo = torch.cat(
            [centers_flat, torch.ones_like(centers_flat[..., :1])], dim=-1,
        )
        point_depth = -(centers_homo @ camera.world2camera.T)[..., 2]  # (N,)
        return uv, point_depth

    @staticmethod
    def _cameraDepthRange(
        camera: Camera,
        vertices_tensor: torch.Tensor,
    ) -> Optional[Tuple[float, float]]:
        """从候选 mesh 顶点估计该相机所需的 (near, far) 裁剪面。

        第一性原理：nvdiffrast 默认 near/far 是为单位尺度场景设定的；当相机
        距离或场景尺度偏离默认 [near, far] 时，候选 mesh 会被裁掉，导致无
        VALID 命中、且背景误判 FREE。这里用顶点在相机前方的真实深度范围动态
        设定裁剪面，并留出余量与正下限，保证整段候选几何都在裁剪范围内。

        Returns:
            (near, far)；若没有任何顶点落在相机前方则返回 None（沿用默认）。
        """
        if vertices_tensor.shape[0] == 0:
            return None

        verts_homo = torch.cat(
            [vertices_tensor,
             torch.ones_like(vertices_tensor[..., :1])], dim=-1,
        )
        # 相机前方正深度 = -Z_camera。
        depth = -(verts_homo @ camera.world2camera.T)[..., 2]
        front = depth[depth > 0]
        if front.numel() == 0:
            return None

        d_min = float(front.min().item())
        d_max = float(front.max().item())
        # 余量：near 取最小深度的 0.5 倍（且有正下限），far 取最大深度的 2 倍，
        # 充分包裹候选几何，避免边界顶点被裁。
        near = max(d_min * 0.5, 1e-4)
        far = max(d_max * 2.0, near * 10.0)
        return near, far

    @staticmethod
    def _renderClassify(
        camera_list: List[Camera],
        vertices: torch.Tensor,
        faces: torch.Tensor,
        triangle_to_voxel: torch.Tensor,
        R: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """渲染候选体素并集边界 mesh，一次性产出 VALID 与 FREE 证据。

        对每个相机渲染稳定遮挡关系的 depth（关闭抗锯齿），同时得到：
          - 被击中的 face id -> 真正可见的候选体素（VALID）。
          - voxel 中心相对渲染表面深度的位置 -> 自由空间证据（FREE）。

        FREE 判定（第一性原理）：候选 mesh 是该视角的占据几何「地面真值」。
        一个 voxel 在某相机下属于自由空间，当且仅当从相机到该 voxel 中心的
        视线在到达占据表面之前就经过了它，即：
          - voxel 中心投影像素命中 mesh，且 voxel 深度 < 渲染表面深度（更近）；
          - 或 voxel 中心投影像素是背景（整条视线无占据）。
        这保证了「比任意 VALID/占据表面更靠近相机且视线无遮挡」的 voxel 必为
        FREE —— 不会再出现比 VALID 更近却 UNKNOWN 的矛盾。

        Returns:
            visible_candidate: (R, R, R) bool，跨视角可见候选体素并集。
            free_any:          (R, R, R) bool，至少一个视角给出 FREE 证据。
            observed_any:      (R, R, R) bool，voxel 中心至少被一个相机看到
                               （投影落在画面内且在相机前方）。
        """
        visible = torch.zeros((R, R, R), dtype=torch.bool, device=device)
        free_any = torch.zeros((R, R, R), dtype=torch.bool, device=device)
        observed_any = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        if faces.shape[0] == 0:
            return visible, free_any, observed_any

        # 延迟导入：nvdiffrast 仅在真正渲染时才需要，使纯 CPU 逻辑
        # （体素化 / 边界 mesh）在无 nvdiffrast 环境也能使用。
        from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

        # nvdiffrast 不读取 visual，传入纯几何 trimesh 即可。
        mesh = trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy().astype(np.float64),
            faces=faces.detach().cpu().numpy().astype(np.int64),
            process=False,
        )

        visible_flat = visible.reshape(-1)
        triangle_to_voxel = triangle_to_voxel.to(device)

        # voxel 中心只依赖 (R, dtype, device)，跨相机复用以避免每个相机重复
        # 创建 R^3 个中心点。
        centers_cache: dict = {}
        # 同一 device 上的 float32 顶点张量也跨相机复用。
        vertices_cache: dict = {}

        for camera in camera_list:
            vertices_tensor = vertices_cache.get(camera.device)
            if vertices_tensor is None:
                vertices_tensor = vertices.to(
                    dtype=torch.float32, device=camera.device,
                )
                vertices_cache[camera.device] = vertices_tensor

            # 动态裁剪面：用候选 mesh 顶点的真实深度范围设定 near/far，避免
            # 相机距离/尺度偏离默认裁剪范围时把候选几何裁掉。
            depth_range = VolumeMarker._cameraDepthRange(camera, vertices_tensor)
            near_far = {} if depth_range is None else {
                'near': depth_range[0], 'far': depth_range[1],
            }

            render_result = NVDiffRastRenderer.render(
                mesh=mesh,
                camera=camera,
                render_types=['depth'],
                vertices_tensor=vertices_tensor,
                enable_antialias=False,
                enable_lighting=False,
                **near_far,
            )

            rast_out = render_result['rasterize_output']  # (H, W, 4)
            rendered_depth = render_result['depth']  # (H, W) 相机前方正距离

            height, width = int(rendered_depth.shape[0]), int(rendered_depth.shape[1])
            hit_mask = rast_out[..., 3] > 0  # (H, W)

            # --- VALID：命中 face -> voxel ---
            tri_id = rast_out[..., 3].reshape(-1).long()  # (H*W,)
            hit = tri_id > 0
            if bool(hit.any()):
                local_face_id = (tri_id[hit] - 1).to(device)
                hit_voxel_flat = triangle_to_voxel[local_face_id].unique()
                visible_flat[hit_voxel_flat] = True

            # --- FREE：voxel 中心 vs 渲染表面深度 ---
            cache_key = (camera.dtype, camera.device)
            centers_flat = centers_cache.get(cache_key)
            if centers_flat is None:
                centers_flat = VolumeMarker.createVoxelCenters(
                    R, camera.dtype, camera.device,
                ).reshape(-1, 3)  # (R^3, 3)
                centers_cache[cache_key] = centers_flat

            uv, point_depth = VolumeMarker._projectCentersForCamera(
                camera, centers_flat,
            )

            sampled_depth, surface_hit, background = \
                VolumeMarker._sampleRenderedDepth(
                    rendered_depth, hit_mask, uv, height, width,
                )

            eps = torch.clamp(point_depth.abs() * 1e-4, min=1e-6)
            # voxel 中心明确位于渲染表面前方（更靠近相机）。
            in_front = surface_hit & (point_depth < sampled_depth - eps)
            voxel_free = (in_front | background).reshape(R, R, R).to(device)

            # observed：投影落在画面内（surface_hit 或 background 都意味着
            # voxel 中心在相机前方且 UV 在画面内）。
            observed = (surface_hit | background).reshape(R, R, R).to(device)

            free_any = free_any | voxel_free
            observed_any = observed_any | observed

        return visible_flat.reshape(R, R, R), free_any, observed_any

    # ------------------------------------------------------------------
    # L1 距离场 / FREE_KN 标签（高效并行原子函数）
    # ------------------------------------------------------------------
    @staticmethod
    def _minPlusTransform1D(
        distance: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """沿单轴做精确 1D min-plus 距离传播，返回新张量（不原地改写）。

        第一性原理：1D 距离变换
            out[i] = min_j (dist[j] + |i - j|)
        可拆成前向 + 后向两个单调扫描，并用「坐标补偿 + 单调 cummin」无分支
        实现，避免 Hillis-Steele 倍增中重叠切片 in-place 写入对内核语义的
        依赖：
            forward[i]  = i  + min_{j <= i}(dist[j] - j)   = i  + cummin(dist - i)
            backward[i] = -i + min_{j >= i}(dist[j] + j)   = -i + cummin_rev(dist + i)
            out[i]      = min(forward[i], backward[i])

        其中坐标 i 沿 axis 取值 0..L-1。cummin 是无重叠读写的纯前缀算子，
        全张量并行、CPU / GPU 通用，无任何主机-设备同步或数据相关分支。
        """
        L = distance.shape[axis]
        shape = [1] * distance.ndim
        shape[axis] = L
        coord = torch.arange(
            L, dtype=distance.dtype, device=distance.device,
        ).reshape(shape)

        # forward: i + cummin_{j<=i}(dist[j] - j)
        forward = coord + torch.cummin(distance - coord, dim=axis).values

        # backward: -i + cummin_{j>=i}(dist[j] + j)，用 flip 把后缀 cummin 转成前缀
        plus = (distance + coord).flip(axis)
        backward = -coord + torch.cummin(plus, dim=axis).values.flip(axis)

        return torch.minimum(forward, backward)

    @staticmethod
    def computeL1DistanceToMask(source_mask: torch.Tensor) -> torch.Tensor:
        """计算每个 voxel 到 source_mask 的精确 L1（曼哈顿）距离场。

        第一性原理：L1 距离变换可按轴分离——N 维距离 = 依次沿每个轴做一次
        1D min-plus 距离传播。每次 1D 传播用「坐标补偿 + 单调 cummin」实现
        （见 _minPlusTransform1D），总计 N 次 cummin 算子，全张量并行、
        CPU / GPU 通用，无主机-设备同步、无早停分支、无重叠切片原地写。

        Args:
            source_mask: (R, R, R) bool（或任意维 bool），距离源（距离 0）。

        Returns:
            与输入同形状的 int64 精确 L1 距离场；source 为空时全为
            INF = sum(shape)（严格大于网格内最大可能距离，对应「无限远」
            哨兵语义；对立方 (R,R,R) 即 3R）。
        """
        inf = int(sum(source_mask.shape))
        distance = torch.where(
            source_mask,
            torch.zeros_like(source_mask, dtype=torch.int64),
            torch.full_like(source_mask, inf, dtype=torch.int64),
        )

        for axis in range(source_mask.ndim):
            distance = VolumeMarker._minPlusTransform1D(distance, axis)

        return distance.clamp(max=inf)

    @staticmethod
    def assignFreeLabels(
        labels: torch.Tensor,
        free_mask: torch.Tensor,
    ) -> torch.Tensor:
        """把 free_mask 指定的 voxel 写为 FREE_KN = -K。

        K = 到最近「非 FREE」voxel（VALID 或 UNKNOWN）的 L1 距离。
        第一性原理：FREE 的层级度量的是「离开自由空间需要走多远」，
        VALID（占据）与 UNKNOWN（遮挡/无信息）同为非自由边界，距离源
        必须包含两者——否则紧贴 UNKNOWN 的 FREE 会被远处 VALID 错误地
        赋予过大的 K。

        UNKNOWN / VALID 不动；free_mask 处一次向量化写入 -distance。
        若距离源为空（整个空间都是 FREE），距离场自然全为 INF = 3R，
        与「无限远 FREE」哨兵 freeLabelInf(R) = -(3R) 一致。

        Args:
            labels: (R, R, R) int64 标签场（UNKNOWN=0 / VALID=1）。
            free_mask: (R, R, R) bool，需要赋 FREE_KN 标签的 voxel。

        Returns:
            (R, R, R) int64 新标签场（不修改输入）。
        """
        labels = labels.clone()

        # 距离源 = 所有非 FREE voxel（VALID 与 UNKNOWN）。
        distance = VolumeMarker.computeL1DistanceToMask(~free_mask)
        labels[free_mask] = -distance[free_mask]
        return labels

    # ------------------------------------------------------------------
    # FREE / UNKNOWN 判定（非候选体素）
    # ------------------------------------------------------------------
    @staticmethod
    def createGridVertices(
        volume_resolution: int,
        dtype,
        device: str,
    ) -> torch.Tensor:
        """
        在 [-0.5, 0.5] 之间创建 (R+1)^3 个 voxel 顶点。
        返回: (R+1, R+1, R+1, 3)，按 (x, y, z) 的 ij 顺序排列。
        """
        coords = torch.linspace(
            -0.5, 0.5, volume_resolution + 1, dtype=dtype, device=device,
        )
        return VolumeMarker._makeIjkGrid(coords)

    @staticmethod
    def createVoxelCenters(
        volume_resolution: int,
        dtype,
        device: str,
    ) -> torch.Tensor:
        """
        在 [-0.5, 0.5] 之间创建 R^3 个 voxel 的中心点。
        voxel (i, j, k) 中心 = -0.5 + (idx + 0.5) / R。
        返回: (R, R, R, 3)，按 (x, y, z) 的 ij 顺序排列。
        """
        coords = (
            torch.arange(volume_resolution, dtype=dtype, device=device) + 0.5
        ) / volume_resolution - 0.5
        return VolumeMarker._makeIjkGrid(coords)

    # ------------------------------------------------------------------
    # 标签合并原子
    # ------------------------------------------------------------------
    @staticmethod
    def _classifyFreeMask(
        candidate: torch.Tensor,
        free_any: torch.Tensor,
        observed_any: torch.Tensor,
        unobserved_policy: str = 'free',
    ) -> torch.Tensor:
        """从渲染证据派生「应赋 FREE_KN」的非候选 voxel mask。

        第一性原理：FREE 只可能是非候选 voxel（候选要么 VALID 要么 UNKNOWN）。
        一个非候选 voxel 判为 FREE 的依据有两类：
          - 任一视角给出直接 FREE 证据（在渲染表面前方或投影到背景）；
          - 从未被任何相机观测到（视锥外/相机后方），按 unobserved_policy 决定。

        Args:
            candidate: (R, R, R) bool 候选占据体素。
            free_any: (R, R, R) bool，至少一个视角给出 FREE 证据。
            observed_any: (R, R, R) bool，至少被一个相机观测到。
            unobserved_policy: 未观测非候选的归类策略：
                'free'    -> 视为 FREE（无物体信息，默认，兼容现有下游）；
                'unknown' -> 视为 UNKNOWN（保守，不假设视锥外为空）。

        Returns:
            (R, R, R) bool，需要写 FREE_KN 标签的 voxel。
        """
        non_candidate = ~candidate
        if unobserved_policy == 'free':
            return non_candidate & (free_any | (~observed_any))
        if unobserved_policy == 'unknown':
            return non_candidate & free_any
        raise ValueError(
            '[ERROR][VolumeMarker::_classifyFreeMask] '
            f"unobserved_policy must be 'free' or 'unknown', got {unobserved_policy}"
        )

    @staticmethod
    def _mergeLabels(
        R: int,
        visible_candidate: torch.Tensor,
        free_mask: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """合并 VALID / FREE_KN / UNKNOWN 为最终标签场。

        UNKNOWN(0) 为底；visible_candidate -> VALID(1)；free_mask -> FREE_KN(-K)。
        visible_candidate 与 free_mask 不相交（前者候选、后者非候选），合并
        顺序无歧义。
        """
        labels = VolumeMarker._emptyLabels(R, VolumeMarker.UNKNOWN, device)
        labels = torch.where(
            visible_candidate,
            torch.full_like(labels, VolumeMarker.VALID),
            labels,
        )
        return VolumeMarker.assignFreeLabels(labels, free_mask)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    @staticmethod
    def markVisible(
        camera_list: List[Camera],
        volume_resolution: int,
        geometry: Optional[Union[
        o3d.geometry.TriangleMesh,
        o3d.geometry.PointCloud,
        trimesh.PointCloud,
        trimesh.Trimesh,
        np.ndarray,
        torch.Tensor,
    ]] = None,
    ) -> torch.Tensor:
        """基于候选占据 + 立方体 mesh 渲染对空间体素打标签。

        算法（第一性原理重构）：
          1. 候选占据：
               - 若提供 geometry，按其真实类型转成候选占据体素：
                   * 三角网格（open3d.TriangleMesh / trimesh.Trimesh，含面）->
                     用 open3d 在表面按 volume_resolution 的 K 倍分辨率做面积
                     均匀采样得到稠密点云再体素化，覆盖网格穿过的所有体素；
                   * 点云 / numpy / tensor / 点云文件 -> 点体素化（点落在哪个
                     体素）；
                 提供 geometry 时只需相机位姿/内参来渲染遮挡，不要求 depth。
               - 否则用所有 camera.toCCM(use_mask=False) 的并集体素化，此路径
                 要求每个相机已加载 depth 与 valid_depth_mask。
          2. 边界 mesh：对候选体素并集构造只含外露面的立方体表面 mesh，并记录
             每个三角对应的 voxel。
          3. 单次渲染、统一判定（_renderClassify）：从每个相机用 nvdiffrast
             渲染该 mesh 的 depth（关闭抗锯齿），同源得到两类证据：
               - face id 命中 -> 真正可见的候选体素 -> VALID；
               - voxel 中心相对渲染表面深度的位置 -> FREE / 遮挡。
             候选 mesh 是该视角的占据几何「地面真值」，因此遮挡关系自洽：
               * voxel 中心在渲染表面前方（更近）或投影到背景 -> 该视角 FREE；
               * voxel 中心在渲染表面后方（被遮挡）-> 该视角无信息。
             这从根本上消除「比某个 VALID 更靠近相机却被判 UNKNOWN」的矛盾。
          4. 标签合并：
               - 候选且任一视角可见 -> VALID；
               - 候选但被所有视角完全遮挡 -> UNKNOWN（不被 FREE 改写）；
               - 非候选且任一视角 FREE -> FREE；
               - 非候选且从未被任何相机观测到 -> FREE（无物体信息）；
               - 其余非候选（被遮挡、无 FREE 证据）-> UNKNOWN。
             若不存在任何候选体素，则空间无占据，整体为 FREE。
          5. FREE_KN 标签（assignFreeLabels）：对所有 FREE voxel 全量计算
             到最近非 FREE voxel（VALID 或 UNKNOWN）的 L1 距离 K，写为
             FREE_KN = -K；若整个空间都是 FREE（无任何 VALID/UNKNOWN），
             所有 FREE 写为哨兵 -(3R)。

        返回: (R, R, R) 的 int64 tensor，编码 UNKNOWN=0, VALID=1,
        FREE_KN=-K（K = 到最近 VALID/UNKNOWN 的 L1 距离；全 FREE 时为 -(3R)）。
        """
        R = VolumeMarker._requirePositiveResolution(volume_resolution, 'markVisible')

        if len(camera_list) == 0:
            # 没有任何视角，整个空间一定没有物体信息，且无 VALID -> 哨兵 FREE。
            return VolumeMarker._emptyLabels(R, VolumeMarker.freeLabelInf(R))

        ref_camera = camera_list[0]
        device = ref_camera.device
        dtype = ref_camera.dtype

        # --- 1. 候选占据体素 ---
        # depth 依赖收口在 _buildCandidateOccupancy：仅 geometry 为 None 的
        # depth 候选路径才要求 depth；显式 geometry 只需相机位姿/内参。
        candidate = VolumeMarker._buildCandidateOccupancy(
            camera_list, geometry, R, dtype, device,
        )

        # --- 无候选体素：空间无占据，整体为 FREE（无 VALID -> 哨兵） ---
        if not bool(candidate.any()):
            return VolumeMarker._emptyLabels(
                R, VolumeMarker.freeLabelInf(R), device,
            )

        # --- 2 & 3. 边界 mesh + 单次渲染同源产出 VALID/FREE/observed ---
        vertices, faces, triangle_to_voxel = \
            VolumeMarker._buildVoxelBoundaryMesh(candidate, R)

        visible_candidate, free_any, observed_any = \
            VolumeMarker._renderClassify(
                camera_list, vertices, faces, triangle_to_voxel, R, device,
            )

        # --- 4. 标签合并 ---
        free_mask = VolumeMarker._classifyFreeMask(
            candidate, free_any, observed_any, unobserved_policy='free',
        )

        # --- 5. VALID / FREE_KN / UNKNOWN 合并（含全量 L1 距离 -> -K） ---
        return VolumeMarker._mergeLabels(R, visible_candidate, free_mask, device)
