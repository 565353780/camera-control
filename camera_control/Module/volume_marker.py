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
          - open3d.TriangleMesh / trimesh.Trimesh（含面）-> 网格体素化：默认走
            确定性精确 SAT 三角-体素相交（_voxelizeMesh -> _voxelizeTrianglesExact），
            覆盖网格穿过的每个体素，不依赖随机采样，也不走「直接取顶点当点云」
            的稀疏近似（open3d 表面采样仅作为 _voxelizeMeshSampled fallback 保留）；
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
    def _depthObservationDomain(camera: Camera) -> torch.Tensor:
        """返回某相机 depth 网格上的可信观测域 (Hd, Wd) bool。

        第一性原理：候选只能来自相机的可信观测域。
          - 无 camera.mask：观测域 = valid_depth_mask（所有有效深度像素）。
          - 有 camera.mask：观测域 = valid_depth_mask 且落在 mask 内。mask 是
            「该相机的可信观测域」，mask 外既不是占据也不是空，不得贡献候选。
        mask 与 depth 分辨率可能不同，统一用 depth UV 在 mask 上最近邻采样对齐。
        """
        domain = camera.valid_depth_mask
        if getattr(camera, 'mask', None) is None:
            return domain
        uv = camera.toDepthUV()  # (Hd, Wd, 2)
        in_mask = camera.sampleMaskAtUV(uv).to(
            dtype=torch.bool, device=domain.device,
        )
        return domain & in_mask

    @staticmethod
    def _collectCandidatePoints(
        camera_list: List[Camera],
        dtype,
        device: str,
    ) -> torch.Tensor:
        """合并所有相机可信观测域内的真实表面点，返回 (N, 3)。

        每个相机只取「valid_depth_mask 且（若有 mask）落在 mask 内」的像素的
        CCM 点：mask 外的深度点不属于该相机的可信观测域，不进入候选。
        """
        VolumeMarker._requireDepthLoaded(camera_list, '_collectCandidatePoints')

        all_points: List[torch.Tensor] = []
        for camera in camera_list:
            ccm = camera.toCCM(use_mask=False)  # (H, W, 3)
            domain = VolumeMarker._depthObservationDomain(camera)  # (H, W)
            points = ccm[domain]  # (P, 3)
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
    def _sampleRenderedDepthAtUV(
        rendered_depth: torch.Tensor,
        hit_mask: torch.Tensor,
        observation_mask: torch.Tensor,
        uv: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在某次渲染结果上按任意点的 UV 做最近邻采样（中心或 8 顶点通用）。

        Args:
            rendered_depth: (H, W) 渲染 mesh 的相机前方正距离，背景为 0。
            hit_mask: (H, W) bool，像素是否命中 mesh 表面。
            observation_mask: (H, W) bool，该相机的可信观测域（mask 外为 False）。
            uv: (N, 2) 归一化 UV [0, 1]，与 camera UV 约定一致。
            height/width: 渲染图分辨率（须与 observation_mask 同形状）。

        Returns:
            sampled_depth: (N,) 命中处渲染表面深度（背景/越界/被遮挡为 0）。
            surface_hit:   (N,) bool，UV 在画面内且该像素命中渲染表面。
            in_domain:     (N,) bool，UV 在画面内且该像素位于相机观测域内。
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
        pix_obs = observation_mask[v_nearest, u_nearest]

        surface_hit = in_frame & pix_hit
        in_domain = in_frame & pix_obs

        sampled_depth = torch.where(
            surface_hit, raw_depth, torch.zeros_like(raw_depth),
        )
        return sampled_depth, surface_hit, in_domain

    @staticmethod
    def _projectPointsForCamera(
        camera: Camera,
        points_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把任意世界坐标点投影到某相机，返回 (uv, point_depth)。

        既用于 voxel 中心，也用于 voxel 的 8 个顶点（调用方先 flatten 再 reshape）。

        Args:
            points_flat: (N, 3) 点的世界坐标（与相机同 dtype/device）。

        Returns:
            uv:          (N, 2) 归一化 UV [0, 1]，与 camera UV 约定一致。
            point_depth: (N,) 点在相机坐标系下的前方正深度（相机后方为负）。
        """
        uv = camera.project_points_to_uv(points_flat)  # (N, 2)

        points_homo = torch.cat(
            [points_flat, torch.ones_like(points_flat[..., :1])], dim=-1,
        )
        point_depth = -(points_homo @ camera.world2camera.T)[..., 2]  # (N,)
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
    def _createVoxelCorners(
        R: int,
        dtype,
        device: str,
    ) -> torch.Tensor:
        """生成 (R, R, R, 8, 3) 每个体素的 8 个顶点世界坐标。

        voxel (i, j, k) 的顶点 = -0.5 + (idx + c) / R，c in {0, 1}，与
        createGridVertices() 的网格顶点约定完全一致：顶点落在体素 8 个角。
        顶点顺序固定为 (cx, cy, cz) 的二进制枚举：
            0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0)
            4:(0,0,1) 5:(1,0,1) 6:(0,1,1) 7:(1,1,1)
        顺序仅影响列对齐，「8 顶点全满足」的判定与顺序无关。
        """
        voxel_size = 1.0 / R
        # 每个体素最小角（c=0,0,0）的世界坐标沿各轴 = idx / R - 0.5。
        low = torch.arange(R, dtype=dtype, device=device) / R - 0.5  # (R,)
        lx, ly, lz = torch.meshgrid(low, low, low, indexing='ij')
        low_grid = torch.stack([lx, ly, lz], dim=-1)  # (R, R, R, 3)

        corner_unit = torch.tensor(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            ],
            dtype=dtype, device=device,
        ) * voxel_size  # (8, 3)

        return low_grid.unsqueeze(-2) + corner_unit  # (R, R, R, 8, 3)

    @staticmethod
    def _cameraPixelObservationMask(
        camera: Camera,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """单相机像素级可信观测域 (H, W) bool，对齐到渲染分辨率。

        第一性原理：mask 是「该相机能提供可信观测的画面区域」，不是空域。
          - camera.mask is None：整幅画面都可观测 -> 全 True。
          - 存在 mask：用 sampleMaskWithSize 在渲染分辨率上对齐采样，mask 内
            为 True；mask 外不得产生 VALID/FREE 证据（保持 UNKNOWN）。
        """
        if getattr(camera, 'mask', None) is None:
            return torch.ones(
                (height, width), dtype=torch.bool, device=camera.device,
            )
        return camera.sampleMaskWithSize(width, height).to(
            dtype=torch.bool, device=camera.device,
        )

    @staticmethod
    def _trimeshFromTensors(
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> trimesh.Trimesh:
        """从顶点/面张量构造纯几何 trimesh（nvdiffrast 不读取 visual）。"""
        return trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy().astype(np.float64),
            faces=faces.detach().cpu().numpy().astype(np.int64),
            process=False,
        )

    @staticmethod
    def _renderMeshDepthForCamera(
        mesh: trimesh.Trimesh,
        camera: Camera,
        vertices_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """单相机渲染给定 mesh 的稳定遮挡 depth（关闭抗锯齿）。

        候选边界 mesh 与 visible-VALID-only mesh 共用此原子，统一处理动态
        near/far 裁剪面，避免相机尺度偏离默认裁剪范围时几何被裁掉。

        Returns:
            rendered_depth: (H, W) 相机前方正距离，背景为 0。
            hit_mask:       (H, W) bool，像素是否命中 mesh 表面。
            rast_out:       (H, W, 4) 光栅化输出（第 4 通道为 1 起的三角 id）。
            height, width:  渲染分辨率。
        """
        # 延迟导入：nvdiffrast 仅在真正渲染时才需要，使纯 CPU 逻辑
        # （体素化 / 边界 mesh）在无 nvdiffrast 环境也能使用。
        from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

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
        height = int(rendered_depth.shape[0])
        width = int(rendered_depth.shape[1])
        hit_mask = rast_out[..., 3] > 0  # (H, W)
        return rendered_depth, hit_mask, rast_out, height, width

    @staticmethod
    def _visibleVoxelsFromRaster(
        rast_out: torch.Tensor,
        triangle_to_voxel: torch.Tensor,
        R: int,
        device: str,
        pixel_observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        """把渲染命中的 face 映射成真正可见的候选 voxel（受观测域门控）。

        只统计「命中 mesh 且像素位于观测域内」的命中：mask 外的命中不属于该
        相机的可信观测，不算 VALID。

        Returns:
            (R, R, R) bool，该视角可见的候选体素。
        """
        visible_flat = torch.zeros(R * R * R, dtype=torch.bool, device=device)
        tri_id = rast_out[..., 3].reshape(-1).long()  # (H*W,)
        obs = pixel_observation_mask.reshape(-1).to(tri_id.device)
        hit = (tri_id > 0) & obs
        if bool(hit.any()):
            local_face_id = (tri_id[hit] - 1).to(device)
            hit_voxel_flat = triangle_to_voxel[local_face_id].unique()
            visible_flat[hit_voxel_flat] = True
        return visible_flat.reshape(R, R, R)

    @staticmethod
    def _freeEvidenceEightVerticesInFront(
        depth8: torch.Tensor,
        sampled_depth8: torch.Tensor,
        surface_hit8: torch.Tensor,
        in_domain8: torch.Tensor,
    ) -> torch.Tensor:
        """8 顶点严格 FREE 证明：唯有整个体素都在可见 VALID 表面前方才成立。

        第一性原理：FREE 必须是「被可信射线证明为空」的体素。voxel 中心单点
        在表面前方只能证明中心点为空，不能证明整个体素为空。严格判据要求
        体素的 8 个顶点同时满足：
          - in_domain：顶点投影落在画面内且位于相机可信观测域（mask 内）；
          - surface_hit：顶点投影像素命中 visible-VALID 渲染表面（背后确有
            真正可见的 VALID 作为遮挡终点）；
          - 严格前方：顶点的相机前方深度 < 该像素 visible-VALID 渲染深度 - eps。
        只要某个视角让全部 8 顶点都通过，则相机与可见 VALID 表面之间完整包住
        该体素，可证明其为 FREE。

        Args:
            depth8/sampled_depth8/surface_hit8/in_domain8: 均为 (N, 8)。

        Returns:
            (N,) bool，该视角下体素是否获得 FREE 证据。
        """
        eps = torch.clamp(depth8.abs() * 1e-4, min=1e-6)
        in_front = surface_hit8 & in_domain8 & (depth8 < sampled_depth8 - eps)
        return in_front.all(dim=-1)

    @staticmethod
    def _renderVisibilityEvidence(
        camera_list: List[Camera],
        candidate: torch.Tensor,
        R: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """两阶段渲染：先确认真正可见 VALID，再用 8 顶点证明派生 FREE。

        阶段一（确认 VALID）：渲染候选体素并集的边界 mesh，命中的 face ->
        voxel（受每个相机 mask 观测域门控）-> 跨视角并集得到 visible_candidate。

        阶段二（证明 FREE）：仅由 visible_candidate 重建边界 mesh（FREE 的唯一
        遮挡终点只能是真正可见的 VALID 表面），再逐相机渲染该 visible-VALID
        mesh；对每个体素投影其 8 个顶点，做 _freeEvidenceEightVerticesInFront
        的严格证明，跨视角并集得到 free_evidence。

        没有任何真正可见 VALID 时（如稀疏点云某方向无候选阻挡），不存在可信
        遮挡终点，free_evidence 全 False —— 该方向保持 UNKNOWN。

        Returns:
            visible_candidate: (R, R, R) bool，跨视角真正可见的候选体素（VALID）。
            free_evidence:     (R, R, R) bool，拥有 8 顶点 visible-VALID 前方证明
                               的体素（尚未排除候选，最终 FREE 见 _deriveStrictFreeMask）。
        """
        visible = torch.zeros((R, R, R), dtype=torch.bool, device=device)
        free_evidence = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        # --- 阶段一：渲染候选边界 mesh，确认真正可见的 VALID 体素 ---
        cand_vertices, cand_faces, cand_tri2voxel = \
            VolumeMarker._buildVoxelBoundaryMesh(candidate, R)
        if cand_faces.shape[0] == 0:
            return visible, free_evidence

        cand_mesh = VolumeMarker._trimeshFromTensors(cand_vertices, cand_faces)
        cand_tri2voxel = cand_tri2voxel.to(device)

        # 同一 device 上的 float32 顶点张量跨相机复用。
        cand_vert_cache: dict = {}
        for camera in camera_list:
            vtensor = cand_vert_cache.get(camera.device)
            if vtensor is None:
                vtensor = cand_vertices.to(
                    dtype=torch.float32, device=camera.device,
                )
                cand_vert_cache[camera.device] = vtensor

            _, _, rast_out, height, width = \
                VolumeMarker._renderMeshDepthForCamera(cand_mesh, camera, vtensor)
            pixel_obs = VolumeMarker._cameraPixelObservationMask(
                camera, height, width,
            )
            visible = visible | VolumeMarker._visibleVoxelsFromRaster(
                rast_out, cand_tri2voxel, R, device, pixel_obs,
            )

        # 没有任何真正可见 VALID -> 没有可信遮挡终点 -> 不产生 FREE 证据。
        if not bool(visible.any()):
            return visible, free_evidence

        # --- 阶段二：仅由真正可见 VALID 体素重建边界 mesh，作为 FREE 唯一遮挡终点 ---
        valid_vertices, valid_faces, _ = \
            VolumeMarker._buildVoxelBoundaryMesh(visible, R)
        if valid_faces.shape[0] == 0:
            return visible, free_evidence

        valid_mesh = VolumeMarker._trimeshFromTensors(valid_vertices, valid_faces)

        # voxel 8 顶点只依赖 (R, dtype, device)，跨相机复用。
        corners_cache: dict = {}
        valid_vert_cache: dict = {}
        for camera in camera_list:
            vtensor = valid_vert_cache.get(camera.device)
            if vtensor is None:
                vtensor = valid_vertices.to(
                    dtype=torch.float32, device=camera.device,
                )
                valid_vert_cache[camera.device] = vtensor

            rendered_depth, hit_mask, _, height, width = \
                VolumeMarker._renderMeshDepthForCamera(valid_mesh, camera, vtensor)
            pixel_obs = VolumeMarker._cameraPixelObservationMask(
                camera, height, width,
            )

            cache_key = (camera.dtype, camera.device)
            corners_flat = corners_cache.get(cache_key)
            if corners_flat is None:
                corners_flat = VolumeMarker._createVoxelCorners(
                    R, camera.dtype, camera.device,
                ).reshape(-1, 8, 3)  # (R^3, 8, 3)
                corners_cache[cache_key] = corners_flat

            num_voxels = corners_flat.shape[0]
            uv, point_depth = VolumeMarker._projectPointsForCamera(
                camera, corners_flat.reshape(-1, 3),  # (R^3 * 8, 3)
            )
            sampled_depth, surface_hit, in_domain = \
                VolumeMarker._sampleRenderedDepthAtUV(
                    rendered_depth, hit_mask, pixel_obs, uv, height, width,
                )

            cam_free = VolumeMarker._freeEvidenceEightVerticesInFront(
                point_depth.reshape(num_voxels, 8),
                sampled_depth.reshape(num_voxels, 8),
                surface_hit.reshape(num_voxels, 8),
                in_domain.reshape(num_voxels, 8),
            ).reshape(R, R, R).to(device)

            free_evidence = free_evidence | cam_free

        return visible, free_evidence

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
    def _deriveStrictFreeMask(
        candidate: torch.Tensor,
        free_evidence: torch.Tensor,
    ) -> torch.Tensor:
        """严格 FREE = 非候选 且 拥有 8 顶点 visible-VALID 前方证明。

        第一性原理：FREE 只可能是非候选 voxel（候选要么 VALID 要么 UNKNOWN），
        且只能来自 _renderVisibilityEvidence 的 8 顶点严格证明（整个体素被夹在
        相机与真正可见 VALID 表面之间）。除此之外的一切非候选体素 —— 被遮挡、
        视锥外、相机后方、稀疏点云孔洞、mask 外、background 未命中 —— 一律保持
        UNKNOWN，不再有任何 background / 未观测 / center-only 启发式释放。

        Args:
            candidate: (R, R, R) bool 候选占据体素。
            free_evidence: (R, R, R) bool，8 顶点 visible-VALID 前方证明。

        Returns:
            (R, R, R) bool，需要写 FREE_KN 标签的 voxel。
        """
        return (~candidate) & free_evidence

    @staticmethod
    def _classifyFreeMask(
        candidate: torch.Tensor,
        free_any: torch.Tensor,
        observed_any: torch.Tensor,
        unobserved_policy: str = 'unknown',
    ) -> torch.Tensor:
        """[LEGACY] center/background 证据下的 FREE 归类（不再用于严格主流程）。

        严格主流程已改用 _deriveStrictFreeMask（仅 8 顶点 visible-VALID 前方
        证明）。此 helper 保留仅为向后兼容与单元覆盖，新路径不应使用；默认
        unobserved_policy 已从 'free' 改为 'unknown'，与第一性原理一致
        （未观测不是空）。

        Args:
            candidate: (R, R, R) bool 候选占据体素。
            free_any: (R, R, R) bool，至少一个视角给出 FREE 证据。
            observed_any: (R, R, R) bool，至少被一个相机观测到。
            unobserved_policy: 未观测非候选的归类策略：
                'unknown' -> 视为 UNKNOWN（保守，默认）；
                'free'    -> 视为 FREE（旧启发式，仅兼容）。

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
    def _emptyEvidenceLabels(
        R: int,
        empty_policy: str,
        device: Optional[str],
    ) -> torch.Tensor:
        """无任何可信证据（无相机 / 无候选）时的标签场。

        第一性原理：没有候选或没有相机时，不存在任何「证明为空」的可信射线，
        严格语义下整个空间应为 UNKNOWN（未观测不是空）。仅当下游显式要求
        兼容旧「空候选 = 全 FREE 哨兵」行为时，才用 'free_inf'。

        Args:
            empty_policy: 'unknown' -> 全 UNKNOWN(0)（严格默认）；
                          'free_inf' -> 全哨兵 FREE -(3R)（旧兼容行为）。
        """
        if empty_policy == 'unknown':
            return VolumeMarker._emptyLabels(R, VolumeMarker.UNKNOWN, device)
        if empty_policy == 'free_inf':
            return VolumeMarker._emptyLabels(
                R, VolumeMarker.freeLabelInf(R), device,
            )
        raise ValueError(
            '[ERROR][VolumeMarker::markVisible] '
            f"empty_policy must be 'unknown' or 'free_inf', got {empty_policy}"
        )

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
        empty_policy: str = 'unknown',
    ) -> torch.Tensor:
        """基于候选占据 + 两阶段渲染对空间体素打严格可信标签。

        第一性原理：VALID / FREE 只能来自可信观测证据，其余一律 UNKNOWN。

          1. 候选占据（潜在 VALID）：
               - 提供 geometry：按真实类型体素化（三角网格走 SAT 精确表面三角
                 体素化，得到无缝隙的表面壳；点云 / numpy / tensor / 文件走点
                 体素化）。此路径只需相机位姿/内参来渲染遮挡，不要求 depth；
                 显式 geometry 不按相机 mask 裁剪候选几何。
               - geometry 为 None：用所有相机可信观测域内（valid_depth_mask
                 且 mask 内）的 CCM 点并集体素化。此路径要求每个相机已加载
                 depth 与 valid_depth_mask；mask 外的深度点不进入候选。
          2. 边界 mesh：对候选体素并集构造只含外露面的立方体表面 mesh。
          3. 两阶段渲染（_renderVisibilityEvidence）：
               - 阶段一：渲染候选边界 mesh，命中的 face -> voxel（受每个相机
                 mask 观测域门控）-> 跨视角并集得到真正可见的 VALID；
               - 阶段二：仅由真正可见 VALID 重建边界 mesh 再渲染，对每个非候选
                 体素做「8 顶点全部位于 mask 内、命中 visible-VALID 表面、且严格
                 在其前方」的证明，跨视角并集得到 FREE 证据。
             任何方向若没有真正可见的 VALID 作为遮挡终点（如稀疏点云无候选
             阻挡），则该方向不产生 FREE，全部保持 UNKNOWN。
          4. 标签合并：
               - 候选且任一视角可见 -> VALID；
               - 非候选且有 8 顶点 visible-VALID 前方证明 -> FREE（不可改写）；
               - 其余一切（被遮挡候选、视锥外、相机后方、mask 外、background
                 未命中、稀疏孔洞、中心在前方但 8 顶点未全证明）-> UNKNOWN。
          5. FREE_KN 标签（assignFreeLabels）：对所有 FREE voxel 全量计算到最近
             非 FREE voxel（VALID 或 UNKNOWN）的 L1 距离 K，写为 FREE_KN = -K。

        无相机或无候选时按 empty_policy（默认 'unknown' -> 全 UNKNOWN；
        'free_inf' 保留旧的全哨兵 FREE 兼容行为）。

        返回: (R, R, R) 的 int64 tensor，编码 UNKNOWN=0, VALID=1,
        FREE_KN=-K（K = 到最近 VALID/UNKNOWN 的 L1 距离；全 FREE 时为 -(3R)）。
        """
        R = VolumeMarker._requirePositiveResolution(volume_resolution, 'markVisible')

        if len(camera_list) == 0:
            # 没有任何视角，不存在任何可信观测证据 -> 严格默认全 UNKNOWN。
            return VolumeMarker._emptyEvidenceLabels(R, empty_policy, None)

        ref_camera = camera_list[0]
        device = ref_camera.device
        dtype = ref_camera.dtype

        # --- 1. 候选占据体素 ---
        # 文件路径只解析一次，候选构建复用同一份几何。
        resolved_geometry = geometry
        if isinstance(geometry, str):
            resolved_geometry = VolumeMarker._loadGeometryFile(geometry)

        # depth 依赖收口在 _buildCandidateOccupancy：仅 geometry 为 None 的
        # depth 候选路径才要求 depth；显式 geometry 只需相机位姿/内参。
        candidate = VolumeMarker._buildCandidateOccupancy(
            camera_list, resolved_geometry, R, dtype, device,
        )

        # --- 无候选体素：无可信证据 -> 按 empty_policy（严格默认全 UNKNOWN） ---
        if not bool(candidate.any()):
            return VolumeMarker._emptyEvidenceLabels(R, empty_policy, device)

        # --- 2 & 3. 两阶段渲染：真正可见 VALID + 8 顶点 FREE 证明 ---
        visible_candidate, free_evidence = \
            VolumeMarker._renderVisibilityEvidence(
                camera_list, candidate, R, device,
            )

        # --- 4. 严格 FREE = 非候选 且 8 顶点 visible-VALID 前方证明 ---
        free_mask = VolumeMarker._deriveStrictFreeMask(candidate, free_evidence)

        # --- 5. VALID / FREE_KN / UNKNOWN 合并（含全量 L1 距离 -> -K） ---
        return VolumeMarker._mergeLabels(R, visible_candidate, free_mask, device)
