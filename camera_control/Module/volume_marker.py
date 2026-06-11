import os
import torch
import trimesh
import numpy as np
import open3d as o3d

from typing import List, Optional, Tuple, Union

from camera_control.Config.visible import (
    VISIBLE_LABEL_UNKNOWN,
    VISIBLE_LABEL_FREE,
    VISIBLE_LABEL_VALID,
    VISIBLE_LABEL_FREE_1N,
    VISIBLE_LABEL_FREE_2N,
    visibleLabelFreeKN,
)
from camera_control.Module.camera import Camera


class VolumeMarker(object):
    UNKNOWN: int = VISIBLE_LABEL_UNKNOWN
    FREE: int = VISIBLE_LABEL_FREE
    VALID: int = VISIBLE_LABEL_VALID
    FREE_1N: int = VISIBLE_LABEL_FREE_1N
    FREE_2N: int = VISIBLE_LABEL_FREE_2N

    def __init__(self) -> None:
        return

    @staticmethod
    def freeLabelKN(k: int) -> int:
        """FREE_KN 标签编码（K >= 1），转发 Config.visible.visibleLabelFreeKN。"""
        return visibleLabelFreeKN(k)

    # ------------------------------------------------------------------
    # 体素 / 网格基础原子函数
    # ------------------------------------------------------------------
    @staticmethod
    def _requirePositiveResolution(volume_resolution: int, caller: str) -> int:
        """校验体素分辨率为正，返回 int 化后的 R。"""
        if volume_resolution <= 0:
            raise ValueError(
                f'[ERROR][VolumeMarker::{caller}] '
                f'volume_resolution must be positive, got {volume_resolution}'
            )
        return int(volume_resolution)

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

    # 体采样相对体素的过采样倍率：在 K*R 分辨率下采样，确保采样点最大间距
    # 远小于体素边长，从而网格穿过的每个体素都被至少一个采样点覆盖。
    MESH_SAMPLE_OVERSAMPLE_K: int = 3
    # 采样点数上限，避免巨大网格 + 高分辨率时点数爆炸（仍足以覆盖体素）。
    MESH_SAMPLE_MAX_POINTS: int = 8_000_000

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
        """按目标点间距 d = 1/(K*R) 估算面积均匀采样所需点数。

        第一性原理：要让网格穿过的每个体素（边长 1/R）都被采样点覆盖，采样点
        在面上的间距需远小于体素边长。取目标间距 d = 1/(K*R)，单位面积约需
        1/d^2 个点，再乘安全系数 2 抵抗均匀采样的局部稀疏与边角缺口。
        """
        d = 1.0 / (K * R)
        n = int(np.ceil(area / (d * d) * 2.0))
        # 至少给足一个 K*R 网格量级的点；并限制上限。
        n = max(n, (K * R) ** 2)
        return int(min(n, VolumeMarker.MESH_SAMPLE_MAX_POINTS))

    @staticmethod
    def _voxelizeMesh(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """把三角网格按体素分辨率 R 求占用，返回 (R,R,R) bool 占据 mask。

        实现（面向性能的体采样近似）：用 open3d 在网格表面做面积均匀采样，
        采样分辨率取体素分辨率的 K 倍（点间距 ~ 1/(K*R) 远小于体素边长
        1/R），再把稠密采样点体素化。当 K 足够大时，网格穿过的每个体素都至少
        含一个采样点，从而等价于「网格表面真实占用」，但避免了逐三角形精确
        相交测试的开销，对大网格 + 高分辨率显著更快。

        坐标约定与体素网格一致：体素覆盖 [-0.5, 0.5]^3，
        voxel index = floor((point + 0.5) * R)。
        """
        if faces.numel() == 0 or vertices.numel() == 0:
            return torch.zeros((R, R, R), dtype=torch.bool, device=device)

        K = VolumeMarker.MESH_SAMPLE_OVERSAMPLE_K

        vertices_np = vertices.detach().cpu().numpy().astype(np.float64)
        faces_np = faces.detach().cpu().numpy().astype(np.int32)

        area = VolumeMarker._meshSurfaceArea(
            vertices.to(torch.float64), faces,
        )
        if area <= 0.0:
            # 退化（零面积）网格：直接用顶点当点云体素化。
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
            return VolumeMarker._voxelizeMesh(vertices, faces, R, device)

        points = VolumeMarker._geometryToPoints(geometry, dtype, device)
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

            render_result = NVDiffRastRenderer.render(
                mesh=mesh,
                camera=camera,
                render_types=['depth'],
                vertices_tensor=vertices_tensor,
                enable_antialias=False,
                enable_lighting=False,
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
    # L1 距离场 / FREE_KN 分层（高效并行原子函数）
    # ------------------------------------------------------------------
    @staticmethod
    def _dilateMask6(mask: torch.Tensor) -> torch.Tensor:
        """对 (R, R, R) bool mask 做一次 6-邻域膨胀，返回新 mask。

        实现为 6 个方向的切片平移按位或，纯张量运算，无 padding 拷贝，
        CPU / GPU 通用且全并行。
        """
        out = mask.clone()
        out[1:, :, :] |= mask[:-1, :, :]
        out[:-1, :, :] |= mask[1:, :, :]
        out[:, 1:, :] |= mask[:, :-1, :]
        out[:, :-1, :] |= mask[:, 1:, :]
        out[:, :, 1:] |= mask[:, :, :-1]
        out[:, :, :-1] |= mask[:, :, 1:]
        return out

    @staticmethod
    def computeL1DistanceToMask(
        source_mask: torch.Tensor,
        max_distance: int,
    ) -> torch.Tensor:
        """计算每个 voxel 到 source_mask 的 L1（6-邻域逐步）距离，截断于 max_distance。

        第一性原理：L1 距离为 k 的 voxel 集合 = source 做 k 次 6-邻域膨胀的
        波前（新增层）。逐层膨胀即逐层写入距离，每层一次张量平移 + 逻辑或，
        复杂度 O(max_distance * R^3)，无任何逐 voxel Python 循环，K 任意可扩展。

        Args:
            source_mask: (R, R, R) bool，距离源（距离 0）。
            max_distance: 距离上限（>= 0）。

        Returns:
            (R, R, R) int64 距离场：source 处为 0，波前层依次为 1..max_distance，
            超出 max_distance 的 voxel 为 -1（哨兵）。
        """
        if max_distance < 0:
            raise ValueError(
                '[ERROR][VolumeMarker::computeL1DistanceToMask] '
                f'max_distance must be >= 0, got {max_distance}'
            )

        distance = torch.full_like(source_mask, -1, dtype=torch.int64)
        distance[source_mask] = 0

        reached = source_mask
        for k in range(1, max_distance + 1):
            new = VolumeMarker._dilateMask6(reached) & (~reached)
            if not bool(new.any()):
                break
            distance[new] = k
            reached = reached | new

        return distance

    @staticmethod
    def markFreeNeighborLevels(
        labels: torch.Tensor,
        max_k: int = 2,
    ) -> torch.Tensor:
        """把 FREE voxel 按到 VALID 的 L1 距离分层改写为 FREE_KN（K=1..max_k）。

        只改写 labels == FREE 且距离恰为 k 的 voxel 为 freeLabelKN(k)；
        UNKNOWN / VALID 不动，距离 > max_k 的 FREE voxel 保持 FREE。

        Args:
            labels: (R, R, R) int64 标签场（UNKNOWN/FREE/VALID 编码）。
            max_k: 分层上限（>= 0，0 表示不分层，原样返回副本）。

        Returns:
            (R, R, R) int64 新标签场（不修改输入）。
        """
        labels = labels.clone()
        if max_k <= 0:
            return labels

        valid_mask = labels == VolumeMarker.VALID
        if not bool(valid_mask.any()):
            return labels

        distance = VolumeMarker.computeL1DistanceToMask(valid_mask, max_k)

        free_mask = labels == VolumeMarker.FREE
        for k in range(1, max_k + 1):
            level = free_mask & (distance == k)
            labels[level] = VolumeMarker.freeLabelKN(k)

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
        free_neighbor_levels: int = 2,
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
               - 否则用所有 camera.toCCM(use_mask=False) 的并集体素化。
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
          5. FREE_KN 分层（markFreeNeighborLevels）：FREE voxel 中到最近
             VALID 的 L1 距离恰为 k（k=1..free_neighbor_levels）的改写为
             FREE_KN = VALID + k；其余 FREE 保持不变。free_neighbor_levels=0
             可关闭，调成 3/4 即得 FREE_3N/4N。

        返回: (R, R, R) 的 int64 tensor，编码 UNKNOWN=-1, FREE=0, VALID=1,
        FREE_1N=2, FREE_2N=3（FREE_KN = VALID + K）。
        """
        R = VolumeMarker._requirePositiveResolution(volume_resolution, 'markVisible')

        if len(camera_list) == 0:
            # 没有任何视角，整个空间一定没有物体信息。
            return VolumeMarker._emptyLabels(R, VolumeMarker.FREE)

        ref_camera = camera_list[0]
        device = ref_camera.device
        dtype = ref_camera.dtype

        VolumeMarker._requireDepthLoaded(camera_list, 'markVisible')

        # --- 1. 候选占据体素 ---
        if geometry is not None:
            candidate = VolumeMarker._geometryToOccupancy(
                geometry, R, dtype, device,
            )
        else:
            candidate_points = VolumeMarker._collectCandidatePoints(
                camera_list, dtype, device,
            )
            candidate = VolumeMarker._voxelizePoints(candidate_points, R, device)

        # --- 无候选体素：空间无占据，整体为 FREE ---
        if not bool(candidate.any()):
            return VolumeMarker._emptyLabels(R, VolumeMarker.FREE, device)

        labels = VolumeMarker._emptyLabels(R, VolumeMarker.UNKNOWN, device)
        non_candidate = ~candidate

        # --- 2 & 3. 边界 mesh + 单次渲染同源产出 VALID/FREE/observed ---
        vertices, faces, triangle_to_voxel = \
            VolumeMarker._buildVoxelBoundaryMesh(candidate, R)

        visible_candidate, free_any, observed_any = \
            VolumeMarker._renderClassify(
                camera_list, vertices, faces, triangle_to_voxel, R, device,
            )

        # --- 4. 标签合并 ---
        # 非候选 -> FREE，当且仅当任一视角给出 FREE 证据，或从未被任何相机
        # 观测到（视线外，一定无物体信息）。两段 FREE 写入合并为一次。
        free_update = non_candidate & (free_any | (~observed_any))
        labels = torch.where(
            free_update,
            torch.full_like(labels, VolumeMarker.FREE),
            labels,
        )

        # 候选且任一视角可见 -> VALID（最后覆盖，确保不被 FREE 降级）。
        labels = torch.where(
            visible_candidate,
            torch.full_like(labels, VolumeMarker.VALID),
            labels,
        )

        # --- 5. FREE_KN 分层 ---
        labels = VolumeMarker.markFreeNeighborLevels(
            labels, max_k=free_neighbor_levels,
        )

        return labels
