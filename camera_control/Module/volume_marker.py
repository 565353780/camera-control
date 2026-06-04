import os
import torch
import trimesh
import numpy as np

from typing import List, Optional, Tuple, Union

from camera_control.Config.visible import (
    VISIBLE_LABEL_UNKNOWN,
    VISIBLE_LABEL_FREE,
    VISIBLE_LABEL_VALID,
)
from camera_control.Module.camera import Camera


class VolumeMarker(object):
    UNKNOWN: int = VISIBLE_LABEL_UNKNOWN
    FREE: int = VISIBLE_LABEL_FREE
    VALID: int = VISIBLE_LABEL_VALID

    def __init__(self) -> None:
        return

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
        finite = torch.isfinite(points).all(dim=-1)
        inside = (
            finite
            & (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5)
            & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
            & (points[:, 2] >= -0.5) & (points[:, 2] <= 0.5)
        )
        if not bool(inside.any()):
            return occupancy

        idx = torch.floor((points[inside] + 0.5) * R).long().clamp(0, R - 1)
        flat_idx = (idx[:, 0] * R + idx[:, 1]) * R + idx[:, 2]

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
        all_points: List[torch.Tensor] = []
        for camera in camera_list:
            if camera.depth is None or camera.valid_depth_mask is None:
                raise ValueError(
                    '[ERROR][VolumeMarker::_collectCandidatePoints] '
                    'camera missing depth or valid_depth_mask; call loadDepth first.'
                )
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
        M = occ_idx.shape[0]

        empty = (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.int64, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )
        if M == 0:
            return empty

        voxel_size = 1.0 / R

        # 6 个面方向：邻居偏移 + 该面 4 个角点（单位立方体局部坐标 0/1）。
        # 角点顺序保证三角化后外法线朝向 candidate 外侧（这里只用于遮挡，
        # nvdiffrast 不做背面剔除，朝向不影响命中判定，仅保持几何闭合）。
        face_defs = [
            # +x 面
            ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            # -x 面
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            # +y 面
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            # -y 面
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            # +z 面
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            # -z 面
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ]

        base = (occ_idx.to(torch.float32) - 0.5 * R) * voxel_size  # voxel 最小角世界坐标
        i = occ_idx[:, 0]
        j = occ_idx[:, 1]
        k = occ_idx[:, 2]
        voxel_flat = (i * R + j) * R + k  # (M,)

        all_quad_verts: List[torch.Tensor] = []
        all_quad_voxel: List[torch.Tensor] = []

        for (dx, dy, dz), corners in face_defs:
            ni = i + dx
            nj = j + dy
            nk = k + dz

            out_of_grid = (
                (ni < 0) | (ni >= R)
                | (nj < 0) | (nj >= R)
                | (nk < 0) | (nk >= R)
            )

            ni_c = ni.clamp(0, R - 1)
            nj_c = nj.clamp(0, R - 1)
            nk_c = nk.clamp(0, R - 1)
            neighbor_occupied = candidate[ni_c, nj_c, nk_c] & (~out_of_grid)

            exposed = ~neighbor_occupied  # (M,)
            if not bool(exposed.any()):
                continue

            sel_base = base[exposed]  # (E, 3)
            sel_voxel = voxel_flat[exposed]  # (E,)

            corner_offsets = torch.tensor(
                corners, dtype=torch.float32, device=device,
            ) * voxel_size  # (4, 3)

            # (E, 4, 3)
            quad = sel_base.unsqueeze(1) + corner_offsets.unsqueeze(0)
            all_quad_verts.append(quad)
            all_quad_voxel.append(sel_voxel)

        if len(all_quad_verts) == 0:
            return empty

        quad_verts = torch.cat(all_quad_verts, dim=0)  # (Q, 4, 3)
        quad_voxel = torch.cat(all_quad_voxel, dim=0)  # (Q,)
        Q = quad_verts.shape[0]

        vertices = quad_verts.reshape(-1, 3)  # (Q*4, 3)

        # 每个 quad -> 2 个三角，顶点索引基于 quad 在 vertices 中的偏移。
        quad_offset = torch.arange(Q, device=device, dtype=torch.int64) * 4
        tri0 = torch.stack(
            [quad_offset + 0, quad_offset + 1, quad_offset + 2], dim=-1,
        )
        tri1 = torch.stack(
            [quad_offset + 0, quad_offset + 2, quad_offset + 3], dim=-1,
        )
        faces = torch.stack([tri0, tri1], dim=1).reshape(-1, 3)  # (Q*2, 3)

        triangle_to_voxel = quad_voxel.repeat_interleave(2)  # (Q*2,)

        return vertices, faces, triangle_to_voxel

    # ------------------------------------------------------------------
    # 用 nvdiffrast 确认真正可见的候选体素
    # ------------------------------------------------------------------
    @staticmethod
    def _renderVisibleVoxels(
        camera_list: List[Camera],
        vertices: torch.Tensor,
        faces: torch.Tensor,
        triangle_to_voxel: torch.Tensor,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """渲染候选体素并集边界 mesh，返回各相机可见的候选体素并集 (R, R, R) bool。

        对每个相机渲染稳定遮挡关系的 depth（关闭抗锯齿），从 rasterize 输出
        中读取被击中的 face id，映射回 voxel flat index 标记为可见。
        """
        visible = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        if faces.shape[0] == 0:
            return visible

        # 延迟导入：nvdiffrast 仅在真正渲染时才需要，使纯 CPU 逻辑
        # （体素化 / 边界 mesh / FREE 判定）在无 nvdiffrast 环境也能使用。
        from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

        # nvdiffrast 不读取 visual，传入纯几何 trimesh 即可。
        mesh = trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy().astype(np.float64),
            faces=faces.detach().cpu().numpy().astype(np.int64),
            process=False,
        )

        visible_flat = visible.reshape(-1)
        triangle_to_voxel = triangle_to_voxel.to(device)

        for camera in camera_list:
            render_result = NVDiffRastRenderer.render(
                mesh=mesh,
                camera=camera,
                render_types=['depth'],
                vertices_tensor=vertices.to(
                    dtype=torch.float32, device=camera.device,
                ),
                enable_antialias=False,
                enable_lighting=False,
            )

            # rasterize_output[..., 3] 是 1-based triangle id，0 为背景。
            rast_out = render_result['rasterize_output']  # (H, W, 4)
            tri_id = rast_out[..., 3].reshape(-1).long()  # (H*W,)
            hit = tri_id > 0
            if not bool(hit.any()):
                continue

            local_face_id = (tri_id[hit] - 1).to(device)
            hit_voxel_flat = triangle_to_voxel[local_face_id].unique()
            visible_flat[hit_voxel_flat] = True

        return visible_flat.reshape(R, R, R)

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
        xs, ys, zs = torch.meshgrid(coords, coords, coords, indexing='ij')
        vertices = torch.stack([xs, ys, zs], dim=-1)
        return vertices

    @staticmethod
    def sampleVertexDepth(
        camera: Camera,
        uv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        复刻 DepthChannel.queryUVPoints 的最近邻 depth 采样逻辑，但跳过 3D 反投影，
        以最高效的方式得到每个顶点采样到的 render depth 及其状态。

        uv: (N, 2) 归一化 [0, 1]，与 camera UV 约定一致。
        返回:
            sampled_depth:   (N,) 采样到的 render depth（背景/越界处为 0）。
            depth_positive:  (N,) bool，UV 在画面内且采到正深度表面（depth 有效）。
            depth_background: (N,) bool，UV 在画面内但该像素无物体（depth≈0，背景射线）。
        """
        in_frame = (
            torch.isfinite(uv).all(dim=-1)
            & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        )

        uv_safe = torch.where(
            in_frame.unsqueeze(-1), uv, torch.zeros_like(uv),
        )

        u_nearest = (uv_safe[:, 0] * camera.depth_width).long().clamp(
            0, camera.depth_width - 1,
        )
        v_nearest = ((1.0 - uv_safe[:, 1]) * camera.depth_height).long().clamp(
            0, camera.depth_height - 1,
        )

        raw_depth = camera.depth[v_nearest, u_nearest]
        depth_valid = camera.valid_depth_mask[v_nearest, u_nearest]

        is_background = in_frame & (~depth_valid) & (raw_depth <= 1e-5)

        depth_positive = in_frame & depth_valid
        sampled_depth = torch.where(
            depth_positive, raw_depth, torch.zeros_like(raw_depth),
        )
        return sampled_depth, depth_positive, is_background

    @staticmethod
    def _stackCorners(vertex_field: torch.Tensor) -> torch.Tensor:
        """
        将 (R+1, R+1, R+1) 的顶点场转换为 (R, R, R, 8) 的 voxel 8 顶点堆叠。
        顶点 (i, j, k) 是 voxel (i, j, k) 的一个角。
        """
        corners = [
            vertex_field[di:di + (vertex_field.shape[0] - 1),
                         dj:dj + (vertex_field.shape[1] - 1),
                         dk:dk + (vertex_field.shape[2] - 1)]
            for di in (0, 1)
            for dj in (0, 1)
            for dk in (0, 1)
        ]
        return torch.stack(corners, dim=-1)

    @staticmethod
    def _computeFreeEvidence(
        camera_list: List[Camera],
        R: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于角点 depth 采样，统计每个 voxel 的 FREE 与 observed 证据。

        FREE 证据：某相机下 voxel 的 8 个角点都位于正深度表面前方，或都落在
        背景射线（depth≈0）上，则该 voxel 在该相机下属于自由空间。

        Returns:
            free_any:    (R, R, R) bool，至少一个相机给出 FREE 证据。
            observed_any:(R, R, R) bool，至少一个相机的 8 角全部给出可解释观测。
        """
        free_any = torch.zeros((R, R, R), dtype=torch.bool, device=device)
        observed_any = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        grid_shape = (R + 1, R + 1, R + 1)

        for camera in camera_list:
            vertices = VolumeMarker.createGridVertices(
                R, camera.dtype, camera.device,
            )
            flat_vertices = vertices.reshape(-1, 3)

            uv = camera.project_points_to_uv(flat_vertices)  # (N, 2)

            vertices_homo = torch.cat(
                [flat_vertices, torch.ones_like(flat_vertices[..., :1])], dim=-1,
            )
            point_depth = -(vertices_homo @ camera.world2camera.T)[..., 2]  # (N,)

            sampled_depth, depth_positive, depth_background = \
                VolumeMarker.sampleVertexDepth(camera, uv)

            eps = torch.clamp(point_depth.abs() * 1e-4, min=1e-6)
            diff = point_depth - sampled_depth

            observed = depth_positive | depth_background
            front_surface = depth_positive & (diff < -eps)
            front = front_surface | depth_background

            observed_v = observed.reshape(grid_shape).to(device)
            front_v = front.reshape(grid_shape).to(device)

            corners_observed = VolumeMarker._stackCorners(observed_v)
            corners_front = VolumeMarker._stackCorners(front_v)

            voxel_observed = corners_observed.all(dim=-1)  # (R, R, R)
            all_front = corners_front.all(dim=-1)

            voxel_free = voxel_observed & all_front

            free_any = free_any | voxel_free
            observed_any = observed_any | voxel_observed

        return free_any, observed_any

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    @staticmethod
    def markVisible(
        camera_list: List[Camera],
        volume_resolution: int,
        points: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
    ) -> torch.Tensor:
        """基于候选占据 + 立方体 mesh 渲染对空间体素打标签。

        算法（第一性原理重构）：
          1. 候选占据：
               - 若提供 points（数组/张量/文件路径），优先用 points 体素化得到
                 候选 VALID voxel；
               - 否则用所有 camera.toCCM(use_mask=False) 的并集体素化。
          2. 边界 mesh：对候选体素并集构造只含外露面的立方体表面 mesh，并记录
             每个三角对应的 voxel。
          3. 可见性渲染：从每个相机用 nvdiffrast 渲染该 mesh 的 depth（关闭抗
             锯齿），读取被击中的 face id，映射回 voxel —— 这些是真正能被某个
             相机看到的候选体素，标为 VALID。
          4. 被所有视角的渲染 depth 完全遮挡的候选体素（误标注 VALID），保持
             UNKNOWN，后续不再改写。
          5. FREE / UNKNOWN：仅对非候选体素，沿用基于角点 depth 的 FREE 判定：
             8 角全在表面前方或全为背景射线 -> FREE；其余既非 FREE 又非候选的
             voxel 保持 UNKNOWN。从未被任何相机观测到的非候选 voxel -> FREE。

        返回: (R, R, R) 的 int64 tensor，编码 UNKNOWN=-1, FREE=0, VALID=1。
        """
        if volume_resolution <= 0:
            raise ValueError(
                '[ERROR][VolumeMarker::markVisible] '
                f'volume_resolution must be positive, got {volume_resolution}'
            )

        R = int(volume_resolution)

        if len(camera_list) == 0:
            # 没有任何视角，整个空间一定没有物体信息。
            return torch.full((R, R, R), VolumeMarker.FREE, dtype=torch.int64)

        ref_camera = camera_list[0]
        device = ref_camera.device
        dtype = ref_camera.dtype

        for camera in camera_list:
            if camera.depth is None or camera.valid_depth_mask is None:
                raise ValueError(
                    '[ERROR][VolumeMarker::markVisible] '
                    'camera missing depth or valid_depth_mask; call loadDepth first.'
                )

        # --- 1. 候选占据体素 ---
        if points is not None:
            candidate_points = VolumeMarker._normalizePoints(points, dtype, device)
        else:
            candidate_points = VolumeMarker._collectCandidatePoints(
                camera_list, dtype, device,
            )

        candidate = VolumeMarker._voxelizePoints(candidate_points, R, device)

        labels = torch.full(
            (R, R, R), VolumeMarker.UNKNOWN, dtype=torch.int64, device=device,
        )

        # --- 2 & 3. 边界 mesh + 渲染可见候选体素 ---
        if bool(candidate.any()):
            vertices, faces, triangle_to_voxel = \
                VolumeMarker._buildVoxelBoundaryMesh(candidate, R)

            visible_candidate = VolumeMarker._renderVisibleVoxels(
                camera_list, vertices, faces, triangle_to_voxel, R, device,
            )

            # 可见候选 -> VALID；其余候选（被完全遮挡）保持 UNKNOWN。
            labels = torch.where(
                visible_candidate,
                torch.full_like(labels, VolumeMarker.VALID),
                labels,
            )
        else:
            visible_candidate = torch.zeros(
                (R, R, R), dtype=torch.bool, device=device,
            )

        # --- 4 & 5. 仅对非候选体素判定 FREE / UNKNOWN ---
        free_any, observed_any = VolumeMarker._computeFreeEvidence(
            camera_list, R, device,
        )

        non_candidate = ~candidate

        # 非候选 + 有 FREE 证据 -> FREE。
        free_update = non_candidate & free_any
        labels = torch.where(
            free_update,
            torch.full_like(labels, VolumeMarker.FREE),
            labels,
        )

        # 非候选 + 从未被任何相机观测到 -> FREE（一定没有物体信息）。
        never_seen = non_candidate & (~observed_any)
        labels = torch.where(
            never_seen,
            torch.full_like(labels, VolumeMarker.FREE),
            labels,
        )

        return labels
