import torch

from typing import List, Tuple

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

        与旧实现的关键差异：把采样像素拆成三类状态而非单一 valid，
        以便上层单独处理 depth=0 的背景射线（应支持 FREE 而非 UNKNOWN）。

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

        # NaN/越界的 UV 会被 clamp 到边缘像素，因此先填一个安全值再用 in_frame 屏蔽。
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

        # depth=0（或近 0）代表该射线打到背景，没有可见表面。这里区别于
        # NaN / 过大等真正无效的 depth：只有恰为背景才提供 FREE 证据。
        is_background = in_frame & (~depth_valid) & (raw_depth <= 1e-5)

        depth_positive = in_frame & depth_valid
        sampled_depth = torch.where(
            depth_positive, raw_depth, torch.zeros_like(raw_depth),
        )
        return sampled_depth, depth_positive, is_background

    @staticmethod
    def splatCCMToVoxels(
        camera: Camera,
        R: int,
        device: str,
    ) -> torch.Tensor:
        """
        把当前 depth 反投影出的真实表面点（CCM）体素化为 (R, R, R) 的 bool mask。

        这是对角点采样的补充证据：即使某个 voxel 的 8 个角点都落在表面后方，
        或都投影到 depth=0 的背景区域，只要该 voxel 内确实有 depth 投影出的 3D 点，
        就应被标记为 VALID（表面穿过 voxel 内部）。

        坐标约定与体素网格一致：体素覆盖 [-0.5, 0.5]^3，
        voxel index = floor((point + 0.5) * R)，(x, y, z) 对应 (i, j, k)。
        """
        surface_voxel = torch.zeros(
            (R, R, R), dtype=torch.bool, device=device,
        )

        # use_mask=False：只依赖 valid_depth_mask（正深度），不引入额外 image mask，
        # 避免漏掉确实存在的表面点。
        ccm = camera.toCCM(use_mask=False)  # (H, W, 3)
        valid = camera.valid_depth_mask  # (H, W)

        points = ccm[valid]  # (P, 3)
        if points.numel() == 0:
            return surface_voxel

        idx = torch.floor((points + 0.5) * R).long()  # (P, 3)

        inside = (
            (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5)
            & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
            & (points[:, 2] >= -0.5) & (points[:, 2] <= 0.5)
        )
        if not bool(inside.any()):
            return surface_voxel

        idx = idx[inside].clamp(0, R - 1)  # (Q, 3)

        flat_idx = (idx[:, 0] * R + idx[:, 1]) * R + idx[:, 2]  # (Q,)
        surface_flat = surface_voxel.reshape(-1)
        surface_flat[flat_idx.unique()] = True
        return surface_flat.reshape(R, R, R)

    @staticmethod
    def markVisible(
        camera_list: List[Camera],
        volume_resolution: int,
    ) -> torch.Tensor:
        """
        基于多相机 depth 对空间体素打标签。

        算法（每个 camera 同时使用两类证据）：
          1. 在 [-0.5, 0.5] 创建 R^3 个 voxel 的 (R+1)^3 顶点网格。
          2. 投影所有顶点到 UV，取相机前方距离 point_depth = -Z，
             并按最近邻规则采样像素，区分三类状态：
               - positive: 采到正深度表面，比较 point_depth 与表面 depth；
               - background: 像素 depth=0，射线打到背景，无可见物体；
               - 其它（NaN/越界/无效）: 不提供证据。
          3. 角点证据 reduce 到 voxel：
               - 8 角全为 front 或 background -> FREE；
               - 跨越表面（既有正深度 front 又有 behind）或角点落在表面带 -> VALID。
                 注意：背景 free 不参与穿越判定，避免 depth=0/depth>0 边界处
                 VALID 沿视线向后误传。
          4. 表面点证据：把当前 depth 反投影的 CCM 点直接体素化为 VALID，
             修复表面穿过 voxel 内部但 8 角都未跨面的漏检。
          5. 多视角合并：VALID 覆盖一切，FREE 不覆盖已有 VALID。
          6. 不落在任意视角内的 voxel -> FREE。

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

        labels = torch.full(
            (R, R, R), VolumeMarker.UNKNOWN, dtype=torch.int64, device=device,
        )
        seen_any = torch.zeros((R, R, R), dtype=torch.bool, device=device)

        grid_shape = (R + 1, R + 1, R + 1)

        for camera in camera_list:
            if camera.depth is None or camera.valid_depth_mask is None:
                raise ValueError(
                    '[ERROR][VolumeMarker::markVisible] '
                    'camera missing depth or valid_depth_mask; call loadDepth first.'
                )

            vertices = VolumeMarker.createGridVertices(
                R, camera.dtype, camera.device,
            )  # (R+1, R+1, R+1, 3)
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

            # 顶点级证据：
            #   - observed: 该角点给出了可解释的观测（正深度表面 或 背景射线）。
            #   - front: 角点位于自由空间——正深度表面前方，或该射线为背景（无物体）。
            #   - front_surface: 仅“正深度表面前方”，不含背景。用于表面穿越判定。
            #   - behind / surface: 仅对正深度像素有意义。
            observed = depth_positive | depth_background
            front_surface = depth_positive & (diff < -eps)
            front = front_surface | depth_background
            behind = depth_positive & (diff > eps)
            surface = depth_positive & (diff.abs() <= eps)

            observed_v = observed.reshape(grid_shape)
            front_v = front.reshape(grid_shape)
            front_surface_v = front_surface.reshape(grid_shape)
            behind_v = behind.reshape(grid_shape)
            surface_v = surface.reshape(grid_shape)

            # 把顶点级布尔量 reduce 到 voxel 级：每个 voxel 取其 8 个顶点切片。
            corners_observed = VolumeMarker._stackCorners(observed_v)  # (R, R, R, 8)
            corners_front = VolumeMarker._stackCorners(front_v)
            corners_front_surface = VolumeMarker._stackCorners(front_surface_v)
            corners_behind = VolumeMarker._stackCorners(behind_v)
            corners_surface = VolumeMarker._stackCorners(surface_v)

            # 8 个顶点均给出可解释观测，才参与前/表面/后判断。
            voxel_observed = corners_observed.all(dim=-1)  # (R, R, R)
            all_front = corners_front.all(dim=-1)
            has_front_surface = corners_front_surface.any(dim=-1)
            has_behind = corners_behind.any(dim=-1)
            has_surface = corners_surface.any(dim=-1)

            voxel_free = voxel_observed & all_front
            # 表面穿越只看“正深度表面前方 + 表面后方”的角点组合。
            # 背景角点提供的 free 不参与穿越判定，否则会在 depth=0 与 depth>0
            # 的边界处把 VALID 沿视线方向向后误传。
            voxel_valid = voxel_observed & (
                (has_front_surface & has_behind) | has_surface
            )

            # 表面点证据：CCM 反投影点直接落入的 voxel 视为 VALID。
            surface_voxel = VolumeMarker.splatCCMToVoxels(camera, R, device)
            voxel_valid = voxel_valid | surface_voxel

            seen_any = seen_any | voxel_observed | surface_voxel

            # FREE 不覆盖已有 VALID；VALID 覆盖一切。
            # 先处理 FREE（含本视角新判出的 VALID 不应被 FREE 覆盖），再统一覆盖 VALID。
            free_update = voxel_free & (~voxel_valid) & (labels != VolumeMarker.VALID)
            labels = torch.where(
                free_update,
                torch.full_like(labels, VolumeMarker.FREE),
                labels,
            )
            labels = torch.where(
                voxel_valid,
                torch.full_like(labels, VolumeMarker.VALID),
                labels,
            )

        # 不落在任意视角内的 voxel 一定没有物体信息。
        labels = torch.where(
            seen_any,
            labels,
            torch.full_like(labels, VolumeMarker.FREE),
        )

        return labels

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
