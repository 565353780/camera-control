import torch
import trimesh
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Union

from camera_control.Config.visible import (
    VISIBLE_LABEL_UNKNOWN,
    VISIBLE_LABEL_VALID,
    VISIBLE_COLOR_VALID,
    VISIBLE_COLOR_UNKNOWN,
    VISIBLE_COLOR_FREE_NEAR,
    VISIBLE_COLOR_FREE_FAR,
)


def toPcd(
    points: Union[torch.Tensor, np.ndarray, list],
    color: list=[1, 0, 0],
) -> o3d.geometry.PointCloud:
    if isinstance(points, list):
        points = np.array(points)
    elif isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if points.shape[-1] == 4:
        points = points[..., :3]

    points = points.reshape(-1, 3).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.paint_uniform_color(color)
    return pcd

def create_coordinate_frame(origin=np.array([0, 0, 0]), size=0.2):
    """
    创建坐标轴

    Args:
        origin: 原点位置
        size: 轴的长度

    Returns:
        coord_frame: 坐标轴LineSet
    """
    points = np.array([
        origin,  # 0: 原点
        origin + [size, 0, 0],  # 1: X轴
        origin + [0, size, 0],  # 2: Y轴
        origin + [0, 0, size],  # 3: Z轴
    ])

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector([
        [0, 1],  # X轴（红色）
        [0, 2],  # Y轴（绿色）
        [0, 3],  # Z轴（蓝色）
    ])
    lines.colors = o3d.utility.Vector3dVector([
        [1, 0, 0],  # X轴红色
        [0, 1, 0],  # Y轴绿色
        [0, 0, 1],  # Z轴蓝色
    ])

    return lines

def create_line_set(
    start_pos: Union[torch.Tensor, np.ndarray, list],
    end_pos: Union[torch.Tensor, np.ndarray, list],
    color=[1, 0, 0],
) -> o3d.geometry.LineSet:
    """
    创建线段集合

    Args:
        start_pos: 起点位置，可以是单个点(3或1x3)或多个点(Nx3)
        end_pos: 终点位置，可以是单个点(3或1x3)或多个点(Nx3)
        color: 线段颜色，默认红色

    支持的4种情况：
        1. 单点到单点：一条线段
        2. 单点到多点：从单个起点到多个终点，多条线段
        3. 多点到单点：从多个起点到单个终点，多条线段
        4. 相同数量的起点和终点：逐点连线，N条线段

    Returns:
        line_set: LineSet对象
    """
    # 转换为numpy数组
    def to_numpy(x):
        if isinstance(x, list):
            x = np.asarray(x)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64)

    start_pos = to_numpy(start_pos)
    end_pos = to_numpy(end_pos)

    if start_pos.shape[-1] != 3:
        start_pos = start_pos[..., :3]

    if end_pos.shape[-1] != 3:
        end_pos = end_pos[..., :3]

    start_pos = start_pos.reshape(-1, 3)
    end_pos = end_pos.reshape(-1, 3)

    num_start = start_pos.shape[0]
    num_end = end_pos.shape[0]

    # 判断并处理4种情况
    if num_start == 1 and num_end == 1:
        # 情况1：单点到单点
        points = np.vstack([start_pos, end_pos])
        lines = np.array([[0, 1]])
    elif num_start == 1 and num_end > 1:
        # 情况2：单点到多点
        points = np.vstack([start_pos, end_pos])
        lines = np.array([[0, i+1] for i in range(num_end)])
    elif num_start > 1 and num_end == 1:
        # 情况3：多点到单点
        points = np.vstack([start_pos, end_pos])
        end_idx = num_start  # 终点的索引
        lines = np.array([[i, end_idx] for i in range(num_start)])
    elif num_start == num_end:
        # 情况4：相同数量的起点和终点，逐点连线
        points = np.vstack([start_pos, end_pos])
        lines = np.array([[i, i + num_start] for i in range(num_start)])
    else:
        raise ValueError(
            f"不支持的输入形状组合：start_pos有{num_start}个点，end_pos有{num_end}个点。"
            f"仅支持：1-1, 1-N, N-1, 或 N-N（N相同）"
        )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

def _tetraTemplate(radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成以原点为中心、外接球半径为 radius 的正四面体模板。

    4 个顶点取自正方体的交替角点，天然构成正四面体；面索引顶点顺序使
    外法线朝外，便于着色/法线计算。

    Returns:
        vertices: (4, 3) float64 顶点。
        faces: (4, 3) int64 三角面索引。
    """
    tetra_unit = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetra_unit /= np.sqrt(3.0)  # 归一化到外接球半径 1
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    return tetra_unit * radius, faces


def _lerpColors(
    c0: Union[list, np.ndarray],
    c1: Union[list, np.ndarray],
    t: np.ndarray,
) -> np.ndarray:
    """对 (M,) 的插值参数 t（[0, 1]）做 c0 -> c1 的向量化线性颜色插值。

    Returns:
        (M, 3) float64 颜色数组。
    """
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return c0[None, :] + (c1 - c0)[None, :] * t[:, None]


# FREE_KN 四面体尺寸端点（相对基准半径的缩放）：K=1 最大，K=K_max 最小。
FREE_RADIUS_SCALE_NEAR: float = 0.6
FREE_RADIUS_SCALE_FAR: float = 0.15


def _freeLevelStyle(
    k_values: np.ndarray,
    k_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """为每个 FREE voxel 的层级 K 计算 (radius_scale, color)。

    K=1 -> 蓝色、尺寸 FREE_RADIUS_SCALE_NEAR；K=k_max -> 红色、尺寸
    FREE_RADIUS_SCALE_FAR；中间按 t = (K-1)/(k_max-1) 线性插值。

    Args:
        k_values: (M,) int，每个 FREE voxel 的 K（>= 1）。
        k_max: 标签场中实际出现的最大 K（>= 1）。

    Returns:
        radius_scales: (M,) float64 半径缩放。
        colors: (M, 3) float64 颜色。
    """
    k_values = np.asarray(k_values, dtype=np.float64)
    if k_max <= 1:
        t = np.zeros_like(k_values)
    else:
        t = (k_values - 1.0) / (float(k_max) - 1.0)

    radius_scales = (
        FREE_RADIUS_SCALE_NEAR
        + (FREE_RADIUS_SCALE_FAR - FREE_RADIUS_SCALE_NEAR) * t
    )
    colors = _lerpColors(VISIBLE_COLOR_FREE_NEAR, VISIBLE_COLOR_FREE_FAR, t)
    return radius_scales, colors


def _batchTetraMesh(
    centers: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量构建 M 个四面体的 (vertices, faces, vertex_colors)。

    全向量化广播：verts = centers[:,None,:] + template_unit[None,:,:] *
    radii[:,None,None]，支持逐四面体独立的半径与颜色。

    Args:
        centers: (M, 3) 四面体中心。
        radii: (M,) 每个四面体的外接球半径。
        colors: (M, 3) 每个四面体的 RGB 颜色。

    Returns:
        vertices: (M*4, 3) float64。
        faces: (M*4, 3) int64（顶点索引从 0 起，调用方负责整体偏移）。
        vertex_colors: (M*4, 3) float64。
    """
    template_unit, template_faces = _tetraTemplate(1.0)
    num_template_vertices = template_unit.shape[0]
    M = centers.shape[0]

    verts = (
        centers[:, None, :]
        + template_unit[None, :, :] * radii[:, None, None]
    ).reshape(-1, 3)

    face_offsets = np.arange(M, dtype=np.int64) * num_template_vertices
    faces = (
        template_faces[None, :, :] + face_offsets[:, None, None]
    ).reshape(-1, 3)

    vertex_colors = np.repeat(colors, num_template_vertices, axis=0)
    return verts, faces, vertex_colors


def toVisibleVolumeMesh(
    labels: Union[torch.Tensor, np.ndarray],
    tetra_radius_ratio: float = 0.25,
    max_free_k: Optional[int] = None,
    show_valid: bool = True,
    show_unknown: bool = True,
    show_free: bool = True,
    mesh_type: str = 'open3d',
) -> Union[o3d.geometry.TriangleMesh, trimesh.Trimesh]:
    """
    将 VolumeMarker.markVisible 输出的 (R, R, R) 标签可视化为单个 TriangleMesh。

    可视化方式：在每个待绘 voxel 的 center 放一个四面体（4 顶点、4 三角面），
    相比球面可把导出数据量减小一个数量级以上。四面体外接半径以
    tetra_radius_ratio * (voxel 边长) 为基准，按标签缩放：
    - VALID（1）   -> 绿色，最大尺寸（1.0 倍基准）；
    - UNKNOWN（0） -> 灰色，中等尺寸（0.7 倍）；
    - FREE_KN（-K）-> 颜色按 t = (K-1)/(K_max-1) 从蓝色渐变到红色，
      尺寸从 0.6 倍线性减小到 0.15 倍（K_max 为标签场中实际最大 K，
      含全 FREE 时的 -(3R) 哨兵层）。

    Args:
        labels: (R, R, R) int 张量，编码 UNKNOWN=0, VALID=1, FREE_KN=-K。
        tetra_radius_ratio: 基准四面体外接球半径相对 voxel 边长的比例。
        max_free_k: 只绘制 K <= max_free_k 的 FREE 层（None 表示全画）；
            用于避免全空间 R^3 个四面体导出过大。
        show_valid / show_unknown / show_free: 分别控制是否绘制 VALID /
            UNKNOWN / FREE_KN 三类 voxel（默认全画，保持向后兼容）。调试
            单类中间结果（如只看候选或只看 FREE）时可关闭其余类，避免被
            外围 UNKNOWN 灰色四面体淹没。

        mesh_type: 返回类型，``'open3d'``（默认）返回
            ``open3d.geometry.TriangleMesh``；``'trimesh'`` 经 ``toTrimesh``
            完整继承 v/f/color 后返回 ``trimesh.Trimesh``。

    Returns:
        合并后的单个网格（类型由 ``mesh_type`` 决定）。
    """
    assert mesh_type in ('open3d', 'trimesh'), (
        f'[ERROR][render::toVisibleVolumeMesh] unsupported mesh_type={mesh_type}, '
        "expected one of ['open3d', 'trimesh']"
    )

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)

    if labels.ndim != 3 or labels.shape[0] != labels.shape[1] or labels.shape[0] != labels.shape[2]:
        raise ValueError(
            '[ERROR][render::toVisibleVolumeMesh] '
            f'labels must be a cubic (R, R, R) array, got shape {labels.shape}'
        )

    R = int(labels.shape[0])
    voxel_size = 1.0 / R
    base_radius = tetra_radius_ratio * voxel_size

    # voxel (i, j, k) 中心：-0.5 + (idx + 0.5) / R
    idx = (np.arange(R, dtype=np.float64) + 0.5) / R - 0.5

    def _centersOf(mask: np.ndarray) -> np.ndarray:
        ii, jj, kk = np.where(mask)
        return np.stack([idx[ii], idx[jj], idx[kk]], axis=-1)

    all_centers = []
    all_radii = []
    all_colors = []

    # --- VALID / UNKNOWN：固定颜色与尺寸（按开关过滤）---
    for label_value, color, radius_scale, enabled in (
        (VISIBLE_LABEL_VALID, VISIBLE_COLOR_VALID, 1.0, show_valid),
        (VISIBLE_LABEL_UNKNOWN, VISIBLE_COLOR_UNKNOWN, 0.7, show_unknown),
    ):
        if not enabled:
            continue
        centers = _centersOf(labels == label_value)
        M = centers.shape[0]
        if M == 0:
            continue
        all_centers.append(centers)
        all_radii.append(np.full((M,), radius_scale * base_radius))
        all_colors.append(np.tile(np.asarray(color, dtype=np.float64), (M, 1)))

    # --- FREE_KN：按 K 渐变颜色与尺寸 ---
    free_mask = labels < 0
    if max_free_k is not None:
        free_mask &= labels >= -int(max_free_k)
    if not show_free:
        free_mask = np.zeros_like(free_mask)

    if bool(free_mask.any()):
        k_values = -labels[free_mask].astype(np.int64)  # (M,) K >= 1
        # K_max 取全标签场中实际出现的最大 K（不受 max_free_k 过滤影响，
        # 保证同一标签场在不同过滤参数下颜色一致）。
        k_max = int(-labels[labels < 0].min())

        radius_scales, colors = _freeLevelStyle(k_values, k_max)

        all_centers.append(_centersOf(free_mask))
        all_radii.append(radius_scales * base_radius)
        all_colors.append(colors)

    mesh = o3d.geometry.TriangleMesh()
    if len(all_centers) == 0:
        if mesh_type == 'trimesh':
            from camera_control.Method.mesh import toTrimesh
            return toTrimesh(mesh)
        return mesh

    vertices, faces, vertex_colors = _batchTetraMesh(
        np.concatenate(all_centers, axis=0),
        np.concatenate(all_radii, axis=0),
        np.concatenate(all_colors, axis=0),
    )

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()

    if mesh_type == 'trimesh':
        from camera_control.Method.mesh import toTrimesh
        return toTrimesh(mesh)

    return mesh
