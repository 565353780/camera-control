import torch
import numpy as np
import open3d as o3d
from typing import Union

from typing import Tuple

from camera_control.Config.visible import (
    VISIBLE_LABEL_UNKNOWN,
    VISIBLE_LABEL_VALID,
    VISIBLE_LABEL_FREE_1N,
    VISIBLE_LABEL_FREE_2N,
    VISIBLE_COLOR_VALID,
    VISIBLE_COLOR_UNKNOWN,
    VISIBLE_COLOR_FREE_1N,
    VISIBLE_COLOR_FREE_2N,
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


def toVisibleVolumeMesh(
    labels: Union[torch.Tensor, np.ndarray],
    tetra_radius_ratio: float = 0.25,
) -> o3d.geometry.TriangleMesh:
    """
    将 VolumeMarker.markVisible 输出的 (R, R, R) 标签可视化为单个 TriangleMesh。

    可视化方式：在每个非 FREE voxel 的 center 放一个四面体（4 顶点、4 三角面），
    相比球面可把导出数据量减小一个数量级以上。四面体外接半径以
    tetra_radius_ratio * (voxel 边长) 为基准，按标签分级缩放，便于清晰分辨：
    - Valid   -> 绿色，最大尺寸（1.0 倍基准）
    - Unknown -> 灰色，中等尺寸（0.7 倍）
    - Free_1N -> 浅蓝，较小尺寸（0.45 倍）
    - Free_2N -> 浅橙，最小尺寸（0.28 倍）
    - Free    -> 不绘制

    Args:
        labels: (R, R, R) int 张量，编码 UNKNOWN=-1, FREE=0, VALID=1,
            FREE_1N=2, FREE_2N=3。
        tetra_radius_ratio: 基准四面体外接球半径相对 voxel 边长的比例。

    Returns:
        合并后的单个 o3d.geometry.TriangleMesh。
    """
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

    # voxel (i, j, k) 中心：-0.5 + (idx + 0.5) / R
    idx = (np.arange(R, dtype=np.float64) + 0.5) / R - 0.5

    # 配置驱动：(label, color, radius_ratio_scale)，FREE 不绘制。
    label_render_configs = [
        (VISIBLE_LABEL_VALID, VISIBLE_COLOR_VALID, 1.0),
        (VISIBLE_LABEL_UNKNOWN, VISIBLE_COLOR_UNKNOWN, 0.7),
        (VISIBLE_LABEL_FREE_1N, VISIBLE_COLOR_FREE_1N, 0.45),
        (VISIBLE_LABEL_FREE_2N, VISIBLE_COLOR_FREE_2N, 0.28),
    ]

    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for label_value, color, radius_scale in label_render_configs:
        ii, jj, kk = np.where(labels == label_value)
        if ii.size == 0:
            continue

        radius = radius_scale * tetra_radius_ratio * voxel_size
        template_vertices, template_faces = _tetraTemplate(radius)
        num_template_vertices = template_vertices.shape[0]

        centers = np.stack([idx[ii], idx[jj], idx[kk]], axis=-1)  # (M, 3)
        M = centers.shape[0]

        # (M, V, 3): 每个 center 平移一份模板顶点。
        verts = template_vertices[None, :, :] + centers[:, None, :]
        verts = verts.reshape(-1, 3)

        # (M, F, 3): 每份面索引加上对应四面体的顶点偏移。
        face_offsets = (
            vertex_offset + np.arange(M, dtype=np.int64) * num_template_vertices
        )
        faces = template_faces[None, :, :] + face_offsets[:, None, None]
        faces = faces.reshape(-1, 3)

        colors = np.tile(
            np.asarray(color, dtype=np.float64), (verts.shape[0], 1),
        )

        all_vertices.append(verts)
        all_faces.append(faces)
        all_colors.append(colors)
        vertex_offset += verts.shape[0]

    mesh = o3d.geometry.TriangleMesh()
    if len(all_vertices) == 0:
        return mesh

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh
