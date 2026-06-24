import torch
import random
import trimesh
import numpy as np
from typing import List, Optional

from camera_control.Module.camera import Camera


def sampleFibonacciPolars(num_polars: int) -> np.ndarray:
    """
    使用Fibonacci球面采样生成均匀分布的极角 (phi, theta)。

    Args:
        num_polars: 极角对的数量

    Returns:
        polars: shape (num_polars, 2) 的数组，每行为 (phi, theta)；
               phi 为极角/天顶角 [0, pi]，theta 为方位角。
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(num_polars, dtype=np.float64)
    if num_polars > 1:
        y = 1.0 - (i / (num_polars - 1)) * 2.0
    else:
        y = np.array([0.0])
    phi = np.arccos(np.clip(y, -1.0, 1.0))
    theta = golden_angle * i
    return np.stack([phi, theta], axis=1)

def sampleFibonacciRotations(num_rotations: int) -> np.ndarray:
    """
    基于 sampleFibonacciPolars 并行生成 num_rotations 个 3x3 旋转矩阵。
    R = Rz(phi) @ Ry(theta)，其中 phi 为极角/天顶角 [0, pi]（与 z 轴夹角），
    theta 为方位角（由 Fibonacci 黄金角递增）。

    注意：此处 phi/theta 的含义与 sampleFibonacciPolars 一致：
    polars[:, 0] = phi（天顶角），polars[:, 1] = theta（方位角）。

    Args:
        num_rotations: 旋转数量

    Returns:
        R: shape (num_rotations, 3, 3) 的旋转矩阵数组
    """
    polars = sampleFibonacciPolars(num_rotations)  # (n, 2), (phi, theta)
    phi = polars[:, 0]
    theta = polars[:, 1]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    n = num_rotations
    Rz = np.zeros((n, 3, 3), dtype=np.float64)
    Rz[:, 0, 0] = cphi
    Rz[:, 0, 1] = -sphi
    Rz[:, 1, 0] = sphi
    Rz[:, 1, 1] = cphi
    Rz[:, 2, 2] = 1.0
    Ry = np.zeros((n, 3, 3), dtype=np.float64)
    Ry[:, 0, 0] = cth
    Ry[:, 0, 2] = sth
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sth
    Ry[:, 2, 2] = cth
    return np.matmul(Rz, Ry)

def sampleFibonacciDirections(
    num_directions: int,
) -> np.ndarray:
    polars = sampleFibonacciPolars(num_directions)  # (num_points, 2), (phi, theta)
    phi = polars[:, 0]
    theta = polars[:, 1]
    sin_phi = np.sin(phi)
    x = sin_phi * np.sin(theta)
    y = sin_phi * np.cos(theta)
    z = np.cos(phi)
    directions = np.stack([x, y, z], axis=1)
    return directions

def sampleRandomUp(pos: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    """
    随机生成一个垂直于视线方向的单位向量作为up方向。

    Args:
        pos: 相机位置, shape (3,)
        look_at: 注视点位置, shape (3,)

    Returns:
        up: 随机的up方向单位向量, shape (3,)
    """
    forward = look_at - pos
    forward = forward / np.linalg.norm(forward)

    rand_vec = np.random.randn(3)
    rand_vec -= np.dot(rand_vec, forward) * forward
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    return rand_vec

def fovFromFillRatio(
    object_radius: float,
    camera_to_center_dist: float,
    fill_ratio: float,
    fov_min_degree: float,
    fov_max_degree: float,
) -> float:
    """由"填充比"几何把相机到物体中心的距离映射成水平 fov(度)。

    物体近似为半径 ``object_radius`` 的包围球、相机距中心 ``camera_to_center_dist``，
    令物体投影半径占半画面的比例为 ``fill_ratio`` (ρ)。针孔投影下
        ρ = (R / dist) / tan(fov/2)
    解出
        fov = 2 · arctan( R / (ρ · dist) )
    含义：
      - ρ→1 物体恰好占满画面，ρ 越小留边越多 —— ρ 单一控制"物体在画面里的大小"。
      - dist 越大 fov 越小，正好抵消距离带来的缩放，所以 **dist 只决定透视强度**
        (近=广角强透视，远=窄角近正交)，不再影响物体大小。
    结果 clamp 到 ``[fov_min_degree, fov_max_degree]``；远距离触底时 fov 不再变窄、
    物体随之在画面中变小，这本身就是天然的尺度增强。
    """
    ratio = max(float(fill_ratio), 1e-6)
    dist = max(float(camera_to_center_dist), 1e-6)
    half_angle = np.arctan(float(object_radius) / (ratio * dist))
    fov_degree = float(np.degrees(2.0 * half_angle))
    return float(np.clip(fov_degree, fov_min_degree, fov_max_degree))


def sampleFillRatio(spec) -> float:
    """从填充比 ρ 规格抽一个标量。两种写法:
      - 扁平 ``[lo, hi]``: 在该区间均匀采样(单段)。
      - 分段 ``[[lo1,hi1], [lo2,hi2], ...]``: **各段等概率**先选一段、再段内均匀
        —— 与各段宽度无关, 因此可写 ``[[0.5,1.0],[1.0,3.0]]`` 得到 50%/50% 的混合
        (一半 ρ≤1 整物体带边距, 一半 ρ>1 居中放大裁切的特写)。
    ρ 的含义见 :func:`fovFromFillRatio`: ρ→1 占满画面, <1 留边, >1 放大出框被裁。
    """
    seg = spec
    if isinstance(spec[0], (list, tuple, np.ndarray)):   # 分段混合: 等概率选段
        seg = spec[np.random.randint(len(spec))]
    return float(np.random.uniform(float(seg[0]), float(seg[1])))


def azimuthWedgeMask(
    directions: np.ndarray,
    up_direction,
    *,
    apply_prob: float = 0.0,
    wedge_fractions: List[float] = [0.5, 0.25],
    generator: Optional[random.Random] = None,
) -> Optional[np.ndarray]:
    """随机把候选相机方向限制到绕 up 轴的一个方位角扇区(wedge), 模拟"贴墙/墙角拍摄"。

    现训练默认相机分布在整个上半球(用户绕物体转一整圈), 但贴着墙拍只能绕半圈
    (方位角 ~180°), 墙角只能绕 1/4 圈(~90°)。本原子在「候选方向」层面按 first-
    principles 过滤(而非采样后丢弃)::

      * 以概率 ``1 - apply_prob`` 不做限制, 返回 ``None``(无操作, 调用方按全 360° 处理);
      * 以概率 ``apply_prob`` 等概率从 ``wedge_fractions`` 选一个扇宽占比 ``frac``
        (0.5=半圈/贴墙, 0.25=四分之一圈/墙角), 选一个随机中心方位角 ``θ0∈[0,2π)``,
        返回布尔 mask: 候选方向绕 up 轴的方位角落在 ``[θ0−w/2, θ0+w/2]`` (mod 2π,
        ``w = frac·2π``) 内为 True。

    方位角在「垂直于 up 轴的平面」里度量, 与天顶角/俯仰角解耦, 因此扇区只压缩"绕物体
    转的水平角度", 不限制俯仰——正是贴墙/墙角场景的几何。中心 ``θ0`` 随机, 所以平面内
    参考轴的选取无关紧要。

    与 :func:`flux_mv...random_valid_to_unknown` 同构: 关键字 tunables + 早退无操作 +
    每次调用独立随机 + 纯函数(无副作用)。``None`` 表示"本次不限制"(让调用方原样处理),
    否则返回 ``(N,)`` 的 bool mask。

    Args:
        directions:        (N, 3) 候选相机方向(单位向量, 全球面 Fibonacci 候选)。
        up_direction:      绕之度量方位角的竖直轴(= 上半球过滤的 up 轴), shape (3,)。
        apply_prob:        本次调用施加扇区限制的概率。
        wedge_fractions:   可选扇宽占比集合(占整个 360° 的比例), 施加时等概率选一个。
        generator:         可选 ``random.Random`` 复现实例(默认用全局 ``random``)。
    Returns:
        ``None``(不限制) 或 (N,) bool mask(扇区内为 True)。
    """
    if apply_prob <= 0.0 or not wedge_fractions:
        return None
    rng = generator if generator is not None else random
    if rng.random() > apply_prob:
        return None

    up = np.asarray(up_direction, dtype=np.float64)
    up_norm = np.linalg.norm(up)
    if up.shape != (3,) or up_norm == 0.0:
        return None
    up = up / up_norm

    # 在垂直于 up 的平面里取一组正交基 (e1, e2) 作为方位角的参考系。
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, up) * up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    azimuth = np.arctan2(directions @ e2, directions @ e1)   # (N,) ∈ (-π, π]
    frac = float(rng.choice(list(wedge_fractions)))
    half_width = np.pi * frac                                # 半扇宽 = (frac·2π)/2
    center = rng.uniform(0.0, 2.0 * np.pi)
    # 角差 wrap 到 [-π, π]; |Δ| ≤ half_width 即落在扇区内。
    delta = np.abs((azimuth - center + np.pi) % (2.0 * np.pi) - np.pi)
    return delta <= half_width


def sampleCameras(
    mesh: trimesh.Trimesh,
    candidate_camera_num: int=720,
    camera_num: int = 4,
    camera_dist_range: List[float] = [2.5, 2.5],
    width: int = 518,
    height: int = 518,
    fovx_degree_range: List[float] = [30.0, 90.0],
    dtype = torch.float32,
    device: str = 'cuda:0',
    focus_center_ratio: float=1.0,
    up_direction: Optional[List[float]] = [0, 0, 1],
    all_camera_upper_ratio: float=0.0,
    all_camera_upper_direction: List[float] = [0, 0, 1],
    fov_fill_ratio_range: Optional[List[float]] = None,
    azimuth_wedge_prob: float = 0.0,
    azimuth_wedge_fractions: List[float] = [0.5, 0.25],
) -> List[Camera]:
    """
    创建围绕mesh均匀分布的相机和深度数据

    Args:
        mesh: 输入的三角网格
        camera_num: 相机数量
        camera_dist: 相机距离mesh中心的距离
        width: 图像宽度
        height: 图像高度
        fx: 焦距x
        fy: 焦距y
        up_direction: up向量, 如果不提供则随机生成up方向
        all_camera_upper_direction: 上半球过滤的参考方向, 与up_direction解耦
        fov_fill_ratio_range: 不为 None 时启用"fov 随距离联动"——每个相机先抽距离、
            再从该区间抽填充比 ρ，由 :func:`fovFromFillRatio` 用物体包围半径 + 相机到
            中心距离导出 fovx(度)，并 clamp 到 ``fovx_degree_range`` 作为上下限。此时
            ``fovx_degree_range`` 不再是采样区间而是 fov 的 clamp 边界。为 None 时保持
            旧行为：fovx 在 ``fovx_degree_range`` 上独立均匀采样。

    Returns:
        camera_list: 相机列表
    """
    # 计算mesh的bbox center
    bbox = mesh.bounds  # shape: (2, 3), [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    bbox_center = (bbox[0] + bbox[1]) / 2.0

    # 物体包围球半径(到 bbox 中心的最远顶点距离)——fov 联动模式下用它把"距离"换算成
    # "物体张角"。逐 mesh 真实计算, 不依赖任何归一化假设。
    object_radius = float(np.linalg.norm(mesh.vertices - bbox_center, axis=1).max())

    # 使用Fibonacci球面采样生成均匀分布的相机位置
    camera_directions = sampleFibonacciDirections(candidate_camera_num)

    upper_dir = np.asarray(all_camera_upper_direction, dtype=np.float64)
    upper_dir_norm = np.linalg.norm(upper_dir)
    filter_by_upper = upper_dir.shape == (3,) and upper_dir_norm > 0

    # 候选方向掩码: 先全 True, 再按各自概率与"上半球"、"方位角扇区"求交。两个增强独立,
    # 默认(azimuth_wedge_prob=0)时退化为原始的上半球/全球面逻辑——逐分支语义不变。
    candidate_mask = np.ones(candidate_camera_num, dtype=bool)

    # 上半球过滤(原逻辑): 以 all_camera_upper_ratio 概率限制到 dir·up ≥ 0。
    if filter_by_upper and all_camera_upper_ratio > 0 and random.random() <= all_camera_upper_ratio:
        candidate_mask &= (camera_directions @ (upper_dir / upper_dir_norm)) >= 0

    # 方位角扇区过滤(贴墙/墙角增强): 绕 up 轴限制到半圈/四分之一圈, 见 azimuthWedgeMask。
    wedge_mask = azimuthWedgeMask(
        camera_directions, all_camera_upper_direction,
        apply_prob=azimuth_wedge_prob, wedge_fractions=azimuth_wedge_fractions,
    )
    if wedge_mask is not None:
        candidate_mask &= wedge_mask

    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) >= camera_num:
        sampled_indices = random.sample(candidate_indices.tolist(), camera_num)
    else:
        # 限制后候选不足(极窄扇区/候选过少): 回退全 360° 池, 保证能取到 camera_num 个。
        sampled_indices = random.sample(range(candidate_camera_num), camera_num)

    sampled_camera_directions = camera_directions[sampled_indices]

    # 创建相机列表
    camera_list = []
    for i in range(camera_num):
        focus_center = np.random.rand() <= focus_center_ratio
        if focus_center:
            look_at = bbox_center
        else:
            look_at = mesh.vertices[np.random.randint(0, mesh.vertices.shape[0])]

        camera_dist = np.random.uniform(camera_dist_range[0], camera_dist_range[1])

        camera_position = look_at + camera_dist * sampled_camera_directions[i]

        if up_direction is None:
            camera_up = sampleRandomUp(camera_position, look_at)
        else:
            camera_up = up_direction

        if fov_fill_ratio_range is None:
            fovx_degree = np.random.uniform(*fovx_degree_range)
        else:
            # fov 随距离联动: 抽填充比 ρ, 用"相机到 bbox 中心距离"(对 look_at 偏心也成立)
            # + 物体包围半径导出 fovx, 使物体大小由 ρ 控制、距离只控制透视强度。
            # ρ 支持分段等概率混合(见 sampleFillRatio), 如 [[0.5,1.0],[1.0,3.0]] = 50/50。
            fill_ratio = sampleFillRatio(fov_fill_ratio_range)
            dist_to_center = float(np.linalg.norm(camera_position - bbox_center))
            fovx_degree = fovFromFillRatio(
                object_radius, dist_to_center, fill_ratio,
                fovx_degree_range[0], fovx_degree_range[1],
            )

        camera = Camera(
            width=width,
            height=height,
            fovx_degree=fovx_degree,
            pos=camera_position,
            look_at=bbox_center,
            up=camera_up,
            dtype=dtype,
            device=device,
        )
        camera_list.append(camera)

    return camera_list
