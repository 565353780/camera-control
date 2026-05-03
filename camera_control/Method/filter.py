import numpy as np

from typing import Any

from camera_control.Method.data import toNumpy


def _maskByBBox(
    xyz_np: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    return np.all((xyz_np >= bbox_min) & (xyz_np <= bbox_max), axis=1)


def _octreeCoreMask(
    xyz_np: np.ndarray,
    leaf_count_ratio: float,
    octree_max_depth: int,
    min_leaf_points: int,
    min_remaining_points: int,
    max_depth_reject_ratio: float,
) -> np.ndarray:
    '''
    基于自适应八叉树的逐深度叶子占用率筛选 core 表面点。

    与静态八叉树的关键区别: 每进入下一个 depth 之前, 用当前仍保留的点重新计算
    bbox, 再对这个 bbox 做该 depth 的体素化与占用率剔除。这样在 "离群点从物体
    向远处延伸" 的极端场景下, 第 1 层先剔掉远端稀疏链, 之后 bbox 会大幅收紧,
    第 2~N 层再以更细尺度识别越来越靠近物体的低密度离群点, 而不会被极端 bbox
    把整个物体表面压成单个 voxel 的退化失败。

    每个 depth 的剔除规则: 叶子节点点数小于该深度最大叶子节点点数
    leaf_count_ratio 倍 (默认 1%) 的体素中所有点全部抛弃。

    早停: 若某个 depth 提议的剔除比例超过 max_depth_reject_ratio, 视作阈值
    已开始误删真实表面 (例如低密度但合法的表面区域), 撤销该层并停止迭代;
    这条规则把 "用 1% 阈值识别离群体素" 与 "在均匀阈值下不再继续切真实表面"
    两个第一性原理结合。

    返回长度 n 的 bool mask, True 表示在所有 (实际生效的) 深度都属于高占用
    叶子。
    '''
    n = int(xyz_np.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.bool_)

    keep_mask = np.ones(n, dtype=np.bool_)
    leaf_count_ratio = float(leaf_count_ratio)
    min_leaf_points = max(1, int(min_leaf_points))
    min_remaining_points = max(4, int(min_remaining_points))
    max_depth_reject_ratio = max(0.0, float(max_depth_reject_ratio))

    print('[INFO][filter::_octreeCoreMask]')
    print('\t scan octree leaves per depth (adaptive bbox)...')

    for depth in range(1, int(octree_max_depth) + 1):
        sub_indices = np.flatnonzero(keep_mask)
        sub_n = int(sub_indices.size)
        if sub_n < min_remaining_points:
            break

        sub_xyz = xyz_np[sub_indices]
        bbox_min = sub_xyz.min(axis=0).astype(np.float64)
        bbox_max = sub_xyz.max(axis=0).astype(np.float64)
        extent = bbox_max - bbox_min

        if not np.all(np.isfinite(extent)) or float(np.max(extent)) <= 0.0:
            break

        safe_extent = np.where(extent > 0.0, extent, 1.0)
        rel = (sub_xyz - bbox_min) / safe_extent

        grid = 1 << depth
        idx_xyz = np.floor(rel * float(grid)).astype(np.int64)
        np.clip(idx_xyz, 0, grid - 1, out=idx_xyz)
        linear = (idx_xyz[:, 0] * grid + idx_xyz[:, 1]) * grid + idx_xyz[:, 2]

        _, inverse, counts = np.unique(
            linear, return_inverse=True, return_counts=True,
        )
        max_count = int(counts.max())
        threshold = max(min_leaf_points, int(np.ceil(max_count * leaf_count_ratio)))
        leaf_keep = counts >= threshold
        local_rejected = ~leaf_keep[inverse]
        depth_reject_count = int(local_rejected.sum())
        depth_reject_ratio = depth_reject_count / float(sub_n) if sub_n > 0 else 0.0

        print(
            '\t depth', depth,
            '| sub_n', sub_n,
            '| leaves', int(counts.size),
            '| max_leaf_count', max_count,
            '| threshold', threshold,
            '| reject', depth_reject_count,
            '| reject_ratio', round(depth_reject_ratio, 4),
            '| extent', np.round(extent, 4).tolist(),
        )

        if depth_reject_ratio > max_depth_reject_ratio:
            print(
                '\t -> stop: depth_reject_ratio',
                round(depth_reject_ratio, 4),
                '>',
                max_depth_reject_ratio,
                '(treat as surface erosion, undo this depth)',
            )
            break

        if depth_reject_count > 0:
            keep_mask[sub_indices[local_rejected]] = False

    return keep_mask


def _expandedBBoxMask(
    xyz_np: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    expand_ratio: float,
) -> np.ndarray:
    '''
    基于 core bbox 做一次性有限扩张, 召回 core 阶段被过度删除的边缘表面点。
    不做 "扩到无新增点为止" 的迭代扩张, 避免沿连续离群链把远端点重新拉回来。
    '''
    bbox_min = bbox_min.astype(np.float64).reshape(3)
    bbox_max = bbox_max.astype(np.float64).reshape(3)
    pad = (bbox_max - bbox_min) * float(expand_ratio)
    return _maskByBBox(xyz_np, bbox_min - pad, bbox_max + pad)


def searchMainClusterPointMask(
    points: Any,
    leaf_count_ratio: float = 0.01,
    octree_max_depth: int = 8,
    min_leaf_points: int = 1,
    min_remaining_points: int = 16,
    max_depth_reject_ratio: float = 0.01,
    bbox_expand_ratio: float = 0.05,
    min_core_keep_ratio: float = 0.5,
) -> np.ndarray:
    '''
    返回主物体点云掩码: True 为保留, False 为离群点。

    算法 (基于第一性原理):
      1. 在原始 xyz 上做自适应八叉树占用率筛选: 每个 depth 用当前仍保留的点重
         算 bbox, 再做该 depth 的体素化, 把叶子点数少于最大叶子点数
         leaf_count_ratio 倍的低占用体素整体抛弃。bbox 随每层收紧, 自然适配
         "离群点从物体延伸到远处" 的多尺度场景, 得到 core 表面点集。
         若某 depth 触发的剔除比例超过 max_depth_reject_ratio, 撤销该层并停止
         迭代, 视作 1% 阈值已经开始切真实表面。
      2. 用 core 点集的 bbox 做一次有限的 bbox_expand_ratio 比例扩张, 仅召回
         紧邻 core bbox 的边缘表面点; 不沿离群链做无限扩张。

    超参 (调用方仅需传 points; 其余仅供调参):
      - leaf_count_ratio:        每深度叶子保留阈值占最大叶子点数的比例 (默认 1%)
      - octree_max_depth:        最大扫描深度
      - min_leaf_points:         叶子保留的下界点数, 防止阈值退化为 0
      - min_remaining_points:    剩余点数低于该值时停止更深的 octree 迭代
      - max_depth_reject_ratio:  单层剔除比例上限, 超过即视为开始切表面并停止
      - bbox_expand_ratio:       一次性 bbox 扩张比例
      - min_core_keep_ratio:     core 保留比例不足时退化为全保留, 防止误删全场

    输入无效或点数过少时返回全 True 掩码; 无法确定点数时返回 shape (0,)
    的 bool 数组。
    '''
    xyz_np = toNumpy(points, np.float64)

    if xyz_np.ndim != 2 or xyz_np.shape[1] != 3:
        if xyz_np.ndim == 2 and int(xyz_np.shape[0]) > 0:
            return np.ones(int(xyz_np.shape[0]), dtype=np.bool_)
        return np.array([], dtype=np.bool_)

    n = int(xyz_np.shape[0])
    if n < 4:
        return np.ones(n, dtype=np.bool_)

    core_mask = _octreeCoreMask(
        xyz_np=xyz_np,
        leaf_count_ratio=float(leaf_count_ratio),
        octree_max_depth=int(octree_max_depth),
        min_leaf_points=int(min_leaf_points),
        min_remaining_points=int(min_remaining_points),
        max_depth_reject_ratio=float(max_depth_reject_ratio),
    )

    core_count = int(core_mask.sum())
    min_core_count = int(np.ceil(float(n) * float(min_core_keep_ratio)))

    print('[INFO][filter::searchMainClusterPointMask]')
    print('\t total', n, '| core', core_count, '| min_core_count', min_core_count)

    if core_count == 0 or core_count < min_core_count:
        print('[WARN][filter::searchMainClusterPointMask]')
        print('\t core kept too few points, fall back to all True')
        return np.ones(n, dtype=np.bool_)

    core_xyz = xyz_np[core_mask]
    bbox_min = core_xyz.min(axis=0)
    bbox_max = core_xyz.max(axis=0)

    expanded_mask = _expandedBBoxMask(
        xyz_np=xyz_np,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        expand_ratio=float(bbox_expand_ratio),
    )

    print('\t kept after bbox expand:', int(expanded_mask.sum()))

    return expanded_mask.astype(np.bool_)
