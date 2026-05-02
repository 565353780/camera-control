import numpy as np
import open3d as o3d

from tqdm import trange
from typing import Any, Tuple, Union

from camera_control.Method.data import toNumpy
from camera_control.Method.pcd import toPcd


def _sampleProxyPoints(
    xyz_np: np.ndarray,
    proxy_point_num: int = 20000,
) -> np.ndarray:
    '''
    使用 Open3D voxel downsample 快速得到近似均匀的代理点。
    通过 bbox 估计初始 voxel size, 再二分搜索到接近 proxy_point_num 的点数。
    点数不足时直接返回原始点。
    '''
    n = int(xyz_np.shape[0])
    target = int(proxy_point_num)
    if target <= 0 or n <= target:
        return xyz_np

    pcd = toPcd(xyz_np)
    bbox_min = np.min(xyz_np, axis=0)
    bbox_max = np.max(xyz_np, axis=0)
    bbox_extent = np.maximum(bbox_max - bbox_min, 0.0)
    positive_extent = bbox_extent[bbox_extent > 0.0]

    print('[INFO][filter::_sampleProxyPoints]')
    print('\t start sample proxy points...')
    print('\t point num from', n, 'to around', target)

    if positive_extent.size == 0 or not np.all(np.isfinite(positive_extent)):
        return xyz_np

    min_count = max(1, int(np.floor(target * 0.8)))
    max_count = max(min_count, int(np.ceil(target * 1.2)))
    max_iter = 16

    def _downsample(voxel_size: float) -> np.ndarray:
        sampled = pcd.voxel_down_sample(float(voxel_size))
        return np.asarray(sampled.points, dtype=np.float64)

    def _is_valid_count(count: int) -> bool:
        return min_count <= count <= max_count

    bbox_measure = float(np.prod(positive_extent))
    voxel_size = float(
        np.power(bbox_measure / float(target), 1.0 / float(positive_extent.size)),
    )
    if voxel_size <= 0.0 or not np.isfinite(voxel_size):
        return xyz_np

    best_proxy = _downsample(voxel_size)
    best_count = int(best_proxy.shape[0])
    best_error = abs(best_count - target)
    if best_count == 0:
        return xyz_np
    if _is_valid_count(best_count):
        print('\t sampled point num:', best_count)
        return best_proxy

    low_size = None
    high_size = None

    if best_count > max_count:
        low_size = voxel_size
        search_size = voxel_size
        for _ in range(max_iter):
            search_size *= 2.0
            proxy_np = _downsample(search_size)
            count = int(proxy_np.shape[0])
            if count == 0:
                break
            error = abs(count - target)
            if error < best_error:
                best_proxy = proxy_np
                best_count = count
                best_error = error
            if _is_valid_count(count):
                print('\t sampled point num:', count)
                return proxy_np
            if count < min_count:
                high_size = search_size
                break
            low_size = search_size
    else:
        high_size = voxel_size
        search_size = voxel_size
        for _ in range(max_iter):
            search_size *= 0.5
            proxy_np = _downsample(search_size)
            count = int(proxy_np.shape[0])
            if count == 0:
                break
            error = abs(count - target)
            if error < best_error:
                best_proxy = proxy_np
                best_count = count
                best_error = error
            if _is_valid_count(count):
                print('\t sampled point num:', count)
                return proxy_np
            if count > max_count:
                low_size = search_size
                break
            high_size = search_size

    if low_size is None or high_size is None:
        print('\t sampled point num:', best_count)
        return best_proxy

    for _ in range(max_iter):
        mid_size = (low_size + high_size) * 0.5
        proxy_np = _downsample(mid_size)
        count = int(proxy_np.shape[0])
        if count == 0:
            high_size = mid_size
            continue

        error = abs(count - target)
        if error < best_error:
            best_proxy = proxy_np
            best_count = count
            best_error = error

        if _is_valid_count(count):
            print('\t sampled point num:', count)
            return proxy_np

        if count > max_count:
            low_size = mid_size
        else:
            high_size = mid_size

    print('\t sampled point num:', best_count)
    return best_proxy

def _computeKNearestNeighborDistances(
    xyz_np: np.ndarray,
    k: int,
) -> np.ndarray:
    '''
    使用 Open3D KDTree 计算每个点到自身以外 k 个最近点的欧氏距离 (N, k)。
    '''
    n = int(xyz_np.shape[0])
    k = max(1, min(int(k), n - 1))

    pcd = toPcd(xyz_np)
    tree = o3d.geometry.KDTreeFlann(pcd)

    knn_dist_np = np.empty((n, k), dtype=np.float64)
    query_count = k + 1
    print('[INFO][filter::_computeKNearestNeighborDistances]')
    print('\t start search knn dists...')
    for i in trange(n):
        found, _, sq_dists = tree.search_knn_vector_3d(xyz_np[i], query_count)
        sq = np.asarray(sq_dists, dtype=np.float64)
        if found < query_count:
            # 重复点等极端情况下查询不足 k+1 个邻居, 用最后一个距离填充
            fill_value = float(sq[-1]) if sq.size > 0 else 0.0
            padded = np.full(query_count, fill_value, dtype=np.float64)
            if sq.size > 0:
                padded[: sq.size] = sq
            sq = padded
        # search_knn_vector_3d 返回平方距离, 第 0 个是自身 (0.0)
        knn_dist_np[i] = np.sqrt(np.maximum(sq[1:query_count], 0.0))

    return knn_dist_np


def _sparseOutlierThreshold(scores: np.ndarray) -> float:
    '''
    根据局部稀疏度分数自动寻找离群阈值。
    优先使用最高分尾部小簇的明显对数间隙；没有明显间隙时使用保守分位数兜底。
    '''
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return float('inf')

    sorted_scores = np.sort(finite_scores.astype(np.float64))
    count = int(sorted_scores.size)
    tiny = float(np.finfo(sorted_scores.dtype).eps)
    median = float(sorted_scores[count // 2])
    mad = float(np.median(np.abs(sorted_scores - median)))
    robust_limit = median + 8.0 * 1.4826 * mad

    q_values = np.quantile(sorted_scores, [0.95, 0.99, 0.995, 0.999])
    fallback_limit = max(robust_limit, float(q_values[2]))

    threshold = fallback_limit
    gap_reject_count = 0

    if count >= 8:
        log_scores = np.log(np.clip(sorted_scores, a_min=tiny, a_max=None))
        gaps = log_scores[1:] - log_scores[:-1]
        tail_start = max(0, int(count * 0.90) - 1)
        if tail_start < int(gaps.size):
            tail_gaps = gaps[tail_start:]
            local_idx = int(np.argmax(tail_gaps))
            gap_value = float(tail_gaps[local_idx])
            gap_index = tail_start + local_idx
            gap_reject_count = count - gap_index - 1
            max_reject_count = max(16, int(count * 0.02))
            if (
                gap_value >= float(np.log(2.5))
                and 1 <= gap_reject_count <= max_reject_count
            ):
                left = float(sorted_scores[gap_index])
                right = float(sorted_scores[gap_index + 1])
                threshold = float(np.sqrt(left * right))

            # 对扫描物体点云，近离群点通常表现为最高分尾部中的小簇。
            # 只在最多剔除 10 个点时放宽到较小尾部间隙，避免误删真实稀疏表面。
            tight_tail_reject_limit = min(10, max(1, int(count * 0.001)))
            tight_start = max(0, count - tight_tail_reject_limit - 1)
            if tight_start < int(gaps.size):
                tight_gaps = gaps[tight_start:]
                candidate_mask = tight_gaps >= float(np.log(1.5))
                if bool(candidate_mask.any()):
                    candidate_local = int(np.flatnonzero(candidate_mask)[0])
                    selected_gap_index = tight_start + candidate_local
                    selected_gap_reject_count = count - selected_gap_index - 1
                    if selected_gap_reject_count > gap_reject_count:
                        left = float(sorted_scores[selected_gap_index])
                        right = float(sorted_scores[selected_gap_index + 1])
                        threshold = float(np.sqrt(left * right))

    return threshold


def _maskByBBox(
    xyz_np: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    return np.all((xyz_np >= bbox_min) & (xyz_np <= bbox_max), axis=1)


def _expandMaskByBBox(
    xyz_np: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    expand_ratio: float = 0.1,
) -> np.ndarray:
    '''
    基于初始 bbox 在 xyz_np 上逐轮膨胀, 拾回可能被 kNN 稀疏度误删的表面点。
    每轮按初始 bbox 的 xyz 边长膨胀 expand_ratio, 没有新增点则停止。
    '''
    bbox_min = bbox_min.astype(np.float64).reshape(3)
    bbox_max = bbox_max.astype(np.float64).reshape(3)
    expand_step = (bbox_max - bbox_min) * float(expand_ratio)

    expanded_mask = _maskByBBox(xyz_np, bbox_min, bbox_max)

    cur_min = bbox_min.copy()
    cur_max = bbox_max.copy()

    while True:
        cur_min = cur_min - expand_step
        cur_max = cur_max + expand_step
        new_mask = _maskByBBox(xyz_np, cur_min, cur_max)

        if int(new_mask.sum()) <= int(expanded_mask.sum()):
            return expanded_mask

        expanded_mask = new_mask


def _proxyMainClusterBBox(
    proxy_xyz: np.ndarray,
    knn_k: int,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    '''
    在代理点集上估计主物体的 bbox。失败时返回 None, 表示降级使用全部点。
    '''
    n = int(proxy_xyz.shape[0])
    if n < 4:
        return None

    density_k = max(1, min(int(knn_k), n - 1))
    knn_dist = _computeKNearestNeighborDistances(xyz_np=proxy_xyz, k=density_k)
    density_score = knn_dist[:, -1]
    density_threshold = _sparseOutlierThreshold(density_score)
    density_mask = density_score <= density_threshold

    if not bool(density_mask.any()):
        return None

    kept = proxy_xyz[density_mask]
    bbox_min = kept.min(axis=0)
    bbox_max = kept.max(axis=0)
    return bbox_min, bbox_max


def searchMainClusterPointMask(
    points: Any,
    chunk_size: int = 4096,
    knn_k: int = 16,
    proxy_point_num: int = 20000,
    expand_ratio: float = 0.1,
) -> np.ndarray:
    '''
    返回主物体点云掩码: True 为保留, False 为离群点。

    算法:
      1. 对原始 xyz 做 Open3D voxel downsample 得到约 proxy_point_num 个
         近似均匀代理点 (点数不足时直接使用原始点)。
      2. 在代理点上计算每个点到自身以外第 k 个最近点的距离作为局部稀疏度分数,
         结合尾部对数间隙 / MAD + 高分位数兜底确定离群阈值, 得到代理主簇 bbox。
      3. 在原始 xyz 上以代理主簇 bbox 为初始范围, 按 bbox 边长 expand_ratio
         逐轮膨胀, 直到没有新增点, 得到最终保留 mask。

    chunk_size 已废弃, 仅为兼容旧调用保留。

    输入无效或点数过少时返回全 True 掩码; 无法确定点数时返回 shape (0,)
    的 bool 数组。
    '''
    _ = chunk_size

    xyz_np = toNumpy(points, np.float64)

    if xyz_np.ndim != 2 or xyz_np.shape[1] != 3:
        if xyz_np.ndim == 2 and int(xyz_np.shape[0]) > 0:
            return np.ones(int(xyz_np.shape[0]), dtype=np.bool_)
        return np.array([], dtype=np.bool_)

    n = int(xyz_np.shape[0])
    if n < 4:
        return np.ones(n, dtype=np.bool_)

    proxy_xyz = _sampleProxyPoints(xyz_np, proxy_point_num=int(proxy_point_num))
    bbox = _proxyMainClusterBBox(proxy_xyz=proxy_xyz, knn_k=int(knn_k))

    if bbox is None:
        return np.ones(n, dtype=np.bool_)

    bbox_min, bbox_max = bbox
    expanded_mask = _expandMaskByBBox(
        xyz_np=xyz_np,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        expand_ratio=float(expand_ratio),
    )

    return expanded_mask.astype(np.bool_)
