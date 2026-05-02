import torch
import numpy as np

from typing import Union

from camera_control.Method.data import toTensor


@torch.no_grad()
def _computeKNearestNeighborDistances(
    xyz: torch.Tensor,
    k: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    '''
    分块计算每个点到自身以外 k 个最近点的距离 (N, k)。
    '''
    n = int(xyz.shape[0])
    device = xyz.device
    k = max(1, min(int(k), n - 1))
    knn_dist = torch.empty((n, k), dtype=xyz.dtype, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = xyz[start:end]
        dist = torch.cdist(chunk, xyz, p=2)
        topk_vals, _ = torch.topk(dist, k=k + 1, dim=1, largest=False)
        knn_dist[start:end] = topk_vals[:, 1:]

    return knn_dist


def _sparseOutlierThreshold(scores: torch.Tensor) -> float:
    '''
    根据局部稀疏度分数自动寻找离群阈值。
    优先使用最高分尾部小簇的明显对数间隙；没有明显间隙时使用保守分位数兜底。
    '''
    finite_scores = scores[torch.isfinite(scores)]
    if finite_scores.numel() == 0:
        return float('inf')

    sorted_scores = torch.sort(finite_scores.float()).values
    count = int(sorted_scores.numel())
    tiny = torch.finfo(sorted_scores.dtype).eps
    median = float(sorted_scores[count // 2])
    mad = float(torch.median(torch.abs(sorted_scores - median)))
    robust_limit = median + 8.0 * 1.4826 * mad

    q_values = torch.quantile(
        sorted_scores,
        torch.tensor([0.95, 0.99, 0.995, 0.999], device=sorted_scores.device),
    )
    fallback_limit = max(robust_limit, float(q_values[2]))

    threshold = fallback_limit
    gap_reject_count = 0

    if count >= 8:
        log_scores = torch.log(torch.clamp(sorted_scores, min=tiny))
        gaps = log_scores[1:] - log_scores[:-1]
        tail_start = max(0, int(count * 0.90) - 1)
        if tail_start < int(gaps.numel()):
            tail_gaps = gaps[tail_start:]
            max_gap, local_idx = torch.max(tail_gaps, dim=0)
            gap_index = int(tail_start + int(local_idx))
            gap_value = float(max_gap)
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
            if tight_start < int(gaps.numel()):
                tight_gaps = gaps[tight_start:]
                candidate_mask = tight_gaps >= float(np.log(1.5))
                if bool(candidate_mask.any()):
                    candidate_local = torch.nonzero(candidate_mask, as_tuple=False)[0, 0]
                    selected_gap_index = int(tight_start + int(candidate_local))
                    selected_gap_reject_count = count - selected_gap_index - 1
                    if selected_gap_reject_count > gap_reject_count:
                        left = float(sorted_scores[selected_gap_index])
                        right = float(sorted_scores[selected_gap_index + 1])
                        threshold = float(np.sqrt(left * right))

    return threshold


def _expandMaskByBBox(
    xyz: torch.Tensor,
    mask: torch.Tensor,
    expand_ratio: float = 0.1,
) -> torch.Tensor:
    '''
    基于当前主簇 bbox 逐步膨胀, 拾回可能被 kNN 稀疏度误删的表面点。
    每轮按初始 bbox 的 xyz 边长膨胀 expand_ratio, 没有新增点则停止。
    '''
    if int(mask.sum()) == 0:
        return mask

    expanded_mask = mask.clone()
    kept_xyz = xyz[expanded_mask]
    bbox_min = kept_xyz.amin(dim=0)
    bbox_max = kept_xyz.amax(dim=0)
    expand_step = (bbox_max - bbox_min) * float(expand_ratio)

    while True:
        test_min = bbox_min - expand_step
        test_max = bbox_max + expand_step
        new_mask = ((xyz >= test_min) & (xyz <= test_max)).all(dim=1) | expanded_mask

        if int(new_mask.sum()) <= int(expanded_mask.sum()):
            return expanded_mask

        expanded_mask = new_mask
        kept_xyz = xyz[expanded_mask]
        bbox_min = kept_xyz.amin(dim=0)
        bbox_max = kept_xyz.amax(dim=0)


@torch.no_grad()
def searchMainClusterPointMask(
    points: Union[torch.Tensor, np.ndarray, list],
    chunk_size: int = 4096,
    knn_k: int = 16,
) -> np.ndarray:
    '''
    返回主物体点云掩码: True 为保留, False 为离群点。

    算法:
      1. 计算每个点到自身以外第 k 个最近点的距离作为局部稀疏度分数。
      2. 在稀疏度分数的高分尾部寻找明显间隙, 删除间隙右侧的小簇。
      3. 若没有明显尾部间隙, 使用 MAD + 高分位数的保守阈值兜底。
      4. 基于筛选后的主簇 bbox 逐轮膨胀, 拾回可能被误删的物体表面点。

    knn_k 控制局部稀疏度使用第几个近邻距离。

    输入无效或点数过少时返回全 True 掩码; 无法确定点数时返回 shape (0,)
    的 bool 数组。
    '''
    if isinstance(points, torch.Tensor):
        xyz = points.detach()
    else:
        xyz = toTensor(points)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        if xyz.ndim == 2 and int(xyz.shape[0]) > 0:
            return np.ones(int(xyz.shape[0]), dtype=np.bool_)
        return np.array([], dtype=np.bool_)

    if not torch.is_floating_point(xyz):
        xyz = xyz.float()

    n = int(xyz.shape[0])
    if n < 4:
        return np.ones(n, dtype=np.bool_)

    density_k = max(1, min(int(knn_k), n - 1))
    knn_dist = _computeKNearestNeighborDistances(
        xyz=xyz,
        k=density_k,
        chunk_size=int(chunk_size),
    )
    density_score = knn_dist[:, -1]
    density_threshold = _sparseOutlierThreshold(density_score)
    density_mask = density_score <= density_threshold
    expanded_mask = _expandMaskByBBox(xyz=xyz, mask=density_mask, expand_ratio=0.1)

    return expanded_mask.cpu().numpy().astype(np.bool_)
