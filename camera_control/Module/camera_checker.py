import numpy as np

from typing import Iterable, List, Optional, Tuple

from camera_control.Module.camera import Camera


# 默认的 camera list 升级判定阈值，集中放在类上方便外部覆盖与文档化。
# 这些数值来自 VGGT vs VGGT-Omega 的实测经验，可以按项目需要调优。
_DEFAULT_POS_MEDIAN_REL_THRESH = 0.05
_DEFAULT_POS_P95_REL_THRESH = 0.15
_DEFAULT_ROT_MEDIAN_DEG_THRESH = 5.0
_DEFAULT_ROT_P95_DEG_THRESH = 15.0
_DEFAULT_ROT_MAX_DEG_THRESH = 90.0
_DEFAULT_CAND_RADIUS_MIN_RATIO = 0.05
_DEFAULT_MIN_DIRECTION_SPREAD_DEG = 30.0
_DEFAULT_DIRECTION_SPREAD_RATIO = 0.5
_DEFAULT_MATCH_MAX_ITER = 5


class CameraChecker(object):
    """围绕 camera list 的纯几何分析 / 退化检查工具集。

    所有方法都是 ``@staticmethod``，只依赖相机对象暴露的
    ``pos`` / ``forwardDirection`` / ``upDirection``（torch.Tensor，shape (3,)）
    与可选的 ``image_id`` 属性，刻意不耦合任何 pipeline / IO / 模型推理逻辑，
    便于其他项目直接复用作为相机序列健康度检查模块。
    """

    POS_MEDIAN_REL_THRESH = _DEFAULT_POS_MEDIAN_REL_THRESH
    POS_P95_REL_THRESH = _DEFAULT_POS_P95_REL_THRESH
    ROT_MEDIAN_DEG_THRESH = _DEFAULT_ROT_MEDIAN_DEG_THRESH
    ROT_P95_DEG_THRESH = _DEFAULT_ROT_P95_DEG_THRESH
    ROT_MAX_DEG_THRESH = _DEFAULT_ROT_MAX_DEG_THRESH
    CAND_RADIUS_MIN_RATIO = _DEFAULT_CAND_RADIUS_MIN_RATIO
    MIN_DIRECTION_SPREAD_DEG = _DEFAULT_MIN_DIRECTION_SPREAD_DEG
    DIRECTION_SPREAD_RATIO = _DEFAULT_DIRECTION_SPREAD_RATIO
    MATCH_MAX_ITER = _DEFAULT_MATCH_MAX_ITER

    def __init__(self) -> None:
        return

    # ------------------------------------------------------------------
    # 基础向量提取
    # ------------------------------------------------------------------

    @staticmethod
    def _tensorVectorToNumpy(
        vector,
        fallback: Optional[np.ndarray]=None,
    ) -> np.ndarray:
        """把单个相机方向 / 位置向量转为 float64 numpy，必要时回退到 fallback。

        当向量近似零向量（数值上无法定义方向）时，返回 ``fallback`` 的拷贝；
        若没有提供 fallback，则返回原始向量未归一化结果，由上层决定如何处理。
        """
        arr = np.asarray(vector.detach().cpu().numpy(), dtype=np.float64).reshape(3)
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            if fallback is None:
                return arr
            return np.asarray(fallback, dtype=np.float64).reshape(3).copy()
        return arr / norm

    @staticmethod
    def cameraPositions(camera_list: Iterable[Camera]) -> np.ndarray:
        """返回相机位置矩阵 ``(N, 3)``，float64。"""
        pos_list = []
        for cam in camera_list:
            pos = np.asarray(
                cam.pos.detach().cpu().numpy(),
                dtype=np.float64,
            ).reshape(3)
            pos_list.append(pos)
        if len(pos_list) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(pos_list, axis=0)

    @staticmethod
    def cameraForwardDirections(camera_list: Iterable[Camera]) -> np.ndarray:
        """返回相机前向方向矩阵 ``(N, 3)``，已单位化。

        近零向量回退为 ``[1, 0, 0]``，避免下游归一化除零。
        """
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        dirs = [
            CameraChecker._tensorVectorToNumpy(cam.forwardDirection, fallback)
            for cam in camera_list
        ]
        if len(dirs) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(dirs, axis=0)

    @staticmethod
    def cameraUpDirections(camera_list: Iterable[Camera]) -> np.ndarray:
        """返回相机 up 方向矩阵 ``(N, 3)``，已单位化。

        近零向量回退为 ``[0, 0, 1]``。
        """
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dirs = [
            CameraChecker._tensorVectorToNumpy(cam.upDirection, fallback)
            for cam in camera_list
        ]
        if len(dirs) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(dirs, axis=0)

    # ------------------------------------------------------------------
    # 通用向量 / 角度 / 轨迹度量
    # ------------------------------------------------------------------

    @staticmethod
    def normalizeVectors(vectors: np.ndarray, eps: float=1e-12) -> np.ndarray:
        """按行做单位化，加 ``eps`` 防止除零。"""
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + eps)

    @staticmethod
    def trajectoryRadius(positions: np.ndarray) -> float:
        """以中位距离衡量相机轨迹的"半径"。

        对离群点比平均距离更鲁棒，适合用作"轨迹是否塌缩"的尺度参考。
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if positions.shape[0] == 0:
            return 0.0
        center = positions.mean(axis=0)
        return float(np.median(np.linalg.norm(positions - center, axis=1)))

    @staticmethod
    def directionMaxAngleDeg(directions: np.ndarray) -> float:
        """所有方向两两夹角中的最大值（度）。

        刻画"朝向是否过于一致"——若所有相机朝向几乎平行，
        最大夹角会接近 0，这通常是位姿估计退化的信号。
        """
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if directions.shape[0] < 2:
            return 0.0
        dots = np.clip(directions @ directions.T, -1.0, 1.0)
        # 对角线 dot=1，不会变成最小值。
        return float(np.degrees(np.arccos(np.min(dots))))

    @staticmethod
    def rotateDirections(directions: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """对方向集做右乘旋转 ``directions @ R``，并重新单位化。"""
        rotated = np.asarray(directions, dtype=np.float64) @ np.asarray(
            rotation, dtype=np.float64,
        )
        return CameraChecker.normalizeVectors(rotated)

    @staticmethod
    def angleMatrixDeg(
        src_dirs: np.ndarray,
        dst_dirs: np.ndarray,
    ) -> np.ndarray:
        """所有 src 与 dst 方向两两夹角矩阵（度），shape ``(Ns, Nd)``。"""
        cos_ang = np.clip(
            np.asarray(src_dirs, dtype=np.float64)
            @ np.asarray(dst_dirs, dtype=np.float64).T,
            -1.0,
            1.0,
        )
        return np.degrees(np.arccos(cos_ang))

    @staticmethod
    def pairedAngleDeg(
        src_dirs: np.ndarray,
        dst_dirs: np.ndarray,
    ) -> np.ndarray:
        """逐对方向夹角向量（度），shape ``(N,)``，要求两者长度一致。"""
        cos_ang = np.clip(
            np.sum(
                np.asarray(src_dirs, dtype=np.float64)
                * np.asarray(dst_dirs, dtype=np.float64),
                axis=1,
            ),
            -1.0,
            1.0,
        )
        return np.degrees(np.arccos(cos_ang))

    # ------------------------------------------------------------------
    # 对齐与匹配
    # ------------------------------------------------------------------

    @staticmethod
    def fitAnisotropicAlignment(
        src_pos: np.ndarray,
        dst_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """拟合 ``src @ A + t ~= dst``，允许旋转+平移+各向异性缩放。

        返回 ``(A, t, R)``：
          - ``A`` 为 3x3 一般线性矩阵（含各向异性缩放/剪切自由度的总变换）。
          - ``t`` 为 3 维平移。
          - ``R`` 为 ``A`` 的最近正交旋转矩阵（SVD 去剪切后的纯旋转），
            用于把朝向向量从 src 坐标系旋转到 dst 坐标系，做角度比较。
        """
        src = np.asarray(src_pos, dtype=np.float64).reshape(-1, 3)
        dst = np.asarray(dst_pos, dtype=np.float64).reshape(-1, 3)

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        A, *_ = np.linalg.lstsq(src_centered, dst_centered, rcond=None)
        t = dst_mean - src_mean @ A

        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return A, t, R

    @staticmethod
    def minimumCostAssignment(
        cost: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """求解方阵最小代价分配（匈牙利算法），不依赖 scipy。

        返回 ``(row_ids, col_ids)``，``row_ids`` 总是 ``[0..n-1]``，
        ``col_ids[i]`` 是第 ``i`` 行配对的列索引。
        """
        cost = np.asarray(cost, dtype=np.float64)
        n, m = cost.shape
        if n == 0 or n != m:
            raise ValueError('cost must be a non-empty square matrix.')

        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(m + 1, dtype=np.float64)
        p = np.zeros(m + 1, dtype=np.int64)
        way = np.zeros(m + 1, dtype=np.int64)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf, dtype=np.float64)
            used = np.zeros(m + 1, dtype=bool)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

                for j in range(0, m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        col_for_row = np.empty(n, dtype=np.int64)
        for j in range(1, m + 1):
            col_for_row[p[j] - 1] = j - 1
        return np.arange(n, dtype=np.int64), col_for_row

    @staticmethod
    def imageIdInitialOrder(
        ref_list: List[Camera],
        cand_list: List[Camera],
    ) -> Tuple[List[Camera], bool]:
        """若两侧都有唯一且匹配的 ``image_id``，按 ref 顺序重排 cand。

        返回 ``(reordered_cand_list, used_image_id_order)``。无法使用 image_id
        时，直接返回 cand 的浅拷贝并标记 False。
        """
        ref_ids = [getattr(c, 'image_id', None) for c in ref_list]
        cand_ids = [getattr(c, 'image_id', None) for c in cand_list]
        if (
            all(image_id is not None for image_id in ref_ids)
            and all(image_id is not None for image_id in cand_ids)
            and len(set(ref_ids)) == len(ref_ids)
            and set(ref_ids) == set(cand_ids)
        ):
            cand_by_id = {
                getattr(camera, 'image_id'): camera
                for camera in cand_list
            }
            return [cand_by_id[image_id] for image_id in ref_ids], True
        return list(cand_list), False

    @staticmethod
    def cameraMatchCost(
        ref_pos: np.ndarray,
        ref_dirs: np.ndarray,
        ref_up_dirs: np.ndarray,
        cand_pos_aligned: np.ndarray,
        cand_dirs_rotated: np.ndarray,
        cand_up_dirs_rotated: np.ndarray,
        scale_ref: float,
        pos_rel_scale: float=_DEFAULT_POS_P95_REL_THRESH,
        rot_deg_scale: float=_DEFAULT_ROT_P95_DEG_THRESH,
    ) -> np.ndarray:
        """构造 ref 与 cand 之间的两两匹配代价矩阵。

        位置部分按 ref 轨迹半径归一化；旋转部分取 forward / up 夹角的最大值。
        ``*_scale`` 用于把两类残差缩到同一尺度后求和，默认值与升级判定阈值一致。
        """
        pos_rel = (
            np.linalg.norm(
                cand_pos_aligned[:, None, :] - ref_pos[None, :, :], axis=2,
            )
            / max(scale_ref, 1e-6)
        )
        forward_deg = CameraChecker.angleMatrixDeg(cand_dirs_rotated, ref_dirs)
        up_deg = CameraChecker.angleMatrixDeg(cand_up_dirs_rotated, ref_up_dirs)
        rot_deg = np.maximum(forward_deg, up_deg)
        return (
            pos_rel / max(pos_rel_scale, 1e-6)
            + rot_deg / max(rot_deg_scale, 1e-6)
        )

    @staticmethod
    def matchCameraListsByGeometry(
        ref_list: List[Camera],
        cand_list: List[Camera],
        max_iter: int=_DEFAULT_MATCH_MAX_ITER,
        pos_rel_scale: float=_DEFAULT_POS_P95_REL_THRESH,
        rot_deg_scale: float=_DEFAULT_ROT_P95_DEG_THRESH,
    ) -> Tuple[
        Optional[List[Camera]],
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        bool,
        int,
    ]:
        """通过几何代价迭代匹配两个等长 camera list。

        步骤：先按 ``image_id`` 给出初始顺序，再反复用各向异性对齐 + 匈牙利
        分配收敛到稳定排列。

        返回 ``(matched_cand_list, (A, t, R), used_image_id_order, changed_count)``。
        若两边长度不一致返回 ``(None, None, False, 0)``。
        """
        if len(ref_list) != len(cand_list):
            return None, None, False, 0

        cand_ordered, used_image_id_order = CameraChecker.imageIdInitialOrder(
            ref_list,
            cand_list,
        )

        ref_pos = CameraChecker.cameraPositions(ref_list)
        ref_dirs = CameraChecker.cameraForwardDirections(ref_list)
        ref_up_dirs = CameraChecker.cameraUpDirections(ref_list)
        cand_pos_all = CameraChecker.cameraPositions(cand_ordered)
        cand_dirs_all = CameraChecker.cameraForwardDirections(cand_ordered)
        cand_up_dirs_all = CameraChecker.cameraUpDirections(cand_ordered)
        scale_ref = CameraChecker.trajectoryRadius(ref_pos)

        cand_order = np.arange(len(cand_ordered), dtype=np.int64)
        changed_count = 0

        for _ in range(max_iter):
            A, t, R = CameraChecker.fitAnisotropicAlignment(
                cand_pos_all[cand_order],
                ref_pos,
            )
            cand_pos_aligned = cand_pos_all @ A + t
            cand_dirs_rotated = CameraChecker.rotateDirections(cand_dirs_all, R)
            cand_up_dirs_rotated = CameraChecker.rotateDirections(cand_up_dirs_all, R)
            cost = CameraChecker.cameraMatchCost(
                ref_pos,
                ref_dirs,
                ref_up_dirs,
                cand_pos_aligned,
                cand_dirs_rotated,
                cand_up_dirs_rotated,
                scale_ref,
                pos_rel_scale=pos_rel_scale,
                rot_deg_scale=rot_deg_scale,
            )

            cand_indices, ref_indices = CameraChecker.minimumCostAssignment(cost)
            new_cand_order = np.empty_like(cand_order)
            for cand_idx, ref_idx in zip(cand_indices, ref_indices):
                new_cand_order[ref_idx] = cand_idx

            if np.array_equal(new_cand_order, cand_order):
                break

            changed_count = int(np.count_nonzero(new_cand_order != cand_order))
            cand_order = new_cand_order

        A, t, R = CameraChecker.fitAnisotropicAlignment(
            cand_pos_all[cand_order],
            ref_pos,
        )
        matched_cand_list = [cand_ordered[i] for i in cand_order]
        return matched_cand_list, (A, t, R), used_image_id_order, changed_count

    # ------------------------------------------------------------------
    # 面向业务的检查入口
    # ------------------------------------------------------------------

    @staticmethod
    def checkCameraForwardDirectionSpread(
        camera_list: List[Camera],
        min_spread_deg: float=45.0,
        min_camera_num: int=2,
        log_prefix: str='CameraChecker',
    ) -> Tuple[bool, float]:
        """校验相机 forward 方向分布是否足够大。

        用最大两两夹角衡量分布广度，过小通常说明相机位姿估计退化。

        Args:
            camera_list: 任意可迭代的相机集合，每个相机需要提供 ``forwardDirection``。
            min_spread_deg: 允许的最大夹角下限（不含）。
            min_camera_num: 计算夹角所需的最小相机数量。
            log_prefix: 打印日志时使用的模块前缀。

        Returns:
            ``(is_valid, max_angle_deg)``：相机数量不足或最大夹角不大于
            ``min_spread_deg`` 时返回 ``(False, max_angle_deg)``，否则
            返回 ``(True, max_angle_deg)``。
        """
        camera_num = len(camera_list)
        if camera_num < min_camera_num:
            print('[ERROR][' + log_prefix + '::checkCameraForwardDirectionSpread]')
            print('\t camera num too small to estimate poses!')
            print('\t camera_num:', camera_num)
            return False, 0.0

        dirs = CameraChecker.cameraForwardDirections(camera_list)
        max_angle_deg = CameraChecker.directionMaxAngleDeg(dirs)

        if max_angle_deg <= min_spread_deg:
            print('[ERROR][' + log_prefix + '::checkCameraForwardDirectionSpread]')
            print('\t camera forward direction spread too small,')
            print('\t treat as camera pose estimation failure!')
            print('\t max_angle_deg:', max_angle_deg)
            print('\t min_spread_deg:', min_spread_deg)
            return False, max_angle_deg

        return True, max_angle_deg

    @staticmethod
    def checkCameraListUpgrade(
        ref_list: List[Camera],
        cand_list: List[Camera],
        pos_median_rel_thresh: float=_DEFAULT_POS_MEDIAN_REL_THRESH,
        pos_p95_rel_thresh: float=_DEFAULT_POS_P95_REL_THRESH,
        rot_median_deg_thresh: float=_DEFAULT_ROT_MEDIAN_DEG_THRESH,
        rot_p95_deg_thresh: float=_DEFAULT_ROT_P95_DEG_THRESH,
        rot_max_deg_thresh: float=_DEFAULT_ROT_MAX_DEG_THRESH,
        cand_radius_min_ratio: float=_DEFAULT_CAND_RADIUS_MIN_RATIO,
        min_direction_spread_deg: float=_DEFAULT_MIN_DIRECTION_SPREAD_DEG,
        direction_spread_ratio: float=_DEFAULT_DIRECTION_SPREAD_RATIO,
        match_max_iter: int=_DEFAULT_MATCH_MAX_ITER,
        min_camera_num: int=3,
        log_prefix: str='CameraChecker',
    ) -> Tuple[bool, List[Camera]]:
        """判断 ``cand_list`` 是否是 ``ref_list`` 的可信"升级"候选。

        典型用法：``ref_list`` 是稳定参考（如 VGGT），``cand_list`` 是潜在
        更优但可能退化的结果（如 VGGT-Omega）。只有当 cand 自身没有退化、
        且在旋转 + 平移 + 各向异性缩放意义下能很好对齐到 ref 时，才返回 True。

        返回 ``(is_valid_upgrade, matched_cand_list)``。即便升级被拒绝，
        也尽量返回按几何匹配后排好序的 cand_list，方便调用方做后续选择。
        """
        if len(ref_list) < min_camera_num or len(cand_list) < min_camera_num:
            return False, list(cand_list)
        if len(ref_list) != len(cand_list):
            return False, list(cand_list)

        ref_pos = CameraChecker.cameraPositions(ref_list)
        cand_pos = CameraChecker.cameraPositions(cand_list)

        ref_dirs = CameraChecker.cameraForwardDirections(ref_list)
        cand_dirs = CameraChecker.cameraForwardDirections(cand_list)
        ref_up_dirs = CameraChecker.cameraUpDirections(ref_list)

        ref_radius = CameraChecker.trajectoryRadius(ref_pos)
        cand_radius = CameraChecker.trajectoryRadius(cand_pos)
        ref_dir_spread_deg = CameraChecker.directionMaxAngleDeg(ref_dirs)
        cand_dir_spread_deg = CameraChecker.directionMaxAngleDeg(cand_dirs)

        print('[INFO][' + log_prefix + '::checkCameraListUpgrade]')
        print('\t ref_radius:', ref_radius)
        print('\t cand_radius:', cand_radius)
        print('\t ref_direction_spread_deg:', ref_dir_spread_deg)
        print('\t cand_direction_spread_deg:', cand_dir_spread_deg)

        # 退化 1：cand 相机位置塌缩到一起。
        if cand_radius <= 1e-8:
            print('\t reject upgrade: camera positions collapsed.')
            return False, list(cand_list)
        if ref_radius > 1e-8:
            radius_ratio = cand_radius / ref_radius
            if radius_ratio < cand_radius_min_ratio:
                print('\t reject upgrade: camera radius is too small.')
                print('\t radius_ratio:', radius_ratio)
                return False, list(cand_list)

        # 退化 2：cand 相机朝向近似一致，但 ref 是宽视角观察。
        ref_has_wide_view = ref_dir_spread_deg >= min_direction_spread_deg
        cand_is_too_uniform = (
            cand_dir_spread_deg < min_direction_spread_deg
            or cand_dir_spread_deg
            < ref_dir_spread_deg * direction_spread_ratio
        )
        if ref_has_wide_view and cand_is_too_uniform:
            print('\t reject upgrade: camera directions are too uniform.')
            return False, list(cand_list)

        try:
            matched_cand_list, alignment, used_image_id_order, changed_count = (
                CameraChecker.matchCameraListsByGeometry(
                    ref_list,
                    cand_list,
                    max_iter=match_max_iter,
                    pos_rel_scale=pos_p95_rel_thresh,
                    rot_deg_scale=rot_p95_deg_thresh,
                )
            )
        except np.linalg.LinAlgError:
            return False, list(cand_list)

        if matched_cand_list is None or alignment is None:
            return False, list(cand_list)

        print('[INFO][' + log_prefix + '::checkCameraListUpgrade]')
        print('\t initial_match:', 'image_id' if used_image_id_order else 'list_order')
        print('\t geometry_match_changed:', changed_count)

        A, t, R = alignment
        cand_pos_matched = CameraChecker.cameraPositions(matched_cand_list)
        cand_dirs_matched = CameraChecker.cameraForwardDirections(matched_cand_list)
        cand_up_dirs_matched = CameraChecker.cameraUpDirections(matched_cand_list)

        aligned_cand = cand_pos_matched @ A + t
        pos_residual = np.linalg.norm(aligned_cand - ref_pos, axis=1)
        scale_ref = max(ref_radius, 1e-6)

        pos_rel = pos_residual / scale_ref
        pos_median_rel = float(np.median(pos_rel))
        pos_p95_rel = float(np.quantile(pos_rel, 0.95))

        rotated_cand_dirs = CameraChecker.rotateDirections(cand_dirs_matched, R)
        rotated_cand_up_dirs = CameraChecker.rotateDirections(cand_up_dirs_matched, R)
        forward_ang_deg = CameraChecker.pairedAngleDeg(rotated_cand_dirs, ref_dirs)
        up_ang_deg = CameraChecker.pairedAngleDeg(rotated_cand_up_dirs, ref_up_dirs)
        ang_deg = np.maximum(forward_ang_deg, up_ang_deg)
        rot_median_deg = float(np.median(ang_deg))
        rot_p95_deg = float(np.quantile(ang_deg, 0.95))
        rot_max_deg = float(np.max(ang_deg))

        print('[INFO][' + log_prefix + '::checkCameraListUpgrade]')
        print('\t pos_median_rel:', pos_median_rel)
        print('\t pos_p95_rel:', pos_p95_rel)
        print('\t rot_median_deg:', rot_median_deg)
        print('\t rot_p95_deg:', rot_p95_deg)
        print('\t rot_max_deg:', rot_max_deg)

        if pos_median_rel > pos_median_rel_thresh:
            return False, matched_cand_list
        if pos_p95_rel > pos_p95_rel_thresh:
            return False, matched_cand_list
        if rot_median_deg > rot_median_deg_thresh:
            return False, matched_cand_list
        if rot_p95_deg > rot_p95_deg_thresh:
            return False, matched_cand_list
        if rot_max_deg > rot_max_deg_thresh:
            return False, matched_cand_list
        return True, matched_cand_list
