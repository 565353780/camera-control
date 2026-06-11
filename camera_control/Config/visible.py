# 体素可见性标签编码（第一性原理：用整数符号区分三类语义）：
#   UNKNOWN = 0：无任何信息；
#   VALID   = 1：候选占据且至少一个视角可见；
#   FREE_KN = -K（K >= 1）：自由空间，K 为到最近非 FREE voxel
#   （VALID 或 UNKNOWN，二者同为非自由边界）的 L1 距离。
# 当整个空间都是 FREE（无任何 VALID/UNKNOWN）时，距离无定义，统一编码为
# -(3R)（R 为体素分辨率），严格大于网格内最大可能 L1 距离 3(R-1)，
# 即「无限远 FREE」哨兵。
VISIBLE_LABEL_UNKNOWN: int = 0
VISIBLE_LABEL_VALID: int = 1


def visibleLabelFreeKN(k: int) -> int:
    """返回 FREE_KN 标签编码：-K（K >= 1）。"""
    if k < 1:
        raise ValueError(
            '[ERROR][visible::visibleLabelFreeKN] '
            f'k must be >= 1, got {k}'
        )
    return -k


def visibleLabelFreeInf(resolution: int) -> int:
    """返回「无限远 FREE」哨兵编码：-(3R)，用于场景中无任何 VALID voxel。"""
    if resolution <= 0:
        raise ValueError(
            '[ERROR][visible::visibleLabelFreeInf] '
            f'resolution must be positive, got {resolution}'
        )
    return -3 * resolution


# 可视化时各标签对应的颜色（RGB，范围 [0, 1]）。
VISIBLE_COLOR_VALID = [0.0, 1.0, 0.0]
VISIBLE_COLOR_UNKNOWN = [0.5, 0.5, 0.5]
# FREE_KN 渐变端点：K=1 为蓝色，K=K_max 为红色，中间线性插值。
VISIBLE_COLOR_FREE_NEAR = [0.0, 0.0, 1.0]
VISIBLE_COLOR_FREE_FAR = [1.0, 0.0, 0.0]
