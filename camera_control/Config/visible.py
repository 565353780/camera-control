VISIBLE_LABEL_UNKNOWN: int = -1
VISIBLE_LABEL_FREE: int = 0
VISIBLE_LABEL_VALID: int = 1

# FREE_KN：FREE voxel 中到最近 VALID 的 L1 距离恰为 K 的 voxel。
# 编码采用可推广公式 VALID + K，未来 K=3/4 直接用 visibleLabelFreeKN(k)。
VISIBLE_LABEL_FREE_1N: int = 2
VISIBLE_LABEL_FREE_2N: int = 3


def visibleLabelFreeKN(k: int) -> int:
    """返回 FREE_KN 标签编码：VALID + K（K >= 1）。"""
    if k < 1:
        raise ValueError(
            '[ERROR][visible::visibleLabelFreeKN] '
            f'k must be >= 1, got {k}'
        )
    return VISIBLE_LABEL_VALID + k


# 可视化时各标签对应的颜色（RGB，范围 [0, 1]）。
VISIBLE_COLOR_VALID = [0.0, 1.0, 0.0]
VISIBLE_COLOR_UNKNOWN = [0.5, 0.5, 0.5]
VISIBLE_COLOR_FREE_1N = [0.55, 0.75, 1.0]
VISIBLE_COLOR_FREE_2N = [1.0, 0.85, 0.55]
