"""根据静态相编号和 z 坐标，划分三个区域和五类物理边。"""

import numpy as np


def assign_node_regions(phase, coordinates, bounds):
    """返回 [N]：0=负极，1=隔膜，2=正极。

    phase 2 的电解液可能位于负极或正极，需要与 phase 5 的 z 范围比较。
    使用原始 float64 坐标，以保留参考代码对边界的判定精度。
    """
    if coordinates.dtype != np.float64 or coordinates.shape != (len(phase), 3):
        raise ValueError("区域判定需要 [N,3] 的 float64 坐标。")
    if bounds.shape != (3, 2) or not np.all(bounds[:, 1] > bounds[:, 0]):
        raise ValueError("物理坐标范围应为有效的 [3,2] 数组。")
    if not np.isfinite(coordinates).all() or not np.isfinite(bounds).all():
        raise ValueError("坐标和范围必须有限。")
    z = coordinates[:, 0]
    separator_z = z[phase == 5]
    if len(separator_z) == 0:
        raise ValueError("缺少隔膜节点，无法划分三个区域。")
    tolerance = 8 * np.finfo(coordinates.dtype).eps * max(1.0, float(np.abs(z).max()))
    low, high = separator_z.min(), separator_z.max()

    region = np.full(len(phase), -1, dtype=np.int64)
    region[np.isin(phase, [1, 6])] = 0
    region[phase == 5] = 1
    region[np.isin(phase, [3, 4, 7])] = 2
    region[(phase == 2) & (z < low - tolerance)] = 0
    region[(phase == 2) & (z > high + tolerance)] = 2
    if (region < 0).any():
        raise ValueError("存在不能按参考规则划分的节点，例如位于隔膜 z 范围内的 phase 2。")
    return region


def assign_edge_groups(node_region, connections, relations):
    """返回 [E]：负极内部、隔膜内部、正极内部、负极-隔膜、隔膜-正极。"""
    ends = np.sort(node_region[connections], axis=1)
    groups = np.full(len(connections), -1, dtype=np.int64)
    same_region = ends[:, 0] == ends[:, 1]
    groups[same_region] = ends[same_region, 0]
    groups[(ends[:, 0] == 0) & (ends[:, 1] == 1)] = 3
    groups[(ends[:, 0] == 1) & (ends[:, 1] == 2)] = 4
    if (groups < 0).any():
        raise ValueError("存在参考模型不支持的负极到正极直接连接。")

    expected_groups = {0: 0, 1: 0, 2: 0, 4: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 1}
    for relation, expected in expected_groups.items():
        if not np.all(groups[relations == relation] == expected):
            raise ValueError(f"关系 {relation} 与区域分组不一致。")
    if not np.isin(groups[relations == 3], [0, 2]).all():
        raise ValueError("电极电解液内部边必须属于负极或正极。")
    if not np.isin(groups[relations == 5], [3, 4]).all():
        raise ValueError("电极-隔膜电解液边必须属于两个界面之一。")
    return groups
