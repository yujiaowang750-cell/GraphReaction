"""静态节点、边和整图几何特征；统计方式保持参考模型的普通 z-score。"""

import numpy as np
import torch

from normalization import zscore


def build_static_features(raw_graph, statistics):
    """返回 [N,4]：归一化 z、y、x，加标准化体积。

    7 维 phase one-hot 在 node_features.py 中拼接。
    坐标直接使用数据中已归一化的坐标，体积使用训练统计量。
    """
    volume = zscore(raw_graph["node_volume_um3"], statistics["geometry"]["node_volume_um3"])
    features = np.column_stack([raw_graph["node_coords_normalized"], volume])
    return torch.tensor(features, dtype=torch.float32)


def build_edge_geometry(raw_graph, statistics):
    """返回原始物理边的 [E,4]：[面积, 起点段长度, 中间段长度, 终点段长度]。

    两端长度共用一份训练统计量；中间段长度使用自己的统计量。
    """
    geometry = statistics["geometry"]
    lengths = raw_graph["throat_conduit_lengths_um"]
    return np.column_stack([
        zscore(raw_graph["throat_area_um2"], geometry["throat_area_um2"]),
        zscore(lengths[:, 0], geometry["conduit_endpoint_length_um"]),
        zscore(lengths[:, 1], geometry["conduit_throat_length_um"]),
        zscore(lengths[:, 2], geometry["conduit_endpoint_length_um"]),
    ])


def build_global_geometry(raw_graph, statistics):
    """[z_min,z_max,y_min,y_max,x_min,x_max] -> 标准化后的 [6]。"""
    bounds = raw_graph["sample_coordinate_range_zyx_um"].reshape(6)
    result = zscore(bounds, statistics["geometry"]["sample_coordinate_range_zyx_um"])
    return torch.tensor(result, dtype=torch.float32)
