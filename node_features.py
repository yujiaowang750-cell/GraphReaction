"""X13：七相 one-hot、Z/Y/X、体积、浓度槽、电势槽。"""

import numpy as np
import torch
from torch.nn import functional as F

from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS, NODE_INPUT_DIM
from normalization import zscore
from field_validation import validate_physical_fields, field_error


def build_node_features(current_state, graph, statistics):
    """CPU 上将 [N,2] 原物理状态变为 [N,13] float32，节点顺序不变。

    有效字段分别用原统计或样本 cmax 变换；不存在的动态槽保持为零。
    两类浓度只是在不同相共用一列，没有混合统计，也没有按数值判断有效性。
    """
    validate_physical_fields(current_state, graph)
    state = current_state.numpy()  # 原始 float64，浓度 mol/m^3，电势 V。
    values = np.zeros((graph.num_nodes, 2), dtype=np.float64)
    feature_slots = (0, 1, 0, 0, 1, 1, 1)
    for field_index, field in enumerate(FIELD_NAMES):
        nodes = graph.field_nodes[field].numpy()
        raw_values = state[nodes, FIELD_STATE_COLUMNS[field_index]]
        try:
            if field_index in (2, 3):
                scale = float(graph.concentration_scale[field_index])
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("cmax 必须为有限正数")
                transformed = raw_values / scale
            else:
                transformed = zscore(raw_values, statistics["dynamic"][field], "input_")
        except (KeyError, ValueError) as error:
            raise field_error(graph, field, f"输入变换失败：{error}") from error
        bad = np.flatnonzero(~np.isfinite(transformed))
        if len(bad):
            raise field_error(graph, field, "变换后出现 NaN/Inf", int(nodes[bad[0]]))
        slot = feature_slots[field_index]
        values[nodes, slot] = transformed
    phase_one_hot = F.one_hot(graph.node_phase - 1, num_classes=7).float()
    features = torch.cat([phase_one_hot, graph.x_static, torch.tensor(values, dtype=torch.float32)], dim=1)
    if features.shape != (graph.num_nodes, NODE_INPUT_DIM):
        raise ValueError("X13 需要 7 维相编码 + 4 维静态量 + 2 维动态槽。")
    bad = torch.nonzero(~torch.isfinite(features))
    if len(bad):
        node, column = bad[0].tolist()
        raise field_error(graph, f"X13[{column}]", "float32 输入出现 NaN/Inf", node)
    return features
