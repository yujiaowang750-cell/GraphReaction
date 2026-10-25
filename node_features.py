"""把当前物理状态整理为共享节点编码器的 17 维输入。"""

import numpy as np
import torch
from torch.nn import functional as F

from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS
from normalization import zscore


def build_node_features(current_state, graph, statistics):
    """[N,2] 原始状态 + 静态图 -> [N,17]，此函数在 CPU 上进行。

    输入顺序：[theta, ce_norm, phi_norm, 三个有效 mask, z,y,x,volume_norm, 七相 one-hot]。
    theta 仅属于石墨/AM；ce 仅属于电解液；phi 属于电解液/AM/CBD/正集流体。
    石墨电势不属于这套参考模型的输入或预测字段；phase 6 的三个 mask 都为零。
    """
    state = current_state.numpy()  # 原始 float64，浓度 mol/m^3，电势 V。
    values = np.zeros((graph.num_nodes, 3), dtype=np.float64)
    mask = np.zeros_like(values)
    feature_slots = (1, 2, 0, 0, 2, 2, 2)
    for field_index, field in enumerate(FIELD_NAMES):
        nodes = graph.field_nodes[field].numpy()
        raw_values = state[nodes, FIELD_STATE_COLUMNS[field_index]]
        if field_index in (2, 3):
            transformed = raw_values / float(graph.concentration_scale[field_index])
        else:
            transformed = zscore(raw_values, statistics["dynamic"][field], "input_")
        slot = feature_slots[field_index]
        values[nodes, slot] = transformed
        mask[nodes, slot] = 1
    phase_one_hot = F.one_hot(graph.node_phase - 1, num_classes=7).float()
    return torch.cat([
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        graph.x_static, phase_one_hot,
    ], dim=1)
