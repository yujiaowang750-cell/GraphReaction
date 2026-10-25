"""标准化增量标签，以及预测增量到原物理状态的恢复。"""

import torch

from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS


def build_delta_targets(current_state, future_state, data):
    """返回七字段字典，每项 [该字段节点数]，保留 float64 标签精度。"""
    targets = {}
    for index, field in enumerate(FIELD_NAMES):
        nodes = data.field_nodes[field]
        column = FIELD_STATE_COLUMNS[index]
        difference = future_state[nodes, column] - current_state[nodes, column]
        targets[field] = (difference - data.delta_mean[index]) / data.delta_std[index]
    return targets


def recover_physical_state(current_state, predicted_delta, data):
    """七字段增量 -> 未来 [N,2] float64，以及 [N,2] 物理字段有效 mask。

    data.delta_mean/std 已将石墨和 AM 的增量尺度乘 cmax，因此结果直接是浓度。
    phase 6 电势固定 0 V；石墨电势不预测，返回值中该位置为无效占位零。
    不对预测浓度或电势作硬截断。
    """
    future = torch.zeros_like(current_state)
    mask = torch.zeros_like(current_state, dtype=torch.bool)
    for index, field in enumerate(FIELD_NAMES):
        nodes = data.field_nodes[field]
        column = FIELD_STATE_COLUMNS[index]
        future[nodes, column] = (
            current_state[nodes, column] + predicted_delta[field].double() * data.delta_std[index]
            + data.delta_mean[index]
        )
        mask[nodes, column] = True
    mask[data.node_phase == 6, 1] = True
    return future, mask
