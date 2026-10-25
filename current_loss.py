"""由原始接触面积、固相端长度和恢复后的电势计算正极集流体电流。"""

import torch

from model_config import AM_CONDUCTIVITY, CBD_CONDUCTIVITY


def predict_collector_current(future_state, data):
    """[N,2] 原物理状态 -> 电流标量 A；每条无向接触只计算一次。"""
    u, v = data.edge_index_u
    phase = data.node_phase
    selected = (phase[u] == 7) | (phase[v] == 7)
    u, v = u[selected], v[selected]
    collector_is_start = phase[u] == 7
    solid = torch.where(collector_is_start, v, u)
    collector = torch.where(collector_is_start, u, v)
    if not torch.all((phase[solid] == 3) | (phase[solid] == 4)):
        raise ValueError("正极集流体只能连接 AM 或 CBD 固相。")
    lengths = data.throat_conduit_lengths_um[selected]
    solid_length = torch.where(collector_is_start, lengths[:, 2], lengths[:, 0])
    if (solid_length <= 0).any():
        raise ValueError("集流体接触的固相端长度必须大于零。")
    area = data.throat_area_um2[selected]
    conductivity = torch.where(phase[solid] == 3, area.new_tensor(AM_CONDUCTIVITY), area.new_tensor(CBD_CONDUCTIVITY))
    conductance = conductivity * area * 1e-12 / (solid_length * 1e-6)
    return (conductance * (future_state[solid, 1] - future_state[collector, 1])).sum()


def compute_current_loss(future_state, data):
    predicted_current = predict_collector_current(future_state, data)
    # 参考模型按当前轨迹的 sample_manifest 读取施加电流。
    applied_current = data.applied_current_A
    loss = ((predicted_current - applied_current) / abs(applied_current)).square()
    return loss, predicted_current
