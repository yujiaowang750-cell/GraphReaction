"""节点与反应单元之间的差值消息，分别按各自的反应关联数取均值。"""

import torch

from aggregation import mean_by_index


def reaction_messages(node_hidden, reaction_hidden, rate, data, update_reaction=True):
    """返回节点反应量 [N,64]、可选单元反应量 [R,64]、关联消息 q [L,64]。

    对一条成员记录：q = a * (该反应单元状态 - 该节点状态)。
    节点接收 q 的均值；反应单元接收 -q 的均值。两者都读取更新前的状态。
    update_reaction=False 时仍计算 q 和节点均值，单元反应量返回 None。
    """
    if len(reaction_hidden) == 0:
        unit_message = torch.zeros_like(reaction_hidden) if update_reaction else None
        return torch.zeros_like(node_hidden), unit_message, node_hidden.new_zeros((0, node_hidden.shape[1]))
    q = rate * (reaction_hidden[data.reaction_index] - node_hidden[data.reaction_node])
    node_message, _ = mean_by_index(q, data.reaction_node, len(node_hidden))
    unit_message = None
    if update_reaction:
        unit_message, _ = mean_by_index(-q, data.reaction_index, len(reaction_hidden))
    return node_message, unit_message, q
