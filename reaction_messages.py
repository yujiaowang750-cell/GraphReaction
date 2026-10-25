"""节点与反应单元之间的差值消息，分别按各自的反应关联数取均值。"""

import torch

from aggregation import mean_by_index


def reaction_messages(node_hidden, reaction_hidden, rate, data):
    """H [N,64]、Z [R,64]、a [L,1] -> 节点反应量 [N,64]、单元反应量 [R,64]。

    对一条成员记录：q = a * (该反应单元状态 - 该节点状态)。
    节点接收 q 的均值；反应单元接收 -q 的均值。两者都读取更新前的状态。
    """
    if len(reaction_hidden) == 0:
        return torch.zeros_like(node_hidden), torch.zeros_like(reaction_hidden)
    q = rate * (reaction_hidden[data.reaction_index] - node_hidden[data.reaction_node])
    node_message, _ = mean_by_index(q, data.reaction_node, len(node_hidden))
    unit_message, _ = mean_by_index(-q, data.reaction_index, len(reaction_hidden))
    return node_message, unit_message
