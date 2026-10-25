"""五组物理边各自计算消息，再按每个节点的全部物理入边数求均值。"""

import torch
from torch import nn

from aggregation import mean_by_index
from model_config import HIDDEN_DIM, LAYER_NORM_EPS, EDGE_GROUP_NAMES


class GraphMessages(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(188, HIDDEN_DIM),  # 两端节点 64+64，两端说明向量 30+30。
                nn.ReLU(),
                nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            for _ in EDGE_GROUP_NAMES
        ])

    def forward(self, node_hidden, data):
        """[N,64] + 有向物理图 -> 每个节点的物理图消息 [N,64]。"""
        sender, receiver = data.edge_index
        message_input = torch.cat([
            node_hidden[sender], node_hidden[receiver],
            data.edge_metadata, data.edge_metadata[data.edge_reverse],
        ], dim=1)  # [2E,188]
        messages = node_hidden.new_zeros((len(sender), HIDDEN_DIM))
        for group, mlp in enumerate(self.mlps):
            edges = torch.where(data.edge_group == group)[0]
            if len(edges) > 0:
                messages = messages.index_copy(0, edges, mlp(message_input[edges]))
        # 五组只决定使用哪个 MLP；聚合时统一除以全部物理入度。
        mean_messages, _ = mean_by_index(messages, receiver, len(node_hidden))
        return mean_messages
