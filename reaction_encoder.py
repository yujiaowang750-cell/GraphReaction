"""从初始节点状态建立反应单元隐状态 Z0，并计算每条成员记录的固定 rate。"""

import torch
from torch import nn

from aggregation import mean_by_index
from model_config import HIDDEN_DIM, LAYER_NORM_EPS, REACTION_PORT_DIM


class ReactionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.port_encoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM + REACTION_PORT_DIM, HIDDEN_DIM),  # 成员节点64 + 反应属性30 = 94。
            nn.ReLU(), nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
        )
        self.initializer = nn.Sequential(
            nn.Linear(65, HIDDEN_DIM),  # 成员均值 64 + log(1+成员数) 1。
            nn.ReLU(), nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
        )
        # 负极、正极各一个Linear(30,1)；不读取节点动态状态。
        self.rate_layers = nn.ModuleList([nn.Linear(REACTION_PORT_DIM, 1) for _ in range(2)])

    def forward(self, node_hidden, data):
        """返回反应单元状态 [R,64] 和成员 rate [L,1]，四次更新共用这个 rate。"""
        num_units = len(data.reaction_center)
        if data.reaction_metadata.shape != (len(data.reaction_node), REACTION_PORT_DIM):
            raise ValueError("反应属性必须是 [P_R,30]；请用 build_reaction_topology 重建，不能直接复用b12缓存。")
        if num_units == 0:
            return node_hidden.new_zeros((0, HIDDEN_DIM)), node_hidden.new_zeros((0, 1))
        inputs = torch.cat([node_hidden[data.reaction_node], data.reaction_metadata], dim=1)
        ports = self.port_encoder(inputs)  # [L,64]
        pooled, size = mean_by_index(ports, data.reaction_index, num_units)
        initial_state = self.initializer(torch.cat([pooled, torch.log1p(size).unsqueeze(1)], dim=1))

        member_region = data.reaction_region[data.reaction_index]
        logits = ports.new_zeros((len(ports), 1))
        for region, layer in enumerate(self.rate_layers):
            members = torch.where(member_region == region)[0]
            if len(members) > 0:
                logits = logits.index_copy(0, members, layer(data.reaction_metadata[members]))
        return initial_state, torch.sigmoid(logits)
