"""负极、隔膜、正极各有一个控制 MLP，输入节点状态和固定全局上下文。"""

import torch
from torch import nn

from model_config import HIDDEN_DIM, LAYER_NORM_EPS, REGION_NAMES


class NodeControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, HIDDEN_DIM), nn.ReLU(),
                nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            for _ in REGION_NAMES
        ])

    def forward(self, node_hidden, global_hidden, node_region):
        """[N,64] + [64] -> 控制量 U [N,64]。"""
        inputs = torch.cat([node_hidden, global_hidden.expand(len(node_hidden), -1)], dim=1)
        controls = torch.zeros_like(node_hidden)
        for region, mlp in enumerate(self.mlps):
            nodes = torch.where(node_region == region)[0]
            if len(nodes) > 0:
                controls = controls.index_copy(0, nodes, mlp(inputs[nodes]))
        return controls
