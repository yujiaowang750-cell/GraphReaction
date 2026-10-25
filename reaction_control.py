"""反应单元的控制 MLP：负极和正极两组，输入单元状态与固定全局上下文。"""

import torch
from torch import nn

from model_config import HIDDEN_DIM, LAYER_NORM_EPS


class ReactionControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, HIDDEN_DIM), nn.ReLU(),
                nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            for _ in range(2)
        ])

    def forward(self, reaction_hidden, global_hidden, reaction_region):
        """[R,64] + [64] -> 反应控制量 U_R [R,64]。"""
        inputs = torch.cat([reaction_hidden, global_hidden.expand(len(reaction_hidden), -1)], dim=1)
        controls = torch.zeros_like(reaction_hidden)
        for region, mlp in enumerate(self.mlps):
            units = torch.where(reaction_region == region)[0]
            if len(units) > 0:
                controls = controls.index_copy(0, units, mlp(inputs[units]))
        return controls
