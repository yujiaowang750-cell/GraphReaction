"""七相均值各 64 维，加 6 维坐标范围：454 -> 64。"""

import torch
from torch import nn

from model_config import HIDDEN_DIM, LAYER_NORM_EPS


class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7 * HIDDEN_DIM + 6, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
        )

    def forward(self, node_hidden, phase_nodes, global_geometry):
        """[N,64] + 七相节点索引 + [6] -> 整图向量 [64]。

        同一编码器调用两次：处理器前得到固定上下文 g0，处理器后得到解码用 g1。
        """
        phase_means = []
        for phase in range(1, 8):
            nodes = phase_nodes[phase]
            if len(nodes) == 0:
                raise ValueError(f"缺少 phase {phase}，无法分相均值池化。")
            phase_means.append(node_hidden[nodes].mean(dim=0))
        global_input = torch.cat(phase_means + [global_geometry], dim=0)
        return self.mlp(global_input)
