"""所有相共用一个节点编码器：13 -> 64，ReLU 后接 LayerNorm。"""

from torch import nn

from model_config import HIDDEN_DIM, LAYER_NORM_EPS, NODE_INPUT_DIM


class NodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(NODE_INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
        )

    def forward(self, node_features):
        """[节点数,13] -> [节点数,64]。"""
        if node_features.ndim != 2 or node_features.shape[1] != NODE_INPUT_DIM:
            raise ValueError("新版节点编码器需要 [N,13]；旧 X17/checkpoint 不能直接裁剪或恢复。")
        return self.mlp(node_features)
