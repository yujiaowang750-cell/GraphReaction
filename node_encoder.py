"""所有相共用一个节点编码器：17 -> 64，ReLU 后接 LayerNorm。"""

from torch import nn

from model_config import HIDDEN_DIM, LAYER_NORM_EPS


class NodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(17, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM, eps=LAYER_NORM_EPS),
        )

    def forward(self, node_features):
        """[节点数,17] -> [节点数,64]。"""
        return self.mlp(node_features)
