"""总模型只组织调用：节点编码 -> g0 -> 四次同步更新 -> g1 -> 分相解码。"""

from torch import nn

from node_encoder import NodeEncoder
from global_encoder import GlobalEncoder
from processor import Processor
from decoder import PhaseDecoder
from model_config import HIDDEN_DIM
from schema import validate_features


class GraphReactionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_encoder = NodeEncoder()  # H0
        self.global_encoder = GlobalEncoder()  # g0
        self.processor = Processor()  # H4、Z3、rate
        self.decoder = PhaseDecoder()  # 七字段预测

    def forward(self, data, compute_terminal_reaction=False, capture=False):
        """data.x [N,13] -> 原七字段预测；可选保存诊断过程。"""
        validate_features(data)
        hidden = data.x.new_zeros((data.num_nodes, HIDDEN_DIM))
        for phase in range(1, 8):
            nodes = data.phase_nodes[phase]
            # 分相取行以保持参考代码的运算顺序；所有相调用同一个节点编码器。
            hidden = hidden.index_copy(0, nodes, self.node_encoder(data.x[nodes]))
        global_before = self.global_encoder(hidden, data.phase_nodes, data.global_geometry)
        hidden, reaction_hidden, rate, reaction_step, trace = self.processor(
            hidden, global_before, data, compute_terminal_reaction, capture)
        global_after = self.global_encoder(hidden, data.phase_nodes, data.global_geometry)
        prediction = self.decoder(hidden, global_after, data)
        if trace is not None:
            trace["g1"] = global_after
        return {
            "prediction": prediction,
            "node_hidden": hidden,
            "reaction_hidden": reaction_hidden,
            "reaction_rate": rate,
            "reaction_hidden_step": reaction_step,
            "trace": trace,
        }
