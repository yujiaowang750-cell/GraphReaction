"""四次同步更新：物理图分支、反应分支和控制分支共同更新节点与反应状态。"""

from torch import nn

from graph_messages import GraphMessages
from node_control import NodeControl
from reaction_encoder import ReactionEncoder
from reaction_control import ReactionControl
from reaction_messages import reaction_messages
from model_config import PROCESSOR_STEPS, STEP_SIZE, CONTROL_WEIGHT


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        # 保留参考模型的参数注册顺序，便于核对同种子的初始化结果。
        self.graph_messages = GraphMessages()
        self.node_control = NodeControl()
        self.reaction_encoder = ReactionEncoder()
        self.reaction_control = ReactionControl()

    def forward(self, node_hidden, global_hidden, data):
        """H0 [N,64]、g0 [64] -> H4 [N,64]、Z4 [R,64]、rate [L,1]。

        四次更新共享同一套参数，g0 和 rate 只在循环前计算一次，梯度保持连接。
        四次是网络内部迭代，不代表向未来预测四个时间步。
        """
        reaction_hidden, rate = self.reaction_encoder(node_hidden, data)
        for _ in range(PROCESSOR_STEPS):
            graph_message = self.graph_messages(node_hidden, data)
            node_control = self.node_control(node_hidden, global_hidden, data.node_region)
            node_reaction, unit_reaction = reaction_messages(node_hidden, reaction_hidden, rate, data)
            unit_control = self.reaction_control(reaction_hidden, global_hidden, data.reaction_region)

            # 所有右侧量都来自旧 H、旧 Z；先算两份新值，再一起替换。
            next_nodes = node_hidden + STEP_SIZE * (graph_message + node_reaction + CONTROL_WEIGHT * node_control)
            next_units = reaction_hidden + STEP_SIZE * (unit_reaction + CONTROL_WEIGHT * unit_control)
            node_hidden, reaction_hidden = next_nodes, next_units
        return node_hidden, reaction_hidden, rate
