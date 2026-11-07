"""共享参数的四轮同步更新：默认得到 H4/Z3，诊断模式才补算 Z4。"""

from torch import nn

from graph_messages import GraphMessages
from node_control import NodeControl
from reaction_encoder import ReactionEncoder
from reaction_control import ReactionControl
from reaction_messages import reaction_messages
from model_config import PROCESSOR_STEPS, STEP_SIZE, CONTROL_WEIGHT
from trace import record_capture, validate_capture


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        # 保留参考模型的参数注册顺序，便于核对同种子的初始化结果。
        self.graph_messages = GraphMessages()
        self.node_control = NodeControl()
        self.reaction_encoder = ReactionEncoder()
        self.reaction_control = ReactionControl()

    def advance(self, node_hidden, reaction_hidden, global_hidden, rate, data, update_reaction=True):
        """右侧全用旧 H/Z；关闭 update_reaction 时不计算 DR、UR 和 Z_next。"""
        graph_message = self.graph_messages(node_hidden, data)
        # 节点控制器计算节点控制信号
        node_control = self.node_control(node_hidden, global_hidden, data.node_region)
        node_reaction, unit_reaction, q = reaction_messages(
            node_hidden, reaction_hidden, rate, data, update_reaction)
        unit_control, next_units = None, None
        if update_reaction:
            # 反应控制器计算反应控制信号
            unit_control = self.reaction_control(reaction_hidden, global_hidden, data.reaction_region)
            # 更新反应隐藏状态
            next_units = reaction_hidden + STEP_SIZE * (unit_reaction + CONTROL_WEIGHT * unit_control)
        # 更新节点隐藏状态
        next_nodes = node_hidden + STEP_SIZE * (graph_message + node_reaction + CONTROL_WEIGHT * node_control)
        parts = dict(Dgraph=graph_message, R=node_reaction, U=node_control,
                     q=q, DR=unit_reaction, UR=unit_control)
        return next_nodes, next_units, parts

    def forward(self, node_hidden, global_hidden, data, compute_terminal_reaction=False, capture=False):
        """Z0、g0、rate 每次前向只计算一次；循环中不 detach，不重建 Z。

        capture 仅保存已计算的张量。compute_terminal_reaction=True 才计算末轮 Z4。
        返回最后可用的 Z 及真实轮次，默认是 3，完整诊断时是 4。
        """
        reaction_hidden, rate = self.reaction_encoder(node_hidden, data)
        # 节点隐藏状态与超边隐藏状态分别记录
        trace = dict(H=[node_hidden], Z=[reaction_hidden], steps=[]) if capture else None
        reaction_step = 0
        # Processor_step是4轮
        for step in range(PROCESSOR_STEPS):
            update_reaction = step < PROCESSOR_STEPS - 1 or compute_terminal_reaction
            next_nodes, next_units, parts = self.advance(
                node_hidden, reaction_hidden, global_hidden, rate, data, update_reaction)
            if capture:
                record_capture(trace, next_nodes, next_units, parts)
            node_hidden = next_nodes
            if next_units is not None:
                reaction_hidden = next_units
                reaction_step = step + 1
        if capture:
            trace.update(a=rate, g0=global_hidden, reaction_hidden_step=reaction_step)
            validate_capture(trace, PROCESSOR_STEPS, compute_terminal_reaction)
        return node_hidden, reaction_hidden, rate, reaction_step, trace
