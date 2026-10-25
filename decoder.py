"""六个独立线性解码头；每个头读取最终节点状态和最终全局上下文。"""

import torch
from torch import nn
from torch.nn import functional as F

from model_config import DECODER_DIMS, FIELD_NAMES


class PhaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({str(phase): nn.Linear(128, size) for phase, size in DECODER_DIMS.items()})

    def forward(self, node_hidden, global_hidden, data):
        """[N,64] + g1 [64] -> 七字段字典，各字段是 [对应节点数] 的标准化增量。

        phase 1 输出 theta；phase 2/5 输出 ce、phi；phase 3 输出 theta、phi；
        phase 4/7 只输出 phi；phase 6 不设预测头。
        """
        outputs = node_hidden.new_zeros((len(node_hidden), 2))
        for phase in DECODER_DIMS:
            nodes = data.phase_nodes[phase]
            inputs = torch.cat([node_hidden[nodes], global_hidden.expand(len(nodes), -1)], dim=1)
            values = self.heads[str(phase)](inputs)
            outputs = outputs.index_copy(0, nodes, F.pad(values, (0, 2 - values.shape[1])))
        # outputs 是临时解码槽位：单字段头的电势位于第 0 列，不是原始状态的列定义。
        field_columns = (0, 1, 0, 0, 1, 0, 0)
        return {field: outputs[data.field_nodes[field], column] for field, column in zip(FIELD_NAMES, field_columns)}
