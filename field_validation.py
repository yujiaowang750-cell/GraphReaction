"""字段是否存在由 phase 决定；错误指出样本、时间、字段和节点。"""

import torch

from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS

FIELD_PHASES = ((2, 5), (2, 5), (1,), (3,), (3,), (4,), (7,))


def field_error(graph, field, reason, node=None, time_index=None):
    sample = getattr(graph, "sample_id", "unknown")
    time = time_index if time_index is not None else getattr(graph, "input_time_index", "unknown")
    return ValueError(f"sample={sample}, time={time}, field={field}, node={node}: {reason}")


def validate_physical_fields(state, graph, time_index=None):
    """检查全部应有字段及节点映射；不检查本模型根本不用的占位槽。"""
    if state.shape != (graph.num_nodes, 2):
        raise field_error(graph, "state", "物理状态应为 [N,2]", time_index=time_index)
    if not torch.isin(graph.node_phase, graph.node_phase.new_tensor(list(range(1, 8)))).all():
        raise field_error(graph, "phase", "相编号必须为 1～7", time_index=time_index)
    for field, phases, column in zip(FIELD_NAMES, FIELD_PHASES, FIELD_STATE_COLUMNS):
        expected = torch.where(torch.isin(graph.node_phase, graph.node_phase.new_tensor(phases)))[0]
        nodes = graph.field_nodes.get(field)
        if nodes is None:
            raise field_error(graph, field, "缺少应有字段的节点映射", time_index=time_index)
        if nodes.ndim != 1 or not torch.equal(nodes.sort().values, expected):
            missing = expected[~torch.isin(expected, nodes)]
            node = int(missing[0]) if len(missing) else None
            raise field_error(graph, field, "字段节点映射缺失、重复或与 phase 不符", node, time_index)
        bad = torch.where(~torch.isfinite(state[nodes, column]))[0]
        if len(bad):
            raise field_error(graph, field, "有效物理字段出现 NaN/Inf", int(nodes[bad[0]]), time_index)
