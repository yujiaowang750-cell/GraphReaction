"""每条原始物理边展开成两个相邻方向，并构造每个起点的 30 维说明向量。"""

import numpy as np
import torch

from model_config import RELATION_PHASES, PAIR_PORT_DIM
from regions import assign_edge_groups


def build_graph_topology(phase, connections, relations, geometry, node_region):
    """返回物理图所需的张量字典。

    两个方向按 i->j、j->i 相邻排列；reverse[k] = k ^ 1。
    30 维顺序：类型 one-hot 14 + 起点相 one-hot 7 + 角色 one-hot 4
                + 几何 4 + 几何有效标记 1。
    类型 0～11 是物理关系；12、13 留给反应单元。物理边角色为 0。
    """
    num_nodes, num_edges = len(phase), len(connections)
    if connections.shape != (num_edges, 2) or relations.shape != (num_edges,) or geometry.shape != (num_edges, 4):
        raise ValueError("物理边连接、关系和几何的形状不一致。")
    if not np.issubdtype(connections.dtype, np.integer) or not np.issubdtype(relations.dtype, np.integer):
        raise ValueError("连接和关系编号必须为整数。")
    if not np.isin(phase, range(1, 8)).all() or not np.isin(relations, range(12)).all():
        raise ValueError("相编号应为 1～7，关系编号应为 0～11。")
    if connections.size and (connections.min() < 0 or connections.max() >= num_nodes):
        raise ValueError("边端点超出节点范围。")
    if not np.array_equal(np.sort(phase[connections], axis=1), np.asarray(RELATION_PHASES)[relations]):
        raise ValueError("关系编号与两端节点的相类型不一致。")
    if not np.isfinite(geometry).all():
        raise ValueError("边几何特征包含非有限值。")

    groups = assign_edge_groups(node_region, connections, relations)
    sender = connections.reshape(-1)
    receiver = connections[:, ::-1].reshape(-1)
    # 每条原物理边的两个方向使用相同的关系编号
    directed_relations = np.repeat(relations, 2)
    metadata = np.zeros((2 * num_edges, PAIR_PORT_DIM), dtype=np.float32)
    rows = np.arange(2 * num_edges)
    # 普通物理边使用关系编号0-11，反应单元使用12、13
    metadata[rows, directed_relations] = 1  # 类型 one-hot
    metadata[rows, 14 + phase[sender] - 1] = 1  # 起点相 one-hot
    metadata[:, 21] = 1  # 四种角色中的第 0 种：物理边端点。
    metadata[0::2, 25:29] = geometry
    metadata[1::2, 25:29] = geometry[:, [0, 3, 2, 1]]
    metadata[:, 29] = 1

    return {
        "edge_index": torch.tensor(np.stack([sender, receiver]), dtype=torch.long),
        "edge_metadata": torch.tensor(metadata),             # [2E,30]
        "edge_reverse": torch.tensor(rows ^ 1, dtype=torch.long),
        "edge_group": torch.tensor(np.repeat(groups, 2), dtype=torch.long),
        "edge_physical_index": torch.tensor(np.repeat(np.arange(num_edges), 2)),
        "graph_degree": torch.tensor(np.bincount(receiver, minlength=num_nodes)),
    }
