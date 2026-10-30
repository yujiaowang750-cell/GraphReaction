"""以石墨或 AM 节点为中心，组织 CENTER、ION、ELECTRON 三类反应成员。"""

import numpy as np
import torch

from model_config import REACTION_PORT_DIM


def build_reaction_topology(phase, connections, relations, node_region):
    """一条成员记录表示“某节点属于某反应单元”，不是额外的物理边。

    石墨中心：ION 来自关系 1；ELECTRON 来自关系 0、2。
    AM 中心：ION 来自关系 4；ELECTRON 来自关系 6、7、8。
    没有 ION 的中心不建立单元；没有 ELECTRON 仍保留。
    同一成员的平行物理连接只生成一条成员记录，原物理图仍保留所有连接。
    """

    # 找出所有的反应中心
    centers = np.flatnonzero(np.isin(phase, [1, 3]))
    # 构建每个反应中心的邻居列表
    neighbors = {int(center): [] for center in centers}
    # 只有端点恰好是反应中心时，才记录邻居
    for row, ((u, v), relation) in enumerate(zip(connections, relations)):
        u, v, relation = int(u), int(v), int(relation)
        if u in neighbors:
            neighbors[u].append((v, relation, row))
        if v in neighbors:
            neighbors[v].append((u, relation, row))

    member_nodes, member_units, member_roles = [], [], []
    support_incidence, support_physical_row = [], []
    unit_centers, unit_kinds, unit_regions = [], [], []
    for center in centers:
        center = int(center)
        if phase[center] == 1:
            ion_relation, electron_relations = 1, {0: 1, 2: 6}
            kind = 12
        else:
            ion_relation, electron_relations = 4, {6: 3, 7: 4, 8: 7}
            kind = 13
        ions, electrons = set(), set()
        for node, relation, _ in neighbors[center]:
            if relation == ion_relation and phase[node] == 2:
                ions.add(node)
            if relation in electron_relations and phase[node] == electron_relations[relation]:
                electrons.add(node)
        if not ions:
            continue
        if ions & electrons:
            raise ValueError("同一成员不能同时属于 ION 和 ELECTRON 角色。")
        unit = len(unit_centers)
        unit_centers.append(center)
        unit_kinds.append(kind)
        unit_regions.append(0 if node_region[center] == 0 else 1)  # 反应分支只分负极/正极。
        for role, nodes in ((1, [center]), (2, sorted(ions)), (3, sorted(electrons))):
            for node in nodes:
                if node_region[center] not in (0, 2) or node_region[node] != node_region[center]:
                    raise ValueError("反应成员不能跨区域，也不能属于隔膜。")
                # 支撑行仅用于核查：反应成员去重，但平行物理行全部保留。
                if role != 1:
                    allowed = {ion_relation} if role == 2 else set(electron_relations)
                    rows = sorted({row for neighbor, relation, row in neighbors[center]
                                   if neighbor == node and relation in allowed})
                    support_incidence.extend([len(member_nodes)] * len(rows))
                    support_physical_row.extend(rows)
                member_nodes.append(node)
                member_units.append(unit)
                member_roles.append(role)

    node = np.asarray(member_nodes, dtype=np.int64)
    unit = np.asarray(member_units, dtype=np.int64)
    role = np.asarray(member_roles, dtype=np.int64)
    kinds = np.asarray(unit_kinds, dtype=np.int64)
    metadata = np.zeros((len(node), REACTION_PORT_DIM), dtype=np.float32)
    rows = np.arange(len(node))
    # 与普通图共用原始30列格式，但每条记录仍属于特定的节点—反应单元关联。
    metadata[rows, kinds[unit]] = 1          # 类型0:14，反应使用12/13。
    metadata[rows, 14 + phase[node] - 1] = 1 # phase14:21，使用当前成员的相。
    metadata[rows, 21 + role] = 1           # 角色21:25，反应使用1/2/3。
    # 反应关联没有单一喉道几何：25:29以及geometry_valid第29列保持0。
    return {
        "reaction_node": torch.tensor(node),                 # [L] 成员的全图节点编号
        "reaction_index": torch.tensor(unit),                # [L] 成员所属反应单元编号
        "reaction_role": torch.tensor(role),                 # [L] 1/2/3
        "reaction_metadata": torch.tensor(metadata),         # [L,30]，空图也保留二维
        "reaction_center": torch.tensor(unit_centers, dtype=torch.long),  # [R]
        "reaction_kind": torch.tensor(kinds),                # [R] 12/13
        "reaction_region": torch.tensor(unit_regions, dtype=torch.long), # [R] 0/1
        "reaction_support_incidence": torch.tensor(support_incidence, dtype=torch.long),
        "reaction_support_physical_row": torch.tensor(support_physical_row, dtype=torch.long),
        "reaction_degree": torch.tensor(np.bincount(node, minlength=len(phase))),
        "reaction_size": torch.tensor(np.bincount(unit, minlength=len(kinds))),
    }
