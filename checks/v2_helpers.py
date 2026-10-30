"""新版检查共用的小图、旧实现读取和容差。只用于检查，不进入生产模型。"""

from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from compare_reference import reference_definitions, new_name
from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS
from field_validation import FIELD_PHASES
from schema import feature_identity
from regions import assign_node_regions
from graph_topology import build_graph_topology
from reaction_topology import build_reaction_topology
from node_features import build_node_features

KEEP = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24]
# 直接沿用原 compare_reference.py 的预测、损失和梯度容差。
TOLERANCES = dict(value=(2e-6, 2e-5), loss=(1e-9, 2e-4), gradient=(2e-5, 2e-4))


def close(actual, expected, report, key, kind="value", exact=False):
    atol, rtol = (0, 0) if exact else TOLERANCES[kind]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    error = float((actual.detach() - expected.detach()).abs().max()) if actual.numel() else 0.0
    report[key] = max(report.get(key, 0.0), error)


def must_fail(call, text):
    try:
        call()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        assert text in str(error), str(error)
        return
    raise AssertionError(f"应该拒绝而未拒绝：{text}")


def small_graph(ref, mode="overlap"):
    phase = np.array([1, 1, 2, 2, 5, 3, 3, 2, 4, 6, 7, 5])
    coords = np.column_stack([
        [.10, .15, .20, .25, .50, .85, .90, .80, .95, 0., 1., .55],
        np.linspace(.91, .13, 12), np.linspace(.03, .69, 12),
    ])
    bounds = np.array([[0., 100.], [0., 30.], [0., 50.]])
    contacts = [(0, 2, 1), (0, 2, 1), (1, 2, 1), (0, 1, 0), (0, 9, 2),
                (3, 4, 5), (7, 4, 5), (2, 3, 3), (4, 11, 11),
                (5, 7, 4), (6, 7, 4), (5, 6, 6), (5, 8, 7), (5, 10, 8), (8, 10, 10)]
    if mode == "size2":
        contacts = [(0, 2, 1), (0, 2, 1), (5, 7, 4)]
    elif mode == "no_reaction":
        contacts = [(3, 4, 5), (7, 4, 5), (5, 10, 8), (8, 10, 10)]
    elif mode == "no_edges":
        contacts = []
    connections = np.array([(u, v) for u, v, _ in contacts], dtype=np.int64).reshape(-1, 2)
    relations = np.array([r for _, _, r in contacts], dtype=np.int64)
    geometry = np.arange(len(contacts) * 4, dtype=np.float64).reshape(-1, 4) / 13
    region = assign_node_regions(phase, coords, bounds)
    old_topology = ref["build"](phase, connections, relations, geometry, coords, bounds, True)
    fields = {name: torch.tensor(np.flatnonzero(np.isin(phase, phases)))
              for name, phases in zip(FIELD_NAMES, FIELD_PHASES)}
    graph = Data(
        num_nodes=len(phase), node_phase=torch.tensor(phase), node_region=torch.tensor(region),
        phase_nodes={p: torch.tensor(np.flatnonzero(phase == p)) for p in range(1, 8)},
        field_nodes=fields, x_static=torch.tensor(np.column_stack([coords, np.arange(12) - 4.]), dtype=torch.float32),
        global_geometry=torch.tensor([-.2, .3, .1, .7, -.4, 1.2]),
        concentration_scale=torch.tensor([1., 1., 100., 250., 1., 1., 1.], dtype=torch.float64),
        delta_mean=torch.zeros(7, dtype=torch.float64),
        delta_std=torch.tensor([.2, .001, .5, .5, .001, .001, .001], dtype=torch.float64),
        edge_index_u=torch.tensor(connections.T),
        throat_area_um2=torch.full((len(contacts),), .001, dtype=torch.float64),
        throat_conduit_lengths_um=torch.ones((len(contacts), 3), dtype=torch.float64),
        applied_current_A=2.95e-7, sample_id="artificial_7phase", input_time_index=0,
        feature_identity=feature_identity(),
        **build_graph_topology(phase, connections, relations, geometry, region),
        **build_reaction_topology(phase, connections, relations, region),
    )
    stats = {"dynamic": {name: dict(input_mean=2. + j, input_effective_std=.5 + j)
                         for j, name in enumerate(FIELD_NAMES)}}
    current = torch.zeros((len(phase), 2), dtype=torch.float64)
    for j, field in enumerate(FIELD_NAMES):
        nodes = fields[field]
        current[nodes, FIELD_STATE_COLUMNS[j]] = torch.arange(len(nodes), dtype=torch.float64) + j
    # 同时覆盖有效物理值为零、非零物理值变换为零。
    current[fields[FIELD_NAMES[0]][:2], 0] = torch.tensor([0., 2.], dtype=torch.float64)
    current[fields[FIELD_NAMES[1]][:2], 1] = torch.tensor([0., 3.], dtype=torch.float64)
    current[0, 0] = 0.
    graph.x = build_node_features(current, graph, stats)
    future = current + .0001
    future[graph.node_phase == 6, 1] = 0
    return graph, current, future, stats, old_topology


def compare_topology(data, old, report):
    names = dict(edge_metadata="metadata", edge_reverse="reverse_port", edge_group="arc_group",
                 edge_physical_index="physical_rows", node_region="node_region", graph_degree="graph_degree",
                 reaction_node="reaction_node", reaction_index="reaction_edge", reaction_center="reaction_center",
                 reaction_kind="reaction_kind", reaction_region="reaction_region", reaction_role="reaction_role",
                 reaction_support_incidence="reaction_support_incidence",
                 reaction_support_physical_row="reaction_support_physical_row",
                 reaction_degree="reaction_degree", reaction_size="reaction_size")
    for new, original in names.items():
        np.testing.assert_array_equal(data[new].cpu().numpy(), old[original], err_msg=new)
    np.testing.assert_array_equal(data.edge_index.cpu().numpy(), np.stack([old["sender"], old["receiver"]]))
    old_metadata = torch.tensor(old["reaction_metadata"])
    removed = sorted(set(range(30)) - set(KEEP))
    assert torch.count_nonzero(old_metadata[:, removed]) == 0
    close(data.reaction_metadata.cpu(), old_metadata, report, "topology_metadata_max", exact=True)
    for start, stop in ((0, 14), (14, 21), (21, 25)):
        assert torch.all(data.reaction_metadata[:, start:stop].sum(1) == 1)
    np.testing.assert_array_equal(torch.bincount(data.reaction_node, minlength=data.num_nodes).numpy(), old["reaction_degree"])
    np.testing.assert_array_equal(torch.bincount(data.reaction_index, minlength=len(data.reaction_center)).numpy(), old["reaction_size"])
    assert data.reaction_metadata.shape == (len(data.reaction_node), 30)


def explicit_x17_to_x13(old, phase):
    new = old.new_zeros((len(old), 13))
    new[:, :7], new[:, 7:10], new[:, 10] = old[:, 10:17], old[:, 6:9], old[:, 9]
    theta = (phase == 1) | (phase == 3)
    electrolyte = (phase == 2) | (phase == 5)
    potential = torch.isin(phase, phase.new_tensor([2, 3, 4, 5, 7]))
    new[theta, 11] = old[theta, 0]
    new[electrolyte, 11] = old[electrolyte, 1]
    new[potential, 12] = old[potential, 2]
    return new


def original_processor_weights(old_processor):
    """仅在对照检查中按模块名复制完整原处理器参数，不裁剪列或迁移节点编码器。"""
    state = {}
    for name, value in old_processor.state_dict().items():
        state[new_name("processor." + name).removeprefix("processor.")] = value.clone()
    return state
