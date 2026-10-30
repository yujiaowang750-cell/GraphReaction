"""A/B：单独检查 X13 列映射、字段有效性与原始拓扑。"""

import torch

from v2_helpers import small_graph, close, must_fail, compare_topology, explicit_x17_to_x13
from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS
from node_features import build_node_features
from initialization import create_model


def original_inputs(ref, graph, current, statistics):
    """独立用旧 adapt_phase 生成 X17，再由显式列映射比较。"""
    phase_fields = {1: (2,), 2: (0, 1), 3: (3, 4), 4: (5,), 5: (0, 1), 6: (), 7: (6,)}
    old = torch.zeros((graph.num_nodes, 17))
    for phase, indices in phase_fields.items():
        nodes = graph.phase_nodes[phase]
        columns = []
        for index in indices:
            raw = current[nodes, FIELD_STATE_COLUMNS[index]]
            if index in (2, 3):
                value = raw / graph.concentration_scale[index]
            else:
                stats = statistics["dynamic"][FIELD_NAMES[index]]
                value = (raw - stats["input_mean"]) / stats["input_effective_std"]
            columns.append(value.float().unsqueeze(1))
        inputs = torch.cat([*columns, graph.x_static[nodes]], dim=1)
        old[nodes] = ref["adapt_phase"](inputs, phase)
    return old


def run_inputs_topology(ref, report):
    data, current, _, stats, old_top = small_graph(ref)
    old = original_inputs(ref, data, current, stats)
    close(data.x, explicit_x17_to_x13(old, data.node_phase), report, "input_mapping_max", exact=True)
    close(data.x[:, 7:11], data.x_static, report, "input_static_max", exact=True)
    assert torch.equal(data.x[:, :7].argmax(1) + 1, data.node_phase)
    assert not torch.equal(data.x[:, 7], data.x[:, 8]) and not torch.equal(data.x[:, 8], data.x[:, 9])
    assert data.x[0, 11] == 0  # 有效 theta=0。
    assert data.x[2, 11] != 0 and data.x[3, 11] == 0  # 原 ce=0，与标准化 ce=0 都有效。
    assert data.x[2, 12] != 0 and data.x[3, 12] == 0
    assert not data.x[torch.isin(data.node_phase, torch.tensor([4, 6, 7])), 11].any()
    assert not data.x[torch.isin(data.node_phase, torch.tensor([1, 6])), 12].any()
    # 改样本 cmax 后，只改变对应 theta 输入。
    other = data.clone()
    other.concentration_scale[2:4] *= 2
    alternate = build_node_features(current, other, stats)
    theta = torch.isin(data.node_phase, torch.tensor([1, 3]))
    close(alternate[theta, 11], data.x[theta, 11] / 2, report, "sample_cmax_max", exact=True)
    # 不使用值是否为零判断有效性；无效物理槽的 NaN 不应污染输入。
    invalid = current.clone()
    invalid[data.node_phase == 6] = float("nan")
    invalid[data.node_phase == 1, 1] = float("inf")
    close(build_node_features(invalid, data, stats), data.x, report, "invalid_slots_max", exact=True)
    for index, field in enumerate(FIELD_NAMES):
        for value in (float("nan"), float("inf"), -float("inf")):
            broken = current.clone()
            node = int(data.field_nodes[field][0])
            broken[node, FIELD_STATE_COLUMNS[index]] = value
            must_fail(lambda: build_node_features(broken, data, stats), f"field={field}, node={node}")
        missing = data.clone()
        missing.field_nodes[field] = missing.field_nodes[field][1:]
        must_fail(lambda: build_node_features(current, missing, stats), f"field={field}")
        missing = data.clone()
        del missing.field_nodes[field]
        must_fail(lambda: build_node_features(current, missing, stats), f"field={field}")
    bad_stats = {"dynamic": dict(stats["dynamic"])}
    del bad_stats["dynamic"][FIELD_NAMES[0]]
    must_fail(lambda: build_node_features(current, data, bad_stats), f"field={FIELD_NAMES[0]}")
    model = create_model()
    seen = []
    hook = model.node_encoder.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
    port_inputs = dict(pair=[], reaction_port=[], rate=[])
    def record(name):
        return lambda module, args: port_inputs[name].append(tuple(args[0].shape))
    port_hooks = [mlp[0].register_forward_pre_hook(record("pair")) for mlp in model.processor.graph_messages.mlps]
    port_hooks.append(model.processor.reaction_encoder.port_encoder[0].register_forward_pre_hook(record("reaction_port")))
    port_hooks.extend(layer.register_forward_pre_hook(record("rate")) for layer in model.processor.reaction_encoder.rate_layers)
    with torch.no_grad():
        result = model(data)
    hook.remove()
    for port_hook in port_hooks:
        port_hook.remove()
    for name, width in (("pair", 188), ("reaction_port", 94), ("rate", 30)):
        assert port_inputs[name] and all(shape[1] == width for shape in port_inputs[name])
    report["actual_port_input_shapes"] = port_inputs
    assert sum(shape[0] for shape in seen) == data.num_nodes and all(shape[1] == 13 for shape in seen)
    assert result["node_hidden"].shape == (data.num_nodes, 64)
    report["encoder_actual_input_shapes"] = seen
    compare_topology(data, old_top, report)
    roles = data.reaction_role[data.reaction_node == 0]
    assert 1 in roles and 3 in roles  # 同一个节点在不同反应单元分别为中心与电子成员。
    assert (data.reaction_degree > 1).any()
    # 平行行保留：两个 0->2 物理接触支撑同一条 ION 关联。
    support = data.reaction_support_physical_row
    assert 0 in support and 1 in support
    assert len(data.edge_physical_index) == 2 * data.edge_index_u.shape[1]
    assert torch.equal(data.reaction_metadata[:, :14].argmax(1), data.reaction_kind[data.reaction_index])
    assert torch.equal(data.reaction_metadata[:, 14:21].argmax(1) + 1, data.node_phase[data.reaction_node])
    assert torch.equal(data.reaction_metadata[:, 21:25].argmax(1), data.reaction_role)
    assert not data.reaction_metadata[:, :12].any() and not data.reaction_metadata[:, 25:].any()
    assert torch.all(data.edge_metadata[:, 21] == 1) and torch.all(data.edge_metadata[:, 29] == 1)
    # 几何值为0仍是有效物理端口；反应端口的几何及有效标记均为0。
    assert data.edge_metadata[0, 25] == 0 and data.edge_metadata[0, 29] == 1
    close(data.edge_metadata[1::2, 25:29], data.edge_metadata[0::2, 25:29][:, [0,3,2,1]],
          report, "reverse_geometry_max", exact=True)
    for mode in ("size2", "no_reaction", "no_edges"):
        graph, _, _, _, old_top = small_graph(ref, mode)
        compare_topology(graph, old_top, report)
        if mode == "size2":
            assert (graph.reaction_size == 2).all() and not (graph.reaction_role == 3).any()
        else:
            assert graph.reaction_metadata.shape == (0, 30)
        with torch.no_grad():
            output = model(graph)
        assert torch.isfinite(output["node_hidden"]).all()
        assert all(torch.isfinite(value).all() for value in output["prediction"].values())
    report["A_X13"] = "PASS"
    report["B_topology"] = "PASS: overlap, roles, parallel edges, size2, zero reaction, zero edges"
