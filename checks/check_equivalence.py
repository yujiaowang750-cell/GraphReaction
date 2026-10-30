"""C/D：恢复b30后与原处理器对照，再单独检查末轮Z跳过；不混入X13编码器变化。"""

import copy
from unittest.mock import patch

import torch

from v2_helpers import close, small_graph, original_processor_weights, new_name
from initialization import create_model
from loss import compute_loss
from state_output import build_delta_targets
from reaction_messages import reaction_messages


def activate_rates_controls(model):
    """非零参数让测试能发现 rate 映射和前三轮 control 的错误。"""
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(128)
        processor = model.processor
        rates = getattr(processor, "reaction_rates", None)
        if rates is None:
            rates = processor.reaction_encoder.rate_layers
        controls = getattr(processor, "reaction_controls", None)
        if controls is None:
            controls = processor.reaction_control.mlps
        for layer in rates:
            layer.weight.uniform_(-.1, .1)
            layer.bias.uniform_(-.1, .1)
        for block in controls:
            block[-1].weight.uniform_(-.03, .03)
            block[-1].bias.uniform_(-.03, .03)


def compare_predictions_and_loss(first, second, current, target, data, report, prefix):
    for field in first:
        close(first[field], second[field], report, prefix + "_prediction_max")
    loss1 = compute_loss(first, target, current, data)
    loss2 = compute_loss(second, target, current, data)
    for key in ("loss", "reference_loss", "main_loss", "current_loss"):
        close(loss1[key], loss2[key], report, prefix + "_loss_max", "loss")
    return loss1["loss"], loss2["loss"]


def run_original_processor(ref, report):
    data, current, future, _, original_top = small_graph(ref)
    target = build_delta_targets(current, future, data)
    old_model = ref["fresh"](2026, "GraphReaction")
    activate_rates_controls(old_model)
    model = create_model()
    old = old_model.processor
    topology = ref["tensor"](original_top)
    model.processor.load_state_dict(original_processor_weights(old), strict=True)
    model.global_encoder.mlp.load_state_dict(old_model.global_encoder.state_dict())
    model.decoder.heads.load_state_dict(old_model.decoders.state_dict())
    # H0/g0 固定并复制，不经过新旧形状不同的节点编码器。
    torch.manual_seed(117)
    h_old = torch.randn(data.num_nodes, 64, requires_grad=True)
    g_old = torch.randn(64, requires_grad=True)
    h_new = h_old.detach().clone().requires_grad_()
    g_new = g_old.detach().clone().requires_grad_()
    topology.reaction.metadata.requires_grad_()
    data.reaction_metadata = data.reaction_metadata.detach().clone().requires_grad_()
    old_ports = old.reaction_port_encoder(torch.cat([h_old[topology.reaction.node], topology.reaction.metadata], 1))
    new_ports = model.processor.reaction_encoder.port_encoder(torch.cat([h_new[data.reaction_node], data.reaction_metadata], 1))
    close(new_ports, old_ports, report, "reference_port_max")
    old_graph_message, _, _ = old.graph_messages(h_old, topology)
    new_graph_message = model.processor.graph_messages(h_new, data)
    close(new_graph_message, old_graph_message, report, "reference_graph_message_max")
    ho, zo, ao, old_trace, _ = old(h_old, g_old, topology, True, True)
    hn, zn, an, _, trace = model.processor(h_new, g_new, data, compute_terminal_reaction=True, capture=True)
    close(an, ao, report, "reference_rate_max")
    close(trace["Z"][0], old_trace[0][1], report, "reference_Z0_max")
    assert not torch.all(an == .5)
    for (old_h, old_z), new_h, new_z in zip(old_trace, trace["H"], trace["Z"]):
        close(new_h, old_h, report, "reference_H_max")
        close(new_z, old_z, report, "reference_Z_max")
    old_g1 = model.global_encoder(ho, data.phase_nodes, data.global_geometry)
    new_g1 = model.global_encoder(hn, data.phase_nodes, data.global_geometry)
    old_prediction = model.decoder(ho, old_g1, data)
    new_prediction = model.decoder(hn, new_g1, data)
    old_loss, new_loss = compare_predictions_and_loss(old_prediction, new_prediction, current, target, data, report, "reference")
    old_parameters, new_parameters = dict(old.named_parameters()), dict(model.processor.named_parameters())
    readout_parameters = list(model.global_encoder.parameters()) + list(model.decoder.parameters())
    old_grads = torch.autograd.grad(old_loss, [h_old, g_old, topology.reaction.metadata, *old_parameters.values(), *readout_parameters])
    new_grads = torch.autograd.grad(new_loss, [h_new, g_new, data.reaction_metadata, *new_parameters.values(), *readout_parameters])
    for a, b in zip(new_grads[:2], old_grads[:2]):
        close(a, b, report, "reference_input_gradient_max", "gradient")
    close(new_grads[2], old_grads[2], report, "reference_metadata_gradient_max", "gradient")
    new_by_name = dict(zip(new_parameters, new_grads[3:]))
    for (name, _), gradient in zip(old_parameters.items(), old_grads[3:]):
        key = new_name("processor." + name).removeprefix("processor.")
        close(new_by_name[key], gradient, report, "reference_parameter_gradient_max", "gradient")
    for old_gradient, new_gradient in zip(old_grads[3 + len(old_parameters):], new_grads[3 + len(new_parameters):]):
        close(new_gradient, old_gradient, report, "reference_readout_gradient_max", "gradient")
    report["C_original_processor"] = "PASS: complete original b30 weights, nonzero rates/controls, fixed H0/g0"


def run_terminal_skip(ref, report):
    data, current, future, _, _ = small_graph(ref)
    target = build_delta_targets(current, future, data)
    normal = create_model()
    activate_rates_controls(normal)
    diagnostic = copy.deepcopy(normal)
    counts = dict(graph=0, node_reaction=0, unit_reaction=0, node_control=0, reaction_control=0, initializer=0)
    def counter(name):
        def hook(*args):
            counts[name] += 1
        return hook
    handles = [normal.processor.graph_messages.register_forward_hook(counter("graph")),
               normal.processor.node_control.register_forward_hook(counter("node_control")),
               normal.processor.reaction_control.register_forward_hook(counter("reaction_control")),
               normal.processor.reaction_encoder.initializer.register_forward_hook(counter("initializer"))]
    def counted_messages(h, z, a, graph, update_reaction=True):
        counts["node_reaction"] += 1
        counts["unit_reaction"] += int(update_reaction)
        return reaction_messages(h, z, a, graph, update_reaction)
    with patch("processor.reaction_messages", side_effect=counted_messages):
        output = normal(data, capture=True)
    for handle in handles:
        handle.remove()
    assert counts == dict(graph=4, node_reaction=4, unit_reaction=3, node_control=4, reaction_control=3, initializer=1), counts
    complete = diagnostic(data, compute_terminal_reaction=True, capture=True)
    assert len(output["trace"]["H"]) == 5 and len(output["trace"]["Z"]) == 4
    assert len(complete["trace"]["Z"]) == 5
    assert output["reaction_hidden_step"] == 3 and complete["reaction_hidden_step"] == 4
    assert output["trace"]["steps"][-1]["DR"] is None and output["trace"]["steps"][-1]["UR"] is None
    for name in ("H", "Z"):
        for value, expected in zip(output["trace"][name], complete["trace"][name]):
            close(value, expected, report, "terminal_" + name + "_max")
    tr = output["trace"]
    q = tr["a"] * (tr["Z"][3][data.reaction_index] - tr["H"][3][data.reaction_node])
    close(tr["steps"][3]["q"], q, report, "terminal_q_max", exact=True)
    # 用保留的旧状态单独补算 Z4，不能使用 H4。
    _, expected_z4, _ = normal.processor.advance(tr["H"][3], tr["Z"][3], tr["g0"], tr["a"], data)
    close(complete["trace"]["Z"][4], expected_z4, report, "terminal_diagnostic_Z4_max")
    loss1, loss2 = compare_predictions_and_loss(output["prediction"], complete["prediction"], current, target, data, report, "terminal")
    parameters1, parameters2 = list(normal.parameters()), list(diagnostic.parameters())
    grad1 = torch.autograd.grad(loss1, parameters1 + tr["Z"] + [tr["a"], tr["g0"]], retain_graph=True)
    grad2 = torch.autograd.grad(loss2, parameters2)
    for a, b in zip(grad1[:len(parameters1)], grad2):
        close(a, b, report, "terminal_parameter_gradient_max", "gradient")
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grad1[len(parameters1):])
    # 正常调用不 capture 时仍应与诊断结果一致。
    with torch.no_grad():
        plain = normal(data)
    for field in plain["prediction"]:
        close(plain["prediction"][field], output["prediction"][field], report, "terminal_plain_prediction_max", exact=True)
    assert plain["trace"] is None
    # 原版固定 K=4；直接测试 advance 关闭开关时确实无需计算任何下一 Z。
    with patch.object(normal.processor.reaction_control, "forward", side_effect=AssertionError("不应计算 UR")):
        h1, z1, parts = normal.processor.advance(tr["H"][0], tr["Z"][0], tr["g0"], tr["a"], data, False)
    assert z1 is None and parts["DR"] is None and parts["UR"] is None
    close(h1, tr["H"][1], report, "single_advance_max", exact=True)
    # 临时令已有轮数常量 K=1，验证完整循环的末轮规则；不改正式配置。
    with patch("processor.PROCESSOR_STEPS", 1):
        with patch.object(normal.processor.reaction_control, "forward", side_effect=AssertionError("K=1 不应计算 UR")):
            one = normal(data, capture=True)
        one_full = diagnostic(data, compute_terminal_reaction=True, capture=True)
    assert len(one["trace"]["H"]) == 2 and len(one["trace"]["Z"]) == 1
    assert one["reaction_hidden_step"] == 0 and one_full["reaction_hidden_step"] == 1
    assert one["trace"]["steps"][0]["DR"] is None
    one_loss, full_loss = compare_predictions_and_loss(one["prediction"], one_full["prediction"], current, target, data, report, "K1")
    one_grad = torch.autograd.grad(one_loss, parameters1, allow_unused=True)
    full_grad = torch.autograd.grad(full_loss, parameters2, allow_unused=True)
    for left, right in zip(one_grad, full_grad):
        if left is None or right is None:
            assert left is None and right is None
        else:
            close(left, right, report, "K1_gradient_max", "gradient")
    report["D_terminal_skip"] = "PASS: same predictions, CE2/Q_ref, gradients; Z0..Z3/a/g0 gradients connected"
    report["processor_call_counts"] = counts
    report["K1_boundary"] = "PASS: H0/Z0 -> H1, default returns Z0; diagnostic Z1 does not affect prediction/loss/gradient"
