"""用训练实际读取路径记录每个模块的维度；只做检查，不保存模型或启动训练。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from initialization import create_model
from loss import compute_loss
from model_config import FIELD_NAMES, MODEL_SCHEMA
from state_output import recover_physical_state
from training_windows import TrainingWindows, WindowReader


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}


def describe(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))


def check_tensor(value, shape, dtype, device):
    assert tuple(value.shape) == tuple(shape), (value.shape, shape)
    assert value.dtype == dtype, (value.dtype, dtype)
    assert value.device.type == device
    if value.is_floating_point():
        assert torch.isfinite(value).all()
    return describe(value)


def add_layer_hooks(model, records):
    """记录每个 Linear、ReLU、LayerNorm 的真实输入和输出，重复形状只累计次数。"""
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            value = inputs[0]
            if isinstance(module, nn.Linear):
                assert value.shape[-1] == module.in_features
                expected_shape = (*value.shape[:-1], module.out_features)
            else:
                expected_shape = value.shape
                if isinstance(module, nn.LayerNorm):
                    assert tuple(value.shape[-len(module.normalized_shape):]) == module.normalized_shape
            check_tensor(output, expected_shape, torch.float32, value.device.type)
            signature = dict(input=describe(value), output=describe(output))
            entries = records.setdefault(name, [])
            for entry in entries:
                if entry["shapes"] == signature:
                    entry["calls"] += 1
                    break
            else:
                entries.append(dict(shapes=signature, calls=1))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU, nn.LayerNorm)):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def check_window(model, reader, scope, sid, time_index):
    data, current, future, targets = reader.window(scope, sid, time_index)
    n = data.num_nodes
    e = data.edge_index_u.shape[1]
    r = len(data.reaction_center)
    l = len(data.reaction_node)
    device = current.device.type
    report = dict(scope=scope, sample_index=sid, sample_id=data.sample_id, time_index=time_index,
                  counts=dict(nodes=n, physical_edges=e, directed_edges=2*e, reactions=r, incidences=l))
    report["phase_nodes"] = {str(p): len(data.phase_nodes[p]) for p in range(1, 8)}
    report["node_regions"] = torch.bincount(data.node_region, minlength=3).tolist()
    report["edge_groups"] = torch.bincount(data.edge_group, minlength=5).tolist()
    report["reaction_regions"] = torch.bincount(data.reaction_region, minlength=2).tolist()
    report["incidence_regions"] = torch.bincount(data.reaction_region[data.reaction_index], minlength=2).tolist()
    report["nodes_without_reactions"] = int((data.reaction_degree == 0).sum())

    float_shapes = dict(x=(n, 13), x_static=(n, 4), global_geometry=(6,),
                        edge_metadata=(2*e, 30), reaction_metadata=(l, 30))
    integer_shapes = dict(node_phase=(n,), node_region=(n,), edge_index=(2, 2*e), edge_index_u=(2, e),
                          edge_reverse=(2*e,), edge_group=(2*e,), edge_physical_index=(2*e,),
                          reaction_node=(l,), reaction_index=(l,), reaction_role=(l,),
                          reaction_center=(r,), reaction_kind=(r,), reaction_region=(r,))
    double_shapes = dict(throat_area_um2=(e,), throat_conduit_lengths_um=(e, 3),
                         concentration_scale=(7,), delta_mean=(7,), delta_std=(7,))
    report["data_tensors"] = {}
    for shapes, dtype in ((float_shapes, torch.float32), (integer_shapes, torch.int64), (double_shapes, torch.float64)):
        for name, shape in shapes.items():
            report["data_tensors"][name] = check_tensor(data[name], shape, dtype, device)
    for name, value in (("current", current), ("future", future)):
        report["data_tensors"][name] = check_tensor(value, (n, 2), torch.float64, device)

    records = {}
    handles = add_layer_hooks(model, records)
    model.zero_grad(set_to_none=True)
    try:
        output = model(data, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    report["layers"] = records

    expected_calls = {"node_encoder.mlp.0": (7, 13, 64), "global_encoder.mlp.0": (2, 454, 64),
                      "processor.reaction_encoder.port_encoder.0": (1, 94, 64),
                      "processor.reaction_encoder.initializer.0": (1, 65, 64)}
    for prefix, counts, calls, width in (
            ("processor.graph_messages.mlps", report["edge_groups"], 4, 188),
            ("processor.node_control.mlps", report["node_regions"], 4, 128),
            ("processor.reaction_control.mlps", report["reaction_regions"], 3, 128)):
        for group, count in enumerate(counts):
            expected_calls[f"{prefix}.{group}.0"] = (calls if count else 0, width, 64)
            expected_calls[f"{prefix}.{group}.3"] = (calls if count else 0, 64, 64)
    for group, count in enumerate(report["incidence_regions"]):
        expected_calls[f"processor.reaction_encoder.rate_layers.{group}"] = (1 if count else 0, 30, 1)
    for phase, width in ((1, 1), (2, 2), (3, 2), (4, 1), (5, 2), (7, 1)):
        expected_calls[f"decoder.heads.{phase}"] = (1, 128, width)
    for name, (calls, in_dim, out_dim) in expected_calls.items():
        entries = records.get(name, [])
        assert sum(entry["calls"] for entry in entries) == calls, name
        for entry in entries:
            assert entry["shapes"]["input"]["shape"][-1] == in_dim, name
            assert entry["shapes"]["output"]["shape"][-1] == out_dim, name

    trace = output["trace"]
    assert len(trace["H"]) == 5 and len(trace["Z"]) == 4
    assert output["reaction_hidden_step"] == 3
    report["H"] = [check_tensor(h, (n, 64), torch.float32, device) for h in trace["H"]]
    report["Z"] = [check_tensor(z, (r, 64), torch.float32, device) for z in trace["Z"]]
    check_tensor(output["node_hidden"], (n, 64), torch.float32, device)
    check_tensor(output["reaction_hidden"], (r, 64), torch.float32, device)
    check_tensor(output["reaction_rate"], (l, 1), torch.float32, device)
    report["global_states"] = {key: check_tensor(trace[key], (64,), torch.float32, device) for key in ("g0", "g1")}
    report["steps"] = []
    for step, parts in enumerate(trace["steps"]):
        shapes = dict(Dgraph=(n, 64), R=(n, 64), U=(n, 64), q=(l, 64))
        item = {key: check_tensor(parts[key], shape, torch.float32, device) for key, shape in shapes.items()}
        for key in ("DR", "UR"):
            if step < 3:
                item[key] = check_tensor(parts[key], (r, 64), torch.float32, device)
            else:
                assert parts[key] is None
                item[key] = None
        assert not parts["R"][data.reaction_degree == 0].any()
        report["steps"].append(item)

    report["fields"] = {}
    for field in FIELD_NAMES:
        count = len(data.field_nodes[field])
        report["fields"][field] = dict(
            prediction=check_tensor(output["prediction"][field], (count,), torch.float32, device),
            target=check_tensor(targets[field], (count,), torch.float64, device))
    physical, valid = recover_physical_state(current, output["prediction"], data)
    report["physical_state"] = check_tensor(physical, (n, 2), torch.float64, device)
    report["physical_mask"] = check_tensor(valid, (n, 2), torch.bool, device)
    # 用真实标准化标签恢复未来状态，检查输出列、cmax 和增量统计是否对应。
    restored_target, target_mask = recover_physical_state(current, targets, data)
    assert torch.equal(valid, target_mask)
    recovery_error = (restored_target[valid] - future[valid]).abs().max().item()
    torch.testing.assert_close(restored_target[valid], future[valid], atol=2e-6, rtol=2e-5)
    report["target_recovery_max_absolute_error"] = recovery_error

    losses = compute_loss(output["prediction"], targets, current, data)
    report["losses"] = {name: check_tensor(value, (), torch.float64, device)
                        for name, value in losses.items() if name != "field_losses"}
    report["field_losses"] = {name: check_tensor(value, (), torch.float64, device)
                              for name, value in losses["field_losses"].items()}
    losses["loss"].backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    report["backward"] = "PASS: all registered parameters have finite gradients (zero is allowed)"
    report["status"] = "PASS"
    return report


def main():
    hashes = source_hashes()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TrainingWindows()
    reader = WindowReader(dataset, device)
    model = create_model(device=device)
    report = dict(checked_at=datetime.now().astimezone().isoformat(), model_schema=MODEL_SCHEMA,
                  device=device, torch_version=str(torch.__version__), production_source_sha256=hashes,
                  note="two real windows; fresh model; no optimizer step, checkpoint write or formal training")
    names = ("node_encoder", "global_encoder", "processor.graph_messages", "processor.node_control",
             "processor.reaction_encoder.port_encoder", "processor.reaction_encoder.initializer",
             "processor.reaction_encoder.rate_layers", "processor.reaction_control", "decoder")
    report["parameters"] = {name: sum(p.numel() for p in model.get_submodule(name).parameters()) for name in names}
    report["parameters"]["total"] = sum(p.numel() for p in model.parameters())
    assert sum(report["parameters"][name] for name in names) == report["parameters"]["total"] == 186695
    report["windows"] = []
    for scope, sid, time_index in dataset.panel:
        window = check_window(model, reader, scope, sid, time_index)
        report["windows"].append(window)
        print(f"PASS {scope} sample={sid} t={time_index}: {window['counts']}")
    assert source_hashes() == hashes
    report["production_sources_unchanged"] = True
    report["status"] = "PASS"
    path = HERE / "checks" / "module_dimensions_result.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {path}")
    print(json.dumps(report["parameters"], indent=2))


if __name__ == "__main__":
    main()
