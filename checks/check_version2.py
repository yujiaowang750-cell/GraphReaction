"""沿用检查入口：验证X13、恢复的b30端口、末轮Z跳过及兼容性，不启动正式训练。"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_helpers import (HERE, KEEP, TOLERANCES, reference_definitions, small_graph, close,
                        compare_topology, explicit_x17_to_x13, must_fail)
from check_inputs_topology import run_inputs_topology
from check_equivalence import run_original_processor, run_terminal_skip
from compare_reference import source_hashes
from checkpoint import parameter_audit, save_checkpoint, load_checkpoint
from data_loader import get_sample_folder, load_static_graph, load_window, load_physical_state
from initialization import create_model
from loss import compute_loss
from normalization import load_statistics
from schema import model_configuration, configuration_hash, statistics_identity
from reaction_cache import (reaction_cache_path, save_reaction_cache, load_reaction_cache,
                            reaction_source_identity)
from state_output import build_delta_targets, recover_physical_state
from training_config import training_configuration


def check_initialization(report):
    model = create_model()
    audit = parameter_audit(model)
    assert {k: audit[k] for k in ("node_encoder", "reaction_port_encoder", "reaction_rates", "total")} == {
        "node_encoder": 1024, "reaction_port_encoder": 6208, "reaction_rates": 62, "total": 186695}
    same = create_model()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, dict(same.named_parameters())[name])
    p = model.processor
    for rates in p.reaction_encoder.rate_layers:
        assert not rates.weight.any() and not rates.bias.any()
    for group in (p.node_control.mlps, p.reaction_control.mlps):
        for block in group:
            assert not block[-1].weight.any() and not block[-1].bias.any()
    for group in (p.graph_messages.mlps, p.node_control.mlps, p.reaction_control.mlps, p.reaction_encoder.rate_layers):
        for block in group[1:]:
            for first, second in zip(group[0].parameters(), block.parameters()):
                assert torch.equal(first, second) and first.data_ptr() != second.data_ptr()
    report["parameter_audit"] = audit
    saved_config = json.loads((HERE / "configs/restored_b30.json").read_text(encoding="utf-8"))
    assert saved_config["configuration"] == model_configuration()
    assert saved_config["configuration_hash"] == configuration_hash()
    assert saved_config["initialization"] == model.initialization_manifest
    report["initialization"] = "PASS: repeatable fresh X13/b30, rate=0.5, control last=0, copies have independent storage"
    settings = training_configuration()
    assert settings["monitor"] == "online_train_CE2" and not settings["validate_each_epoch"]
    assert settings["early_stopping"] == 20 and settings["max_epochs"] == 141
    assert settings["scheduler"] == "CosineAnnealingLR" and settings["T_max"] == 141
    assert settings["learning_rate"] == .001 and settings["eta_min"] == .00001
    report["retained_training_settings"] = settings
    return model


def check_compatibility(ref, report):
    data, current, future, stats, old = small_graph(ref)
    model = create_model()
    with torch.no_grad():
        expected = model(data)["prediction"]
    stats_id = statistics_identity(stats)
    dataset_id = "artificial-fixture-unchanged"
    with tempfile.TemporaryDirectory(prefix="graphreaction_b30_") as temp:
        path = Path(temp) / "model.pt"
        save_checkpoint(path, model, stats_id, dataset_id)
        restored = create_model(seed=17)
        load_checkpoint(path, restored, stats_id, dataset_id)
        with torch.no_grad():
            actual = restored(data)["prediction"]
        for field in expected:
            close(actual[field], expected[field], report, "checkpoint_roundtrip_max", exact=True)
        must_fail(lambda: load_checkpoint(path, restored, "wrong-stats", dataset_id), "训练统计身份")
        must_fail(lambda: load_checkpoint(path, restored, stats_id, "wrong-data"), "数据集")
        old_path = Path(temp) / "old.pt"
        torch.save({"model": ref["fresh"](2026, "GraphReaction").state_dict()}, old_path)
        must_fail(lambda: load_checkpoint(old_path, restored, stats_id, dataset_id), "旧 checkpoint")
        saved = torch.load(path, weights_only=True)
        previous = copy.deepcopy(saved)
        previous["model_schema"] = "graphreaction_x13_b12_v2"
        torch.save(previous, old_path)
        must_fail(lambda: load_checkpoint(old_path, restored, stats_id, dataset_id), "旧 checkpoint")
        saved["state_dict"]["node_encoder.mlp.0.weight"] = torch.zeros(64, 17)
        torch.save(saved, old_path)
        must_fail(lambda: load_checkpoint(old_path, restored, stats_id, dataset_id), "size mismatch")
        saved["configuration_hash"] = "old-configuration"
        torch.save(saved, old_path)
        must_fail(lambda: load_checkpoint(old_path, restored, stats_id, dataset_id), "配置身份")
        topology = {key: data[key] for key in data.keys() if key.startswith("reaction_")}
        source_id = reaction_source_identity(data.node_phase.numpy(), data.edge_index_u.T.numpy(),
                                              data.edge_metadata[::2, :12].argmax(1).numpy(), data.node_region.numpy())
        cache_path = reaction_cache_path(temp, source_id)
        save_reaction_cache(cache_path, topology, source_id)
        cached = load_reaction_cache(cache_path, source_id)
        for name in topology:
            assert torch.equal(cached[name], topology[name])
        must_fail(lambda: load_reaction_cache(cache_path, "other-source"), "来源不一致")
        torch.save({"schema": "reaction_kind2_phase7_role3_v2", "source_identity": source_id, "topology": topology}, old_path)
        must_fail(lambda: load_reaction_cache(old_path, source_id), "旧反应缓存")
        np.savez(Path(temp) / "old.npz", **old)
        must_fail(lambda: load_reaction_cache(Path(temp) / "old.npz", source_id), "旧反应缓存")
        bad = copy.deepcopy(topology)
        bad["reaction_metadata"] = torch.tensor(old["reaction_metadata"][:, KEEP])
        must_fail(lambda: save_reaction_cache(cache_path, bad, source_id), "[P_R,30]")
        bad = copy.deepcopy(topology)
        bad["reaction_metadata"][0, 29] = 1
        must_fail(lambda: save_reaction_cache(cache_path, bad, source_id), "geometry_valid必须为0")
        bad = copy.deepcopy(topology)
        bad["reaction_metadata"][0, 21:25] = torch.tensor([1., 0., 0., 0.])
        must_fail(lambda: save_reaction_cache(cache_path, bad, source_id), "角色1/2/3")
        # 形状正确但旧身份仍不能送入模型。
        broken = data.clone()
        broken.feature_identity = {"node": "x17"}
        must_fail(lambda: model(broken), "schema 不兼容")
        broken = data.clone()
        broken.reaction_metadata = torch.zeros(len(data.reaction_node), 12)
        must_fail(lambda: model(broken), "reaction_metadata")
        broken = data.clone()
        broken.edge_metadata = torch.zeros(data.edge_index.shape[1], 12)
        must_fail(lambda: model(broken), "edge_metadata")
        # 缺少动态文件时直接报出路径，不静默填 0。
        broken.state_nodes = {"electrolyte": data.field_nodes[next(iter(data.field_nodes))]}
        must_fail(lambda: load_physical_state(Path(temp) / "missing_sample", 0, broken), "electrolyte_state.npy")
    report["checkpoint_and_cache"] = "PASS: b30 roundtrip; reject X17/b12 identities, wrong shapes, geometry/role errors, stats and source mismatch"


def optimizer_smoke(model, data, current, future):
    target = build_delta_targets(current, future, data)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 仅一次可丢弃更新，不定义正式训练协议。
    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    losses = compute_loss(output["prediction"], target, current, data)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    before = model.node_encoder.mlp[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.node_encoder.mlp[0].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return float(losses["loss"].detach())


def check_training_checkpoint(ref, report):
    """一个人工窗口的可丢弃AdamW更新，核对新身份下完整训练状态的恢复。"""
    from train import new_optimizer, initial_state
    from learning_rate import create_scheduler
    from training_checkpoint import save_training_checkpoint, load_training_checkpoint, rng_state
    from check_training_metrics import assert_nested_equal
    data, current, future, _, _ = small_graph(ref)
    model = create_model()
    optimizer = new_optimizer(model)
    scheduler = create_scheduler(optimizer)
    target = build_delta_targets(current, future, data)
    loss = compute_loss(model(data)["prediction"], target, current, data)["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    state = initial_state()
    state.update(epoch=1, updates=1)
    identity = dict(model=model_configuration(), training=training_configuration(), dataset="artificial-check-only")
    with tempfile.TemporaryDirectory(prefix="b30_training_checkpoint_") as temp:
        path = Path(temp) / "last.pt"
        save_training_checkpoint(path, model, optimizer, scheduler, state, identity)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        restored = create_model(seed=17)
        restored_optimizer = new_optimizer(restored)
        restored_scheduler = create_scheduler(restored_optimizer)
        torch.rand(5)  # 故意改变随机状态，加载后应恢复到保存时。
        loaded = load_training_checkpoint(path, restored, restored_optimizer, restored_scheduler, identity)
        for name, value in model.state_dict().items():
            close(restored.state_dict()[name], value, report, "training_checkpoint_parameter_max", exact=True)
        assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        assert_nested_equal(restored_scheduler.state_dict(), scheduler.state_dict())
        assert_nested_equal(loaded["state"], state)
        assert_nested_equal(rng_state(), saved["rng"])
        saved["schema"] = "graphreaction_x13_b12_cosine_training_v1"
        old_path = Path(temp) / "b12.pt"
        torch.save(saved, old_path)
        must_fail(lambda: load_training_checkpoint(old_path, restored, restored_optimizer, restored_scheduler, identity), "schema")
    report["training_checkpoint"] = "PASS: AdamW/scheduler/RNG/state restored exactly; old b12 training identity rejected"


def check_real_windows(ref, report):
    root = HERE.parent / "Datasets"
    if not (root / "normalization_stats.json").exists():
        report["not_run"].append("真实窗口：Datasets/normalization_stats.json 不存在")
        return
    dataset = ref["Dataset"](root)
    statistics = load_statistics(root)
    hashes_before = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                     for p in ("normalization_stats.json", "dataset_manifest.json", "trajectory_split.json")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report["real_windows"] = []
    for sample_id in (1, 0):
        folder = get_sample_folder(root, sample_id)
        graph = load_static_graph(folder, statistics)
        time_index = graph.input_time_indices[0]
        sample = dataset.sample(sample_id)
        old_graph = ref["GraphTensor"](sample, "cpu")
        inputs = old_graph.inputs(sample.state(time_index))
        old_x = torch.zeros(graph.num_nodes, 17)
        for phase in range(1, 8):
            old_x[old_graph.ids[phase]] = ref["adapt_phase"](inputs[phase], phase)
        data, current, future = load_window(folder, time_index, statistics, graph)
        close(data.x, explicit_x17_to_x13(old_x, data.node_phase), report, "real_input_mapping_max", exact=True)
        raw = sample.graph
        topology = ref["build"](sample.phase, raw["edge_conns"], raw["edge_relation"], ref["raw_geometry"](raw, dataset.stats),
                                raw["node_coords_normalized"], raw["sample_coordinate_range_zyx_um"], True)
        compare_topology(data, topology, report)
        target = build_delta_targets(current, future, data)
        recovered, mask = recover_physical_state(current, target, data)
        torch.testing.assert_close(recovered[mask], future[mask], atol=1e-10, rtol=1e-10)
        data, current, future = data.to(device), current.to(device), future.to(device)
        model = create_model(device=device)
        value = optimizer_smoke(model, data, current, future)
        report["real_windows"].append(dict(sample_index=sample_id, time_index=time_index, device=device,
                                            nodes=data.num_nodes, reaction_incidences=len(data.reaction_node), CE2=value))
        print(f"PASS real window sample={sample_id}, device={device}, forward/loss/backward/one discarded update", flush=True)
    # 可选缓存的真实读取往返；缓存只写临时目录。
    with tempfile.TemporaryDirectory(prefix="graphreaction_b30_real_") as temp:
        first = load_static_graph(folder, statistics, cache_root=temp)
        second = load_static_graph(folder, statistics, cache_root=temp)
        for name in first.keys():
            if name.startswith("reaction_"):
                assert torch.equal(first[name], second[name])
    assert hashes_before == {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in hashes_before}
    report["dataset_and_statistics_unchanged"] = hashes_before


def main():
    torch.set_num_threads(2)
    report = dict(status="running", configuration=model_configuration(), configuration_hash=configuration_hash(),
                  tolerances=TOLERANCES, not_run=[])
    ref = reference_definitions()
    check_initialization(report)
    run_inputs_topology(ref, report)
    print("PASS A/B: X13 and topology", flush=True)
    run_original_processor(ref, report)
    print("PASS C: restored b30 processor matches original with identical H0/g0 and weights", flush=True)
    run_terminal_skip(ref, report)
    print("PASS D: terminal Z skip equivalence and call counts", flush=True)
    check_compatibility(ref, report)
    check_training_checkpoint(ref, report)
    for mode in ("overlap", "size2", "no_reaction", "no_edges"):
        data, current, future, _, _ = small_graph(ref, mode)
        optimizer_smoke(create_model(), data, current, future)
    check_real_windows(ref, report)
    report["E_integration"] = "PASS"
    report["not_applicable"] = [
        "当前项目只实现 GraphReaction，无共享 GraphOnly 构造器；未新增 GraphOnly。",
        "当前处理器默认常量 K=4，没有运行时 K 参数；测试临时将常量改为1，已验证 K=1 完整前向边界。",
        "本次不启动正式训练或完整数据评价；只运行人工图和两个真实窗口的可丢弃更新。",
    ]
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in source_hashes.items())
    report["reference_sources_unchanged"] = True
    report["source_sha256"] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in HERE.rglob("*.py") if "__pycache__" not in p.parts}
    report["status"] = "PASS" if not report["not_run"] else "PARTIAL"
    output = HERE / "checks" / "b30_restored_result.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["status"], output, flush=True)
    print(json.dumps({k: v for k, v in report.items() if k.endswith("_max")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
