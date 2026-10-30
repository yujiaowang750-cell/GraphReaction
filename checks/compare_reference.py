"""只读等价性检查：不启动参考工程的训练器，也不修改参考目录。

从项目根目录运行 .venv/Scripts/python.exe GraphReaction/checks/compare_reference.py。
为了避开参考工程导入时的日志、缓存和调度代码，只提取其原始计算定义。
"""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from data_loader import get_sample_folder, load_static_graph, load_window
from initialization import create_model
from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS
from normalization import load_statistics
from state_output import build_delta_targets, recover_physical_state
from loss import compute_loss
from regions import assign_node_regions
from graph_topology import build_graph_topology
from reaction_topology import build_reaction_topology

REFERENCE_ROOT = Path(r'D:\Github Project\subdatasets\processed_v1\reproduction_proxy')
STUDY = REFERENCE_ROOT / 'experiments/three_region_hybrid_graph_v1/studies/hybrid3r_seed2026_20261010_072635'
DATA_ROOT = HERE.parent / 'Datasets'
source_hashes = {}


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def definitions(path, names, environment):
    source = path.read_bytes()
    source_hashes[str(path)] = hashlib.sha256(source).hexdigest()
    tree = ast.parse(source.decode('utf-8-sig'), str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
                selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), environment)


def reference_definitions():
    module = ModuleType('graphreaction_original_calculations')
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update(torch=torch, nn=nn, F=F, np=np, Path=Path, json=json,
               hashlib=hashlib, dataclass=dataclass, ensure=ensure,
               VARIANTS=('GraphOnly', 'GraphReaction'))
    definitions(REFERENCE_ROOT / 'reference/a0_source/data.py',
                {'FIELDS', 'PARTS', 'RELATIONS', 'read_json', 'zscore', 'Dataset', 'Sample'}, env)
    definitions(REFERENCE_ROOT / 'reference/a0_source/model.py', {'GraphTensor'}, env)
    definitions(REFERENCE_ROOT / 'reference/a0_source/physics.py', {'loss_terms', 'reconstruct_current'}, env)
    definitions(REFERENCE_ROOT / 'src/shared_model.py', {'DIMS', 'SLOTS', 'adapt_phase'}, env)
    definitions(STUDY / 'src/regions.py', {'assign', 'classify_pairs'}, env)
    definitions(STUDY / 'src/reaction_topology.py', {'build_reaction'}, env)
    definitions(STUDY / 'src/graph_topology.py', {'raw_geometry', 'build', 'Reaction', 'Topology', 'tensor'}, env)
    definitions(STUDY / 'src/model.py', {'encoder', 'control', 'message', 'mean_index', 'apply_group',
                                      'HybridProcessor', 'HybridModel', 'fresh'}, env)
    return env


PREFIXES = {
    'node_encoder.': 'node_encoder.mlp.',
    'global_encoder.': 'global_encoder.mlp.',
    'processor.pair_messages.': 'processor.graph_messages.mlps.',
    'processor.node_controls.': 'processor.node_control.mlps.',
    'processor.reaction_port_encoder.': 'processor.reaction_encoder.port_encoder.',
    'processor.reaction_initializer.': 'processor.reaction_encoder.initializer.',
    'processor.reaction_rates.': 'processor.reaction_encoder.rate_layers.',
    'processor.reaction_controls.': 'processor.reaction_control.mlps.',
    'decoders.': 'decoder.heads.',
}


def new_name(name):
    for old, new in PREFIXES.items():
        if name.startswith(old):
            return new + name[len(old):]
    raise KeyError(name)


def copy_weights(old_state, model):
    model.load_state_dict({new_name(name): value for name, value in old_state.items()}, strict=True)


def check_topology(data, original):
    pairs = {
        'edge_metadata': 'metadata', 'edge_reverse': 'reverse_port',
        'edge_group': 'arc_group', 'edge_physical_index': 'physical_rows',
        'node_region': 'node_region', 'graph_degree': 'graph_degree',
        'reaction_node': 'reaction_node', 'reaction_index': 'reaction_edge',
        'reaction_metadata': 'reaction_metadata', 'reaction_center': 'reaction_center',
        'reaction_kind': 'reaction_kind', 'reaction_region': 'reaction_region',
        'reaction_role': 'reaction_role',
    }
    for name, old in pairs.items():
        np.testing.assert_array_equal(data[name].numpy(), original[old], err_msg=name)
    np.testing.assert_array_equal(data.edge_index.numpy(), np.stack([original['sender'], original['receiver']]))


def main():
    from model_config import NODE_INPUT_DIM
    if NODE_INPUT_DIM != 17:
        raise ValueError("这是历史 X17 版本的完整网络等价检查；新版请运行 checks/check_version2.py。历史 comparison_result.json 保留不覆盖。")
    torch.set_num_threads(2)
    ref = reference_definitions()
    dataset = ref['Dataset'](DATA_ROOT)
    stats = load_statistics(DATA_ROOT)
    original_model = ref['fresh'](2026, 'GraphReaction')
    rewritten_model = create_model(2026)
    assert sum(p.numel() for p in rewritten_model.parameters()) == 186951
    for name, value in original_model.state_dict().items():
        torch.testing.assert_close(value, rewritten_model.state_dict()[new_name(name)], rtol=0, atol=0)
    report = {'parameter_count': 186951, 'initial_parameters_identical': True,
              'topology_graphs': 0, 'cases': [], 'status': 'running'}

    # 全部训练/验证图的实际拓扑逐元素比较；不读取测试图。
    for sample_id, sample_meta in dataset.samples.items():
        if sample_meta['split'] not in ('train', 'validation'):
            continue
        folder = dataset.path(sample_id)
        graph = load_static_graph(folder, stats)
        with np.load(folder / 'static_graph.npz', allow_pickle=False) as file:
            raw = {name: file[name] for name in file.files}
        geo = ref['raw_geometry'](raw, dataset.stats)
        topology = ref['build'](raw['node_phase'], raw['edge_conns'], raw['edge_relation'], geo,
                                raw['node_coords_normalized'], raw['sample_coordinate_range_zyx_um'], True)
        check_topology(graph, topology)
        report['topology_graphs'] += 1
    print('PASS: 79 real graph topologies and all initial parameter values', flush=True)

    # 小图检查平行连接、只有中心+ION 的反应、没有反应单元的情况。
    phase = np.array([1, 2, 2, 3, 4, 5, 6, 7])
    coords = np.zeros((8, 3), dtype=np.float64)
    coords[:, 0] = [.1, .2, .8, .9, .9, .5, 0., 1.]
    bounds = np.array([[0., 100.], [0., 10.], [0., 10.]])
    region = assign_node_regions(phase, coords, bounds)
    for conns, relations in [(np.array([[0, 1], [0, 1], [3, 2]]), np.array([1, 1, 4])),
                              (np.array([[1, 5], [2, 5]]), np.array([5, 5])),
                              (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64))]:
        geo = np.zeros((len(conns), 4), dtype=np.float64)
        topology = ref['build'](phase, conns, relations, geo, coords, bounds, True)
        data = dict(node_region=torch.tensor(region), **build_graph_topology(phase, conns, relations, geo, region),
                    **build_reaction_topology(phase, conns, relations, region))
        # 上面 check_topology 只需要字典读取，edge_index 使用属性在此转成简单 Data。
        from torch_geometric.data import Data
        check_topology(Data(**data), topology)
    report['artificial_topologies_passed'] = True

    checkpoint_path = STUDY / 'runs/H3R_CE2_GraphReaction_seed2026/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for weights_name in ('initial', 'trained'):
        if weights_name == 'trained':
            original_model.load_state_dict(checkpoint['model'], strict=True)
            copy_weights(checkpoint['model'], rewritten_model)
        original_model.eval()
        rewritten_model.eval()
        for sample_id in (1, 7, 45):
            sample = dataset.sample(sample_id)
            graph = ref['GraphTensor'](sample, 'cpu')
            raw = sample.graph
            topology = ref['build'](sample.phase, raw['edge_conns'], raw['edge_relation'], ref['raw_geometry'](raw, dataset.stats),
                                    raw['node_coords_normalized'], raw['sample_coordinate_range_zyx_um'], True)
            graph.hybrid = ref['tensor'](topology, 'cpu')
            new_static = load_static_graph(sample.path, stats)
            for time_index in (sample.meta['input_time_indices'][0], sample.meta['input_time_indices'][-1]):
                current_old, future_old = sample.state(time_index), sample.state(time_index + 1)
                inputs = graph.inputs(current_old)
                data, current, future = load_window(sample.path, time_index, stats, new_static)
                for phase in range(1, 8):
                    torch.testing.assert_close(data.x[data.phase_nodes[phase]], ref['adapt_phase'](inputs[phase], phase), rtol=0, atol=0)
                torch.testing.assert_close(data.global_geometry, graph.bounds, rtol=0, atol=0)
                target = build_delta_targets(current, future, data)
                with torch.no_grad():
                    expected = original_model(graph, inputs)
                    actual = rewritten_model(data)['prediction']
                    errors = []
                    for field, old_value in zip(FIELD_NAMES, expected):
                        torch.testing.assert_close(actual[field], old_value, atol=2e-6, rtol=2e-5)
                        errors.append(float((actual[field] - old_value).abs().max()))
                    terms = ref['loss_terms'](graph, graph.physical(current_old), graph.physical(future_old), expected)
                    losses = compute_loss(actual, target, current, data)
                    ce2 = (2 * terms[3][0] + terms[3][1:].sum()) / 8 + .1 * terms[2]
                    torch.testing.assert_close(losses['loss'], ce2, atol=1e-9, rtol=2e-4)
                    torch.testing.assert_close(losses['reference_loss'], terms[0], atol=1e-9, rtol=2e-4)
                    recovered, mask = recover_physical_state(current, actual, data)
                    old_recovered = graph.next_state(graph.physical(current_old), expected)
                    for index, field in enumerate(FIELD_NAMES):
                        torch.testing.assert_close(recovered[data.field_nodes[field], FIELD_STATE_COLUMNS[index]],
                                                   old_recovered[index], atol=1e-7, rtol=1e-8)
                    truth_recovered, truth_mask = recover_physical_state(current, target, data)
                    torch.testing.assert_close(truth_recovered[truth_mask], future[truth_mask], atol=1e-10, rtol=1e-10)
                report['cases'].append({'weights': weights_name, 'sample_index': sample_id, 'time_index': time_index,
                                        'max_prediction_difference': max(errors), 'loss_difference': float(abs(losses['loss'] - ce2)),
                                        'N': data.num_nodes, 'E': data.edge_index_u.shape[1],
                                        'R': len(data.reaction_center), 'L': len(data.reaction_node)})
                print('PASS:', weights_name, 'sample', sample_id, 't', time_index, 'max prediction error', max(errors), flush=True)

    # 在已有训练权重上比较梯度，避免只检查初始控制分支为零的特殊情况。
    original_model.zero_grad(set_to_none=True)
    rewritten_model.zero_grad(set_to_none=True)
    old_pred = original_model(graph, inputs)
    new_pred = rewritten_model(data)['prediction']
    old_terms = ref['loss_terms'](graph, graph.physical(current_old), graph.physical(future_old), old_pred)
    ((2 * old_terms[3][0] + old_terms[3][1:].sum()) / 8 + .1 * old_terms[2]).backward()
    compute_loss(new_pred, target, current, data)['loss'].backward()
    new_parameters = dict(rewritten_model.named_parameters())
    maximum_gradient_difference = 0.0
    for name, parameter in original_model.named_parameters():
        new_gradient = new_parameters[new_name(name)].grad
        assert parameter.grad is not None and new_gradient is not None, name
        torch.testing.assert_close(parameter.grad, new_gradient, atol=2e-5, rtol=2e-4)
        maximum_gradient_difference = max(maximum_gradient_difference, float((parameter.grad - new_gradient).abs().max()))
    report['maximum_gradient_difference'] = maximum_gradient_difference
    report['reference_sources_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in source_hashes.items())
    assert report['reference_sources_unchanged']
    report['reference_source_sha256'] = source_hashes
    report['rewritten_source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    report['status'] = 'PASS'
    (HERE / 'checks' / 'comparison_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL_CHECKS_PASS', 'gradient difference', maximum_gradient_difference, flush=True)


if __name__ == '__main__':
    main()
