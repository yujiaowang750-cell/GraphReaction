"""读取一张静态图和一个 t -> t+1 窗口；不负责训练或修改原数据。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from graph_topology import build_graph_topology
from model_config import FIELD_NAMES
from node_features import build_node_features
from reaction_topology import build_reaction_topology
from regions import assign_node_regions
from static_features import build_static_features, build_edge_geometry, build_global_geometry
from field_validation import validate_physical_fields
from schema import feature_identity, statistics_identity
from reaction_cache import reaction_source_identity, reaction_cache_path, load_reaction_cache, save_reaction_cache


def get_sample_folder(dataset_root, sample_index):
    """sample_index 查总清单，不能根据轨迹文件名中的 rXXX 猜测。"""
    dataset_root = Path(dataset_root)
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    samples = {sample["sample_index"]: sample for sample in manifest["samples"]}
    return dataset_root / samples[sample_index]["processed_relative_dir"]


def load_static_graph(sample_folder, statistics, cache_root=None):
    """读取原始物理图，计算静态特征、三个区域和反应单元，返回 CPU 上的 Data。"""
    sample_folder = Path(sample_folder)
    # 用来确认模型是单步预测
    metadata = json.loads((sample_folder / "sample_manifest.json").read_text(encoding="utf-8"))
    if metadata["prediction_horizon_intervals"] != 1:
        raise ValueError("当前参考模型使用 H=1，即原数组中相隔一个时间步的目标。")
    # 加载数据相关的静态图信息
    with np.load(sample_folder / "static_graph.npz", allow_pickle=False) as arrays:
        raw = {name: arrays[name] for name in arrays.files}
    with np.load(sample_folder / "node_mapping.npz", allow_pickle=False) as arrays:
        mapping = {name: arrays[name] for name in arrays.files}
    phase = raw["node_phase"]  # 取出每个节点的材料相编号（1-7）
    # 按照节点相和坐标，给节点划分空间区域：正负极和隔膜区
    node_region = assign_node_regions(phase, raw["node_coords_normalized"], raw["sample_coordinate_range_zyx_um"])
    geometry = build_edge_geometry(raw, statistics)
    # 构建图拓扑
    graph_topology = build_graph_topology(phase, raw["edge_conns"], raw["edge_relation"], geometry, node_region)
    # 构建反应拓扑（用来计算反应发生的局部信息）
    source_id = reaction_source_identity(phase, raw["edge_conns"], raw["edge_relation"], node_region)
    cache_path = reaction_cache_path(cache_root, source_id) if cache_root is not None else None
    if cache_path is not None and cache_path.exists():
        reaction_topology = load_reaction_cache(cache_path, source_id)
    else:
        reaction_topology = build_reaction_topology(phase, raw["edge_conns"], raw["edge_relation"], node_region)
        if cache_path is not None:
            save_reaction_cache(cache_path, reaction_topology, source_id)

    phase_nodes = {p: torch.tensor(np.flatnonzero(phase == p)) for p in range(1, 8)}
    # 效验
    if any(len(nodes) == 0 for nodes in phase_nodes.values()):
        raise ValueError("分相全局池化需要图中七相均存在。")
    if len(phase_nodes[7]) != 1:
        raise ValueError("参考电流公式要求一个正极集流体节点。")
    state_nodes = {name: torch.tensor(mapping[name + "_node_index"], dtype=torch.long)
                   for name in ("electrolyte", "graphite", "positive_am", "positive_cbd", "collector")}
    all_state_nodes = torch.cat(list(state_nodes.values()))
    if not torch.equal(all_state_nodes.sort().values, torch.arange(len(phase))):
        raise ValueError("动态节点映射必须不重不漏地覆盖静态图节点。")
    # 将节点相，索引以及相名称组合拼接
    field_node_arrays = (
        state_nodes["electrolyte"], state_nodes["electrolyte"], state_nodes["graphite"],
        state_nodes["positive_am"], state_nodes["positive_am"], state_nodes["positive_cbd"], phase_nodes[7],
    )
    # 找出石墨和AM的最大锂浓度，为后续归一化作准备
    cmax = torch.tensor([
        1.0, 1.0, metadata["negative_graphite_max_concentration_mol_m3"],
        metadata["positive_am_max_concentration_mol_m3"], 1.0, 1.0, 1.0,
    ], dtype=torch.float64)
    # 训练统计中的石墨/AM 增量是锂化度；乘 cmax 后转成原浓度单位。
    delta_mean = torch.tensor([statistics["dynamic"][field]["delta_mean"] for field in FIELD_NAMES], dtype=torch.float64) * cmax
    delta_std = torch.tensor([statistics["dynamic"][field]["delta_effective_std"] for field in FIELD_NAMES], dtype=torch.float64) * cmax
    if not torch.isfinite(delta_mean).all() or not torch.isfinite(delta_std).all() or (delta_std <= 0).any():
        raise ValueError("增量训练统计无效。")
    # 组装PyG的Data对象，将所有静态信息都组合起来
    return Data(
        num_nodes=len(phase), node_phase=torch.tensor(phase, dtype=torch.long),
        node_region=torch.tensor(node_region), phase_nodes=phase_nodes,
        state_nodes=state_nodes, field_nodes=dict(zip(FIELD_NAMES, field_node_arrays)),
        x_static=build_static_features(raw, statistics),
        global_geometry=build_global_geometry(raw, statistics),
        edge_index_u=torch.tensor(raw["edge_conns"].T, dtype=torch.long),
        throat_area_um2=torch.tensor(raw["throat_area_um2"], dtype=torch.float64),
        throat_conduit_lengths_um=torch.tensor(raw["throat_conduit_lengths_um"], dtype=torch.float64),
        concentration_scale=cmax, delta_mean=delta_mean, delta_std=delta_std,
        applied_current_A=float(metadata["applied_current_A"]),
        sample_id=metadata["sample_id"], input_time_indices=metadata["input_time_indices"],
        feature_identity=feature_identity(), statistics_identity=statistics_identity(statistics),
        **graph_topology, **reaction_topology,
    )


def load_physical_state(sample_folder, time_index, graph, dynamic_arrays=None):
    """按原节点映射读取 [N,2] float64：[浓度 mol/m^3, 电势 V]。

    不存在浓度的节点第一列为零；读取到的石墨电势不会进入此模型的输入或标签。
    """
    state = torch.zeros((graph.num_nodes, 2), dtype=torch.float64)
    field_counts = {"electrolyte": 2, "graphite": 2, "positive_am": 2, "positive_cbd": 1, "collector": 1}
    for name, columns in field_counts.items():
        nodes = graph.state_nodes[name]
        arrays = (dynamic_arrays[name] if dynamic_arrays is not None else
                  np.load(Path(sample_folder) / "dynamic" / f"{name}_state.npy", mmap_mode="r", allow_pickle=False))
        if arrays.ndim != 3 or arrays.shape[1:] != (len(nodes), columns):
            raise ValueError(f"sample={sample_folder}, time={time_index}, field={name}: 动态数组与节点映射不匹配。")
        if not 0 <= time_index < len(arrays):
            raise IndexError(f"{name} 中不存在时间索引 {time_index}。")
        values = torch.tensor(arrays[time_index], dtype=torch.float64)
        if columns == 2:
            state[nodes] = values
        else:
            state[nodes, 1] = values[:, 0]
    validate_physical_fields(state, graph, time_index)
    return state


def load_window(sample_folder, time_index, statistics, static_graph=None, device="cpu"):
    """返回 (data, current_state, future_state)，状态均为原物理单位 [N,2]。

    data.x 为 [N,13]，只由当前时刻计算。future_state 单独返回，模型不会读取它。
    连续读取同一轨迹时，可以复用 static_graph；原静态图不会被当前窗口覆盖。
    """
    if static_graph is None:
        static_graph = load_static_graph(sample_folder, statistics)
    if time_index not in static_graph.input_time_indices:
        raise ValueError("time_index 必须来自该样本的 input_time_indices。")
    current = load_physical_state(sample_folder, time_index, static_graph)
    future = load_physical_state(sample_folder, time_index + 1, static_graph)
    data = static_graph.clone()
    data.input_time_index = time_index
    data.target_time_index = time_index + 1
    data.x = build_node_features(current, data, statistics)
    return data.to(device), current.to(device), future.to(device)
