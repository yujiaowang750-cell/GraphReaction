"""32档索引、权重、原两级打乱顺序，以及容量为一条轨迹的读取缓存。"""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_loader import load_static_graph, load_physical_state
from node_features import build_node_features
from normalization import load_statistics
from state_output import build_delta_targets
from run_files import file_hash

DEFAULT_PROXY = Path(r"D:\Github Project\subdatasets\processed_v1\proxy_runs\time_v1")


class TrainingWindows:
    def __init__(self, proxy_root=DEFAULT_PROXY, full_root=None):
        # 加载32档数据
        self.mode = "proxy32"
        self.proxy_root = Path(proxy_root).resolve()
        manifest = json.loads((self.proxy_root / "proxy_manifest.json").read_text(encoding="utf-8"))
        self.root = (self.proxy_root / manifest["dataset_root"]).resolve()
        self.full_root = Path(full_root or Path(__file__).resolve().parent.parent / "Datasets").resolve()
        for root in (self.root, self.full_root):
            for name, digest in manifest["source_sha256"].items():
                if file_hash(root / name) != digest:
                    raise ValueError(f"数据源身份不一致：{root / name}")
        names = ("window_index_32.npz", "window_weights_32.npz", "selected_windows_32.csv")
        for name in names:
            if file_hash(self.proxy_root / name) != manifest["output_sha256"][name]:
                raise ValueError(f"32档文件身份不一致：{name}")
        self.identity = dict(source=manifest["source_sha256"], proxy_manifest=file_hash(self.proxy_root / "proxy_manifest.json"),
                             proxy_files={name: file_hash(self.proxy_root / name) for name in names})
        metadata = json.loads((self.root / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.samples = {s["sample_index"]: s for s in metadata["samples"]}
        self.statistics = load_statistics(self.root)
        with np.load(self.proxy_root / "window_index_32.npz", allow_pickle=False) as index, np.load(
                self.proxy_root / "window_weights_32.npz", allow_pickle=False) as weights:
            self.rows = {}
            for scope, split in (("train", "train"), ("proxy", "validation")):
                self.rows[scope] = [dict(sid=int(sid), t=int(t), weight=float(weight), population_weight=float(pop))
                    for sid, t, weight, pop in zip(index[split+"_sample_index"], index[split+"_input_time_index"],
                        weights[split+"_mean_loss_weight"], weights[split+"_population_weight"])]
        with np.load(self.full_root / "window_index.npz", allow_pickle=False) as index:
            self.rows["full"] = [dict(sid=int(sid), t=int(t), weight=1., population_weight=1.)
                for sid, t in zip(index["validation_sample_index"], index["validation_input_time_index"])]
        self.by_pair = {scope: {(r["sid"], r["t"]): r for r in rows} for scope, rows in self.rows.items()}
        for scope, expected_graphs, expected_windows in (("train", 65, 2080), ("proxy", 14, 448), ("full", 14, 4917)):
            rows = self.rows[scope]
            assert len(rows) == len(self.by_pair[scope]) == expected_windows
            assert len({r["sid"] for r in rows}) == expected_graphs
            for row in rows:
                sample = self.samples[row["sid"]]
                assert sample["split"] == ("train" if scope == "train" else "validation")
                assert row["t"] in sample["input_time_indices"]
            if scope != "full":
                split = "train" if scope == "train" else "validation"
                population = np.array([r["population_weight"] for r in rows])
                mean = np.array([r["weight"] for r in rows])
                n = manifest["source_counts"][split]["windows"]
                assert np.isfinite(population).all() and (population > 0).all()
                np.testing.assert_allclose(mean, population * len(rows) / n, rtol=1e-14, atol=0)
                assert np.isclose(population.sum(), n, rtol=1e-12)
        # 使用子集已定义的8个时间分层；不发明新的SOC/物理阶段。
        self.bins = {}
        for sid in {r["sid"] for rows in self.rows.values() for r in rows}:
            times = sorted(self.samples[sid]["input_time_indices"])
            for bin_id, part in enumerate(np.array_split(times, manifest["parameters"]["bins"])):
                for t in part:
                    self.bins[(sid, int(t))] = bin_id
        with (self.proxy_root / "selected_windows_32.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                scope = "train" if row["split"] == "train" else "proxy"
                pair = (int(row["sample_index"]), int(row["input_time_index"]))
                assert self.bins[pair] == int(row["bin_id"])
                expected = self.by_pair[scope][pair]
                assert expected["weight"] == float(row["mean_loss_weight"])
                assert expected["population_weight"] == float(row["population_weight"])
        self.panel = [(scope, self.rows[scope][0]["sid"], self.rows[scope][0]["t"]) for scope in ("train", "proxy")]

    def epoch_order(self, epoch, seed=2026):
        """重放原 MT19937 两级 shuffle；不消耗全局 NumPy/RNG 状态。"""
        rng = np.random.RandomState(seed)
        for _ in range(epoch):
            groups = {}
            for row in self.rows["train"]:
                groups.setdefault(row["sid"], []).append(row["t"])
            samples = list(groups)
            rng.shuffle(samples)
            order = []
            for sid in samples:
                times = groups[sid]
                rng.shuffle(times)
                order.extend((sid, t) for t in times)
        return order

    def pairs(self, scope):
        return [(r["sid"], r["t"]) for r in self.rows[scope]]

    def weight(self, scope, sid, t):
        row = self.by_pair[scope][(sid, t)]
        return row["weight"] if scope == "train" else row["population_weight"]


def order_hash(order):
    return hashlib.sha256(np.asarray(order, dtype="<i8").tobytes(order="C")).hexdigest()


class WindowReader:
    def __init__(self, dataset, device):
        self.dataset, self.device = dataset, torch.device(device)
        self.key = None
        self.cpu_graph = self.device_graph = self.arrays = None
        self.loaded_samples = set()
        self.on_graph = None

    def window(self, scope, sid, t):
        if scope not in self.dataset.by_pair:
            raise ValueError(f"未启用数据范围 {scope}；不读取测试集或未选择的数据集。")
        if (sid, t) not in self.dataset.by_pair[scope]:
            raise ValueError(f"禁止读取 scope={scope} 中未选择的窗口 {(sid,t)}")
        root = self.dataset.full_root if scope == "full" else self.dataset.root
        if self.key != (str(root), sid):
            self.cpu_graph = self.device_graph = self.arrays = None
            sample = self.dataset.samples[sid]
            self.folder = root / sample["processed_relative_dir"]
            self.cpu_graph = load_static_graph(self.folder, self.dataset.statistics)
            self.device_graph = self.cpu_graph.clone().to(self.device)
            self.arrays = {name: np.load(self.folder / "dynamic" / f"{name}_state.npy", mmap_mode="r", allow_pickle=False)
                           for name in self.cpu_graph.state_nodes}
            self.key = (str(root), sid)
            self.loaded_samples.add(sid)
            if self.on_graph is not None:
                self.on_graph(sid, self.cpu_graph)
        current = load_physical_state(self.folder, t, self.cpu_graph, self.arrays)
        future = load_physical_state(self.folder, t + 1, self.cpu_graph, self.arrays)
        # 静态张量只读共享；本窗口仅新建 x 和时间属性，不原地修改静态图。
        cpu = copy.copy(self.cpu_graph)
        cpu.input_time_index = t
        data = copy.copy(self.device_graph)
        data.x = build_node_features(current, cpu, self.dataset.statistics).to(self.device)
        data.input_time_index, data.target_time_index = t, t + 1
        current, future = current.to(self.device), future.to(self.device)
        target = build_delta_targets(current, future, data)
        return data, current, future, target
