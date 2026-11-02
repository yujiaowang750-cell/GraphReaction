"""只读取 trajectory_split.json 指定的14条完整测试轨迹。"""

import json
from pathlib import Path

import numpy as np

from normalization import load_statistics
from run_files import file_hash


class TestWindows:
    def __init__(self, dataset_root):
        self.root = self.full_root = Path(dataset_root).resolve()
        self.mode = "full_test"
        split = json.loads((self.root / "trajectory_split.json").read_text(encoding="utf-8"))
        manifest = json.loads((self.root / "dataset_manifest.json").read_text(encoding="utf-8"))
        groups = {name: set(ids) for name, ids in split["trajectory_ids"].items()}
        if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("训练、验证、测试轨迹不能重叠。")
        selected = [s for s in manifest["samples"] if s["trajectory_id"] in groups["test"]]
        self.samples = {s["sample_index"]: s for s in selected}
        if len(self.samples) != len(selected) or any(s["split"] != "test" for s in selected):
            raise ValueError("测试样本索引重复，或split标签不一致。")
        if {s["trajectory_id"] for s in selected} != groups["test"]:
            raise ValueError("测试轨迹清单不完整。")
        if len(groups["test"]) != split["trajectory_counts"]["test"] or len(selected) != split["sample_counts"]["test"]:
            raise ValueError("测试轨迹数与划分不一致。")
        with np.load(self.root / "window_index.npz", allow_pickle=False) as index:
            ids, times = index["test_sample_index"], index["test_input_time_index"]
            if ids.ndim != 1 or ids.shape != times.shape or ids.dtype.kind not in "iu" or times.dtype.kind not in "iu":
                raise ValueError("测试窗口索引必须为形状相同的一维整数数组。")
            pairs = [(int(sid), int(t)) for sid, t in zip(ids, times)]
        expected = {(s["sample_index"], int(t)) for s in selected for t in s["input_time_indices"]}
        if len(pairs) != len(set(pairs)) or set(pairs) != expected or len(pairs) != split["window_counts"]["test"]:
            raise ValueError("测试窗口必须不重不漏地覆盖测试轨迹，不得混入训练或验证窗口。")
        self.rows = {"test": [dict(sid=sid, t=t) for sid, t in pairs]}
        self.by_pair = {"test": {(row["sid"], row["t"]): row for row in self.rows["test"]}}
        self.statistics = load_statistics(self.root)  # 只读原训练统计，不在测试集拟合。
        names = ("trajectory_split.json", "dataset_manifest.json", "window_index.npz", "normalization_stats.json")
        self.identity = dict(mode="full_test", source={name: file_hash(self.root / name) for name in names})

    def pairs(self, scope):
        if scope != "test":
            raise ValueError("本入口只允许测试集。")
        return [(row["sid"], row["t"]) for row in self.rows[scope]]

    def weight(self, scope, sid, t):
        if scope != "test" or (sid, t) not in self.by_pair["test"]:
            raise ValueError("不属于测试窗口集合。")
        return 1.
