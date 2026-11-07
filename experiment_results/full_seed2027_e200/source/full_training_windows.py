"""按 trajectory_split.json 读取完整训练/验证窗口，不读取测试集状态。"""

import json
from pathlib import Path

import numpy as np

from normalization import load_statistics
from run_files import file_hash
from training_windows import TrainingWindows


class FullTrainingWindows(TrainingWindows):
    """只复用原类的打乱、pairs、weight 方法，不调用32档数据的初始化。"""

    def __init__(self, dataset_root=None):
        self.mode = "full"
        self.root = Path(dataset_root or Path(__file__).resolve().parent.parent / "Datasets").resolve()
        self.full_root = self.root
        self.proxy_root = None
        split = json.loads((self.root / "trajectory_split.json").read_text(encoding="utf-8"))
        metadata = json.loads((self.root / "dataset_manifest.json").read_text(encoding="utf-8"))
        groups = {name: set(ids) for name, ids in split["trajectory_ids"].items()}
        if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("trajectory_split.json 的训练、验证、测试轨迹不能重叠。")

        # 这里只保留训练和验证样本；测试样本不会进入可读取窗口集合。
        selected = [s for s in metadata["samples"] if s["trajectory_id"] in groups["train"] | groups["validation"]]
        self.samples = {s["sample_index"]: s for s in selected}
        if len(self.samples) != len(selected):
            raise ValueError("dataset_manifest.json 存在重复 sample_index。")
        self.statistics = load_statistics(self.root)  # 沿用已有训练统计，不重新拟合。
        names = ("trajectory_split.json", "dataset_manifest.json", "window_index.npz", "normalization_stats.json")
        self.identity = dict(mode="full_train_validation", source={name: file_hash(self.root / name) for name in names})

        self.rows, self.by_pair = {}, {}
        with np.load(self.root / "window_index.npz", allow_pickle=False) as index:
            for scope, split_name in (("train", "train"), ("full", "validation")):
                ids = index[split_name + "_sample_index"]
                times = index[split_name + "_input_time_index"]
                if ids.ndim != 1 or times.shape != ids.shape or ids.dtype.kind not in "iu" or times.dtype.kind not in "iu":
                    raise ValueError(f"{split_name} 的窗口索引应为形状相同的一维整数数组。")
                samples = [s for s in selected if s["trajectory_id"] in groups[split_name]]
                if any(s["split"] != split_name for s in samples):
                    raise ValueError(f"{split_name} 的样本标签与 trajectory_split.json 不一致。")
                if {s["trajectory_id"] for s in samples} != groups[split_name]:
                    raise ValueError(f"{split_name} 的轨迹清单不完整。")
                if len(groups[split_name]) != split["trajectory_counts"][split_name] or len(samples) != split["sample_counts"][split_name]:
                    raise ValueError(f"{split_name} 的轨迹数或样本数与划分文件不一致。")

                pairs = [(int(sid), int(t)) for sid, t in zip(ids, times)]
                expected = {(s["sample_index"], int(t)) for s in samples for t in s["input_time_indices"]}
                if len(pairs) != len(set(pairs)):
                    raise ValueError(f"{split_name} 存在重复窗口。")
                if set(pairs) != expected:
                    raise ValueError(f"{split_name} 窗口必须完整匹配划分和 input_time_indices，不能遗漏或混入其他集合。")
                if len(pairs) != split["window_counts"][split_name]:
                    raise ValueError(f"{split_name} 的窗口数与划分文件不一致。")
                # 全量窗口全部参与，每个窗口权重为1，不使用子集抽样补偿权重。
                self.rows[scope] = [dict(sid=sid, t=t, weight=1., population_weight=1.) for sid, t in pairs]
                self.by_pair[scope] = {(r["sid"], r["t"]): r for r in self.rows[scope]}

        # 保留既有指标中的8段时间统计；仅影响分段报告，不影响训练顺序或损失。
        self.stage_definition = "eight equal-count time strata per trajectory; retained reporting rule; not SOC or inferred physical phases"
        self.bins = {}
        for sid, sample in self.samples.items():
            for bin_id, part in enumerate(np.array_split(sorted(sample["input_time_indices"]), 8)):
                for t in part:
                    self.bins[(sid, int(t))] = bin_id
        self.panel = [(scope, self.rows[scope][0]["sid"], self.rows[scope][0]["t"]) for scope in ("train", "full")]
