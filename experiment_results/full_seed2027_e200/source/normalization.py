"""读取参考模型使用的训练统计量，并提供普通 z-score。"""

import json
from pathlib import Path

import numpy as np


def load_statistics(dataset_root):
    path = Path(dataset_root) / "normalization_stats.json"
    statistics = json.loads(path.read_text(encoding="utf-8"))
    if statistics["statistics_source"] != "all_windows_from_train_trajectories_only":
        raise ValueError("归一化统计必须只来自训练集。")
    return statistics


def zscore(values, statistics, prefix=""):
    """(x - mean) / effective_std；先以 float64 计算，再在网络入口转 float32。

    本参考模型没有在这里取 log，也不临时重新估计当前图的均值和标准差。
    常数通道的有效标准差由数据文件提供；非零的小标准差也按原值使用。
    """
    mean = statistics[prefix + "mean"]
    std = statistics[prefix + "effective_std"]
    if not np.isfinite(std) or std <= 0:
        raise ValueError("有效标准差必须为有限正数。")
    return (np.asarray(values, dtype=np.float64) - mean) / std
