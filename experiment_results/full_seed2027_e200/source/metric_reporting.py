"""新增指标：窗口算术平均，以及不加权的节点合并 NRMSE_delta。"""

import math

import torch

from model_config import FIELD_NAMES
from training_metrics import SHORT_FIELDS, MetricMean
from run_files import write_csv, write_json
from group_metrics import group_metrics


class EpochMetrics:
    """每个窗口记录一次；梯度累积的除数不参与指标统计。"""

    def __init__(self, saved=None):
        self.mean = MetricMean(saved["mean"] if saved else None)
        self.sse = {name: 0. for name in SHORT_FIELDS}
        self.count = {name: 0 for name in SHORT_FIELDS}
        if saved:
            self.sse.update(saved["sse"])
            self.count.update(saved["count"])

    @torch.no_grad()
    def add(self, values, prediction, target):
        # values 复用 compute_loss/metric_values 的结果；这里不重新定义损失。
        self.mean.add(values)
        errors = [prediction[field].detach().double() - target[field] for field in FIELD_NAMES]
        sums = torch.stack([error.square().sum() for error in errors]).cpu().tolist()
        for short, error, total in zip(SHORT_FIELDS, errors, sums):
            if not math.isfinite(total):
                raise ValueError(f"{short} 的标准化增量平方误差非有限。")
            self.sse[short] += total
            self.count[short] += error.numel()

    def merge(self, other):
        for name in self.mean.sums:
            self.mean.sums[name] += other.mean.sums[name]
        self.mean.denominator += other.mean.denominator
        self.mean.windows += other.mean.windows
        for field in SHORT_FIELDS:
            self.sse[field] += other.sse[field]
            self.count[field] += other.count[field]

    def state(self):
        return dict(mean=self.mean.state(), sse=self.sse.copy(), count=self.count.copy())

    def fields(self):
        """先合并所有有效节点 SSE/count，最后开一次平方；不分区域加权。"""
        result = {}
        for field in SHORT_FIELDS:
            mse = self.sse[field] / self.count[field] if self.count[field] else None
            result[field] = dict(standardized_node_pooled_MSE=mse,
                                NRMSE_delta=sqrt_mse(mse), count=self.count[field],
                                standardized_SSE=self.sse[field])
        return result


def sqrt_mse(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("节点合并 MSE 必须为有限非负数。")
    return math.sqrt(value)


def fields_from_saved(overall):
    """只把每窗权重为1的旧节点合并MSE标为论文所要求的不加权NRMSE。"""
    if str(overall.get("weight_rule")) != "1":
        raise ValueError("旧评价使用了抽样窗口权重，不能冒充未加权 NRMSE_delta；需要原始节点充分统计或重新评价。")
    fields = {}
    for field in SHORT_FIELDS:
        saved = overall["physical_node_pooled"][field]
        mse = saved["standardized_node_pooled_MSE"]
        if not math.isclose(float(saved["weighted_count"]), float(saved["count"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("评价声称窗口权重为1，但节点加权计数与真实计数不一致。")
        if mse is not None and saved["count"] <= 0:
            raise ValueError("非空节点MSE必须具有正的有效节点计数。")
        fields[field] = dict(standardized_node_pooled_MSE=mse,
                             NRMSE_delta=sqrt_mse(mse), count=saved["count"])
    return fields


def save_nrmse(folder, fields, *, scope, checkpoint=None, windows=None):
    """新增两个文件，原 overall_metrics.json 和 field_metrics.csv 保持原格式。"""
    rows = [dict(field=field, **fields[field]) for field in SHORT_FIELDS]
    columns = ["field", "standardized_node_pooled_MSE", "NRMSE_delta", "count"]
    write_csv(folder / "nrmse_delta.csv", [{key: row[key] for key in columns} for row in rows], columns)
    write_json(folder / "nrmse_delta.json", dict(
        format_version="unweighted_node_pooled_nrmse_delta_v1", scope=scope,
        checkpoint=checkpoint, windows=windows, fields=fields,
        definition="sqrt(sum of standardized increment squared errors / all valid node occurrences)",
        window_weight=1, field_weight=1, electrolyte_region_weight="none; include all phase2/5 nodes"))


def epoch_columns(prefix, metrics):
    """把一个 epoch 的窗口平均和节点合并指标放入新增列。"""
    values = metrics.mean.result()
    values.update(group_metrics(values))
    columns = {f"{prefix}_{name}": value for name, value in values.items()}
    for field, values in metrics.fields().items():
        columns[f"{prefix}_{field}_NRMSE_delta"] = values["NRMSE_delta"]
        columns[f"{prefix}_{field}_node_pooled_MSE"] = values["standardized_node_pooled_MSE"]
    return columns
