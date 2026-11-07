"""从同一次前向的原损失取出日志数值；所有统计都脱离计算图。"""

import math

import numpy as np
import torch

from model_config import FIELD_NAMES

SHORT_FIELDS = ("ce", "phi_e", "cg", "cp", "phi_p", "phi_CBD", "V")
BASE_METRICS = ("CE2", "Q_ref") + tuple(f"{name}_MSE" for name in SHORT_FIELDS) + (
    "ce_phase2_MSE", "ce_phase5_MSE", "phi_e_phase2_MSE", "phi_e_phase5_MSE", "current_relative_MSE")


@torch.no_grad()
def metric_values(losses, prediction, target, data):
    values = [losses["loss"].detach(), losses["reference_loss"].detach()]
    values.extend(losses["field_losses"][field].detach() for field in FIELD_NAMES)
    for field in FIELD_NAMES[:2]:
        phase = data.node_phase[data.field_nodes[field]]
        error = (prediction[field].detach().double() - target[field]).square()
        values.extend(error[phase == p].mean() for p in (2, 5))
    values.append(losses["current_loss"].detach())
    numbers = torch.stack(values).cpu().numpy()
    if not np.isfinite(numbers).all():
        raise FloatingPointError(f"非有限损失/指标：sample={data.sample_id}, time={data.input_time_index}")
    return dict(zip(BASE_METRICS, numbers.tolist()))


class MetricMean:
    def __init__(self, saved=None):
        self.sums = {name: 0. for name in BASE_METRICS}
        self.denominator, self.windows = 0., 0
        if saved is not None:
            self.sums, self.denominator, self.windows = saved["sums"].copy(), saved["denominator"], saved["windows"]

    def add(self, values, weight=1., denominator=None):
        """在线训练 denominator=1；Proxy/固定评价默认分母为权重和。"""
        for name in BASE_METRICS:
            self.sums[name] += weight * values[name]
        self.denominator += weight if denominator is None else denominator
        self.windows += 1

    def result(self):
        if self.denominator == 0:
            return {**{name: None for name in BASE_METRICS}, "current_relative_RMSE_pct": None}
        result = {name: value / self.denominator for name, value in self.sums.items()}
        result["current_relative_RMSE_pct"] = 100 * math.sqrt(result["current_relative_MSE"])
        return result

    def state(self):
        return dict(sums=self.sums.copy(), denominator=self.denominator, windows=self.windows)


class ErrorSums:
    """节点级充分统计：权重同时乘 SSE、SAE、count；不先开方再平均。"""
    def __init__(self):
        self.sse = self.sae = self.count = self.standardized_sse = 0.
        self.raw_count = self.windows = 0
        self.max_abs = None

    def add(self, physical_error, standardized_error, weight):
        error, standardized = np.asarray(physical_error), np.asarray(standardized_error)
        if error.size == 0:
            return
        if not np.isfinite(error).all() or not np.isfinite(standardized).all():
            raise FloatingPointError("评价误差中出现 NaN/Inf。")
        self.sse += weight * float(np.square(error).sum())
        self.sae += weight * float(np.abs(error).sum())
        self.standardized_sse += weight * float(np.square(standardized).sum())
        self.count += weight * error.size
        self.raw_count += error.size
        self.windows += 1
        maximum = float(np.abs(error).max())
        self.max_abs = maximum if self.max_abs is None else max(self.max_abs, maximum)

    def result(self):
        return dict(physical_RMSE=math.sqrt(self.sse / self.count) if self.count else None,
                    physical_MAE=self.sae / self.count if self.count else None,
                    standardized_node_pooled_MSE=self.standardized_sse / self.count if self.count else None,
                    max_abs_error=self.max_abs, count=self.raw_count, weighted_count=self.count,
                    windows=self.windows, physical_SSE=self.sse, physical_SAE=self.sae)
