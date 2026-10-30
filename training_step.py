"""一次 AdamW 更新：最多4个窗口依次反向、裁剪一次、更新一次。"""

import math

import torch

from loss import compute_loss
from training_metrics import MetricMean, metric_values
from metric_reporting import EpochMetrics
from training_config import CLIP_NORM


def train_group(model, reader, pairs, optimizer, collect_metrics=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = MetricMean()
    metrics.unweighted = EpochMetrics()
    for sid, t in pairs:
        data, current, future, target = reader.window("train", sid, t)
        prediction = model(data)["prediction"]
        losses = compute_loss(prediction, target, current, data)
        weight = reader.dataset.weight("train", sid, t)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"非有限训练目标：sample={sid}, time={t}")
        if collect_metrics:
            values = metric_values(losses, prediction, target, data)
            metrics.add(values, weight, denominator=1.)  # 原在线指标、原best选择口径。
            metrics.unweighted.add(values, prediction, target)  # 新增：窗口算术平均和节点合并误差。
        # 最后不足4个时按真实窗口数除；保持原梯度累积逻辑。
        (losses["loss"] * weight / len(pairs)).backward()
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" in str(error):
            raise FloatingPointError("nonfinite_gradient_pre_clip；本次 optimizer.step 未执行。") from error
        raise
    norm = float(norm)
    if not math.isfinite(norm):
        raise FloatingPointError("非有限裁剪前梯度范数；本次 optimizer.step 未执行。")
    optimizer.step()
    return metrics, norm
