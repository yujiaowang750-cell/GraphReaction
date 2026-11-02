"""每轮结束后固定当前权重，评价完整验证集；不反向、不更新参数。"""

import math

import torch

from loss import compute_loss
from metric_reporting import EpochMetrics
from training_metrics import metric_values
from training_checkpoint import rng_state, restore_rng


@torch.no_grad()
def validate_full(model, reader, smoke_windows=None):
    """返回窗口平均CE2/Q_ref、字段MSE及节点合并指标。

    full在当前数据读取器中表示完整验证集。正式训练不截取窗口；
    smoke_windows只用于明确标记的短流程检查。
    """
    pairs = reader.dataset.pairs("full")
    if smoke_windows is not None:
        pairs = pairs[:smoke_windows]
    if not pairs:
        raise ValueError("验证集不能没有窗口。")
    previous_mode = model.training
    saved_rng = rng_state()
    metrics = EpochMetrics()
    model.eval()
    try:
        for step, (sid, t) in enumerate(pairs, 1):
            data, current, _, target = reader.window("full", sid, t)
            prediction = model(data)["prediction"]
            losses = compute_loss(prediction, target, current, data)
            metrics.add(metric_values(losses, prediction, target, data), prediction, target)
            if step % 500 == 0 or step == len(pairs):
                print(f"epoch validation: {step}/{len(pairs)}", flush=True)
    finally:
        model.train(previous_mode)
        restore_rng(saved_rng)
    return metrics


def update_validation_best(state, metrics, epoch, updates):
    """严格下降才更新验证best并清零；相等或上升都算一轮未改善。"""
    value = metrics["CE2"]
    if not math.isfinite(value):
        raise FloatingPointError("验证CE2不是有限值。")
    improved = state["best_val_CE2"] is None or value < state["best_val_CE2"]
    if improved:
        state.update(best_val_CE2=value, best_val_epoch=epoch, best_val_update=updates,
                     best_val_Q_ref=metrics["Q_ref"], early_stop_counter=0)
    else:
        state["early_stop_counter"] += 1
    return improved
