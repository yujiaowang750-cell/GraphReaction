"""141 个 epoch 的余弦退火：学习率从 0.001 降到 0.00001。

调用方式（完整训练入口尚未编写）：
    optimizer = 原优化器(..., lr=INITIAL_LR)
    scheduler = create_scheduler(optimizer)
    for epoch in range(NUM_EPOCHS):
        # 先完成本轮所有训练窗口和 optimizer.step()。
        # 本轮的实际学习率在 scheduler.step() 之前读取。
        scheduler.step()  # 一轮一次，不传入 loss，也不在每个窗口后调用。

t 表示已经完成的 epoch 数：训练开始 t=0，第141轮结束 t=141。
第141轮训练使用 eta(140)，该轮结束后才更新为 eta(141)=MIN_LR。
严格使用给定公式的 T=141，不改成140。没有 warmup 或热重启。
训练预算用完后停止调用 scheduler.step()，避免越过余弦最低点后回升。
"""

import math

from torch.optim.lr_scheduler import CosineAnnealingLR

NUM_EPOCHS = 141
INITIAL_LR = 1e-3
MIN_LR = 1e-5


def create_scheduler(optimizer):
    """给新建的优化器配置余弦退火；优化器本身仍沿用原训练设置。"""
    for group in optimizer.param_groups:
        if not math.isclose(float(group["lr"]), INITIAL_LR, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"请先用 lr={INITIAL_LR} 创建优化器，再创建余弦调度器。")
    return CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR)
