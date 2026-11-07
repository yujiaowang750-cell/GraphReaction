"""新实验200个epoch的余弦退火：学习率从0.001降到0.00001。

调用方式（train.py在每轮结束时调用）：
    optimizer = 原优化器(..., lr=INITIAL_LR)
    scheduler = create_scheduler(optimizer)
    for epoch in range(NUM_EPOCHS):
        # 先完成本轮所有训练窗口和 optimizer.step()。
        # 本轮的实际学习率在 scheduler.step() 之前读取。
        scheduler.step()  # 一轮一次，不传入 loss，也不在每个窗口后调用。

t表示已完成的epoch数：训练开始t=0，第200轮结束t=200。
第200轮训练使用eta(199)，该轮结束后更新为eta(200)=MIN_LR。
新实验T=200；没有warmup或热重启，提前停止时无需到达最低学习率。
训练预算用完后停止调用 scheduler.step()，避免越过余弦最低点后回升。
"""

import math

from torch.optim.lr_scheduler import CosineAnnealingLR

NUM_EPOCHS = 200
INITIAL_LR = 1e-3
MIN_LR = 1e-5


def create_scheduler(optimizer):
    """给新建的优化器配置余弦退火；优化器本身仍沿用原训练设置。"""
    for group in optimizer.param_groups:
        if not math.isclose(float(group["lr"]), INITIAL_LR, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"请先用 lr={INITIAL_LR} 创建优化器，再创建余弦调度器。")
    return CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR)
