"""只核对调度公式与调用时序；一个虚拟参数，不读取数据、不训练模型。"""

import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from learning_rate import NUM_EPOCHS, INITIAL_LR, MIN_LR, create_scheduler


def main():
    config = json.loads((HERE / "configs/learning_rate.json").read_text(encoding="utf-8"))
    assert config["max_epochs"] == config["T_max"] == NUM_EPOCHS == 141
    assert config["initial_lr"] == INITIAL_LR == 1e-3
    assert config["eta_min"] == MIN_LR == 1e-5
    # 不涉及模型训练，只模拟每轮先 optimizer.step、再 scheduler.step 的先后顺序。
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=INITIAL_LR)
    scheduler = create_scheduler(optimizer)
    rates = [optimizer.param_groups[0]["lr"]]
    used_rates = []
    for _ in range(NUM_EPOCHS):
        used_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    expected = [MIN_LR + (INITIAL_LR - MIN_LR) / 2 * (1 + math.cos(math.pi * t / NUM_EPOCHS))
                for t in range(NUM_EPOCHS + 1)]
    for actual, formula in zip(rates, expected):
        assert math.isclose(actual, formula, rel_tol=1e-12, abs_tol=1e-15), (actual, formula)
    assert all(left >= right for left, right in zip(rates, rates[1:]))
    assert used_rates[0] == INITIAL_LR
    assert used_rates[-1] > MIN_LR
    assert rates[-1] == MIN_LR and scheduler.last_epoch == NUM_EPOCHS
    # 初始学习率写错时必须报错，而不是把错误的值当作 eta_max。
    bad_optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    try:
        create_scheduler(bad_optimizer)
    except ValueError:
        pass
    else:
        raise AssertionError("错误的初始学习率未被拒绝")
    report = dict(
        status="PASS", scheduler="CosineAnnealingLR", checked_times=len(rates),
        maximum_formula_error=max(abs(a - b) for a, b in zip(rates, expected)),
        first_epoch_lr=used_rates[0], last_epoch_lr=used_rates[-1],
        after_epoch_141_lr=rates[-1], model_training_started=False,
    )
    path = HERE / "checks/learning_rate_result.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
