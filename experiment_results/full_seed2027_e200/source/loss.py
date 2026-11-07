"""原 GraphReaction 的 CE2 训练损失和 Q_ref 参考指标，不引入新的损失项。"""

import torch

from current_loss import compute_current_loss
from model_config import FIELD_NAMES, FIELD_WEIGHTS, CURRENT_LOSS_WEIGHT
from state_output import recover_physical_state


def compute_loss(prediction, target, current_state, data):
    """七字段标准化增量 MSE + 电流辅助损失，均保留 float64 物理计算。

    电解液两个字段分别先算 phase 2、phase 5 的节点均值，再以 0.5:0.5 合并。
    正极集流体电势已经属于七字段，不另外叠加旧模型中的电压辅助项。
    """
    field_losses = {}
    for index, field in enumerate(FIELD_NAMES):
        if prediction[field].shape != target[field].shape or target[field].numel() == 0:
            raise ValueError(f"{field} 的预测和目标必须非空且形状一致。")
        squared_error = (prediction[field].double() - target[field]).square()
        if index < 2:
            phase = data.node_phase[data.field_nodes[field]]
            # 计算电解液两个字段的增量 MSE 并合并
            value = 0.5 * (squared_error[phase == 2].mean() + squared_error[phase == 5].mean())
        else:
            value = squared_error.mean()
        field_losses[field] = value
    values = torch.stack(list(field_losses.values()))
    weights = values.new_tensor(FIELD_WEIGHTS)
    # 计算七字段标准化增量 MSE
    main_loss = (values * weights).sum() / weights.sum()
    future, _ = recover_physical_state(current_state, prediction, data)
    current_loss, predicted_current = compute_current_loss(future, data)
    return {
        "loss": main_loss + CURRENT_LOSS_WEIGHT * current_loss,  # 原训练目标 CE2
        "main_loss": main_loss,
        "current_loss": current_loss,
        "predicted_current_A": predicted_current,
        "field_losses": field_losses,
        "reference_loss": values.mean() + CURRENT_LOSS_WEIGHT * current_loss,  # 原 Q_ref
    }
