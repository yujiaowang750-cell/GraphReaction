"""从已有七字段MSE计算分组指标；只整理日志，不参与反向传播。"""


def group_metrics(values):
    """C沿用ce权重2，P只含三个电势字段；V和电流I单列。

    CE2 = 4/8*C + 3/8*P + 1/8*V + 0.1*I。
    输入是同一统计口径的字段均值，不对梯度累积再次除法。
    """
    keys = ("ce_MSE", "cg_MSE", "cp_MSE", "phi_e_MSE", "phi_p_MSE",
            "phi_CBD_MSE", "V_MSE", "current_relative_MSE", "CE2")
    names = ("C_MSE", "P_MSE", "V_MSE", "I_relative_MSE",
             "C_CE2_contribution", "P_CE2_contribution", "V_CE2_contribution",
             "I_CE2_contribution", "main_loss", "total_loss")
    if any(values.get(key) is None for key in keys):
        return dict.fromkeys(names)
    c = (2 * values["ce_MSE"] + values["cg_MSE"] + values["cp_MSE"]) / 4
    p = (values["phi_e_MSE"] + values["phi_p_MSE"] + values["phi_CBD_MSE"]) / 3
    v, i = values["V_MSE"], values["current_relative_MSE"]
    return dict(C_MSE=c, P_MSE=p, V_MSE=v, I_relative_MSE=i,
                C_CE2_contribution=c * 4/8, P_CE2_contribution=p * 3/8,
                V_CE2_contribution=v / 8, I_CE2_contribution=0.1 * i,
                main_loss=(4*c + 3*p + v) / 8, total_loss=values["CE2"])
