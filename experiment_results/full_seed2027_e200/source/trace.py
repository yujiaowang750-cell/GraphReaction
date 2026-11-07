"""诊断历史按 H、Z 分别存储，不假设两种状态有相同轮数。"""


def record_capture(trace, next_nodes, next_units, parts):
    trace["H"].append(next_nodes)
    if next_units is not None:
        trace["Z"].append(next_units)
    trace["steps"].append(parts)


def validate_capture(trace, steps, compute_terminal_reaction=False):
    expected_z = steps + 1 if compute_terminal_reaction else steps
    if len(trace["H"]) != steps + 1 or len(trace["Z"]) != expected_z or len(trace["steps"]) != steps:
        raise ValueError("trace 轮数错误：默认 H0～H4、Z0～Z3；完整诊断包含 Z4。")
    for index, parts in enumerate(trace["steps"]):
        calculated = index < steps - 1 or compute_terminal_reaction
        if (parts["DR"] is not None) != calculated or (parts["UR"] is not None) != calculated:
            raise ValueError("未计算的 DR/UR 必须标记为 None，不能伪造为零。")
