"""固定模型评价：原损失口径与物理节点汇总分开记录，只允许 train/proxy/full。"""

import time
from pathlib import Path

import numpy as np
import torch

from loss import compute_loss
from model_config import FIELD_NAMES, FIELD_STATE_COLUMNS
from state_output import recover_physical_state
from training_metrics import SHORT_FIELDS, MetricMean, ErrorSums, metric_values
from run_files import write_json, write_csv
from metric_reporting import EpochMetrics, save_nrmse

FIELD_COLUMNS = ["scope", "field", "phase", "unit", "standardized_phase_balanced_MSE",
                 "physical_RMSE", "physical_MAE", "standardized_node_pooled_MSE", "max_abs_error",
                 "count", "weighted_count", "windows", "physical_SSE", "physical_SAE"]
REGION_COLUMNS = ["scope", "region_or_interface", "field", "unit"] + FIELD_COLUMNS[5:]
STAGE_COLUMNS = ["scope", "stage", "stage_kind", "field", "unit"] + FIELD_COLUMNS[5:]


@torch.no_grad()
def validate_proxy(model, reader, pairs):
    model.eval()
    mean = MetricMean()
    for sid, t in pairs:
        data, current, future, target = reader.window("proxy", sid, t)
        prediction = model(data)["prediction"]
        losses = compute_loss(prediction, target, current, data)
        mean.add(metric_values(losses, prediction, target, data), reader.dataset.weight("proxy", sid, t))
    return mean.result()


def physical_error_arrays(prediction, target, current, future, data):
    """recover 已经把 cg/cp 变成 mol/m^3；这里只对电势误差做一次 V->mV。"""
    recovered, _ = recover_physical_state(current, prediction, data)
    errors = {}
    for index, (field, short) in enumerate(zip(FIELD_NAMES, SHORT_FIELDS)):
        nodes, column = data.field_nodes[field], FIELD_STATE_COLUMNS[index]
        factor = 1. if column == 0 else 1000.
        physical = (recovered[nodes, column] - future[nodes, column]) * factor
        standardized = prediction[field].double() - target[field]
        errors[short] = (physical.detach().cpu().numpy(), standardized.detach().cpu().numpy())
    return recovered, errors


@torch.no_grad()
def evaluate_scope(model, reader, scope, folder, identity, checkpoint, pairs=None):
    if scope not in ("train", "proxy", "full"):
        raise ValueError("评价只允许 train/proxy/full，不读取测试集。")
    model.eval()
    pairs = reader.dataset.pairs(scope) if pairs is None else pairs
    means, fields, regions, stages, trajectories = MetricMean(), {}, {}, {}, {}
    reported = EpochMetrics()
    current_sums = ErrorSums()
    voltage_current_rows = []
    started = time.perf_counter()
    for step, (sid, t) in enumerate(pairs, 1):
        data, current, future, target = reader.window(scope, sid, t)
        prediction = model(data)["prediction"]
        losses = compute_loss(prediction, target, current, data)
        values = metric_values(losses, prediction, target, data)
        reported.add(values, prediction, target)
        weight = reader.dataset.weight(scope, sid, t)
        means.add(values, weight)
        trajectories.setdefault(sid, MetricMean()).add(values, weight)
        recovered, errors = physical_error_arrays(prediction, target, current, future, data)
        graph = reader.cpu_graph
        # 接口节点是原跨区物理边的去重端点，只选该字段确实有效的节点。
        boundaries = {name: np.unique(graph.edge_index[:, graph.edge_group == group].numpy())
                      for group, name in ((3, "NEG-SEP"), (4, "SEP-POS"))}
        stage = f"bin{reader.dataset.bins[(sid,t)]}"
        for field, short in zip(FIELD_NAMES, SHORT_FIELDS):
            physical_error, standardized_error = errors[short]
            nodes = graph.field_nodes[field].numpy()
            phase = graph.node_phase[nodes].numpy()
            region = graph.node_region[nodes].numpy()
            fields.setdefault((short, "all"), ErrorSums()).add(physical_error, standardized_error, weight)
            for p in sorted(set(phase.tolist())):
                mask = phase == p
                fields.setdefault((short, str(p)), ErrorSums()).add(physical_error[mask], standardized_error[mask], weight)
            masks = {name: region == r for r, name in enumerate(("NEG", "SEP", "POS"))}
            masks.update({name: np.isin(nodes, ends) for name, ends in boundaries.items()})
            for name, mask in masks.items():
                regions.setdefault((name, short), ErrorSums()).add(physical_error[mask], standardized_error[mask], weight)
            stages.setdefault((stage, short), ErrorSums()).add(physical_error, standardized_error, weight)
        predicted_current = float(losses["predicted_current_A"])
        error_current = predicted_current - data.applied_current_A
        current_sums.add(np.array([error_current]), np.array([error_current / abs(data.applied_current_A)]), weight)
        collector = data.phase_nodes[7]
        predicted_v = float(recovered[collector, 1].item())
        true_v = float(future[collector, 1].item())
        voltage_current_rows.append(dict(scope=scope, sample_index=sid, input_time_index=t, target_time_index=t+1,
            weight=weight, voltage_pred_V=predicted_v, voltage_true_V=true_v, voltage_error_mV=1000*(predicted_v-true_v),
            current_pred_A=predicted_current, current_applied_A=data.applied_current_A, current_error_A=error_current))
        if step % 500 == 0 or step == len(pairs):
            print(f"evaluation {scope}: {step}/{len(pairs)}", flush=True)
    result = means.result()
    def unit(short):
        return "mol/m^3" if short in ("ce", "cg", "cp") else "mV"
    field_rows = []
    for (short, phase), accumulator in fields.items():
        balanced = result[short+"_MSE"] if phase == "all" else result.get(f"{short}_phase{phase}_MSE")
        field_rows.append(dict(scope=scope, field=short, phase=phase, unit=unit(short),
                               standardized_phase_balanced_MSE=balanced, **accumulator.result()))
    current_result = current_sums.result()
    overall = dict(format_version="graphreaction_metrics_v1", scope=scope, identity=identity,
        checkpoint=checkpoint, status="completed", windows=len(pairs), graphs=len({sid for sid,t in pairs}),
        weight_rule="1" if getattr(reader.dataset, "mode", "proxy32") == "full" else
                    {"train":"mean_loss_weight", "proxy":"population_weight", "full":"1"}[scope],
        window_weight_sum=means.denominator, losses=result,
        physical_node_pooled={name: dict(unit=unit(name), **fields[(name,"all")].result()) for name in SHORT_FIELDS},
        current=dict(RMSE_A=current_result["physical_RMSE"], MAE_A=current_result["physical_MAE"],
                     relative_RMSE_pct=result["current_relative_RMSE_pct"], count=len(pairs)),
        stage_definition=getattr(reader.dataset, "stage_definition", "existing proxy_manifest eight time strata; not SOC or inferred physical phases"),
        seconds=time.perf_counter()-started, test_access=0, optimizer_updates=0)
    folder = Path(folder)
    write_json(folder / "overall_metrics.json", overall)
    write_csv(folder / "field_metrics.csv", field_rows, FIELD_COLUMNS)
    write_csv(folder / "region_metrics.csv", [dict(scope=scope, region_or_interface=region, field=short,
              unit=unit(short), **a.result()) for (region, short), a in regions.items()], REGION_COLUMNS)
    write_csv(folder / "stage_metrics.csv", [dict(scope=scope, stage=stage, stage_kind="existing_time_stratum", field=short,
              unit=unit(short), **a.result()) for (stage, short), a in stages.items()], STAGE_COLUMNS)
    write_json(folder / "trajectory_metrics.json", {sid: a.result() for sid, a in trajectories.items()})
    write_csv(folder / "voltage_current.csv", voltage_current_rows, list(voltage_current_rows[0]))
    # 单独新增文件，原指标的窗口权重、列名和数值保持不变。
    save_nrmse(folder, reported.fields(), scope=scope, checkpoint=checkpoint, windows=len(pairs))
    write_json(folder / "window_mean_metrics.json", dict(
        format_version="window_arithmetic_metrics_v1", scope=scope, checkpoint=checkpoint,
        windows=len(pairs), window_weight=1, metrics=reported.mean.result()))
    return overall
