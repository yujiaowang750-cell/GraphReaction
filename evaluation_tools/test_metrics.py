"""测试集单步评价；复用原损失及误差累加器，不改变指标定义。"""

from pathlib import Path
import time

import numpy as np
import torch

from evaluation import physical_error_arrays, FIELD_COLUMNS
from loss import compute_loss
from metric_reporting import EpochMetrics, save_nrmse
from model_config import FIELD_NAMES
from run_files import write_csv, write_json
from training_metrics import MetricMean, ErrorSums, metric_values, SHORT_FIELDS


@torch.no_grad()
def evaluate_test(model, reader, folder, identity, checkpoint):
    model.eval()
    pairs = reader.dataset.pairs("test")
    reported, fields, trajectories = EpochMetrics(), {}, {}
    current_errors, window_rows = ErrorSums(), []
    started = time.perf_counter()
    for step, (sid, t) in enumerate(pairs, 1):
        data, current, future, target = reader.window("test", sid, t)
        prediction = model(data)["prediction"]
        losses = compute_loss(prediction, target, current, data)
        values = metric_values(losses, prediction, target, data)
        reported.add(values, prediction, target)
        trajectories.setdefault(sid, MetricMean()).add(values)
        recovered, errors = physical_error_arrays(prediction, target, current, future, data)
        graph = reader.cpu_graph
        for field, short in zip(FIELD_NAMES, SHORT_FIELDS):
            physical_error, standardized_error = errors[short]
            phase = graph.node_phase[graph.field_nodes[field]].numpy()
            fields.setdefault((short, "all"), ErrorSums()).add(physical_error, standardized_error, 1.)
            for p in sorted(set(phase.tolist())):
                selected = phase == p
                fields.setdefault((short, str(p)), ErrorSums()).add(physical_error[selected], standardized_error[selected], 1.)
        current_pred = float(losses["predicted_current_A"])
        current_error = current_pred - data.applied_current_A
        current_errors.add(np.array([current_error]), np.array([current_error / abs(data.applied_current_A)]), 1.)
        collector = data.phase_nodes[7]
        voltage_pred, voltage_true = float(recovered[collector, 1].item()), float(future[collector, 1].item())
        window_rows.append(dict(scope="test", sample_index=sid, sample_id=data.sample_id,
            input_time_index=t, target_time_index=t+1, weight=1.,
            voltage_pred_V=voltage_pred, voltage_true_V=voltage_true,
            voltage_error_mV=1000*(voltage_pred-voltage_true), current_pred_A=current_pred,
            current_applied_A=data.applied_current_A, current_error_A=current_error))
        if step % 500 == 0 or step == len(pairs):
            print(f"evaluation test: {step}/{len(pairs)}", flush=True)
    result, current_result = reported.mean.result(), current_errors.result()
    def unit(short):
        return "mol/m^3" if short in ("ce", "cg", "cp") else "mV"
    rows = []
    for (short, phase), accumulator in fields.items():
        balanced = result[short+"_MSE"] if phase == "all" else result.get(f"{short}_phase{phase}_MSE")
        rows.append(dict(scope="test", field=short, phase=phase, unit=unit(short),
                         standardized_phase_balanced_MSE=balanced, **accumulator.result()))
    overall = dict(format_version="graphreaction_test_metrics_v1", scope="test", identity=identity,
        checkpoint=checkpoint, status="completed", windows=len(pairs), graphs=len(trajectories),
        weight_rule="1", window_weight_sum=reported.mean.denominator, losses=result,
        physical_node_pooled={short: dict(unit=unit(short), **fields[(short, "all")].result()) for short in SHORT_FIELDS},
        current=dict(RMSE_A=current_result["physical_RMSE"], MAE_A=current_result["physical_MAE"],
                     relative_RMSE_pct=result["current_relative_RMSE_pct"], count=len(pairs)),
        seconds=time.perf_counter()-started, test_access=len(pairs), optimizer_updates=0,
        prediction_mode="single step: true state at t -> prediction at t+1; no rollout")
    folder = Path(folder)
    write_json(folder / "overall_metrics.json", overall)
    write_csv(folder / "field_metrics.csv", rows, FIELD_COLUMNS)
    write_json(folder / "trajectory_metrics.json", {sid: a.result() for sid, a in trajectories.items()})
    trajectory_rows = [dict(sample_index=sid, trajectory_id=reader.dataset.samples[sid]["trajectory_id"],
                            windows=a.windows, **a.result()) for sid, a in trajectories.items()]
    write_csv(folder / "trajectory_metrics.csv", trajectory_rows, list(trajectory_rows[0]))
    write_csv(folder / "voltage_current.csv", window_rows, list(window_rows[0]))
    save_nrmse(folder, reported.fields(), scope="test", checkpoint=checkpoint, windows=len(pairs))
    write_json(folder / "window_mean_metrics.json", dict(format_version="window_arithmetic_metrics_v1",
        scope="test", checkpoint=checkpoint, windows=len(pairs), window_weight=1, metrics=result))
    return overall
