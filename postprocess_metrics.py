"""从已有结果整理 CE2、Q_ref、NRMSE_delta 和七字段曲线，不运行模型。"""

import argparse
import math
from pathlib import Path

from metric_reporting import fields_from_saved, save_nrmse
from plot_metrics import plot_metric_curves
from run_files import read_csv, write_csv, write_json, file_hash
from training_metrics import BASE_METRICS, SHORT_FIELDS
import json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("指标中存在 NaN/Inf。")
    return result


def build_epoch_rows(history, training, evaluations):
    """只使用真实存在的值；旧抽样加权训练记录不能直接改名为算术平均。"""
    rows = []
    for old in history:
        row = dict(epoch=int(old["epoch"]), validation_status="not_recorded",
                   validation_scope=None, train_metric_source=None)
        for metric in BASE_METRICS:
            canonical = f"train_{metric}"
            if number(old.get(canonical)) is not None:
                row[canonical] = number(old[canonical])
                row["train_metric_source"] = "epoch_window_arithmetic_mean"
            elif training.get("train_weight") == 1:
                row[canonical] = number(old.get(f"online_train_{metric}"))
                row["train_metric_source"] = "online_train_alias; window weights are 1"
            else:
                row[canonical] = None
                row["train_metric_source"] = "unavailable: legacy sampled-window weighting"
            row[f"val_{metric}"] = number(old.get(f"val_{metric}"))
        if row["val_CE2"] is not None:
            row["validation_status"] = "recorded_epoch_validation"
            row["validation_scope"] = "full" if training.get("dataset_mode") == "full" else "proxy"
        for prefix in ("train", "val"):
            for field in SHORT_FIELDS:
                for suffix in ("node_pooled_MSE", "NRMSE_delta"):
                    key = f"{prefix}_{field}_{suffix}"
                    row[key] = number(old.get(key))
        rows.append(row)

    validation_scope = "full" if training.get("dataset_mode") == "full" else "proxy"
    for overall, arithmetic in evaluations:
        if overall["scope"] != validation_scope or arithmetic is None:
            continue
        epoch = overall["checkpoint"]["epoch"]
        row = next((r for r in rows if r["epoch"] == epoch), None)
        if row is None:
            raise ValueError(f"评价checkpoint第{epoch}轮不在history中。")
        # 固定checkpoint评价只能放在真实的那个epoch，不能铺满历史。
        for metric in BASE_METRICS:
            value = arithmetic[metric]
            existing = row[f"val_{metric}"]
            if existing is not None and not math.isclose(existing, value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"第{epoch}轮的已有验证值与固定checkpoint评价不一致。")
            row[f"val_{metric}"] = value
        row["validation_status"] = "fixed_checkpoint_evaluation"
        row["validation_scope"] = validation_scope
    return rows


def summarize_selection(history, rows, summary):
    """保留实验实际选择依据，不把事后的一次验证伪装成验证选best。"""
    epoch = summary["best_epoch"]
    old = next(r for r in history if int(r["epoch"]) == epoch)
    row = next(r for r in rows if r["epoch"] == epoch)
    metric = summary["selection_metric"]
    source = "val" if metric == "val_CE2" else "online_train"
    selected_q = number(old.get("best_Q_ref_at_CE2_best"))
    if selected_q is None:
        selected_q = row["val_Q_ref"] if source == "val" else number(old.get("online_train_Q_ref"))
    q_rows = [(number(r.get(f"{source}_Q_ref")), int(r["epoch"])) for r in history]
    q_rows = [(v, e) for v, e in q_rows if v is not None]
    minimum = min(q_rows) if q_rows else (None, None)
    return dict(selection_metric=metric, best_epoch=epoch,
                best_CE2=summary.get("best_train_CE2") if source != "val" else row["val_CE2"],
                best_Q_ref_at_CE2_best=selected_q, best_Q_ref_scope=source,
                Q_ref_min_diagnostic=minimum[0], Q_ref_min_epoch_diagnostic=minimum[1],
                val_CE2_at_saved_best=row["val_CE2"], val_Q_ref_at_saved_best=row["val_Q_ref"],
                validation_selected_best_epoch=epoch if source == "val" else None)


def process_run(run, output=None):
    run = Path(run).resolve()
    output = Path(output).resolve() if output else run / "metrics_v2"
    # 新目录保存派生结果，不能指向原始评价目录。
    if output == run or output == run / "evaluation" or (run / "evaluation") in output.parents:
        raise ValueError("请将派生指标放在独立目录，例如 run/metrics_v2。")
    history = read_csv(run / "history.csv")
    if not history:
        raise ValueError("history.csv 没有已完成的epoch。")
    config, summary = read_json(run / "config.json"), read_json(run / "training_summary.json")
    protected = [run / name for name in ("history.csv", "history.json", "config.json", "training_summary.json", "loss_curve.png")]
    protected += list((run / "checkpoints").glob("*.pt"))
    protected += [p for p in (run / "evaluation").rglob("*") if p.is_file()]
    before = {str(p.relative_to(run)): file_hash(p) for p in protected if p.exists()}
    output.mkdir(parents=True, exist_ok=True)
    evaluations, available_nrmse, unavailable = [], {}, []
    for path in sorted((run / "evaluation").glob("*/overall_metrics.json")):
        overall = read_json(path)
        if overall["identity"] != config["identity"] or overall["status"] != "completed":
            raise ValueError(f"评价身份或状态不匹配：{path}")
        checkpoint = Path(overall["checkpoint"]["path"])
        if not checkpoint.exists() or file_hash(checkpoint) != overall["checkpoint"]["sha256"]:
            raise ValueError(f"评价checkpoint文件哈希不匹配：{checkpoint}")
        arithmetic = overall["losses"] if str(overall.get("weight_rule")) == "1" else None
        mean_path = path.parent / "window_mean_metrics.json"
        if mean_path.exists():
            saved = read_json(mean_path)
            if saved["checkpoint"] != overall["checkpoint"]:
                raise ValueError("窗口算术平均的checkpoint不一致。")
            arithmetic = saved["metrics"]
        evaluations.append((overall, arithmetic))
        nrmse_path = path.parent / "nrmse_delta.json"
        if nrmse_path.exists():
            saved = read_json(nrmse_path)
            if saved["checkpoint"] != overall["checkpoint"] or saved["window_weight"] != 1:
                raise ValueError("NRMSE来源的checkpoint或窗口权重不匹配。")
            fields = saved["fields"]
        else:
            try:
                fields = fields_from_saved(overall)
            except ValueError as error:
                unavailable.append(dict(scope=overall["scope"], reason=str(error)))
                continue
        save_nrmse(output / overall["scope"], fields, scope=overall["scope"],
                   checkpoint=overall["checkpoint"], windows=overall["windows"])
        available_nrmse[overall["scope"]] = fields

    rows = build_epoch_rows(history, config["identity"]["training"], evaluations)
    validation_scope = "full" if config["identity"]["training"].get("dataset_mode") == "full" else "proxy"
    for overall, _ in evaluations:
        if overall["scope"] == validation_scope and validation_scope in available_nrmse:
            row = next(r for r in rows if r["epoch"] == overall["checkpoint"]["epoch"])
            for field, metrics in available_nrmse[validation_scope].items():
                row[f"val_{field}_NRMSE_delta"] = metrics["NRMSE_delta"]
                row[f"val_{field}_node_pooled_MSE"] = metrics["standardized_node_pooled_MSE"]
    selection = summarize_selection(history, rows, summary)
    write_csv(output / "epoch_metrics.csv", rows, list(rows[0]))
    write_json(output / "selection.json", selection)
    plot_metric_curves(rows, output, selection)
    missing = [row["epoch"] for row in rows if row["val_CE2"] is None]
    checkpoint_names = [p.name for p in (run / "checkpoints").glob("*.pt")]
    availability = dict(format_version="metric_reporting_v2", run=str(run),
        train_epochs=sum(row["train_CE2"] is not None for row in rows), total_epochs=len(rows),
        validation_epochs=[row["epoch"] for row in rows if row["val_CE2"] is not None],
        missing_validation_epochs=missing, selection=selection, nrmse_scopes=list(available_nrmse),
        unavailable_nrmse=unavailable, saved_checkpoint_files=checkpoint_names,
        reconstruction="No retraining, no model inference, no missing-epoch interpolation",
        train_nrmse_epochs=sum(row["train_ce_NRMSE_delta"] is not None for row in rows),
        original_file_hashes=before)
    write_json(output / "availability.json", availability)
    lines = ["# CE2、Q_ref 与 NRMSE_delta 派生指标", "",
             f"来源实验：`{run.name}`。本次只读取历史结果，没有训练或模型前向。", "",
             f"原实验选择依据：`{selection['selection_metric']}`；best epoch：{selection['best_epoch']}。",
             f"best_Q_ref_at_CE2_best = {selection['best_Q_ref_at_CE2_best']}，统计范围为 `{selection['best_Q_ref_scope']}`。",
             "它表示CE2最佳轮对应的Q_ref，不表示Q_ref自身达到最低。", "",
             f"训练CE2/Q_ref可用轮数：{availability['train_epochs']}/{len(rows)}。",
             f"验证可用轮次：{availability['validation_epochs']}。缺少的{len(missing)}轮保持空值，图上不连出虚构曲线。",
             "历史只保留best/last时，无法仅凭这两份权重恢复其余轮的验证结果。重新评价last只能补它对应的一轮。",
             "训练每轮字段MSE采用窗口/区域平均，不能由它反推节点合并的训练NRMSE_delta；缺少的充分统计必须用对应checkpoint重新评价。", "",
             "NRMSE_delta = sqrt(standardized_node_pooled_MSE)，不使用字段权重或电解液区域权重。",
             "旧子集评价若包含population_weight，不能将加权节点MSE直接冒充不加权公式；availability.json会指出不可恢复项。", "",
             "## 文件", "", "- [每轮指标](epoch_metrics.csv)", "- [选择记录](selection.json)",
             "- [CE2/Q_ref曲线](ce2_q_ref_curves.png)", "- [七字段MSE曲线](field_mse_curves.png)",
             "- [可恢复性与原文件哈希](availability.json)", ""]
    for scope in available_nrmse:
        lines.append(f"- [{scope}的七字段NRMSE_delta]({scope}/nrmse_delta.csv)")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: file_hash(run / name) for name in before}
    if before != after:
        raise RuntimeError("原始实验文件发生变化。")
    return availability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = process_run(args.run_dir, args.output)
    print(f"训练轮数：{result['train_epochs']}；可用验证轮次：{result['validation_epochs']}")
    print(f"NRMSE_delta可用范围：{result['nrmse_scopes']}；原文件哈希保持一致。")


if __name__ == "__main__":
    main()
