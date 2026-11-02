"""固定138/154轮权重，补算全量训练集，复用验证结果，导出四组同口径CSV。"""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from continue_training import executable_tree
from evaluation import evaluate_scope
from full_training_windows import FullTrainingWindows
from initialization import create_model
from run_files import file_hash, read_csv, write_csv, write_json
from schema import model_configuration
from train import configure_runtime, model_hash, runtime_sources
from training_checkpoint import TRAINING_CHECKPOINT_SCHEMA
from training_windows import WindowReader

RUNS = CODE / "training_runs"
EXPERIMENTS = {
    138: (RUNS / "full_train_ce2_20261026_215337",
          RUNS / "full_train_ce2_20261026_215337/evaluation/full"),
    154: (RUNS / "full_continue_193_fixed_lr_20261030_101435",
          RUNS / "validation_best154_20261030_194340"),
}
TEST_CSV = RUNS / "test_best138_20261030_192828/trajectory_metrics.csv"
FIELDS = ("ce", "phi_e", "cg", "cp", "phi_p", "phi_CBD", "V")
EXTRA_COLUMNS = ["checkpoint_epoch", "dataset_split", "evaluation_mode",
                 "main_loss", "current_loss_weight", "weighted_current_loss",
                 "total_loss", "reference_main_loss", "reference_total_loss"]
ATOL = 1e-12  # 沿用项目指标检查的绝对容差，不放宽。


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_checkpoint(run, epoch, dataset):
    config = read_json(run / "config.json")
    path = run / "checkpoints/best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != TRAINING_CHECKPOINT_SCHEMA or saved["identity"] != config["identity"]:
        raise ValueError(f"epoch {epoch}: checkpoint与原实验身份不一致")
    if saved["state"]["epoch"] != epoch:
        raise ValueError(f"所选best不是第{epoch}轮，拒绝替换权重")
    if saved["identity"]["model"] != model_configuration() or saved["identity"]["data"] != dataset.identity:
        raise ValueError("模型、数据划分或训练归一化统计与原实验不一致")
    if Path(config["dataset_root"]).resolve() != dataset.root:
        raise ValueError("不是原全量数据集目录")
    # 原141轮实验后只增加过指标记录和续训入口；模型/损失/数据计算必须一致。
    for name, digest in config["identity"]["source"].items():
        snapshot = run / "source" / name
        if file_hash(snapshot) != digest:
            raise ValueError(f"原实验源码快照改变：{name}")
        if name not in {"train.py", "training_step.py", "training_config.py", "evaluation.py"}:
            if executable_tree(snapshot) != executable_tree(CODE / name):
                raise ValueError(f"当前计算源码不同于原实验：{name}")
    checkpoint = dict(path=str(path), sha256=file_hash(path), epoch=epoch,
                      update=saved["state"]["updates"])
    return saved, checkpoint


def split_loss(values, epoch, split):
    """从已有字段MSE拆出主损失/电流项，保留原CE2和Q_ref数值。"""
    losses = [float(values[f"{field}_MSE"]) for field in FIELDS]
    main = (2 * losses[0] + math.fsum(losses[1:])) / 8
    reference = math.fsum(losses) / 7
    current = 0.1 * float(values["current_relative_MSE"])
    return dict(checkpoint_epoch=epoch, dataset_split=split,
                evaluation_mode="fixed_checkpoint_eval",
                main_loss=main, current_loss_weight=0.1, weighted_current_loss=current,
                total_loss=float(values["CE2"]), reference_main_loss=reference,
                reference_total_loss=float(values["Q_ref"]))


def export_group(folder, scope, epoch, dataset, checkpoint, identity, output):
    """读取已有离线结果，校验完整覆盖，并沿用测试CSV的全部原始列。"""
    overall = read_json(folder / "overall_metrics.json")
    if (overall["status"] != "completed" or overall["scope"] != scope
            or overall["checkpoint"] != checkpoint or overall["identity"] != identity
            or overall["weight_rule"] != "1" or overall["optimizer_updates"] != 0):
        raise ValueError(f"不能复用身份/范围不一致的评价：{folder}")
    expected = dataset.pairs(scope)
    windows = read_csv(folder / "voltage_current.csv")
    actual = [(int(row["sample_index"]), int(row["input_time_index"])) for row in windows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("评价窗口不完整、重复或混入其他划分")
    if any(row["scope"] != scope or float(row["weight"]) != 1
           or int(row["target_time_index"]) != int(row["input_time_index"]) + 1 for row in windows):
        raise ValueError("评价范围、窗口权重或单步时间索引错误")
    counts = Counter(sid for sid, _ in actual)
    if (overall["windows"] != len(expected) or overall["graphs"] != len(counts)
            or overall["window_weight_sum"] != len(expected)):
        raise ValueError("总窗口数/轨迹数/分母不一致")
    trajectory = read_json(folder / "trajectory_metrics.json")
    if {int(sid) for sid in trajectory} != set(counts):
        raise ValueError("逐轨迹指标未完整覆盖所选数据")
    split = "train" if scope == "train" else "validation"
    columns = list(read_csv(TEST_CSV)[0])  # 直接沿用测试CSV的原字段和顺序。
    rows = []
    errors = {}

    def check(name, actual_value, expected_value):
        error = abs(float(actual_value) - float(expected_value))
        if not math.isfinite(error) or error > ATOL:
            raise ValueError(f"{name}: 误差{error}超过{ATOL}")
        errors[name] = error

    def check_losses(label, values, extra):
        check(label + ":CE2", values["CE2"], extra["main_loss"] + extra["weighted_current_loss"])
        check(label + ":Q_ref", values["Q_ref"], extra["reference_main_loss"] + extra["weighted_current_loss"])
        for field in ("ce", "phi_e"):
            check(label + ":" + field, values[field + "_MSE"],
                  (values[field + "_phase2_MSE"] + values[field + "_phase5_MSE"]) / 2)

    for sid in sorted(counts):
        values = trajectory[str(sid)]
        row = dict(sample_index=sid, trajectory_id=dataset.samples[sid]["trajectory_id"],
                   windows=counts[sid], **values)
        extra = split_loss(values, epoch, split)
        check_losses(str(sid), values, extra)
        row.update(extra)
        rows.append({name: row[name] for name in columns + EXTRA_COLUMNS})
    values = overall["losses"]
    extra = split_loss(values, epoch, split)
    check_losses("overall", values, extra)
    for metric in values:
        if metric == "current_relative_RMSE_pct":
            check(metric, values[metric], 100 * math.sqrt(values["current_relative_MSE"]))
        else:
            check("trajectory_merge:" + metric, values[metric],
                  math.fsum(row[metric] * row["windows"] for row in rows) / len(expected))
    current_mse = math.fsum((float(row["current_error_A"]) / abs(float(row["current_applied_A"])))**2
                            for row in windows) / len(expected)
    check("current_from_window_CSV", values["current_relative_MSE"], current_mse)
    basename = f"epoch{epoch}_{split}"
    write_csv(output / f"{basename}.csv", rows, columns + EXTRA_COLUMNS)
    summary = dict(checkpoint_epoch=epoch, dataset_split=split, trajectories=len(rows),
                   windows=len(expected), **values)
    summary.update(extra)
    write_csv(output / f"{basename}_overall.csv", [summary], list(summary))
    return summary, dict(source=str(folder), checkpoint=checkpoint,
                         trajectories=len(rows), windows=len(expected),
                         csv=f"{basename}.csv", overall_csv=f"{basename}_overall.csv",
                         window_coverage="exact; no duplicates", tolerance=ATOL,
                         max_absolute_error=max(errors.values()), errors=errors)


def protected_hashes():
    paths = [RUNS / "active_training.json"]
    for run, validation in EXPERIMENTS.values():
        paths += [run / name for name in ("config.json", "history.csv", "training_summary.json",
                                         "checkpoints/best.pt", "checkpoints/last.pt")]
        paths += [p for p in validation.iterdir() if p.is_file()]
    return {str(path): file_hash(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="只核验并导出已有验证结果")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = FullTrainingWindows()
    for scope, count, graphs in (("train", 22927, 65), ("full", 4917, 14)):
        pairs = dataset.pairs(scope)
        if len(pairs) != count or len({sid for sid, _ in pairs}) != graphs:
            raise ValueError("全量训练/验证划分不符合本次要求")
    before = protected_hashes()
    sources_before = runtime_sources()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = dict(format_version="fixed_138_154_train_validation_v1",
                    dataset_identity=dataset.identity, script_sha256=file_hash(__file__),
                    model=model_configuration(), source=sources_before)
    manifest_path = output / "export_config.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("输出目录属于其他配置，拒绝覆盖")
    write_json(manifest_path, manifest)
    summaries, checks = {}, {}
    payloads = {}
    for epoch, (run, validation) in EXPERIMENTS.items():
        saved, checkpoint = checked_checkpoint(run, epoch, dataset)
        payloads[epoch] = (saved, checkpoint)
        summaries[f"{epoch}_validation"], checks[f"{epoch}_validation"] = export_group(
            validation, "full", epoch, dataset, checkpoint, saved["identity"], output)
        print(f"Reused validation epoch={epoch}: 14 trajectories, 4917 windows", flush=True)
    if args.prepare_only:
        print("Prepared: checkpoints verified; both validation CSVs exported; training evaluation still missing.", flush=True)
        return
    configure_runtime(args.device)
    for epoch, (saved, checkpoint) in payloads.items():
        folder = output / f"evaluation_epoch{epoch}_train"
        if not (folder / "overall_metrics.json").exists():
            write_json(output / "status.json", dict(status="running", checkpoint_epoch=epoch,
                       scope="train", windows=22927, started_at=stamp, optimizer_updates=0))
            model = create_model(device=args.device)
            model.load_state_dict(saved["model"], strict=True)
            initial_hash = model_hash(model)
            reader = WindowReader(dataset, args.device)
            print(f"START fixed epoch={epoch}, train=65 trajectories /22927 windows", flush=True)
            result = evaluate_scope(model, reader, "train", folder, saved["identity"], checkpoint)
            if model_hash(model) != initial_hash or model.training:
                raise RuntimeError("固定模型权重或eval状态发生变化")
            if reader.loaded_samples != {sid for sid, _ in dataset.pairs("train")}:
                raise RuntimeError("实际训练轨迹读取集合不正确")
            write_json(folder / "evaluation_audit.json", dict(status="PASS", checkpoint=checkpoint,
                model_weights_unchanged=True, optimizer_updates=0, normalization_refitted=False,
                loaded_train_samples=sorted(reader.loaded_samples), training_windows=22927,
                validation_dynamic_reads=0, test_dynamic_reads=0))
            print(f"DONE epoch={epoch}, seconds={result['seconds']:.2f}, CE2={result['losses']['CE2']}", flush=True)
            del model, reader
        audit = read_json(folder / "evaluation_audit.json")
        if audit["status"] != "PASS" or audit["checkpoint"] != checkpoint:
            raise ValueError("已有训练评价缺少匹配的完整审计，不能复用")
        key = f"{epoch}_train"
        summaries[key], checks[key] = export_group(folder, "train", epoch, dataset,
                                                   checkpoint, saved["identity"], output)
        write_csv(output / "summary.csv", list(summaries.values()), list(summaries[key]))
    if before != protected_hashes() or sources_before != runtime_sources():
        raise RuntimeError("原训练记录、权重、验证结果或训练计算源码发生变化")
    ordered = [summaries[f"{epoch}_{split}"] for epoch in (138, 154) for split in ("train", "validation")]
    write_csv(output / "summary.csv", ordered, list(ordered[0]))
    write_json(output / "export_checks.json", dict(status="PASS", groups=checks,
        max_absolute_error=max(item["max_absolute_error"] for item in checks.values()),
        tolerance=ATOL, original_files_unchanged=True, original_hashes=before,
        training_sources_unchanged=True, optimizer_updates=0, test_dynamic_reads=0,
        training_status="still paused at epoch155", selection_rule_unchanged="online_train_CE2"))
    write_json(output / "status.json", dict(status="completed", completed_at=datetime.now(
        ZoneInfo("Asia/Shanghai")).isoformat(), groups=4, optimizer_updates=0))
    print(json.dumps(dict(status="completed", output=str(output), results=ordered), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
