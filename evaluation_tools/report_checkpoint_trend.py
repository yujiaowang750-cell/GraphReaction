"""把已完成的固定评价与历史在线指标分组整理，输出CSV和科研曲线。"""

import argparse
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from run_files import file_hash, read_csv, write_csv, write_json
from export_checkpoint_comparison import FIELDS, RUNS, read_json

EPOCHS = (138, 141, 154, 155)
PANELS = (
    ("CE2", "Total CE2"),
    ("concentration_group_MSE", "Concentration group: (2 ce + cg + cp) / 4"),
    ("potential_group_MSE", "Potential group: (phi_e + phi_p + phi_CBD) / 3"),
    ("V_MSE", "Terminal voltage: standardized increment MSE"),
    ("current_relative_MSE", "Current: relative squared error"),
    ("Q_ref", "Equal-field reference: Q_ref"),
)


def grouped(values):
    """组内均值和对CE2的贡献分别记录；端电压不重复计入电势组。"""
    v = {name: float(values[name]) for name in [f"{f}_MSE" for f in FIELDS]
         + ["CE2", "Q_ref", "current_relative_MSE"]}
    concentration_sum = 2 * v["ce_MSE"] + v["cg_MSE"] + v["cp_MSE"]
    potential_sum = v["phi_e_MSE"] + v["phi_p_MSE"] + v["phi_CBD_MSE"]
    v.update(concentration_group_MSE=concentration_sum / 4,
             potential_group_MSE=potential_sum / 3,
             concentration_CE2_contribution=concentration_sum / 8,
             potential_CE2_contribution=potential_sum / 8,
             voltage_CE2_contribution=v["V_MSE"] / 8,
             current_CE2_contribution=0.1 * v["current_relative_MSE"],
             current_relative_RMSE_pct=100 * math.sqrt(v["current_relative_MSE"]))
    total = math.fsum(v[name + "_CE2_contribution"] for name in
                     ("concentration", "potential", "voltage", "current"))
    error = abs(v["CE2"] - total)
    if not math.isfinite(error) or error > 1e-12:
        raise ValueError(f"分组贡献不能恢复原CE2：误差{error}")
    v["CE2_reconstruction_error"] = error
    return v


def draw_groups(rows, path, online=False):
    """固定评价只画四个实测点；在线训练单独画，不填充缺失验证结果。"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    styles = (("train", "#2457A6", "o", "Train (fixed weights)"),
              ("validation", "#C06A19", "s", "Validation (fixed weights)"))
    if online:
        styles = (("train", "#2457A6", "", "Online train: changing weights within each epoch"),)
    for axis, (metric, title) in zip(axes.flat, PANELS):
        for split, color, marker, label in styles:
            points = sorted([r for r in rows if r["dataset_split"] == split], key=lambda r: r["epoch"])
            axis.plot([r["epoch"] for r in points], [r[metric] for r in points],
                      color=color, marker=marker, markersize=6, linewidth=1.5,
                      linestyle="-" if online else "--", label=label)
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Dimensionless loss")
        axis.grid(alpha=0.22)
        if online:
            axis.set_yscale("log")
            axis.axvline(141, color="#777777", linewidth=1, linestyle=":")
            axis.set_xticks((1, 30, 60, 90, 120, 141, 155))
        else:
            axis.set_xticks(EPOCHS)
            axis.set_xlim(137, 156)
            axis.set_ylim(bottom=0)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(fontsize=9, loc="best")
    title = ("Online training groups | epochs 1-155 | logarithmic y axes" if online else
             "Fixed-checkpoint evaluation | 65 train / 14 validation trajectories\n"
             "Only four saved checkpoints; dashed segments connect observations, not per-epoch measurements")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def draw_fields(rows, path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    for axis, field in zip(axes.flat, FIELDS):
        for split, color, marker in (("train", "#2457A6", "o"), ("validation", "#C06A19", "s")):
            points = [r for r in rows if r["dataset_split"] == split]
            axis.plot([r["epoch"] for r in points], [r[field + "_MSE"] for r in points],
                      color=color, marker=marker, linestyle="--", linewidth=1.5, label=split)
        axis.set_title(field, fontsize=12)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Standardized increment MSE")
        axis.set_xticks(EPOCHS)
        axis.set_xlim(137, 156)
        axis.set_ylim(0, max(r[field + "_MSE"] for r in rows) * 1.1)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", ncol=2, frameon=False)
    axes.flat[-1].text(.05, .78,
        "ce / phi_e retain phase 2 / phase 5 weights of 0.5 / 0.5.\n\n"
        "cg / cp: standardized lithiation-fraction increments.\n\n"
        "V: positive collector potential (negative collector fixed at 0 V).\n\n"
        "Missing checkpoints are not interpolated or reconstructed.",
        va="top", fontsize=11, linespacing=1.7)
    fig.suptitle("Seven physical fields | fixed-checkpoint evaluation", fontsize=15)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    checks = read_json(folder / "evaluation_checks.json")
    if checks["status"] != "PASS" or len(checks["groups"]) != 8:
        raise ValueError("先完成并检查全部8组固定评价")
    summary = read_csv(folder / "summary.csv")
    if [(int(r["checkpoint_epoch"]), r["dataset_split"]) for r in summary] != [
            (e, s) for e in EPOCHS for s in ("train", "validation")]:
        raise ValueError("固定评价的轮次/范围不完整")
    group_rows, field_rows, trajectory_rows, source_hashes = [], [], [], {}
    for row in summary:
        epoch, split = int(row["checkpoint_epoch"]), row["dataset_split"]
        scope = "train" if split == "train" else "full"
        source = Path(checks["groups"][f"{epoch}_{scope}"]["source"]) / "overall_metrics.json"
        raw = read_json(source)
        source_hashes[str(source)] = file_hash(source)
        group = dict(epoch=epoch, dataset_split=split, evaluation_mode="fixed_checkpoint_eval",
                     trajectories=int(row["trajectories"]), windows=int(row["windows"]), **grouped(row))
        group.update(voltage_physical_RMSE_mV=raw["physical_node_pooled"]["V"]["physical_RMSE"],
                     current_physical_RMSE_A=raw["current"]["RMSE_A"])
        group_rows.append(group)
        for field in FIELDS:
            pooled = raw["physical_node_pooled"][field]
            field_rows.append(dict(epoch=epoch, dataset_split=split, field=field,
                windows=int(row["windows"]), standardized_field_MSE=float(row[field + "_MSE"]),
                standardized_node_pooled_MSE=pooled["standardized_node_pooled_MSE"],
                NRMSE_delta=math.sqrt(pooled["standardized_node_pooled_MSE"]),
                physical_RMSE=pooled["physical_RMSE"], physical_MAE=pooled["physical_MAE"],
                physical_unit=pooled["unit"]))
        for trajectory in read_csv(folder / f"epoch{epoch}_{split}.csv"):
            trajectory_rows.append(dict(epoch=epoch, dataset_split=split,
                sample_index=int(trajectory["sample_index"]), trajectory_id=trajectory["trajectory_id"],
                windows=int(trajectory["windows"]), **grouped(trajectory)))
    write_csv(folder / "group_summary.csv", group_rows, list(group_rows[0]))
    write_csv(folder / "field_trends.csv", field_rows, list(field_rows[0]))
    write_csv(folder / "group_trajectory_metrics.csv", trajectory_rows, list(trajectory_rows[0]))

    history_path = RUNS / "full_continue_193_fixed_lr_20261030_101435/history.csv"
    history = read_csv(history_path)
    if [int(r["epoch"]) for r in history] != list(range(1, 156)):
        raise ValueError("在线历史不是完整1-155轮")
    source_hashes[str(history_path)] = file_hash(history_path)
    online = []
    for row in history:
        values = {name.removeprefix("online_train_"): value for name, value in row.items()
                  if name.startswith("online_train_")}
        online.append(dict(epoch=int(row["epoch"]), dataset_split="train",
                           evaluation_mode="online_train", lr=float(row["lr"]), **grouped(values)))
    write_csv(folder / "online_train_groups.csv", online, list(online[0]))

    changes = []
    movements = ((138, 141), (141, 154), (154, 155), (138, 154), (138, 155))
    metrics = [key for key, _ in PANELS] + [f + "_MSE" for f in FIELDS if f != "V"]
    for split in ("train", "validation"):
        lookup = {r["epoch"]: r for r in group_rows if r["dataset_split"] == split}
        for start, end in movements:
            for metric in metrics:
                a, b = lookup[start][metric], lookup[end][metric]
                changes.append(dict(dataset_split=split, start_epoch=start, end_epoch=end,
                    metric=metric, start_value=a, end_value=b, absolute_change=b-a,
                    relative_change_pct=100*(b/a-1) if a else None))
    write_csv(folder / "interval_changes.csv", changes, list(changes[0]))

    trajectory_counts = []
    for split in ("train", "validation"):
        for start, end in movements:
            a = {r["sample_index"]: r for r in trajectory_rows if r["dataset_split"] == split and r["epoch"] == start}
            b = {r["sample_index"]: r for r in trajectory_rows if r["dataset_split"] == split and r["epoch"] == end}
            if set(a) != set(b):
                raise ValueError("不同checkpoint的轨迹集合不一致")
            for metric in ("CE2", "potential_group_MSE", "V_MSE"):
                trajectory_counts.append(dict(dataset_split=split, start_epoch=start, end_epoch=end,
                    metric=metric, improved=sum(b[sid][metric] < a[sid][metric] for sid in a),
                    worsened=sum(b[sid][metric] > a[sid][metric] for sid in a),
                    unchanged=sum(b[sid][metric] == a[sid][metric] for sid in a), trajectories=len(a)))
    write_csv(folder / "trajectory_improvement_counts.csv", trajectory_counts, list(trajectory_counts[0]))

    directions = {}
    for split in ("train", "validation"):
        points = [r for r in group_rows if r["dataset_split"] == split]
        directions[split] = {metric: dict(
            values=[r[metric] for r in points],
            strictly_decreasing=all(b[metric] < a[metric] for a, b in zip(points, points[1:])),
            strictly_increasing=all(b[metric] > a[metric] for a, b in zip(points, points[1:])))
            for metric, _ in PANELS}
    max_error = max(r["CE2_reconstruction_error"] for r in group_rows + online + trajectory_rows)
    write_json(folder / "group_checks.json", dict(status="PASS", groups=8, fields=len(field_rows),
        trajectory_rows=len(trajectory_rows), online_epochs=len(online), source_hashes=source_hashes,
        report_script_sha256=file_hash(__file__),
        max_CE2_reconstruction_error=max_error, tolerance=1e-12,
        fixed_evaluation_separate_from_online=True, checkpoint_epochs=list(EPOCHS),
        directions_at_saved_points=directions, gradient_contributions_measured=False,
        selection_rule_unchanged="online_train_CE2"))
    draw_groups(group_rows, folder / "fixed_group_trends.png")
    draw_fields(group_rows, folder / "fixed_field_trends.png")
    draw_groups(online, folder / "online_group_trends.png", online=True)
    print("Saved grouped metrics and three figures. Max CE2 reconstruction error:", max_error, flush=True)
    for row in group_rows:
        print({k: row[k] for k in ("epoch", "dataset_split", "CE2", "concentration_group_MSE",
                                    "potential_group_MSE", "V_MSE", "current_relative_MSE")}, flush=True)


if __name__ == "__main__":
    main()
