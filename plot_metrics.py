"""画已记录的指标；缺失验证轮次留空，不插值，也不复制best结果。"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training_metrics import SHORT_FIELDS


FIELD_LABELS = ("Electrolyte concentration", "Electrolyte potential", "Graphite lithiation",
                "AM lithiation", "AM potential", "CBD potential", "Collector potential")


def values(rows, key):
    return np.asarray([float(row[key]) if row.get(key) not in (None, "") else np.nan for row in rows])


def mark_best(axis, selection):
    epoch = selection.get("best_epoch")
    if epoch is None:
        return
    source = "validation" if selection.get("selection_metric") == "val_CE2" else "train"
    axis.axvline(epoch, color="0.4", linestyle="--", linewidth=1,
                 label=f"Saved best by {source} CE2 (epoch {epoch})")


def plot_metric_curves(rows, folder, selection):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    epochs = values(rows, "epoch")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, metric in zip(axes, ("CE2", "Q_ref")):
        for prefix, label, color in (("train", "Train (online)", "tab:blue"),
                                     ("val", "Validation (fixed checkpoint)", "tab:orange")):
            y = values(rows, f"{prefix}_{metric}")
            if np.isfinite(y).any():
                axis.plot(epochs, y, color=color, label=label,
                          marker="o" if prefix == "val" else None, markersize=4)
        mark_best(axis, selection)
        axis.set(xlabel="Epoch", ylabel=metric, yscale="log")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    count = int(np.isfinite(values(rows, "val_CE2")).sum())
    fig.suptitle(f"CE2 and Q_ref | recorded validation epochs: {count}/{len(rows)}")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"ce2_q_ref_curves.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for axis, field, label in zip(axes.flat, SHORT_FIELDS, FIELD_LABELS):
        for prefix, name, color in (("train", "Train (online)", "tab:blue"),
                                    ("val", "Validation", "tab:orange")):
            y = values(rows, f"{prefix}_{field}_MSE")
            if np.isfinite(y).any():
                axis.plot(epochs, y, label=name, color=color,
                          marker="o" if prefix == "val" else None, markersize=4)
        mark_best(axis, selection)
        axis.set(title=label, xlabel="Epoch", ylabel="Standardized increment MSE", yscale="log")
        axis.grid(alpha=.25)
    axes.flat[7].axis("off")
    axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=10)
    axes.flat[8].axis("off")
    axes.flat[8].text(.05, .65, "Missing validation epochs remain blank.\n\n"
                     "Electrolyte MSE uses 0.5 / 0.5 phase balance.\n"
                     "These field losses are not node-pooled NRMSE_delta.",
                     va="top", fontsize=10, linespacing=1.8)
    fig.suptitle("Seven physical-field losses by epoch", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"field_mse_curves.{suffix}", dpi=180)
    plt.close(fig)
