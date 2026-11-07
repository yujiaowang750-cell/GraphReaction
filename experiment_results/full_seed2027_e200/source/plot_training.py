"""从已经保存的历史记录绘制训练/Proxy曲线，不重新运行模型。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, path):
    if not history:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes[:2], ("CE2", "Q_ref")):
        axis.plot(epochs, [row[f"online_train_{metric}"] for row in history], label="Online train")
        # 续训前的验证记录保留；后续未验证的轮次留空，不能伪造为0或延长旧曲线。
        validated = [row for row in history if row.get(f"proxy_{metric}") not in (None, "")]
        if validated:
            axis.plot([row["epoch"] for row in validated],
                      [row[f"proxy_{metric}"] for row in validated], label="Earlier proxy validation")
        full_validated = [row for row in history if row.get(f"val_{metric}") not in (None, "")]
        if full_validated:
            axis.plot([row["epoch"] for row in full_validated],
                      [row[f"val_{metric}"] for row in full_validated], label="Full validation (fixed)")
        axis.set(xlabel="Epoch", ylabel=metric, yscale="log")
        axis.legend()
        axis.grid(alpha=.25)
    axes[2].plot(epochs, [row["lr"] for row in history])
    axes[2].set(xlabel="Epoch", ylabel="Learning rate")
    axes[2].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=160)
    plt.close(fig)
