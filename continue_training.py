"""把已完成141轮的全量实验接着训练；原实验只读，新目录记录续训。"""

import argparse
import ast
import copy
import json
from datetime import datetime
from pathlib import Path

import torch

from learning_rate import MIN_LR, NUM_EPOCHS
from metric_reporting import EpochMetrics
from run_files import copy_atomic, file_hash, write_csv, write_json
from schema import model_configuration, content_hash
from training_checkpoint import TRAINING_CHECKPOINT_SCHEMA
from training_config import training_configuration
from training_metrics import BASE_METRICS


def executable_tree(path):
    """比较计算代码时忽略注释和docstring，不忽略实际语句。"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def read_parent(parent_run, dataset, epochs):
    """显式检查可延长的旧实验；不绕过普通 --resume 的严格身份检查。"""
    parent_run = Path(parent_run).resolve()
    config = json.loads((parent_run / "config.json").read_text(encoding="utf-8"))
    identity = config["identity"]
    old = identity["training"]
    if (identity["model"] != model_configuration() or identity["data"] != dataset.identity
            or old["dataset_mode"] != "full"):
        raise ValueError("续训要求相同的X13/b30模型、全量数据和划分。")
    if old["max_epochs"] != NUM_EPOCHS or old["T_max"] != NUM_EPOCHS or epochs <= NUM_EPOCHS:
        raise ValueError("此入口只延长已完成的141轮余弦实验，目标轮数必须大于141。")
    current = training_configuration("full")
    for key in ("optimizer", "learning_rate", "weight_decay", "betas", "adam_eps", "accumulation",
                "clip_norm", "scheduler", "eta_min", "monitor", "best_rule", "early_stopping",
                "validate_each_epoch", "amp", "tf32", "train_weight", "evaluation_scopes", "seed"):
        if old[key] != current[key]:
            raise ValueError(f"原训练协议不一致：{key}")
    # 这四个文件已专门对拍：仅增加指标、配置身份及本次续训入口。
    reviewed = {"train.py", "training_step.py", "training_config.py", "evaluation.py"}
    here = Path(__file__).resolve().parent
    for name, digest in identity["source"].items():
        saved = parent_run / "source" / name
        if file_hash(saved) != digest:
            raise ValueError(f"原实验源码快照哈希不匹配：{name}")
        if name not in reviewed and executable_tree(saved) != executable_tree(here / name):
            raise ValueError(f"模型或数据计算代码与原实验不一致：{name}")
    paths = {name: parent_run / "checkpoints" / f"{name}.pt" for name in ("last", "best")}
    payloads = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    for name, payload in payloads.items():
        if payload["schema"] != TRAINING_CHECKPOINT_SCHEMA or payload["identity"] != identity:
            raise ValueError(f"原{name}断点身份不匹配。")
        if payload["pending_gradients"] is not None or payload["accumulation_pending_windows"] != 0:
            raise ValueError("不能从未完成的梯度累积中延长实验。")
    state = payloads["last"]["state"]
    if (state["epoch"] != NUM_EPOCHS or state["cursor"] != 0 or not state["complete"]
            or state["stop_reason"] != "epoch_budget" or len(state["history"]) != NUM_EPOCHS):
        raise ValueError("必须使用完整结束第141轮的last.pt。")
    if state["early_stop_counter"] >= old["early_stopping"]:
        raise ValueError("原实验已触发早停，不能通过续训重置早停计数。")
    scheduler = payloads["last"]["scheduler"]
    if scheduler["T_max"] != NUM_EPOCHS or scheduler["last_epoch"] != NUM_EPOCHS:
        raise ValueError("原余弦调度器尚未正确完成141轮。")
    if any(group["lr"] != MIN_LR for group in payloads["last"]["optimizer"]["param_groups"]):
        raise ValueError("原last断点学习率不是1e-5。")
    best_epoch = state["best_epoch"]
    best_row = min(state["history"], key=lambda row: row["online_train_CE2"])
    if (payloads["best"]["state"]["epoch"] != best_epoch or best_row["epoch"] != best_epoch
            or best_row["online_train_CE2"] != state["best_train_CE2"]):
        raise ValueError("原best权重、训练CE2选择记录不一致。")
    info = dict(schema="full_141_to_fixed_lr_v1", parent_run=str(parent_run),
                parent_identity_hash=content_hash(identity), parent_config_sha256=file_hash(parent_run / "config.json"),
                parent_last_sha256=file_hash(paths["last"]), parent_best_sha256=file_hash(paths["best"]),
                start_epoch=NUM_EPOCHS + 1, fixed_learning_rate=MIN_LR,
                inherited_best_epoch=best_epoch, inherited_best_train_CE2=state["best_train_CE2"],
                inherited_early_stop_counter=state["early_stop_counter"], reviewed_source_files=sorted(reviewed),
                old_scheduler=scheduler, old_training_schema=old["schema"])
    return dict(info=info, last=payloads["last"], best=payloads["best"])


def continued_state(payload):
    """延长预算，但不重置更新次数、随机状态、best或早停计数。"""
    state = copy.deepcopy(payload["state"])
    state.update(complete=False, stop_reason=None, pending_diagnostic=False,
                 reported=EpochMetrics().state())
    best_row = state["history"][state["best_epoch"] - 1]
    state["best_Q_ref_at_CE2_best"] = best_row["online_train_Q_ref"]
    return state


def copy_parent_records(run, parent, state, history_columns):
    """原文件不改写；新目录继承日志、best原文件以及可追溯来源。"""
    source = Path(parent["info"]["parent_run"])
    copy_atomic(source / "update_metrics.csv", run / "update_metrics.csv")
    copy_atomic(source / "checkpoints/best.pt", run / "checkpoints/best.pt")
    for row in state["history"]:
        for key in history_columns:
            row.setdefault(key, None)
        for metric in (*BASE_METRICS, "current_relative_RMSE_pct"):
            row[f"train_{metric}"] = row[f"online_train_{metric}"]
        selected = state["history"][row["best_epoch"] - 1]
        row["best_Q_ref_at_CE2_best"] = selected["online_train_Q_ref"]
        row["best_Q_ref_scope"] = "online_train"
    write_csv(run / "history.csv", state["history"], history_columns)
    write_json(run / "history.json", state["history"])
    write_json(run / "continuation.json", parent["info"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=193)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = json.loads((args.parent_run / "config.json").read_text(encoding="utf-8"))
    from full_training_windows import FullTrainingWindows
    from train import run_training
    dataset = FullTrainingWindows(config["dataset_root"])
    run = args.run_dir or Path(__file__).resolve().parent / "training_runs" / (
        "full_continue_fixed_lr_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_training(run, dataset, args.epochs, args.device, continue_from=args.parent_run)


if __name__ == "__main__":
    main()
