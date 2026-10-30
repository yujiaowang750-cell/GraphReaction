"""GraphReaction：只训练，按训练CE2选best，停止后固定best评价。

从项目根目录启动：.venv/Scripts/python.exe GraphReaction/train.py
全量训练：... GraphReaction/train.py --dataset full --full-root Datasets
断点继续：... GraphReaction/train.py --resume <run目录>
短流程检查：... GraphReaction/train.py --epochs 2 --smoke-windows 8
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from diagnostics import TopologyCoverage, run_diagnostics, DIAGNOSTIC_COLUMNS, RATIO_EPS
from evaluation import evaluate_scope
from initialization import create_model
from learning_rate import create_scheduler, NUM_EPOCHS, INITIAL_LR
from plot_training import plot_history
from run_files import write_json, append_csv, file_hash, copy_atomic
from schema import model_configuration, configuration_hash, content_hash
from training_checkpoint import save_training_checkpoint, load_training_checkpoint, reconcile_csv
from training_config import training_configuration, SEED, ACCUMULATION, CLIP_NORM, EARLY_STOPPING
from training_metrics import MetricMean, BASE_METRICS
from training_step import train_group
from training_windows import TrainingWindows, WindowReader, DEFAULT_PROXY, order_hash
from full_training_windows import FullTrainingWindows
from metric_reporting import EpochMetrics, epoch_columns
from plot_metrics import plot_metric_curves

HERE = Path(__file__).resolve().parent
UPDATE_COLUMNS = ["update", "epoch", "cursor", "lr", "grad_norm_pre_clip", "clipped", "nonfinite_grad_count", "status", "order_sha256"]
HISTORY_COLUMNS = ["epoch", "updates", "lr", "next_lr", "order_sha256", "scheduler_name", "scheduler_config_hash",
    "best_proxy_Q_ref", "best_epoch", "epochs_since_best", "early_stop_counter", "is_best", "stop_reason",
    "grad_norm_pre_clip_mean", "grad_norm_pre_clip_max", "grad_clip_fraction", "nonfinite_grad_count",
    "epoch_train_seconds", "epoch_proxy_seconds", "cuda_max_memory_bytes",
    "selection_metric", "best_value", "best_train_CE2", "validation_status"] + [
    f"{scope}_{name}" for scope in ("online_train", "proxy") for name in (*BASE_METRICS,"current_relative_RMSE_pct")]
# 只为新实验扩列；旧history和旧MSE列不改写。
REPORT_COLUMNS = list(epoch_columns("train", EpochMetrics())) + list(epoch_columns("val", EpochMetrics()))
HISTORY_COLUMNS += ["best_Q_ref_at_CE2_best", "best_Q_ref_scope"] + REPORT_COLUMNS


def configure_runtime(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不静默改成CPU正式训练。")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-6, betas=(.9,.999), eps=1e-8)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def runtime_sources():
    return {p.name: file_hash(p) for p in sorted(HERE.glob("*.py")) if p.name != "model_guide.py"}


def initial_state():
    return dict(epoch=0, cursor=0, updates=0, best_proxy_Q_ref=None, best_train_CE2=None,
        best_Q_ref_at_CE2_best=None, reported=EpochMetrics().state(),
        best_epoch=0, best_update=0, selection_start_epoch=1,
        early_stop_counter=0, history=[], online=MetricMean().state(), grad_sum=0., grad_max=0., grad_clipped=0,
        epoch_steps=0, epoch_train_seconds=0., training_seconds=0., complete=False, stop_reason=None,
        pending_diagnostic=False, diagnostics=[], last_update_row=None, nonfinite_grad_count=0)


def write_definitions(run, identity, dataset):
    # 此文件定义日志口径；不是另一个训练算法配置。
    full = identity["training"]["dataset_mode"] == "full"
    weight_note = ("全量训练和完整验证的窗口权重均为1；online_train按窗口数平均。" if full else
                   "固定best评价 train/proxy/full 的窗口权重依次为 mean_loss_weight/population_weight/1。")
    stage_note = ("保留每条轨迹按时间等数量划分8段的报告口径，不读取子集索引，不是SOC或推断物理阶段。" if full else
                  "stage使用原proxy_manifest的8个时间分层bin，不是SOC阶段，也不人为新增物理阶段。")
    final_note = ("最终best确定后才诊断及评价full，即14条完整验证轨迹；不评价测试集。" if full else
                  "最终best确定后才诊断及评价train/proxy/full。")
    text = f"""# GraphReaction 指标定义（metrics_v1）

- online_train_*：随参数更新的在线窗口指标；sum(mean_loss_weight * value)/实际窗口数。
- train_*：新增的窗口算术平均；每个窗口只统计一次，不再除梯度累积步数。全量训练中与online_train_*相同。
- val_*：保持不进行逐轮验证，因此本轮留空。最终固定best的真实验证点可由postprocess_metrics.py整理到独立表格。
- train/val_*_NRMSE_delta：所有有效节点的标准化增量SSE/count后开方，不用字段或电解液区域权重。
- best_Q_ref_at_CE2_best：刷新训练CE2最佳轮时，一并保存该轮online_train_Q_ref；不是Q_ref历史最小值。
- proxy_*：本轮训练不进行验证，始终留空，validation_status=not_run。
- {weight_note}
- CE2=(2*ce_MSE+其余六字段MSE)/8+0.1*current_relative_MSE；Q_ref=七字段MSE均值+0.1*current_relative_MSE。
- ce/phi_e 的MSE先对phase2/5分别节点平均，再以0.5:0.5合并。分相列单独列出。
- 所有 *_MSE 默认是标准化10秒增量误差，无量纲；cg/cp是原锂化度增量标准化空间。
- current_relative_MSE 是未乘0.1的电流相对平方项；RMSE_pct=100*sqrt(聚合MSE)。
- 物理浓度恢复为mol/m^3后不再乘cmax；电势误差只转换一次到mV。
- physical_RMSE/MAE按权重*SSE、SAE、节点count累加，最后开方/相除；不等同于phase平衡的训练loss。
- count为未加权有效节点数，weighted_count为原scope权重加权的有效节点数；空集合count=0、指标null/CSV空。
- region/interface只统计该字段有效节点；接口为原跨区物理边去重端点。无字段的phase6不填零参与物理误差。
- {stage_note}
- lr是该epoch训练实际值；next_lr是epoch末scheduler.step()之后的值。余弦T_max=141，不传验证loss。
- best依据每轮online_train_CE2严格降低；连续20轮未改善停止，最多总计141轮。
- 在线CE2来自本轮各次参数更新时的loss平均，不是用轮末固定模型重新评价训练集。
- 已完成实验保留原目录；新实验重新初始化，在独立目录从第1轮开始选best。
- grad_norm_pre_clip在4窗累积完成后，直接取clip_grad_norm_返回值；阈值1；随后optimizer.step。
- update_metrics.csv每次成功optimizer.step记录一次；异常单独保存failure.json，不把失败更新算作成功。
- 后续训练期间不进行验证或额外机制前向；{final_note}
- RMS=sqrt(mean(x**2))，count为张量元素数；属于隐藏表示量，无物理单位。a.std使用总体标准差。
- R/Dgraph比值的eps=1e-12，仅表示隐藏幅值；q/a不是物理电流。
- 默认H0..H4、Z0..Z3；DR3/UR3为not_computed和null，Z4不伪造。
- 原始化物理电流按AM/CBD-正集流体接触重建，与LI使用同一结果。
- checkpoint先原子提交，再落对应CSV；恢复只补最后一个未落日志行，检测到超前/重复记录即拒绝。
"""
    path = Path(run) / "metrics/metric_definitions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    write_json(path.with_suffix(".json"), dict(format_version="metrics_v1", identity=identity,
        history_columns=HISTORY_COLUMNS, update_columns=UPDATE_COLUMNS, diagnostic_columns=DIAGNOSTIC_COLUMNS,
        diagnostic_panel=dataset.panel, diagnostic_interval=None, diagnostic_when="final_best_only", ratio_eps=RATIO_EPS))


def run_training(run, dataset, epochs=NUM_EPOCHS, device="cuda", smoke_windows=None, resume=False,
                 pause_after_epoch=None, pause_after_updates=None):
    """pause_*仅供恢复测试；正式命令行没有这些参数。"""
    configure_runtime(device)
    run = Path(run).resolve()
    settings = training_configuration(getattr(dataset, "mode", "proxy32"))
    settings.update(max_epochs=epochs, smoke_windows=smoke_windows, device=device)
    identity = dict(model=model_configuration(), model_configuration_hash=configuration_hash(),
        training=settings, data=dataset.identity, source=runtime_sources(), seed=SEED,
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__), smoke=smoke_windows is not None)
    identity["configuration_hash"] = content_hash(identity)
    if resume:
        saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if saved_config["identity"] != identity:
            raise ValueError("续训配置与原run不一致，不能混用旧版本或数据。")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", dict(identity=identity, created_utc=datetime.now(timezone.utc).isoformat(),
            proxy_root=str(dataset.proxy_root) if dataset.proxy_root is not None else None,
            dataset_root=str(dataset.root), full_root=str(dataset.full_root),
            selection="online_train_CE2 strict decrease; no epoch validation", diagnostic_panel=dataset.panel))
        source_folder = run / "source"
        source_folder.mkdir()
        for name in identity["source"]:
            shutil.copy2(HERE / name, source_folder / name)
        write_definitions(run, identity, dataset)
    model = create_model(SEED, device)
    optimizer = new_optimizer(model)
    scheduler = create_scheduler(optimizer)  # 先创建，再恢复optimizer/scheduler状态。
    reader = WindowReader(dataset, device)
    reader.on_graph = TopologyCoverage(run, identity).add
    state = initial_state()
    last_path, best_path = run / "checkpoints/last.pt", run / "checkpoints/best.pt"
    if resume:
        saved = load_training_checkpoint(last_path, model, optimizer, scheduler, identity)
        state = saved["state"]
        reconcile_csv(run / "update_metrics.csv", state["updates"], state["last_update_row"], UPDATE_COLUMNS, "update")
        reconcile_csv(run / "history.csv", state["epoch"], state["history"][-1] if state["history"] else None, HISTORY_COLUMNS, "epoch")
    def persist():
        save_training_checkpoint(last_path, model, optimizer, scheduler, state, identity, state["last_update_row"])
    def ensure_best():
        if state["best_epoch"] == state["epoch"] and state["epoch"] > 0 and state["cursor"] == 0:
            if not best_path.exists() or torch.load(best_path,map_location="cpu",weights_only=True)["state"]["epoch"] != state["best_epoch"]:
                copy_atomic(last_path, best_path)
    if not resume:
        persist()
    ensure_best()
    print(f"Run: {run}\nDevice={device}; parameters={sum(p.numel() for p in model.parameters())}; smoke={smoke_windows is not None}\n"
          f"Dataset={settings['dataset_mode']}; train_windows={len(dataset.rows['train'])}; final_evaluation={settings['evaluation_scopes']}", flush=True)
    evaluation_status = {name: "not_run" for name in settings["evaluation_scopes"]}
    try:
        write_summary(run, state, identity, evaluation_status, "training")
        while not state["complete"] and state["epoch"] < epochs:
            epoch = state["epoch"] + 1
            order = dataset.epoch_order(epoch, SEED)
            if smoke_windows is not None:
                order = order[:smoke_windows]
            oh = order_hash(order)
            if state["cursor"] and state.get("active_order_hash") != oh:
                raise ValueError("恢复的窗口顺序不一致。")
            state["active_order_hash"] = oh
            lr = optimizer.param_groups[0]["lr"]
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            online = MetricMean(state["online"])
            reported = EpochMetrics(state["reported"])
            for offset in range(state["cursor"], len(order), ACCUMULATION):
                started = time.perf_counter()
                group = order[offset:offset+ACCUMULATION]
                values, norm = train_group(model, reader, group, optimizer)
                for name in BASE_METRICS:
                    online.sums[name] += values.sums[name]
                online.denominator += values.denominator
                online.windows += values.windows
                reported.merge(values.unweighted)
                state["cursor"] = offset + len(group)
                state["updates"] += 1
                state["epoch_steps"] += 1
                state["grad_sum"] += norm
                state["grad_max"] = max(state["grad_max"], norm)
                state["grad_clipped"] += int(norm > CLIP_NORM)
                state["online"] = online.state()
                state["reported"] = reported.state()
                elapsed = time.perf_counter() - started
                state["epoch_train_seconds"] += elapsed
                state["training_seconds"] += elapsed
                row = dict(update=state["updates"], epoch=epoch, cursor=state["cursor"], lr=lr,
                    grad_norm_pre_clip=norm, clipped=norm > CLIP_NORM, nonfinite_grad_count=0, status="ok", order_sha256=oh)
                state["last_update_row"] = row
                persist()
                append_csv(run / "update_metrics.csv", row, UPDATE_COLUMNS)
                if state["updates"] % 100 == 0:
                    print(f"progress epoch={epoch}/{epochs} windows={state['cursor']}/{len(order)} updates={state['updates']}", flush=True)
                if pause_after_updates is not None and state["updates"] == pause_after_updates:
                    return dict(status="test_pause", state=state, run=str(run))
            train_metrics = online.result()
            is_best = state["best_train_CE2"] is None or train_metrics["CE2"] < state["best_train_CE2"]
            if is_best:
                state.update(best_train_CE2=train_metrics["CE2"], best_epoch=epoch, best_update=state["updates"], early_stop_counter=0)
                state["best_Q_ref_at_CE2_best"] = train_metrics["Q_ref"]
            else:
                state["early_stop_counter"] += 1
            scheduler.step()
            reason = "early_stopping_20_epochs_without_strict_improvement" if state["early_stop_counter"] >= EARLY_STOPPING else "epoch_budget" if epoch == epochs else None
            history = dict(epoch=epoch, updates=state["updates"], lr=lr, next_lr=optimizer.param_groups[0]["lr"],
                order_sha256=oh, scheduler_name=type(scheduler).__name__, scheduler_config_hash=content_hash(settings),
                best_proxy_Q_ref=None, best_train_CE2=state["best_train_CE2"],
                selection_metric="online_train_CE2", best_value=state["best_train_CE2"], validation_status="not_run",
                best_epoch=state["best_epoch"], epochs_since_best=epoch-state["best_epoch"],
                early_stop_counter=state["early_stop_counter"], is_best=is_best, stop_reason=reason,
                grad_norm_pre_clip_mean=state["grad_sum"]/state["epoch_steps"], grad_norm_pre_clip_max=state["grad_max"],
                grad_clip_fraction=state["grad_clipped"]/state["epoch_steps"], nonfinite_grad_count=state["nonfinite_grad_count"],
                epoch_train_seconds=state["epoch_train_seconds"], epoch_proxy_seconds=None,
                cuda_max_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
            history.update({"online_train_"+name: value for name,value in train_metrics.items()})
            history.update(epoch_columns("train", reported))
            history.update({key: None for key in epoch_columns("val", EpochMetrics())})
            history.update(best_Q_ref_at_CE2_best=state["best_Q_ref_at_CE2_best"], best_Q_ref_scope="online_train")
            state["history"].append(history)
            state.update(epoch=epoch, cursor=0, online=MetricMean().state(), grad_sum=0., grad_max=0., grad_clipped=0,
                         reported=EpochMetrics().state(),
                         epoch_steps=0, epoch_train_seconds=0., complete=reason is not None, stop_reason=reason,
                         pending_diagnostic=False)
            persist()
            ensure_best()
            append_csv(run / "history.csv", history, HISTORY_COLUMNS)
            write_json(run / "history.json", state["history"])
            print(f"epoch={epoch}/{epochs} updates={state['updates']} lr={lr:.4g} next_lr={history['next_lr']:.4g} "
                  f"online_train_CE2={history['online_train_CE2']:.7g} best_train_CE2={state['best_train_CE2']:.7g} "
                  f"best_epoch={state['best_epoch']} grad_norm={history['grad_norm_pre_clip_mean']:.5g} "
                  f"elapsed={history['epoch_train_seconds']:.1f}s best={is_best}; validation=not_run", flush=True)
            write_summary(run, state, identity, evaluation_status, "training" if not state["complete"] else "evaluating")
            if epoch == 1 or epoch % 10 == 0 or state["complete"]:
                plot_history(state["history"], run / "loss_curve.png")
                plot_metric_curves(state["history"], run / "metrics_v2", dict(
                    best_epoch=state["best_epoch"], selection_metric="online_train_CE2"))
            if pause_after_epoch == epoch:
                return dict(status="test_pause", state=state, run=str(run))
        write_summary(run, state, identity, evaluation_status, "evaluating")
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if best["identity"] != identity:
            raise ValueError("best checkpoint身份不一致。")
        model.load_state_dict(best["model"], strict=True)
        checkpoint = dict(path=str(best_path), sha256=file_hash(best_path), epoch=state["best_epoch"], update=state["best_update"])
        run_diagnostics(model, reader, run, state["best_epoch"], state["best_update"], model_hash(model), tag="best")
        for scope in settings["evaluation_scopes"]:
            folder = run / "evaluation" / scope
            result_path = folder / "overall_metrics.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result["identity"] != identity or result["checkpoint"] != checkpoint:
                    raise ValueError("已有评价不属于当前best。")
            else:
                pairs = dataset.pairs(scope)
                if smoke_windows is not None:
                    pairs = pairs[:smoke_windows]
                result = evaluate_scope(model, reader, scope, folder, identity, checkpoint, pairs)
            evaluation_status[scope] = "completed"
            write_summary(run, state, identity, evaluation_status, "evaluating")
        summary = write_summary(run, state, identity, evaluation_status, "completed")
        # 后处理已有日志和最终评价；不额外运行模型，也不回写历史文件。
        from postprocess_metrics import process_run
        process_run(run)
        print("TRAINING_AND_EVALUATION_COMPLETE", run, flush=True)
        return summary
    except BaseException as error:
        # 不把未完成的累积/更新写成已提交断点；last.pt仍是最近完整更新。
        if "nonfinite_gradient_pre_clip" in str(error):
            state["nonfinite_grad_count"] += 1
        write_json(run / "failure.json", dict(type=type(error).__name__, message=str(error),
                   epoch=state["epoch"], cursor=state["cursor"], nonfinite_grad_count=state["nonfinite_grad_count"],
                   committed_checkpoint=str(last_path)))
        write_summary(run, state, identity, evaluation_status, "interrupted" if isinstance(error,KeyboardInterrupt) else "failed")
        raise


def write_summary(run, state, identity, evaluation_status, status):
    history = state["history"]
    summary = dict(format_version="training_summary_v1", run=str(run), identity=identity, status=status,
        seed=SEED, optimizer="AdamW", scheduler="CosineAnnealingLR", initial_lr=INITIAL_LR,
        final_lr=history[-1]["next_lr"] if history else INITIAL_LR, epochs=state["epoch"], updates=state["updates"],
        best_epoch=state["best_epoch"], best_update=state["best_update"], best_train_CE2=state["best_train_CE2"],
        best_Q_ref_at_CE2_best=state["best_Q_ref_at_CE2_best"], best_Q_ref_scope="online_train",
        selection_metric="online_train_CE2", selection_start_epoch=state["selection_start_epoch"],
        validation_during_training=False,
        stop_reason=state["stop_reason"] if status not in ("failed","interrupted") else status,
        training_seconds=state["training_seconds"], checkpoints=dict(best=str(run/"checkpoints/best.pt"),last=str(run/"checkpoints/last.pt")),
        evaluations=evaluation_status.copy(), diagnostics_epochs=state["diagnostics"], best_diagnostics=(run/"diagnostics/capture_best.csv").exists(),
        smoke=identity["smoke"], test_access=0)
    summary["nonfinite_grad_count"] = state["nonfinite_grad_count"]
    write_json(run / "training_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("proxy32", "full"), default="proxy32")
    parser.add_argument("--proxy-root", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--full-root", type=Path, default=HERE.parent / "Datasets")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--smoke-windows", type=int)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.epochs <= NUM_EPOCHS or args.smoke_windows is not None and args.smoke_windows < 1:
        parser.error("epochs应在1～141内，smoke-windows必须为正数。")
    if args.resume:
        saved = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))
        config = saved["identity"]["training"]
        if config.get("dataset_mode", "proxy32") == "full":
            dataset = FullTrainingWindows(saved["dataset_root"])
        else:
            dataset = TrainingWindows(saved["proxy_root"], saved["full_root"])
        run_training(args.resume, dataset, config["max_epochs"], config["device"], config["smoke_windows"], resume=True)
    else:
        dataset = FullTrainingWindows(args.full_root) if args.dataset == "full" else TrainingWindows(args.proxy_root, args.full_root)
        run = args.run_dir
        if run is None:
            prefix = ("smoke_" if args.smoke_windows is not None else "") + args.dataset + "_train_ce2"
            number = 1
            while (HERE / "training_runs" / f"{prefix}_{number:03d}").exists():
                number += 1
            run = HERE / "training_runs" / f"{prefix}_{number:03d}"
        run_training(run, dataset, args.epochs, args.device, args.smoke_windows)


if __name__ == "__main__":
    main()
