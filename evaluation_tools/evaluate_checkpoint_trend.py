"""只用已保存权重补充138/141/154/155轮的固定训练、验证评价。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from evaluation import evaluate_scope
from full_training_windows import FullTrainingWindows
from initialization import create_model
from run_files import file_hash, write_csv, write_json
from train import configure_runtime, model_hash, runtime_sources
from training_windows import WindowReader
from export_checkpoint_comparison import (
    EXPERIMENTS, RUNS, checked_checkpoint, export_group, protected_hashes, read_json,
)

PREVIOUS = RUNS / "offline_138_154_train_validation_20261030"
POINTS = {
    138: (EXPERIMENTS[138][0], "best.pt"),
    141: (EXPERIMENTS[138][0], "last.pt"),
    154: (EXPERIMENTS[154][0], "best.pt"),
    155: (EXPERIMENTS[154][0], "last.pt"),
}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def load_points(dataset):
    """复用已核对的模型/源码检查，再明确检查last的轮次，绝不把last当best。"""
    parents = {}
    for best_epoch in (138, 154):
        run = EXPERIMENTS[best_epoch][0]
        parents[run] = checked_checkpoint(run, best_epoch, dataset)
    result = {}
    for epoch, (run, name) in POINTS.items():
        parent, _ = parents[run]
        path = run / "checkpoints" / name
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != parent["schema"] or saved["identity"] != parent["identity"]:
            raise ValueError(f"{epoch}轮checkpoint身份不匹配")
        if saved["state"]["epoch"] != epoch or saved["state"]["cursor"] != 0:
            raise ValueError(f"{path}不是指定轮次的完整epoch边界")
        checkpoint = dict(path=str(path), sha256=file_hash(path), epoch=epoch,
                          update=saved["state"]["updates"])
        result[epoch] = saved, checkpoint
    return result


def original_hashes():
    hashes = protected_hashes()
    for path in PREVIOUS.rglob("*"):
        if path.is_file():
            hashes[str(path)] = file_hash(path)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = FullTrainingWindows()
    for scope, windows, graphs in (("train", 22927, 65), ("full", 4917, 14)):
        pairs = dataset.pairs(scope)
        if len(pairs) != windows or len({sid for sid, _ in pairs}) != graphs:
            raise ValueError("全量训练/验证划分与预期不一致")
    payloads = load_points(dataset)
    before = original_hashes()
    sources = runtime_sources()
    config = dict(format_version="stored_checkpoint_group_trend_v1",
        dataset=dataset.identity, source=sources, evaluation_script_sha256=file_hash(__file__),
        checkpoints={str(epoch): checkpoint for epoch, (_, checkpoint) in payloads.items()},
        grouping=dict(concentration="(2*ce_MSE+cg_MSE+cp_MSE)/4",
                      potential="(phi_e_MSE+phi_p_MSE+phi_CBD_MSE)/3",
                      voltage="V_MSE", current="current_relative_MSE"),
        checkpoint_selection="unchanged: online_train_CE2", new_training=False)
    config_path = output / "evaluation_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("输出目录配置不一致，拒绝覆盖")
    write_json(config_path, config)

    inventory = []
    for path in sorted(RUNS.rglob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        comparable = saved.get("identity", {}).get("data") == dataset.identity
        inventory.append(dict(path=str(path), epoch=saved["state"]["epoch"],
            sha256=file_hash(path), comparable_full_experiment=comparable,
            reason="same full dataset lineage" if comparable else "different subset experiment; excluded"))
    write_csv(output / "checkpoint_inventory.csv", inventory, list(inventory[0]))
    print("Available full checkpoints: 138(best),141(last),154(best),155(last). No earlier/middle checkpoint found.", flush=True)

    summaries, checks = {}, {}
    # 先复用上次完成的四组；本次只额外计算141、155两个轮次。
    for epoch in (138, 154):
        saved, checkpoint = payloads[epoch]
        for scope in ("train", "full"):
            folder = PREVIOUS / f"evaluation_epoch{epoch}_train" if scope == "train" else EXPERIMENTS[epoch][1]
            key = f"{epoch}_{scope}"
            summaries[key], checks[key] = export_group(folder, scope, epoch, dataset,
                                                      checkpoint, saved["identity"], output)
            print(f"REUSED epoch={epoch}, scope={scope}, windows={summaries[key]['windows']}", flush=True)
    if args.prepare_only:
        print("Prepared: four checkpoints verified; existing four evaluations reused.", flush=True)
        return

    configure_runtime(args.device)
    for epoch in (141, 155):
        saved, checkpoint = payloads[epoch]
        model = create_model(device=args.device)
        model.load_state_dict(saved["model"], strict=True)
        initial_hash = model_hash(model)
        for scope in ("train", "full"):
            split = "train" if scope == "train" else "validation"
            folder = output / f"evaluation_epoch{epoch}_{split}"
            if not (folder / "overall_metrics.json").exists():
                pairs = dataset.pairs(scope)
                write_json(output / "status.json", dict(status="running", epoch=epoch,
                    split=split, windows=len(pairs), stage_started_at=timestamp(), optimizer_updates=0))
                reader = WindowReader(dataset, args.device)
                print(f"START epoch={epoch}, split={split}, windows={len(pairs)}", flush=True)
                result = evaluate_scope(model, reader, scope, folder, saved["identity"], checkpoint)
                if model.training or model_hash(model) != initial_hash:
                    raise RuntimeError("评价模式或固定权重发生变化")
                if reader.loaded_samples != {sid for sid, _ in pairs}:
                    raise RuntimeError("实际读取轨迹集合不符合指定范围")
                write_json(folder / "evaluation_audit.json", dict(status="PASS", checkpoint=checkpoint,
                    model_weights_unchanged=True, eval_mode=True, optimizer_updates=0,
                    normalization_refitted=False, test_dynamic_reads=0,
                    loaded_samples=sorted(reader.loaded_samples), scope=scope, windows=len(pairs)))
                print(f"DONE epoch={epoch}, split={split}, seconds={result['seconds']:.2f}, CE2={result['losses']['CE2']}", flush=True)
                del reader
            audit = read_json(folder / "evaluation_audit.json")
            if audit["status"] != "PASS" or audit["checkpoint"] != checkpoint:
                raise ValueError("已有补评结果的审计记录不匹配")
            key = f"{epoch}_{scope}"
            summaries[key], checks[key] = export_group(folder, scope, epoch, dataset,
                                                      checkpoint, saved["identity"], output)
            write_csv(output / "summary.csv", list(summaries.values()), list(summaries[key]))
        del model
    ordered = [summaries[f"{epoch}_{scope}"] for epoch in POINTS for scope in ("train", "full")]
    write_csv(output / "summary.csv", ordered, list(ordered[0]))
    if before != original_hashes() or sources != runtime_sources():
        raise RuntimeError("原实验、既有结果或训练源码发生变化")
    write_json(output / "evaluation_checks.json", dict(status="PASS", groups=checks,
        max_absolute_error=max(c["max_absolute_error"] for c in checks.values()), tolerance=1e-12,
        original_hashes=before, original_files_unchanged=True, training_sources_unchanged=True,
        optimizer_updates=0, test_dynamic_reads=0, training_still_paused_at=155))
    write_json(output / "status.json", dict(status="evaluation_completed", completed_at=timestamp(),
                                           groups=8, optimizer_updates=0))
    print(f"All eight fixed-checkpoint evaluations saved: {output}", flush=True)


if __name__ == "__main__":
    main()
