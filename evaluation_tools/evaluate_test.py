"""固定指定best，对14条完整测试轨迹评价；不训练、不选择新best。"""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import sys

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from continue_training import executable_tree
from initialization import create_model
from run_files import file_hash, write_json
from schema import model_configuration
from train import configure_runtime, model_hash, runtime_sources
from training_checkpoint import TRAINING_CHECKPOINT_SCHEMA
from training_windows import WindowReader
from test_windows import TestWindows
from test_metrics import evaluate_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, default=138)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
    "--checkpoint-name",
    choices=["best.pt", "best_train.pt", "best_val.pt"],
    default="best.pt",
    )
    args = parser.parse_args()
    model_run = args.model_run.resolve()
    config = json.loads((model_run / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = model_run / "checkpoints" / args.checkpoint_name
    digest = file_hash(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["schema"] != TRAINING_CHECKPOINT_SCHEMA or payload["identity"] != config["identity"]:
        raise ValueError("best身份与原实验不一致。")
    if payload["state"]["epoch"] != args.expected_epoch:
        raise ValueError("best不是明确指定的轮次，拒绝换用其它权重。")
    if payload["identity"]["model"] != model_configuration():
        raise ValueError("当前模型结构与checkpoint不同。")
    # 指标记录/训练入口允许有已审查的改动；计算模块必须与原实验一致。
    for name, source_hash in config["identity"]["source"].items():
        saved = model_run / "source" / name
        if file_hash(saved) != source_hash:
            raise ValueError(f"原实验源码快照已变化：{name}")
        if name not in {"train.py", "training_step.py", "training_config.py", "evaluation.py"}:
            if executable_tree(saved) != executable_tree(CODE / name):
                raise ValueError(f"当前计算模块与原实验不同：{name}")
    dataset = TestWindows(config["dataset_root"])
    if dataset.identity["source"] != config["identity"]["data"]["source"]:
        raise ValueError("数据/训练归一化统计与原实验不同。")
    if len(dataset.samples) != 14 or len(dataset.pairs("test")) != 4941:
        raise ValueError("当前任务必须覆盖14条测试轨迹、4941个窗口。")
    checkpoint = dict(path=str(checkpoint_path), sha256=digest, epoch=args.expected_epoch,
                      update=payload["state"]["updates"])
    identity = dict(model=model_configuration(), dataset=dataset.identity,
                    checkpoint_training_identity=payload["identity"], source=runtime_sources(),
                    test_source={p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = (args.output or CODE / "training_runs" / f"test_best{args.expected_epoch}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "evaluation_config.json", dict(identity=identity, checkpoint=checkpoint,
        dataset_root=str(dataset.root), device=args.device, windows=4941, trajectories=14,
        selection="fixed checkpoint specified before test; no selection using test results"))
    configure_runtime(args.device)
    model = create_model(device=args.device)
    model.load_state_dict(payload["model"], strict=True)
    initial_weights = model_hash(model)
    reader = WindowReader(dataset, args.device)
    print(f"Output: {output}\nCheckpoint epoch={args.expected_epoch}; sha256={digest}\nTest trajectories=14; windows=4941; device={args.device}", flush=True)
    result = evaluate_test(model, reader, output, identity, checkpoint)
    if model_hash(model) != initial_weights or file_hash(checkpoint_path) != digest:
        raise RuntimeError("固定模型或原checkpoint在评价中发生变化。")
    if reader.loaded_samples != set(dataset.samples):
        raise RuntimeError("实际读取的测试轨迹没有完整匹配划分。")
    write_json(output / "evaluation_audit.json", dict(status="PASS", checkpoint_epoch=args.expected_epoch,
        checkpoint_sha256=digest, original_checkpoint_unchanged=True, model_weights_unchanged=True,
        loaded_test_samples=sorted(reader.loaded_samples), test_trajectories=14, test_windows=4941,
        optimizer_updates=0, normalization_refitted=False, training_or_validation_dynamic_reads=0))
    print(json.dumps(dict(status="completed", output=str(output), seconds=result["seconds"], losses=result["losses"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
