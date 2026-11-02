"""固定指定best，单独评价14条完整验证轨迹；不恢复训练、不重选best。"""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import sys

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from evaluation import evaluate_scope
from full_training_windows import FullTrainingWindows
from initialization import create_model
from run_files import file_hash, write_json
from schema import model_configuration
from train import configure_runtime, model_hash, runtime_sources
from training_checkpoint import TRAINING_CHECKPOINT_SCHEMA
from training_windows import WindowReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run = args.model_run.resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    path = run / "checkpoints/best.pt"
    original = {name: file_hash(run / name) for name in (
        "checkpoints/best.pt", "checkpoints/last.pt", "config.json", "history.csv", "training_summary.json")}
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != TRAINING_CHECKPOINT_SCHEMA or saved["identity"] != config["identity"]:
        raise ValueError("best权重与实验身份不一致。")
    if saved["state"]["epoch"] != args.expected_epoch:
        raise ValueError("best不是指定轮次，拒绝换用其它权重。")
    if saved["identity"]["model"] != model_configuration() or saved["identity"]["source"] != runtime_sources():
        raise ValueError("当前模型或计算源码不同于保存的实验。")
    dataset = FullTrainingWindows(config["dataset_root"])
    if dataset.identity != saved["identity"]["data"]:
        raise ValueError("验证数据或训练归一化统计与原实验不同。")
    pairs = dataset.pairs("full")
    if len(pairs) != 4917 or len({sid for sid, t in pairs}) != 14:
        raise ValueError("本次评价必须包含14条完整验证轨迹、4917个窗口。")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = (args.output or CODE / "training_runs" / f"validation_best{args.expected_epoch}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = dict(path=str(path), sha256=original["checkpoints/best.pt"],
                      epoch=args.expected_epoch, update=saved["state"]["updates"])
    write_json(output / "evaluation_config.json", dict(identity=saved["identity"], checkpoint=checkpoint,
        device=args.device, evaluation_script_sha256=file_hash(__file__),
        prediction_mode="true state at t -> one step prediction at t+1; no rollout",
        selection="checkpoint specified before evaluation; no best selection"))
    configure_runtime(args.device)
    model = create_model(device=args.device)
    model.load_state_dict(saved["model"], strict=True)
    before = model_hash(model)
    reader = WindowReader(dataset, args.device)
    print(f"Output: {output}\nCheckpoint epoch={args.expected_epoch}; sha256={checkpoint['sha256']}\nValidation trajectories=14; windows=4917; device={args.device}", flush=True)
    result = evaluate_scope(model, reader, "full", output, saved["identity"], checkpoint, pairs)
    if model_hash(model) != before:
        raise RuntimeError("评价过程中模型权重发生变化。")
    if reader.loaded_samples != {sid for sid, t in pairs}:
        raise RuntimeError("实际读取的验证轨迹不完整。")
    if original != {name: file_hash(run / name) for name in original}:
        raise RuntimeError("原实验的记录或断点发生变化。")
    write_json(output / "evaluation_audit.json", dict(status="PASS", checkpoint=checkpoint,
        original_files_unchanged=True, original_file_hashes=original, model_weights_unchanged=True,
        loaded_validation_samples=sorted(reader.loaded_samples), validation_trajectories=14,
        validation_windows=4917, optimizer_updates=0, normalization_refitted=False,
        test_dynamic_reads=0, training_dynamic_reads=0))
    print(json.dumps(dict(status="completed", output=str(output), seconds=result["seconds"], losses=result["losses"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
