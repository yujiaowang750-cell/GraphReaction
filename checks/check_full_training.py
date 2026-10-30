"""检查全量划分、训练/验证隔离及短流程；测试结果不作为正式实验结果。"""

import ast
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_training_windows import FullTrainingWindows
from run_files import file_hash
from training_windows import WindowReader
from training_config import training_configuration
from check_training_metrics import ArtificialDataset, ArtificialReader, assert_nested_equal
from check_train_only import train_only_tests, early_stop_test
from v2_helpers import reference_definitions
import train


def must_reject(call, message):
    try:
        call()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("应当拒绝这个输入")


def check_indices(dataset, folder, report):
    assert dataset.mode == "full" and dataset.proxy_root is None
    assert dataset.root == dataset.full_root == (HERE.parent / "Datasets").resolve()
    assert set(dataset.rows) == {"train", "full"}
    for scope, graphs, windows in (("train", 65, 22927), ("full", 14, 4917)):
        pairs = dataset.pairs(scope)
        assert len(pairs) == windows and len({sid for sid, _ in pairs}) == graphs
        assert all(dataset.weight(scope, sid, t) == 1. for sid, t in pairs)
    order = dataset.epoch_order(2)
    assert order == dataset.epoch_order(2)
    assert len(order) == len(set(order)) == 22927 and set(order) == set(dataset.pairs("train"))
    reader = WindowReader(dataset, "cpu")
    for scope in ("test", "proxy"):
        must_reject(lambda: reader.window(scope, 0, 0), "未启用数据范围")
    must_reject(lambda: reader.window("train", dataset.rows["full"][0]["sid"], 0), "禁止读取")
    assert reader.loaded_samples == set()

    # 只复制清单/索引作错误输入测试，不复制或修改原始轨迹。
    bad_root = folder / "bad_metadata"
    bad_root.mkdir()
    for name in ("trajectory_split.json", "dataset_manifest.json", "normalization_stats.json"):
        shutil.copy2(dataset.root / name, bad_root / name)
    with np.load(dataset.root / "window_index.npz", allow_pickle=False) as index:
        arrays = {split + suffix: index[split + suffix].copy()
                  for split in ("train", "validation") for suffix in ("_sample_index", "_input_time_index")}
    for kind in ("duplicate", "missing", "validation_leak"):
        bad = {name: value.copy() for name, value in arrays.items()}
        if kind == "duplicate":
            for suffix in ("_sample_index", "_input_time_index"):
                bad["train" + suffix][1] = bad["train" + suffix][0]
            message = "重复窗口"
        elif kind == "missing":
            for suffix in ("_sample_index", "_input_time_index"):
                bad["train" + suffix] = bad["train" + suffix][:-1]
            message = "完整匹配"
        else:
            for suffix in ("_sample_index", "_input_time_index"):
                bad["train" + suffix][0] = bad["validation" + suffix][0]
            message = "完整匹配"
        np.savez(bad_root / "window_index.npz", **bad)
        must_reject(lambda: FullTrainingWindows(bad_root), message)
    report["full_indices"] = dict(status="PASS", train_graphs=65, train_windows=22927,
                                   validation_graphs=14, validation_windows=4917, weights=1,
                                   rejected=["test scope", "proxy scope", "validation in train", "duplicate window", "missing window"])


def check_full_flow(dataset, reader_class, folder, device, report, label, epochs, windows):
    final_started = False
    reads, evaluations = [], []
    original_diagnostics, original_evaluate = train.run_diagnostics, train.evaluate_scope
    run = folder / label

    class GuardReader(reader_class):
        def window(self, scope, sid, t):
            assert scope in ("train", "full")
            assert final_started or scope == "train", "训练阶段不能读取验证状态"
            reads.append((final_started, scope, sid, t))
            return super().window(scope, sid, t)

    def final_diagnostics(model, reader, path, epoch, updates, checkpoint, tag=None):
        nonlocal final_started
        assert tag == "best"
        saved = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
        for name, value in model.state_dict().items():
            assert torch.equal(value.cpu(), saved["model"][name])
        final_started = True
        return original_diagnostics(model, reader, path, epoch, updates, checkpoint, tag)

    def evaluate(model, reader, scope, *args):
        assert final_started and scope == "full"
        evaluations.append(scope)
        return original_evaluate(model, reader, scope, *args)

    with patch("train.WindowReader", GuardReader), patch("train.run_diagnostics", final_diagnostics), patch("train.evaluate_scope", evaluate):
        summary = train.run_training(run, dataset, epochs=epochs, device=device, smoke_windows=windows)
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    selected = min(history, key=lambda row: float(row["online_train_CE2"]))
    assert summary["best_epoch"] == int(selected["epoch"])
    assert summary["evaluations"] == {"full": "completed"} and evaluations == ["full"]
    assert all(row["validation_status"] == "not_run" and not row["proxy_CE2"] for row in history)
    assert sum(not final for final, *_ in reads) == epochs * windows
    assert not (run / "evaluation/train").exists() and not (run / "evaluation/proxy").exists()
    result = json.loads((run / "evaluation/full/overall_metrics.json").read_text(encoding="utf-8"))
    assert result["weight_rule"] == "1" and result["windows"] == min(windows, len(dataset.rows["full"]))
    report[label] = dict(status="PASS", device=device, training_windows=epochs*windows,
                        validation_calls_during_training=0, final_scopes=evaluations,
                        final_validation_windows=result["windows"], best_epoch=summary["best_epoch"],
                        test_state_reads=0, updates=summary["updates"])


def main():
    dataset = FullTrainingWindows()
    data_hashes = dataset.identity["source"]
    # 与已经通过维度检查的实现比对，模型、损失、优化器更新函数均应保持原样。
    baseline = json.loads((HERE / "checks/module_dimensions_result.json").read_text(encoding="utf-8"))["production_source_sha256"]
    allowed = {"train.py", "training_windows.py", "training_config.py", "evaluation.py"}
    comment_changes = []
    for name, digest in baseline.items():
        if name in allowed or file_hash(HERE / name) == digest:
            continue
        # 用户阅读时添加过注释：使用原实验源码核对AST，仍拒绝任何计算逻辑差异。
        original = HERE / "training_runs/x13_b30_train_ce2_20261026_092100/source" / name
        assert file_hash(original) == digest, name
        assert ast.dump(ast.parse(original.read_text(encoding="utf-8-sig"))) == ast.dump(
            ast.parse((HERE / name).read_text(encoding="utf-8-sig"))), name
        comment_changes.append(name)
    settings = training_configuration("full")
    assert settings["evaluation_scopes"] == ["full"] and settings["train_weight"] == 1
    assert settings["monitor"] == "online_train_CE2" and settings["early_stopping"] == 20
    assert settings["T_max"] == settings["max_epochs"] == 141 and not settings["validate_each_epoch"]
    report = dict(status="running", settings=settings, data_identity=dataset.identity,
                  preexisting_comment_changes=comment_changes)
    train.configure_runtime("cpu")
    ref = reference_definitions()
    with tempfile.TemporaryDirectory(prefix="graphreaction_full_checks_") as temp:
        folder = Path(temp)
        check_indices(dataset, folder, report)
        artificial = ArtificialDataset(ref)
        artificial.mode = "full"
        artificial.proxy_root = None
        artificial.identity["dataset_mode"] = "full"
        artificial.rows.pop("proxy")
        artificial.by_pair.pop("proxy")
        artificial.panel = [("train", 1, 0), ("full", 0, 0)]
        artificial.weight = lambda scope, sid, t: 1.
        check_full_flow(artificial, ArtificialReader, folder, "cpu", report, "artificial_full_flow", 3, 8)
        # 原32档流程和严格20轮早停仍应有效。
        train_only_tests(ref, folder, report)
        early_stop_test(ref, folder, report)
        # 全量模式断点恢复与连续执行应产生相同模型和优化器。
        with patch("train.WindowReader", ArtificialReader):
            continuous = folder / "full_continuous"
            resumed = folder / "full_resumed"
            train.run_training(continuous, artificial, epochs=2, device="cpu", smoke_windows=8)
            train.run_training(resumed, artificial, epochs=2, device="cpu", smoke_windows=8, pause_after_updates=1)
            train.run_training(resumed, artificial, epochs=2, device="cpu", smoke_windows=8, resume=True)
        a = torch.load(continuous / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        b = torch.load(resumed / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        for name in ("model", "optimizer", "scheduler", "rng"):
            assert_nested_equal(a[name], b[name])
        report["full_resume"] = "PASS: identical model, optimizer, scheduler and RNG"
        if not torch.cuda.is_available():
            raise RuntimeError("正式全量训练需要CUDA；未执行GPU短流程。")
        check_full_flow(dataset, WindowReader, folder, "cuda", report, "real_full_gpu_smoke", 1, 7)
    assert all(file_hash(dataset.root / name) == digest for name, digest in data_hashes.items())
    report.update(status="PASS", model_and_loss_unchanged=True, source_data_unchanged=True)
    output = HERE / "checks/full_training_result.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FULL_TRAINING_CHECKS_PASS", output)


if __name__ == "__main__":
    main()
