"""只训练选best、早停、保存占用修复、精确断点恢复的针对性检查。"""

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_training_metrics import ArtificialDataset, ArtificialReader, resume_tests
from v2_helpers import HERE, reference_definitions
import train
from run_files import replace_with_retry, write_json, read_csv, file_hash
from training_metrics import MetricMean, BASE_METRICS


def locked_file_tests(folder, report):
    """实际用Windows文件共享标志占用目标，然后释放；另模拟持续失败。"""
    old, temporary = folder / "locked.pt", folder / "locked.tmp"
    old.write_bytes(b"old checkpoint")
    temporary.write_bytes(b"new checkpoint")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # 允许读写，但暂不允许删除/替换，模拟另一个进程正在查看checkpoint。
    handle = kernel.CreateFileW(str(old), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    errors = []
    original = Path.replace
    def observed_replace(source, target):
        try:
            return original(source, target)
        except PermissionError as error:
            errors.append(error.winerror)
            raise
    def release():
        time.sleep(.35)
        kernel.CloseHandle(handle)
    worker = threading.Thread(target=release)
    worker.start()
    try:
        with patch.object(Path, "replace", observed_replace):
            replace_with_retry(temporary, old)
    finally:
        worker.join()
    assert errors and old.read_bytes() == b"new checkpoint"
    report["real_windows_lock"] = dict(status="PASS", retry_errors=errors)

    old.write_bytes(b"keep valid checkpoint")
    temporary.write_bytes(b"new candidate")
    error = PermissionError("simulated persistent Windows access denial")
    error.winerror = 5
    with patch.object(Path, "replace", side_effect=error) as replacing, patch("run_files.time.sleep"):
        try:
            replace_with_retry(temporary, old)
        except PermissionError:
            pass
        else:
            raise AssertionError("持续失败必须明确报错")
    assert replacing.call_count == 10
    assert old.read_bytes() == b"keep valid checkpoint" and temporary.read_bytes() == b"new candidate"
    report["persistent_lock"] = "PASS: bounded 10 attempts, old checkpoint and candidate retained"


def train_only_tests(ref, folder, report):
    dataset = ArtificialDataset(ref)
    final_started = False
    scopes = []
    group_calls = []
    real_diagnostics, real_group = train.run_diagnostics, train.train_group
    run = folder / "train_only"

    class GuardReader(ArtificialReader):
        def window(self, scope, sid, t):
            # 包括机制诊断在内，训练尚未停止时不得读取任何验证窗口。
            assert scope == "train" or final_started, "训练期间读取了验证集"
            scopes.append((final_started, scope))
            return super().window(scope, sid, t)

    def guarded_group(*args, **kwargs):
        assert not final_started
        group_calls.append(1)
        return real_group(*args, **kwargs)

    def final_diagnostics(model, reader, path, epoch, updates, checkpoint, tag=None):
        nonlocal final_started
        assert tag == "best" and len(group_calls) == 6
        saved = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(value.cpu(), saved["model"][key]) for key,value in model.state_dict().items())
        final_started = True
        return real_diagnostics(model, reader, path, epoch, updates, checkpoint, tag)

    with patch("train.WindowReader", GuardReader), patch("train.train_group", guarded_group), \
         patch("train.run_diagnostics", final_diagnostics):
        summary = train.run_training(run, dataset, epochs=3, device="cpu", smoke_windows=8)
    history = read_csv(run / "history.csv")
    chosen = min(history, key=lambda row: float(row["online_train_CE2"]))
    assert summary["best_epoch"] == int(chosen["epoch"])
    assert summary["best_train_CE2"] == float(chosen["online_train_CE2"])
    assert all(row["validation_status"] == "not_run" and row["proxy_CE2"] == "" for row in history)
    assert sum(not done for done, scope in scopes) == 24
    assert summary["diagnostics_epochs"] == [] and summary["best_diagnostics"]
    assert summary["evaluations"] == dict(train="completed",proxy="completed",full="completed")
    report["train_only"] = dict(status="PASS", training_window_calls=24, validation_calls_during_training=0,
        best_epoch=summary["best_epoch"], best_train_CE2=summary["best_train_CE2"], final_scopes=list(summary["evaluations"]))


def early_stop_test(ref, folder, report):
    """人为给定CE2，验证相等不算改善，第20次未改善触发；不依赖模型能否收敛。"""
    dataset = ArtificialDataset(ref)
    calls = 0
    def controlled_group(model, reader, group, optimizer):
        nonlocal calls
        epoch = calls // 2 + 1
        calls += 1
        # 标量参数做一次真实optimizer更新，保留scheduler的正确调用顺序。
        optimizer.zero_grad(set_to_none=True)
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        result = MetricMean()
        from metric_reporting import EpochMetrics
        result.unweighted = EpochMetrics()
        for _ in group:
            values = {key: 2. if epoch == 1 else 1. for key in BASE_METRICS}
            result.add(values, 1, denominator=1)
            result.unweighted.mean.add(values)
        return result, 1.
    run = folder / "early_stop"
    with patch("train.WindowReader", ArtificialReader), patch("train.train_group", controlled_group), \
         patch("train.plot_history"):
        summary = train.run_training(run, dataset, epochs=30, device="cpu", smoke_windows=8)
    rows = read_csv(run / "history.csv")
    assert summary["epochs"] == 22 and summary["best_epoch"] == 2
    assert rows[-2]["early_stop_counter"] == "19" and rows[-1]["early_stop_counter"] == "20"
    assert sum(row["is_best"] == "True" for row in rows) == 2
    assert summary["stop_reason"] == "early_stopping_20_epochs_without_strict_improvement"
    report["early_stop"] = "PASS: best epoch2, ties do not improve, exactly20 misses stop at epoch22"


def old_run_rejection(report):
    parent = HERE / "training_runs/proxy32_cosine_001"
    if not parent.exists():
        report["old_run_rejected"] = "not_run: prior run unavailable"
        return
    from training_windows import TrainingWindows
    before = file_hash(parent / "checkpoints/last.pt")
    try:
        train.run_training(parent, TrainingWindows(), resume=True)
    except ValueError as error:
        assert "续训配置与原run不一致" in str(error)
    else:
        raise AssertionError("不得将旧Proxy选模运行静默当成训练选模运行")
    assert file_hash(parent / "checkpoints/last.pt") == before
    report["old_run_rejected"] = "PASS: incompatible protocol rejected; old checkpoint unchanged"


def main():
    train.configure_runtime("cpu")
    number = 1
    while (HERE / "training_runs" / f"train_only_checks_{number:03d}").exists(): number += 1
    folder = HERE / "training_runs" / f"train_only_checks_{number:03d}"
    folder.mkdir(parents=True)
    report = dict(status="running")
    ref = reference_definitions()
    locked_file_tests(folder, report)
    train_only_tests(ref, folder, report)
    early_stop_test(ref, folder, report)
    resume_tests(ref, report, folder)
    old_run_rejection(report)
    report.update(status="PASS", runtime_source=train.runtime_sources())
    write_json(folder / "test_results.json", report)
    print("TRAIN_ONLY_CHECKS_PASS", folder, flush=True)


if __name__ == "__main__":
    main()
