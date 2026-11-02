"""当前实验的有限回归：模型/损失、累积尾组、实际维度；报告写入新目录。"""

import argparse
import ast
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from v2_helpers import reference_definitions
from check_inputs_topology import run_inputs_topology
from check_equivalence import run_original_processor, run_terminal_skip
from check_training_metrics import scalar_tests, logging_and_diagnostics, ArtificialDataset, ArtificialReader
from check_metric_reporting import node_pooling_test
from check_module_dimensions import check_window
from check_dual_best import compare_tensors
from full_training_windows import FullTrainingWindows
from initialization import create_model
from training_windows import WindowReader
from training_step import train_group
from train import configure_runtime, new_optimizer, HISTORY_COLUMNS
from loss import compute_loss
from run_files import write_json
from training_config import CLIP_NORM, SEED


def check_tail_group(ref):
    """4+3个窗口：逐窗反向和先平均loss再反向应得到相同的两次更新。"""
    dataset = ArtificialDataset(ref)
    dataset.weight = lambda *args: 1.
    reader = ArtificialReader(dataset, "cpu")
    actual = create_model(SEED, "cpu")
    expected = copy.deepcopy(actual)
    optimizer, reference_optimizer = new_optimizer(actual), new_optimizer(expected)
    metrics_count, maximum_error = 0, 0.
    pairs = dataset.pairs("train")[:7]
    for offset in (0, 4):
        group = pairs[offset:offset+4]
        with patch.object(optimizer, "step", wraps=optimizer.step) as step, \
             patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero, \
             patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
            metrics, _ = train_group(actual, reader, group, optimizer)
            assert step.call_count == zero.call_count == clip.call_count == 1
        metrics_count += metrics.unweighted.mean.windows
        reference_optimizer.zero_grad(set_to_none=True)
        losses = []
        for sid, t in group:
            data, current, _, target = reader.window("train", sid, t)
            losses.append(compute_loss(expected(data)["prediction"], target, current, data)["loss"])
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), CLIP_NORM, error_if_nonfinite=True)
        reference_optimizer.step()
        maximum_error = max(maximum_error, compare_tensors(actual.state_dict(), expected.state_dict()))
        compare_tensors(optimizer.state_dict(), reference_optimizer.state_dict())
    assert metrics_count == 7
    assert all(float(state["step"]) == 2 for state in optimizer.state.values())
    return dict(group_sizes=[4, 3], optimizer_updates=2, metric_windows=7,
                max_parameter_error=maximum_error, atol=2e-5, rtol=2e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_runtime("cpu")
    report = dict(status="running", formal_training_started=False, test_dynamic_reads=0)
    files = list(HERE.glob("*.py")) + list((HERE/"checks").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    report["syntax_files"] = len(files)
    assert len(HISTORY_COLUMNS) == len(set(HISTORY_COLUMNS))
    ref = reference_definitions()
    for check in (run_inputs_topology, run_original_processor, run_terminal_skip,
                  scalar_tests, logging_and_diagnostics, node_pooling_test):
        check(ref, report) if check != node_pooling_test else check(report)
        print("PASS", check.__name__, flush=True)
    report["accumulation_tail"] = check_tail_group(ref)
    dataset = FullTrainingWindows(args.dataset_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_runtime(device)
    model = create_model(SEED, device)
    reader = WindowReader(dataset, device)
    report["device"] = device
    report["parameters"] = sum(p.numel() for p in model.parameters())
    assert report["parameters"] == 186695
    report["real_windows"] = [check_window(model, reader, scope, sid, t) for scope, sid, t in dataset.panel]
    report["data_windows"] = {scope: len(dataset.pairs(scope)) for scope in ("train", "full")}
    report["status"] = "passed"
    write_json(args.output/"checks.json", report)
    print("MODEL_AND_METRIC_CHECKS_PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()
