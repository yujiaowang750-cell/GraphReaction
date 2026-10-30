"""人工小图验证141轮断点延长，不启动真实训练，不改历史实验。"""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "checks")]

import train
from continue_training import read_parent
from check_training_metrics import ArtificialDataset, ArtificialReader, assert_nested_equal
from initialization import create_model
from learning_rate import create_scheduler, MIN_LR
from run_files import file_hash, read_csv, write_csv, write_json
from schema import model_configuration, configuration_hash, content_hash
from training_checkpoint import save_training_checkpoint, load_training_checkpoint
from training_config import training_configuration
from training_metrics import BASE_METRICS
from training_step import train_group
from v2_helpers import reference_definitions


def make_parent(path, dataset, best_value=0.):
    """合成141轮边界；只做一次小图更新以生成真实AdamW动量，不伪装真实实验。"""
    path.mkdir()
    settings = training_configuration("full")
    settings.update(device="cpu", smoke_windows=5)
    identity = dict(model=model_configuration(), model_configuration_hash=configuration_hash(),
                    training=settings, data=dataset.identity, source=train.runtime_sources(), seed=2026,
                    runtime=dict(torch=str(torch.__version__)), smoke=True)
    identity["configuration_hash"] = content_hash(identity)
    write_json(path / "config.json", dict(identity=identity))
    (path / "source").mkdir()
    for name in identity["source"]:
        shutil.copy2(HERE / name, path / "source" / name)
    model = create_model()
    optimizer = train.new_optimizer(model)
    scheduler = create_scheduler(optimizer)
    train_group(model, ArtificialReader(dataset, "cpu"), dataset.pairs("train")[:2], optimizer)
    for _ in range(141):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == MIN_LR
    state = train.initial_state()
    state.update(epoch=141, updates=141, best_epoch=138, best_update=138,
                 best_train_CE2=best_value, early_stop_counter=3, complete=True, stop_reason="epoch_budget")
    state.pop("reported")
    state.pop("best_Q_ref_at_CE2_best")
    for epoch in range(1, 142):
        ce2 = best_value if epoch == 138 else best_value + max(1., best_value)
        row = dict(epoch=epoch, updates=epoch, lr=MIN_LR, next_lr=MIN_LR,
                   is_best=epoch in (1, 138), best_epoch=138 if epoch >= 138 else 1,
                   validation_status="not_run", selection_metric="online_train_CE2")
        row.update({"online_train_" + key: ce2 for key in (*BASE_METRICS, "current_relative_RMSE_pct")})
        state["history"].append(row)
    rows = [dict(update=i, epoch=i, cursor=5, lr=MIN_LR, grad_norm_pre_clip=0., clipped=False,
                 nonfinite_grad_count=0, status="ok", order_sha256="synthetic") for i in range(1, 142)]
    state["last_update_row"] = rows[-1]
    write_csv(path / "update_metrics.csv", rows, train.UPDATE_COLUMNS)
    save_training_checkpoint(path / "checkpoints/last.pt", model, optimizer, scheduler, state, identity)
    best_state = copy.deepcopy(state)
    best_state.update(epoch=138, updates=138, history=state["history"][:138], complete=False,
                      stop_reason=None, early_stop_counter=0)
    save_training_checkpoint(path / "checkpoints/best.pt", model, optimizer, scheduler, best_state, identity)
    return identity


def load_last(run):
    return torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=True)


def main():
    train.configure_runtime("cpu")
    dataset = ArtificialDataset(reference_definitions())
    dataset.mode = "full"
    dataset.weight = lambda *args: 1.
    dataset.panel = [("train", 1, 0), ("full", 0, 0)]
    report = dict(formal_training=False, real_data_evaluation=False)
    with tempfile.TemporaryDirectory(prefix="graphreaction_continue_") as tmp:
        root = Path(tmp)
        parent = root / "parent"
        make_parent(parent, dataset)
        hashes = {str(p.relative_to(parent)): file_hash(p) for p in parent.rglob("*") if p.is_file()}
        with patch("train.WindowReader", ArtificialReader):
            a, b = root / "continuous", root / "resumed"
            train.run_training(a, dataset, 143, "cpu", smoke_windows=5, continue_from=parent)
            train.run_training(b, dataset, 143, "cpu", smoke_windows=5, continue_from=parent,
                               pause_after_updates=142)
            train.run_training(b, dataset, 143, "cpu", smoke_windows=5, resume=True)
            one, two = load_last(a), load_last(b)
            for key in ("model", "optimizer", "scheduler", "rng"):
                assert_nested_equal(one[key], two[key])
            assert_nested_equal(one["state"]["reported"], two["state"]["reported"])
            assert one["state"]["best_epoch"] == 138 and one["state"]["early_stop_counter"] == 5
            assert file_hash(a / "checkpoints/best.pt") == file_hash(parent / "checkpoints/best.pt")
            for row in read_csv(a / "history.csv")[141:]:
                assert float(row["lr"]) == float(row["next_lr"]) == MIN_LR
                assert row["train_CE2"] == row["online_train_CE2"] and row["val_CE2"] == ""
            for row in read_csv(a / "history.csv")[:141]:
                assert row["train_ce_NRMSE_delta"] == ""
            report["resume_parameter_optimizer_rng_max_error"] = 0.
            report["fixed_lr_and_inherited_best"] = "PASS: 142/143 at 1e-5, best138 retained, counter3->5"
            # 0为不可严格降低的历史best，必须在累计20轮未改善时停于158，而非再等20轮。
            stopped = train.run_training(root / "early_stop", dataset, 193, "cpu", smoke_windows=5,
                                         continue_from=parent)
            assert stopped["epochs"] == 158 and stopped["best_epoch"] == 138
            report["early_stop"] = "PASS: inherited counter=3; stops at158 after17 additional epochs"
            new_parent = root / "new_best_parent"
            make_parent(new_parent, dataset, best_value=1e100)
            run = root / "new_best"
            result = train.run_training(run, dataset, 142, "cpu", smoke_windows=5, continue_from=new_parent)
            assert result["best_epoch"] == 142
            best = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
            assert best["identity"] == load_last(run)["identity"]
            report["new_best"] = "PASS: new training CE2 best replaces inherited best and is finally evaluated"
        assert hashes == {str(p.relative_to(parent)): file_hash(p) for p in parent.rglob("*") if p.is_file()}
        report["parent_files_unchanged"] = True
        # 普通恢复接口仍拒绝旧身份；只有显式continue入口可以建立有来源的新实验。
        model = create_model(); optimizer = train.new_optimizer(model); scheduler = create_scheduler(optimizer)
        try:
            load_training_checkpoint(parent / "checkpoints/last.pt", model, optimizer, scheduler, one["identity"])
        except ValueError:
            report["strict_resume"] = "PASS: incompatible old identity rejected"
        else:
            raise AssertionError("旧断点不能静默恢复成新版身份")
        bad = copy.copy(dataset); bad.identity = {"different": True}
        try:
            read_parent(parent, bad, 193)
        except ValueError:
            report["wrong_data_rejected"] = True
        else:
            raise AssertionError("不同数据必须拒绝")
    report["status"] = "PASS"
    write_json(HERE / "checks/continuation_result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
