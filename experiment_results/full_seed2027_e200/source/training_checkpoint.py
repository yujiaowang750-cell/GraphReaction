"""训练专用完整断点；先提交 last.pt，再落对应CSV行，恢复时补齐至多一个缺行。"""

import random
from pathlib import Path

import numpy as np
import torch

from run_files import append_csv, read_csv, replace_with_retry, copy_atomic, file_hash

TRAINING_CHECKPOINT_SCHEMA = "graphreaction_x13_b30_cosine_training_v2"


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=state[0], keys=torch.from_numpy(state[1].astype(np.int64)),
                position=state[2], has_gauss=state[3], cached_gaussian=state[4]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["name"], numpy["keys"].numpy().astype(np.uint32), numpy["position"],
                         numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(path, model, optimizer, scheduler, state, identity, update_row=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=TRAINING_CHECKPOINT_SCHEMA, identity=identity, model=model.state_dict(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), state=state,
        initialization=model.initialization_manifest, rng=rng_state(), last_update_row=update_row,
        pending_gradients=None, accumulation_pending_windows=0)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    replace_with_retry(temporary, path)


def load_training_checkpoint(path, model, optimizer, scheduler, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != TRAINING_CHECKPOINT_SCHEMA or payload.get("identity") != identity:
        raise ValueError("训练断点 schema/配置/数据/源码身份不一致；不兼容旧X17、b12或旧调度器断点。")
    if payload["accumulation_pending_windows"] != 0 or payload["pending_gradients"] is not None:
        raise ValueError("只允许在完成 optimizer.step 的边界恢复。")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    model.initialization_manifest = payload["initialization"]
    restore_rng(payload["rng"])
    return payload


def archive_epoch_checkpoint(last_path, epoch):
    """轮末last已经包含完整状态，独立复制保存；既有轮次绝不静默覆盖。

    在last提交后、归档复制前中断时，恢复可用同一last补齐归档。
    必须在完成训练、验证、scheduler.step和早停计数后调用。
    """
    path = Path(last_path).with_name(f"epoch_{epoch:04d}.pt")
    if path.exists():
        if file_hash(path) != file_hash(last_path):
            raise ValueError(f"已有第{epoch}轮checkpoint与已提交last不一致，拒绝覆盖：{path}")
    else:
        copy_atomic(last_path, path)
    return path


def reconcile_csv(path, committed_count, pending_row, columns, id_column):
    """CSV比checkpoint少最后一行时补写；其它不一致拒绝，不静默重放更新。"""
    rows = read_csv(path)
    ids = [int(row[id_column]) for row in rows]
    if ids != list(range(1, len(rows)+1)) or len(rows) > committed_count:
        raise ValueError(f"日志与checkpoint提交边界不一致：{path}")
    if len(rows) == committed_count:
        return
    if len(rows) == committed_count-1 and pending_row is not None and int(pending_row[id_column]) == committed_count:
        append_csv(path, pending_row, columns)
        return
    raise ValueError(f"日志缺失多于一个已提交记录，拒绝静默续训：{path}")
