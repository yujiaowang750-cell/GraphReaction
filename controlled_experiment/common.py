"""小型公共函数：读取原断点、严格恢复和分组损失。"""

import hashlib
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from initialization import create_model
from model_config import FIELD_NAMES, MODEL_SCHEMA
from schema import configuration_hash
from training_checkpoint import TRAINING_CHECKPOINT_SCHEMA, restore_rng
from train import new_optimizer, model_hash
from run_files import file_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def tensor_tree_hash(value):
    """包含优化器矩、步数和随机状态；字典键排序以避免保存顺序影响身份。"""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def load_parent(path, dataset, expected_sha, epoch=138):
    if file_hash(path) != expected_sha:
        raise ValueError("起点 checkpoint SHA256 不匹配，拒绝混入其他实验。")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    identity = saved["identity"]
    if saved["schema"] != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("起点不是原完整训练断点。")
    if identity["model"]["model_schema"] != MODEL_SCHEMA or identity["model_configuration_hash"] != configuration_hash():
        raise ValueError("模型 schema/配置不匹配。")
    if identity["data"] != dataset.identity:
        raise ValueError("原划分、窗口、归一化或数据 manifest 身份不匹配。")
    if saved["state"]["epoch"] != epoch or saved["state"]["cursor"] != 0:
        raise ValueError("起点不是指定 epoch 的完整边界。")
    if saved["pending_gradients"] is not None or saved["accumulation_pending_windows"] != 0:
        raise ValueError("起点有未提交梯度。")
    for key in ("optimizer", "rng", "scheduler"):
        if key not in saved:
            raise ValueError(f"缺少 {key}；不能声称完整恢复。请先设计共同权重热启动方案。")
    return saved


def initialize_branch(saved, device):
    model = create_model(device=device)
    model.load_state_dict(saved["model"], strict=True)
    model.initialization_manifest = saved["initialization"]
    optimizer = new_optimizer(model)
    # CPU上的load_state_dict可能复用输入张量；复制后A/B及原payload的矩状态才独立。
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    # 恢复原 AdamW 一、二阶矩和 step；本分支明确改为固定学习率。
    for group in optimizer.param_groups:
        group["lr"] = 1e-5
    restore_rng(saved["rng"])
    return model, optimizer


def contributions(losses):
    f = losses["field_losses"]
    ce, pe, cg, cp, pp, cbd, v = [f[name] for name in FIELD_NAMES]
    return {"C": (2 * ce + cg + cp) / 8,
            "P": (pe + pp + cbd) / 8,
            "V": v / 8, "I": .1 * losses["current_loss"]}


def objective(losses, beta=1.):
    """固定分母8；beta=1直接返回原张量，保持原CE2数值和梯度。"""
    if beta == 1.:
        return losses["loss"]
    return losses["loss"] + (beta - 1.) * contributions(losses)["P"]


def grouped(values):
    c = (2 * values["ce_MSE"] + values["cg_MSE"] + values["cp_MSE"]) / 4
    p = sum(values[name + "_MSE"] for name in ("phi_e", "phi_p", "phi_CBD")) / 3
    return {**values, "C_mean": c, "P_mean": p, "C_contribution": c / 2,
            "P_contribution": p * 3 / 8, "V_contribution": values["V_MSE"] / 8,
            "I_contribution": .1 * values["current_relative_MSE"]}
