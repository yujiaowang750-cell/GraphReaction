"""只保存/恢复同一新版模型；不加载旧 optimizer、scheduler 或 X17 权重。"""

from pathlib import Path
import pickle

import torch

from model_config import MODEL_SCHEMA
from schema import model_configuration, configuration_hash


def parameter_audit(model):
    groups = {
        "node_encoder": model.node_encoder,
        "reaction_port_encoder": model.processor.reaction_encoder.port_encoder,
        "reaction_rates": model.processor.reaction_encoder.rate_layers,
    }
    return {
        **{name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()},
        "total": sum(p.numel() for p in model.parameters()),
        "shapes": {name: list(p.shape) for name, p in model.named_parameters()},
    }


def save_checkpoint(path, model, statistics_id, dataset_id):
    """两个数据身份由调用者传入；它们不随模型 schema 改名。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        model_schema=MODEL_SCHEMA, configuration=model_configuration(),
        configuration_hash=configuration_hash(), statistics_identity=statistics_id,
        dataset_identity=dataset_id, initialization=model.initialization_manifest,
        parameter_audit=parameter_audit(model), state_dict=model.state_dict(),
    ), path)


def load_checkpoint(path, model, statistics_id, dataset_id):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError("旧 checkpoint 或不兼容序列化格式；请使用新版 create_model 重新初始化。") from error
    if not isinstance(saved, dict) or saved.get("model_schema") != MODEL_SCHEMA:
        raise ValueError("旧 checkpoint 与当前X13/b30模型不兼容，请 create_model 重新初始化；不自动迁移X17或b12权重。")
    if saved.get("configuration") != model_configuration() or saved.get("configuration_hash") != configuration_hash():
        raise ValueError("checkpoint 配置身份不一致，不能恢复。")
    if saved.get("statistics_identity") != statistics_id or saved.get("dataset_identity") != dataset_id:
        raise ValueError("checkpoint 的数据集或训练统计身份不一致。")
    if saved.get("parameter_audit") != parameter_audit(model):
        raise ValueError("checkpoint 参数形状/数量不兼容。")
    initialization = saved.get("initialization", {})
    if initialization.get("model_schema") != MODEL_SCHEMA or initialization.get("configuration_hash") != configuration_hash():
        raise ValueError("初始化 manifest 不属于当前新版，不能复用旧 paired 初始化。")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.initialization_manifest = initialization
    return model
