"""新版身份与维度检查；模型 schema 和原数据/归一化统计身份分开保存。"""

import hashlib
import json

from model_config import (
    MODEL_SCHEMA, NODE_SCHEMA, PAIR_SCHEMA, REACTION_SCHEMA,
    NODE_INPUT_DIM, PAIR_PORT_DIM, REACTION_PORT_DIM, HIDDEN_DIM,
    PROCESSOR_STEPS, STEP_SIZE, CONTROL_WEIGHT, LAYER_NORM_EPS,
    FIELD_NAMES, FIELD_WEIGHTS, CURRENT_LOSS_WEIGHT,
)


def content_hash(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_identity():
    return dict(node=NODE_SCHEMA, pair=PAIR_SCHEMA, reaction=REACTION_SCHEMA)


def model_configuration():
    return dict(
        model_schema=MODEL_SCHEMA, feature_identity=feature_identity(),
        node_input_dim=NODE_INPUT_DIM, pair_port_dim=PAIR_PORT_DIM,
        reaction_port_dim=REACTION_PORT_DIM, hidden_dim=HIDDEN_DIM,
        processor_steps=PROCESSOR_STEPS, eta=STEP_SIZE, gamma=CONTROL_WEIGHT,
        activation="ReLU", layer_norm_eps=LAYER_NORM_EPS,
        compute_terminal_reaction=False, fields=list(FIELD_NAMES),
        decoder_columns=[0, 1, 0, 0, 1, 0, 0], field_weights=list(FIELD_WEIGHTS),
        current_loss_weight=CURRENT_LOSS_WEIGHT,
        input_dtype="float32", physical_and_loss_dtype="float64",
        normalization="unchanged_saved_train_statistics", prediction_horizon=1,
    )


def configuration_hash():
    return content_hash(model_configuration())


def statistics_identity(statistics):
    """只对已有统计做身份校验，不拟合、不改动数据文件。"""
    return content_hash(statistics)


def validate_features(data):
    if getattr(data, "feature_identity", None) != feature_identity():
        raise ValueError("输入 schema 不兼容：请用新版 load_static_graph/load_window 重建X13和两类30维端口。")
    shapes = {
        "x": (data.num_nodes, NODE_INPUT_DIM),
        "edge_metadata": (data.edge_index.shape[1], PAIR_PORT_DIM),
        "reaction_metadata": (len(data.reaction_node), REACTION_PORT_DIM),
    }
    for name, shape in shapes.items():
        if tuple(data[name].shape) != shape:
            raise ValueError(f"{name} 需要 {shape}，实际为 {tuple(data[name].shape)}；不能裁剪旧缓存绕过检查。")
