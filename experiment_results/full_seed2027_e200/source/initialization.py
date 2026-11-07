"""初始化参数；同组模块初值相同，但训练参数各自独立。"""

import torch
from torch import nn

from model import GraphReactionModel
from model_config import MODEL_SCHEMA
from schema import configuration_hash
from checkpoint import parameter_audit


def create_model(seed=2026, device="cpu"):
    """创建模型并执行原初始化"""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = GraphReactionModel()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        processor = model.processor
        for layer in processor.reaction_encoder.rate_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)  # sigmoid(0)=0.5。
        for controls in (processor.node_control.mlps, processor.reaction_control.mlps):
            for mlp in controls:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)  # 初始控制量为零。

        groups = (
            processor.graph_messages.mlps, processor.node_control.mlps,
            processor.reaction_control.mlps, processor.reaction_encoder.rate_layers,
        )
        for modules in groups:
            for module in modules[1:]:
                module.load_state_dict(modules[0].state_dict())
    model.initialization_manifest = dict(
        model_schema=MODEL_SCHEMA, configuration_hash=configuration_hash(), seed=seed,
        rule="Xavier; bias=0; LayerNorm=1/0; rate=0; control_last=0; group_copy_independent_storage",
        parameter_audit=parameter_audit(model),
    )
    return model.to(device)
