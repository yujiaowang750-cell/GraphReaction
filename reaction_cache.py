"""恢复30维反应属性后的可选缓存；与旧12维缓存使用不同身份和目录。"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from model_config import REACTION_SCHEMA, REACTION_PORT_DIM

CACHE_FOLDER = "reaction_b30_restored_v3"


def reaction_source_identity(phase, connections, relations, node_region):
    digest = hashlib.sha256()
    for array in (phase, connections, relations, node_region):
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def reaction_cache_path(cache_root, source_id):
    return Path(cache_root) / CACHE_FOLDER / f"{source_id}.pt"


def validate_reaction_cache(topology):
    required = ("reaction_node", "reaction_index", "reaction_role", "reaction_metadata",
                "reaction_center", "reaction_kind", "reaction_region", "reaction_degree",
                "reaction_size", "reaction_support_incidence", "reaction_support_physical_row")
    if not isinstance(topology, dict) or any(name not in topology for name in required):
        raise ValueError(f"反应缓存字段不完整；请用 build_reaction_topology 重建到 {CACHE_FOLDER}。")
    metadata = topology.get("reaction_metadata")
    nodes = topology.get("reaction_node")
    if nodes is None or metadata is None or metadata.shape != (len(nodes), REACTION_PORT_DIM):
        raise ValueError(f"反应缓存必须为 [P_R,30]；请用 build_reaction_topology 重建到 {CACHE_FOLDER}，不要改旧manifest。")
    if not torch.isfinite(metadata).all() or not ((metadata == 0) | (metadata == 1)).all():
        raise ValueError("反应缓存属性必须是有限 one-hot。")
    for start, end in ((0, 14), (14, 21), (21, 25)):
        if not (metadata[:, start:end].sum(1) == 1).all():
            raise ValueError("反应缓存的类型、成员 phase 或关联 role 分段无效。")
    if metadata[:, :12].any() or metadata[:, 21].any() or metadata[:, 25:].any():
        raise ValueError("反应端口只能使用类型12/13、角色1/2/3，几何及geometry_valid必须为0。")
    unit, role = topology["reaction_index"], topology["reaction_role"]
    kind, region = topology["reaction_kind"], topology["reaction_region"]
    if unit.shape != nodes.shape or role.shape != nodes.shape or region.shape != kind.shape:
        raise ValueError("反应缓存的关联与区域形状不匹配，请重建。")
    if len(unit) and (unit.min() < 0 or unit.max() >= len(kind)):
        raise ValueError("反应缓存的单元索引越界，请重建。")
    if not torch.isin(kind, kind.new_tensor([12, 13])).all() or not torch.equal(region, kind - 12):
        raise ValueError("反应区域必须为 NEG=0、POS=1，请重建。")
    if not torch.equal(metadata[:, :14].argmax(1), kind[unit]) or not torch.equal(metadata[:, 21:25].argmax(1), role):
        raise ValueError("反应缓存 one-hot 与类型/关联角色不一致，请重建。")


def save_reaction_cache(path, topology, source_id):
    validate_reaction_cache(topology)
    path = Path(path)
    if path.parent.name != CACHE_FOLDER:
        raise ValueError(f"当前反应缓存必须写入独立的 {CACHE_FOLDER} 目录。")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=REACTION_SCHEMA, source_identity=source_id, topology=topology), path)


def load_reaction_cache(path, source_id):
    if Path(path).suffix != ".pt":
        raise ValueError(f"旧反应缓存格式不兼容；请用 build_reaction_topology 重建到 {CACHE_FOLDER}。")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError(f"旧反应缓存或不兼容序列化格式；请重建到 {CACHE_FOLDER}，不修改旧文件。") from error
    if not isinstance(saved, dict) or saved.get("schema") != REACTION_SCHEMA:
        raise ValueError(f"旧反应缓存不兼容；请用 build_reaction_topology 重建到 {CACHE_FOLDER}，原缓存不修改。")
    if saved.get("source_identity") != source_id:
        raise ValueError("反应缓存的物理图来源不一致，请重建。")
    validate_reaction_cache(saved["topology"])
    return saved["topology"]
