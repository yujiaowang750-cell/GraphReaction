"""读取一个真实窗口，运行完整 GraphReaction 前向、物理恢复和损失计算。"""

from pathlib import Path

import torch

from data_loader import get_sample_folder, load_static_graph, load_window
from initialization import create_model
from loss import compute_loss
from normalization import load_statistics
from state_output import build_delta_targets, recover_physical_state


def main():
    # 改这三个位置，就可以检查另一条轨迹或另一个原始时间索引。
    dataset_root = Path(__file__).resolve().parents[1] / "Datasets"
    sample_index = 1
    time_index = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)

    statistics = load_statistics(dataset_root)
    sample_folder = get_sample_folder(dataset_root, sample_index)
    graph = load_static_graph(sample_folder, statistics)
    data, current, future = load_window(sample_folder, time_index, statistics, graph, device)
    target = build_delta_targets(current, future, data)
    model = create_model(seed=2026, device=device)
    model.eval()
    with torch.no_grad():
        output = model(data)
        losses = compute_loss(output["prediction"], target, current, data)
        predicted_state, physical_mask = recover_physical_state(current, output["prediction"], data)

    print("样本：", data.sample_id, "；输入时间索引：", time_index, "；设备：", device)
    print("模型参数量：", sum(parameter.numel() for parameter in model.parameters()))
    print("节点输入：", tuple(data.x.shape))
    print("物理边：", data.edge_index_u.shape[1], "；有向边：", data.edge_index.shape[1])
    print("反应单元：", len(data.reaction_center), "；成员记录：", len(data.reaction_node))
    for name in ("node_hidden", "reaction_hidden", "reaction_rate"):
        print(name, tuple(output[name].shape))
    for field, prediction in output["prediction"].items():
        print(field, "预测：", tuple(prediction.shape), "；标签：", tuple(target[field].shape))
    print("未来物理状态：", tuple(predicted_state.shape), "；有效 mask：", tuple(physical_mask.shape))
    print("CE2 总损失：", float(losses["loss"]))
    print("七字段加权主损失：", float(losses["main_loss"]))
    print("电流辅助损失：", float(losses["current_loss"]))
    print("预测电流 A：", float(losses["predicted_current_A"]))
    print("这是随机初始化模型的调用检查，数值不代表训练后的预测效果。")


if __name__ == "__main__":
    main()
