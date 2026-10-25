r"""
GraphReaction：从输入到输出的阅读顺序

重写依据：用户指定 study 下的 src/model.py、graph_topology.py、
reaction_topology.py、regions.py，以及它们实际调用的数据和物理公式。
这里只保留 GraphReaction 模型，不包含训练队列、资源门槛或实验调度。
参考版本：hybrid3r_seed2026_20261010_072635，初始化种子 2026。

一、先看整体

    当前状态 X(t) + 静态节点特征
                |
             [N,17]
                |
       共享节点编码器 17 -> 64
                |
          初始节点状态 H0
          /             \
    分相池化并编码       反应成员编码、均值池化
       得到 g0            得到 Z0 和成员 rate
          \             /
          四次同步更新 H、Z
       /        |          \
   物理图消息  反应差值消息  区域控制量
                |
            最终节点 H4
                |
        再次分相池化得到 g1
                |
       [H4;g1] -> 六个分相线性头
                |
        七个字段的标准化增量
                |
     反标准化 + 当前物理状态 X(t)
                |
         未来状态 X(t+1)

N 是节点数，E 是原始物理边数，R 是反应单元数，L 是反应成员记录数。
一个节点可属于多个反应单元，因此 L 通常不等于 N。
四次同步更新属于网络内部计算，预测目标仍为原数据的 t+1（10 秒后）。

二、数据与特征文件

normalization.py
    从当前项目 Datasets/normalization_stats.json 读取训练统计量。
    保留有效标准差；验证时使用同一份统计。

data_loader.py
    get_sample_folder：按 sample_index 查找样本文件夹。
    load_static_graph：读取静态图，连接区域、物理图、反应单元和几何处理函数。
    load_window：读取指定 t 和 t+1，返回 data、current、future。
    模型只接收 data，其中 x 只包含 t 时刻信息。
    current/future 为 [N,2]，列为原始浓度 mol/m^3、电势 V。

static_features.py
    静态节点 [N,4]：已归一化 z/y/x + 体积 z-score。
    原始边几何 [E,4]：面积、起点长度、中间长度、终点长度。
    两端长度共用统计，中间长度单独统计。
    全局范围 [6]：z_min,z_max,y_min,y_max,x_min,x_max 做 z-score。
    这些量都使用保存的训练统计，不做当前图 log(1+x) 标准化。

node_features.py
    节点输入共 17 维：
      0～2   theta、ce_norm、phi_norm
      3～5   对应的三个有效 mask
      6～9   z、y、x、volume_norm
      10～16 phase 1～7 的 one-hot
    phase 1：theta；phase 2/5：ce、phi；phase 3：theta、phi；
    phase 4/7：phi；phase 6：没有动态输入。
    石墨/AM 浓度除以各自 cmax 得到 theta，其余有效字段用训练输入统计。

regions.py
    节点划分为负极 0、隔膜 1、正极 2。
    phase 2 根据 z 坐标在隔膜 phase 5 范围的哪一侧划分。
    物理边划分为三个区域内部边、负极-隔膜界面、隔膜-正极界面，共五组。

graph_topology.py
    每条原物理边保留两个相邻方向，所有平行连接均保留。
    每条有向边起点有一个 30 维说明向量：
      类型 one-hot 14 + 节点相 one-hot 7 + 角色 one-hot 4 + 几何 4 + 几何标记 1。
    类型 0～11 是物理关系，12/13 留给负极/正极反应单元。
    物理边角色为 0，反向边交换起终点长度。
    没有额外的“二元超边隐状态”，物理边只负责传递消息。

reaction_topology.py
    以 phase 1 石墨或 phase 3 AM 为中心。
    每个单元包含 CENTER（角色1）、ION（角色2）、ELECTRON（角色3）。
    ION 是同区域、一步相邻的 phase 2 电解液。
    ELECTRON 是按指定关系连接的同区域电子侧邻居。
    中心必须有 ION；没有 ELECTRON 也保留，两个成员的单元也保留。
    同一角色内去掉重复成员，不删除原始物理图中的平行边。
    反应成员也使用 30 维说明向量，其中几何四维和几何标记均为零。

三、模型计算文件

node_encoder.py
    共享投影：17 -> 64 -> ReLU -> LayerNorm。
    所有相共用同一份参数。

global_encoder.py
    七相各求 64 维节点均值，拼接 6 维范围：454 -> 64 -> ReLU -> LayerNorm。
    输出一维整图向量 [64]。
    H0 生成 g0，H4 生成 g1，两次使用同一个编码器。
    g0 在四次更新中固定且保留梯度；g1 只用于解码。

graph_messages.py
    五组各有独立 MLP：188 -> 64 -> ReLU -> LayerNorm -> 64。
    188 = 起点状态64 + 终点状态64 + 起点说明30 + 反向端说明30。
    各方向的消息汇到接收节点，除以接收节点的全部物理入边数。
    不是五组各自求均值后再求和。

reaction_encoder.py
    成员投影：[节点64;成员说明30] = 94 -> 64 -> ReLU -> LayerNorm。
    同一反应单元的成员均值为 [R,64]。
    拼接 log(1+成员数)：65 -> 64 -> ReLU -> LayerNorm，得到 Z0。
    负极/正极各一个 Linear(30,1)，再 Sigmoid，得到成员 rate [L,1]。
    rate 只读取成员说明向量，一次前向计算一次，四次更新复用。

reaction_messages.py / aggregation.py
    每条成员记录 q = rate * (Z[所属单元] - H[成员节点])。
    节点按自身关联的反应成员记录数，对 q 求均值，得到 D_node_reaction。
    单元按自身成员数，对 -q 求均值，得到 D_unit_reaction。
    没有成员或邻居的位置返回零，分母不是 rate 之和。

node_control.py / reaction_control.py
    节点三个区域各有一份控制 MLP；反应单元负极/正极各一份。
    都是 [自身64;g0的64] -> 128 -> 64 -> ReLU -> LayerNorm -> 64。
    节点输出 U，单元输出 U_R。

processor.py
    每次更新同时读取旧 H 和旧 Z：
      H_new = H + 0.25 * (D_graph + D_node_reaction + 0.1*U)
      Z_new = Z + 0.25 * (D_unit_reaction + 0.1*U_R)
    共执行四次；四次使用同一套参数。
    更新末尾不额外添加 LayerNorm、门控或投影。

decoder.py
    拼接最终节点 H4 和最终全局 g1，每个节点 128 维。
    phase 1/4/7：Linear(128,1)。
    phase 2/3/5：Linear(128,2)。
    phase 6 无解码头。输出线性，允许负增量。
    最终输出字典的七个字段依次是：
      电解液浓度、电解液电势、石墨锂化度、AM锂化度、AM电势、CBD电势、正集流体电势。
    每项形状是 [对应字段的节点数]；电解液字段覆盖 phase 2 和 phase 5。

initialization.py
    create_model() 执行原初始化，默认 seed=2026。
    Linear 使用 Xavier，偏置为零；LayerNorm 权重1、偏置0。
    控制 MLP 的最后一层置零；rate 权重和偏置置零，初始 rate=0.5。
    同组模块复制相同初值，但参数存储独立。
    模型参数总数应为 186951。

model.py
    只组织上述模块；这是检查整体前向顺序的入口。

四、输出与损失

state_output.py
    build_delta_targets：原始未来减当前，再按训练增量均值/标准差标准化。
    recover_physical_state：预测反标准化后加回当前物理状态。
    石墨/AM 增量尺度乘 cmax 后再回加，输出浓度仍为 mol/m^3。
    phase 6 电势固定为零；未预测的石墨电势及不存在的浓度位置由 mask 标为无效。

current_loss.py
    由 AM/CBD 到正极集流体的原始物理接触计算电流。
    G = sigma * 面积(m^2) / 固相一侧长度(m)。
    I_pred = sum(G * (phi_solid - phi_collector))。
    I_app 读取当前轨迹的 sample_manifest；每条无向接触只计一次。

loss.py
    七字段各在标准化增量空间算 MSE。
    两个电解液字段都先按 phase 2、phase 5 分别求均值，再 0.5:0.5 合并。
    CE2 = (2*L_ce + 其余六个字段的loss) / 8 + 0.1*L_current。
    正极集流体电势属于上述七字段，不另加一项电压辅助损失。
    保留原 Q_ref 参考指标：七字段等权均值 + 0.1*L_current。

五、最小调用

    statistics = load_statistics(dataset_root)
    folder = get_sample_folder(dataset_root, sample_index=1)
    graph = load_static_graph(folder, statistics)
    data, current, future = load_window(folder, 0, statistics, graph, device)
    target = build_delta_targets(current, future, data)
    model = create_model(seed=2026, device=device)
    output = model(data)
    losses = compute_loss(output['prediction'], target, current, data)

从项目根目录运行：
    .venv/Scripts/python.exe GraphReaction/use_model.py

这一轮只重写模型、数据处理、物理恢复和损失，不启动训练。
"""
