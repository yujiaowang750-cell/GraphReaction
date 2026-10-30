# GraphReaction 当前超参数说明

核对日期：2026-10-27。

本文按当前代码和本次全量实验保存的配置整理，模型版本为 `graphreaction_x13_b30_skipz4_v3`。节点输入为 13 维，普通图和反应端口均为 30 维，默认跳过末轮 Z4。本文描述当前版本，不修改历史实验的定义。

本次实验配置：[full_train_ce2_20261026_215337/config.json](training_runs/full_train_ce2_20261026_215337/config.json)。

## 1. 模型结构与消息更新参数

主要位置：[model_config.py](model_config.py)。

| 参数 | 当前值 | 作用 |
|---|---|---|
| `HIDDEN_DIM` | 64 | 节点 H、反应超边 Z、全局 g 的隐藏维度 |
| `PROCESSOR_STEPS` | 4 | 每次前向中的节点更新轮数 |
| `STEP_SIZE` | 0.25 | 隐状态更新步长 eta |
| `CONTROL_WEIGHT` | 0.1 | 节点和反应 control 项的权重 gamma |
| 激活函数 | ReLU | 编码器、普通图消息和 control 网络使用 |
| rate 激活函数 | Sigmoid | 把反应关联强度映射到 0～1 |
| `LAYER_NORM_EPS` | 1e-5 | LayerNorm 的数值稳定项 |
| Dropout | 未使用 | 当前模型没有 Dropout 层 |

每轮节点更新为：

$$
H^{k+1}=H^k+0.25\left(D^k+R^k+0.1U^k\right).
$$

前三轮的反应超边更新为：

$$
Z^{k+1}=Z^k+0.25\left(DR^k+0.1UR^k\right).
$$

其中，D 是普通图消息，R 是节点收到的反应消息，U 是节点 control；DR 和 UR 分别是超边收到的反应消息与超边 control。所有右侧使用同一轮更新前的状态。

四轮依次为 `H0/Z0 → H1/Z1 → H2/Z2 → H3/Z3 → H4`。最后一轮仍计算节点反应消息，但默认不计算 DR、UR 和 Z4。`compute_terminal_reaction=True` 仅用于显式诊断，可补算 Z4。

实现位置：[processor.py](processor.py)。

## 2. 各模块的层数、维度与分组

下表中的箭头表示 Linear 层的输入、输出维度。

| 模块 | 当前维度 | 分组及共享方式 | 文件 |
|---|---|---|---|
| 节点编码器 | 13 → 64 | 七相节点共用一个编码器 | [node_encoder.py](node_encoder.py) |
| 全局编码器 | 454 → 64 | 初始化 g0 和最终读出 g1 共用一个编码器 | [global_encoder.py](global_encoder.py) |
| 反应端口编码器 | 94 → 64 | 所有反应关联共用一个编码器 | [reaction_encoder.py](reaction_encoder.py) |
| 反应初始化器 | 65 → 64 | 所有反应单元共用一个 initializer | [reaction_encoder.py](reaction_encoder.py) |
| 普通图消息 MLP | 188 → 64 → 64 | 负极内部、隔膜内部、正极内部、负极—隔膜、隔膜—正极，共 5 组 | [graph_messages.py](graph_messages.py) |
| 节点 control MLP | 128 → 64 → 64 | 负极、隔膜、正极，共 3 组 | [node_control.py](node_control.py) |
| 反应 control MLP | 128 → 64 → 64 | 负极、正极，共 2 组 | [reaction_control.py](reaction_control.py) |
| 反应 rate 网络 | 30 → 1 | 负极、正极，共 2 组，输出后接 Sigmoid | [reaction_encoder.py](reaction_encoder.py) |
| 分相解码器 | 128 → 1 或 2 | phase 1、2、3、4、5、7 各一个头，共 6 个 | [decoder.py](decoder.py) |

节点、全局、反应端口编码器和反应初始化器均为 `Linear → ReLU → LayerNorm`，各含一个 Linear。

普通图消息和两类 control 网络均为 `Linear → ReLU → LayerNorm → Linear`，各含两个 Linear。解码头为单个 Linear。phase 1、4、7 输出 1 维，phase 2、3、5 输出 2 维；phase 6 不设置预测头。

四轮更新共享参数；不同区域或消息组的网络参数彼此独立。Z0、g0 和 rate 在每次前向中各初始化或计算一次，g0 和 rate 在内部各轮固定，但保留梯度。

## 3. 训练与优化器参数

主要位置：[training_config.py](training_config.py)。优化器在 [train.py](train.py) 的 `new_optimizer()` 中实际创建。

| 参数 | 当前值 | 作用 |
|---|---|---|
| 优化器 | AdamW | 更新模型参数 |
| `INITIAL_LR` | 0.001 | 初始学习率 |
| `MIN_LR` / `eta_min` | 0.00001 | 余弦退火最低学习率 |
| 调度器 | CosineAnnealingLR | 按余弦曲线调整学习率 |
| `T_max` | 141 | 余弦下降周期，单位为 epoch |
| 最大训练轮数 | 141 | 本次实验的训练预算上限 |
| `weight_decay` | 1e-6 | AdamW 权重衰减系数 |
| `betas` | (0.9, 0.999) | Adam 一阶、二阶动量系数 |
| `adam_eps` | 1e-8 | 优化器数值稳定项 |
| 每次前向的图窗口数 | 1 | 单个窗口独立前向、反向 |
| `ACCUMULATION` | 4 | 通常累计 4 个窗口的梯度再更新一次参数 |
| `CLIP_NORM` | 1.0 | 每次参数更新前的梯度范数裁剪阈值 |
| `SEED` | 2026 | 初始化和随机打乱使用的种子 |
| `EARLY_STOPPING` | 20 | 连续 20 轮选择指标没有严格降低时停止 |
| best 选择指标 | 每轮平均训练 CE2 | 指标严格降低时保存 best |

梯度累积在 [training_step.py](training_step.py) 的 `train_group()` 中实现：每个窗口的损失先除以本组窗口数再反向传播；一组结束后裁剪梯度，再调用一次 `optimizer.step()`。最后一组不足 4 个窗口时，按真实窗口数平均。

`online_train_CE2` 是一轮训练过程中各窗口损失的平均，模型参数在这一轮内持续更新。它不是在轮末固定模型后，重新遍历整个训练集得到的损失。

### 学习率公式与调用时机

位置：[learning_rate.py](learning_rate.py)。

$$
\operatorname{lr}(t)=10^{-5}+\frac{10^{-3}-10^{-5}}{2}
\left(1+\cos\left(\frac{\pi t}{141}\right)\right).
$$

这里 t 是已经完成的 epoch 数；它与消息更新步长 `STEP_SIZE=0.25` 是不同的量。

- 第 1 轮训练使用 `lr(0)=0.001`。
- 每轮训练完成后调用一次 `scheduler.step()`。
- 第 141 轮训练使用 `lr(140)`，该轮结束后才得到 `lr(141)=0.00001`。
- 没有 warmup 或热重启；调度器不接收训练或验证损失。
- 如果提前停止，仍沿用 T=141 的曲线，不重新缩短周期。

## 4. 损失权重

权重定义在 [model_config.py](model_config.py)，计算在 [loss.py](loss.py) 的 `compute_loss()` 中。

| 字段或损失项 | 当前权重 |
|---|---:|
| 电解液浓度 ce | 2 |
| 电解液电势 phi_e | 1 |
| 石墨嵌锂分数 theta_Gr | 1 |
| 正极 AM 嵌锂分数 theta_AM | 1 |
| 正极 AM 电势 phi_AM | 1 |
| 正极 CBD 电势 phi_CBD | 1 |
| 正极集流体电势 phi_cc+ | 1 |
| 电流辅助损失 `CURRENT_LOSS_WEIGHT` | 0.1 |

七个字段在标准化增量空间计算 MSE。电解液浓度和电势分别先计算 phase 2、phase 5 内的节点平均 MSE，再按 `0.5 : 0.5` 合并。这两个区域系数目前直接写在 `loss.py` 中。

训练目标为：

$$
\mathrm{CE2}=\frac{2L_{c_e}+\sum_{f\ne c_e}L_f}{8}+0.1L_I.
$$

用于记录比较的参考指标为：

$$
Q_{\mathrm{ref}}=\frac{1}{7}\sum_f L_f+0.1L_I.
$$

这里的求和均针对上述七个预测字段。正极集流体电势已经包含在七字段损失中，没有额外叠加独立电压辅助项。当前按 CE2 反向传播和选择 best；Q_ref 用于记录，不参与本次 best 选择。

## 5. 初始化与数值设置

初始化位置：[initialization.py](initialization.py)。

| 对象 | 初始化方式 |
|---|---|
| 普通 Linear 权重 | Xavier uniform |
| Linear 偏置 | 0 |
| LayerNorm 权重 / 偏置 | 1 / 0 |
| rate 层权重 / 偏置 | 0 / 0，因此初始 rate 为 sigmoid(0)=0.5 |
| 两类 control 的末层权重 / 偏置 | 0 / 0，因此初始 control 输出为零 |
| 同类分组网络 | 复制初始参数值，但保持各组参数存储独立 |

rate 的初值 0.5 是初始化规则；训练后 rate 由网络参数决定，不是始终固定为 0.5。

数值和运行设置位于 [train.py](train.py)、[training_config.py](training_config.py) 及相关特征、损失模块：

| 设置 | 当前值 |
|---|---|
| 本次训练设备 | CUDA GPU |
| 网络输入及隐藏表示 | float32 |
| 原始物理状态、增量目标、物理恢复及损失计算 | float64 |
| AMP 混合精度 | 关闭 |
| TF32 | 关闭 |
| PyTorch CPU 线程数 | 4 |
| 数据加载 worker 数 | 0 |
| 静态图缓存数量 | 1 |
| `cudnn_deterministic` | True |
| `deterministic_algorithms` | False |
| 窗口打乱 | 先打乱轨迹，再打乱各轨迹内窗口，使用 MT19937 |

上述种子和后端设置不表示所有 GPU 运算都保证逐位确定性。

## 6. 本次全量实验的数据和验证流程

数据目录为 `D:\Github Project\NewProgramByMe\Datasets`，使用 [trajectory_split.json](../Datasets/trajectory_split.json) 中的划分。

| 项目 | 当前设置 |
|---|---|
| 数据模式 | full |
| 训练数据 | 65 条完整轨迹，共 22,927 个窗口 / epoch |
| 最终验证数据 | 14 条完整验证轨迹，共 4,917 个窗口 |
| 窗口权重 | 训练和完整验证均为 1 |
| 预测跨度 | 一个原数据时间间隔，当前为 10 秒 |
| 每轮验证 | 不进行 |
| 最终验证 | 训练结束后加载训练 CE2 最低的 best，在完整验证集上评估 |
| 验证输入 | 每个窗口使用真实当前状态，单步预测下一时刻 |
| 测试集 | 本次实验不评估 |
| 诊断 | 训练结束后对 best 的指定窗口进行诊断 |

训练入口为 [train.py](train.py)，完整训练集窗口读取为 [full_training_windows.py](full_training_windows.py)。本节描述本次 full 实验，32 档子集模式另有对应的窗口和评价设置。

## 7. 与超参数区分的输入定义、物理参数

| 项目 | 当前定义或数值 | 来源 |
|---|---|---|
| 节点输入维度 | 13：七相 one-hot + Z/Y/X + 体积 + 浓度槽 + 电势槽 | [node_features.py](node_features.py) |
| 普通图端口维度 | 30 | [graph_topology.py](graph_topology.py) |
| 反应端口维度 | 30 | [reaction_topology.py](reaction_topology.py) |
| 全局输入维度 | 7 × 64 + 6 = 454 | [global_encoder.py](global_encoder.py) |
| 石墨最大浓度 | 33133 mol/m³ | 各样本 `sample_manifest.json` |
| 正极 AM 最大浓度 | 48900 mol/m³ | 各样本 `sample_manifest.json` |
| AM 电子电导率 | 0.01 S/m | `AM_CONDUCTIVITY` |
| CBD 电子电导率 | 760 S/m | `CBD_CONDUCTIVITY` |
| 归一化统计 | 已保存的训练集统计，不在当前训练中重新拟合 | [normalization.py](normalization.py) |

最大浓度由 [data_loader.py](data_loader.py) 按样本读取。当前检查的 93 条轨迹均为上述数值；石墨和 AM 的输入分别按 `c / cmax` 转换为嵌锂分数。它们是数据提供的材料参数，不是通过观察轨迹中的浓度最大值计算出来的。

输入维度和字段定义决定数据接口，材料参数决定物理计算含义，均应与学习率、隐藏维度、更新轮数等超参数区分。

## 8. 查看或调整时对应的文件

| 要查看的设置 | 代码位置 |
|---|---|
| 隐藏维度、更新轮数、eta、gamma、LayerNorm eps | [model_config.py](model_config.py) |
| 七字段权重、电流损失权重 | [model_config.py](model_config.py) |
| 电解液两个区域的损失合并权重 | [loss.py](loss.py) |
| 初始学习率、最低学习率、余弦周期 | [learning_rate.py](learning_rate.py) |
| 种子、梯度累积、裁剪阈值、早停轮数 | [training_config.py](training_config.py) |
| 优化器实际创建、训练流程及后端设置 | [train.py](train.py) |
| 单次梯度累积和参数更新 | [training_step.py](training_step.py) |
| 激活函数、MLP 层数 | 各 encoder、messages、control 模块 |
| 参数初始化规则 | [initialization.py](initialization.py) |

部分配置是说明记录，实际创建参数同时位于实现文件中。例如 AdamW 的 `weight_decay`、`betas`、`eps` 在 `training_config.py` 中记录，在 `train.py` 中实际传入，调整时应保持一致。部分层的输入维度仍明确写为 128 或 65，调整隐藏维度时也需要同步核对对应拼接和 Linear 层。

实验目录的 `config.json` 与 `source/` 保存启动时的配置和代码。修改当前源码不会自动改变已经启动的训练进程所使用的模型与训练设置。
