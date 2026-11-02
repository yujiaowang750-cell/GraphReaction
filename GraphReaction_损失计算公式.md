# GraphReaction 损失计算公式与指标统计

整理日期：2026-10-30。

本文按照当前项目代码整理，适用于当前 X13 节点输入、30 维普通边与反应端口的 GraphReaction。公式以当前全量数据实验为统计背景。

当前反向传播使用 **CE2**，最佳模型按 **每轮在线训练 CE2 最低**选择。Q_ref 和 NRMSE_delta 用于报告，不参与最佳模型选择。

## 1. 标准化增量标签

对一个输入窗口，模型输入真实的当前状态，预测下一个评价时刻的标准化增量。当前单步预测间隔为 10 秒，记为 $\Delta t$。

节点 $i$、物理字段 $f$ 的真实标签为：

$$
y_{i,f}
=
\frac{x_{i,f}(t+\Delta t)-x_{i,f}(t)-\mu_{\Delta f}}
{\sigma_{\Delta f}}.
$$

其中：

- $x_{i,f}(t)$：当前物理状态。
- $\mu_{\Delta f}$：已经保存的训练集增量均值，对应 `data.delta_mean`。
- $\sigma_{\Delta f}$：已经保存的训练集增量有效标准差，对应 `data.delta_std`。
- $\hat y_{i,f}$：模型输出的标准化预测增量，对应 `prediction[field]`。
- $y_{i,f}$：标准化真实增量，对应 `target[field]`。

这些统计量沿用数据集中的训练统计，不在验证或测试时重新拟合。它们是增量标签的统计量，与节点编码器当前状态输入的标准化统计量用途不同。

石墨和 AM 的原始状态存储的是浓度，保存的对应增量统计以嵌锂分数表示。数据加载时，代码将对应的增量均值和标准差同时乘以各样本的 $c_{\max}$，转换到浓度单位。因此，使用原浓度构造的标准化增量，与使用嵌锂分数构造的标准化增量一致；恢复物理状态时也直接得到浓度。

对应代码：[state_output.py](state_output.py) 的 `build_delta_targets`，以及 [data_loader.py](data_loader.py) 中的 `delta_mean`、`delta_std` 构造。

## 2. 单相、单字段 MSE

记 $\mathcal V_{p,f}$ 为 phase $p$ 中字段 $f$ 的有效节点集合。一个窗口内的单相字段损失为：

$$
L_{p,f}
=
\frac{1}{|\mathcal V_{p,f}|}
\sum_{i\in\mathcal V_{p,f}}
\left(\hat y_{i,f}-y_{i,f}\right)^2.
$$

也就是先计算每个有效节点的标准化增量平方误差，再在该相内取平均。这里的 MSE 是无量纲的标准化增量误差，不是直接用原物理单位计算的 MSE。

## 3. 七个物理字段及电解液区域合并

| 字段符号 | 日志简称 | 物理含义 | 对应 phase | 单窗口字段损失 |
| --- | --- | --- | --- | --- |
| $c_e$ | `ce` | 电解液浓度 | 2、5 | $L_{c_e}=0.5L_{2,c_e}+0.5L_{5,c_e}$ |
| $\phi_e$ | `phi_e` | 电解液电势 | 2、5 | $L_{\phi_e}=0.5L_{2,\phi_e}+0.5L_{5,\phi_e}$ |
| $\theta_{\mathrm{Gr}}$ | `cg` | 石墨嵌锂分数 | 1 | $L_{\theta_{\mathrm{Gr}}}=L_{1,\theta_{\mathrm{Gr}}}$ |
| $\theta_{\mathrm{AM}}$ | `cp` | AM 嵌锂分数 | 3 | $L_{\theta_{\mathrm{AM}}}=L_{3,\theta_{\mathrm{AM}}}$ |
| $\phi_{\mathrm{AM}}$ | `phi_p` | AM 固相电势 | 3 | $L_{\phi_{\mathrm{AM}}}=L_{3,\phi_{\mathrm{AM}}}$ |
| $\phi_{\mathrm{CBD}}$ | `phi_CBD` | CBD 固相电势 | 4 | $L_{\phi_{\mathrm{CBD}}}=L_{4,\phi_{\mathrm{CBD}}}$ |
| $V$ | `V` | 正集流体电势 | 7 | $L_V=L_{7,V}$ |

电解液浓度和电势都先分别计算 phase 2（电极区）与 phase 5（隔膜区）的节点平均 MSE，然后两个区域各占 0.5：

$$
L_f
=
\frac{\alpha_{\mathrm{elec}}L_{2,f}+\alpha_{\mathrm{sep}}L_{5,f}}
{\alpha_{\mathrm{elec}}+\alpha_{\mathrm{sep}}}
=0.5L_{2,f}+0.5L_{5,f},
\qquad f\in\{c_e,\phi_e\},
$$

其中 $\alpha_{\mathrm{elec}}=\alpha_{\mathrm{sep}}=0.5$。两个区域的节点数不同，也不会改变这里各占一半的权重。

phase 6 负集流体电势固定为 0，不单独预测或计算字段损失；石墨电势也不属于当前七字段输出。由于负集流体电势固定为 0，正集流体电势对应当前模型的电压 $V$。

对应代码：[loss.py](loss.py) 的 `compute_loss`，字段顺序见 [model_config.py](model_config.py) 的 `FIELD_NAMES`。

## 4. 物理电流重建与辅助损失

### 4.1 恢复未来物理状态

模型输出标准化增量后，按下面的公式恢复未来状态：

$$
\hat x_{i,f}(t+\Delta t)
=x_{i,f}(t)+\mu_{\Delta f}+\sigma_{\Delta f}\hat y_{i,f}.
$$

电流重建使用恢复后的物理电势，不使用隐藏表示或反应关联 rate 作为物理电流。

### 4.2 计算正集流体接触边的电导

对连接 AM/CBD 与正集流体的物理边 $e$：

$$
G_e=\sigma_{p(e)}\frac{A_e}{L_{e,\mathrm{solid}}}.
$$

其中 $A_e$ 为原始物理接触面积，$L_{e,\mathrm{solid}}$ 为三段 conduit 长度中固相节点一侧的端点长度。计算时不使用经过 log1p 和 z-score 变换的几何特征。

材料电导率为：

$$
\sigma_{\mathrm{AM}}=0.01\ \mathrm{S/m},
\qquad
\sigma_{\mathrm{CBD}}=760\ \mathrm{S/m}.
$$

存储的面积单位为 $\mu\mathrm m^2$，长度单位为 $\mu\mathrm m$，因此代码中的单位换算为：

$$
G_e
=\sigma_{p(e)}
\frac{A_e^{(\mu\mathrm m^2)}\times10^{-12}}
{L_{e,\mathrm{solid}}^{(\mu\mathrm m)}\times10^{-6}}.
$$

$G_e$ 的单位是 S，再乘电势差 V 后得到电流 A。

### 4.3 预测总电流及损失

将每条接触统一理解为“固相节点到正集流体节点”，预测总电流为：

$$
\hat I
=\sum_{e\in\mathcal E_{\mathrm{cc+}}}
G_e\left[
\hat\phi_{\mathrm{solid},s(e)}(t+\Delta t)
-\hat\phi_{\mathrm{cc+},c(e)}(t+\Delta t)
\right].
$$

这里 $s(e)$ 为固相端节点，$c(e)$ 为正集流体端节点。每条无向物理接触只计算一次，普通图双向消息不会使电流重复计数。

电流损失是相对误差的平方：

$$
\boxed{
L_I=\left(\frac{\hat I-I_{\mathrm{app}}}{|I_{\mathrm{app}}|}\right)^2
}.
$$

$I_{\mathrm{app}}$ 使用当前轨迹配置中的施加电流，对应 `data.applied_current_A`。

对应代码：[state_output.py](state_output.py) 的 `recover_physical_state`；[current_loss.py](current_loss.py) 的 `predict_collector_current`、`compute_current_loss`。

## 5. 优化目标 CE2

当前七字段权重为：

$$
(w_{c_e},w_{\phi_e},w_{\theta_{\mathrm{Gr}}},w_{\theta_{\mathrm{AM}}},
w_{\phi_{\mathrm{AM}}},w_{\phi_{\mathrm{CBD}}},w_V)
=(2,1,1,1,1,1,1).
$$

主字段损失对应 `main_loss`：

$$
L_{\mathrm{main}}
=\frac{\sum_f w_fL_f}{\sum_f w_f}
=\frac{
2L_{c_e}+L_{\phi_e}+L_{\theta_{\mathrm{Gr}}}+L_{\theta_{\mathrm{AM}}}
+L_{\phi_{\mathrm{AM}}}+L_{\phi_{\mathrm{CBD}}}+L_V
}{8}.
$$

加入电流辅助项，得到用于反向传播的 CE2：

$$
\boxed{
L_{\mathrm{CE2}}
=\frac{
2L_{c_e}+L_{\phi_e}+L_{\theta_{\mathrm{Gr}}}+L_{\theta_{\mathrm{AM}}}
+L_{\phi_{\mathrm{AM}}}+L_{\phi_{\mathrm{CBD}}}+L_V
}{8}+0.1L_I
}.
$$

分母 8 来自权重之和 $2+6\times1$。正集流体电势损失 $L_V$ 已经包含在七字段中，当前公式没有额外再叠加一个电压辅助损失项。

对应代码：[loss.py](loss.py) 中返回的 `loss`；权重见 [model_config.py](model_config.py) 的 `FIELD_WEIGHTS` 和 `CURRENT_LOSS_WEIGHT`。

## 6. 等权参考指标 Q_ref

Q_ref 使用同一批预测结果和同一组字段损失，只将七字段之间的权重改为等权：

$$
\boxed{
Q_{\mathrm{ref}}
=\frac{
L_{c_e}+L_{\phi_e}+L_{\theta_{\mathrm{Gr}}}+L_{\theta_{\mathrm{AM}}}
+L_{\phi_{\mathrm{AM}}}+L_{\phi_{\mathrm{CBD}}}+L_V
}{7}+0.1L_I
}.
$$

| 比较项 | CE2 | Q_ref |
| --- | --- | --- |
| 电解液浓度字段权重 | 2 | 1 |
| 其余六字段权重 | 各 1 | 各 1 |
| 七字段加权和的分母 | 8 | 7 |
| 电解液两个区域的权重 | 0.5 / 0.5 | 0.5 / 0.5 |
| 电流损失系数 | 0.1 | 0.1 |
| 用于反向传播 | 是 | 否 |
| 用于当前 best 选择 | 是，按每轮训练均值 | 否 |

对应代码：[loss.py](loss.py) 中返回的 `reference_loss`。

## 7. 每轮统计、梯度累积与最佳模型选择

### 7.1 按窗口计算，再按窗口平均

当前全量数据中每个窗口的权重为 1。一个 epoch 有 $W$ 个训练窗口，则：

$$
\mathrm{train\_CE2}
=\frac{1}{W}\sum_{w=1}^{W}L_{\mathrm{CE2}}^{(w)},
\qquad
\mathrm{train\_Q_{ref}}
=\frac{1}{W}\sum_{w=1}^{W}Q_{\mathrm{ref}}^{(w)}.
$$

各字段日志中的 MSE 也按窗口取算术平均：

$$
\overline L_f=\frac{1}{W}\sum_{w=1}^{W}L_f^{(w)}.
$$

评价固定 checkpoint 时，用该评价集的窗口数和窗口损失代入相同公式。这里是窗口平均，不是先对各轨迹平均，再对轨迹取平均。

在线训练指标来自训练过程中各窗口当时的模型预测；一轮内模型参数会更新。因此，在线训练 CE2 与训练结束后固定某个 checkpoint 重新评价整个训练集的 CE2，是两个不同的统计过程。

当前训练流程不逐轮验证。未做验证的 epoch 没有对应的验证 CE2/Q_ref；这些位置应留空。

### 7.2 梯度累积

每组最多累积 4 个窗口的梯度，然后进行一次梯度裁剪和优化器更新。当前全量数据的组目标为：

$$
L_{\mathrm{group}}
=\frac{1}{B}\sum_{w=1}^{B}L_{\mathrm{CE2}}^{(w)},
\qquad 1\leq B\leq4.
$$

最后一组不足 4 个窗口时，使用实际窗口数 $B$ 作分母。日志记录每个窗口原始的 CE2/Q_ref，不会因为梯度累积再次除以 $B$。

### 7.3 最佳模型选择

当前规则为：

$$
\boxed{
\mathrm{best\_epoch}
=\arg\min_e\mathrm{online\_train\_CE2}(e)
}.
$$

只有训练 CE2 严格降低时才更新 best；相等时保留先前 best。Q_ref 和 NRMSE_delta 不参与选择。best 对应的 Q_ref 表示 CE2 最佳 checkpoint 所对应的参考指标，不表示 Q_ref 自身的历史最小值。

对应代码：[training_step.py](training_step.py) 的 `train_group`、[training_metrics.py](training_metrics.py) 的 `MetricMean`、[train.py](train.py) 中的 `is_best` 判断。

## 8. 单步归一化增量 RMSE：NRMSE_delta

该指标分别对七个字段计算。记 $\mathcal V_f^{(w)}$ 为窗口 $w$ 中字段 $f$ 的全部有效节点集合，先累计平方误差和与节点出现次数：

$$
S_f=\sum_{w=1}^{W}\sum_{i\in\mathcal V_f^{(w)}}
\left(\hat y_{w,i,f}-y_{w,i,f}\right)^2,
\qquad
C_f=\sum_{w=1}^{W}|\mathcal V_f^{(w)}|.
$$

节点合并 MSE 为：

$$
\mathrm{standardized\_node\_pooled\_MSE}_f=\frac{S_f}{C_f}.
$$

最后只开一次平方：

$$
\boxed{
\mathrm{NRMSE}_{\Delta,f}
=\sqrt{\frac{S_f}{C_f}}
=\sqrt{\mathrm{standardized\_node\_pooled\_MSE}_f}
}.
$$

这里每个窗口中的有效节点都算一次。例如同一个节点出现在 100 个评价窗口中，就贡献 100 次误差和 100 个计数。

计算时不使用 CE2 的字段权重，也不使用电解液区域的 0.5/0.5 权重。电解液字段直接合并 phase 2 和 phase 5 的全部有效节点。

因此，不能一般性地将日志中的 `ce_MSE` 或 `phi_e_MSE` 直接开平方当作 NRMSE_delta：前者是区域等权后再按窗口平均，后者是所有有效节点合并统计。它们的分母和权重不同。

对应代码：[metric_reporting.py](metric_reporting.py) 的 `EpochMetrics.add`、`EpochMetrics.fields`、`save_nrmse`。输出字段包括 `standardized_node_pooled_MSE`、`NRMSE_delta` 和 `count`，保留原有 MSE。

## 9. 电流相对 RMSE 与指标名称对照

日志中的电流相对 RMSE 百分比，由窗口平均电流相对 MSE 开平方得到：

$$
\mathrm{current\_relative\_RMSE\_pct}
=100\sqrt{\frac{1}{W}\sum_{w=1}^{W}L_I^{(w)}}.
$$

若不同轨迹的施加电流不同，每个窗口先使用各自的 $I_{\mathrm{app}}$ 计算相对误差，再合并统计。

| 代码或日志字段 | 含义 |
| --- | --- |
| `loss` / `CE2` | 七字段 CE2 加权平均，加 0.1 倍电流损失 |
| `main_loss` | CE2 中的七字段加权平均，不含电流项 |
| `reference_loss` / `Q_ref` | 七字段等权平均，加 0.1 倍电流损失 |
| `field_losses` / `*_MSE` | 七字段各自的标准化增量 MSE；电解液保留区域等权 |
| `current_loss` / `current_relative_MSE` | 电流相对误差平方；汇总时按窗口平均 |
| `current_relative_RMSE_pct` | 窗口平均电流相对 MSE 开平方，再乘 100 |
| `standardized_node_pooled_MSE` | 全部有效节点的标准化增量平方误差合并均值 |
| `NRMSE_delta` | 节点合并标准化增量 MSE 的平方根 |

以上各项使用同一批预测结果。本文记录当前实现；历史实验的保存位置和指标补充情况可另见 [GraphReaction_指标说明.md](GraphReaction_指标说明.md)。
