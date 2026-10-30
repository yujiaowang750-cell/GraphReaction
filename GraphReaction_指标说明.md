# GraphReaction：CE2、Q_ref 与单步 NRMSE_delta

整理日期：2026-10-30。适用于当前 X13、普通/反应端口 b30、四轮共享参数更新的 GraphReaction。

## 1. 本次确认的训练策略

用户本次已确认：保持当前 **train_CE2 选 best**，训练优化目标仍为 CE2，不增加逐轮验证。已有实验按照其真实选择依据保存；验证缺失的轮次留空。

因此当前规则是 `best_epoch = argmin(online_train_CE2)`，不是验证 CE2 选 best。当前全量训练每窗权重为 1，新增 `train_CE2` 与原 `online_train_CE2` 相同。

Q_ref 和 NRMSE_delta 不参与反向传播、best 选择、早停或学习率调整。早停仍依据训练 CE2 连续 20 轮未严格降低；余弦退火设置不变。

## 2. 先计算各字段的标准化增量误差

对一个窗口、一个字段 f，真实标签为：

$$
y_{i,f}=\frac{x_{i,f}(t+10)-x_{i,f}(t)-\mu_{\Delta f}}{\sigma_{\Delta f}}.
$$

模型预测为同一空间内的增量 `prediction[f]`。误差为 `prediction[f] - target[f]`。归一化统计只读取已保存的训练统计；不重新拟合，不对验证集单独拟合。

七字段顺序保持：电解液浓度 ce、电解液电势 phi_e、石墨嵌锂分数 theta_Gr、AM 嵌锂分数 theta_AM、AM 电势 phi_AM、CBD 电势 phi_CBD、正极集流体电势 phi_cc+。

单相字段使用该字段有效节点的平均平方误差。电解液浓度和电势都使用：

$$
L_f=0.5\operatorname{MSE}_{phase2,f}+0.5\operatorname{MSE}_{phase5,f}.
$$

这一步保留现有训练损失的区域等权规则。

## 3. CE2 的准确公式

$$
\mathrm{CE2}=
\frac{2L_{ce}+L_{\phi_e}+L_{\theta_{Gr}}+L_{\theta_{AM}}+L_{\phi_{AM}}+L_{\phi_{CBD}}+L_{\phi_{cc+}}}{8}
+0.1L_I.
$$

分母为字段权重之和 `2+1+1+1+1+1+1=8`。

电流项为：

$$
L_I=\left(\frac{\hat I-I_{app}}{|I_{app}|}\right)^2.
$$

电流由恢复后的物理电势、原物理面积和固相端长度重建。没有把隐藏反应消息或 rate 当作物理电流。

## 4. Q_ref 的准确公式

$$
Q_{ref}=
\frac{L_{ce}+L_{\phi_e}+L_{\theta_{Gr}}+L_{\theta_{AM}}+L_{\phi_{AM}}+L_{\phi_{CBD}}+L_{\phi_{cc+}}}{7}
+0.1L_I.
$$

七字段使用算术平均，分母为 7；电流权重同样为 0.1。正极集流体电势已包含在七字段中，没有独立的额外电压惩罚项。

当前公式与本地原版说明 `C:\Users\13789\Downloads\GraphReaction模型代码详解_zh.md` 第 13 节的 Q_ref 一致。尚未提供包含“公式（54）”的论文完整总损失原文，因此不能据此断言与那篇论文的全部损失定义完全一致。只有当论文采用同样的七字段、标准化、区域平均及 0.1 电流项时，才完全一致。

实际损失函数仍为 [loss.py](loss.py) 的 `compute_loss()`，本次没有修改该函数或电流计算公式。新增统计直接复用其返回的 `loss`、`reference_loss`、`field_losses`。

## 5. 每个 epoch 怎样统计

对一轮内 W 个窗口，新增指标为：

$$
train\_CE2=\frac1W\sum_w CE2_w,\qquad
train\_Q_{ref}=\frac1W\sum_w Q_{ref,w}.
$$

每个训练窗口只前向一次，CE2 与 Q_ref 来自同一次预测。先记录未经梯度累积缩放的指标，再执行原有 `(CE2 * weight / len(pairs)).backward()`；指标不再除以梯度累积步数。

当前不逐轮验证，所以新历史中的 `val_CE2`、`val_Q_ref` 留空。最终固定 best 的验证结果计算相同的窗口算术平均，后处理将该真实结果放在 best 对应的那个 epoch。

原 `online_train_*`、`proxy_*` 和 MSE 列保持原意。旧 32 档实验中存在抽样窗口权重，不能把加权旧列直接改名为算术平均。本次新增 `train_*` 统计用窗口权重 1；原抽样优化权重与旧 best 依据均保留。全量模式没有这个差别，因为全部窗口权重原本就是 1。

## 6. best 对应的 Q_ref

每次训练 CE2 严格降低时，同时记录：

- `best_epoch`：CE2 最低轮。
- `best_Q_ref_at_CE2_best`：该轮对应的训练 Q_ref。
- `best_Q_ref_scope=online_train`：明确它的统计来源。

派生 `selection.json` 另外保存 `Q_ref_min_diagnostic` 与对应 epoch，仅用于诊断，不选择 checkpoint。

本次全量实验记录如下：

| 指标 | 值 |
|---|---:|
| best epoch，由 train_CE2 选出 | 138 |
| best train CE2 | 0.0005597272136835062 |
| best_Q_ref_at_CE2_best，训练在线统计 | 0.0005643919075431654 |
| 同一 best 的完整验证 CE2 | 0.0007373676242401624 |
| 同一 best 的完整验证 Q_ref | 0.0007863455564530354 |

这次训练 Q_ref 自身的最低轮也恰好是 138，但这不是指标定义保证的结果。

## 7. 单步 NRMSE_delta

按照本次明确给出的公式，对每个字段分别计算：

$$
\mathrm{NRMSE}_{\Delta,f}=
\sqrt{\frac{\sum_w\sum_{i\in V_{w,f}}(\hat y_{w,i,f}-y_{w,i,f})^2}
{\sum_w |V_{w,f}|}}
=\sqrt{\mathrm{standardized\_node\_pooled\_MSE}_f}.
$$

- 合并全部单步评价窗口和全部有效节点，最后只开一次平方。
- 不使用 CE2 的字段权重。
- 不使用电解液 phase 2/5 的 0.5/0.5 区域权重；这两相的全部有效节点直接参与统计。
- 当前完整验证每窗权重为 1，可以直接复用已保存的节点合并 MSE。
- 旧子集评价若使用 `population_weight`，其节点合并 MSE 已加权，不能冒充这里的不加权统计。新评价会在同一批预测上另外积累不加权 SSE/count。
- `sqrt(字段/窗口平均MSE)` 与这里的 `NRMSE_delta` 不应混为同一指标。

第 138 轮 best 在完整验证集上的结果：

| 字段 | NRMSE_delta |
|---|---:|
| 电解液浓度 | 0.02086230975 |
| 电解液电势 | 0.00797457686 |
| 石墨嵌锂分数 | 0.04988135515 |
| AM 嵌锂分数 | 0.04819140894 |
| AM 电势 | 0.00849680875 |
| CBD 电势 | 0.00939699185 |
| 正极集流体电势 | 0.00915977329 |

## 8. 历史数据能恢复到什么程度

| 内容 | 当前可恢复性 |
|---|---|
| 141 轮 train_CE2 / train_Q_ref | 完整，直接读取 history.csv |
| 七字段的 141 轮训练 MSE 曲线 | 完整，直接读取 history.csv |
| 验证 CE2 / Q_ref / 七字段 MSE | 只有第 138 轮 best 的真实评价点 |
| best 完整验证的七字段 NRMSE_delta | 完整，直接对节点合并 MSE 开平方 |
| 141 轮训练 NRMSE_delta | 旧记录缺少每字段逐节点 SSE/count，不能由字段平均 MSE 反推 |
| 其余 140 轮验证指标 | 未记录；只剩 best/last 权重，不能恢复完整曲线 |
| last 权重的验证结果 | 可以单独重新评价第 141 轮，但本次未执行 |

本次没有重新训练、没有重做真实数据评价，也没有补造缺失值。原 history.csv、overall_metrics.json、权重及原图均经哈希核对保持不变。

## 9. 修改文件与新增输出

| 文件 | 作用 |
|---|---|
| [metric_reporting.py](metric_reporting.py) | 窗口算术平均、节点 SSE/count、NRMSE_delta、新增评价文件 |
| [training_step.py](training_step.py) | 从同一预测记录新增统计，原 backward 和 optimizer.step 不变 |
| [train.py](train.py) | 新增历史列、best 对应 Q_ref、可恢复的 epoch 统计与曲线调用 |
| [training_config.py](training_config.py) | 明确新指标口径和新配置身份，monitor 仍为 online_train_CE2 |
| [evaluation.py](evaluation.py) | 保留原指标格式，另外输出不加权 NRMSE 与窗口算术平均 |
| [postprocess_metrics.py](postprocess_metrics.py) | 只读取已保存结果，生成派生表格、曲线和缺失情况说明 |
| [plot_metrics.py](plot_metrics.py) | 两套损失及七字段曲线；缺失验证轮次留空 |
| [checks/check_metric_reporting.py](checks/check_metric_reporting.py) | 手算、梯度无扰动、断点恢复、选择逻辑、历史文件保护检查 |
| [checks/check_train_only.py](checks/check_train_only.py) | 人工早停夹具适配新增统计接口，断言仍是 train_CE2 选择 |

本次派生目录为 [metrics_v2](training_runs/full_train_ce2_20261026_215337/metrics_v2/README.md)：

- `epoch_metrics.csv`：明确的 train/val CE2、Q_ref、七字段 MSE；不可恢复的位置留空。
- `selection.json`：真实 best 依据及对应 Q_ref。
- `full/nrmse_delta.csv`、`full/nrmse_delta.json`：完整验证七字段节点合并指标。
- `ce2_q_ref_curves.png/.svg`：两套损失曲线。
- `field_mse_curves.png/.svg`：七字段 MSE 曲线。
- `availability.json`：数据可恢复性、缺失轮次与原文件哈希。

以后新实验的 `history.csv` 会扩充 `train_*`、`val_*`、节点合并指标和 best 对应 Q_ref。最终评价额外保存 `nrmse_delta.csv/json` 与 `window_mean_metrics.json`；原评价文件的字段和数值定义保持不变。

指标状态加入 checkpoint 后，配置身份版本已更新。旧实验依然保留原配置和源码快照；不改写旧 manifest，也不绕过身份检查继续写入旧运行目录。

## 10. 命令与验证

从项目根目录后处理已有全量实验：

```powershell
.\.venv\Scripts\python.exe GraphReaction\postprocess_metrics.py --run-dir GraphReaction\training_runs\full_train_ce2_20261026_215337
```

运行人工图检查：

```powershell
.\.venv\Scripts\python.exe GraphReaction\checks\check_metric_reporting.py
```

检查结果：[checks/metric_reporting_result.json](checks/metric_reporting_result.json)。CE2/Q_ref 手算最大误差为 0，节点合并 NRMSE 手算最大误差为 0；启用新增统计与关闭统计的参数更新最大差异为 0；连续运行与断点恢复的参数最大差异为 0。人工检查覆盖不等节点数、最后不足 4 个窗口的累积组、两个电解液字段的区域权重、Q_ref 最低轮与 CE2 最低轮不同时的记录、20 轮早停与缺失验证数据。

这些检查只对临时人工小图执行少量优化器更新；没有启动正式训练或真实数据重评价。
