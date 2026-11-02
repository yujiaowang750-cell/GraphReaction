# seed2027：每轮完整验证、双best与验证早停

本次修改只在原 `GraphReaction` 目录；`GraphReaction_incremental/GraphReaction` 副本没有同步修改。旧实验记录与权重不改写。

## 当前设置

| 设置 | 值 |
| --- | --- |
| 训练seed | 2027 |
| 最大epoch | 200 |
| 余弦周期 | 200，初始LR=0.001、最低LR=0.00001 |
| 梯度累积/裁剪 | 4窗口 / 范数1，保持原逻辑 |
| 优化目标 | 原CE2，不改变损失公式 |
| 每轮验证 | 当轮权重固定，完整14条验证轨迹、4,917窗口 |
| 早停 | 验证CE2连续20轮没有严格刷新历史最低值 |
| 改善阈值 | 不额外设置；min_delta=0，相等不算改善 |
| 测试集 | 不读取动态数据、不参与选择 |

训练集best的依据仍是本轮在线训练CE2，不是用轮末权重重新评价整个训练集的误差。

## 一轮的调用顺序

1. `train.py` 调用 `training_step.train_group` 完成本轮全部训练窗口。
2. 当前模型不再更新，调用 `epoch_validation.validate_full`。`model.eval()`与`torch.no_grad()`用于验证，每个窗口复用原 `compute_loss`。
3. 验证窗口CE2作算术平均得到val_CE2；同一次预测也记录Q_ref、七字段MSE与节点合并NRMSE_delta。
4. 独立比较训练CE2与验证CE2，决定是否更新各自的best。
5. 只有验证CE2刷新最低值，早停计数才清零；否则加1。训练CE2下降不会清零。
6. 余弦scheduler在epoch结束时step一次，不接收验证loss。
7. 将本轮模型、优化器、scheduler、RNG、双best信息和早停计数保存到last，独立归档为 `epoch_XXXX.pt`，再将符合条件的last原子复制为对应best。每轮刷新曲线。
8. 达到200轮或验证计数达到20则结束，加载best_val输出详细验证结果。

验证直接使用内存中固定的轮末权重；随后保存的checkpoint就是这份权重，不需要为了固定评价而反复从磁盘加载。验证期间没有backward或optimizer.step，结束后恢复原训练模式与随机状态。

下一轮从上一轮结束时的模型继续训练，不能先加载best_val再训练。两份best只是同一训练过程中挑出的两个保存点；如果最佳轮次相同，两份权重可以相同。

## 早停示例

若第40轮取得验证最低值，第41～60轮都未低于该值，第60轮结束后早停。中途只有训练loss改善不影响这个计数。若第59轮验证CE2哪怕只下降很小，也会更新best_val并将计数清零。

## 文件对应关系

- `learning_rate.py`：`NUM_EPOCHS=200`同时控制默认最大预算与余弦周期。
- `training_config.py`：全量模式设置 `validate_each_epoch=True`，记录双best及验证早停策略。原32档入口没有在本次改成逐轮完整验证。
- `epoch_validation.py`：`validate_full`负责固定验证，`update_validation_best`负责严格比较与计数。
- `train.py`：在每轮训练后调用验证，保存两个best，写日志，按验证CE2早停。
- `training_checkpoint.py`：保存完整last；`archive_epoch_checkpoint`归档各轮完整状态，已有不同内容的轮次文件拒绝覆盖。
- `group_metrics.py`：从已有七字段MSE整理C、P、V、I及其CE2贡献，不另算预测或改变反向目标。
- `postprocess_metrics.py`：保持参与选择的当轮验证记录；最终详细复评独立保留，避免CUDA轻微重复差异覆盖best选择依据。
- `plot_training.py`、`plot_metrics.py`：绘制训练/验证曲线，指标曲线分别标记训练best和验证best。
- `checks/check_dual_best.py`：隔离检查，不是正式训练入口。

## 输出

```text
checkpoints/
  epoch_0001.pt   第1轮完整训练状态
  epoch_0002.pt   第2轮完整训练状态（后续每轮独立保留）
  best_train.pt    在线训练CE2最低的轮末模型
  best_val.pt      完整验证CE2最低的模型
  last.pt          最近已提交的训练状态，供断点恢复
history.csv/json   每轮train/val CE2、Q_ref、7字段、C/P/V/I、两个best轮次、验证早停计数
training_summary.json
metrics_v2/        CE2/Q_ref曲线、七字段曲线、C/P/V/I曲线；结束后另存epoch_metrics.csv和选择信息
evaluation/full/  最终best_val的详细验证结果
```

`best_epoch`为原训练best字段，`best_val_epoch`为验证best。summary中的 `final_evaluation_checkpoint=best_val` 明确最终评价来源，不混淆两者。

新协议身份已改变，不能用新源码直接resume旧协议实验。断点恢复需使用本次新实验自己的目录和相同代码。验证中断时，最后训练更新已存在last中；恢复后补做该轮验证，不重复已经提交的训练更新。

每份epoch归档包含 `model`、AdamW的 `optimizer`、`scheduler`、Python/NumPy/Torch/CUDA的 `rng`、`state`（epoch、窗口游标、更新数、历史指标、两个best与早停计数）、初始化和配置/数据/源码身份；不是仅保存模型权重。轮末归档在scheduler.step之后保存，优化器中的学习率供下一轮使用。

恢复最近进度使用 `--resume 新实验目录`，入口读取该目录的 `checkpoints/last.pt`，不能用best替换last继续训练。epoch归档支持后续固定评价或显式恢复状态；当前命令行不支持自动回退到任意历史轮，不能直接覆盖last来回退，以免与CSV历史冲突。

## 分组指标

所有字段先按原规则统计窗口MSE，ce和phi_e仍先在phase2/phase5之间做0.5/0.5平均。

| 名称（train_ / val_ 前缀） | 定义 | 对CE2的贡献 |
| --- | --- | --- |
| C_MSE | (2×ce_MSE + cg_MSE + cp_MSE)/4 | C_CE2_contribution = C/2 |
| P_MSE | (phi_e_MSE + phi_p_MSE + phi_CBD_MSE)/3 | P_CE2_contribution = 3P/8 |
| V_MSE | 正集流体电势的标准化增量MSE | V_CE2_contribution = V/8 |
| I_relative_MSE | 电流相对误差平方的窗口均值 | I_CE2_contribution = 0.1I |

`main_loss`为C/P/V三项贡献之和，`total_loss`直接复用原CE2。P不包含V。七字段是ce、phi_e、cg、cp、phi_p、phi_CBD、V。训练曲线为在线训练指标，验证曲线为每轮固定模型评价；不把在线训练指标说成固定checkpoint训练误差。

曲线每轮输出 `metrics_v2/ce2_q_ref_curves.png/svg`、`field_mse_curves.png/svg`、`group_loss_curves.png/svg`，分别标记两个best。每次只更新本次新run目录内的曲线，不改旧实验。

## 新实验启动

在 `NewProgramByMe` 根目录：

```powershell
& ".\.venv\Scripts\python.exe" -u `
  ".\GraphReaction\train.py" `
  --dataset full `
  --full-root "D:\Github Project\NewProgramByMe\Datasets" `
  --device cuda `
  --epochs 200 `
  --run-dir ".\GraphReaction\training_runs\full_seed2027_e200_dualbest_r001"
```

新目录必须尚不存在。正式训练未自动启动。

中断恢复（使用上面同一个新实验目录，配置和代码需保持一致）：

```powershell
& ".\.venv\Scripts\python.exe" -u ".\GraphReaction\train.py" `
  --resume ".\GraphReaction\training_runs\full_seed2027_e200_dualbest_r001"
```

必要检查命令：

```powershell
.\.venv\Scripts\python.exe -u GraphReaction\checks\check_dual_best.py --dataset-root Datasets --output GraphReaction\training_runs\dual_best_checks_seed2027
```

短检查只用少量真实窗口，显式标记smoke，不是对完整验证集效果的估计；不会读取测试集动态数据。

## 前一轮检查记录（保留历史）

- CPU隔离检查通过：相等/上升累计20轮、任意严格下降清零、验证不改参数/RNG、缺失best副本恢复、训练中途和epoch边界恢复。
- 连续运行与恢复运行的模型最大绝对差1.11e-6、优化器最大差1.36e-8，均在原梯度容差atol2e-5/rtol2e-4内；scheduler和RNG一致。原四份历史checkpoint与数据身份文件SHA256保持不变。
- GPU只运行2个短epoch，每轮3个训练、3个验证窗口。实际得到best_train=第2轮、best_val=第1轮、早停计数1；最终详细评价读取第1轮验证best。三份checkpoint元数据和曲线标记均已读回检查。
- CPU报告：`training_runs/dual_best_checks_seed2027/checks.json`。
- GPU报告：`training_runs/dual_best_gpu_smoke_seed2027/dual_best_readback_checks.json`。
- 正式200轮训练没有启动，完整4,917窗口逐轮验证没有在短检查中运行。

## 本轮追加检查

本轮检查目录是 `training_runs/epoch_outputs_checks_20261031`，与以上历史报告分开。核查时发现当前train.py仍为旧的不逐轮验证流程，已经按已确认需求接回验证/双best/验证早停。`configs/learning_rate.json`中的旧141轮说明同步为200，实际调度仍由learning_rate.py控制。

- `cpu/checks.json`：验证不更新参数和随机状态、20轮早停、完整轮归档、缺失副本恢复、轮中和轮末恢复、已有目录与归档防覆盖、七字段和分组指标。
- `model_metrics/checks.json`：人工图输入/拓扑、处理器和损失、末轮Z跳过、梯度、4+3累积尾组，以及训练/验证各一个真实窗口的所有模块维度。
- `learning_rate.json`：200轮余弦公式和配置一致性。
- `gpu/`：两轮短流程，每轮7个训练窗口与7个验证窗口；不代表完整训练或模型精度。

当前入口仍需显式传 `--dataset full`；默认proxy32入口不在本轮改变为逐轮完整验证。历史141轮续训入口及独立A/B审计配置不用于本次新seed实验；其源码身份检查不应绕过。主目录代码与旧 `GraphReaction_incremental` 副本没有自动同步。
