# 固定第138轮best的完整测试集评价

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -u GraphReaction\evaluation_tools\evaluate_test.py --model-run GraphReaction\training_runs\full_train_ce2_20261026_215337 --expected-epoch 138 --device cuda
```

每次新建独立 `training_runs/test_best138_时间戳` 目录。模型权重来自原全量实验的 `checkpoints/best.pt`，启动时检查它确实是第138轮，以及模型结构、数据文件和训练统计的身份一致。

测试范围是 `Datasets/trajectory_split.json` 中14条测试轨迹、4,941个单步窗口。每窗输入真实t时刻状态，预测t+1；不做自回归滚动预测，不创建优化器，不依据测试结果选择checkpoint。

## 代码顺序

1. `test_windows.py`：只构造测试窗口，校验轨迹互斥、窗口不重不漏，读取原训练统计。
2. `evaluate_test.py`：校验指定checkpoint，使用原WindowReader和模型，调用测试统计函数。
3. `test_metrics.py`：复用原 `compute_loss`、`metric_values`、`physical_error_arrays` 和误差累加器。

这些文件位于独立子目录，不改变当前暂停实验的训练源码身份。

## 指标口径

- CE2 = `(2*ce_MSE + 其余六字段MSE之和)/8 + 0.1*current_relative_MSE`。
- Q_ref = `七字段MSE算术平均 + 0.1*current_relative_MSE`。
- ce和phi_e的字段MSE仍为电极电解液、隔膜电解液的0.5/0.5区域平均。
- 先逐窗计算损失，再对4,941个窗口作算术平均；各窗口权重都是1。
- NRMSE_delta按字段将全部有效节点的标准化增量平方误差合并，除节点总数后开方；不使用字段权重或区域等权。
- cg/cp的标准化增量对应石墨/AM嵌锂分数；恢复后的浓度误差单位为mol/m³。
- 电势、正极集流体电压的物理误差单位为mV，电流为A。
- 七字段沿用GraphReaction定义：ce、phi_e、cg、cp、phi_p、phi_CBD、V。phase6电势固定0，石墨电势不是独立输出字段。

## 输出

- `overall_metrics.json`：CE2、Q_ref、各字段损失、物理误差、总体电流误差。
- `field_metrics.csv`：七字段及有效相位的误差统计。
- `nrmse_delta.csv/json`：各字段节点合并MSE与NRMSE_delta。
- `trajectory_metrics.csv/json`：14条轨迹分别的损失。
- `voltage_current.csv`：4,941个窗口的电压、电流预测、真值/施加值及误差。
- `evaluation_config.json`：指定权重、数据与计算代码身份。
- `evaluation_audit.json`：测试轨迹覆盖情况、原权重未变、优化器更新为0等检查。

不覆盖已有训练日志、验证结果、best/last权重，也不恢复暂停的训练。

## 第154轮best的验证集复查

```powershell
.\.venv\Scripts\python.exe -u GraphReaction\evaluation_tools\evaluate_validation.py --model-run GraphReaction\training_runs\full_continue_193_fixed_lr_20261030_101435 --expected-epoch 154 --device cuda
```

此入口直接复用原 `evaluate_scope`，只评价原14条验证轨迹的4,917个窗口。
结果写入独立 `training_runs/validation_best154_时间戳`，可与第138轮已有的验证结果比较，不重新选择best、不读取测试集、不恢复训练。
