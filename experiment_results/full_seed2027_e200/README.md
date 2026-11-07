# GraphReaction：seed2027 全量训练与验证实验

本归档保存200轮正式训练及完整验证结果，不包含任何测试集评价、冒烟测试、恢复测试或代码检查运行结果。

## 实验设置与最佳模型

- 随机种子：2027；模型参数量：186,695。
- 完成200轮，因达到轮数上限结束，没有触发早停。
- 训练集：65条轨迹，每轮22,927个窗口。
- 验证集：14条轨迹，每轮4,917个窗口。
- 优化目标：CE2；训练best按在线训练CE2选择，验证best按固定权重验证CE2选择。
- 早停：验证CE2连续20轮未刷新最低值；结束时计数为5。
- 学习率：200轮余弦退火，从0.001降至0.00001。

| 模型 | epoch | 对应CE2 |
|---|---:|---:|
| 训练best | 189 | 0.0004328377915113856（在线训练） |
| 验证best | 195 | 0.0006170460928160809（轮末验证） |
| 验证best最终完整评价 | 195 | 0.0006170461633385933 |

训练指标是在每轮不断更新参数时累计的在线统计，不是固定该轮checkpoint重新评价整个训练集的结果。第195轮的最终验证Q_ref为0.0006593200692276412；该次独立评价与轮末验证存在微小数值差异，两份原始记录均保留。

## 查看结果

- [原训练配置](config.json)、[训练摘要](training_summary.json)、[200轮历史](history.csv)。
- [最终验证总体指标](validation_best195/overall_metrics.json)。
- [验证七字段NRMSE_delta](validation_best195/nrmse_delta.csv)、[字段误差](validation_best195/field_metrics.csv)。
- [逐窗口电压与电流](validation_best195/voltage_current.csv)。
- [CE2与Q_ref曲线](metrics_v2/ce2_q_ref_curves.png)、[七字段MSE曲线](metrics_v2/field_mse_curves.png)。
- [C/P/V/I分组曲线](metrics_v2/group_loss_curves.png)。
- [按参考样式绘制的七字段验证MSE](requested_plots/validation_seven_fields_mse_20261107_091829/validation_seven_fields_mse.png)。
- [原指标定义](metrics/metric_definitions.md)、[逐文件SHA256清单](manifest.json)。

NRMSE_delta按全部有效节点的标准化增量平方误差合并后开方，不使用字段权重或电解液区域等权规则。字段损失MSE沿用原损失口径，其中电解液浓度和电势使用电极区/隔膜区0.5/0.5平均。

## 源码与权重

[source/](source/)是实际训练时的源码快照，逐文件原始哈希保留在config.json。本次归档前的仓库提交为`f45cc9316173cc1bf158d63927f613ed46d7d298`；归档提交另包含测试入口选择权重文件的源码改动，不包含测试结果。

完整权重及更新日志在[对应Release](https://github.com/yujiaowang750-cell/GraphReaction/releases/tag/exp-seed2027-full-e200-20261107)的`exp-seed2027-full-e200-20261107-checkpoints.zip`附件中，ZIP内包含：

- checkpoints/epoch_0001.pt 到 epoch_0200.pt；
- checkpoints/best_train.pt（189轮）、best_val.pt（195轮）、last.pt（200轮）；
- update_metrics.csv。

这些均为完整训练断点，包含模型、优化器、调度器和随机状态。达到200轮后的延长训练需要适配续训入口；普通--resume不会自动增加轮数。

附件SHA256：`3d5b64068d4871a602f7ddb05f004f8cad549403bb03a834e8d08936bf37096e`。

下载后可使用PowerShell核验：

```powershell
Get-FileHash -Algorithm SHA256 ./exp-seed2027-full-e200-20261107-checkpoints.zip
```

将输出与manifest.json的release_asset.sha256比较。manifest.json还记录每个压缩包成员及每个复制文件的SHA256。原实验目录保持不变。
