# 从141轮续训至最多193轮

本次续训保留当前模型与新增指标记录。原全量实验目录保持原样，续训使用新的独立目录。

## 保持和继承的内容

- 数据仍为 `Datasets/trajectory_split.json` 指定的65条完整训练轨迹，共22,927个窗口。
- 从原 `checkpoints/last.pt` 的第141轮继续，下一轮是142；不从第138轮best倒退训练。
- 继承模型参数、AdamW动量、随机数状态、累计更新次数，以及原始训练统计。
- 保留历史best：第138轮，训练CE2为 `0.0005597272136835062`。
- best仍按每轮训练CE2严格降低选择，反向目标仍为CE2。
- 不进行逐轮验证；停止后用最终best评价14条完整验证轨迹，测试集不动。
- 历史早停计数为3，续训不清零；连续20轮未改善停止，193轮是上限。

## 学习率

第1～141轮保留原余弦记录；第142轮起固定为 `1e-5`。
代码使用 `ConstantLR(factor=1, total_iters=1)`，各轮 `step()` 不改变学习率。
原余弦调度器的状态记录在 `continuation.json` 中，不继续推进原余弦。
普通从零开始的训练仍按原141轮余弦配置执行。

## 启动与恢复

在项目根目录运行（第一次建立续训实验）：

```powershell
.\.venv\Scripts\python.exe -u GraphReaction\continue_training.py --parent-run "GraphReaction\training_runs\full_train_ce2_20261026_215337" --epochs 193 --device cuda
```

如续训本身中断，使用已生成的新目录恢复，不能再次从旧目录建立另一份续训：

```powershell
.\.venv\Scripts\python.exe -u GraphReaction\train.py --resume "GraphReaction\training_runs\<本次续训目录>"
```

## 文件及身份

- `continue_training.py`：检查旧模型、数据、源码快照、断点和训练协议；继承旧状态与日志。
- `train.py`：执行原训练循环，按显式续训配置使用固定学习率。
- `continuation.json`：原目录、原config/last/best哈希、继承的best和早停计数。
- 新 `config.json` 和 `source/`：记录此次代码与固定学习率协议；不修改原实验manifest。
- 新 `last.pt`：保存可恢复的完整续训状态。
- 新目录最初的 `best.pt` 与原best文件完全相同；若有新best才替换。最终读取继承best时显式核对来源哈希，不使用 `strict=False`。
- 新目录继承1～141轮日志，并新增后续轮次。旧轮次缺少的节点合并NRMSE仍留空，不补造数据。
- 普通 `--resume` 的源码、数据、配置身份检查仍然严格。跨版本延长只通过明确的续训入口进行。

## 验证

```powershell
.\.venv\Scripts\python.exe -u GraphReaction\checks\check_continuation.py
```

检查使用临时目录和人工图，包括：固定学习率、完整及不足4窗的梯度累积、中途恢复对拍、继承best、新best替换、20轮早停、原文件不变、拒绝错误数据和不兼容的普通恢复。
结果位于 `checks/continuation_result.json`。没有为这些检查启动正式训练。
