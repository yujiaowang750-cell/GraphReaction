# GraphReaction

当前正式训练入口为仓库根目录的 `train.py`。从 GitHub 克隆后直接运行 `python train.py`，不需要再加一层 `GraphReaction/`。

## 当前实验

| 设置 | 当前值 |
| --- | --- |
| seed | 2027，见 `training_config.py` |
| 输入/隐藏维数 | 节点13维，隐藏64维，普通边和反应端口均30维 |
| 参数量 | 186,695 |
| 更新 | 4轮节点更新，默认只生成Z0～Z3 |
| 数据 | 完整65条训练轨迹、14条验证轨迹；划分与统计沿用原数据文件 |
| 窗口 | 22,927个训练窗口、4,917个验证窗口；10秒单步预测 |
| 优化目标 | 原CE2，公式见 `GraphReaction_损失计算公式.md` |
| 预算/学习率 | 最多200轮；余弦T_max=200，0.001下降至0.00001 |
| best | 分别保存在线训练CE2最优与固定验证CE2最优 |
| 验证与早停 | 每轮固定模型评价完整验证集；验证CE2连续20轮未严格降低则早停 |
| 恢复 | 每轮完整checkpoint；last另在每次完整优化器更新后保存 |

测试集不参与上述训练、验证或模型选择。训练曲线是在线指标，验证曲线是固定轮末模型的评价指标。

## Linux / SSH服务器下载

在服务器选择一个新的代码目录，执行：

```bash
git clone https://github.com/yujiaowang750-cell/GraphReaction.git
cd GraphReaction
git log -1 --oneline
```

仓库不包含Datasets、训练权重、完整实验输出或虚拟环境。数据需要已单独放在服务器上，并保持原目录结构；不要重新生成划分或归一化统计。

已有克隆且没有本地改动时，可在其目录中执行 `git pull --ff-only`。正在运行实验的目录应保留原源码；新版本另建目录，避免断点恢复的源码身份不一致。

## Python环境

本机已验证环境：Python 3.13.9、PyTorch 2.14.1+cu126、NumPy 2.5.4、Matplotlib 3.11.2、PyG 2.8.1。服务器Linux环境尚未实际运行，先按下面的短检查确认。

如果服务器已有可用的CUDA PyTorch环境，激活该环境后安装其余依赖。否则先创建独立环境，再按[PyTorch官方安装页](https://pytorch.org/get-started/locally/)选择Linux、Pip及与服务器驱动适配的CUDA版本安装PyTorch。这里不根据尚未检查的服务器驱动强行指定CUDA包。

```bash
# 新环境示例；使用你在服务器安装的Python解释器。
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 先执行官方页面对应的CUDA PyTorch安装命令，再执行：
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

`requirements.txt`固定本机已验证的三个直接依赖，不包含PyTorch，避免覆盖服务器已配置的CUDA版本。安装失败时先核对Python/包源及版本，不将失败的安装当作已完成。

本模型使用PyG的Data容器和PyTorch原生计算，不依赖torch-scatter/torch-sparse等额外扩展；PyG的基本安装方式见[官方说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

## 数据路径

将下面的路径改成服务器上实际Datasets的绝对路径：

```bash
DATASET_ROOT="/absolute/path/to/Datasets"
```

该目录至少包含：

```text
trajectory_split.json
dataset_manifest.json
window_index.npz
normalization_stats.json
各样本的目录（由dataset_manifest.json的processed_relative_dir指定）/
  sample_manifest.json
  static_graph.npz
  node_mapping.npz
  dynamic/*_state.npy
```

用 `--dataset full --full-root "$DATASET_ROOT"` 选择完整数据。不要省略 `--dataset full`，因为旧默认入口仍为proxy32。

## 先做短检查

```bash
python -u train.py \
  --dataset full --full-root "$DATASET_ROOT" \
  --device cuda --epochs 2 --smoke-windows 7 \
  --run-dir training_runs/server_smoke_seed2027_001
```

这只运行2个短epoch，每轮7个训练/7个验证窗口，检查GPU前向、损失、反向、4+3累积尾组、checkpoint和曲线输出。它不代表正式模型效果；正式训练不能带 `--smoke-windows`。

## 开始新的正式实验

```bash
python -u train.py \
  --dataset full --full-root "$DATASET_ROOT" \
  --device cuda --epochs 200 \
  --run-dir training_runs/full_seed2027_e200_server_001
```

新实验目录必须尚不存在。代码从SEED=2027重新初始化，不会加载本机正在训练的模型。SSH断开后需要继续运行时，可在服务器已有的tmux会话中启动此命令。

中断后恢复同一次服务器实验：

```bash
python -u train.py --resume training_runs/full_seed2027_e200_server_001
```

恢复读取该目录的last.pt和config.json；同一实验的数据位置、源码和运行配置需要保持一致。仅克隆代码不能恢复本机某一轮的进度，本机训练的checkpoint没有上传到Git。

## 输出

```text
training_runs/full_seed2027_e200_server_001/
  config.json                 实际配置、源码/数据身份
  source/                     本次源码快照
  checkpoints/epoch_0001.pt    每轮完整状态，后续轮次独立保存
  checkpoints/best_train.pt   在线训练CE2最低的轮末模型
  checkpoints/best_val.pt     完整验证CE2最低的模型
  checkpoints/last.pt         最近完成更新的训练状态
  history.csv / history.json  train/val CE2、Q_ref、七字段、C/P/V/I与早停记录
  update_metrics.csv          优化器更新、学习率及梯度裁剪记录
  training_summary.json
  loss_curve.png
  metrics_v2/                 CE2/Q_ref、七字段、C/P/V/I的训练/验证曲线
  evaluation/full/            最终best_val的详细验证结果
```

七字段为ce、phi_e、cg、cp、phi_p、phi_CBD、V。C=(2ce+cg+cp)/4，P=(phi_e+phi_p+phi_CBD)/3，V单列，I为电流相对MSE。CE2=C/2+3P/8+V/8+0.1I；分组与字段MSE保留原区域等权口径。

## 其他目录

- `checks/`：本机检查代码与历史验收记录。部分旧对拍检查依赖未入库的参考项目，不作为服务器首次启动的必需步骤；服务器先用上面的smoke命令。
- `audit_tools/`、`evaluation_tools/`：已有审计与离线评价工具；其中历史默认路径需按实际环境传参或调整。测试集评价工具不会由train.py调用。
- `controlled_experiment/`：之前准备的、从epoch138出发的独立A/B方案，保留当时的配置身份。它不是当前seed2027实验入口，旧manifest未随新版重写，不能绕过身份校验直接用于当前训练。
- 更详细的保存/恢复说明见 [逐轮验证与双best](GraphReaction_逐轮验证与双best.md)。
