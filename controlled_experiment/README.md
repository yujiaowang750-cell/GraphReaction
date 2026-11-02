# 固定分母8的独立A/B实验

这是新分支入口，不替换旧 `train.py`。原训练继续暂停于155，正式best仍为154。
本机只完成审计与短检查；本文件的正式命令尚未执行。

## 实验约定

`L_beta = 原CE2 + (beta-1)*(L_phi_e+L_phi_p+L_phi_CBD)/8`。

- A：beta=1；B：候选beta=1.25，尚未验证收益。
- 浓度、电压、电流的绝对系数不变。与旧计划分母 `5+3*beta` 不同，后者会同时缩放非电势物理场，并改变它们相对电流项的权重。旧计划保留为历史，不覆盖。
- 共同起点：全量实验138轮best，SHA256 `bddb304aaa19de2b8dc0c00acc438cc91b45741049c584b3dfa98f08d1b9d689`。
- 起点具备完整模型、AdamW一二阶矩/step、CPU/CUDA/NumPy/Python随机状态。初始化时独立复制优化器状态，重置学习率为固定1e-5；原余弦调度器不再推进。
- 这是从138出发的新分支，不是原139～148轮轨迹的原样重放。诊断不会污染分支初态，每次均重新加载父断点。
- 10个新增epoch，使用原139～148的确定性窗口排列；每轮65条训练轨迹22,927窗口，5,732次更新，累积4、尾组3。两组更新预算相同。
- 模型/数据/归一化/区域等权规则/精度/裁剪不变。每轮固定评价完整训练集及14条验证轨迹，不读取测试动态数据。
- 正式best仍按**在线训练的原CE2**严格降低选择；不依据L_beta、Q_ref或验证指标。新分支早停计数从0开始，10轮预算内不会达到原20轮阈值，因此按共同10轮终点比较。
- `online.L_beta`与`online.CE2`分开，固定评价使用原CE2/Q_ref/C/P/V/I。V是第7字段；P组仅三个电势字段。所有损失按窗口平均，节点合并NRMSE另外保存。
- 主要对照为共同第10轮终点，同时报告完整曲线和最后3轮均值、标准差、范围；不事后挑有利轮次。

## 阅读顺序

1. `configs/A_beta1.json`、`B_beta1p25.json`：唯一区别为beta及标签。
2. `common.py`：严格检查父断点/数据身份，恢复独立初态，定义目标和C/P/V/I。
3. `run.py:train_group`：4个窗口累计反向，组末裁剪与更新；`evaluate_epoch`不更新权重并恢复RNG。
4. `run.py:run`：每轮保存独立checkpoint，评价后记录结果；继续时不重复已提交epoch。
5. `compare.py`：检查两组初态、顺序和预算，输出第10轮与末3轮对照和曲线。
6. `check_local.py`：只用于隔离验证，不是正式训练入口。

## 云端准备与命令

本机没有连接服务器，也没有上传。将包解压到独立代码目录，先按增量清单检查文件，不对服务器旧目录使用删除同步。
将已有Datasets和指定138断点放在包外；包不包含任何数据或权重。参照 `environment_observed.txt` 配置服务器环境。
`runtime_manifest.json` 校验包内计算代码；数据四个身份文件与父断点必须匹配；旧版本/子集权重将被拒绝。

在服务器的 `GraphReaction` 目录中：

```bash
cp controlled_experiment/cloud_paths.example.sh controlled_experiment/cloud_paths.local.sh
# 编辑local文件中的四个绝对路径，再加载：
source controlled_experiment/cloud_paths.local.sh

# 先check：零更新，每个集合仅2窗口。
bash controlled_experiment/cloud_run.sh check A
bash controlled_experiment/cloud_run.sh check B

# 再smoke：2个短epoch，每轮仅7训练窗口及2+2评价窗口。
bash controlled_experiment/cloud_run.sh smoke A
bash controlled_experiment/cloud_run.sh smoke B

# 人工核对两个initial_identity.json一致，短检查均成功后，才运行完整实验。
bash controlled_experiment/cloud_run.sh train A
bash controlled_experiment/cloud_run.sh train B

"$PYTHON_BIN" controlled_experiment/compare.py \
  --a "$OUTPUT_ROOT/train_A" --b "$OUTPUT_ROOT/train_B" \
  --output "$OUTPUT_ROOT/comparison_A_B"
```

命令串行运行两组，避免共享GPU导致速度和内存干扰。输出目录已存在会拒绝启动。
需要继续一个已存在的新分支时：`bash controlled_experiment/cloud_run.sh train A --resume`。
恢复边界为**完整epoch末尾**；epoch中断会从上一个已提交epoch重新执行该轮，不宣称中途逐更新精确续训。
如首轮尚未提交checkpoint，另选新的空输出目录重启。评价中断只补已保存模型的评价，不重放该轮训练。
禁止对旧正式实验使用本入口的resume。

## 输出

- `initial_identity.json`：模型、AdamW、RNG和第一轮顺序哈希；A/B应相同。
- `checkpoints/epoch001.pt`～`epoch010.pt`：每轮独立模型+优化器+RNG。另有本分支best/last，绝不写原实验。
- `epochs/epochNNN.json`：online原CE2、L_beta、Q_ref、字段、C/P/V/I；fixed_train/fixed_validation独立记录；原CE2选择结果。
- `updates/epochNNN.csv`：每组窗口数、裁剪前范数、是否触发、学习率。
- `evaluation/epochNNN/{train,full}`：原评价文件、节点合并MSE/NRMSE、电压电流、逐轨迹/区域/时间箱结果。`full`明确指验证集。
- `summary.json`：共同预算终点、末3轮波动及本分支best。
- 比较目录：`comparison.json`、`curves.csv`、`fixed_curves.png`、`online_objectives.png`。

## 保留与兼容性

保留服务器的 `cloud_paths.local.sh`、虚拟环境、数据、旧checkpoint、旧训练目录以及用户自定义调度脚本；均不覆盖、不删除。
上传包附带当前原运行源码作为身份依赖。若服务器已有不同版本，先放入新的代码目录或逐文件手动合并，不能直接覆盖仍用于旧实验的目录。
旧权重只作为明确指定的138父断点；新分支checkpoint有独立schema。旧优化器并非丢弃重置：保留矩与step，只改已约定的学习率。
跨CPU/GPU或不同CUDA硬件不承诺逐位一致。原CUDA协议存在小幅重复波动；本轮没有为消除该波动擅自改成新数值协议。
