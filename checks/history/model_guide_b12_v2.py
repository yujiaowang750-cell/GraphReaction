r"""
GraphReaction v2：X13 + reaction_b12 + 默认不计算末轮 Z4

更新：现已接入32档训练、余弦退火、Proxy选best、指标与断点恢复，入口见第十节。
第八/九节提及“尚无训练入口”描述的是此前实现阶段，不是当前状态。

这是对可读版 GraphReaction 的三项新修改，不是原实验已有的设计。
历史说明原文见 checks/history/model_guide_v1.py；历史 comparison_result.json 保留。
新版报告为 checks/version2_result.json。X13 与原 X17 完整网络不等价。

一、整体调用

load_static_graph：原物理图 -> 普通双向图、反应成员、静态量。
load_window：原始当前/未来状态 -> 仅用当前状态建立 X13。
create_model：按原 Xavier 等规则重新初始化新版。
model(data)：X13 -> H0 -> g0、Z0、a -> H4/Z3 -> g1 -> 七字段。
build_delta_targets：当前/未来原物理状态 -> 七字段标准化增量标签。
compute_loss：原 CE2 训练目标与 Q_ref 指标。
recover_physical_state：标准化增量 -> 未来原物理场。

模型不读取 future。预测仍为原数组 t+1，即 10 秒后的单步。
默认四轮是网络内部更新，不是向未来预测四个物理时间步。

二、节点输入 X13

列 0:7   phase1～phase7 one-hot
列 7:10  原归一化 Z、Y、X，严格保持顺序
列 10    原训练统计标准化的体积
列 11    浓度类槽
列 12    电势类槽

phase    浓度类槽                  电势类槽
  1      cg / 样本 cmax_g           0
  2      原标准化 ce                原标准化 phi_e
  3      cp / 样本 cmax_p           原标准化 phi_p
  4      0                          原标准化 phi_CBD
  5      原标准化 ce                原标准化 phi_e
  6      0                          0
  7      0                          原标准化正集流体电势

node_features.build_node_features：有效字段分别变换后填槽，不合并统计，不裁剪。
不存在的字段才填零；真实有效值为零仍正常保留。
field_validation.validate_physical_fields：由 phase 核对字段、节点映射和有限性。
缺失及 NaN/Inf 报告 sample/time/field/node。原物理 mask 和字段集合不删除。
仍先 float64 变换，再转 float32；保持原节点顺序和设备转移方式。

测试先用旧适配器生成 X17，再按下式独立映射；生产路径直接生成 X13：
new[:7]=old[10:17]；new[7:10]=old[6:9]；new[10]=old[9]。
phase1/3 的 new[11]=old[0]；phase2/5 的 new[11]=old[1]。
phase2/3/4/5/7 的 new[12]=old[2]；其余动态槽为零。
两种浓度在不同相共用一列，旧 17 维节点权重不能通用无损裁剪成 13 维。

三、反应属性 reaction_b12

列 0:2   NEG/POS 反应 one-hot，旧 kind=12/13 显式映射为 0/1
列 2:9   当前关联成员节点的 phase1～phase7
列 9:12  当前关联的 CENTER/ION/ELECTRON，旧 role=1/2/3 映射为 0/1/2

同一节点在不同反应单元中的角色可以不同；不能总用中心节点的 phase。
反应区域只有 NEG=0/POS=1；节点区域是 NEG=0/SEP=1/POS=2。
共享端口编码：[H成员64;b12] -> 76 -> 64 -> ReLU -> LayerNorm。
全部成员一起做算术均值，拼 log1p(成员数)，共享 initializer 仍为 65 -> 64。
负极/正极各 Linear(12,1)，sigmoid 后得到 rate [P_R,1]。
Z0、a 只在每次前向开始生成一次；a 与 g0 在内部各轮固定，但梯度不断开。

构图仍以 phase1/3 为中心，ION/ELECTRON 沿用原关系规则。
无 ION 不建反应单元；无 ELECTRON 仍保留合法 size2 单元。
成员排序、按角色去重、重叠超边、关联数和区域归属不变。
reaction_support_incidence / reaction_support_physical_row 仅供核查支撑物理行。
反应成员去重不删除其平行支撑行。空反应属性固定为 (0,12)。

仅反应属性压缩可独立等价迁移。测试 KEEP：
[12,13,14,15,16,17,18,19,20,22,23,24]
确认旧 b30 丢弃列全为零后：
new_port_weight = cat(old_weight[:,:64], old_weight[:,[64+k for k in KEEP]])
new_rate_weight = old_rate_weight[:,KEEP]
偏置、LayerNorm、initializer 和其他对应参数原样复制。
对拍函数见 checks/v2_helpers.mapped_processor_weights；不用于默认恢复完整旧模型。

四、模块维度

模块                     输入 -> 输出                   数量
节点编码                 13 -> 64                       共享 1 个
全局编码                 7*64+6=454 -> 64                共享 1 个，计算 g0/g1
普通图消息               64+64+30+30=188 -> 64 -> 64     5 组
节点 control             128 -> 64 -> 64                3 组
反应端口编码             76 -> 64                       共享 1 个
反应 initializer         65 -> 64                       共享 1 个
反应 rate                12 -> 1 -> sigmoid             2 组
反应 control             128 -> 64 -> 64                2 组
phase1/4/7 解码          128 -> 1                       各相独立
phase2/3/5 解码          128 -> 2                       各相独立

H、Z、g 都为 64 维。编码器为 Linear/ReLU/LayerNorm；
消息/control 为 Linear/ReLU/LayerNorm/Linear。
普通图仍保留全部原物理边和平行边，每条两个相邻方向，reverse[k]=k^1。
普通端口 30 维：类型14+phase7+role4+几何4+有效标记1。
物理边 role=0，几何为面积/起点段/中段/终点段；反向交换起终点段。
五组分别为负极/隔膜/正极内部、负极-隔膜、隔膜-正极。
聚合除以全部物理入度，不是五组分别平均后求和。

五、同步更新与诊断

k=0：H0/Z0 -> H1/Z1
k=1：H1/Z1 -> H2/Z2
k=2：H2/Z2 -> H3/Z3
k=3：H3/Z3 -> H4，默认不计算 Z4

每轮读取同一份旧状态：
q = a * (Z[单元] - H[节点])
R = 按节点反应关联数对 q 求均值
D = 普通图消息按全部物理入度求均值
U = node_control(H,g0)
H_next = H + 0.25 * (D + R + 0.1*U)
前三轮再计算：
DR = 按单元成员数对 -q 求均值
UR = reaction_control(Z,g0)
Z_next = Z + 0.25 * (DR + 0.1*UR)

先算分量再赋值，不原地修改 H/Z，不 detach，参数跨轮共享。
末轮 DR/UR/Z_next 的计算跳过，但 q/R/D/U 全部保留。
最后只由 H4 重算 g1，解码不依赖 Z4。

output = model(data)  # 默认不 capture，也不计算 Z4
output = model(data, capture=True)  # 保存 H0..H4、Z0..Z3
output = model(data, capture=True, compute_terminal_reaction=True)  # 诊断补算 Z4

诊断 Z4 与 H4 同轮产生，使用 H3/Z3，不能用 H4 回填。
output['prediction'] 七字段接口不变。
output['reaction_hidden_step'] 默认为3，完整诊断为4，对应实际返回的 Z。
output['trace'] 默认 None；开启后含 H、Z 两个不同长度的列表和 steps 列表。
steps 中每项为 Dgraph/R/U/q/DR/UR，默认末项 DR/UR 为 None，不伪造 Z4。
另有 a/g0/g1/reaction_hidden_step。
trace.py 的 record_capture/validate_capture 按此契约记录和检查。

六、输出与损失不变

七字段：ce、phi_e、theta_g、theta_p、phi_p、phi_CBD、phi_cc+。
临时解码槽拆分索引固定 [0,1,0,0,1,0,0]，不由 X13 输入槽推导。
原 [N,2] 物理状态的电势位于第1列，因此 FIELD_STATE_COLUMNS 与解码槽不同。
石墨电势不预测；phase6 电势固定零；有效物理输出 mask 保留。
石墨/AM 增量尺度乘样本 cmax 恢复浓度；无硬裁剪。
电流使用原面积、固相端长度及 AM/CBD 电导率0.01/760 S/m。
I_app 从当前样本清单读取，不使用网络归一化几何重建电流。
两个电解液字段都先按 phase2/5 分别求 MSE，再以0.5:0.5平均。
CE2 = (2*L_ce + 其他六字段loss之和)/8 + 0.1*L_current
Q_ref = 七字段loss等权平均 + 0.1*L_current
输入/隐状态 float32，原物理状态/标签/损失 float64，没有新损失。

七、版本、缓存、初始化与 checkpoint

model_config.py：维度与版本常量。
schema.py：feature_identity、model_configuration、configuration_hash、statistics_identity。
configs/version2.json：新版配置、哈希、种子2026初始化 manifest 与实测参数审计。
模型 schema 不改变原数据集身份，不拟合或改写 normalization_stats.json。

默认 load_static_graph 直接构图，不写缓存。
可选 load_static_graph(folder, statistics, cache_root=Path('GraphReaction/.cache'))
从项目根目录调用时，输出：
GraphReaction/.cache/reaction_b12_v2/<原图来源hash>.pt
hash 覆盖 phase、原物理连接、关系、节点区域，缓存另存独立反应 schema。
旧 b30/.npz 缓存明确拒绝；重新构图写入 v2 目录，不改旧 manifest、不补零、不截断。

checkpoint.save_checkpoint(path, model, statistics_id, dataset_id)
checkpoint.load_checkpoint(path, model, statistics_id, dataset_id)
保存配置/哈希、原数据与统计身份、初始化 manifest、参数审计、state_dict。
statistics_id 可取 data.statistics_identity；dataset_id 使用原数据集清单身份或哈希。
恢复前核对这些身份，再 strict=True 加载。
旧 X17 checkpoint/旧 paired 初始化明确拒绝。create_model 默认从头初始化。
不保存或恢复 optimizer/scheduler；本项目目前也没有训练入口。

初始化保留 Xavier、bias=0、LayerNorm=1/0；rate=0，所以 a=0.5；control末层为0。
同组初值复制但 storage 独立。形状变化会改变随机数消耗，不与旧网络按相同seed对拍。
实测：节点编码器1024、反应端口编码器5056、两套rate合计26，总计185507。
原186951减1444；跳过 Z4 不删除参数。

八、源码差异与检查

当前源码使用 PyG Data.edge_metadata/reaction_metadata，没有 top.reaction 对象。
只实现 GraphReaction，没有 GraphOnly；固定 K=4，没有运行时 K 配置。
改动前没有缓存/checkpoint/capture/record_capture/validate_capture 接口；新版按需补入小文件。
改动前没有返回反应支撑物理行，本次补出核查字段，与参考原始构图逐项比较。
用户已有 data_loader.py、regions.py 注释保留。

从 NewProgramByMe 根目录执行：
.venv/Scripts/python.exe GraphReaction/checks/check_version2.py
.venv/Scripts/python.exe GraphReaction/use_model.py

A/B：七相 X13、零值、缺字段、NaN/Inf、非对称 ZYX、旧适配器列映射、实际编码器输入；
平行边、重叠超边、同节点不同角色、size2、零反应、空图、物理支撑行。
C：固定 H0/g0，显式映射反应权重，非零rate，比较端口/Z0/a/处理器/读出/损失/梯度。
D：非零reaction control末层，比较默认与完整诊断 H/Z/预测/CE2/Q_ref/梯度，核对4/3调用次数。
测试中临时将轮数常量设为1，检查 K=1 默认仅生成 H1/Z0；正式配置仍为4。
E：初始化审计、新checkpoint和缓存往返、旧版拒绝、人工图与两个真实窗口的可丢弃更新。
沿用原容差：预测 atol=2e-6/rtol=2e-5；损失1e-9/2e-4；梯度2e-5/2e-4。
结果和最大误差见 checks/version2_result.json；不启动正式训练、完整评价或覆盖旧验收记录。

九、后续新增的学习率策略：余弦退火

原参考工程训练器使用 ReduceLROnPlateau；当前简化版尚无完整训练入口。
现在新增 learning_rate.py，供后续入口使用；本次不改变前三项模型修改。
NUM_EPOCHS=141、INITIAL_LR=0.001、MIN_LR=0.00001 集中在该文件中。
create_scheduler(optimizer) 返回 CosineAnnealingLR(T_max=141, eta_min=0.00001)。
优化器的初始 lr 必须设为 INITIAL_LR；优化器类型及其他参数沿用原协议。
每个 epoch 的 optimizer.step() 全部完成后，调用一次 scheduler.step()。
不再调用 scheduler.step(validation_loss)；余弦调度不监控验证损失。
t 表示已经完成的 epoch 数：首轮用 eta(0)，第141轮用 eta(140)，结束后到 eta(141)。
因此第141轮实际训练学习率略大于1e-5；结束后的学习率才精确达到1e-5。
严格保持公式中的 T=141，无 warmup/热重启；满141轮后停止调度。
配置记录在 configs/learning_rate.json，与模型架构身份分开记录。
正式训练尚未接入，旧 early stopping、模型选择及数据流程不在此模块中实现或更改。
可执行 checks/check_learning_rate.py 核对全部142个时间点，仅使用虚拟参数，不训练模型。

十、32档正式训练与指标记录（本次新增）

用户已明确确认：每轮做Proxy验证，按population加权Q_ref严格降低选best；
最多141轮，连续20轮未改善则早停。余弦T_max=141不因提前停止而改变。
用户本次明确授权正式训练，因此附件中历史“不要启动正式训练”的边界不适用于本次启动。

训练入口：train.py 的 run_training/main。
    .venv/Scripts/python.exe GraphReaction/train.py
    .venv/Scripts/python.exe GraphReaction/train.py --resume <已有run目录>
    .venv/Scripts/python.exe GraphReaction/train.py --epochs 2 --smoke-windows 8

training_config.py：seed2026、AdamW(lr=.001,weight_decay=1e-6,betas=.9/.999,eps=1e-8)，
梯度累积4、clip_norm=1、早停20、4个CPU线程、AMP/TF32关闭，模型结构和损失不变。
training_windows.py：读取proxy_runs/time_v1的32档索引/权重；核验源哈希和划分。
    train=65图/2080窗，proxy=14图/448窗；full来自当前项目Datasets验证集14图/4917窗。
    训练按原MT19937先打乱轨迹再打乱轨迹内窗口，4窗一组可跨轨迹；每轮520次更新。
    不用子集相邻行代替原t+1；只缓存一条轨迹的静态图/只读动态mmap。
    data_loader.load_physical_state新增可选dynamic_arrays参数，原计算及默认调用不变。
training_step.py：train_group，零梯度->逐窗加权CE2反向->裁剪->一次optimizer.step。
    最后一组不足4窗按实际组长度除。监控直接读取原loss，禁止额外训练前向/反向。
training_metrics.py：metric_values/MetricMean/ErrorSums，保存脱离计算图的充分统计。
    在线训练为sum(mean_loss_weight*value)/窗口数；Proxy为sum(population_weight*value)/权重和。
    ce/phi_e保持phase2/5各自平均后等权，另写分项；电流MSE未乘0.1，RMSE%=100*sqrt(聚合MSE)。
evaluation.py：固定best后依次评价train/proxy/full；窗口权重依次为mean_loss_weight/population_weight/1。
    标准化增量MSE与物理节点池化RMSE/MAE分列。物理SSE/SAE/count同时乘scope权重。
    cg/cp恢复为mol/m^3后不再乘cmax；电势误差仅一次V->mV；电流仍走原物理接触计算。
    区域只取有效字段；接口为原跨区物理边去重端点。
    stage复用原manifest的8个时间分层bin；不是SOC阶段或新推断的物理阶段。
diagnostics.py：run_diagnostics/capture_rows，首轮、每10轮和最终best捕获固定两个窗口。
    固定面板为首个train样本及首个proxy样本的首个选择窗口，记录于config。
    H0..H4、Z0..Z3、q/R/D/U和a区域/角色分组均保存；DR3/UR3为null/not_computed。
    RMS为sqrt(mean(x**2))；空反应量null/not_applicable；比例eps=1e-12，仅隐藏幅值诊断。
    topology_coverage.json每图只计算一次，含phase/region覆盖、关联度、普通零入度。
training_checkpoint.py：每次完成optimizer更新后原子提交完整last.pt，随后写update_metrics.csv。
    内容包括严格版本/数据/源码身份、模型、AdamW、余弦状态、RNG、epoch/cursor、在线充分统计。
    若中断在提交checkpoint之后、CSV写入之前，恢复只补最后一行；重复/超前/多行缺失均拒绝。
    每轮完成后更新history、严格选best，best也包含完整恢复状态。旧模型专用checkpoint接口仍独立。
plot_training.py：从history画CE2、Q_ref和学习率曲线，不再运行模型。

新run位于GraphReaction/training_runs/proxy32_cosine_XXX；smoke和人工测试使用单独目录。
    config.json：模型/训练配置、数据/源码哈希、版本、固定诊断面板；source保存本次运行源码快照。
    metrics/metric_definitions.md/json：实际列名、口径、单位、分母/权重和诊断频率。
    update_metrics.csv：每次成功optimizer.step后的lr、裁剪前norm、异常计数与order_hash。
    history.csv/json：每轮online_train_*/proxy_*、lr/next_lr、best/早停、梯度及耗时。
    checkpoints/best.pt/last.pt：完整训练断点。
    diagnostics/mechanism_metrics.csv：低频机制量；capture_*.csv是各次原始诊断记录。
    evaluation/{train,proxy,full}/：overall_metrics.json、field/region/stage_metrics.csv，
        trajectory_metrics.json以及逐窗voltage_current.csv。
    training_summary.json：当前状态、best来源、停止原因和实际评价完成状态；未运行项not_run。
    loss_curve.png：在线训练/Proxy曲线；failure.json仅在异常或中断时写出。

完整验证集只在训练停止且best固定后使用，不参与学习率、best或早停选择；不读取测试集、不跑rollout。
日志恢复和RNG测试：checks/check_training_metrics.py；人工断点后继续两轮与连续路径参数误差为0。
两个epoch、每scope8窗的GPU smoke已通过，明确标记smoke；不是正式实验性能结果。
正式运行进度见training_runs/active_training.json指向的run与stdout日志。

十一、2026-10-26：重新开始，仅以训练CE2选best（当前规则）

本节替代第十节的逐轮Proxy验证/选模规则；第十节保留为旧实验说明。
用户最终要求重新训练，不接续第58轮：新模型、AdamW和余弦调度器均重新初始化，
seed仍为2026，最多141轮。每轮只使用32档65条训练轨迹/2080窗。
best依据online_train_CE2严格降低保存本轮末权重；相等不算改善。
连续20轮未改善则早停。在线loss来自本轮训练过程中的参数，不额外进行轮末训练集前向。
余弦仍为T_max=141、0.001降至0.00001；早停不重新缩放余弦周期。
训练期间既不做Proxy验证，也不进行原先会访问Proxy小面板的机制前向。
训练结束固定best后，才执行机制诊断，并评价train/proxy/full及输出原有全部指标。
模型、物理恢复、CE2/Q_ref计算、图拓扑、batch累积与数值精度保持不变。

training_config.py记录monitor=online_train_CE2、validate_each_epoch=False，
schema为graphreaction_train_ce2_selection_v2。train.py记录best_train_CE2，
history中的proxy_*和epoch_proxy_seconds为空，validation_status=not_run，不能用0代替未验证。
plot_training.py允许没有验证曲线，照常绘制在线CE2/Q_ref和学习率。
新运行位于training_runs/proxy32_train_ce2_XXX；旧proxy32_cosine_001保持原样。
新旧配置/源码身份不同，--resume不会静默加载旧选模协议；同版本新运行仍可精确断点续训。

保存修复：run_files.replace_with_retry只对Windows错误5/32/33做最多10次替换尝试，
等待总计最多约6.5秒。每次先完整写临时文件，再替换目标，不先删除旧checkpoint。
短暂占用消失后继续；持续拒绝访问仍报错，保留旧断点和临时候选文件。
training_checkpoint.save_training_checkpoint以及JSON/CSV写入使用该函数；
best用copy_atomic先复制临时文件再替换，避免覆盖到一半留下损坏的best。
原错误的具体占用进程没有证据确定；这里修复的是瞬时拒绝访问导致直接退出的处理。

检查：checks/check_train_only.py。
覆盖真实Windows文件占用后释放、持续占用保留原断点、训练期间零验证窗口读取、
CE2严格选模、相等20轮后早停、停止后从best评价、轮中/轮末恢复与旧运行拒绝。
真实GPU短流程：train.py --epochs 2 --smoke-windows 8，另存独立smoke目录。
"""
