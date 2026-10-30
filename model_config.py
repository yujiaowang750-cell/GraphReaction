"""30维普通/反应端口；保留X13和末轮不计算Z4。"""

HIDDEN_DIM = 64
NODE_INPUT_DIM = 13
PAIR_PORT_DIM = 30
REACTION_PORT_DIM = 30
# 端口列定义采用30维输入；完整模型使用X13。
MODEL_SCHEMA = "graphreaction_x13_b30_skipz4_v3"
NODE_SCHEMA = "phase7_zyx_volume_concentration_potential_v2"
PAIR_SCHEMA = "physical_pair_b30_v1"
REACTION_SCHEMA = "reaction_kind14_phase7_role4_geometry_b30_v3"
PROCESSOR_STEPS = 4
STEP_SIZE = 0.25
CONTROL_WEIGHT = 0.1
LAYER_NORM_EPS = 1e-5

# 区域编号：负极、隔膜、正极。
REGION_NAMES = ("negative", "separator", "positive")

# 物理边分成五组，每组使用自己的消息 MLP。
EDGE_GROUP_NAMES = (
    "negative_internal", "separator_internal", "positive_internal",
    "negative_separator", "separator_positive",
)

# 关系编号 0～11 对应的两端相编号，保留参考代码中的顺序。
RELATION_PHASES = (
    (1, 1), (1, 2), (1, 6), (2, 2), (2, 3), (2, 5),
    (3, 3), (3, 4), (3, 7), (4, 4), (4, 7), (5, 5),
)

# 七个预测场的顺序。这里没有石墨电势；正极集流体电势属于七个字段之一。
FIELD_NAMES = (
    "electrolyte_concentration_mol_m3",
    "electrolyte_potential_v",
    "negative_graphite_lithiation_fraction",
    "positive_am_lithiation_fraction",
    "positive_am_solid_potential_v",
    "positive_cbd_solid_potential_v",
    "positive_collector_potential_v",
)
DECODER_DIMS = {1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 7: 1}
# 原始 [N,2] 状态的列：0=浓度，1=电势；与七字段顺序对应。
FIELD_STATE_COLUMNS = (0, 1, 0, 0, 1, 1, 1)
FIELD_WEIGHTS = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
CURRENT_LOSS_WEIGHT = 0.1

# 与原始几何配合计算物理电流，单位 S/m。
AM_CONDUCTIVITY = 0.01
CBD_CONDUCTIVITY = 760.0
