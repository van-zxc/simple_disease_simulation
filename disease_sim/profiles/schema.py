"""档案 schema 的数据定义层：版本语义、覆盖白名单、年龄/曲线/默认值表。

职责：保存"档案允许长什么样"的静态数据，供 loader/validate/derive/apply 共享。
非职责：不做 I/O、不做校验、不做派生；本模块是 profiles 子包的依赖根。
版本语义（只增不改，旧档案按原路径继续可用）：
  v1 旧 SEIR：仅 ``model_overrides`` 简档或基础 ``epidemiology`` 派生（toy.json）；
  v2 = v1 + ``natural_history``（无症状/隔离/住院/时变传染力/年龄相对表）+ ``medical``；
  v3 = v2 + ``interventions``（检测 / 追踪隔离 / 行为反馈 / 报告）；
  v4 = v3 + ``heterogeneity``（传染力 sigma、依从性）+ ``mixing``（家庭乘子 / 年龄同配 / 症状居家）。
不变式：``model_overrides`` 只能覆盖 ``ALLOWED_OVERRIDES`` 中的 CONFIG 键（v1 简档的安全边界）；
所有默认值"关闭即中性"——老档案不写新块时，行为与旧版本等价。
"""

from __future__ import annotations

from pathlib import Path


# 不变式：路径假设 <repo>/disease_sim/profiles/schema.py —— parents[2] 才是仓库根；
# 移动本文件、或把 profiles 打包进别处，diseases/ 就会指向错误目录。
PROFILE_DIR = Path(__file__).resolve().parents[2] / "diseases"

# 不变式：默认档案必须是 v4 且四层社会响应全开（默认演示与回归测试依赖其行为）。
DEFAULT_PROFILE = "covid19"

SCHEMA_VERSION = 4              # 不变式：等于 SCHEMA_VERSIONS 的最大值；新档案一律写当前版本
SCHEMA_VERSIONS = (1, 2, 3, 4)  # 不变式：版本白名单，不在其中的 schema_version 一律拒绝



# 注意：以下常量必须与模型内部实现一致（config/contacts/model 侧改动时同步），
# 否则派生出的 beta 与 hazard 会和仿真实际使用的参数脱节。
FRAMES_PER_DAY_KEY = "DAY_FRAMES"   # 不变式：档案 time.frames_per_day 与 CONFIG 的该键同义
CONTACT_RATE_PER_FRAME = 24.6       # 不变式：活动模型每帧有效接触数（夜间≈17、工作时段 85–99、日均≈25）
TRANSMIT_BETA_EXPONENT = 1.5        # 不变式：传播概率 p = beta**1.5 × transmit_base_prob
DEATH_HAZARD_FRACTION = 0.05       # 不变式：死亡风险 d_p = mu × medical_pressure × 0.05



# 不变式：``model_overrides`` 只能覆盖下列 CONFIG 键——这是 v1 简档的安全边界；
# 白名单外的键（渲染 / RNG / 人口缩放等）会被 validate 拒绝，防止档案越权改模型结构。
# 注意：新增可覆盖键时必须同步登记到这里与 validate._validate_overrides 的数值检查。
ALLOWED_OVERRIDES = {
    "BASE_TRANSMIT", "BASE_RECOVERY", "BASE_DEATH",
    "MUTATION_RATE", "MUTATION_SCALE", "MUTATION_BOUNDS",
    "INCUBATION_MU", "INCUBATION_SD", "INCUBATION_MIN",
    "INCUBATION_SELF_RECOVERY", "INCUBATING_TRANSMISSION_MULT",
    "MEDICAL_THRESHOLD", "TRANSMIT_BASE_PROB",
    "IMMUNITY_GAIN_PER_INFECTION", "IMMUNITY_DECAY_RATE",
    "MAX_IMMUNITY", "IMMUNITY_EFFECTIVENESS", "IMMUNITY_WANING_THRESHOLD",
    "INITIAL_SEEDS", "INITIAL_IMMUNE_FRACTION", "INITIAL_IMMUNE_LEVEL",
    "FRAMES",
}



# schema v2 起引入的年龄维度：9 档、10 岁一档、末档 80+（v4 混合再映射成 4 个粗组）。
# 不变式：所有 age_* 相对表必须恰好包含这 9 个键（validate._validate_age_table 强制）。
AGE_BANDS = ("0-9", "10-19", "20-29", "30-39", "40-49",
             "50-59", "60-69", "70-79", "80+")

# 旧版 status_mult 的等价曲线：E 期恒 0.15、发病期恒 1.0。
# 不变式：v2+ 档案省略 infectivity_curve 时以它为默认值（v1 档案不走自然史派生，
# 不会读取本曲线）；改动数值等于改动这些省略曲线档案的行为。
LEGACY_INFECTIVITY_CURVE = [[-1.0e9, 0.15], [0.0, 1.0], [1.0e9, 1.0]]

# 无症状者的传染力曲线：恒 0（旧行为——无症状完全不传播）。
LEGACY_ASYMPTOMATIC_CURVE = [[0.0, 0.0]]

# v2 自然史字段的公开参考默认值。注意：derive._derive_natural_history 内联了同值默认，
# 两处必须保持一致（否则"省略键"的语义会随调用路径而变）；本表当前不被包内代码 import。
NATURAL_HISTORY_DEFAULTS = {
    "asymptomatic_fraction": 0.0,
    "asymptomatic_days": 0.0,
    "isolation_probability": 0.0,
    "isolation_delay_days": 0.0,
    "isolation_transmission_mult": 1.0,
    "hospital_delay_days": 0.0,
    "ward_days": 7.0,
    "hospital_death_share": 0.7,
    "untreated_death_mult": 1.0,
    "infectivity_curve": LEGACY_INFECTIVITY_CURVE,
    "asymptomatic_infectivity_curve": LEGACY_ASYMPTOMATIC_CURVE,
}



# schema v3 社会响应层默认值：检测 / 追踪隔离 / 行为反馈 / 报告。
# 不变式：顶层 enabled 与各子块 enabled 默认 True —— 写了 interventions 段的档案，
# 未显式关闭的层都算开启；完全省略该段的档案（v2 及旧档）由 apply 写成全关。
# 不变式：四层子块的键集合同时是白名单（见下方 _INTERVENTION_KEYS），未知键一律拒绝。
INTERVENTIONS_DEFAULTS = {
    "enabled": True,
    "testing": {
        "enabled": True,
        "test_probability_symptomatic": 0.6,
        "test_delay_days": 1.5,
        "test_capacity_per_100k_per_day": 800.0,
        "test_sensitivity": 0.9,
        "isolation_adherence": 0.8,
        "isolation_delay_days": 0.5,
    },
    "tracing": {
        "enabled": True,
        "trace_probability": 0.6,
        "trace_delay_days": 1.0,
        "trace_household": True,
        "trace_venue_fraction": 0.25,
        "quarantine_days": 10.0,
        "quarantine_adherence": 0.8,
        "quarantine_transmission_mult": 0.5,
    },
    "behavior": {
        "enabled": True,
        "response_strength": 0.35,
        "response_threshold_per_100k": 20.0,
        "response_scale_per_100k": 50.0,
        "response_delay_days": 7.0,
        "fatigue_days": 90.0,
        "fatigue_strength": 0.6,
        "min_contact_mult": 0.2,
    },
    "reporting": {
        "enabled": True,
        "reporting_fraction": 0.5,
        "reporting_delay_days": 3.0,
        "reporting_symptomatic_only": True,
    },
}



# 社会响应各子块的已知键集合（由默认值表派生），validate 用它拒绝未知键。
# 注意：新增默认键会自动进入白名单，但必须同时在 validate 里补上区间/布尔检查。
_INTERVENTION_KEYS = {
    "testing": set(INTERVENTIONS_DEFAULTS["testing"]),
    "tracing": set(INTERVENTIONS_DEFAULTS["tracing"]),
    "behavior": set(INTERVENTIONS_DEFAULTS["behavior"]),
    "reporting": set(INTERVENTIONS_DEFAULTS["reporting"]),
}



# schema v4 个体异质性默认值：
#   * infectiousness_sigma：传染力乘子的对数正态 sigma（0 = 无超传播，均值归一为 1）；
#   * compliance_*：依从性个体差异（乘在自然隔离、检测隔离、密接隔离、行为反馈上）。
# 取值范围：sigma∈[0, 3]、compliance_mean∈[0, 1]、compliance_sd∈[0, 0.5]（validate 强制）。
# 注意：关闭块也会被写进 CONFIG（apply._apply_heterogeneity 不看 enabled），因此档案里的
# v4 块即使 enabled=false 也必须写中性值，否则 CONFIG 会残留非中性参数。
HETEROGENEITY_DEFAULTS = {
    "enabled": False,
    "infectiousness_sigma": 0.0,
    "compliance_mean": 1.0,
    "compliance_sd": 0.0,
}



# schema v4 接触混合结构默认值：
#   * household_contact_mult：家庭内接触的相对乘子；
#   * age_assortativity：按 4 个粗年龄组同配（0 = 均匀混合，越大越"同组扎堆"）；
#   * symptom_stay_home_fraction：有症状者自发减少外出的比例。
# 取值范围：household_contact_mult∈[1, 10]、age_assortativity∈[0, 0.9]、
# symptom_stay_home_fraction∈[0, 1]（validate 强制）。
# 不变式：年龄同配矩阵按 E[phi] = 1 归一化，因此总体 R0 期望不变，只改变接触的分配。
MIXING_DEFAULTS = {
    "enabled": False,
    "household_contact_mult": 1.0,
    "age_assortativity": 0.0,
    "symptom_stay_home_fraction": 0.0,
}



# 接触混合用的 4 个粗年龄组（child / adult / older / senior，见 MECHANISMS.md 第 12 节）。
AGE_GROUPS = ("0-19", "20-49", "50-69", "70+")

# 不变式：9 档 AGE_BANDS -> 4 粗组的映射，长度必须为 9（0-19 / 20-49 / 50-69 / 70+）。
AGE_BAND_TO_GROUP = (0, 0, 1, 1, 1, 2, 2, 3, 3)


class ProfileError(ValueError):
    """档案缺失或非法；调用方捕获后打印中文说明并退出，不进入仿真。"""
