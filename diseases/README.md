# 疾病参数档案（diseases/）

> 文档分工：本文件是**档案字段与校验规则的权威清单**（含新增疾病的 9 步流程）；
> 机制怎么用这些参数见 [MECHANISMS.md](../MECHANISMS.md)，缩放/性能见
> [SCALING.md](../SCALING.md)。

本目录存放**可切换的疾病参数档案**（JSON）。模型不把疾病相关数字硬编码在
CONFIG 里，而是启动时从档案加载、校验、派生为每帧参数，再写入 CONFIG。
代码位置：加载/校验/派生/应用在 `disease_sim/profiles/{loader,validate,derive,apply}.py`，
子包默认档案与 schema 常量在 `schema.py`。

当前 schema 版本：

- `schema_version = 4`：在 v3 之上增加 **个体异质性与接触混合**
  （`heterogeneity`：传染力对数正态超传播、依从性个体差异；`mixing`：家庭接触乘子、
  4 组年龄同配、症状者自发居家）。默认 `covid19` 使用它；两块都可单独 `"enabled": false` 关闭。
- `schema_version = 3`：自然史 / 医疗容量（自 v2 起）+ **社会响应层**
  （`interventions`：检测、追踪隔离、行为反馈、报告）。
  `covid19_v2_noiv` 是 v3 档案，但**省略 `interventions` 块**，社会响应全关，
  作为"自然史开启、社会响应关闭"的对照档。
- `schema_version = 1`：旧版 SEIR 机制（`toy`），用于快速演示与统计回归。
  v1 只能用 `model_overrides` 简档或旧的 `epidemiology` 基础派生路径。

机制细节（状态机、自然史、社会响应如何生效）见 [MECHANISMS.md](../MECHANISMS.md)。

## 1. 文件与选择方式

| 文件 | 用途 | schema | 社会响应 | v4 异质性 |
| --- | --- | --- | --- | --- |
| `covid19.json` | COVID-19（SARS-CoV-2 祖先株，2020），默认档案 | 4 | 四层全开 | 开 |
| `covid19_homogeneous.json` | 同一档案但 `heterogeneity`/`mixing` 关闭：**同质对照** | 4 | 四层全开 | 关 |
| `covid19_v2_noiv.json` | 同一 COVID 档案，省略 `interventions` | 3 | 全关（回归档） | 不适用 |
| `toy.json` | 旧版演示参数（`model_overrides` 简档） | 1 | 不适用 | 不适用 |

选择优先级（高 → 低）：

1. 命令行 `--profile NAME`（也支持 `--profile=NAME`）
2. 环境变量 `DISEASE_PROFILE=NAME`
3. 默认值 `covid19`（`disease_sim.profiles.DEFAULT_PROFILE`）

```bash
python simple_disease_simulation.py                          # 默认 covid19（v3）
python simple_disease_simulation.py --profile toy            # 旧演示参数（v1 legacy）
python simple_disease_simulation.py --profile covid19_v2_noiv  # 关闭社会响应
DISEASE_PROFILE=covid19 DISEASE_POPULATION=100000 python simple_disease_simulation.py
SIM_FRAMES=2000 python simple_disease_simulation.py           # 覆盖模拟帧数（快速预览）
```

档案在**人口缩放之前**应用；`diseases/*.json` 也记录在
`benchmarks/golden_*.json` 的回归口径里。

## 2. 单位约定

- 模型 1 帧 = 6 分钟（活动节律），`frames_per_day = 240`。
- **JSON 里的时长全部用「天」**，loader 统一换算为帧：`T = days × 240`。
  这包括自然史时长（潜伏期/感染期/住院延迟/病房病程）、隔离/住院延迟，
  以及 v3 的检测延迟、检测隔离延迟、追踪延迟、隔离天数、行为响应延迟、
  报告延迟。
- beta / gamma / mu 是模型内部的**每帧**概率参数，由 loader 派生，不在 JSON 中手填。
- 传染力曲线的横轴也是「天」，相对**症状出现**（负值=症状前）；无症状曲线相对**感染**。
- v3 的 `test_capacity_per_100k_per_day` 单位是"每 10 万人每日检测数"，
  运行时按人口线性放大为 `round(该值 × N / 100000)`（见 §4.3）。
- 疾病时间刻度与活动节律共用同一时钟（1 帧 = 6 分钟，按天换算）；
  这是抽象约定，不做真实地理/密度校准。

## 3. Schema v4

v4 = v3 + 两个可选顶层块（`heterogeneity`、`mixing`），语义与实现见
[MECHANISMS.md](../MECHANISMS.md) §12。v1/v2/v3 档案不需要任何改动即可继续加载。

顶层字段（`*` 为必填；`epidemiology` 与 `model_overrides` 二选一）：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `schema_version` | 是 | 1/2/3/4 之一；含 `interventions` 时必须 ≥ 3；含 `heterogeneity`/`mixing` 时必须 = 4 |
| `profile_id` | 是 | 必须与文件名一致（loader 用文件名当 expected_id） |
| `label` | 是 | 展示用中文说明 |
| `description` | 否 | 更长说明（工具读取，不参与模型） |
| `time` | 是 | `frames_per_day`（必须与模型一致）+ `simulation_days` 或 `simulation_frames` |
| `population` | 否 | `initial_immune_fraction`（CONFIG 默认 0.01；covid19 取 0.0）、`initial_immune_level`（0.9）、`initial_seeds`（每万人初始感染数，covid19 取 20） |
| `epidemiology` | 二选一 | 流行病学输入，由 loader 派生模型参数 |
| `model_overrides` | 二选一 | 直接覆盖 CONFIG 键（如 toy 档），跳过派生 |
| `natural_history` | 否 | 无症状、隔离行为、住院/床位、年龄别严重性、时变传染力；需要 `epidemiology` |
| `medical` | 否 | `beds_per_1000`、`n_hospitals_per_100k`；需要 `epidemiology` |
| `interventions` | 否（v3） | 社会响应四层；需要 `schema_version ≥ 3`；`enabled=true` 时还需要 `natural_history` 与 `epidemiology` |
| `heterogeneity` | 否（v4） | 个体异质性：传染力 σ 与依从性分布；`enabled=true` 时需要 `natural_history` 与 `epidemiology` |
| `mixing` | 否（v4） | 接触混合：家庭乘子、年龄同配强度、症状居家比例 |
| `model_parameters` | 否 | 派生用的模型结构参数与校准因子 |
| `sources` | 否（建议填） | 来源字典，供各数值的 `source` 引用 |
| `notes` | 否 | 已知限制/口径说明 |

### 3.0 v4 两个新块（完整键与取值范围）

```json
"heterogeneity": {
  "enabled": true,
  "infectiousness_sigma": 1.4,
  "compliance_mean": 0.8,
  "compliance_sd": 0.15
},
"mixing": {
  "enabled": true,
  "household_contact_mult": 3.0,
  "age_assortativity": 0.5,
  "symptom_stay_home_fraction": 0.6
}
```

| 键 | 默认（关闭） | 范围 | 含义 |
| --- | --- | --- | --- |
| `heterogeneity.enabled` | `false` | bool | 总开关；关闭时不消耗任何随机数 |
| `heterogeneity.infectiousness_sigma` | 0.0 | [0, 3] | 传染力乘子 σ（均值 1 的对数正态，clamp [0.05,20] 后归一化） |
| `heterogeneity.compliance_mean` | 1.0 | [0, 1] | 依从性均值 |
| `heterogeneity.compliance_sd` | 0.0 | [0, 0.5] | 依从性标准差；0 时全体依从性=1 |
| `mixing.enabled` | `false` | bool | 总开关；关闭时不消耗任何随机数 |
| `mixing.household_contact_mult` | 1.0 | [1, 10] | 同户接触对的传播概率乘子 |
| `mixing.age_assortativity` | 0.0 | [0, 0.9] | 4 组年龄同配强度（φ 归一化保 R0 期望不变） |
| `mixing.symptom_stay_home_fraction` | 0.0 | [0, 1] | 有症状者自发居家比例 |

校验规则（不满足即 `ProfileError` 并非零码退出）：

- `schema_version` 不在 `{1,2,3,4}`；`profile_id` 与文件名不一致；缺 `time`
  或 `frames_per_day`；`epidemiology`/`model_overrides` 都没有。
- `natural_history`/`medical` 出现在 `model_overrides` 档案中。
- `interventions` 出现在 `schema_version < 3` 的档案中；`interventions.enabled=true`
  但缺 `natural_history`。`interventions` 顶层或各子块出现未知键。
- `heterogeneity`/`mixing` 出现在 `schema_version < 4` 的档案中；`heterogeneity.enabled=true`
  但缺 `natural_history`；两个块的未知键或数值越界（见上表）。
- 各字段做数值范围/类型检查（见 §3.4 与各小节的默认值表）。

### 3.1 `epidemiology`（推荐模式）

每个数值对象统一为 `{"value": ..., "range": [lo, hi], "source": "...", "note": "..."}`。
`incubation_days` 例外，使用 `mean/sd/min/max`。

| 键 | 单位 | 用途 |
| --- | --- | --- |
| `r0` | 无量纲 | 目标基本再生数 |
| `incubation_days` | 天 | 潜伏期均值/标准差/最短/最长（`max` 仅记录） |
| `infectious_days` | 天 | 发病后可传染期均值（≈1/gamma） |
| `ifr` | 比例 | 每例感染的死亡概率 |
| `symptomatic_fraction` | 比例 | 有症状比例；自然史启用时派生无症状比例 `a = 1 - 该值` |
| `presymptomatic_transmission_fraction` | 比例 | 症状前传播占比（v1 用来反推潜伏期乘子；自然史启用后由曲线直接积分） |
| `immunity_days` | 天 | 感染后保护从 `gain` 衰减到 `waning_threshold` 的天数 |
| `hospital_beds_per_1000` | 张/千人 | 医疗容量（也接受 `medical.beds_per_1000`） |
| `hospitalization_rate` | 比例 | 症状者住院比例（用于住院/死亡路径拆分） |
| `serial_interval_days` | 天 | 仅用于经验校验，不参与派生 |

### 3.2 `natural_history`（v2 起）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `asymptomatic_fraction` | 0.0 | 无症状比例；无症状不进医院、不隔离（covid19 显式 0.4） |
| `asymptomatic_days` | 0.0 | 无症状固定病程（天），`asymptomatic_fraction > 0` 时必须显式给出 |
| `isolation_probability` | 0.0 | 有症状者被自然隔离的概率 |
| `isolation_delay_days` | 0.0 | 发病到隔离的延迟 |
| `isolation_transmission_mult` | 1.0 | 隔离后传播乘子（对家庭内/外接触统一乘，近似"减少接触"） |
| `hospital_delay_days` | 0.0 | 发病到住院的延迟 |
| `ward_days` | 7.0 | 住院病程（天） |
| `hospital_death_share` | 0.7 | 总死亡中发生在医院内的比例 |
| `untreated_death_mult` | 1.0 | 需住院但无床位者相对 `p_ward_death` 的死亡倍数 |
| `infectivity_curve` | 旧版常数曲线 | 相对症状出现天数的传染力（分段线性，`[[day, mult], ...]`） |
| `asymptomatic_infectivity_curve` | 旧版常数曲线 | 相对感染天数的无症状传染力 |
| `age_ifr_relative` | 全 1 | 9 档年龄（0-9/10-19/…/80+）相对死亡风险 |
| `age_hospitalization_relative` | 全 1 | 相对住院概率 |
| `age_susceptibility` | 全 1 | 相对易感性 |

上表"默认"指**档案省略该键时** `schema.NATURAL_HISTORY_DEFAULTS` 的取值，不是推荐值；
`covid19` 显式写出全部键（无症状 0.4、病程 6 天、隔离 60%、医院死亡占比 0.7 等）。
`natural_history` 整体缺省时模型走 legacy SEIR（`NATURAL_HISTORY_ENABLED=False`）。

年龄表是**相对值**

：主文件初始化时按实际人口年龄分布归一化为均值 1，因此不改变
档案名义的总体 IFR / 住院率 / 易感性，只改变个体间的排序。

### 3.3 `medical`（v2 起）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `beds_per_1000` | 由 `epidemiology.hospital_beds_per_1000` 得到 | 床位密度 |
| `n_hospitals_per_100k` | 10 | 每 10 万人医院数量（按人口缩放，仅用于提示/后续扩展） |

### 3.4 `interventions`（v3 社会响应层）

`interventions` 只允许 `enabled` + `testing`/`tracing`/`behavior`/`reporting`
四个块；各块只允许下表键名（出现未知键直接报 `ProfileError`）。
`interventions.enabled` 默认 `true`；每个子块的 `enabled` 默认 `true`。
总开关或子块关闭时，对应机制不生效且不消耗随机数。

**`testing`（检测 → 隔离）**

| 键 | 默认 | 范围/单位 | 运行时映射 |
| --- | --- | --- | --- |
| `enabled` | true | 布尔 | `TESTING_ENABLED`（与总开关 AND） |
| `test_probability_symptomatic` | 0.6 | [0,1] | `TEST_PROBABILITY_SYMPTOMATIC`：发病时愿意检测的概率 |
| `test_delay_days` | 1.5 | ≥0，天 | `TEST_DELAY_FRAMES`：预约到出结果 |
| `test_capacity_per_100k_per_day` | 800 | ≥0，人/10万/日 | `TEST_CAPACITY_PER_100K_PER_DAY`：每日检测能力 |
| `test_sensitivity` | 0.9 | [0,1] | `TEST_SENSITIVITY`：灵敏度 |
| `isolation_adherence` | 0.8 | [0,1] | `TEST_ISOLATION_ADHERENCE`：阳性后居家依从性 |
| `isolation_delay_days` | 0.5 | ≥0，天 | `TEST_ISOLATION_DELAY_FRAMES`：阳性到隔离的延迟 |

**`tracing`（追踪 → 密接隔离）**

| 键 | 默认 | 范围/单位 | 运行时映射 |
| --- | --- | --- | --- |
| `enabled` | true | 布尔 | `TRACING_ENABLED`（与总开关 AND） |
| `trace_probability` | 0.6 | [0,1] | `TRACE_PROBABILITY`：检测阳性者触发追踪的概率 |
| `trace_delay_days` | 1.0 | ≥0，天 | `TRACE_DELAY_FRAMES`：触发到隔离密接者的延迟 |
| `trace_household` | true | 布尔 | `TRACE_HOUSEHOLD`：是否纳入同户成员 |
| `trace_venue_fraction` | 0.25 | [0,1] | `TRACE_VENUE_FRACTION`：工作/学校同场所成员的纳入比例 |
| `quarantine_days` | 10 | >0，天 | `QUARANTINE_DAYS_FRAMES`：密接隔离时长 |
| `quarantine_adherence` | 0.8 | [0,1] | `QUARANTINE_ADHERENCE`：密接者隔离依从性 |
| `quarantine_transmission_mult` | 0.5 | [0,1] | `QUARANTINE_TRANSMISSION_MULT`：密接隔离者传播乘子 |

**`behavior`（行为反馈）**

| 键 | 默认 | 范围/单位 | 运行时映射 |
| --- | --- | --- | --- |
| `enabled` | true | 布尔 | `BEHAVIOR_ENABLED`（与总开关 AND） |
| `response_strength` | 0.35 | [0,1] | `BEHAVIOR_RESPONSE_STRENGTH`：最大接触降幅 |
| `response_threshold_per_100k` | 20 | ≥0，每 10 万人 7 天报告发病率 | `BEHAVIOR_THRESHOLD_PER_100K`：sigmoid 中心 |
| `response_scale_per_100k` | 50 | >0 | `BEHAVIOR_SCALE_PER_100K`：sigmoid 斜率尺度 |
| `response_delay_days` | 7 | ≥0，天 | `BEHAVIOR_RESPONSE_DELAY_FRAMES`：报告到行为响应的延迟 |
| `fatigue_days` | 90 | >0，天 | `BEHAVIOR_FATIGUE_DAYS`：疲劳时间尺度 |
| `fatigue_strength` | 0.6 | ≥0 | `BEHAVIOR_FATIGUE_STRENGTH`：疲劳强度 |
| `min_contact_mult` | 0.2 | [0,1] | `BEHAVIOR_MIN_CONTACT_MULT`：乘子下限 |

**`reporting`（报告）**

| 键 | 默认 | 范围/单位 | 运行时映射 |
| --- | --- | --- | --- |
| `enabled` | true | 布尔 | `REPORTING_ENABLED`（与总开关 AND） |
| `reporting_fraction` | 0.5 | [0,1] | `REPORTING_FRACTION`：有症状发病被报告比例 |
| `reporting_delay_days` | 3 | ≥0，天 | `REPORTING_DELAY_FRAMES`：发病到报告延迟 |
| `reporting_symptomatic_only` | true | 布尔 | `REPORTING_SYMPTOMATIC_ONLY`；当前实现只报告有症状发病（该键保留作档案声明） |

### 3.5 `model_parameters`（可选）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `contact_rate_per_frame` | `24.6` | 10k 基准下每帧有效接触数（活动模型实测校准；是 `CONTACT_RATE_PER_FRAME` 常量） |
| `r0_calibration_factor` | `1.0` | 直接缩放 p_symp：经验 R0 偏高/偏低时微调（需在 notes 说明） |
| `transmit_base_prob` | `0.08` | 模型的 `TRANSMIT_BASE_PROB` |
| `transmit_beta_exponent` | `1.5` | 模型里 `p ∝ beta^1.5` 的指数 |
| `immunity_gain_per_infection` | `0.6` | 每次感染获得的免疫力增量（toy 档为 0.4） |
| `immunity_effectiveness` | `3` | 免疫抵抗 `(1-immunity)^3` 的指数 |
| `immunity_waning_threshold` | `0.1` | 免疫力衰减到该值后重新易感 |
| `incubation_self_recovery_rate` | `0.0` | 潜伏期每帧自愈概率（toy 档 0.0005） |
| `mutation` | `{enabled: false}` | `{enabled, rate, scale, bounds{beta,gamma,mu}}`；关闭时 `MUTATION_RATE=0` |

### 3.6 `model_overrides`（简档模式 / v1）

把派生结果直接写成 CONFIG 键值，loader 原样覆盖并跳过流行病学派生。
`toy.json` 用该模式逐位复刻旧版演示参数。允许覆盖的键（白名单，其它键报错）：

```
BASE_TRANSMIT, BASE_RECOVERY, BASE_DEATH,
MUTATION_RATE, MUTATION_SCALE, MUTATION_BOUNDS,
INCUBATION_MU, INCUBATION_SD, INCUBATION_MIN,
INCUBATION_SELF_RECOVERY, INCUBATING_TRANSMISSION_MULT,
MEDICAL_THRESHOLD, TRANSMIT_BASE_PROB,
IMMUNITY_GAIN_PER_INFECTION, IMMUNITY_DECAY_RATE,
MAX_IMMUNITY, IMMUNITY_EFFECTIVENESS, IMMUNITY_WANING_THRESHOLD,
INITIAL_SEEDS, INITIAL_IMMUNE_FRACTION, INITIAL_IMMUNE_LEVEL,
FRAMES
```

注意：`model_overrides` 档案跳过派生，也不读取 `natural_history`/`medical`/
`interventions`（校验器会直接拒绝含这些段但无 `epidemiology` 的档案）。
新增简档时，`FRAMES`、`INCUBATION_SD`、`INCUBATION_MIN`、
`INCUBATION_SELF_RECOVERY`、`INCUBATING_TRANSMISSION_MULT`、
`INITIAL_IMMUNE_FRACTION`、`INITIAL_IMMUNE_LEVEL`、`MUTATION_BOUNDS` 这些是
新增键，旧档需要显式写全才能"逐位一致"。

## 4. 派生公式（与 `disease_sim/profiles/derive.py` 一致）

设 `fpd = 240`。

### 4.1 基础量

```
T_inc   = incubation_days.mean      × fpd
sd_inc  = incubation_days.sd        × fpd
min_inc = incubation_days.min       × fpd
T_symp  = infectious_days           × fpd
gamma   = 1 / T_symp

mu      = ifr × gamma / (0.05 × (1 - ifr))          # 旧版死亡/医疗阈值路径使用

p_symp  = r0 × r0_calibration_factor / (C × T_eff)   # C = 24.6（活动模型实测）
beta    = (p_symp / transmit_base_prob) ^ (1 / 1.5)

immunity_decay = (gain - waning_threshold) / (immunity_days × fpd)
frames  = simulation_days × fpd   （或直接 simulation_frames）
```

- **legacy（v1，无自然史）路径**：`T_eff = T_symp + m × T_inc`，其中
  `m = presym_frac × T_symp / ((1-presym_frac) × T_inc)`；医疗阈值用
  `medical_threshold_per_10k = beds_per_1000 × 10 / hospitalization_rate`。
- **自然史路径**：`T_eff` 由传染力曲线数值积分得到（见 4.2），并启用
  无症状/隔离/床位机制。

### 4.2 自然史派生

设 `a = asymptomatic_fraction`、`ifr` 为总感染病死率、`ifr_hosp` 为症状者住院比例：

```
cfr_symp      = ifr / (1 - a)                                  # 症状者病死率
p_ward_death  = cfr_symp × hospital_death_share / hospitalization_rate
p_comm_death  = cfr_symp × (1 - hospital_death_share) / (1 - hospitalization_rate)
untreated_base = p_ward_death（若 >0），否则 cfr_symp
p_untreated   = min(0.95, untreated_base × untreated_death_mult)   # 需住院但无床位

comm_frames             = max(1, T_symp)       # 社区病程帧数
ward_frames             = max(1, ward_days × fpd)
community_death_hazard  = p_comm_death / comm_frames
community_recovery_rate = (1 - p_comm_death) / comm_frames
ward_death_hazard       = p_ward_death / ward_frames
ward_recovery_rate      = (1 - p_ward_death) / ward_frames
untreated_death_hazard  = p_untreated / ward_frames

T_eff = ∫ [ (1-a) × sym_curve(t) × isolation_factor(t)
          + a × asym_curve(t) ] dt          # 逐帧数值积分
isolation_factor(t) = 1                                    (t 未到隔离延迟)
                    = (1 - p_iso) + p_iso × iso_mult       (已到隔离延迟)
presym_fraction = (症状前积分) / T_eff      # 写回派生结果
```

- `sym_curve(t)`：E/I 阶段按 `infectivity_curve` 插值（相对症状出现天数）；
  积分窗口 `sym_lo = max(curve[0].day, -T_inc/fpd)`、
  `sym_hi = max(sym_lo + 0.1, min(curve[-1].day, T_symp/fpd))`。
- `asym_curve(t)`：A 阶段按 `asymptomatic_infectivity_curve` 插值，窗口
  `[max(0, curve[0].day), min(curve[-1].day, asymptomatic_days)]`；
  仅当 `a > 0` 且 `asymptomatic_days > 0` 时计入。
- 派生时校验 `r0_check = C × T_eff × p_symp` 等于目标 R0（严格相等），
  `cfr_symp/p_ward_death/p_comm_death` 必须 < 1，`T_eff > 0`，beta 落在变异
  边界内；任何不一致都会报 `ProfileError` 并以非零码退出。

`untreated_death_mult` 放大的是**本应住院者的死亡风险**（相对 `p_ward_death`），
不是普通社区病程：床位挤兑时，需要住院却无床位的人按更接近重症的风险死亡，
因此容量不足时总病死率会高于名义 IFR，而不是低于它。

### 4.3 社会响应派生（v3）

`_derive_interventions` 把 JSON 的"天"换算成帧，关闭时返回全默认值并置
`enabled=false`：

```
enabled                       = interventions.enabled（默认 True）
test_delay_frames             = test_delay_days × fpd
test_isolation_delay_frames   = isolation_delay_days × fpd
trace_delay_frames            = trace_delay_days × fpd
quarantine_days_frames        = quarantine_days × fpd
behavior_response_delay_frames= response_delay_days × fpd
reporting_delay_frames        = reporting_delay_days × fpd
其余概率/乘子/阈值直接透传
```

- 每个子块 `enabled=false` 时该块不写入（子开关为 false）；总开关关闭时全部不写入。
- 应用阶段：`TESTING_ENABLED = testing.enabled AND interventions.enabled`，
  `TRACING_ENABLED` / `BEHAVIOR_ENABLED` / `REPORTING_ENABLED` 同理；
  `INTERVENTIONS_ENABLED` 要求运行时 `NATURAL_HISTORY_ENABLED`（校验失败会报错）。
- 检测能力在**初始化时**按人口缩放（不是派生阶段）：
  `test_capacity_day = round(test_capacity_per_100k_per_day × N / 100000)`。
- 检测/追踪/行为/报告的运行逻辑（每日排程、延迟、依从、乘子应用位置）
  见 [MECHANISMS.md](../MECHANISMS.md) §7。

随后主程序按 `DISEASE_POPULATION` 缩放：`INITIAL_SEEDS`、床位总数
（`beds_per_1000 × N / 1000`）、医院数量、检测能力、旧版 `MEDICAL_THRESHOLD`
乘以人口比；`beta/gamma/mu` 与人口无关。顺序必须是**先加载档案，再人口缩放**。

## 5. COVID-19 档案的取值与来源

`covid19.json` 采用 2020 年祖先株的近似值，每项带 `range` 与 `sources`（含 DOI/URL）：

| 输入 | 取值 | 来源 |
| --- | --- | --- |
| R0 | 3.0（2.5–3.5） | Liu et al. 2020, J Travel Med（汇总均值约 3.28） |
| 潜伏期 | 均值 5.5 天、sd 2.3、最短 1 天 | Lauer et al. 2020, Ann Intern Med（中位 5.1，均值 5.5，97.5% 在 11.5 天内） |
| 感染期 | 7 天（5–10） | He et al. 2020, Nat Med（病毒学/传播动力学） |
| IFR | 0.66%（0.39–1.33%） | Verity et al. 2020, Lancet Infect Dis |
| 症状前传播 | 0.4（0.3–0.5） | He et al. 2020（约 44% 发生在症状前） |
| 有症状比例 | 0.6（0.4–0.8） | Oran & Topol 2020 / CDC 估计（无症状比例约 40%） |
| 免疫保护 | 180 天（90–365） | Bobrovitz et al. 2023, Lancet Infect Dis（混合免疫 waning） |
| 床位 | 3/1000（1–5） | WHO/World Bank WDI via Our World in Data |
| 住院率 | 0.08（0.05–0.12） | WHO/CDC 工作口径 |

`natural_history` 中的行为/临床参数（隔离概率 0.6、隔离延迟 1.5 天、
隔离传播乘子 0.15、住院延迟 5 天、住院病程 7 天、医院死亡占比 0.7、
无床位死亡倍数 2.0）是**机制性假设**，不是精确文献值；传染力曲线形状参考
He et al. 2020 的"症状前/发病初高峰"定性结论。年龄别相对表参考 Verity et al.
2020 的 IFR 年龄梯度，并做 9 档粗化。

v3 的 `interventions` 是**社会响应情景参数**（不是流行病学估计值）：

| 块 | 取值 |
| --- | --- |
| testing | 有症状者 60% 愿意检测、1.5 天后出结果（360 帧）、能力 800/10 万/日、灵敏度 90%、阳性依从隔离 80%、隔离延迟 0.5 天（120 帧） |
| tracing | 阳性者 60% 触发追踪、延迟 1 天（240 帧）、同户全量 + 工作/学校 25%、隔离 10 天（2400 帧）、依从 80%、传播乘子 0.5 |
| behavior | 强度 0.35、阈值 20/10 万（7 天报告发病率）、斜率尺度 50、响应延迟 7 天（1680 帧）、疲劳 90 天/强度 0.6、乘子下限 0.2 |
| reporting | 有症状发病 50% 报告、延迟 3 天（720 帧），只报告有症状 |

派生结果（当前，C=24.6）：

```
frames = 10800（45 天）
beta = 0.0126317, gamma = 5.95238e-4（感染期 1680 帧）, mu = 7.90934e-5
incubation = 1320 帧（sd 552, min 240）
自然史：a=0.40（无症状 6 天）、隔离 60%/延迟 1.5 天/传播×0.15、
        住院延迟 5 天、病房 7 天、医院死亡占比 0.7、无床死亡×2.0
症状者病死率 cfr_symp ≈ 0.011；p_ward_death ≈ 0.09625、p_comm_death ≈ 0.00359、
p_untreated ≈ 0.1925
有效传染期积分 T_eff = 1074 帧（4.47 天）→ 名义 beta 使 R0=3
平均代际时间 ≈ 1089 帧（4.5 天）
R0 解析校验 = 3.000（严格自洽）
Euler-Lotka 解析倍增时间 ≈ 2.3 天（核近似）
```

v3 不改变 `epidemiology → beta/gamma/mu` 的派生：解析 R0 仍为 3；社会响应是在
传播、检测、行为环节额外乘因子/改变接触概率，因此**仿真中的有效再生数会低于
解析 R0**，这是预期行为，不是档案错误。

v4 的两个块同样不进入 `epidemiology → beta/gamma/mu` 的派生，且被设计成
**R0 期望不变**：`virus_scale` 归一化到均值 1、年龄同配矩阵满足
`Σ_g Σ_h n_g n_h φ(g,h) = 1`。covid19 的 v4 取值为：传染力 σ=1.4（实测方差 ≈3.8，
因 20 上限截断低于理论 6.07）、依从性 0.80±0.15、同户×3.0、年龄同配 0.5、
症状居家 0.6。这些是**先验常数**，不是拟合结果；`covid19_homogeneous.json` 把它们
全部关掉，用于对照"方差效应"（攻击率跨度从 1.6–9.9% 收窄到 5–10%）。

## 6. 校验与校准

```bash
# 1) 解析式检查（秒级）：派生参数、R0 自洽、传染力核/代际时间
python tools/profile_check.py --profile covid19

# 2) 经验检查（约数分钟）：10k 人口跑 6000 帧，拟合早期增长率并报告经验 R0
python tools/profile_check.py --profile covid19 --empirical --frames 6000

# 3) 冒烟（入口直接跑 30 帧）
python simple_disease_simulation.py --profile <name> --frames 30

# 4) 多 seed 量级对照（可选，观察攻击率/住院/超传播）
python tools/ensemble.py --profile <name> --population 10000 --frames 3000 --seeds 1-8 --jobs 4
```

> 历史上的结构断言与逐位金标工具已在 2026-11 移除；
> `benchmarks/` 里的 `golden_*.json` 只作长跑/量级参照。

`--empirical` 的拟合方式：把初始感染改为**随机分散种子**（避免默认工作场所集中
播种放大早期增长率），累计感染数在 `[max(20,0.002N), 0.15N]` 区间内对
`log(累计感染) ~ 帧号` 做最小二乘，得到增长率 r、倍增时间 `ln2/(r·240)`，再用
Euler-Lotka 关系 `1 = R0 · ∫ g(t)·e^{-r·t} dt`（g 为自然史传染力核归一化后的
代际分布）换算经验 R0。容差 ±30%。

注意：`profile_check.py` 不感知 v3，经验校准按"自然史开启、社会响应关闭"路径
运行；v3 档案的仿真轨迹会因检测/追踪/行为而低于该基准，属预期。

当前验证状态（2026-10-31，由 Lead 执行；测试工具已于 2026-11 移除）：

- `profile_check`：`covid19` 解析式 R0 3.00、代际 4.47 天、倍增 2.3 天，PASS；
- 性能基准（重构前口径）见 [SCALING.md](../SCALING.md) §6。

历史记录（重构前、自然史机制、社会响应未启用的一次 45 天运行，10k、10800 帧、
GPU、`--seed 12345`；当时全部断言通过）：

```
末帧 counts = [2282, 0, 359, 816, 36, 6107, 370, 30]   # S/初始免疫/E/I/D/R/A/H
Inf = 1575, Symp = 816, Hosp = 30/30, Unmet = 0, Med: COLLAPSE
累计感染 ≈ 7718（攻击率 77.2%），死亡 36（占已感染 0.47%；结束时有 1575 人
  仍处于 E/I/A/H 病程中，最终死亡会更高，向 0.66% 量级靠近）
峰值同时发病 I = 1314（13.1%，frame 8800）、无症状峰值 971
床位约 frame 8100 起饱和；死亡平均年龄 74.7 vs 全人群 41.1
update 49.1 ms（中位 52.3）/ draw+blit 62.9 ms / CUDA 127.3 MB / wall ≈ 21 分钟
```

这些数字用于确认"无症状分流、隔离、床位挤兑、年龄别严重性"是否按预期影响
宏观轨迹；v3 开启后轨迹会不同（检测/追踪/行为降低传播），不应把上表当 v3 结果。

## 7. 新增一种疾病

1. 复制 `covid19.json` 为 `diseases/<name>.json`，修改 `profile_id`、`label`、`description`。
2. 按第 3 节替换 `epidemiology` 各值；每个数值填 `value`、`range`、`source`。
3. 按疾病设定 `natural_history`（若该病无无症状/住院机制，可把 `asymptomatic_fraction`
   设为 0、`hospitalization_rate` 设为 0，或改用 `model_overrides` 简档）。
4. 需要社会响应时设 `interventions`（v3）：逐块填 `testing`/`tracing`/`behavior`/`reporting`，
   不用的块写 `"enabled": false` 或整块省略；不要引入未知键。
5. **需要个体异质性时设 `heterogeneity` / `mixing`（v4，两块的键与范围见第 3.0 节）**：
   不开就整块省略（等价全关），不要写 `note` 之类未知子键（校验会报错）。
   关闭对照档可参考 `covid19_homogeneous.json`。
6. 在 `sources` 里补 DOI/URL/accessed；`notes` 写清模型结构造成的偏差。
7. 运行 `python tools/profile_check.py --profile <name>` 直到 PASS；必要时用
   `--empirical` 检查经验 R0，再决定是否微调 `r0_calibration_factor`（并在 notes 说明）。
8. 冒烟 + 量级检查：`python simple_disease_simulation.py --profile <name> --frames 300`
   （CLI 覆盖帧数）或 `MPLBACKEND=Agg SIM_FRAMES=300 python simple_disease_simulation.py --profile <name>`；
   然后再跑 `python tools/profile_check.py --profile <name>` 与一次 `tools/ensemble.py` 看量级是否合理。
9. 回归：没有自动化套件；用 `tools/profile_check.py` + 一次短 `ensemble` 观察攻击率/住院量级。
   自建档案不会被任何清单自动纳入；放在 `diseases/` 下即可用 `--profile <name>` 加载。
10. 若疾病时间尺度很短（如流感），可用 `--frames` / `SIM_FRAMES` 或档案的 `simulation_days` 控制运行长度。
11. 如需快速复刻旧行为，用 v1 `model_overrides` 简档（参考 `toy.json`）。

> 逐位基线只用于历史对照；正式测试工具已于 2026-11 移除，现在的自查方式是
> `profile_check` + `ensemble` 量级检查，见 [README](../README.md) 自检段。

## 8. 模型结构限制（档案里已用 notes 标注）

- **有** 无症状/住院状态、自然隔离、年龄别严重性与床位容量；v3 起还有检测、
  追踪、密接隔离、行为反馈与报告层。但仍：
  - 无 ICU/呼吸机/医护人员细分，床位是单一标量容量；
  - 无院内传播；`untreated`（无床位）者居家、不移动，但仍可家庭内传播；
  - 隔离按"传播乘子"近似，自然隔离对家庭内/外接触统一乘，密接隔离另行乘 0.5；
  - 年龄别接触矩阵仍由空间邻近隐式决定（非 POLYMOD 接触矩阵）。
- v3 社会响应是简化机制：只检测有症状自愿者（无无症状筛查/入境检测）、无假阳性
  与复核；追踪只覆盖同户与同工作/学校场所；隔离依从性、行为乘子对全人群相同；
  行为只降低休闲接触与出行；报告只有有症状、固定漏报率。
- 无疫苗、变异免疫逃逸、口罩/学校关闭等 NPI；封控按钮是唯一全局干预。
  `covid19` 的 `model_parameters.mutation.enabled = false`。
- 单标量免疫力：不区分毒株/疫苗/再感染史；再感染严重性不随免疫史变化。
- 疾病时间与行为时钟共用抽象刻度，不做真实日历/地区校准；
  `serial_interval_days` 只做经验对照，不参与派生。
- 攻击率/峰值等"实测数字"是结构参数下的自然结果，不应直接当作真实疫情预测。
