# 机制说明（自然史 / 社会响应 / 医疗容量）

> 文档分工：本文件讲**机制怎么实现**；结构与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，
> 参数与缩放见 [SCALING.md](SCALING.md)，档案字段见 [diseases/README.md](diseases/README.md)。

本文档说明当前代码（`simple_disease_simulation.py` 薄入口 + `disease_sim/` 模块 +
`diseases/*.json` 档案）实现的机制：状态机、档案派生、自然史、年龄别严重性、
住院床位、时变传染力、**schema v3 的社会响应层**（检测、追踪隔离、行为反馈、报告），
以及 **schema v4 的个体异质性与接触混合**（传染力超传播、依从性差异、家庭接触乘子、
4 组年龄同配、症状居家）。所有机制只追求"结构上像真实疾病"，不做真实疫情曲线拟合；
`toy` 档案走 legacy 路径，其 300 帧 counts 在 v4 前后逐位不变。

代码已从单文件拆分为模块（模块名与公开函数名已定稿）：

| 机制 | 实现位置 |
| --- | --- |
| 入口/装配 | `simple_disease_simulation.py`（薄入口）、`disease_sim/__init__.py`（`build_simulation`）、`disease_sim/config.py`（`init_session`/`reset_config`/人口缩放）、`disease_sim/cli.py` |
| 状态初始化 | `disease_sim/state.py`（`build_state`：人口/家庭/场所/v2/v3 状态/日程初始态） |
| 每帧主循环 | `disease_sim/simulation.py`（`update(sim, frame)`） |
| 自然史与医疗 | `disease_sim/simulation.py` + `disease_sim/profiles/derive.py`（`_derive_natural_history`） |
| 社会响应（v3） | `disease_sim/interventions.py`（`_daily_step`/`_process_tests`/`_resolve_traces`/`_advance_quarantine`） |
| 行为反馈应用 | `interventions.py` 计算乘子；`disease_sim/activity.py`（`resample_daily_plan`）与 `simulation.py` 传播循环应用 |
| 活动/出行 | `disease_sim/activity.py`（`_trip_frames`/`_place_spread`/`resample_daily_plan`/`advance_mobility`） |
| 接触对枚举 | `disease_sim/contacts.py`（`_contact_pairs_dense`/`_contact_pairs_grid`） |
| 时变传染力曲线 | `disease_sim/disease.py`（`_curve_symp`/`_curve_asym` 惰性插值表 + `_curve_eval`） |
| 渲染与交互 | `disease_sim/render.py`（`build_render`，`toggle` 挂在 `sim.toggle`） |
| 档案加载/校验/派生/应用 | `disease_sim/profiles/{loader,validate,derive,apply}.py`；根级 `disease_profiles.py` 只是兼容 shim |
| 测量工具 | `tools/profile_check.py`（解析/经验 R0）、`tools/bench.py`（耗时/显存）、`tools/ensemble.py`（多 seed 分布）、`tools/diag.py`（算子/派发归因） |

## 1. 状态与转移

| 状态码 | 名称 | 说明 |
| --- | --- | --- |
| 0 | S 易感 | 可被感染（受免疫力/年龄易感性调制） |
| 1 | 初始免疫 | 开局的免疫者，会随免疫衰减回到 S |
| 2 | E 潜伏 | 已感染、尚无传染力分化（按潜伏期传染力曲线传播） |
| 3 | I 有症状发病 | 有症状、可传播；可能被隔离，或住院 |
| 4 | D 死亡 | 终态 |
| 5 | R 康复 | 获得免疫力，可能随免疫衰减再感染 |
| 6 | A 无症状 | 传染力较低、固定病程、不进医院、不隔离 |
| 7 | H 住院 | 占用床位，不再参与社区传播 |

```
S --感染--> E --潜伏期满--> I（有症状，1-a）
                        \--> A（无症状，a）
I --发病 hospital_delay 天后按年龄概率--> H（有空床） / 回家自愈或死亡（无床则 untreated）
I/A --病程结束--> R 或 D（社区路径，死亡风险×年龄别严重性）
H --病程结束--> R 或 D（住院路径，死亡风险×年龄别严重性）
R/初始免疫 --免疫衰减--> S
```

- E 的传染力按"距症状出现的天数"查症状曲线（对未来发病/无症状者一致）。
- A 的传染力按"感染后天数"查无症状曲线。
- H 不参与社区传播（简化为无院内传播）。
- 自然隔离只影响 I：`isolated=True` 后活动被限制在家（每帧把期望地点设为家，
  在途者改道回家），传播乘 `isolation_transmission_mult`。

### 1.1 社会响应不新增状态码

v3 的检测、追踪、密接隔离都是**标记 + 计时器**，不改变 0–7 状态机：

| 标记/计时器 | 含义 | 效果 |
| --- | --- | --- |
| `wants_test` | 发病时抽到"愿意检测" | 进入检测排队资格 |
| `test_timer` | 检测排队剩余帧数（-1=未预约） | 到期出结果 |
| `tested` | 已出结果 | 每人每次感染最多检一次 |
| `trace_timer` | 阳性后追踪倒计时（-1=未触发） | 到期隔离密接者 |
| `iso_timer` / `isolated` | 病休隔离倒计时/标记 | 居家、传播乘 `isolation_transmission_mult` |
| `quarantined` / `quarantine_timer` | 密接隔离标记/倒计时 | 居家、传播乘 `quarantine_transmission_mult`（仅追踪启用时生效） |

`homebound = isolated | quarantined | status==7 | untreated` 的个体在活动模块中
强制居家。状态 4 的死者不参与任何后续检测/追踪/隔离。

## 2. 疾病档案 schema v4

当前默认档案 `covid19` 为 schema 4；版本历史：v2 = 自然史/医疗容量，
v3 = 加上 `interventions` 社会响应块，v4 = 加上 `heterogeneity`（个体异质性）与
`mixing`（接触混合结构）。下面按 `diseases/covid19.json` 的实际结构给示意
（`epidemiology`/`sources` 省略细节，见 `diseases/README.md`）：

```json
{
  "schema_version": 3,
  "profile_id": "covid19",
  "label": "COVID-19（SARS-CoV-2 祖先株，2020）",
  "description": "……",
  "time": { "frames_per_day": 240, "simulation_days": 45 },
  "population": {
    "initial_immune_fraction": 0.0,
    "initial_immune_level": 0.9,
    "initial_seeds": 20
  },
  "epidemiology": { "r0": { "value": 3.0, "range": [2.5, 3.5], "source": "liu2020" }, "……": "……" },
  "natural_history": {
    "asymptomatic_fraction": 0.4,
    "asymptomatic_days": 6.0,
    "isolation_probability": 0.6,
    "isolation_delay_days": 1.5,
    "isolation_transmission_mult": 0.15,
    "hospital_delay_days": 5.0,
    "ward_days": 7.0,
    "hospital_death_share": 0.7,
    "untreated_death_mult": 2.0,
    "infectivity_curve": [[-6, 0.05], [-3, 0.4], [-1, 0.9], [0, 1.0],
                          [2, 0.8], [5, 0.4], [8, 0.1], [12, 0.0]],
    "asymptomatic_infectivity_curve": [[0, 0.5], [2, 0.6], [5, 0.3],
                                       [8, 0.05], [10, 0.0]],
    "age_ifr_relative": { "0-9": 0.02, "……": "……", "80+": 15.0 },
    "age_hospitalization_relative": { "……": "9 档相对值" },
    "age_susceptibility": { "……": "9 档相对值" }
  },
  "medical": { "beds_per_1000": 3.0, "n_hospitals_per_100k": 10 },
  "interventions": {
    "enabled": true,
    "testing": {
      "enabled": true,
      "test_probability_symptomatic": 0.6,
      "test_delay_days": 1.5,
      "test_capacity_per_100k_per_day": 800,
      "test_sensitivity": 0.9,
      "isolation_adherence": 0.8,
      "isolation_delay_days": 0.5
    },
    "tracing": {
      "enabled": true,
      "trace_probability": 0.6,
      "trace_delay_days": 1.0,
      "trace_household": true,
      "trace_venue_fraction": 0.25,
      "quarantine_days": 10,
      "quarantine_adherence": 0.8,
      "quarantine_transmission_mult": 0.5
    },
    "behavior": {
      "enabled": true,
      "response_strength": 0.35,
      "response_threshold_per_100k": 20,
      "response_scale_per_100k": 50,
      "response_delay_days": 7,
      "fatigue_days": 90,
      "fatigue_strength": 0.6,
      "min_contact_mult": 0.2
    },
    "reporting": {
      "enabled": true,
      "reporting_fraction": 0.5,
      "reporting_delay_days": 3,
      "reporting_symptomatic_only": true
    }
  },
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
  },
  "model_parameters": { "……": "……" },
  "sources": { "……": "……" },
  "notes": ["……"]
}
```

档案规则（由 `profiles/validate.py` 与 `profiles/apply.py` 强制执行）：

- `schema_version` 只能是 1 / 2 / 3 / 4；`interventions` 只在 schema ≥ 3 时允许，
  `heterogeneity` / `mixing` 只在 schema ≥ 4 时允许。
- `interventions` 只允许 `enabled` 与 `testing`/`tracing`/`behavior`/`reporting`
  四个块；各块只允许该块已知键，出现未知键直接报 `ProfileError`。
- `heterogeneity` 只允许 `enabled` / `infectiousness_sigma`（∈[0,3]）/
  `compliance_mean`（∈[0,1]）/ `compliance_sd`（∈[0,0.5]）四个键；
  `enabled = true` 时要求 `natural_history` 与 `epidemiology`。
- `mixing` 只允许 `enabled` / `household_contact_mult`（∈[1,10]）/
  `age_assortativity`（∈[0,0.9]）/ `symptom_stay_home_fraction`（∈[0,1]）四个键。
- 两个 v4 块都可整体省略：省略或 `enabled=false` 时走关闭路径，见 §12.1。
- `interventions.enabled = true` 时档案必须提供 `natural_history` 段；运行时
  `INTERVENTIONS_ENABLED` 也要求 `NATURAL_HISTORY_ENABLED`（否则校验失败）。
- 每个子块可单独写 `"enabled": false` 关闭；总开关关闭时全部子层不生效。
- 档案内时长用「天」，loader 统一乘 `frames_per_day`（240）换算成帧；
  `test_capacity_per_100k_per_day` 的单位是"每 10 万人每日检测数"，运行时按
  人口线性放大（见 §7.2）。
- `--profile` / `--profile=` / 环境变量 `DISEASE_PROFILE` / 默认 `covid19`
  决定加载哪个档案；档案在**人口缩放之前**应用。

## 3. 自然史派生（`_derive_natural_history`）

设 `fpd = 240`（帧/天）、`a = asymptomatic_fraction`、`ifr` 为总感染病死率、
`hospitalization_rate` 为症状者住院比例：

```
cfr_symp      = ifr / (1 - a)                       # 症状者病死率（无症状假设不死）
p_ward_death  = cfr_symp × hospital_death_share / hospitalization_rate
p_comm_death  = cfr_symp × (1 - hospital_death_share) / (1 - hospitalization_rate)
p_untreated   = min(0.95, untreated_base × untreated_death_mult)
                # untreated_base = p_ward_death（若 p_ward_death>0），否则 cfr_symp

comm_frames             = max(1, T_symp)             # 社区病程帧数；T_symp = 感染期 × fpd
ward_frames             = max(1, ward_days × fpd)
community_death_hazard  = p_comm_death / comm_frames
community_recovery_rate = (1 - p_comm_death) / comm_frames
ward_death_hazard       = p_ward_death / ward_frames
ward_recovery_rate      = (1 - p_ward_death) / ward_frames
untreated_death_hazard  = p_untreated / ward_frames
```

- 有效传染期 `T_eff`：把症状曲线（含隔离概率/延迟衰减）与无症状曲线按
  `(1-a)` / `a` 混合，对"感染后每帧"做数值积分；症状前积分 / 总积分给出
  `presymptomatic_curve_fraction`。
  - 症状曲线积分窗口 `sym_lo = max(curve[0].day, -T_inc/fpd)`、
    `sym_hi = max(sym_lo + 0.1, min(curve[last].day, T_symp/fpd))`。
  - 无症状曲线窗口 `[max(0, curve[0].day), min(curve[last].day, asymptomatic_days)]`，
    仅在 `a > 0` 且 `asymptomatic_days > 0` 时计入。
  - 隔离衰减：发病 `isolation_delay` 天之后，曲线期望乘子为
    `(1 - p_iso) + p_iso × isolation_transmission_mult`。
- `p_symp = R0 × r0_calibration_factor / (C × T_eff)`，再
  `beta = (p_symp / transmit_base_prob)^(1 / 1.5)`（与旧路径相同的幂变换）。
  其中 `C = 24.6`（10k 活动模型实测每帧有效接触数）。
- 一致性校验：`r0_check = C × T_eff × p_symp` 必须与目标 R0 严格相等（浮点容差内），
  beta 必须落在变异边界内；`cfr_symp ≥ 1`、`p_ward_death ≥ 1`、`p_comm_death ≥ 1`、
  `T_eff ≤ 0` 都会抛 `ProfileError` 并以非零码退出。

covid19 档案当前派生值（可用 `python tools/profile_check.py --profile covid19`
复算）：`beta = 0.0126317`、`p_symp ≈ 1.1355e-4`、`cfr_symp ≈ 0.011`、
`p_ward_death ≈ 0.09625`、`p_comm_death ≈ 0.00359`、`p_untreated ≈ 0.1925`、
`T_eff = 1074` 帧（4.47 天，= 1073.75 / 240）、R0 校验 `3.000000`、平均代际时间 ≈ 1089 帧（4.54 天）。

## 4. 年龄别严重性

- 初始化时为每个 agent 建立 `age_band`（按 10 岁分段，共 9 档：
  0-9/10-19/…/80+）。
- `severity_mult`（死亡风险相对值）、`hosp_mult`（住院概率相对值）、
  `suscept_mult`（易感性相对值）按年龄表取值后**除以人口加权均值**，保证总体
  与档案名义 IFR/住院率/易感性一致；个体间只保留"老年更危险"的排序。
- 传播时对目标乘 `suscept_mult`；社区/住院死亡判定乘 `severity_mult`；
  住院概率乘 `hosp_mult`。

## 5. 住院与床位机制

- 床位总量 `beds_total = round(beds_per_1000 × N / 1000)`（covid19 3/1000：
  10k→30 床、100k→300 床、1M→3000 床）；医院数量仅用于提示/后续扩展。
- 发病满 `hospital_delay_days` 且有症状的 I 中，按 `hospitalization_rate × hosp_mult`
  抽样是否住院；抽样成功后：
  - 有空床（住院人数 < 床位）→ 状态 7（H）；
  - 无空床 → 标记 `untreated=True`，继续留在家中的有症状状态，死亡风险按
    `untreated_death_hazard`（= `p_untreated / ward_frames`）计算：即"本应住院、
    却没有床位"的重症死亡风险，高于普通社区病程。
- 每帧对"已过住院延迟但仍未入院"的有症状者按发病时长降序补收空床（先到先得）。
- 注：`untreated` 者仍可活动/传播（活动模块将其强制居家）；这是简化，
  未建模"居家隔离但未住院"的接触结构差异。

## 6. 时变传染力

- E/I 用 `infectivity_curve`（相对症状出现天数，负值=症状前）：
  - E：`x = (inf_elapsed - max_incubation) / fpd`（通常为负）；
  - I：`x = symp_elapsed / fpd`；
- A 用 `asymptomatic_infectivity_curve` 的 `x = inf_elapsed / fpd`。
- 隔离后的 I 额外乘 `isolation_transmission_mult`；追踪启用时，密接隔离者
  再乘 `quarantine_transmission_mult`（两个标记同时存在时乘子叠加）。
- 曲线用 `torch.bucketize` + `torch.clamp` 分段线性插值，模块级惰性 LUT 在
  `disease.py`（`_curve_symp`/`_curve_asym`）。

## 7. 社会响应层（schema v3）

### 7.1 开关链与运行前提

- CONFIG 中 v3 参数默认全关（`INTERVENTIONS_ENABLED=false`，四个子开关也 false）。
  档案派生后由 `_apply_interventions` 写入 CONFIG：
  `TESTING_ENABLED = testing.enabled AND interventions.enabled`，
  `TRACING_ENABLED`/`BEHAVIOR_ENABLED`/`REPORTING_ENABLED` 同理。
- `update()` 只在 `IV_ENABLED` 时执行日界事件（`_daily_step`）与密接隔离倒计时
  （`_advance_quarantine`）；检测出结果与追踪到期只在自然史分支内调用。
- 关闭时只做张量初始化与常量判断，**不消耗 torch 随机数**：`toy`（v1）与
  `covid19_v2_noiv`（schema 3、省略 `interventions`）因此保持旧行为。

### 7.2 检测（`TESTING_ON`）

- 日检测能力 `test_capacity_day = round(test_capacity_per_100k_per_day × N / 100000)`，
  随人口**线性放大**：800/10 万/日 → 10k 为 80 人/日、100k 为 800、1M 为 8000。
- 发病时（E→I 分流处）若 `TESTING_ON`，按 `test_probability_symptomatic`
  抽 `wants_test`；只有愿意检测的有症状者可能被排到。
- 每天日界排程：`eligible = status==3 & wants_test & ~tested & ~isolated &
  test_timer<0`；按已发病时长（`symp_elapsed`）降序取前
  `min(当日能力, 合格人数)`，置 `test_timer = max(1, test_delay_frames)`。
- 检测出结果（`_process_tests`）：`tested=True`；按 `test_sensitivity` 判定阳性；
  阳性者按 `isolation_adherence` 决定是否隔离，依从者若尚无自然隔离排队
  （`iso_timer<0`），置 `iso_timer = max(1, isolation_delay_frames)`；
  若 `TRACING_ON`，阳性者再按 `trace_probability` 触发
  `trace_timer = max(1, trace_delay_frames)`。
- 每人每次感染最多检一次（`tested` 在再感染时重置）；无常规/无症状筛查、
  无复核检测、无假阳性；当日能力用不完即作废，不结转。

### 7.3 追踪与密接隔离（`TRACING_ON`）

- 追踪只由**检测阳性者**触发（自然隔离者不触发）；延迟 `trace_delay_days` 后
  由 `_resolve_traces` 展开接触者：
  - `trace_household=true` 时纳入同户全部成员；
  - 工作/学校同场所成员按 `trace_venue_fraction` 比例抽为接触者。
- 接触者是否被隔离 = `rand < p_contact × quarantine_adherence`，其中
  `p_contact` 家庭为 1、场所为 `trace_venue_fraction`；排除本人、死者、
  已隔离者与住院者（状态 7）。
- 被隔离者 `quarantine_timer = quarantine_days × fpd`，期间强制居家
  （`homebound`），传播乘 `quarantine_transmission_mult`；到期自动解除。
- 密接隔离与自然隔离/检测隔离是两套标记；同时存在时两个传播乘子叠加。
  只有密接隔离乘子受 `TRACING_ON` 门控；自然隔离乘子随自然史启用。

> **已修复（2026-11 全量审计）**：原缺陷有两层——`trace_timer` 只在阳性时被**赋值**
> （`max(1, trace_delay_frames)`）、在死者/康复者处清 `-1`，但**没有任何每帧递减**；
> 且 D.5 的门控统计的是 `trace_timer == 0`，而计时器从 ≥1 起，于是门控恒 0。
> 两层叠加使 `_resolve_traces` 从不执行 → v3 追踪与密接隔离在仿真中不生效
> （`n_quarantined` 恒为 0；旧 12-seed 结果里"隔离 0"即由此而来）。
>
> 现在与 `test_timer`/`_process_tests` 完全对称：F 段改为统计 `trace_timer > 0`
> （字段 `_n_trace_pending`，语义="下一帧待递减的追踪数"），`_resolve_traces` 入口先
> `sim.trace_timer.add_(sim.trace_timer > 0, alpha=-1)` 再取 `== 0` 展开。
> **行为影响**（covid19 10k，3000 帧）：隔离总量 p10/p50/p90 0 → **316/346/467**，
> 攻击率 p50 0.050 → **0.040**（追踪压低了传播），二代 var/mean p50 2.32 → **2.62**
> （k 中位 1.31，超传播仍显著）；`toy` 与 `covid19_v2_noiv` 关闭路径计数不变。

### 7.4 行为反馈（`BEHAVIOR_ON`）

- 日界计算一次行为乘子（`_daily_step`）：
  1. 响应延迟 `delay = round(response_delay_frames / fpd)`（按天取整）；
  2. 取 `day - delay` 之前最近 7 天的"发病/报告"计数窗口：报告开启时用
     `report_events`（已报告病例），报告关闭时用 `onset_by_day`（有症状发病）；
  3. `incidence = window / N × 100000`；
  4. `risk = sigmoid((incidence - response_threshold_per_100k) / response_scale_per_100k)`；
  5. `fatigue = 1 / (1 + fatigue_strength × max(0, day_index - delay) / fatigue_days)`；
  6. `mult = 1 - response_strength × risk × fatigue`，clamp 到
     `[min_contact_mult, 1]`；`end <= 0`（尚未到响应时间）时 `mult = 1`。
- 应用位置（**只影响休闲接触与出行，不影响工作/学校/家庭**）：
  - 传播循环中，只有**双方都在休闲场所（`PLACE_LEISURE`）**的接触对乘 `mult`；
  - 每日出行计划重采样时，外出概率乘 `mult`（工作日 0.35 / 周末 0.55 被按比例压低）；
  - 上班/上学通勤与工作/学校/家庭内接触不缩放。
- 乘子对所有 agent 是**同一个标量**（无个体依从性差异）；疲劳按仿真绝对日数
  累积（从第 delay 天起），与疫情是否持续无关——这两点是已知简化。

### 7.5 报告（`REPORTING_ON`）

- 日界：对 `delay_days = round(reporting_delay_frames / fpd)` 天前的
  `onset_by_day` 计数，抽 `Binomial(该日发病数, reporting_fraction)` 作为
  `reported_today`，累加进 `reported_total` 与 `report_events`。
- 报告只统计**有症状发病**（`onset_by_day` 本身就是有症状转归计数），因此
  `reporting_symptomatic_only` 当前在运行时代码中只作为声明保留；
  无症状者、住院、死亡均不进报告口径。
- 报告数供行为层构造窗口；标题显示累计报告数 `Rep`。

### 7.6 计数与显示

`IV_ENABLED` 时标题第三行追加：

```
| Q {quarantined_count} Test {tests_total} Rep {reported_total} | mult {behavior_contact_mult:.2f}
```

其中 Q = 当前密接隔离中人数，Test = 累计检测出结果人数，Rep = 累计报告数，
mult = 当前行为接触乘子（两位小数）。另有内部计数器
`tests_today`/`tests_total`/`quarantine_total`/`onset_total`/`reported_total`
供工具读取。

### 7.7 covid19 档案的 v3 参数（实际值）

| 块 | 参数 | 值 | 含义 |
| --- | --- | --- | --- |
| testing | `test_probability_symptomatic` | 0.6 | 有症状者发病时愿意检测的概率 |
| testing | `test_delay_days` | 1.5 | 预约到出结果的延迟（360 帧） |
| testing | `test_capacity_per_100k_per_day` | 800 | 每 10 万人每日检测能力 |
| testing | `test_sensitivity` | 0.9 | 灵敏度 |
| testing | `isolation_adherence` | 0.8 | 阳性后居家隔离依从性 |
| testing | `isolation_delay_days` | 0.5 | 阳性到隔离的延迟（120 帧） |
| tracing | `trace_probability` | 0.6 | 阳性者触发追踪的概率 |
| tracing | `trace_delay_days` | 1.0 | 追踪延迟（240 帧） |
| tracing | `trace_household` | true | 家庭接触者全量纳入 |
| tracing | `trace_venue_fraction` | 0.25 | 工作/学校接触者纳入比例 |
| tracing | `quarantine_days` | 10 | 密接隔离时长（2400 帧） |
| tracing | `quarantine_adherence` | 0.8 | 密接受追踪后隔离依从性 |
| tracing | `quarantine_transmission_mult` | 0.5 | 密接隔离者传播乘子 |
| behavior | `response_strength` | 0.35 | 最大接触降幅 |
| behavior | `response_threshold_per_100k` | 20 | sigmoid 中心（每 10 万人 7 天报告发病率） |
| behavior | `response_scale_per_100k` | 50 | sigmoid 斜率尺度 |
| behavior | `response_delay_days` | 7 | 报告到行为响应的延迟 |
| behavior | `fatigue_days` | 90 | 疲劳时间尺度 |
| behavior | `fatigue_strength` | 0.6 | 疲劳强度 |
| behavior | `min_contact_mult` | 0.2 | 乘子下限 |
| reporting | `reporting_fraction` | 0.5 | 有症状发病被报告比例 |
| reporting | `reporting_delay_days` | 3 | 发病到报告的延迟（720 帧） |
| reporting | `reporting_symptomatic_only` | true | 只报告有症状（当前实现天然如此） |

## 8. 与 legacy / 关闭社会响应的关系

代码有三条路径，由档案决定：

| 档案 | schema | 自然史 | 社会响应 | 异质性/混合 | 用途 |
| --- | --- | --- | --- | --- | --- |
| `covid19` | 4 | 开 | 四层全开 | 开 | 默认演示（含超传播与年龄同配） |
| `covid19_homogeneous` | 4 | 开 | 四层全开 | **关** | v4 的同质对照（隔离"异质性"本身的效果） |
| `covid19_v2_noiv` | 3（省略 `interventions`） | 开 | 全关 | 关（未声明） | 自然史机制回归（等价于加 v3 之前的行为） |
| `toy` | 1 | 关（legacy 路径） | 关 | 关（未声明） | 快速演示、统计回归与历史对照 |

- `toy` 走 legacy 路径时：状态机仍是旧的 6 状态（A/H 永不出现），旧潜伏/发病/
  死亡康复路径代码不变，随机数调用序列不变 → 旧基线逐位一致；统计 `counts`
  长度为 8 但 `[6]/[7]` 恒为 0。
- 所有 v3 随机抽取都放在 `IV_ENABLED` / 子开关为 True 的分支内；关闭时
  不建检测/追踪随机流。
- 关闭社会响应但开启自然史（`covid19_v2_noiv`）：行为与自然史代码路径一致，
  因不执行任何 v3 分支，RNG 序列与加 v3 前相同；历史快照用它做对照档。

## 9. 已知简化与偏差（相对真实模型）

- 无疫苗、变异/免疫逃逸、口罩/学校关闭等 NPI；封控按钮是唯一非药物干预。
  档案的 `model_parameters.mutation.enabled` 在 covid19 中为 false。
- 无院内传播、ICU/呼吸机细分、医护人员；床位是单一标量容量。
- 无年龄别接触矩阵：v4 只加了 4 组粗年龄同配与家庭接触乘子，接触仍由活动模型的
  空间邻近决定（家庭/工作/学校/休闲混合），没有通勤矩阵或真实 POLYMOD 矩阵。
- 检测只覆盖"有症状且愿意检测"的人：无常规筛查、无无症状检测、无入境/复工检测；
  无假阳性、无复核流程；检测能力按人口线性放大是建模选择，不是政策现实。
- 追踪只由检测阳性触发；只纳入同户与同工作/学校场所成员，不含随机社交网络、
  交通工具、医院等接触；隔离依从性对所有人相同。
- 行为反馈只有休闲接触与出行一个标量乘子：不区分年龄/职业/意愿，不建模
  口罩、社交距离政策、经济停摆；疲劳按绝对日历日累积。
- 报告只有有症状发病一个口径，无年龄/严重程度分层，也无报告修正（漏报率固定）。
- 单标量免疫：不区分毒株/疫苗/再感染史；再感染严重性不随免疫史变化。
- 疾病时间与行为时钟共用抽象刻度（1 帧 = 6 分钟，按天换算），不做真实日历/
  地区校准；社会响应的所有"天"都按 240 帧/天换算。

已知未完成项（如实列出）：

- 重构后大人口（触发 `grid` 接触路径）的性能基准仍由 Lead 统一补齐
  （见 [SCALING.md](SCALING.md) §6.2）；10k 默认走稠密路径，不受影响。
- `tools/profile_check.py` 不感知 v3：它仍按自然史（无社会响应）路径做解析与
  经验校准，因此 covid19 仿真轨迹低于其经验 R0 基准属预期，不代表文档错误。
- 这些一致性检查是结构级校验，不是 v3 参数校准。

## 10. 自检命令

项目不含测试套件；验证方式是跑一次冒烟、看派生值、看多 seed 分布：

```powershell
python simple_disease_simulation.py --profile toy --frames 30       # 入口冒烟
python tools/profile_check.py --profile covid19                   # 解析式：beta/R0/代际/倍增
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 `
    --seeds 1-12 --jobs 4 --json benchmarks/ensemble.json         # 多 seed 分位带
python tools/bench.py --profile covid19 --frames 400 --label after  # 性能口径
```

`benchmarks/` 里的 `golden_*.json` 是历史快照（采集于性能优化前的版本），
只作长跑与量级参照；性能口径见 [SCALING.md](SCALING.md)。


## 11. 工程结构（重构后）

机制实现不再集中于单文件，而是按职责分布在 `disease_sim/` 包中（入口
`simple_disease_simulation.py` 仅做装配与历史接口转发）：

| 机制 | 实现位置 |
| --- | --- |
| 档案 schema 校验 / 派生 / 应用 | `disease_sim/profiles/{schema,loader,validate,derive,apply}.py` |
| 状态码与转移（SEIR + A/H） | `disease_sim/simulation.py`（D 段），状态容器 `disease_sim/state.py` |
| 时变传染力曲线 | `disease_sim/disease.py`（懒构建 LUT） |
| 年龄别严重性 / 床位 | `disease_sim/profiles/derive.py` + `disease_sim/simulation.py` |
| v3 检测/追踪/行为/报告 | `disease_sim/interventions.py` |
| 接触对枚举（dense/grid） | `disease_sim/contacts.py` |
| 出行与场所 | `disease_sim/activity.py` |
| 标题/曲线/散点更新 | `disease_sim/simulation.py`（F 段）+ `disease_sim/render.py` |

依赖方向与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。重构当时不改变任何数值语义：
用原始单文件实现做初始化张量、逐帧 counts、标题、RNG 状态、CONFIG 的全套逐位比对，
covid19 3000 帧（12.5 天）完全一致（性能优化轮后不再逐位，见 §11.1）。

长跑金标（covid19，10k，2026-10-31 采集，**采集于性能优化之前**）：3000 帧末帧
`[9537,0,312,61,0,46,44,0]`；10800 帧（45 天）末帧 `[2405,0,422,779,30,5970,364,30]`
（累计感染 75.95%，床位 30/30 满，`Med: COLLAPSE`，免疫 0.357，再感染率 16.7%）。

### 11.1 性能优化对"逐位"的影响（2026-10-31 晚，S1/S3）

在结构重构完成之后又做了第二轮性能优化，它们**改变了浮点运算顺序，因此不再逐位复现**
（机制语义未变，转移规则/曲线形状/概率/年龄表/床位规则/v3 四层都保持原样）：

- `disease.py`：分段线性曲线改为 0.25 天步长的 dense LUT + 两次 1D gather + lerp。
  与逐段插值的最大偏差 1.2e-7（曲线折点都是整数天，网格与折点对齐）——因此
  "距发病天数 → 传染力"的映射在数值上等价，但末位可能不同。
- `simulation.py`：统计段把 13–24 个标量合并成一个向量，整帧只做一次
  `.cpu()` 同步；均值由 `arr[mask].mean()` 改为 `(arr*mask).sum()/count`
  （避免掩码索引内部的隐式 `nonzero` 同步，数学上同一个均值）。
- `activity.py`：掩码写改整幅 `torch.where`/`masked_fill_`，行程诊断容器按日聚合
  并封顶（不再每帧 `tolist()` 泄漏主机内存）。
- 视觉仍遵守原 contract：15 个 artist、`COLOR_LUT` 8 色、标题三段同样的字段与
  小数位、曲线 500 点滚动窗口；散点抽样上限由 20000 调到 10000（大人口下仍是
  固定随机子集，统计/题目数据永远是全体人口）。

因此验收口径改为统计式：分配守恒（`sum(counts)==N`）、跨 seed 攻击率/病死率量级、
同 seed 重跑一致；`golden_*.json` 保留为历史快照与长跑量级参照。

### 11.2 第三轮优化（Wave 1：整幅化 / 索引复用 / 同步消除）

在同一口径下又做了一轮"减少每帧 kernel 数与主机同步"的实现替换。**机制语义完全不变**
（状态迁移、概率、曲线形状、年龄表、床位规则、v3 四层、可视化 contract 均未改），
但浮点累加顺序与随机数流允许变化（前一轮已取消逐位验收）：

- **活动模型**：停留段由"取停留索引 + 3 次 gather + 2 次掩码写"改成整幅 `where`；
  位置 clamp 合并成一次 elementwise；旅途推进用 `torch.lerp`；`desired`/`traveling`/
  `trip_count` 走预分配缓冲与原地写（`torch.add_(bool)` 由类型提升完成）。
  唯一保留的索引操作是随机游走所需的 `stidx`——因为 `torch.randn((n,2))` 的形状
  必须与"当前停留人数"一致。
- **病程/传播**：新增 8 元素 `_inf_lut`（状态→是否具传染性），一次 gather 取代
  3–5 个比较/或；住院"空床数"改为主机侧记账（帧首住院数 − 本帧病房死亡 + 本帧新收），
  省掉每帧两次 `(status==7).sum().item()` 同步；`p_pair` 先把感染者侧乘子合成一个
  系数再广播到 pair（少 2 次 gather、2 次乘法）；每帧临时 `torch.tensor(...)` 常量
  改用标量版 `torch.where`。
- **统计段**：状态先转 `uint8` 再拷回（1M 下 8 MB→1 MB）；再感染率前缀和只算
  显示窗口所需的尾部（长跑不再每帧对全历史 O(T) 累加）。
- **门控**：v3 的检测/追踪/隔离倒计时用上一帧末的主机计数整块跳过；传播段用
  "帧首无感染者"门控（无感染者不可能产生新感染，精确等价）。

实测（同进程交替 A/B，`tools/ab_iter.py`，配对中位比，噪声地板 ±3%）：covid19 10k
−11.8%、covid19 1M −14.7%、toy 10k −4.0%（toy 两侧 RNG 分叉导致末帧工作集不同，
可比性差）；aten 事件数/帧 978 → 863（−11.8%）。详见 [SCALING.md](SCALING.md) §6.2c。

### 11.3 第四轮优化（Wave 2：算子瘦身 / 接触对 / 渲染）

用户在本轮前明确**不再要求逐位一致**，因此可以删掉"模型自己从不读取"的写操作，
并按 CPU 派发成本（每算子 9–13 µs，见 SCALING §6.7）继续削 kernel 数。机制语义不变：

- **不可观测写**：死亡/康复个体原先会把 5 个 `virus_beta/gamma/mu/sigma/gen` 字段清零，
  但传染力只按 E/I/A 索引读取、病毒均值只对"当前感染者"求、再感染时 transmission 会
  全量覆写这些字段，也没有任何代码读取它们。因此这段清零整体删除
  （疫情期每帧 20–45 次 `index_put`）。
- **批量变异**：三组 `beta/gamma/mu` 的"clone → 掩码随机 → 加性偏移 → clamp → 写回"
  合并成一个 `(3, n)` 张量运算（`stack → rand → delta → clamp_ → 3 次写回`），
  scale/bounds 常量首次调用时按应用档案后的 CONFIG 缓存。
- **免疫增益**：`status.masked_fill_` + `immunity_level.index_add_`/`add_` + `clamp_`
  取代 where 链与逐字段掩码写。
- **系数折叠**：`p_pair` 的感染者侧乘子先合成 `coef = beta_factor*status_mult*(BASE*lockdown)`，
  再对 pair 广播，少 2 次 gather / 2 次乘法。
- **v3 索引化**：检测出结果、追踪展开、隔离倒计时原来都用 `bool(x.any())` 判空 +
  整幅 N 布尔（`sel = zeros(N)`）；改为 `nonzero` + `index_select` + `index_put`，
  既少 kernel 也少主机同步。`_daily_step` 用一次 `sum().item()` 同时当判空与名额。
- **接触对 grid**：目标网格加 1 格外边距，越界邻格自然落到计数为 0 的 padding 格，
  因此不再需要越界掩码（省 4 比较 + 3 与 + 1 cast），邻格步长按 `gxp` 缓存。
  另做过 5 组构造状态校验：dense 与 grid 的 `(src,tar)` 集合完全一致、grid 无
  重复对——这是"按人口切换实现"的正确性保证。
- **渲染**：主散点 marker 由圆形改为小方块（`s=2` 时顶点数约 1/6，Agg 绘制
  37.3 → 22.2 ms/draw）；新增 `--async-render[=K]`（模型逐帧、每 K 帧重绘，默认 K=2）。

实测：covid19 10k **−15.0%**、1M **−8.3%**（配对中位比）；内核数 877 → 833；
1M 接触段 4.216 → 3.617 ms（−14.2%）；10k 交互 amortized 47.4 → 30.6 ms（K=1）、
20.9 ms（K=2，47.9 fps）；当时全量回归 6/6 PASS（v4 前口径）。详见
[SCALING.md](SCALING.md) §6.2d。

## 12. 个体异质性与接触混合（schema v4）

v4 的动机不是"加更多流行病学机制"，而是补上**异质性（方差）**：均匀混合、
Poisson 型传播几乎不会"熄火"，也长不出超级传播者，而这恰是本类 toy model 的教学点
——同样的 R0，不同的个体差异会导致"有的种下去就灭、有的炸开"以及"同样的干预效果
在不同人群里差很多"。v4 只加三个最小结构：传染力个体异质性、依从性个体异质性、
粗粒度接触混合（家庭乘子 + 4 组年龄同配 + 症状居家）。

### 12.1 开关链与向后兼容

- 两个新档案块：`heterogeneity`、`mixing`；`SCHEMA_VERSION = 4`，
  `SCHEMA_VERSIONS = (1, 2, 3, 4)`。
- 运行时 8 个 CONFIG 键默认中性关闭：`HETEROGENEITY_ENABLED=false`、
  `INFECTIOUSNESS_SIGMA=0`、`COMPLIANCE_MEAN=1`、`COMPLIANCE_SD=0`、
  `MIXING_ENABLED=false`、`HOUSEHOLD_CONTACT_MULT=1`、`AGE_ASSORTATIVITY=0`、
  `SYMPTOM_STAY_HOME_FRACTION=0`。`toy`(v1) 与 `covid19_v2_noiv`(v3) 不写这两块
  → 走关闭路径。
- 关闭路径**不消耗任何随机数**（`virus_scale`/`compliance`/`symp_home`/`mix_flat`
  的抽取都在 `if sim.HET_ON / sim.MIX_ON` 内），因此 toy 300 帧 counts 在 v4 前后
  逐位相同：`[18,102,820,6659,1632,769,0,0]`。
- 所有 v4 参数都是**先验选择**，未做数据拟合。

### 12.2 传染力个体异质性（`heterogeneity.infectiousness_sigma`）

- 每个 agent 一个固定乘子 `virus_scale`：初始化抽
  `exp(N(0, σ²) − σ²/2)`（均值 1 的对数正态），clamp 到 `[0.05, 20]`，再按样本均值
  归一化（保证总体 R0 期望不因抽样噪声漂移）。
- 传播时乘在感染者侧：`coef = beta_factor × status_mult × (BASE × lockdown) ×
  virus_scale[infector]`；每个新感染者重新抽自己的乘子（超传播者在整个传染期持续超传播）。
- σ=1.4 → 理论 Var = e^{σ²}−1 ≈ 6.07；实现里 20 上限截断（约 0.23% 上尾）再归一化后
  实测约 3.8（曾用区间断言锁定，测试套件已移除）。
- 实测效果（covid19，10k，12 seeds × 3000 帧，审计修复后复跑）：
  - 跨 seed 攻击率 **1.3% – 9.2%**（同质对照 5.3% – 10.4%），出现"熄火/爆发"分化；
  - 传染期已结束感染者的二代分布 **var/mean p50 2.62**（对照 1.21，接近泊松），
    k ≈ 1.31，零二代比例 24.9%（对照 8.6%）。
- **统计口径提醒**：二代过散布只能统计"传染期已结束"（不在 E/I/A）的感染者；若把
  仍在传染期的人也计入，尾部截断会把方差压平（全部感染者口径下过散布被高估到 ~2.5，
  同质对照也 >2），这是 `ensemble --track-secondary`
  都采用 completed 口径的原因。

### 12.3 依从性个体异质性（`compliance_mean` / `compliance_sd`）

- `compliance ~ clamp(N(mean, sd²), 0, 1)`，是个体固定属性（不随时间/疲劳变化）。
- 乘在四类"需要人配合"的概率上（都只是改阈值，**不改变随机数抽取个数**）：
  - 自然隔离抽签：`ISOLATION_PROBABILITY × compliance[onset_idx]`；
  - 检测阳性后隔离：`TEST_ISOLATION_ADHERENCE × compliance[pos_idx]`；
  - 密接追踪隔离：`p_take × compliance[agents]`；
  - 行为反馈：出门概率 `BASE × (1 − compliance × (1 − behavior_contact_mult))`。
- 关闭时 `compliance ≡ 1` → 与 v3 完全一致。

### 12.4 接触混合结构（`mixing`）

- **家庭乘子** `household_contact_mult`（covid19=3.0）：同户接触对的传播概率乘该系数。
- **4 组年龄同配**：child 0–19 / adult 20–49 / older 50–69 / senior 70+（由 9 档
  `age_band` 映射 `[0,0,1,1,1,2,2,3,3]`）；`φ(g,h) = 1 − s + s·1{g=h}/n_g`
  （`s` = `age_assortativity`，`n_g` 为组占比下限 0.02）。归一化性质
  `Σ_g Σ_h n_g n_h φ(g,h) = 1` → **总体 R0 期望不变**（R0 仍 3.00）。
- **症状居家** `symptom_stay_home_fraction`（covid19=0.6）：抽中的感染者一旦进入
  有症状（状态 3）就自动居家，进入 `homebound`，同时把 `_n_restricted` 计数加上当日 I 数
  （这是"确诊前的自发行为反应"，与检测/隔离相互独立）。
- pair 级额外乘子：`where(same_hh, hh_mult, 1) × mix_flat[g_src×4 + g_tar]`。

### 12.5 观测层：多 seed ensemble（`tools/ensemble.py`）

- 每个 seed 一个独立子进程（隔离档案解析与 CUDA 上下文），汇总 p10/p50/p90、
  可选 CSV 与分位带 PNG；指标：攻击率、峰值感染/住院、峰值日死亡、达峰天数、
  累计感染/死亡/检测/隔离/报告、单帧耗时、显存峰值。
- `--track-secondary` 打开逐感染者二代计数（默认关闭，环境变量
  `SIM_TRACK_SECONDARY`），同时给出 completed 与 all 两个口径。
- 典型用法见 §10；输出 JSON 结构 `{meta, seeds[], summary{}}`。

### 12.6 v4 的已知简化

- 传染力异质性是**个体固定乘子**：不建模场景驱动的超级传播事件（聚会/医院/交通工具），
  也不让传染力随时间变化；k 只是 `infectiousness_sigma` 的直接后果。
- 依从性是固定个体属性，不随疫情阶段/信息变化；疲劳仍由全局 `behavior_contact_mult`
  表达（因此"个体疲劳差异"没有建模）。
- 混合结构只有 4 组年龄 + 家庭乘子，接触对仍由活动模型的空间邻近生成；无通勤矩阵、
  无真实 POLYMOD 矩阵、无场所类型分层（学校/工作/医院）。
- 症状居家只在出现症状后触发；无症状者不减少接触；潜伏期行为不变。
- 参数（σ=1.4、依从性 0.8±0.15、家庭×3、同配 0.5、症状居家 0.6）是先验常数，
  不是拟合结果；v4 不改变"不做绝对预测"的定位。

