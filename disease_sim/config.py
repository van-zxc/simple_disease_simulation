"""配置默认值、会话装配与人口缩放。

配置分层顺序（后写覆盖前写）：_DEFAULTS → 疾病档案 apply（profiles）→ 环境变量/CLI
（SIM_FRAMES、SIM_RENDER_POINTS、DISEASE_SEED、DISEASE_POPULATION）→ 人口缩放
（_apply_population_scale）。

约定：CONFIG 的**字典身份在整个会话内不变**——档案层与各模块都持有同一个 dict 引用
（``from .config import CONFIG``），所以 reset/apply 必须原地 clear+update，绝不能重新
赋值 CONFIG。init_session() 是唯一装配入口，可重复调用（每次先 reset_config）。
"""

import copy
import os
import sys

import torch


from . import profiles as disease_profiles


# 设备在 import 期确定一次：之后所有张量构造/迁移都用它，运行中不切换
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEFAULTS = {
    # 基础参数
    "N_AGENTS": 10000, # 总人口数
    "WIDTH": 100.0, # 区域宽度
    "HEIGHT": 100.0, # 区域高度
    "N_HOTSPOTS": 3, # 热点数量
    "INFECT_RADIUS": 1.8, # 感染距离阈值
    "BASE_TRANSMIT": 0.3, # 基础传播率
    "BASE_RECOVERY": 0.008, # 基础恢复率
    "BASE_DEATH": 0.01, # 基础死亡率
    "MUTATION_RATE": 0.25, # 基础变异率
    "MUTATION_SCALE": 0.15, # 变异幅度
    "INCUBATION_MU": 40.0, # 潜伏期均值
    "MEDICAL_THRESHOLD": 1000, # 医疗系统崩溃阈值
    "TRANSMIT_BASE_PROB": 0.08, # 基础传染概率
    "SEED": 42, # 随机种子（也可用环境变量 DISEASE_SEED 覆盖）

    # 免疫系统参数
    "IMMUNITY_GAIN_PER_INFECTION": 0.4,   # 每次感染获得的免疫力
    "IMMUNITY_DECAY_RATE": 0.0005,        # 免疫力衰减速率（每步）
    "MAX_IMMUNITY": 0.98,                 # 最大免疫力（不会完全免疫）
    "IMMUNITY_EFFECTIVENESS": 3,        # 免疫力效果倍率
    "IMMUNITY_WANING_THRESHOLD": 0.1,   # 免疫低于该阈值后重新变为易感

    # ---- 活动/轨迹模型（行为节律；单位是抽象时间步，不与疾病速率换算）----
    "DAY_FRAMES": 240,                 # 一天 = 240 帧，约 6 分钟/帧
    "CLOCK_START_MIN": 420,            # 第 0 帧对应 07:00
    "WEEKEND_DAYS": [5, 6],            # 周六/周日
    "AGE_CHILD_FRAC": 0.18,            # 儿童占比（现实约 15-25%）
    "AGE_ELDER_FRAC": 0.16,            # 老人占比（现实约 10-20%）
    "ADULT_EMPLOYMENT": 0.65,          # 成人就业率（现实约 60-70%）
    "CHILD_SCHOOL_ENROLL": 0.95,       # 学龄儿童入学率
    "SCHOOL_MIN_AGE": 5,               # 最小入学年龄
    "HOUSEHOLD_SIZE_WEIGHTS": [0.18, 0.30, 0.22, 0.19, 0.11],  # 1-5 人户
    "N_RESIDENTIAL_CLUSTERS": 12,      # 居住区中心数
    "HOME_CLUSTER_STD": 10.0,          # 家庭锚点围绕居住区中心的散布
    "HOME_JITTER_STD": 0.8,            # 同户成员的居住点抖动
    "WORK_START_MIN": 540,             # 09:00 上班
    "WORK_END_MIN": 1050,              # 17:30 下班
    "SCHOOL_START_MIN": 510,           # 08:30 上课
    "SCHOOL_END_MIN": 900,             # 15:00 放学
    "ERRAND_PROB_WEEKDAY": 0.35,       # 工作日外出购物/公园概率
    "ERRAND_PROB_WEEKEND": 0.55,       # 周末外出概率
    "ERRAND_START_MIN": 600,           # 10:00 起可出行
    "ERRAND_END_MIN": 1260,            # 21:00 前结束
    "ERRAND_DURATION_MIN": 90,         # 每次外出约 90 分钟
    "N_WORKPLACES": 20,                # 工作场所数量
    "N_SCHOOLS": 8,                    # 学校数量
    "N_SHOPS": 12,                     # 商店数量
    "N_PARKS": 6,                      # 公园数量
    # 各类场所的散布半径（世界单位）：activity._spread_lut 取 work/school/shop 的 ×0.5
    # 作为停留游走半径，park 只用于外出落点散布（errand_spread = park × 0.5）
    "VENUE_SPREAD": {"work": 8.0, "school": 8.0, "shop": 4.0, "park": 10.0},
    "TRIP_SPEED": 8.0,                 # 抽象速度：单位/帧
    "TRIP_MIN_FRAMES": 2,              # 最短出行帧数
    "TRIP_MAX_FRAMES": 12,             # 最长出行帧数
    "LOCKDOWN_SHELTER": True,          # 封控时返家（False = 旧版原地冻结）
    "INITIAL_SEEDS": 20,               # 初始感染者数量（随后按人口比例缩放）

    # ---- 疾病参数（由 diseases/*.json 档案覆盖；此处是缺省/回归值）----
    "FRAMES": 5000,                    # 模拟总帧数（疾病档案可覆盖；环境变量 SIM_FRAMES 优先）
    "INCUBATION_SD": 10.0,             # 潜伏期标准差（帧）
    "INCUBATION_MIN": 10.0,            # 潜伏期最小值（帧）
    "INCUBATION_SELF_RECOVERY": 0.0005,  # 潜伏期每帧自愈概率
    "INCUBATING_TRANSMISSION_MULT": 0.15,  # 潜伏期传染乘子（相对发病期）
    "INITIAL_IMMUNE_FRACTION": 0.01,   # 初始免疫人口比例
    "INITIAL_IMMUNE_LEVEL": 0.9,       # 初始免疫者的免疫水平
    "MUTATION_BOUNDS": {               # 变异后的参数上下限（档案可覆盖）
        "beta": [0.05, 1.0],
        "gamma": [0.001, 0.05],
        "mu": [0.001, 0.15],
    },

    # ---- 自然史/医疗机制（疾病档案 schema v2 覆盖；默认关闭以保持旧行为）----
    # 全关时走旧路径：只有潜伏/发病/恢复/死亡计时，无床位、无年龄分层、无自然隔离
    "NATURAL_HISTORY_ENABLED": False,
    "ASYMPTOMATIC_FRACTION": 0.0,           # 发病时分流为无症状的比例（0 = 全部有症状）
    "ASYMPTOMATIC_DAYS_FRAMES": 0.0,        # 无症状期时长（帧）：写入 a_timer，到期转有症状
    # 分段线性传染力曲线 [[距发病天数, 相对传染力], ...]：折点必须是整数天且 x 递增，
    # ±1e9 是"无穷远"哨兵（首/末段之外取端点值）；disease 按 0.25 天网格建 LUT
    "INFECTIVITY_CURVE": [[-1.0e9, 0.15], [0.0, 1.0], [1.0e9, 1.0]],
    "ASYMPTOMATIC_INFECTIVITY_CURVE": [[0.0, 0.0]],  # 无症状者曲线（单折点 = 常数 0）
    "ISOLATION_PROBABILITY": 0.0,           # 发病后自然隔离概率（v4 再乘 compliance）
    "ISOLATION_DELAY_FRAMES": 0.0,          # 发病到隔离的延迟（帧）
    "ISOLATION_TRANSMISSION_MULT": 1.0,     # 隔离期间的传染力乘子
    "HOSPITAL_DELAY_FRAMES": 0.0,           # 发病到入院的延迟（帧）
    "COMMUNITY_DEATH_HAZARD": 0.0,          # 未住院者的每帧死亡风险
    "COMMUNITY_RECOVERY_RATE": 0.0,         # 未住院者的每帧恢复率
    "WARD_DEATH_HAZARD": 0.0,               # 住院者的每帧死亡风险
    "WARD_RECOVERY_RATE": 0.0,              # 住院者的每帧恢复率
    "HOSPITAL_DEATH_SHARE": 0.0,            # 总死亡中发生在医院内的比例（derive 据此拆 IFR）
    "UNTREATED_DEATH_MULT": 1.0,            # 需住院但无床位者相对 p_ward_death 的死亡倍数
    "UNTREATED_DEATH_HAZARD": 0.0,          # 无床位等待者的每帧死亡风险
    "HOSPITALIZATION_RATE": 0.0,            # 有症状者需住院比例（再乘 hosp_mult 年龄权重）
    "BEDS_PER_1000": 0.0,                   # 每千人床位数 → beds_total = 该值 × N / 1000
    "N_HOSPITALS_PER_100K": 0.0,            # 每 10 万人医院数（>0 且启用自然史时参与缩放）
    "N_HOSPITALS": 0,                       # 医院总数（由 _apply_population_scale 计算；模拟循环未读）
    "AGE_IFR_RELATIVE": None,               # 年龄组 IFR 相对权重（None = 不分层）
    "AGE_HOSPITALIZATION_RELATIVE": None,   # 年龄组住院率相对权重（None = 不分层）
    "AGE_SUSCEPTIBILITY": None,             # 年龄组易感性相对权重（None = 不分层）

    # ---- 社会响应层（疾病档案 schema v3 覆盖；默认关闭以保持 v2 行为）----
    "INTERVENTIONS_ENABLED": False,
    "TESTING_ENABLED": False,
    "TEST_PROBABILITY_SYMPTOMATIC": 0.0,
    "TEST_DELAY_FRAMES": 0.0,
    "TEST_CAPACITY_PER_100K_PER_DAY": 0.0,
    "TEST_SENSITIVITY": 1.0,
    "TEST_ISOLATION_ADHERENCE": 0.0,
    "TEST_ISOLATION_DELAY_FRAMES": 0.0,
    "TRACING_ENABLED": False,
    "TRACE_PROBABILITY": 0.0,
    "TRACE_DELAY_FRAMES": 0.0,
    "TRACE_HOUSEHOLD": True,
    "TRACE_VENUE_FRACTION": 0.0,
    "QUARANTINE_DAYS_FRAMES": 0.0,
    "QUARANTINE_ADHERENCE": 0.0,
    "QUARANTINE_TRANSMISSION_MULT": 1.0,
    "BEHAVIOR_ENABLED": False,
    "BEHAVIOR_RESPONSE_STRENGTH": 0.0,
    "BEHAVIOR_THRESHOLD_PER_100K": 0.0,
    "BEHAVIOR_SCALE_PER_100K": 1.0,
    "BEHAVIOR_RESPONSE_DELAY_FRAMES": 0.0,
    "BEHAVIOR_FATIGUE_DAYS": 1.0,
    "BEHAVIOR_FATIGUE_STRENGTH": 0.0,
    "BEHAVIOR_MIN_CONTACT_MULT": 0.0,
    "REPORTING_ENABLED": False,
    "REPORTING_FRACTION": 0.0,
    "REPORTING_DELAY_FRAMES": 0.0,
    "REPORTING_SYMPTOMATIC_ONLY": True,

    # ---- 个体异质性（疾病档案 schema v4 覆盖；默认关闭以保持旧行为）----
    # 传染力乘子服从对数正态（均值 1、log 标准差 sigma）：sigma=0 等价于无超传播
    "HETEROGENEITY_ENABLED": False,
    "INFECTIOUSNESS_SIGMA": 0.0,
    "COMPLIANCE_MEAN": 1.0,          # 依从性个体差异（作用在隔离/密接隔离/行为反馈）
    "COMPLIANCE_SD": 0.0,

    # ---- 接触混合结构（schema v4 覆盖；默认关闭）----
    "MIXING_ENABLED": False,
    "HOUSEHOLD_CONTACT_MULT": 1.0,   # 家庭内接触相对乘子
    "AGE_ASSORTATIVITY": 0.0,        # 4 组粗年龄同配强度（0=均匀混合，1=完全同配）
    "SYMPTOM_STAY_HOME_FRACTION": 0.0,  # 有症状者自发减少外出的比例

    # ---- 规模化参数（10k 为基准；人口可用环境变量 DISEASE_POPULATION 覆盖）----
    "CONTACT_MODE": "auto",            # auto=小规模用 dense、大规模用 grid；也可强制 "dense"/"grid"
    "CONTACT_DENSE_LIMIT": 30_000_000, # auto 模式下 n_inf*n_tar 不超过该值则走 dense 全矩阵
    "GRID_CHUNK": 2_000_000,           # 网格候选对分块大小（控制显存峰值）
    "GRID_INF_CHUNK": 100_000,         # 网格枚举时每批处理的感染者数（控制显存峰值）
    # 大人口下散点抽样上限（0 = 全部绘制）。10000 与 20000 在 100k/1M 下
    # 视觉几乎无差别（都是密集点云），但 Agg 绘制耗时约减半。
    "RENDER_MAX_POINTS": 10000,
}

CONFIG: dict = copy.deepcopy(_DEFAULTS)

# 场所码：activity 的 place/desired、render 的静态场所样式都按这个顺序取下标；
# LEISURE 同时覆盖商店与公园（外出落点的实际类型由 activity 的 errand 种类决定）
PLACE_HOME, PLACE_WORK, PLACE_SCHOOL, PLACE_LEISURE = 0, 1, 2, 3


def reset_config() -> None:
    """把 CONFIG 恢复为打包默认值，**字典身份保持不变**。

    必须原地 clear+update（而不是 ``CONFIG = deepcopy(...)``）：档案层与各模块在 import
    时就绑定了同一个 dict 引用，重新赋值会让它们继续看旧字典。init_session 每次先调它，
    所以同一进程内可以重复装配（换档案/换人口）。
    """
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(_DEFAULTS))


def _env_int(name, default):
    """读整数环境变量；缺失/非法时打印警告并回退 default。"""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        print(f"[warn] invalid {name}, fallback to {default}")
        return int(default)


def _resolve_seed() -> int:
    """解析随机种子：DISEASE_SEED 环境变量优先于 CONFIG["SEED"]，非法则回退并警告。"""
    try:
        return int(os.environ.get("DISEASE_SEED", CONFIG["SEED"]))
    except (TypeError, ValueError):
        value = int(CONFIG["SEED"])
        print(f"[warn] invalid DISEASE_SEED, fallback to CONFIG['SEED']={value}")
        return value


# 模块级会话状态：SEED 在 import 期解析一次（之后不变）；init_session 会重新绑定后两者，
# 所以调用方必须按属性名读取，不要在 import 期缓存 PROFILE_INFO/ACTIVE_PROFILE_NAME 的值
SEED: int = _resolve_seed()
PROFILE_INFO: dict = {}
ACTIVE_PROFILE_NAME: str = ""


def _apply_population_scale() -> None:
    """按 DISEASE_POPULATION 缩放世界与各类数量（基准 10k；**非幂等，只调用一次**）。

    面积 ∝ 人口（边长 ×√r）→ 人口密度不变；场所数 ∝ 人口；场所散布半径与通勤速度 ×√r
    → 场所间距与出行帧数分布不变；医疗阈值/初始种子 ∝ 人口。基准人口取 _DEFAULTS 的
    N_AGENTS（此时 CONFIG 刚被 reset_config 重置），所以必须在档案 apply 之后、建 state
    之前调用，否则会缩放到错误基准上。
    """
    _BASE_POP = float(CONFIG["N_AGENTS"])          # 基准 10000
    POPULATION = max(1000, _env_int("DISEASE_POPULATION", CONFIG["N_AGENTS"]))
    _SCALE = POPULATION / _BASE_POP
    _area_scale = _SCALE ** 0.5                    # 面积缩放 → 线性尺寸缩放
    CONFIG["N_AGENTS"] = POPULATION
    # 世界边长按 √r 放大：保持单位面积人口密度与基准一致
    CONFIG["WIDTH"] = CONFIG["WIDTH"] * _area_scale
    CONFIG["HEIGHT"] = CONFIG["HEIGHT"] * _area_scale
    # 场所数按人口比例放大，保证人均场所数与基准一致（至少 1 个）
    CONFIG["N_WORKPLACES"] = max(1, int(round(CONFIG["N_WORKPLACES"] * _SCALE)))
    CONFIG["N_SCHOOLS"] = max(1, int(round(CONFIG["N_SCHOOLS"] * _SCALE)))
    CONFIG["N_SHOPS"] = max(1, int(round(CONFIG["N_SHOPS"] * _SCALE)))
    CONFIG["N_PARKS"] = max(1, int(round(CONFIG["N_PARKS"] * _SCALE)))
    # 至少 4 个居住区中心：人口很小时也要保留多中心聚集结构
    CONFIG["N_RESIDENTIAL_CLUSTERS"] = max(
        4, int(round(CONFIG["N_RESIDENTIAL_CLUSTERS"] * _SCALE)))
    # 场所散布半径随世界尺寸放大，使多个场所之间的间距/密度与基准一致
    CONFIG["VENUE_SPREAD"] = {k: v * _area_scale for k, v in CONFIG["VENUE_SPREAD"].items()}
    # 通勤速度随尺度放大，使出行帧数分布与基准一致
    CONFIG["TRIP_SPEED"] = CONFIG["TRIP_SPEED"] * _area_scale
    # 医疗崩溃阈值按人口比例缩放（基准：10k 人口 / 1000 症状者）
    CONFIG["MEDICAL_THRESHOLD"] = max(1, int(round(CONFIG["MEDICAL_THRESHOLD"] * _SCALE)))
    # 初始感染者数按人口比例缩放（至少 1，否则永远不会爆发）
    CONFIG["INITIAL_SEEDS"] = max(1, int(round(CONFIG["INITIAL_SEEDS"] * _SCALE)))
    # 医院数按"每 10 万人"指标换算；只在启用自然史时才参与计算
    if CONFIG["NATURAL_HISTORY_ENABLED"] and CONFIG["N_HOSPITALS_PER_100K"] > 0:
        CONFIG["N_HOSPITALS"] = max(
            1, int(round(CONFIG["N_HOSPITALS_PER_100K"] * POPULATION / 100000.0)))
    print(f"Population: {POPULATION} (x{_SCALE:.1f}), "
          f"world: {CONFIG['WIDTH']:.1f}x{CONFIG['HEIGHT']:.1f}, "
          f"workplaces: {CONFIG['N_WORKPLACES']}, schools: {CONFIG['N_SCHOOLS']}, "
          f"medical threshold: {CONFIG['MEDICAL_THRESHOLD']}")


def init_session(argv: list[str] | None = None) -> dict:
    """装配一次会话：重置配置 → 设种子 → 应用疾病档案 → env 覆盖 → 人口缩放。

    返回 CONFIG（同一个 dict，原地变更）。可重复调用；档案非法时打印原因后 sys.exit(1)，
    所以拿到返回就意味着装配成功。argv=None 时读 sys.argv[1:]，第一个位置参数是档案名
    （由 profiles.resolve_name 解析）。
    """
    global PROFILE_INFO, ACTIVE_PROFILE_NAME   # 模块级会话状态，由本函数重新绑定
    reset_config()                             # 回到默认值：重复调用不会叠加档案/缩放
    torch.manual_seed(SEED)                    # 全局随机流只在这里播种一次
    print(f"Using device: {device}, seed: {SEED}")
    try:
        name = disease_profiles.resolve_name(sys.argv[1:] if argv is None else argv)
        PROFILE_INFO = disease_profiles.load_and_apply(name, CONFIG)
    except disease_profiles.ProfileError as profile_error:
        print(f"[profile] {profile_error}")
        sys.exit(1)
    # SIM_FRAMES 环境变量优先于档案里的 FRAMES（短跑/调试用）
    CONFIG["FRAMES"] = int(os.environ.get("SIM_FRAMES", CONFIG["FRAMES"]))
    # 渲染点数上限（性能旋钮；默认 10000 见 _DEFAULTS["RENDER_MAX_POINTS"]，非法值忽略）
    _render_points = os.environ.get("SIM_RENDER_POINTS")
    if _render_points:
        try:
            CONFIG["RENDER_MAX_POINTS"] = max(100, int(_render_points))
        except ValueError:
            print(f"[warn] invalid SIM_RENDER_POINTS={_render_points!r}, ignored")
    ACTIVE_PROFILE_NAME = PROFILE_INFO["profile_id"]   # 供工具读取
    print(disease_profiles.format_summary(PROFILE_INFO))
    _apply_population_scale()
    return CONFIG
