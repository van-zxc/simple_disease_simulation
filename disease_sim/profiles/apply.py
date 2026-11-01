"""档案应用：把校验后的档案 / 派生结果写入 CONFIG，并汇总成 PROFILE_INFO。

职责：``apply_profile`` 是唯一会写 CONFIG 的入口（先派生、后覆盖、再二次校验）；
    ``load_and_apply`` = 读取 + 校验 + 应用；``format_summary`` 提供稳定的人读输出。
非职责：不读文件（loader.py）、不校验档案输入（validate.py）、不做参数换算（derive.py）。
入口：``apply_profile(profile, config) -> PROFILE_INFO``（含 derived/heterogeneity/mixing…）。
不变式：model_overrides 与 derived 互斥（v1 简档直接覆盖 CONFIG，不走派生）；
    v4 两个 _apply_* 总会写入块内值，关闭块必须提供中性值，否则 CONFIG 残留非中性参数；
    ``_validate_applied`` 在写入后复查，因为派生结果同样必须自洽。
"""

from __future__ import annotations

import math
import copy

from .schema import (
    AGE_BANDS,
    FRAMES_PER_DAY_KEY,
    HETEROGENEITY_DEFAULTS,
    MIXING_DEFAULTS,
    ProfileError,
    SCHEMA_VERSION,
)
from .loader import load_profile
from .validate import _validate_bounds
from .derive import _derive_interventions, derive_model_parameters


def _apply_interventions(config, result):
    """把 _derive_interventions 的结果（None = 档案无该块）写入 CONFIG。

    注意：``result`` 为空时用关闭态默认值，保证 v1/v2 老档案写入"全关 + 中性参数"，
    因此老档案行为与引入 v3 前等价；子开关与总开关按 AND 生效。
    """
    if not result:
        result = _derive_interventions({"enabled": False}, 1.0)
    enabled = bool(result.get("enabled", False))
    config["INTERVENTIONS_ENABLED"] = enabled
    config["TESTING_ENABLED"] = bool(result.get("testing_enabled", False)) and enabled
    config["TEST_PROBABILITY_SYMPTOMATIC"] = float(
        result.get("test_probability_symptomatic", 0.0))
    config["TEST_DELAY_FRAMES"] = float(result.get("test_delay_frames", 0.0))
    config["TEST_CAPACITY_PER_100K_PER_DAY"] = float(
        result.get("test_capacity_per_100k_per_day", 0.0))
    config["TEST_SENSITIVITY"] = float(result.get("test_sensitivity", 1.0))
    config["TEST_ISOLATION_ADHERENCE"] = float(
        result.get("test_isolation_adherence", 0.0))
    config["TEST_ISOLATION_DELAY_FRAMES"] = float(
        result.get("test_isolation_delay_frames", 0.0))

    config["TRACING_ENABLED"] = bool(result.get("tracing_enabled", False)) and enabled
    config["TRACE_PROBABILITY"] = float(result.get("trace_probability", 0.0))
    config["TRACE_DELAY_FRAMES"] = float(result.get("trace_delay_frames", 0.0))
    config["TRACE_HOUSEHOLD"] = bool(result.get("trace_household", True))
    config["TRACE_VENUE_FRACTION"] = float(result.get("trace_venue_fraction", 0.0))
    config["QUARANTINE_DAYS_FRAMES"] = float(result.get("quarantine_days_frames", 0.0))
    config["QUARANTINE_ADHERENCE"] = float(result.get("quarantine_adherence", 0.0))
    config["QUARANTINE_TRANSMISSION_MULT"] = float(
        result.get("quarantine_transmission_mult", 1.0))

    config["BEHAVIOR_ENABLED"] = bool(result.get("behavior_enabled", False)) and enabled
    config["BEHAVIOR_RESPONSE_STRENGTH"] = float(
        result.get("behavior_response_strength", 0.0))
    config["BEHAVIOR_THRESHOLD_PER_100K"] = float(
        result.get("behavior_threshold_per_100k", 0.0))
    config["BEHAVIOR_SCALE_PER_100K"] = float(
        result.get("behavior_scale_per_100k", 1.0))
    config["BEHAVIOR_RESPONSE_DELAY_FRAMES"] = float(
        result.get("behavior_response_delay_frames", 0.0))
    config["BEHAVIOR_FATIGUE_DAYS"] = float(result.get("behavior_fatigue_days", 1.0))
    config["BEHAVIOR_FATIGUE_STRENGTH"] = float(
        result.get("behavior_fatigue_strength", 0.0))
    config["BEHAVIOR_MIN_CONTACT_MULT"] = float(
        result.get("behavior_min_contact_mult", 0.0))

    config["REPORTING_ENABLED"] = bool(result.get("reporting_enabled", False)) and enabled
    config["REPORTING_FRACTION"] = float(result.get("reporting_fraction", 0.0))
    config["REPORTING_DELAY_FRAMES"] = float(result.get("reporting_delay_frames", 0.0))
    config["REPORTING_SYMPTOMATIC_ONLY"] = bool(
        result.get("reporting_symptomatic_only", True))




def _validate_applied(config: dict):
    """写入 CONFIG 后的最终自洽检查，失败抛 ``ProfileError``。

    为什么要二次校验：派生公式可能把档案参数变成非法组合（自然史每帧 hazard 之和 > 1、
    v1 简档直接覆盖出越界值等），且数值区间与档案原文不同，必须在 apply 之后再查一遍。
    错误消息面向用户中文说明，须保持可读。
    """
    checks = {
        "BASE_TRANSMIT": (0.0, None),
        "BASE_RECOVERY": (0.0, 1.0),
        "BASE_DEATH": (0.0, None),
        "INCUBATION_MU": (1.0, None),
        "INCUBATION_SD": (0.0, None),
        "INCUBATION_MIN": (0.0, None),
        "INCUBATING_TRANSMISSION_MULT": (0.0, 10.0),
        "IMMUNITY_GAIN_PER_INFECTION": (0.0, 1.0),
        "IMMUNITY_DECAY_RATE": (0.0, None),
        "MAX_IMMUNITY": (0.0, 1.0),
        "IMMUNITY_EFFECTIVENESS": (0.0, None),
        "IMMUNITY_WANING_THRESHOLD": (0.0, 1.0),
        "TRANSMIT_BASE_PROB": (0.0, 1.0),
    }
    for key, (lo, hi) in checks.items():
        if key not in config:
            raise ProfileError(f"应用档案后缺少 CONFIG['{key}']")
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ProfileError(f"CONFIG['{key}'] 必须是有限数（实际 {value!r}）")
        if value < lo or (hi is not None and value > hi):
            raise ProfileError(f"CONFIG['{key}']={value} 超出范围 [{lo}, {hi}]")
    if config["MUTATION_RATE"] < 0 or config["MUTATION_RATE"] > 1:
        raise ProfileError("CONFIG['MUTATION_RATE'] 必须在 [0, 1]")
    if int(config["FRAMES"]) < 1:
        raise ProfileError("CONFIG['FRAMES'] 必须至少为 1")
    if float(config["FRAMES"]) != int(config["FRAMES"]):
        raise ProfileError("CONFIG['FRAMES'] 必须是整数")
    if not 0.0 <= float(config["INITIAL_IMMUNE_FRACTION"]) <= 1.0:
        raise ProfileError("CONFIG['INITIAL_IMMUNE_FRACTION'] 必须在 [0, 1]")
    if not 0.0 <= float(config["INITIAL_IMMUNE_LEVEL"]) <= 1.0:
        raise ProfileError("CONFIG['INITIAL_IMMUNE_LEVEL'] 必须在 [0, 1]")
    if int(config["INITIAL_SEEDS"]) < 0:
        raise ProfileError("CONFIG['INITIAL_SEEDS'] 不能为负")
    if int(config["MEDICAL_THRESHOLD"]) < 1:
        raise ProfileError("CONFIG['MEDICAL_THRESHOLD'] 必须至少为 1")
    bounds = config.get("MUTATION_BOUNDS")
    if not isinstance(bounds, dict):
        raise ProfileError("CONFIG['MUTATION_BOUNDS'] 必须是对象")
    _validate_bounds(bounds, "CONFIG.MUTATION_BOUNDS")
    if config["IMMUNITY_GAIN_PER_INFECTION"] <= config["IMMUNITY_WANING_THRESHOLD"]:
        raise ProfileError("IMMUNITY_GAIN_PER_INFECTION 必须大于 IMMUNITY_WANING_THRESHOLD")

    if not isinstance(config.get("NATURAL_HISTORY_ENABLED"), bool):
        raise ProfileError("CONFIG['NATURAL_HISTORY_ENABLED'] 必须是 true/false")
    if config["NATURAL_HISTORY_ENABLED"]:
        for key in ("ASYMPTOMATIC_FRACTION", "ASYMPTOMATIC_DAYS_FRAMES",
                    "ISOLATION_PROBABILITY", "ISOLATION_DELAY_FRAMES",
                    "ISOLATION_TRANSMISSION_MULT", "HOSPITAL_DELAY_FRAMES",
                    "WARD_DEATH_HAZARD", "WARD_RECOVERY_RATE",
                    "COMMUNITY_DEATH_HAZARD", "COMMUNITY_RECOVERY_RATE",
                    "HOSPITAL_DEATH_SHARE", "UNTREATED_DEATH_MULT",
                    "UNTREATED_DEATH_HAZARD",
                    "HOSPITALIZATION_RATE", "BEDS_PER_1000", "N_HOSPITALS_PER_100K"):
            value = config.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)):
                raise ProfileError(f"CONFIG['{key}'] 必须是有限数（实际 {value!r}）")
            if float(value) < 0.0:
                raise ProfileError(f"CONFIG['{key}'] 不能为负（实际 {value}）")
        if not 0.0 <= config["ASYMPTOMATIC_FRACTION"] < 1.0:
            raise ProfileError("CONFIG['ASYMPTOMATIC_FRACTION'] 必须在 [0, 1)")
        if not 0.0 <= config["ISOLATION_PROBABILITY"] <= 1.0:
            raise ProfileError("CONFIG['ISOLATION_PROBABILITY'] 必须在 [0, 1]")
        if not 0.0 <= config["HOSPITALIZATION_RATE"] < 1.0:
            raise ProfileError("CONFIG['HOSPITALIZATION_RATE'] 必须在 [0, 1)")
        if not 0.0 <= config["HOSPITAL_DEATH_SHARE"] <= 1.0:
            raise ProfileError("CONFIG['HOSPITAL_DEATH_SHARE'] 必须在 [0, 1]")
        for key in ("AGE_IFR_RELATIVE", "AGE_HOSPITALIZATION_RELATIVE",
                    "AGE_SUSCEPTIBILITY"):
            table = config.get(key)
            if not isinstance(table, (list, tuple)) or len(table) != len(AGE_BANDS):
                raise ProfileError(f"CONFIG['{key}'] 必须是长度 {len(AGE_BANDS)} 的列表")
            if any(float(v) < 0.0 for v in table):
                raise ProfileError(f"CONFIG['{key}'] 不能含负数")
        if config["COMMUNITY_DEATH_HAZARD"] + config["COMMUNITY_RECOVERY_RATE"] > 1.0:
            raise ProfileError("社区死亡+康复每帧概率之和不能超过 1")
        if config["UNTREATED_DEATH_HAZARD"] + config["COMMUNITY_RECOVERY_RATE"] > 1.0:
            raise ProfileError("无床死亡+社区康复每帧概率之和不能超过 1")
        if config["WARD_DEATH_HAZARD"] + config["WARD_RECOVERY_RATE"] > 1.0:
            raise ProfileError("住院死亡+康复每帧概率之和不能超过 1")

    for key in ("INTERVENTIONS_ENABLED", "TESTING_ENABLED", "TRACING_ENABLED",
                "BEHAVIOR_ENABLED", "REPORTING_ENABLED", "TRACE_HOUSEHOLD",
                "REPORTING_SYMPTOMATIC_ONLY"):
        if not isinstance(config.get(key), bool):
            raise ProfileError(f"CONFIG['{key}'] 必须是 true/false")
    if config["INTERVENTIONS_ENABLED"] and not config["NATURAL_HISTORY_ENABLED"]:
        raise ProfileError("INTERVENTIONS_ENABLED 需要 NATURAL_HISTORY_ENABLED 为 true")

    probability_keys = (
        "TEST_PROBABILITY_SYMPTOMATIC", "TEST_SENSITIVITY",
        "TEST_ISOLATION_ADHERENCE", "TRACE_PROBABILITY",
        "TRACE_VENUE_FRACTION", "QUARANTINE_ADHERENCE",
        "QUARANTINE_TRANSMISSION_MULT", "BEHAVIOR_MIN_CONTACT_MULT",
        "REPORTING_FRACTION")
    for key in probability_keys:
        value = config.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)):
            raise ProfileError(f"CONFIG['{key}'] 必须是有限数（实际 {value!r}）")
        if not 0.0 <= float(value) <= 1.0:
            raise ProfileError(f"CONFIG['{key}'] 必须在 [0, 1]（实际 {value}）")

    non_negative_keys = (
        "TEST_DELAY_FRAMES", "TEST_CAPACITY_PER_100K_PER_DAY",
        "TEST_ISOLATION_DELAY_FRAMES", "TRACE_DELAY_FRAMES",
        "QUARANTINE_DAYS_FRAMES", "BEHAVIOR_RESPONSE_STRENGTH",
        "BEHAVIOR_THRESHOLD_PER_100K", "BEHAVIOR_RESPONSE_DELAY_FRAMES",
        "BEHAVIOR_FATIGUE_STRENGTH", "REPORTING_DELAY_FRAMES")
    for key in non_negative_keys:
        value = config.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)):
            raise ProfileError(f"CONFIG['{key}'] 必须是有限数（实际 {value!r}）")
        if float(value) < 0.0:
            raise ProfileError(f"CONFIG['{key}'] 不能为负（实际 {value}）")

    if config["BEHAVIOR_SCALE_PER_100K"] <= 0.0:
        raise ProfileError("CONFIG['BEHAVIOR_SCALE_PER_100K'] 必须大于 0")
    if config["BEHAVIOR_FATIGUE_DAYS"] <= 0.0:
        raise ProfileError("CONFIG['BEHAVIOR_FATIGUE_DAYS'] 必须大于 0")
    if config["TRACING_ENABLED"] and config["QUARANTINE_DAYS_FRAMES"] <= 0.0:
        raise ProfileError("追踪启用时 CONFIG['QUARANTINE_DAYS_FRAMES'] 必须大于 0")




def _apply_derived(config: dict, derived: dict):
    """把 derived 逐键写入 CONFIG（自然史整块 + 三张年龄表转 list），再写社会响应。

    不变式：``derived["natural_history"]`` 为空表示走旧版 SEIR，此时只写
    ``NATURAL_HISTORY_ENABLED=False``，不清除其他键（保留 CONFIG 既有值）。
    """
    config["BASE_TRANSMIT"] = derived["beta"]
    config["BASE_RECOVERY"] = derived["gamma"]
    config["BASE_DEATH"] = derived["mu"]
    config["INCUBATION_MU"] = derived["incubation_mu"]
    config["INCUBATION_SD"] = derived["incubation_sd"]
    config["INCUBATION_MIN"] = derived["incubation_min"]
    config["INCUBATING_TRANSMISSION_MULT"] = derived["incubating_transmission_mult"]
    config["INCUBATION_SELF_RECOVERY"] = derived["incubation_self_recovery"]
    config["MUTATION_RATE"] = derived["mutation_rate"]
    config["MUTATION_SCALE"] = derived["mutation_scale"]
    config["MUTATION_BOUNDS"] = copy.deepcopy(derived["mutation_bounds"])
    config["IMMUNITY_GAIN_PER_INFECTION"] = derived["immunity_gain"]
    config["IMMUNITY_DECAY_RATE"] = derived["immunity_decay"]
    config["MEDICAL_THRESHOLD"] = max(1, int(round(derived["medical_threshold_per_10k"])))
    config["INITIAL_IMMUNE_FRACTION"] = derived["initial_immune_fraction"]
    config["INITIAL_IMMUNE_LEVEL"] = derived["initial_immune_level"]
    config["INITIAL_SEEDS"] = derived["initial_seeds"]
    config["FRAMES"] = derived["frames"]

    config["BEDS_PER_1000"] = float(derived.get("beds_per_1000", 0.0))
    config["N_HOSPITALS_PER_100K"] = float(derived.get("n_hospitals_per_100k", 0.0))
    config["HOSPITALIZATION_RATE"] = float(derived.get("hosp_rate", 0.0))
    natural = derived.get("natural_history")
    if natural:
        config["NATURAL_HISTORY_ENABLED"] = True
        config["ASYMPTOMATIC_FRACTION"] = natural["asymptomatic_fraction"]
        config["ASYMPTOMATIC_DAYS_FRAMES"] = natural["asymptomatic_days_frames"]
        config["INFECTIVITY_CURVE"] = copy.deepcopy(natural["infectivity_curve"])
        config["ASYMPTOMATIC_INFECTIVITY_CURVE"] = copy.deepcopy(
            natural["asymptomatic_infectivity_curve"])
        config["ISOLATION_PROBABILITY"] = natural["isolation_probability"]
        config["ISOLATION_DELAY_FRAMES"] = natural["isolation_delay_frames"]
        config["ISOLATION_TRANSMISSION_MULT"] = natural["isolation_transmission_mult"]
        config["HOSPITAL_DELAY_FRAMES"] = natural["hospital_delay_frames"]
        config["WARD_DEATH_HAZARD"] = natural["ward_death_hazard"]
        config["WARD_RECOVERY_RATE"] = natural["ward_recovery_rate"]
        config["COMMUNITY_DEATH_HAZARD"] = natural["community_death_hazard"]
        config["COMMUNITY_RECOVERY_RATE"] = natural["community_recovery_rate"]
        config["HOSPITAL_DEATH_SHARE"] = natural["hospital_death_share"]
        config["UNTREATED_DEATH_MULT"] = natural["untreated_death_mult"]
        config["UNTREATED_DEATH_HAZARD"] = natural["untreated_death_hazard"]
        config["AGE_IFR_RELATIVE"] = list(natural["age_ifr_relative"].values())
        config["AGE_HOSPITALIZATION_RELATIVE"] = list(
            natural["age_hospitalization_relative"].values())
        config["AGE_SUSCEPTIBILITY"] = list(natural["age_susceptibility"].values())
    else:
        config["NATURAL_HISTORY_ENABLED"] = False

    _apply_interventions(config, derived.get("interventions"))




def _apply_heterogeneity(config, block):
    """把 heterogeneity 段（schema v4）写入 CONFIG；None = 关闭并写默认值。

    注意：即使 ``enabled=false``，块内出现的键也会照原值写入（不看开关）；
    因此档案/fixture 的关闭块必须使用中性值（sigma 0.0、compliance 1.0 / 0.0），
    否则 CONFIG 会残留非中性参数——这是实际踩过的坑。
    """
    data = dict(HETEROGENEITY_DEFAULTS)
    if isinstance(block, dict):
        data.update({key: value for key, value in block.items() if key in HETEROGENEITY_DEFAULTS})
    config["HETEROGENEITY_ENABLED"] = bool(data.get("enabled", False))
    config["INFECTIOUSNESS_SIGMA"] = float(data.get("infectiousness_sigma", 0.0))
    config["COMPLIANCE_MEAN"] = float(data.get("compliance_mean", 1.0))
    config["COMPLIANCE_SD"] = float(data.get("compliance_sd", 0.0))




def _apply_mixing(config, block):
    """把 mixing 段（schema v4）写入 CONFIG；None = 关闭并写默认值。

    注意：与 _apply_heterogeneity 相同——``enabled=false`` 时块内值仍照写，
    关闭块必须用中性值（家庭×1.0、年龄同配 0.0、症状居家 0.0），否则参数被污染。
    """
    data = dict(MIXING_DEFAULTS)
    if isinstance(block, dict):
        data.update({key: value for key, value in block.items() if key in MIXING_DEFAULTS})
    config["MIXING_ENABLED"] = bool(data.get("enabled", False))
    config["HOUSEHOLD_CONTACT_MULT"] = float(data.get("household_contact_mult", 1.0))
    config["AGE_ASSORTATIVITY"] = float(data.get("age_assortativity", 0.0))
    config["SYMPTOM_STAY_HOME_FRACTION"] = float(
        data.get("symptom_stay_home_fraction", 0.0))




def apply_profile(profile: dict, config: dict) -> dict:
    """把档案写入 CONFIG，返回 PROFILE_INFO（供打印 / 工具读取）。

    model_overrides 档案（v1 简档）直接覆盖 CONFIG、不派生；其余档案走 derive +
    _apply_derived。最后总会写 heterogeneity/mixing 与 FRAMES，并做二次校验。
    不变式：``time.frames_per_day`` 必须等于 CONFIG 的 ``DAY_FRAMES``（容差 1e-9）。
    """
    if not isinstance(config, dict):
        raise ProfileError("apply_profile 需要 CONFIG 字典")
    fpd = float(config[FRAMES_PER_DAY_KEY])
    profile_fpd = float(profile["time"]["frames_per_day"])
    if abs(profile_fpd - fpd) > 1e-9:
        raise ProfileError(
            f"time.frames_per_day={profile_fpd:g} 与模型 {FRAMES_PER_DAY_KEY}={fpd:g} 不一致")

    derived = None
    if "model_overrides" in profile:
        for key, value in profile["model_overrides"].items():
            config[key] = copy.deepcopy(value)
    else:
        derived = derive_model_parameters(profile, config)
        _apply_derived(config, derived)

    population = profile.get("population", {}) or {}
    if "initial_immune_fraction" in population:
        config["INITIAL_IMMUNE_FRACTION"] = float(population["initial_immune_fraction"])
    if "initial_immune_level" in population:
        config["INITIAL_IMMUNE_LEVEL"] = float(population["initial_immune_level"])
    if "initial_seeds" in population:
        config["INITIAL_SEEDS"] = int(population["initial_seeds"])

    time = profile["time"]
    if "simulation_frames" in time:
        config["FRAMES"] = int(time["simulation_frames"])
    elif derived is not None:
        config["FRAMES"] = derived["frames"]
    else:
        config["FRAMES"] = int(round(float(time["simulation_days"]) * fpd))

    if derived is None:
        _apply_interventions(config, None)
    _apply_heterogeneity(config, profile.get("heterogeneity"))
    _apply_mixing(config, profile.get("mixing"))
    _validate_applied(config)

    if derived is None:
        derived = {
            "beta": float(config["BASE_TRANSMIT"]),
            "gamma": float(config["BASE_RECOVERY"]),
            "mu": float(config["BASE_DEATH"]),
            "incubation_mu": float(config["INCUBATION_MU"]),
            "incubation_sd": float(config["INCUBATION_SD"]),
            "incubation_min": float(config["INCUBATION_MIN"]),
            "incubating_transmission_mult": float(config["INCUBATING_TRANSMISSION_MULT"]),
            "mutation_enabled": float(config["MUTATION_RATE"]) > 0,
            "mutation_rate": float(config["MUTATION_RATE"]),
            "immunity_gain": float(config["IMMUNITY_GAIN_PER_INFECTION"]),
            "immunity_decay": float(config["IMMUNITY_DECAY_RATE"]),
            "medical_threshold_per_10k": int(config["MEDICAL_THRESHOLD"]),
            "beds_per_1000": float(config.get("BEDS_PER_1000", 0.0)),
            "n_hospitals_per_100k": float(config.get("N_HOSPITALS_PER_100K", 0.0)),
            "hosp_rate": float(config.get("HOSPITALIZATION_RATE", 0.0)),
            "natural_history_enabled": False,
            "natural_history": None,
            "interventions_enabled": False,
            "interventions": None,
            "frames": int(config["FRAMES"]),
            "r0_target": None,
            "serial_interval_days": None,
        }

    return {
        "profile_id": profile["profile_id"],
        "label": profile["label"],
        "schema_version": int(profile.get("schema_version", SCHEMA_VERSION)),
        "frames_per_day": int(profile["time"]["frames_per_day"]),
        "frames": int(config["FRAMES"]),
        "simulation_days": float(config["FRAMES"]) / fpd,
        "derived": derived,
        "heterogeneity": {
            "enabled": bool(config.get("HETEROGENEITY_ENABLED", False)),
            "infectiousness_sigma": float(config.get("INFECTIOUSNESS_SIGMA", 0.0)),
            "compliance_mean": float(config.get("COMPLIANCE_MEAN", 1.0)),
            "compliance_sd": float(config.get("COMPLIANCE_SD", 0.0)),
        },
        "mixing": {
            "enabled": bool(config.get("MIXING_ENABLED", False)),
            "household_contact_mult": float(config.get("HOUSEHOLD_CONTACT_MULT", 1.0)),
            "age_assortativity": float(config.get("AGE_ASSORTATIVITY", 0.0)),
            "symptom_stay_home_fraction": float(
                config.get("SYMPTOM_STAY_HOME_FRACTION", 0.0)),
        },
    }




def load_and_apply(name: str, config: dict) -> dict:
    """load_profile + apply_profile 的便捷入口（返回 PROFILE_INFO）。

    档案缺失或非法时抛 ``ProfileError``，由调用方负责打印说明并退出。
    """
    profile = load_profile(name)
    return apply_profile(profile, config)




def format_summary(info: dict) -> str:
    """生成多行中文摘要（主程序与工具共用的稳定输出格式）。

    行结构：档案头、模拟帧数、R0/参数行、自然史、社会响应、个体异质性、接触混合；
    仅当对应块存在或启用时追加该行。调用方逐行打印，不要依赖更细的分隔约定。
    """
    derived = info.get("derived", {})
    days = info.get("simulation_days")
    lines = [
        f"档案: {info['profile_id']} — {info.get('label', '')}",
        f"  模拟: {info.get('frames')} 帧 ≈ {days:.1f} 天"
        f"（{info.get('frames_per_day')} 帧/天）",
    ]
    if derived.get("r0_target") is not None:
        lines.append(
            "  R0={r0:.2f}（校验 {rc:.3f}），潜伏期={inc:.0f} 帧，感染期={sym:.0f} 帧，"
            "beta={beta:.5g}, gamma={gamma:.5g}, mu={mu:.5g}".format(
                r0=derived["r0_target"], rc=derived.get("r0_check", float("nan")),
                inc=derived["incubation_mu"], sym=1.0 / derived["gamma"],
                beta=derived["beta"], gamma=derived["gamma"], mu=derived["mu"]))
    else:
        lines.append(
            "  beta={beta:.5g}, gamma={gamma:.5g}, mu={mu:.5g}, "
            "潜伏期={inc:.0f} 帧（model_overrides 档案）".format(
                beta=derived["beta"], gamma=derived["gamma"], mu=derived["mu"],
                inc=derived["incubation_mu"]))
    natural = derived.get("natural_history")
    if natural:
        lines.append(
            "  自然史: 无症状={a:.0%}（{ad:.1f} 天），隔离 p={p:.0%}/延迟 {delay:.1f} 天，"
            "住院延迟 {hd:.1f} 天，床位 {beds:.1f}/千人，有效传染期≈{te:.0f} 帧".format(
                a=natural["asymptomatic_fraction"],
                ad=natural["asymptomatic_days_frames"] / info["frames_per_day"],
                p=natural["isolation_probability"],
                delay=natural["isolation_delay_frames"] / info["frames_per_day"],
                hd=natural["hospital_delay_frames"] / info["frames_per_day"],
                beds=natural["beds_per_1000"],
                te=natural["t_eff_frames"]))
    interventions = derived.get("interventions")
    if interventions and interventions.get("enabled"):
        fpd = float(info["frames_per_day"])
        parts = []
        if interventions.get("testing_enabled"):
            parts.append(
                "检测 p={p:.0%}/延迟 {d:.1f} 天/灵敏度 {s:.0%}".format(
                    p=interventions["test_probability_symptomatic"],
                    d=interventions["test_delay_frames"] / fpd,
                    s=interventions["test_sensitivity"]))
        if interventions.get("tracing_enabled"):
            parts.append(
                "追踪 p={p:.0%}/隔离 {q:.0f} 天".format(
                    p=interventions["trace_probability"],
                    q=interventions["quarantine_days_frames"] / fpd))
        if interventions.get("behavior_enabled"):
            parts.append(
                "行为强度 {s:.0%}/阈值 {t:.0f}/10 万".format(
                    s=interventions["behavior_response_strength"],
                    t=interventions["behavior_threshold_per_100k"]))
        if interventions.get("reporting_enabled"):
            parts.append(
                "报告 {f:.0%}/延迟 {d:.1f} 天".format(
                    f=interventions["reporting_fraction"],
                    d=interventions["reporting_delay_frames"] / fpd))
        if parts:
            lines.append("  社会响应: " + "；".join(parts))
    hetero = info.get("heterogeneity")
    if hetero and hetero.get("enabled"):
        lines.append(
            "  个体异质性: 传染力 sigma={s:.2f}，依从性 {m:.2f}±{sd:.2f}".format(
                s=hetero["infectiousness_sigma"],
                m=hetero["compliance_mean"],
                sd=hetero["compliance_sd"]))
    mixing = info.get("mixing")
    if mixing and mixing.get("enabled"):
        lines.append(
            "  接触混合: 家庭×{hh:.1f}，年龄同配 {a:.0%}，症状居家 {f:.0%}".format(
                hh=mixing["household_contact_mult"],
                a=mixing["age_assortativity"],
                f=mixing["symptom_stay_home_fraction"]))
    return "\n".join(lines)
