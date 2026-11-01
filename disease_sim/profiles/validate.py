"""档案校验：字段/数值/区间/曲线/年龄表，以及 v2 自然史、v3 社会响应、v4 两个块的检查。

职责：把 JSON 档案校验成"可安全派生"的形态；所有错误统一抛 ``ProfileError``。
非职责：不做派生（derive.py）、不写 CONFIG（apply.py）、不读文件（loader.py）。
入口：``validate_profile`` 供 loader 校验原始 JSON；派生结果的二次校验见
``apply._validate_applied``——派生参数同样必须自洽，否则宁可拒绝也不带病仿真。
不变式：未知键一律拒绝（v4 块连 ``note`` 之类子键也不许）；数值区间与错误消息
是用户契约，文案保持中文并带完整字段路径。
"""

from __future__ import annotations

import math

from .schema import (
    AGE_BANDS,
    ALLOWED_OVERRIDES,
    HETEROGENEITY_DEFAULTS,
    MIXING_DEFAULTS,
    ProfileError,
    SCHEMA_VERSIONS,
    _INTERVENTION_KEYS,
)


def _field(obj, key, where):
    """取必填字段；父对象不是 dict 或键缺失时抛 ``ProfileError``（带字段路径）。"""
    if not isinstance(obj, dict) or key not in obj:
        raise ProfileError(f"缺少字段 {where}.{key}")
    return obj[key]




def _number(value, where, *, minimum=None, maximum=None, integer=False):
    """把手写值严格校验为有限数字（可选区间/整数），返回 int 或 float。

    ``bool`` 被显式拒绝：它是 int 子类，但语义上不属于数值参数。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where} 必须是数字，实际 {value!r}")
    if not math.isfinite(float(value)):
        raise ProfileError(f"{where} 必须是有限数")
    if integer:
        if float(value) != int(value):
            raise ProfileError(f"{where} 必须是整数，实际 {value!r}")
        parsed = int(value)
    else:
        parsed = float(value)
    if minimum is not None and parsed < minimum:
        raise ProfileError(f"{where} 不能小于 {minimum}（实际 {parsed}）")
    if maximum is not None and parsed > maximum:
        raise ProfileError(f"{where} 不能大于 {maximum}（实际 {parsed}）")
    return parsed




def _value_entry(epi, key, where, *, minimum=None, maximum=None):
    """校验 epidemiology 的 ``{value, range, source}`` 条目，返回其 ``value``。

    ``range`` 可选；出现时必须是二元序列且下限 ≤ 上限（不要求包住 value）。
    """
    entry = _field(epi, key, where)
    if not isinstance(entry, dict):
        raise ProfileError(f"{where}.{key} 必须是对象，包含 value/range/source")
    value = _number(_field(entry, "value", f"{where}.{key}"),
                    f"{where}.{key}.value", minimum=minimum, maximum=maximum)
    if "range" in entry:
        rng = entry["range"]
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ProfileError(f"{where}.{key}.range 必须是 [下限, 上限]")
        lo = _number(rng[0], f"{where}.{key}.range[0]")
        hi = _number(rng[1], f"{where}.{key}.range[1]")
        if lo > hi:
            raise ProfileError(f"{where}.{key}.range 下限大于上限")
    return value




def _validate_epidemiology(epi):
    """校验 epidemiology 段：R0、潜伏/传染期、IFR、住院率与床位密度等数值区间。

    不变式：概率上界留出余量（IFR≤0.99、症状前传播≤0.9 等），让 derive 的概率运算
    不会退化成 0/1；时长下限 0.01 天，避免派生时除以 0。
    """
    if not isinstance(epi, dict):
        raise ProfileError("epidemiology 必须是 JSON 对象")

    _value_entry(epi, "r0", "epidemiology", minimum=0.0)
    inc = _field(epi, "incubation_days", "epidemiology")
    _number(_field(inc, "mean", "epidemiology.incubation_days"),
            "epidemiology.incubation_days.mean", minimum=0.01)
    if "sd" in inc:
        _number(inc["sd"], "epidemiology.incubation_days.sd", minimum=0.0)
    if "min" in inc:
        _number(inc["min"], "epidemiology.incubation_days.min", minimum=0.0)
    if "max" in inc:
        _number(inc["max"], "epidemiology.incubation_days.max", minimum=0.0)
    _value_entry(epi, "infectious_days", "epidemiology", minimum=0.01)
    _value_entry(epi, "ifr", "epidemiology", minimum=0.0, maximum=0.99)
    _value_entry(epi, "presymptomatic_transmission_fraction", "epidemiology",
                 minimum=0.0, maximum=0.9)
    if "symptomatic_fraction" in epi:
        _value_entry(epi, "symptomatic_fraction", "epidemiology",
                     minimum=0.01, maximum=1.0)
    _value_entry(epi, "immunity_days", "epidemiology", minimum=0.01)
    _value_entry(epi, "hospital_beds_per_1000", "epidemiology", minimum=0.01)
    _value_entry(epi, "hospitalization_rate", "epidemiology",
                 minimum=0.001, maximum=1.0)
    if "serial_interval_days" in epi:
        _value_entry(epi, "serial_interval_days", "epidemiology", minimum=0.01)

    if inc.get("min", 0.0) > inc["mean"]:
        raise ProfileError("epidemiology.incubation_days.min 不能大于 mean")




def _validate_bounds(bounds, where):
    """校验 ``{beta:[lo,hi], gamma:[lo,hi], mu:[lo,hi]}`` 结构，要求非负且 lo < hi。

    同时也用于 ``model_parameters.mutation.bounds``；不满足时抛 ``ProfileError``。
    """
    if not isinstance(bounds, dict):
        raise ProfileError(f"{where} 必须是对象，如 {{beta:[lo,hi], gamma:[lo,hi], mu:[lo,hi]}}")
    for key in ("beta", "gamma", "mu"):
        pair = _field(bounds, key, where)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ProfileError(f"{where}.{key} 必须是 [下限, 上限]")
        lo = _number(pair[0], f"{where}.{key}[0]", minimum=0.0)
        hi = _number(pair[1], f"{where}.{key}[1]", minimum=0.0)
        if lo >= hi:
            raise ProfileError(f"{where}.{key} 下限必须小于上限")




def _validate_model_parameters(mp):
    """校验 model_parameters 覆盖项：接触率、校准因子、传播指数、免疫与变异参数。

    全部键可选；未出现的键由 derive 取默认值（见 schema 常量与 diseases/README.md）。
    """
    if not isinstance(mp, dict):
        raise ProfileError("model_parameters 必须是 JSON 对象")
    if "contact_rate_per_frame" in mp:
        _number(mp["contact_rate_per_frame"], "model_parameters.contact_rate_per_frame",
                minimum=0.01)
    if "r0_calibration_factor" in mp:
        _number(mp["r0_calibration_factor"], "model_parameters.r0_calibration_factor",
                minimum=0.01)
    if "transmit_base_prob" in mp:
        _number(mp["transmit_base_prob"], "model_parameters.transmit_base_prob",
                minimum=0.0001, maximum=1.0)
    if "transmit_beta_exponent" in mp:
        _number(mp["transmit_beta_exponent"], "model_parameters.transmit_beta_exponent",
                minimum=0.01)
    if "immunity_gain_per_infection" in mp:
        _number(mp["immunity_gain_per_infection"],
                "model_parameters.immunity_gain_per_infection", minimum=0.0, maximum=1.0)
    if "immunity_effectiveness" in mp:
        _number(mp["immunity_effectiveness"], "model_parameters.immunity_effectiveness",
                minimum=0.0)
    if "immunity_waning_threshold" in mp:
        _number(mp["immunity_waning_threshold"],
                "model_parameters.immunity_waning_threshold", minimum=0.0, maximum=1.0)
    if "incubation_self_recovery_rate" in mp:
        _number(mp["incubation_self_recovery_rate"],
                "model_parameters.incubation_self_recovery_rate", minimum=0.0, maximum=1.0)
    if "mutation" in mp:
        mut = mp["mutation"]
        if not isinstance(mut, dict):
            raise ProfileError("model_parameters.mutation 必须是对象")
        if "enabled" in mut and not isinstance(mut["enabled"], bool):
            raise ProfileError("model_parameters.mutation.enabled 必须是 true/false")
        if "rate" in mut:
            _number(mut["rate"], "model_parameters.mutation.rate", minimum=0.0, maximum=1.0)
        if "scale" in mut:
            _number(mut["scale"], "model_parameters.mutation.scale", minimum=0.0)
        if "bounds" in mut:
            _validate_bounds(mut["bounds"], "model_parameters.mutation.bounds")




def _validate_curve(curve, where):
    """校验传染力曲线：≥2 个 ``[day, multiplier]`` 点、day 严格递增、乘子非负。

    day 单位是"相对发病日的天"（可为负，表示发病前），乘子无单位（1 = 基准传染力）。
    """
    if not isinstance(curve, (list, tuple)) or len(curve) < 2:
        raise ProfileError(f"{where} 至少需要 2 个 [day, multiplier] 点")
    prev_x = None
    for i, point in enumerate(curve):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ProfileError(f"{where}[{i}] 必须是 [day, multiplier]")
        x = _number(point[0], f"{where}[{i}][0]")
        _number(point[1], f"{where}[{i}][1]", minimum=0.0)
        if prev_x is not None and x <= prev_x:
            raise ProfileError(f"{where} 的 day 必须严格递增")
        prev_x = x




def _validate_age_table(table, where):
    """校验 9 档年龄相对表：键集合必须与 ``AGE_BANDS`` 完全一致，值非负。

    不变式：相对值由主程序按真实人口分布归一化为均值 1，因此这里只要求相对关系
    良定义——不校验总和，也不要求跨表一致。
    """
    if not isinstance(table, dict):
        raise ProfileError(f"{where} 必须是对象，如 {{\"0-9\": 0.002, ..., \"80+\": 1.0}}")
    missing = [b for b in AGE_BANDS if b not in table]
    unknown = sorted(set(table) - set(AGE_BANDS))
    if missing:
        raise ProfileError(f"{where} 缺少年龄档: {', '.join(missing)}")
    if unknown:
        raise ProfileError(f"{where} 含未知年龄档: {', '.join(unknown)}")
    for band in AGE_BANDS:
        _number(table[band], f"{where}.{band}", minimum=0.0)




def _validate_natural_history(nh):
    """校验 natural_history 段：无症状比例、隔离、住院病程、曲线与三张年龄表。

    不变式：``asymptomatic_fraction > 0`` 时必须给出无症状天数与无症状传染力曲线，
    否则派生出的 R0 会漏掉一条传播路径；概率量限制在 [0, 0.99]，避免退化成确定事件。
    """
    if not isinstance(nh, dict):
        raise ProfileError("natural_history 必须是 JSON 对象")
    a = nh.get("asymptomatic_fraction", 0.0)
    a = _number(a, "natural_history.asymptomatic_fraction", minimum=0.0, maximum=0.99)
    asymp_days = _number(nh.get("asymptomatic_days", 0.0),
                         "natural_history.asymptomatic_days", minimum=0.0)
    _number(nh.get("isolation_probability", 0.0),
            "natural_history.isolation_probability", minimum=0.0, maximum=1.0)
    _number(nh.get("isolation_delay_days", 0.0),
            "natural_history.isolation_delay_days", minimum=0.0)
    _number(nh.get("isolation_transmission_mult", 1.0),
            "natural_history.isolation_transmission_mult", minimum=0.0)
    _number(nh.get("hospital_delay_days", 0.0),
            "natural_history.hospital_delay_days", minimum=0.0)
    _number(nh.get("ward_days", 7.0), "natural_history.ward_days", minimum=0.5)
    _number(nh.get("hospital_death_share", 0.7),
            "natural_history.hospital_death_share", minimum=0.0, maximum=1.0)
    _number(nh.get("untreated_death_mult", 1.0),
            "natural_history.untreated_death_mult", minimum=1.0)
    if "infectivity_curve" in nh:
        _validate_curve(nh["infectivity_curve"], "natural_history.infectivity_curve")
    if "asymptomatic_infectivity_curve" in nh:
        _validate_curve(nh["asymptomatic_infectivity_curve"],
                        "natural_history.asymptomatic_infectivity_curve")
    if a > 0.0:
        if asymp_days <= 0.0:
            raise ProfileError(
                "asymptomatic_fraction > 0 时 natural_history.asymptomatic_days 必须 > 0")
        if "asymptomatic_infectivity_curve" not in nh:
            raise ProfileError(
                "asymptomatic_fraction > 0 时需要 natural_history.asymptomatic_infectivity_curve")
    for key in ("age_ifr_relative", "age_hospitalization_relative", "age_susceptibility"):
        if key in nh:
            _validate_age_table(nh[key], f"natural_history.{key}")




def _validate_medical(medical):
    """校验 medical 段：床位密度（beds_per_1000）与每 10 万人医院数，均不得为负。"""
    if not isinstance(medical, dict):
        raise ProfileError("medical 必须是 JSON 对象")
    _number(medical.get("beds_per_1000", 0.0),
            "medical.beds_per_1000", minimum=0.0)
    _number(medical.get("n_hospitals_per_100k", 0.0),
            "medical.n_hospitals_per_100k", minimum=0.0)




def _validate_interventions(interventions):
    """校验 v3 社会响应段：四个子块（检测/追踪/行为/报告）的数值区间、布尔与未知键。

    顶层只允许 ``enabled`` + 四个子块；子块内键集合来自 schema 默认值表，未知键一律拒绝。
    缺省键不在这里报错——derive 会按默认值补齐（见 INTERVENTIONS_DEFAULTS）。
    """
    if not isinstance(interventions, dict):
        raise ProfileError("interventions 必须是 JSON 对象")
    unknown = sorted(set(interventions) -
                     {"enabled", "testing", "tracing", "behavior", "reporting"})
    if unknown:
        raise ProfileError(f"interventions 含未知块: {', '.join(unknown)}")
    if "enabled" in interventions and not isinstance(interventions["enabled"], bool):
        raise ProfileError("interventions.enabled 必须是 true/false")

    for section in ("testing", "tracing", "behavior", "reporting"):
        if section not in interventions:
            continue
        block = interventions[section]
        where = f"interventions.{section}"
        if not isinstance(block, dict):
            raise ProfileError(f"{where} 必须是 JSON 对象")
        unknown = sorted(set(block) - _INTERVENTION_KEYS[section])
        if unknown:
            raise ProfileError(f"{where} 含未知键: {', '.join(unknown)}")
        if "enabled" in block and not isinstance(block["enabled"], bool):
            raise ProfileError(f"{where}.enabled 必须是 true/false")

        if section == "testing":
            _number(block.get("test_probability_symptomatic", 0.0),
                    f"{where}.test_probability_symptomatic", minimum=0.0, maximum=1.0)
            _number(block.get("test_delay_days", 0.0),
                    f"{where}.test_delay_days", minimum=0.0)
            _number(block.get("test_capacity_per_100k_per_day", 0.0),
                    f"{where}.test_capacity_per_100k_per_day", minimum=0.0)
            _number(block.get("test_sensitivity", 1.0),
                    f"{where}.test_sensitivity", minimum=0.0, maximum=1.0)
            _number(block.get("isolation_adherence", 0.0),
                    f"{where}.isolation_adherence", minimum=0.0, maximum=1.0)
            _number(block.get("isolation_delay_days", 0.0),
                    f"{where}.isolation_delay_days", minimum=0.0)
        elif section == "tracing":
            _number(block.get("trace_probability", 0.0),
                    f"{where}.trace_probability", minimum=0.0, maximum=1.0)
            _number(block.get("trace_delay_days", 0.0),
                    f"{where}.trace_delay_days", minimum=0.0)
            if "trace_household" in block and not isinstance(block["trace_household"], bool):
                raise ProfileError(f"{where}.trace_household 必须是 true/false")
            _number(block.get("trace_venue_fraction", 0.0),
                    f"{where}.trace_venue_fraction", minimum=0.0, maximum=1.0)
            _number(block.get("quarantine_days", 1.0),
                    f"{where}.quarantine_days", minimum=0.01)
            _number(block.get("quarantine_adherence", 0.0),
                    f"{where}.quarantine_adherence", minimum=0.0, maximum=1.0)
            _number(block.get("quarantine_transmission_mult", 1.0),
                    f"{where}.quarantine_transmission_mult", minimum=0.0, maximum=1.0)
        elif section == "behavior":
            _number(block.get("response_strength", 0.0),
                    f"{where}.response_strength", minimum=0.0, maximum=1.0)
            _number(block.get("response_threshold_per_100k", 0.0),
                    f"{where}.response_threshold_per_100k", minimum=0.0)
            _number(block.get("response_scale_per_100k", 1.0),
                    f"{where}.response_scale_per_100k", minimum=0.01)
            _number(block.get("response_delay_days", 0.0),
                    f"{where}.response_delay_days", minimum=0.0)
            _number(block.get("fatigue_days", 1.0),
                    f"{where}.fatigue_days", minimum=0.01)
            _number(block.get("fatigue_strength", 1.0),
                    f"{where}.fatigue_strength", minimum=0.0)
            _number(block.get("min_contact_mult", 0.0),
                    f"{where}.min_contact_mult", minimum=0.0, maximum=1.0)
        elif section == "reporting":
            _number(block.get("reporting_fraction", 0.0),
                    f"{where}.reporting_fraction", minimum=0.0, maximum=1.0)
            _number(block.get("reporting_delay_days", 0.0),
                    f"{where}.reporting_delay_days", minimum=0.0)
            if "reporting_symptomatic_only" in block and \
                    not isinstance(block["reporting_symptomatic_only"], bool):
                raise ProfileError(f"{where}.reporting_symptomatic_only 必须是 true/false")




def _validate_heterogeneity(block):
    """校验 v4 个体异质性块：enabled、传染力 sigma、依从性均值与标准差。

    不变式：未知键一律拒绝（v4 块连 ``note`` 之类的说明子键也不允许），
    区间：sigma∈[0, 3]、compliance_mean∈[0, 1]、compliance_sd∈[0, 0.5]。
    """
    if not isinstance(block, dict):
        raise ProfileError("heterogeneity 必须是 JSON 对象")
    where = "heterogeneity"
    unknown = sorted(set(block) - set(HETEROGENEITY_DEFAULTS))
    if unknown:
        raise ProfileError(f"{where} 含未知键: {', '.join(unknown)}")
    if "enabled" in block and not isinstance(block["enabled"], bool):
        raise ProfileError(f"{where}.enabled 必须是 true/false")
    _number(block.get("infectiousness_sigma", 0.0),
            f"{where}.infectiousness_sigma", minimum=0.0, maximum=3.0)
    _number(block.get("compliance_mean", 1.0),
            f"{where}.compliance_mean", minimum=0.0, maximum=1.0)
    _number(block.get("compliance_sd", 0.0),
            f"{where}.compliance_sd", minimum=0.0, maximum=0.5)




def _validate_mixing(block):
    """校验 v4 接触混合块：家庭接触乘子、年龄同配、症状居家比例的取值区间。

    区间：household_contact_mult∈[1, 10]、age_assortativity∈[0, 0.9]、
    symptom_stay_home_fraction∈[0, 1]；未知键一律拒绝。
    """
    if not isinstance(block, dict):
        raise ProfileError("mixing 必须是 JSON 对象")
    where = "mixing"
    unknown = sorted(set(block) - set(MIXING_DEFAULTS))
    if unknown:
        raise ProfileError(f"{where} 含未知键: {', '.join(unknown)}")
    if "enabled" in block and not isinstance(block["enabled"], bool):
        raise ProfileError(f"{where}.enabled 必须是 true/false")
    _number(block.get("household_contact_mult", 1.0),
            f"{where}.household_contact_mult", minimum=1.0, maximum=10.0)
    _number(block.get("age_assortativity", 0.0),
            f"{where}.age_assortativity", minimum=0.0, maximum=0.9)
    _number(block.get("symptom_stay_home_fraction", 0.0),
            f"{where}.symptom_stay_home_fraction", minimum=0.0, maximum=1.0)




def _validate_overrides(overrides):
    """校验 v1 简档的 model_overrides：键必须在 ``ALLOWED_OVERRIDES`` 白名单内。

    ``MUTATION_BOUNDS`` 走边界校验；其余值必须是有限非负数——概率上界留给
    apply._validate_applied 二次检查，因为简档是"绕过派生"的直写路径。
    """
    if not isinstance(overrides, dict):
        raise ProfileError("model_overrides 必须是 JSON 对象")
    unknown = sorted(set(overrides) - ALLOWED_OVERRIDES)
    if unknown:
        raise ProfileError(f"model_overrides 含未知键: {', '.join(unknown)}")
    for key, value in overrides.items():
        where = f"model_overrides.{key}"
        if key == "MUTATION_BOUNDS":
            _validate_bounds(value, where)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(f"{where} 必须是数字，实际 {value!r}")
        if not math.isfinite(float(value)):
            raise ProfileError(f"{where} 必须是有限数")
        if float(value) < 0.0:
            raise ProfileError(f"{where} 不能为负")




def validate_profile(profile, expected_id=None):
    """校验整份档案（版本、身份、时间、人口、各可选块及跨块依赖），无返回值。

    跨块不变式：natural_history/medical 依赖 epidemiology；interventions 需 v≥3，
    heterogeneity/mixing 需 v≥4；启用 v2/v3 块时必须提供自然史与流行病学段。
    ``expected_id`` 由 loader 传入，用于核对 profile_id 与文件名一致。
    """
    if not isinstance(profile, dict):
        raise ProfileError("档案顶层必须是 JSON 对象")
    schema_version = _field(profile, "schema_version", "profile")
    if schema_version not in SCHEMA_VERSIONS:
        raise ProfileError(
            f"schema_version 必须是 {SCHEMA_VERSIONS} 之一（实际 {schema_version!r}）")
    pid = _field(profile, "profile_id", "profile")
    if not isinstance(pid, str) or not pid.strip():
        raise ProfileError("profile.profile_id 必须是非空字符串")
    if expected_id is not None and pid != expected_id:
        raise ProfileError(f"profile_id '{pid}' 与文件名 '{expected_id}.json' 不一致")
    label = _field(profile, "label", "profile")
    if not isinstance(label, str) or not label.strip():
        raise ProfileError("profile.label 必须是非空字符串")

    time = _field(profile, "time", "profile")
    if not isinstance(time, dict):
        raise ProfileError("time 必须是 JSON 对象")
    _number(_field(time, "frames_per_day", "time"), "time.frames_per_day",
            minimum=1, integer=True)
    if "simulation_days" not in time and "simulation_frames" not in time:
        raise ProfileError("time 需要 simulation_days 或 simulation_frames")
    if "simulation_days" in time:
        _number(time["simulation_days"], "time.simulation_days", minimum=0.01)
    if "simulation_frames" in time:
        _number(time["simulation_frames"], "time.simulation_frames", minimum=1, integer=True)

    pop = profile.get("population", {})
    if not isinstance(pop, dict):
        raise ProfileError("population 必须是 JSON 对象")
    if "initial_immune_fraction" in pop:
        _number(pop["initial_immune_fraction"], "population.initial_immune_fraction",
                minimum=0.0, maximum=1.0)
    if "initial_immune_level" in pop:
        _number(pop["initial_immune_level"], "population.initial_immune_level",
                minimum=0.0, maximum=1.0)
    if "initial_seeds" in pop:
        _number(pop["initial_seeds"], "population.initial_seeds",
                minimum=0, integer=True)

    has_epi = "epidemiology" in profile
    has_overrides = "model_overrides" in profile
    if not has_epi and not has_overrides:
        raise ProfileError("档案必须包含 epidemiology 或 model_overrides 之一")
    if has_epi:
        _validate_epidemiology(profile["epidemiology"])
    if has_overrides:
        _validate_overrides(profile["model_overrides"])
    if "model_parameters" in profile:
        _validate_model_parameters(profile["model_parameters"])
    if "natural_history" in profile:
        if not has_epi:
            raise ProfileError("natural_history 需要 epidemiology 段（不支持 model_overrides 档案）")
        _validate_natural_history(profile["natural_history"])
    if "medical" in profile:
        if not has_epi:
            raise ProfileError("medical 需要 epidemiology 段（不支持 model_overrides 档案）")
        _validate_medical(profile["medical"])
    if "interventions" in profile:
        if int(schema_version) < 3:
            raise ProfileError("interventions 需要 schema_version >= 3")
        _validate_interventions(profile["interventions"])
        if profile["interventions"].get("enabled", True):
            if "natural_history" not in profile:
                raise ProfileError("interventions 启用时需要 natural_history 段")
            if not has_epi:
                raise ProfileError("interventions 需要 epidemiology 段")
    if "heterogeneity" in profile:
        if int(schema_version) < 4:
            raise ProfileError("heterogeneity 需要 schema_version >= 4")
        _validate_heterogeneity(profile["heterogeneity"])
        if profile["heterogeneity"].get("enabled", False):
            if "natural_history" not in profile:
                raise ProfileError("heterogeneity 启用时需要 natural_history 段")
            if not has_epi:
                raise ProfileError("heterogeneity 需要 epidemiology 段")
    if "mixing" in profile:
        if int(schema_version) < 4:
            raise ProfileError("mixing 需要 schema_version >= 4")
        _validate_mixing(profile["mixing"])
    if "sources" in profile and not isinstance(profile["sources"], dict):
        raise ProfileError("sources 必须是 JSON 对象")
    if "notes" in profile and not isinstance(profile["notes"], list):
        raise ProfileError("notes 必须是数组")
