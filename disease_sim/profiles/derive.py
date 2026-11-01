"""档案派生：把"天 / 概率"的流行病学参数换算成模型每帧参数（纯函数，不改 CONFIG）。

职责：由 epidemiology/natural_history/medical/interventions 计算 beta/gamma/mu、
每帧 hazard、R0 自洽值、免疫衰减与帧数；被关闭的块给出中性默认值。
非职责：不读文件（loader.py）、不校验用户输入（validate.py）、不写 CONFIG（apply.py）。
入口：``derive_model_parameters(profile, config)`` 返回 derived dict；本模块只读 config
里的 ``DAY_FRAMES`` / ``TRANSMIT_BASE_PROB`` 等默认值，写 CONFIG 与二次校验在 apply.py。
不变式：输入按"天"记录，输出时间量一律换算成"帧"（乘 ``frames_per_day``）；
派生是确定的纯计算（R0 用数值积分而非随机），同一档案 + 同一 config 必得同一结果。
"""

from __future__ import annotations

import math
import copy

from .schema import (
    AGE_BANDS,
    CONTACT_RATE_PER_FRAME,
    DEATH_HAZARD_FRACTION,
    FRAMES_PER_DAY_KEY,
    INTERVENTIONS_DEFAULTS,
    LEGACY_ASYMPTOMATIC_CURVE,
    LEGACY_INFECTIVITY_CURVE,
    ProfileError,
    TRANSMIT_BETA_EXPONENT,
)


def _interp_curve_value(points, x):
    """标量版分段线性插值：返回 ``points`` 在 x 处的插值（域外取端点值）。

    仅在导入期做数值积分时使用，标量实现足够；支持负 day（发病前窗口）。
    """
    if x <= points[0][0]:
        return float(points[0][1])
    if x >= points[-1][0]:
        return float(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            if x1 == x0:
                return float(y1)
            return float(y0) + (float(y1) - float(y0)) * (x - x0) / (x1 - x0)
    return float(points[-1][1])




def _integrate_curve(points, lo_day, hi_day, fpd, *,
                     iso_probability=0.0, iso_delay_day=0.0,
                     iso_mult=1.0, apply_isolation=False):
    """把 [lo_day, hi_day] 上的传染力曲线按帧离散积分（中点法）。

    返回 ``(总积分, 发病前积分)``，单位为"帧 × 相对传染力"。
    不变式：帧数向上取整，短窗口至少 1 帧；``apply_isolation`` 时发病后
    ``t >= iso_delay_day`` 的帧乘 ``(1-p) + p*iso_mult``（隔离概率的期望），
    供 R0 数值积分使用——隔离率因此会直接压低有效传染期。
    """
    n = max(1, int(math.ceil((hi_day - lo_day) * fpd)))
    total = 0.0
    pre = 0.0
    for k in range(n):
        day = lo_day + (k + 0.5) / fpd
        value = _interp_curve_value(points, day)
        factor = 1.0
        if apply_isolation and day > 0.0 and day >= iso_delay_day:
            factor = (1.0 - iso_probability) + iso_probability * iso_mult
        contribution = value * factor
        total += contribution
        if day < 0.0:
            pre += contribution
    return total, pre




def _age_table_or_default(nh, key):
    """取 9 档年龄相对表；档案省略该表时返回全 1（相对风险中性，不改变总体水平）。"""
    table = nh.get(key)
    if table is None:
        return {band: 1.0 for band in AGE_BANDS}
    return {band: float(table[band]) for band in AGE_BANDS}




def _derive_natural_history(profile, config, fpd, t_inc, t_symp, ifr,
                            contact_rate, calibration, r0, base_prob, exponent,
                            hosp_rate, beds_per_1000):
    """schema v2：由自然史/医疗容量字段派生每帧 hazard（含 R0 数值积分）。

    输入时长均为帧（``t_inc`` / ``t_symp``），返回 derived 子 dict（含 ``r0_check`` 自洽值）。
    不变式：社区/住院两条死亡路径按 IFR 守恒拆分；各路径"每帧死亡 + 康复"之和
    等于 1/疗程帧数 ≤ 1，因此逐帧概率不会超过 1。
    """
    nh = profile.get("natural_history") or {}
    medical = profile.get("medical") or {}

    a = float(nh.get("asymptomatic_fraction", 0.0))
    if a >= 1.0:
        raise ProfileError("natural_history.asymptomatic_fraction 必须小于 1")
    asymp_days = float(nh.get("asymptomatic_days", 0.0))
    sym_curve = nh.get("infectivity_curve") or LEGACY_INFECTIVITY_CURVE
    asym_curve = nh.get("asymptomatic_infectivity_curve") or LEGACY_ASYMPTOMATIC_CURVE
    iso_probability = float(nh.get("isolation_probability", 0.0))
    iso_delay_day = float(nh.get("isolation_delay_days", 0.0))
    iso_mult = float(nh.get("isolation_transmission_mult", 1.0))
    hosp_delay_frames = float(nh.get("hospital_delay_days", 0.0)) * fpd
    ward_days = float(nh.get("ward_days", 7.0))
    ward_share = float(nh.get("hospital_death_share", 0.7))
    untreated_mult = float(nh.get("untreated_death_mult", 1.0))
    age_ifr = _age_table_or_default(nh, "age_ifr_relative")
    age_hosp = _age_table_or_default(nh, "age_hospitalization_relative")
    age_suscept = _age_table_or_default(nh, "age_susceptibility")
    n_hospitals = float(medical.get("n_hospitals_per_100k", 0.0))

    # 不变式：死亡在"社区/住院"两条路径间拆分，保证总体每例感染 IFR 与档案一致
    cfr_symp = ifr / (1.0 - a) if a > 0.0 else ifr
    if cfr_symp >= 1.0:
        raise ProfileError(
            f"IFR/(1-asymptomatic_fraction) = {cfr_symp:.3f} >= 1，请检查档案")
    if hosp_rate <= 0.0:
        p_comm_death = cfr_symp
        p_ward_death = 0.0
    elif hosp_rate >= 1.0:
        raise ProfileError("hospitalization_rate 必须小于 1")
    else:
        p_ward_death = cfr_symp * ward_share / hosp_rate
        p_comm_death = cfr_symp * (1.0 - ward_share) / (1.0 - hosp_rate)
    if p_ward_death >= 1.0 or p_comm_death >= 1.0:
        raise ProfileError(
            f"死亡概率超出 [0,1)：社区={p_comm_death:.3f}, 住院={p_ward_death:.3f}；"
            "请检查 IFR/hospitalization_rate/hospital_death_share")
    comm_frames = max(1.0, t_symp)
    ward_frames = max(1.0, ward_days * fpd)
    community_death_hazard = p_comm_death / comm_frames
    community_recovery_rate = (1.0 - p_comm_death) / comm_frames
    ward_death_hazard = p_ward_death / ward_frames
    ward_recovery_rate = (1.0 - p_ward_death) / ward_frames
    # 注意：需要住院但无床位者按"本应住院疗程内的死亡风险"放大；
    # 上限 0.95 留出存活可能，避免医疗崩溃时模型出现必然死亡。
    untreated_base = p_ward_death if p_ward_death > 0.0 else cfr_symp
    untreated_death_prob = min(0.95, untreated_base * untreated_mult)
    untreated_death_hazard = untreated_death_prob / ward_frames

    # 不变式：R0 数值积分的症状前窗口受平均潜伏期限制，发病后窗口受症状期限制
    # （即截断曲线定义域；sym_hi 至少比 sym_lo 多 0.1 天，保证曲线形状可被采到）。
    sym_lo = max(float(sym_curve[0][0]), -t_inc / fpd)
    sym_hi = max(sym_lo + 0.1, min(float(sym_curve[-1][0]), t_symp / fpd))
    sym_total, sym_pre = _integrate_curve(
        sym_curve, sym_lo, sym_hi, fpd,
        iso_probability=iso_probability, iso_delay_day=iso_delay_day,
        iso_mult=iso_mult, apply_isolation=True)
    asym_total = 0.0
    if a > 0.0 and asymp_days > 0.0:
        asym_lo = max(0.0, float(asym_curve[0][0]))
        asym_hi = min(float(asym_curve[-1][0]), asymp_days)
        if asym_hi > asym_lo:
            asym_total, _ = _integrate_curve(asym_curve, asym_lo, asym_hi, fpd)
    t_eff_frames = (1.0 - a) * sym_total + a * asym_total
    if t_eff_frames <= 0.0:
        raise ProfileError("自然史参数导致有效传染期 <= 0，请检查传染力曲线/病程")
    p_symp = r0 * calibration / (contact_rate * t_eff_frames)
    beta = (p_symp / base_prob) ** (1.0 / exponent)
    presym_fraction = sym_pre / sym_total if sym_total > 0.0 else 0.0

    return {
        "beta": beta,
        "p_symp": p_symp,
        "t_eff_frames": t_eff_frames,
        "asymptomatic_fraction": a,
        "asymptomatic_days_frames": asymp_days * fpd,
        "infectivity_curve": [list(map(float, p)) for p in sym_curve],
        "asymptomatic_infectivity_curve": [list(map(float, p)) for p in asym_curve],
        "isolation_probability": iso_probability,
        "isolation_delay_frames": iso_delay_day * fpd,
        "isolation_transmission_mult": iso_mult,
        "hospital_delay_frames": hosp_delay_frames,
        "ward_days_frames": ward_frames,
        "hospital_death_share": ward_share,
        "untreated_death_mult": untreated_mult,
        "untreated_death_prob": untreated_death_prob,
        "untreated_death_hazard": untreated_death_hazard,
        "community_death_hazard": community_death_hazard,
        "community_recovery_rate": community_recovery_rate,
        "ward_death_hazard": ward_death_hazard,
        "ward_recovery_rate": ward_recovery_rate,
        "beds_per_1000": float(beds_per_1000),
        "n_hospitals_per_100k": n_hospitals,
        "age_ifr_relative": age_ifr,
        "age_hospitalization_relative": age_hosp,
        "age_susceptibility": age_suscept,
        "cfr_symp": cfr_symp,
        "p_ward_death": p_ward_death,
        "p_comm_death": p_comm_death,
        "presymptomatic_curve_fraction": presym_fraction,
        "r0_check": contact_rate * t_eff_frames * p_symp,
    }




def _derive_interventions(interventions, fpd):
    """schema v3：把 interventions 段化为每帧参数（关闭时给全量中性默认值）。

    返回扁平 dict，键名对应 CONFIG 的大写键（写入见 apply._apply_interventions）。
    不变式：总开关关闭 -> enabled=False、各子机制关闭且参数取"不生效"中性值；
    子块缺省键用 INTERVENTIONS_DEFAULTS 补齐，使无该块的老档案行为等价。
    """
    if not isinstance(interventions, dict):
        raise ProfileError("interventions 必须是 JSON 对象")
    enabled = bool(interventions.get("enabled", True))

    result = {
        "enabled": False,
        "testing_enabled": False,
        "test_probability_symptomatic": 0.0,
        "test_delay_frames": 0.0,
        "test_capacity_per_100k_per_day": 0.0,
        "test_sensitivity": 1.0,
        "test_isolation_adherence": 0.0,
        "test_isolation_delay_frames": 0.0,
        "tracing_enabled": False,
        "trace_probability": 0.0,
        "trace_delay_frames": 0.0,
        "trace_household": True,
        "trace_venue_fraction": 0.0,
        "quarantine_days_frames": 0.0,
        "quarantine_adherence": 0.0,
        "quarantine_transmission_mult": 1.0,
        "behavior_enabled": False,
        "behavior_response_strength": 0.0,
        "behavior_threshold_per_100k": 0.0,
        "behavior_scale_per_100k": 1.0,
        "behavior_response_delay_frames": 0.0,
        "behavior_fatigue_days": 1.0,
        "behavior_fatigue_strength": 0.0,
        "behavior_min_contact_mult": 0.0,
        "reporting_enabled": False,
        "reporting_fraction": 0.0,
        "reporting_delay_frames": 0.0,
        "reporting_symptomatic_only": True,
    }
    if not enabled:
        return result

    def _block(section):
        """取一个社会响应子块：缺省键用默认值补齐，显式 ``enabled=false`` 时返回 None。"""
        raw = interventions.get(section) or {}
        if not isinstance(raw, dict):
            raise ProfileError(f"interventions.{section} 必须是 JSON 对象")
        if not raw.get("enabled", True):
            return None
        merged = dict(INTERVENTIONS_DEFAULTS[section])
        merged.update(raw)
        return merged

    testing = _block("testing")
    if testing is not None:
        result.update(
            testing_enabled=True,
            test_probability_symptomatic=float(testing["test_probability_symptomatic"]),
            test_delay_frames=float(testing["test_delay_days"]) * fpd,
            test_capacity_per_100k_per_day=float(
                testing["test_capacity_per_100k_per_day"]),
            test_sensitivity=float(testing["test_sensitivity"]),
            test_isolation_adherence=float(testing["isolation_adherence"]),
            test_isolation_delay_frames=float(testing["isolation_delay_days"]) * fpd,
        )

    tracing = _block("tracing")
    if tracing is not None:
        result.update(
            tracing_enabled=True,
            trace_probability=float(tracing["trace_probability"]),
            trace_delay_frames=float(tracing["trace_delay_days"]) * fpd,
            trace_household=bool(tracing["trace_household"]),
            trace_venue_fraction=float(tracing["trace_venue_fraction"]),
            quarantine_days_frames=float(tracing["quarantine_days"]) * fpd,
            quarantine_adherence=float(tracing["quarantine_adherence"]),
            quarantine_transmission_mult=float(
                tracing["quarantine_transmission_mult"]),
        )

    behavior = _block("behavior")
    if behavior is not None:
        result.update(
            behavior_enabled=True,
            behavior_response_strength=float(behavior["response_strength"]),
            behavior_threshold_per_100k=float(
                behavior["response_threshold_per_100k"]),
            behavior_scale_per_100k=float(behavior["response_scale_per_100k"]),
            behavior_response_delay_frames=float(behavior["response_delay_days"]) * fpd,
            behavior_fatigue_days=float(behavior["fatigue_days"]),
            behavior_fatigue_strength=float(behavior["fatigue_strength"]),
            behavior_min_contact_mult=float(behavior["min_contact_mult"]),
        )

    reporting = _block("reporting")
    if reporting is not None:
        result.update(
            reporting_enabled=True,
            reporting_fraction=float(reporting["reporting_fraction"]),
            reporting_delay_frames=float(reporting["reporting_delay_days"]) * fpd,
            reporting_symptomatic_only=bool(reporting["reporting_symptomatic_only"]),
        )

    result["enabled"] = True
    return result




def derive_model_parameters(profile: dict, config: dict) -> dict:
    """把流行病学档案派生为模型每帧参数（纯函数，不修改 config）。

    时长按 fpd 换算；beta/gamma/mu 与症状前传染力 m 由闭式公式得到；自然史存在时
    改走数值积分的 R0 反解；随后叠加免疫衰减、变异与帧数。model_overrides 档案不走这里。
    不变式：``time.frames_per_day`` 必须等于 CONFIG 的 ``DAY_FRAMES``（容差 1e-9
    仅为吸收 JSON 浮点误差）；无自然史时用旧版 ``t_eff = t_symp + m * t_inc``。
    """
    fpd = float(config[FRAMES_PER_DAY_KEY])
    time = profile["time"]
    profile_fpd = float(time["frames_per_day"])
    if abs(profile_fpd - fpd) > 1e-9:
        raise ProfileError(
            f"time.frames_per_day={profile_fpd:g} 与模型 {FRAMES_PER_DAY_KEY}={fpd:g} 不一致")

    if "epidemiology" not in profile:
        raise ProfileError("该档案没有 epidemiology 段，无法派生模型参数（请用 model_overrides）")

    epi = profile["epidemiology"]
    mp = profile.get("model_parameters", {}) or {}
    inc = epi["incubation_days"]

    t_inc = float(inc["mean"]) * fpd
    inc_sd = float(inc.get("sd", 0.0)) * fpd
    inc_min = float(inc.get("min", 0.0)) * fpd
    t_symp = float(
        epi["infectious_days"].get("mean", epi["infectious_days"]["value"])) * fpd
    gamma = 1.0 / t_symp

    ifr = float(epi["ifr"]["value"])
    mu = ifr * gamma / (DEATH_HAZARD_FRACTION * (1.0 - ifr))

    presym = float(epi["presymptomatic_transmission_fraction"]["value"])
    if presym >= 1.0:
        raise ProfileError("presymptomatic_transmission_fraction 必须小于 1")
    m = presym * t_symp / ((1.0 - presym) * t_inc)

    contact_rate = float(mp.get("contact_rate_per_frame", CONTACT_RATE_PER_FRAME))
    calibration = float(mp.get("r0_calibration_factor", 1.0))
    r0 = float(epi["r0"]["value"])
    base_prob = float(mp.get("transmit_base_prob", config["TRANSMIT_BASE_PROB"]))
    exponent = float(mp.get("transmit_beta_exponent", TRANSMIT_BETA_EXPONENT))

    beds = float(epi["hospital_beds_per_1000"]["value"])
    hosp_rate = float(epi["hospitalization_rate"]["value"])
    medical = profile.get("medical") or {}
    beds_per_1000 = float(medical.get("beds_per_1000", beds))
    # 医疗崩溃阈值（每 10 万人）= 每 10 万人床位 / 住院率 = "住院需求正好占满床位"时的
    # 症状者规模；取 medical.beds_per_1000（若档案给了覆盖值），与下方输出的床位口径一致。
    # 【2026-11 审计修复】原式直接除 hosp_rate：hosp_rate=0 会抛 ZeroDivisionError；
    # 且重复用 epi 床位计算，忽略 medical 覆盖值（旧注释误称"值相同"）。
    medical_threshold = (beds_per_1000 * 10.0 / hosp_rate) if hosp_rate > 0.0 else 0.0

    natural = None
    if "natural_history" in profile:
        natural = _derive_natural_history(
            profile, config, fpd, t_inc, t_symp, ifr,
            contact_rate, calibration, r0, base_prob, exponent,
            hosp_rate, beds_per_1000)
        beta = natural["beta"]
        p_symp = natural["p_symp"]
        t_eff = natural["t_eff_frames"]
    else:
        t_eff = t_symp + m * t_inc
        p_symp = r0 * calibration / (contact_rate * t_eff)
        beta = (p_symp / base_prob) ** (1.0 / exponent)

    interventions = None
    if "interventions" in profile:
        if natural is None:
            raise ProfileError("interventions 需要 natural_history 启用")
        interventions = _derive_interventions(profile["interventions"], fpd)

    gain = float(mp.get("immunity_gain_per_infection", 0.6))
    threshold = float(mp.get("immunity_waning_threshold",
                             config["IMMUNITY_WANING_THRESHOLD"]))
    if gain <= threshold:
        raise ProfileError("immunity_gain_per_infection 必须大于 immunity_waning_threshold")
    immunity_days = float(epi["immunity_days"]["value"])
    decay = (gain - threshold) / (immunity_days * fpd)

    mutation = mp.get("mutation", {}) or {}
    mutation_enabled = bool(mutation.get("enabled", False))
    mutation_rate = float(mutation.get("rate", 0.0)) if mutation_enabled else 0.0
    mutation_scale = float(mutation.get("scale", config.get("MUTATION_SCALE", 0.15)))
    mutation_bounds = copy.deepcopy(
        mutation.get("bounds") or config.get("MUTATION_BOUNDS"))
    self_recovery = float(mp.get("incubation_self_recovery_rate", 0.0))

    if "simulation_frames" in time:
        frames = int(time["simulation_frames"])
    else:
        frames = int(round(float(time["simulation_days"]) * fpd))
    if frames < 1:
        raise ProfileError("模拟帧数必须至少为 1")

    derived = {
        "beta": beta,
        "gamma": gamma,
        "mu": mu,
        "incubation_mu": t_inc,
        "incubation_sd": inc_sd,
        "incubation_min": inc_min,
        "incubating_transmission_mult": m,
        "incubation_self_recovery": self_recovery,
        "mutation_enabled": mutation_enabled,
        "mutation_rate": mutation_rate,
        "mutation_scale": mutation_scale,
        "mutation_bounds": mutation_bounds,
        "immunity_gain": gain,
        "immunity_decay": decay,
        "medical_threshold_per_10k": medical_threshold,
        "beds_per_1000": beds_per_1000,
        "n_hospitals_per_100k": (natural or {}).get("n_hospitals_per_100k", 0.0),
        "hosp_rate": hosp_rate,
        "natural_history_enabled": natural is not None,
        "natural_history": natural,
        "interventions_enabled": bool(interventions and interventions["enabled"]),
        "interventions": interventions,
        "initial_immune_fraction": float(
            (profile.get("population") or {}).get("initial_immune_fraction", 0.0)),
        "initial_immune_level": float(
            (profile.get("population") or {}).get("initial_immune_level", 0.9)),
        "initial_seeds": int((profile.get("population") or {}).get("initial_seeds", 20)),
        "frames": frames,
        "r0_target": r0,
        "r0_calibration_factor": calibration,
        "r0_check": contact_rate * t_eff * p_symp,
        "p_symp": p_symp,
        "t_eff": t_eff,
        "contact_rate_per_frame": contact_rate,
        "serial_interval_days": (
            float(epi["serial_interval_days"]["value"])
            if "serial_interval_days" in epi else None),
        "symptomatic_fraction": (
            float(epi["symptomatic_fraction"]["value"])
            if "symptomatic_fraction" in epi else None),
    }
    if mutation_enabled:
        lo = float(mutation_bounds["beta"][0])
        hi = float(mutation_bounds["beta"][1])
        if not (lo <= beta <= hi):
            raise ProfileError(
                f"派生 beta={beta:.6g} 超出 MUTATION_BOUNDS.beta=[{lo}, {hi}]；"
                "请放宽 bounds 或关闭 mutation.enabled")
    return derived
