#!/usr/bin/env python3
"""疾病参数档案校验 / R0 校准工具：判断档案的派生参数是否自洽、R0 是否可信。

输入：diseases/*.json（由 disease_sim.profiles.loader 派生为每帧参数）；输出：控制台报告 + --json。
两条路径：
- 解析式（默认）：核对 beta/gamma/mu/潜伏期/帧数、CONFIG 与派生值一致（容差 1e-12~1e-9）、
  医疗阈值、变异边界；R0 用 Euler-Lotka 反解增长率，要求解析校验相对误差 <=1e-3、代际时间 3-8 天。
- 经验校准（--empirical）：10k 人口跑若干帧，对早期累计感染做对数线性拟合，经同一 Euler-Lotka
  反推经验 R0；与目标偏离 >30% 记 WARN（只告警，不影响进程退出码）。
退出码：0=正常运行；1=档案非法（ProfileError）或 --empirical 且 --frames<100。
已知局限：不感知 v3/v4 机制（社会响应、疫苗/治疗等），经验 R0 是"当前档案+默认开关"下的有效值，
不等于病原体本身的 R0；经验路径对播种位置敏感，所以测量前会把种子随机分散。
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from disease_sim import profiles as dp  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI：参数名与默认值（--frames 6000≈25 天、人口 10000）是文档/CI 引用的契约。"""
    parser = argparse.ArgumentParser(description="疾病参数档案校验 / R0 校准工具")
    parser.add_argument("--profile", default=None,
                        help="档案名（默认 %s）" % dp.DEFAULT_PROFILE)
    parser.add_argument("--empirical", action="store_true",
                        help="跑小规模仿真，拟合并报告早期增长率 / 经验 R0")
    parser.add_argument("--frames", type=int, default=6000,
                        help="empirical 仿真帧数（默认 6000 ≈ 25 天）")
    parser.add_argument("--population", type=int, default=10000,
                        help="empirical 人口（默认 10000）")
    parser.add_argument("--seed", type=int, default=1234,
                        help="随机种子（默认 1234）")
    parser.add_argument("--json", default=None, help="把结果写入 JSON 文件")
    return parser


def import_model(profile_name: str, population: int, seed: int):
    """以指定档案/人口/种子导入模型，返回 (module, 导入期 stdout)。

    参数走环境变量：入口在 import 期读 DISEASE_PROFILE/DISEASE_POPULATION/DISEASE_SEED 建世界，
    所以必须在 exec_module 之前写入，改参数只能靠重开进程而不是重载模块。
    加载失败（档案与模型不匹配）会抛 dp.ProfileError；入口文件缺失抛 SystemExit。
    """
    os.environ["DISEASE_PROFILE"] = str(profile_name)
    os.environ["DISEASE_POPULATION"] = str(int(population))
    os.environ["DISEASE_SEED"] = str(int(seed))
    os.environ.setdefault("MPLBACKEND", "Agg")
    path = ROOT / "simple_disease_simulation.py"
    if not path.is_file():
        raise SystemExit(f"[profile_check] 找不到模型文件: {path}")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        spec = importlib.util.spec_from_file_location("sim_profile_check", str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["sim_profile_check"] = module
        spec.loader.exec_module(module)
    return module, buffer.getvalue()


def _add(checks: list, name: str, ok: bool, detail: str) -> None:
    """追加一条解析式检查：{"name","ok","detail"}，供 _print_report 逐条打印与 JSON 落盘。"""
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _v2_kernel(mod):
    """按 v2 自然史配置构造每帧传染力核（E+I 症状曲线、A 无症状曲线、隔离衰减）。

    返回 (kernel, frames_per_day, t_inc_frames) 或 None（legacy 路径）。
    """
    cfg = mod.CONFIG
    if not cfg.get("NATURAL_HISTORY_ENABLED"):
        return None
    import numpy as np
    info = mod.PROFILE_INFO
    fpd = float(info["frames_per_day"])
    t_inc = float(info["derived"]["incubation_mu"])
    t_symp = 1.0 / float(info["derived"]["gamma"])
    a_frac = float(cfg["ASYMPTOMATIC_FRACTION"])
    a_days = float(cfg["ASYMPTOMATIC_DAYS_FRAMES"])
    iso_prob = float(cfg["ISOLATION_PROBABILITY"])
    iso_delay = float(cfg["ISOLATION_DELAY_FRAMES"])
    iso_mult = float(cfg["ISOLATION_TRANSMISSION_MULT"])

    def interp(points, x):
        arr = np.asarray(points, dtype=float)
        return np.interp(x, arr[:, 0], arr[:, 1], left=arr[0, 1], right=arr[-1, 1])

    # 核长取"5 个症状期/A 期"与 60 天中的最大者：确保尾部已衰减到可忽略，截断不改变 R0/代际估计。
    horizon = int(max(t_inc + t_symp * 5.0, a_days * 5.0, 60.0 * fpd))
    t = np.arange(horizon, dtype=float)
    sym = interp(cfg["INFECTIVITY_CURVE"], (t - t_inc) / fpd)
    t_post = np.maximum(t - t_inc, 0.0)
    # 隔离延迟期满后，被隔离比例 iso_prob 的个体传染力乘 iso_mult，其余不变。
    iso_factor = np.where(t_post >= iso_delay,
                          1.0 - iso_prob + iso_prob * iso_mult, 1.0)
    sym = sym * iso_factor
    sym = sym * np.where(t < t_inc, 1.0, np.exp(-(1.0 / (t_symp / fpd)) * (t_post / fpd)))
    asym = interp(cfg["ASYMPTOMATIC_INFECTIVITY_CURVE"], t / fpd)
    asym = asym * np.where(t < a_days, 1.0, 0.0)
    kernel = (1.0 - a_frac) * sym + a_frac * asym
    if not np.isfinite(kernel).all() or kernel.sum() <= 0:
        return None
    return kernel, fpd, t_inc


def check_analytic(mod) -> dict:
    """对已应用档案的模块做解析式自洽检查，返回报告 dict。

    检查：参数合法性（beta/gamma/mu/潜伏期/帧数/医疗阈值）、CONFIG 与 loader 派生值逐项一致、
    R0 解析自洽（相对误差 <=1e-3）、自然史核的代际时间（3-8 天）与 Euler-Lotka 增长率/倍增时间、
    变异边界。返回 {summary, frames_per_day, derived, analytic, checks}，checks 由 _print_report 打印。
    """
    info = mod.PROFILE_INFO
    cfg = mod.CONFIG
    derived = info["derived"]
    fpd = float(info["frames_per_day"])

    beta = float(derived["beta"])
    gamma = float(derived["gamma"])
    mu = float(derived["mu"])
    t_inc = float(derived["incubation_mu"])
    checks: list = []
    _add(checks, "beta > 0", beta > 0, f"beta={beta:.6g}")
    _add(checks, "gamma in (0,1)", 0.0 < gamma < 1.0, f"gamma={gamma:.6g}")
    _add(checks, "mu > 0", mu > 0, f"mu={mu:.6g}")
    _add(checks, "潜伏期 >= 1 帧", t_inc >= 1.0, f"incubation={t_inc:.0f} 帧")
    _add(checks, "模拟帧数 >= 1", int(info["frames"]) >= 1, f"frames={info['frames']}")

    same = (
        abs(float(cfg["BASE_TRANSMIT"]) - beta) <= 1e-12
        and abs(float(cfg["BASE_RECOVERY"]) - gamma) <= 1e-12
        and abs(float(cfg["BASE_DEATH"]) - mu) <= 1e-12
        and abs(float(cfg["INCUBATION_MU"]) - t_inc) <= 1e-9
        and int(cfg["FRAMES"]) == int(info["frames"])
    )
    _add(checks, "CONFIG 与派生值一致", same,
         "模型 CONFIG 已按档案应用（beta/gamma/mu/incubation/frames）")

    threshold = int(cfg["MEDICAL_THRESHOLD"])
    _add(checks, "医疗阈值 >= 1", threshold >= 1, f"MEDICAL_THRESHOLD={threshold}")

    r0_target = derived.get("r0_target")
    analytic = {"r0_target": r0_target, "r0_check": derived.get("r0_check"),
                "generation_time_frames": None, "doubling_time_days": None,
                "expected_growth_rate_per_frame": None}
    if r0_target is not None:
        r0_check = float(derived.get("r0_check", float("nan")))
        rel = abs(r0_check - r0_target) / max(r0_target, 1e-12)
        # 1e-3 是 loader 与解析式两条实现路径的浮点容差（同一公式、不同运算顺序）。
        _add(checks, "R0 解析自洽", rel <= 1e-3,
             f"R0 目标={r0_target:.4f}, 解析校验={r0_check:.6f}, 相对误差={rel:.2e}")
        import numpy as np
        kernel_info = _v2_kernel(mod)
        if kernel_info is not None:
            kernel, _fpd, _t_inc = kernel_info
            total = float(kernel.sum())
            tgrid = np.arange(kernel.size, dtype=float)
            t_gen = float((tgrid * kernel).sum() / total)
            analytic["kernel_effective_days"] = total / fpd
            analytic["generation_time_days"] = t_gen / fpd

            def _r0_of_r(rate: float) -> float:
                if rate <= 0:
                    return float("inf")
                return 1.0 / float(np.sum(kernel / total * np.exp(-rate * tgrid)))
        else:
            mult = float(derived["incubating_transmission_mult"])
            f_pre = mult * t_inc / (t_inc * mult + 1.0 / gamma) if t_inc > 0 else 0.0
            t_gen = f_pre * (t_inc / 2.0) + (1.0 - f_pre) * (t_inc + 1.0 / gamma)

            def _r0_of_r(rate: float) -> float:
                # Euler-Lotka 自洽：R0(r) = (mult·T_inc + T_symp) / L(r)
                integ = (mult * (1.0 - math.exp(-rate * t_inc)) / rate
                         + math.exp(-rate * t_inc) / (rate + gamma))
                return (mult * t_inc + 1.0 / gamma) / integ if integ > 0 else float("inf")

        if r0_target > 1.0:
            # Euler-Lotka 没有闭式解，二分 R0(r)=r0_target：80 次把区间缩到 1e-18 级，远超 1e-3 容差需求。
            lo, hi = 1e-9, 0.05
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _r0_of_r(mid) < r0_target:
                    lo = mid
                else:
                    hi = mid
            r_lotka = 0.5 * (lo + hi)
            analytic["generation_time_frames"] = t_gen
            analytic["expected_growth_rate_per_frame"] = r_lotka
            analytic["doubling_time_days"] = math.log(2.0) / (r_lotka * fpd)
            if kernel_info is not None:
                # 3-8 天是常见呼吸道病原体的代际时间量级；超出多半是核形状或"天/帧"单位写错。
                _add(checks, "代际时间 3-8 天量级", 3.0 <= t_gen / fpd <= 8.0,
                     f"平均代际时间≈{t_gen:.0f} 帧 ({t_gen / fpd:.1f} 天)")
            else:
                _add(checks, "代际时间 > 潜伏期", t_gen > t_inc,
                     f"平均代际时间≈{t_gen:.0f} 帧 ({t_gen / fpd:.1f} 天)")
        else:
            analytic["generation_time_frames"] = t_gen

    bounds = derived.get("mutation_bounds") or {}
    if derived.get("mutation_enabled"):
        lo_hi = bounds.get("beta")
        if lo_hi:
            in_bounds = float(lo_hi[0]) <= beta <= float(lo_hi[1])
            _add(checks, "beta 在变异边界内", in_bounds,
                 f"bounds={lo_hi}, beta={beta:.6g}")
    else:
        _add(checks, "变异关闭", float(cfg.get("MUTATION_RATE", 0.0)) == 0.0,
             "mutation.enabled=false（祖先株/无抗原漂移档）")

    return {
        "summary": dp.format_summary(info),
        "frames_per_day": fpd,
        "derived": derived,
        "analytic": analytic,
        "checks": checks,
    }


def _reseed_random(mod) -> None:
    """把初始感染改成"随机个体"种子，避免模型默认的工作场所集中播种放大早期增长率。

    只影响 profile_check 的测量进程（导入后的内存对象），不改变模型/档案默认行为。
    """
    torch = mod.torch
    status = mod.status
    status[status == 2] = 0
    idx = torch.randperm(mod.CONFIG["N_AGENTS"], device=mod.device)[: mod.CONFIG["INITIAL_SEEDS"]]
    status[idx] = 2
    mod.virus_beta[idx] = mod.CONFIG["BASE_TRANSMIT"]
    mod.virus_gamma[idx] = mod.CONFIG["BASE_RECOVERY"]
    mod.virus_mu[idx] = mod.CONFIG["BASE_DEATH"]
    mod.virus_sigma[idx] = mod.CONFIG["MUTATION_RATE"]
    mod.virus_gen[idx] = 1
    mod.max_incubation[idx] = mod.CONFIG["INCUBATION_MU"]
    mod.ever_infected[idx] = True
    if mod.CONFIG.get("NATURAL_HISTORY_ENABLED"):
        mod.inf_elapsed[idx] = 0.0
        mod.symp_elapsed[idx] = 0.0
        mod.a_timer[idx] = 0.0
        mod.iso_timer[idx] = -1.0
        mod.isolated[idx] = False
        mod.admission_attempted[idx] = False
        mod.untreated[idx] = False


def run_empirical(mod, frames: int) -> list:
    """跑 update 循环，每 25 帧记录一次累计感染数（种子随机分散）。"""
    update = mod.update
    ever = getattr(mod, "ever_infected", None)
    status = mod.status

    def cumulative() -> int:
        if ever is not None:
            return int(ever.sum().item())
        return int(((status == 2) | (status == 3) | (status == 4) | (status == 5)).sum().item())

    _reseed_random(mod)
    update(0)
    samples = [(0, cumulative())]
    # 每 25 帧采样一次：远小于增长的时间尺度，又足够压低记录/主机同步开销；含 update(0) 共 frames+1 次 update。
    for i in range(1, frames + 1):
        update(i)
        if i % 25 == 0 or i == frames:
            samples.append((i, cumulative()))
    return samples


def fit_growth(samples: list, n_agents: int) -> dict | None:
    """对累计感染数的指数增长段做最小二乘拟合（log 线性），返回斜率/窗口/点数；样本不足返回 None。

    窗口 lo=max(20, 0.2%N)、hi=max(200, 15%N)：下界滤掉播种噪声与个位数波动，上界滤掉饱和段；
    窗口内点 <4 时退回"所有 >= lo 的点"，仍 <3 则 None。斜率单位=1/帧（自然对数）。
    """
    # 下界取 0.2% 人口：再小则整数计数噪声主导 log 斜率；上界 15% 人口：再大就进入饱和段。
    lo = max(20, int(0.002 * n_agents))
    hi = max(200, int(0.15 * n_agents))
    points = [(f, c) for f, c in samples if lo <= c <= hi]
    if len(points) < 4:
        points = [(f, c) for f, c in samples if c >= lo]
    if len(points) < 3:
        return None
    xs = [float(p[0]) for p in points]
    ys = [math.log(float(p[1])) for p in points]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return {"growth_rate_per_frame": r, "points": len(points),
            "window": [int(points[0][0]), int(points[-1][0])],
            "cum_from": points[0][1], "cum_to": points[-1][1]}


def check_empirical(mod, frames: int) -> dict:
    """用经验增长反推 R0：拟合 r -> Euler-Lotka R0(r)=1/Σ(kernel/Σ·e^{-rt})，再与目标 R0 对比。

    只写 report（不新增解析式 checks）；拟合失败或 r<=0 时只记 note 并返回。
    r0_within_30pct = 相对误差 <=0.30：经验路径的容差（早期拟合+结构近似），比解析式 1e-3 宽得多。
    """
    fpd = float(mod.PROFILE_INFO["frames_per_day"])
    derived = mod.PROFILE_INFO["derived"]
    samples = run_empirical(mod, frames)
    fit = fit_growth(samples, int(mod.CONFIG["N_AGENTS"]))
    report: dict = {"frames": frames, "samples": samples[-5:], "fit": fit,
                    "r0_target": derived.get("r0_target"), "r0_estimate": None,
                    "doubling_time_days": None, "generation_time_frames": None}
    if fit is None:
        report["note"] = "样本不足，无法拟合指数增长段（可能未爆发或过快饱和）"
        return report
    r = fit["growth_rate_per_frame"]
    if r <= 0:
        report["note"] = f"拟合增长率为 {r:.3g}，疫情未指数增长"
        return report
    report["doubling_time_days"] = math.log(2.0) / (r * fpd)
    import numpy as np
    kernel_info = _v2_kernel(mod)
    if kernel_info is not None:
        kernel, _fpd, _t_inc = kernel_info
        total = float(kernel.sum())
        tgrid = np.arange(kernel.size, dtype=float)
        report["generation_time_frames"] = float((tgrid * kernel).sum() / total)
        lotka = float(np.sum(kernel / total * np.exp(-r * tgrid)))
        report["lotka_integral_frames"] = lotka
        if lotka > 0:
            report["r0_estimate"] = 1.0 / lotka
    else:
        gamma = float(derived["gamma"])
        t_inc = float(derived["incubation_mu"])
        mult = float(derived["incubating_transmission_mult"])
        t_symp = 1.0 / gamma
        f_pre = mult * t_inc / (mult * t_inc + t_symp) if t_inc > 0 else 0.0
        report["generation_time_frames"] = (
            f_pre * (t_inc / 2.0) + (1.0 - f_pre) * (t_inc + t_symp))
        # Euler-Lotka：1 = q·C·∫ transmission(t)·e^{-rt} dt，其中
        # transmission(t) = mult（0..T_inc，均匀）+ 1（发病期，指数康复）。
        # 对 R0 = q·C·(mult·T_inc + T_symp)，q·C 被消掉：R0 = (mult·T_inc + T_symp) / L(r)。
        lotka = (mult * (1.0 - math.exp(-r * t_inc)) / r
                 + math.exp(-r * t_inc) / (r + gamma))
        report["lotka_integral_frames"] = lotka
        if lotka > 0:
            report["r0_estimate"] = (mult * t_inc + t_symp) / lotka
    if report["r0_target"] is not None and report.get("r0_estimate") is not None:
        rel = abs(report["r0_estimate"] - report["r0_target"]) / report["r0_target"]
        report["r0_relative_error"] = rel
        report["r0_within_30pct"] = rel <= 0.30
    return report


def _print_report(report: dict) -> None:
    """打印人读报告并把 verdict 写回 report；"结论: PASS/WARN" 是输出契约。

    WARN = 解析式有失败项或经验 R0 超 30%；只影响文本呈现，不影响 main 的退出码。
    """
    print(report["summary"])
    print("--- 派生模型参数（每帧单位）---")
    print(f"  beta={report['derived']['beta']:.6g}  "
          f"gamma={report['derived']['gamma']:.6g}  "
          f"mu={report['derived']['mu']:.6g}")
    print(f"  潜伏期均值={report['derived']['incubation_mu']:.0f} 帧  "
          f"sd={report['derived']['incubation_sd']:.0f} 帧  "
          f"min={report['derived']['incubation_min']:.0f} 帧")
    analytic = report["analytic"]
    if analytic["r0_target"] is not None:
        print(f"  R0 目标={analytic['r0_target']:.3f}  "
              f"解析校验={analytic['r0_check']:.6f}  "
              f"平均代际时间≈{analytic['generation_time_frames']:.0f} 帧 "
              f"({analytic['generation_time_frames'] / report['frames_per_day']:.1f} 天)  "
              f"Euler-Lotka 倍增时间≈{analytic['doubling_time_days']:.1f} 天")
    if analytic.get("kernel_effective_days") is not None:
        print(f"  自然史传染力核: 有效传染期积分={analytic['kernel_effective_days']:.2f} 天  "
              f"代际时间={analytic['generation_time_days']:.2f} 天")
    print("--- 解析式检查 ---")
    failed = 0
    for item in report["checks"]:
        status = "PASS" if item["ok"] else "FAIL"
        if not item["ok"]:
            failed += 1
        print(f"  [{status}] {item['name']}: {item['detail']}")
    empirical = report.get("empirical")
    if empirical:
        print("--- 经验检查（小规模仿真）---")
        fit = empirical.get("fit")
        if fit:
            print(f"  拟合窗口: 帧 {fit['window'][0]}..{fit['window'][1]}，"
                  f"累计感染 {fit['cum_from']} -> {fit['cum_to']}（{fit['points']} 点）")
            print(f"  增长率: {fit['growth_rate_per_frame']:.6g} /帧  "
                  f"倍增时间: {empirical['doubling_time_days']:.2f} 天")
            if empirical.get("r0_estimate") is not None:
                print(f"  经验 R0（Euler-Lotka 换算）: {empirical['r0_estimate']:.2f}  "
                      f"（目标 {empirical['r0_target']:.2f}，"
                      f"相对误差 {empirical.get('r0_relative_error', float('nan')):.1%}）")
                if empirical.get("r0_within_30pct") is False:
                    print("  [WARN] 经验 R0 偏离目标 >30%："
                          "可调整 profile 的 model_parameters.r0_calibration_factor")
        else:
            print(f"  {empirical.get('note', '无拟合结果')}")
    failed_empirical = 1 if (empirical and empirical.get("r0_within_30pct") is False) else 0
    verdict = "PASS" if failed == 0 and failed_empirical == 0 else "WARN"
    print(f"结论: {verdict}（解析失败 {failed} 项；经验 R0 "
          f"{'超出容差' if failed_empirical else '未发现偏差'}）")
    report["verdict"] = verdict


def main(argv=None) -> int:
    """CLI 入口，返回退出码：0=报告已生成；1=档案非法（ProfileError）或 --empirical 且 frames<100。

    解析式失败、经验 R0 超容差都只体现在"结论: WARN"里，仍然是 0——这个工具是校准辅助，不是门禁。
    """
    args = build_parser().parse_args(argv)
    profile_name = args.profile or dp.DEFAULT_PROFILE
    try:
        dp.load_profile(profile_name)          # 先做结构与数值校验
    except dp.ProfileError as exc:
        print(f"[profile_check] 档案无效: {exc}")
        return 1
    if args.empirical and args.frames < 100:
        print("[profile_check] --empirical 需要 --frames >= 100")
        return 1
    try:
        mod, _import_log = import_model(profile_name, args.population, args.seed)
    except dp.ProfileError as exc:
        print(f"[profile_check] 应用档案失败: {exc}")
        return 1
    report = check_analytic(mod)
    if args.empirical:
        report["empirical"] = check_empirical(mod, args.frames)
    _print_report(report)
    if args.json:
        payload = {
            "profile": profile_name,
            "population": args.population,
            "seed": args.seed,
            "report": report,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        print(f"JSON 已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
