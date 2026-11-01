"""多 seed 集成（ensemble）运行器：把单条轨迹变成"分布 + 分位带"。

用法（父进程；`--child` 由父进程自动拉起，勿手动用）：
    python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-12 --jobs 4
    python tools/ensemble.py --seeds 1-8 --json out.json --csv out.csv --png out.png --track-secondary
输出：控制台 p10/p50/p90 汇总表；`--json` 完整结果、`--csv` 每 seed 一行、`--png`
分位带图（Agg）；`--png-from-json` 只从已有 JSON 重画图、不重跑模型。

口径陷阱：
- 每 seed 独立子进程（GPU 显存/上下文隔离），并强制 SIM_RENDER_EVERY=0（纯统计）。
- attack_rate = 1 - S/N：toy 档有 1% 初始免疫者（status=1），会被算成"已感染"。
- 二代统计两套口径：completed（只统计传染期已结束者，估过散布/k 用它）与 all
  （含仍在传染期者，右删失会压平方差，只看量级）。
- p10/p50/p90 是 seed 间的分位，不是单条轨迹的时间分位。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY = ROOT / "simple_disease_simulation.py"
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

MARKER = "@@ENSEMBLE@@"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def parse_seeds(spec: str) -> list[int]:
    """''1-12'' / ''1,2,5'' / ''1-3,7'' -> [int, ...]（去重、保序）。"""
    seeds: list[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                lo_i, hi_i = hi_i, lo_i
            seeds.extend(range(lo_i, hi_i + 1))
        else:
            seeds.append(int(chunk))
    out: list[int] = []
    for s in seeds:
        if s not in out:
            out.append(s)
    return out


def pct(values: list[float], q: float) -> float:
    """线性插值分位数（与 numpy.percentile 的默认线性法一致）。"""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def describe(values: list[float]) -> dict:
    """seed 间分布：n/mean/median/p10/p50/p90/min/max；没有有效值时只回 {"n": 0}。"""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p10": round(pct(vals, 10), 4),
        "p50": round(pct(vals, 50), 4),
        "p90": round(pct(vals, 90), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def default_jobs(population: int) -> int:
    """自动并发数：约每 4 核 1 进程（上限 4）；人口 ≥500k 时压到 ≤2（显存/内存）。"""
    cpu = os.cpu_count() or 4
    jobs = max(1, min(4, cpu // 4))
    if population >= 500_000:  # 1M 每进程约 1 GB，保守一点
        jobs = min(jobs, 2)
    return jobs


# ---------------------------------------------------------------------------
# 子进程：跑一个 seed
# ---------------------------------------------------------------------------
def child_main(args: argparse.Namespace) -> int:
    """子进程跑一个 seed：加载 → warm → 逐帧采样 → 可选二代统计 → 一行 MARKER JSON。

    采样记 S/E/A/I/H/D/R 七条曲线与 4 个计数器；帧号从 warm 开始（与父进程一致）。
    """
    import golden_check as gc  # 延迟 import：父进程不必加载 torch
    import torch

    env = {
        "DISEASE_PROFILE": args.profile,
        "DISEASE_POPULATION": str(args.population),
        "DISEASE_SEED": str(args.seed),
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
        # 不渲染：省掉每帧两次显存→主机传输，只留统计所需的 counts。
        "SIM_RENDER_EVERY": "0",
    }
    if args.track_secondary:
        # 只有开这个，disease_sim 才分配 _secondary 张量；父进程据此决定算不算过散布。
        env["SIM_TRACK_SECONDARY"] = "1"
    module, load_s, _ = gc.load_module(ENTRY, env)

    # warm 帧只用来推进 RNG/时钟，不进序列（默认 5 帧，足以越过 kernel 首启）。
    warm = max(0, args.warm)
    for frame in range(warm):
        module.update(frame)

    fpd = int(module.CONFIG.get("DAY_FRAMES", 240))
    n_agents = int(module.status.numel())
    # stride=50 默认每 50 帧记一次（12.5 天里约 60 个采样点）；传 0 会退化成 1。
    stride = max(1, args.stride)

    durations: list[float] = []
    series: dict[str, list[float]] = {"S": [], "E": [], "A": [], "I": [], "H": [], "D": [], "R": []}
    frames_sampled: list[int] = []
    d_prev = int((module.status == 4).sum().item())
    new_deaths_daily: dict[int, int] = {}

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    for offset in range(args.frames):
        frame = warm + offset
        ft0 = time.perf_counter()
        module.update(frame)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        durations.append((time.perf_counter() - ft0) * 1000.0)
        if offset % stride == 0 or offset == args.frames - 1:
            counts = torch.bincount(module.status, minlength=8).cpu().tolist()
            # status 8 槽：0=S 1=初始免疫 2=E 3=I 4=D 5=R 6=A 7=H；槽 1 不进任何曲线
            # （初始免疫者不是 susceptible，画进 S 会与"可感人群"语义不符）。
            for key, idx in (("S", 0), ("E", 2), ("A", 6), ("I", 3), ("H", 7), ("D", 4), ("R", 5)):
                series[key].append(int(counts[idx]))
            frames_sampled.append(frame)
            d_now = int(counts[4])
            day = frame // fpd
            new_deaths_daily[day] = new_deaths_daily.get(day, 0) + max(0, d_now - d_prev)
            d_prev = d_now

    counts_end = torch.bincount(module.status, minlength=8).cpu().tolist()
    s_end, e_end, _, i_end, d_end, r_end, a_end, h_end = (int(v) for v in counts_end)

    # "当前感染" = E+A+I（含无症状 A）；峰值按采样点取，stride 大会略偏低。
    infected_series = [s + a + i for s, a, i in zip(series["E"], series["A"], series["I"])]
    hosp_series = list(series["H"])
    peak_infected = max(infected_series) if infected_series else 0
    peak_hosp = max(hosp_series) if hosp_series else 0
    peak_idx = infected_series.index(peak_infected) if infected_series else 0
    time_to_peak_frames = frames_sampled[peak_idx] if frames_sampled else 0
    peak_new_deaths = max(new_deaths_daily.values()) if new_deaths_daily else 0

    strain_new = 0
    strain_reinf = 0
    try:
        strain_new = int(sum(module.strain_stats.get("new_infections", [])))
        strain_reinf = int(sum(module.strain_stats.get("reinfections", [])))
    except Exception:
        pass

    secondary_stats = None
    if args.track_secondary:
        sec = getattr(module, "_secondary", None)
        if sec is not None:
            sec_cpu = sec.detach().to("cpu").to(torch.float64)
            ever = module.ever_infected.detach().to("cpu")
            status_cpu = module.status.detach().to("cpu")
            # 关键口径：完成统计只取"传染期已结束"的感染者（E/I/A 之外）。仍在传染期的
            # 人二代计数还没记全，右删失会把方差压平（低估 var/mean、高估 k）。
            infectious = (status_cpu == 2) | (status_cpu == 3) | (status_cpu == 6)

            def _sec_block(sel):
                """按布尔掩码算二代分布统计量；样本 <2 返回 None。

                负二项近似 k = mean²/(Var−mean)，仅当 Var>mean 时才有定义（否则 None）。
                """
                vals = sec_cpu[sel]
                if int(vals.numel()) < 2:
                    return None
                mean = float(vals.mean().item())
                var = float(vals.var(unbiased=True).item())
                ratio = var / mean if mean > 0 else float("nan")
                # 负二项近似：Var = mean + mean^2/k  =>  k = mean^2 / (Var - mean)
                k_est = (mean * mean / (var - mean)) if var > mean else None
                vlist = vals.tolist()
                return {
                    "infectors": int(len(vlist)),
                    "mean": round(mean, 4),
                    "var": round(var, 4),
                    "var_over_mean": round(ratio, 4) if ratio == ratio else None,
                    "k_estimate": round(k_est, 4) if k_est is not None else None,
                    "frac_zero": round(sum(1 for v in vlist if v < 0.5) / len(vlist), 4),
                    "frac_ge5": round(sum(1 for v in vlist if v >= 5) / len(vlist), 4),
                }

            completed = _sec_block(ever & (~infectious))
            overall = _sec_block(ever)
            if completed or overall:
                secondary_stats = {"completed": completed, "all": overall}

    result = {
        "seed": args.seed,
        "profile": args.profile,
        "population": args.population,
        "frames": args.frames,
        "warm": warm,
        "stride": stride,
        "load_s": round(load_s, 3),
        "n_agents": n_agents,
        "frames_per_day": fpd,
        "counts_end": [int(v) for v in counts_end],
        # 口径陷阱：1 - S/N 把初始免疫者（槽 1）也算成"感染过"，toy 档会高估约 1 个百分点。
        "attack_rate": round(1.0 - s_end / n_agents, 5),
        "infected_end": e_end + a_end + i_end,
        "peak_infected": peak_infected,
        "peak_hospitalized": peak_hosp,
        "peak_new_deaths_per_day": peak_new_deaths,
        "time_to_peak_days": round(time_to_peak_frames / fpd, 3),
        "total_infections": strain_new,
        "reinfections": strain_reinf,
        "deaths": d_end,
        "recovered": r_end,
        "hospitalized_end": h_end,
        "tests_total": int(getattr(module, "tests_total", 0) or 0),
        "quarantine_total": int(getattr(module, "quarantine_total", 0) or 0),
        "reported_total": int(getattr(module, "reported_total", 0) or 0),
        "onset_total": int(getattr(module, "onset_total", 0) or 0),
        "beds_total": int(getattr(module, "beds_total", 0) or 0),
        "per_frame_ms_mean": round(statistics.fmean(durations), 3) if durations else None,
        "per_frame_ms_median": round(statistics.median(durations), 3) if durations else None,
        "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2)
        if torch.cuda.is_available()
        else 0.0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "frames_sampled": frames_sampled,
        "series": series,
        "secondary": secondary_stats,
    }
    print(MARKER + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------
# 父进程：批量调度 + 汇总
# ---------------------------------------------------------------------------
def run_batch(args: argparse.Namespace, seeds: list[int]) -> tuple[list[dict], list[str]]:
    """分批并发跑 seeds（每批 ≤jobs 个独立子进程），按 seed 排序返回 (结果, 错误)。"""
    results: list[dict] = []
    errors: list[str] = []
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    def run_seed(seed: int) -> tuple[dict | None, list[str]]:
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            "--seed",
            str(seed),
            "--profile",
            args.profile,
            "--population",
            str(args.population),
            "--frames",
            str(args.frames),
            "--warm",
            str(args.warm),
            "--stride",
            str(args.stride),
        ]
        if args.track_secondary:
            cmd.append("--track-secondary")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload = None
        errs: list[str] = []
        for line in (proc.stdout or "").splitlines():
            if line.startswith(MARKER):
                payload = json.loads(line[len(MARKER):])
            elif "Traceback" in line or "Error" in line:
                errs.append(line)
        if payload is None:
            errs.append(f"seed {seed} 未返回结果（exit {proc.returncode}）")
        return payload, errs

    # 简单的分批并发：每批 <= jobs 个 seed
    for start in range(0, len(seeds), args.jobs):
        chunk = seeds[start:start + args.jobs]
        threads: list[threading.Thread] = []
        slots: dict[int, tuple[dict | None, list[str]]] = {}

        def worker(seed: int) -> None:
            slots[seed] = run_seed(seed)

        for seed in chunk:
            t = threading.Thread(target=worker, args=(seed,), daemon=False)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        for seed in chunk:
            payload, errs = slots.get(seed, (None, [f"seed {seed} 无结果"]))
            if payload is not None:
                results.append(payload)
                print(
                    f"  seed {seed:>3}: attack={payload['attack_rate']:.3f} "
                    f"peak_I={payload['peak_infected']:>5} peak_H={payload['peak_hospitalized']:>4} "
                    f"D={payload['deaths']:>4} | {payload['per_frame_ms_median']:.2f} ms/帧",
                    flush=True,
                )
            errors.extend(errs)
    results.sort(key=lambda r: r["seed"])
    return results, errors


def summarize(results: list[dict]) -> dict:
    """把各 seed 结果聚成 seed 间分布；开 --track-secondary 时附两套二代口径汇总。"""
    keys = [
        ("attack_rate", "攻击率"),
        ("peak_infected", "峰值感染"),
        ("peak_hospitalized", "峰值住院"),
        ("peak_new_deaths_per_day", "峰值日死亡"),
        ("time_to_peak_days", "达峰天数"),
        ("total_infections", "累计感染"),
        ("reinfections", "再感染"),
        ("deaths", "死亡"),
        ("tests_total", "检测"),
        ("quarantine_total", "隔离"),
        ("reported_total", "报告"),
        ("per_frame_ms_median", "单帧 ms"),
    ]
    summary: dict[str, dict] = {}
    for key, _label in keys:
        summary[key] = describe([r.get(key) for r in results])

    sec = [r["secondary"] for r in results if r.get("secondary")]
    if sec:
        def _agg(rows):
            """把一组（每 seed 一份的）二代统计聚成 seed 间分布；空行过滤后返回 None。"""
            rows = [row for row in rows if row]
            if not rows:
                return None
            ks = [row["k_estimate"] for row in rows if row.get("k_estimate") is not None]
            return {
                "ratio": describe([row["var_over_mean"] for row in rows]),
                "frac_zero": describe([row["frac_zero"] for row in rows]),
                "frac_ge5": describe([row["frac_ge5"] for row in rows]),
                "k_median": round(statistics.median(ks), 4) if ks else None,
                "infectors_median": round(statistics.median([row["infectors"] for row in rows]), 1),
            }

        # completed 是默认可信口径（右删失已排除）；all 只作对照：方差被压平、k 偏高。
        summary["secondary_completed"] = _agg([s.get("completed") for s in sec])
        summary["secondary_all"] = _agg([s.get("all") for s in sec])
    return summary


def write_csv(path: pathlib.Path, results: list[dict]) -> None:
    """每 seed 一行的标量表（只含标量列；时间序列与二代明细留在 JSON 里）。"""
    cols = [
        "seed", "attack_rate", "peak_infected", "peak_hospitalized", "peak_new_deaths_per_day",
        "time_to_peak_days", "total_infections", "reinfections", "deaths", "recovered",
        "tests_total", "quarantine_total", "reported_total", "onset_total",
        "per_frame_ms_median", "cuda_peak_mb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in results:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def write_png(path: pathlib.Path, results: list[dict], meta: dict) -> None:
    """画感染（E+A+I）与住院人数的 seed 间分位带（p10/p50/p90），Agg 后端出 PNG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fpd = int(meta.get("frames_per_day") or 240)
    length = min(len(r["frames_sampled"]) for r in results)
    frames = np.array(results[0]["frames_sampled"][:length], dtype=float) / fpd

    def bands(key_fn):
        mat = np.array([[key_fn(r)[i] for i in range(length)] for r in results], dtype=float)
        return (
            np.percentile(mat, 10, axis=0),
            np.percentile(mat, 50, axis=0),
            np.percentile(mat, 90, axis=0),
        )

    inf = bands(lambda r: [e + a + i for e, a, i in zip(r["series"]["E"], r["series"]["A"], r["series"]["I"])])
    hosp = bands(lambda r: r["series"]["H"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (lo, mid, hi), title in (
        (axes[0], inf, "Infected (E+A+I)"),
        (axes[1], hosp, "Hospitalized"),
    ):
        ax.fill_between(frames, lo, hi, alpha=0.25, label="p10-p90")
        ax.plot(frames, mid, lw=1.8, label="p50")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[1].set_xlabel("Day")
    title = f"{meta.get('profile')} - N={meta.get('population')} - {len(results)} seeds - ensemble"
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def png_from_json(json_path: pathlib.Path, png_path: pathlib.Path) -> int:
    """只重画分位带图（复用已跑完的 JSON，不重跑模型）。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("seeds") or []
    meta = data.get("meta") or {}
    if not results:
        print(f"[error] {json_path} 里没有 seeds 数据", file=sys.stderr)
        return 1
    write_png(png_path, results, meta)
    print(f"已重画分位带图：{png_path}（来自 {json_path}，{len(results)} seeds）", flush=True)
    return 0


def print_summary(summary: dict, results: list[dict]) -> None:
    """打印 p10/p50/p90 汇总表；开 --track-secondary 时附 completed/all 两套二代口径。"""
    rows = [
        ("攻击率", "attack_rate", 3),
        ("峰值感染", "peak_infected", 0),
        ("峰值住院", "peak_hospitalized", 0),
        ("峰值日死亡", "peak_new_deaths_per_day", 1),
        ("达峰天数", "time_to_peak_days", 1),
        ("累计感染", "total_infections", 0),
        ("再感染", "reinfections", 0),
        ("死亡", "deaths", 0),
        ("检测总量", "tests_total", 0),
        ("隔离总量", "quarantine_total", 0),
        ("报告总量", "reported_total", 0),
        ("单帧 ms", "per_frame_ms_median", 2),
    ]
    print("\n=== 多 seed 汇总（p10 / p50 / p90）===", flush=True)
    print(f"{'指标':<10} {'p10':>10} {'p50':>10} {'p90':>10} {'min':>10} {'max':>10}", flush=True)
    for label, key, nd in rows:
        d = summary.get(key, {})
        if not d or "p10" not in d:
            continue
        fmt = f"{{:>{10}.{nd}f}}"
        print(
            f"{label:<10} " + " ".join(fmt.format(d[k]) for k in ("p10", "p50", "p90", "min", "max")),
            flush=True,
        )
    for tag, title in (
        ("completed", "二代分布过散布（传染期已结束的感染者，seed 内 var/mean）"),
        ("all", "二代分布过散布（全部感染者，含尾部截断，仅参考）"),
    ):
        sec = summary.get(f"secondary_{tag}")
        if not sec:
            continue
        d = sec["ratio"]
        print(
            f"\n{title}：p10={d['p10']:.2f} p50={d['p50']:.2f} p90={d['p90']:.2f}"
            f" | k 估计中位={sec.get('k_median')} | 分母中位={sec.get('infectors_median')}",
            flush=True,
        )
        print(
            f"  零二代比例 p50={sec['frac_zero']['p50']:.3f}，"
            f"≥5 二代比例 p50={sec['frac_ge5']['p50']:.3f}",
            flush=True,
        )


def parent_main(args: argparse.Namespace) -> int:
    """解析 seeds → 分批跑 → 汇总 → 写 json/csv/png 并打印 p10/p50/p90 表。

    退出码：0 = 全部 seed 成功；1 = 有 seed 失败；2 = seeds 解析为空。
    """
    seeds = parse_seeds(args.seeds)
    if not seeds:
        print("没有 seed 可跑（--seeds 解析为空）", file=sys.stderr)
        return 2
    if args.jobs is None:
        args.jobs = default_jobs(args.population)
    args.jobs = max(1, min(args.jobs, len(seeds)))

    print(
        f"ensemble: profile={args.profile} population={args.population} "
        f"frames={args.frames} seeds={len(seeds)} jobs={args.jobs} stride={args.stride}",
        flush=True,
    )
    t0 = time.perf_counter()
    results, errors = run_batch(args, seeds)
    wall = time.perf_counter() - t0

    if not results:
        print(f"\n没有成功结果；错误：\n  " + "\n  ".join(errors[:10]), file=sys.stderr)
        return 1

    meta = {
        "tool": "tools/ensemble.py",
        "profile": args.profile,
        "population": args.population,
        "frames": args.frames,
        "warm": args.warm,
        "stride": args.stride,
        "seeds": seeds,
        "jobs": args.jobs,
        "label": args.label,
        "wall_s": round(wall, 3),
        "n_ok": len(results),
        "n_failed": len(seeds) - len(results),
        "frames_per_day": results[0].get("frames_per_day"),
        "gpu_name": results[0].get("gpu_name"),
        "track_secondary": bool(args.track_secondary),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary = summarize(results)
    payload = {"meta": meta, "summary": summary, "seeds": results}

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv:
        write_csv(pathlib.Path(args.csv), results)
    if args.png:
        try:
            write_png(pathlib.Path(args.png), results, meta)
        except Exception as exc:  # 画图失败不影响统计结果
            print(f"[warn] PNG 生成失败：{exc}", file=sys.stderr)

    print_summary(summary, results)
    print(
        f"\n完成：{len(results)}/{len(seeds)} seeds，wall {wall:.1f}s"
        + (f"（{len(errors)} 条错误）" if errors else ""),
        flush=True,
    )
    if errors:
        print("  错误示例：" + " | ".join(errors[:3]), flush=True)
    return 0 if len(results) == len(seeds) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多 seed ensemble 运行器（分布 + 分位带）")
    p.add_argument("--child", action="store_true", help="子进程模式（父进程自动调用）")
    p.add_argument("--seed", type=int, default=42, help="子进程使用的 seed")
    p.add_argument("--profile", default="covid19", help="疾病档案（默认 covid19）")
    p.add_argument("--population", type=int, default=10000, help="人口（默认 10000）")
    p.add_argument("--frames", type=int, default=3000, help="每个 seed 的帧数（默认 3000 = 12.5 天）")
    p.add_argument("--seeds", default="1-12", help="seed 列表：1-12 / 1,2,5 / 1-3,7")
    p.add_argument("--jobs", type=int, default=None, help="并发进程数（默认按 CPU/人口自动）")
    p.add_argument("--warm", type=int, default=5, help="热身帧数（默认 5）")
    p.add_argument("--stride", type=int, default=50, help="时间序列采样间隔帧（默认 50）")
    p.add_argument("--label", default="", help="给这次运行起个名字，写进 JSON meta")
    p.add_argument("--json", default=None, help="完整结果 JSON 输出路径")
    p.add_argument("--csv", default=None, help="每 seed 一行的 CSV 输出路径")
    p.add_argument("--png", default=None, help="分位带图输出路径（PNG）")
    p.add_argument("--png-from-json", default=None, metavar="JSON",
                   help="只从已有结果 JSON 重画分位带图（不重跑模型，需同时给 --png）")
    p.add_argument("--track-secondary", action="store_true",
                   help="统计二代感染分布（过散布），需要模型侧 SIM_TRACK_SECONDARY 支持")
    return p


def main(argv: list[str] | None = None) -> int:
    """三种入口：--child 子进程 / --png-from-json 只重画图 / 其余走父进程全流程。"""
    args = build_parser().parse_args(argv)
    if args.child:
        return child_main(args)
    if args.png_from_json:
        if not args.png:
            print("[error] --png-from-json 需要同时给 --png 输出路径", file=sys.stderr)
            return 2
        return png_from_json(pathlib.Path(args.png_from_json), pathlib.Path(args.png))
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
