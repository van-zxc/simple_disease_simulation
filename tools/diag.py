"""决定性诊断：把每帧耗时拆成 GPU 内核 / CPU 派发 / 主机同步 / 纯 Python 四块，判断瓶颈在哪。

回答"是 CPU 派发受限还是 GPU 算力受限"：逐帧 sync 的 wall ≈ 真实单帧耗时；不逐帧 sync 的
"CPU 入队"时间若 ≈ wall 且 CPU_time/入队≈100%，说明单线程 CPU 派发打满；入队 << wall 而 GPU
内核时间占满 wall，则为 GPU 算力受限。

用法（进程级位置参数，没有 argparse）：python tools/diag.py [profile] [population] [seed]
默认 covid19 / 10000 / 42；需要可用的 CUDA（内核计时靠 torch.profiler）。

口径与已知偏差：
- CUPTI 会放大 GPU 内核时间（内核总和可 > wall），只看相对占比与 kernel 计数，别当精确耗时；
  "非内核（派发/空闲）= wall - gpu_busy"可能为负，正是放大证据。
- 主机同步点靠给 Tensor.item/.cpu/.numpy/.tolist 打桩统计（调用数/耗时），会轻微扰动被测代码。
输出：控制台分项 + Top 表；JSON 写 benchmarks/diag_<profile>_<pop>.json。不改任何仓内代码。
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import _loader  # noqa: E402

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "covid19"
POP = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42

# 采样窗口：wall/enqueue 各 80 帧够稳定中位数；profiler 只 30 帧（CUPTI 开销大，多采无益）；
# cProfile 40 帧；WARM 20 帧用来跳过首次 kernel 加载/懒编译与缓存构建。
WARM = 20
N_WALL = 80
N_ENQ = 80
N_PROF = 30
N_CPROF = 40


def _load():
    """按环境变量加载入口模型（复用 _loader 的 Agg 打桩），返回 (module, 加载秒数)。"""
    entry = ROOT / "simple_disease_simulation.py"
    env = {
        "DISEASE_PROFILE": PROFILE,
        "DISEASE_POPULATION": str(POP),
        "DISEASE_SEED": str(SEED),
        "MPLBACKEND": "Agg",
    }
    module, load_s, _ = _loader.load_module(entry, env)
    return module, load_s


def measure_wall(mod, start_frame, n):
    """逐帧 sync 前后计时，返回每帧毫秒列表——这是"真实单帧耗时"口径。"""
    times = []
    for i in range(n):
        frame = start_frame + i
        torch.cuda.synchronize()  # 先清空残留异步工作，避免把上一帧的尾巴算进本帧
        t0 = time.perf_counter()
        mod.update(frame)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def measure_enqueue(mod, start_frame, n):
    """不逐帧同步：测 CPU 入队速率（若 ≈ wall 则 CPU 受限）。

    返回 (入队总 ms, 含尾部 sync 的总 ms, 进程 CPU 时间 ms)：入队期间 GPU 可并行跑、队列会变深，
    所以入队 << wall 就说明瓶颈不在 CPU；cpu/入队≈100% 则说明单线程 CPU 已打满。
    """
    torch.cuda.synchronize()
    t_cpu0 = time.process_time()
    t0 = time.perf_counter()
    for i in range(n):
        mod.update(start_frame + i)
    enq = (time.perf_counter() - t0) * 1000.0
    cpu = (time.process_time() - t_cpu0) * 1000.0
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) * 1000.0
    return enq, total, cpu


def measure_syncs(mod, start_frame, n):
    """统计每帧的主机同步点：给 Tensor.item/.cpu/.numpy/.tolist 打桩计数并计时，返回 (调用数/帧, 耗时/帧)。

    打桩会包一层 Python 调用，会轻微扰动被测代码；finally 里必须还原，否则本进程后续测量全部失真。
    """
    import numpy as torch_np  # noqa: F401  仅保证导入顺序无关

    counters = Counter()
    spent = Counter()
    originals = {}
    for name in ("item", "cpu", "numpy", "tolist"):
        fn = getattr(torch.Tensor, name)
        originals[name] = fn

        def make(name_, fn_):
            def wrapper(self, *a, **k):
                t0 = time.perf_counter()
                out = fn_(self, *a, **k)
                dt = (time.perf_counter() - t0) * 1000.0
                counters[name_] += 1
                spent[name_] += dt
                return out

            return wrapper

        setattr(torch.Tensor, name, make(name, fn))
    try:
        for i in range(n):
            mod.update(start_frame + i)
        torch.cuda.synchronize()
    finally:
        for name, fn in originals.items():
            setattr(torch.Tensor, name, fn)
    return {k: v / n for k, v in counters.items()}, {k: v / n for k, v in spent.items()}


def measure_profiler(mod, start_frame, n):
    """用 torch.profiler（CPU+CUDA）采 n 帧，汇总 kernel 数/时长分布、CPU aten 自身时间与 Top 表。

    返回 dict；注意 CUPTI 口径的 GPU 时间会被放大（内核总和可 > wall），只用于看相对占比与计数。
    """
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(start_frame, start_frame + n):
            mod.update(i)
        torch.cuda.synchronize()

    kernels = []
    for ev in prof.events():
        for k in getattr(ev, "kernels", ()) or ():
            kernels.append(k)

    durations = [k.duration for k in kernels]
    total_gpu_us = sum(durations)
    cpu_self = Counter()
    aten_cpu_total = 0.0
    for ev in prof.key_averages():
        cpu_self[ev.key] = ev.self_cpu_time_total
        aten_cpu_total += ev.self_cpu_time_total
    gpu_by_op = Counter()
    for ev in prof.key_averages():
        gpu_by_op[ev.key] += ev.self_device_time_total

    buckets = Counter()
    for d in durations:
        if d < 2:
            buckets["<2us"] += 1
        elif d < 5:
            buckets["2-5us"] += 1
        elif d < 10:
            buckets["5-10us"] += 1
        elif d < 20:
            buckets["10-20us"] += 1
        elif d < 50:
            buckets["20-50us"] += 1
        else:
            buckets["≥50us"] += 1

    by_name = Counter()
    for k in kernels:
        by_name[k.name] += 1

    return {
        "frames": n,
        "kernel_count": len(kernels),
        "kernel_count_per_frame": len(kernels) / n,
        "gpu_us_per_frame": total_gpu_us / n,
        "gpu_median_us": statistics.median(durations) if durations else 0.0,
        "gpu_mean_us": (total_gpu_us / len(kernels)) if kernels else 0.0,
        "gpu_max_us": max(durations) if durations else 0.0,
        "buckets": dict(buckets),
        "cpu_aten_ms_per_frame": aten_cpu_total / 1000.0 / n,
        "top_cpu_ops": cpu_self.most_common(12),
        "top_gpu_ops": gpu_by_op.most_common(12),
        "top_kernels_by_count": by_name.most_common(12),
    }


def main():
    """采样并打印诊断，同时把同一份指标写 benchmarks/diag_<profile>_<pop>.json；无返回值。

    顺序：加载→预热 WARM 帧→wall→enqueue→主机同步→profiler→cProfile；每段接着上一段末尾的帧号，
    保证各段覆盖连续且互不重叠的帧区间。
    """
    print(f"加载 {PROFILE} / population={POP} / seed={SEED} ...", flush=True)
    mod, load_s = _load()
    print(f"加载耗时 {load_s:.2f}s，N={mod.ACTIVE_PROFILE_NAME} n={mod.CONFIG.get('N_AGENTS')}", flush=True)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"GPU total {total:.0f} MiB", flush=True)

    # warm
    for i in range(WARM):
        mod.update(i)
    torch.cuda.synchronize()

    f0 = WARM
    wall = measure_wall(mod, f0, N_WALL)
    f1 = f0 + N_WALL
    enq, total_after_sync, cpu_used = measure_enqueue(mod, f1, N_ENQ)
    f2 = f1 + N_ENQ
    sync_calls, sync_spent = measure_syncs(mod, f2, N_WALL)
    f3 = f2 + N_WALL
    prof = measure_profiler(mod, f3, N_PROF)
    f4 = f3 + N_PROF

    # cProfile：纯 Python 时间分布
    pr = cProfile.Profile()
    pr.enable()
    for i in range(N_CPROF):
        mod.update(f4 + i)
    torch.cuda.synchronize()
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    stats.print_stats(14)
    cprof_text = buf.getvalue()
    total_cprof = stats.total_tt
    update_tt = None
    for key, value in stats.stats.items():
        # key=(文件, 行号, 函数名)；value[3] 是该函数自身 tottime（不含被调函数），故只有 update 才算"Python 开销"。
        if key[2] == "update":
            update_tt = value[3]
            break

    wall_med = statistics.median(wall)
    wall_mean = statistics.mean(wall)
    enq_per_frame = enq / N_ENQ
    gpu_busy = prof["gpu_us_per_frame"] / 1000.0
    idle = wall_med - gpu_busy  # 可能为负：CUPTI 放大 GPU 时间，负值本身就是"统计口径被放大"的证据
    sync_total = sum(sync_spent.values())  # 打桩统计的主机同步耗时，不含 CUPTI 放大

    print("\n" + "=" * 78)
    print(f"决定性诊断：{PROFILE} population={POP}")
    print("=" * 78)
    print(f"帧时间（逐帧 sync）      wall mean={wall_mean:.2f} ms  median={wall_med:.2f} ms  min={min(wall):.2f}  max={max(wall):.2f}")
    print(f" CPU 入队（不逐帧 sync）  enqueue={enq_per_frame:.2f} ms/帧  进程 CPU 时间={cpu_used / N_ENQ:.2f} ms/帧  含尾部 sync 总时={total_after_sync:.0f} ms")
    print(f" CPU 占用率               CPU_time/enqueue = {cpu_used / enq * 100:.1f}%（≈100% 说明单线程 CPU 打满）")
    print(f" GPU 内核时间            {gpu_busy:.2f} ms/帧（CUPTI 口径）")
    print(f" 非内核（派发/空闲）      {idle:.2f} ms/帧 = wall - gpu_busy")
    print(f" 内核数                  {prof['kernel_count_per_frame']:.1f} 个/帧，中位 {prof['gpu_median_us']:.2f} us，均值 {prof['gpu_mean_us']:.2f} us，最大 {prof['gpu_max_us']:.0f} us")
    print(f" 内核时长分布            {prof['buckets']}")
    print(f" CPU 侧 aten 自身时间     {prof['cpu_aten_ms_per_frame']:.2f} ms/帧（CUPTI 口径，含探针放大）")
    print(f" 主机同步点              {sync_total:.2f} ms/帧  调用数/帧={ {k: round(v,2) for k,v in sync_calls.items()} }")
    print(f" Python（cProfile）      update 自身 tottime={update_tt and update_tt/ N_CPROF * 1000:.2f} ms/帧，全部函数合计={total_cprof / N_CPROF * 1000:.2f} ms/帧")
    print(f" 折算 GPU 利用率          gpu_busy/wall = {gpu_busy / wall_med * 100:.1f}%（单进程）")
    print("\nTop CPU 侧 aten（self CPU，us/30帧）：")
    for name, us in prof["top_cpu_ops"]:
        print(f"  {us / 1000.0 / N_PROF:8.3f} ms/帧  {name}")
    print("\nTop GPU 内核时间（按 op 归集）：")
    for name, us in prof["top_gpu_ops"]:
        print(f"  {us / 1000.0 / N_PROF:8.3f} ms/帧  {name}")
    print("\nTop 内核（按调用次数）：")
    for name, cnt in prof["top_kernels_by_count"]:
        print(f"  {cnt / N_PROF:8.2f} 次/帧  {name}")
    print("\ncProfile 头部：")
    for line in cprof_text.splitlines()[:22]:
        print("  " + line)

    out = {
        "profile": PROFILE,
        "population": POP,
        "seed": SEED,
        "wall_ms": {"mean": wall_mean, "median": wall_med, "min": min(wall), "max": max(wall)},
        "enqueue_ms_per_frame": enq_per_frame,
        "gpu_ms_per_frame": gpu_busy,
        "non_kernel_ms_per_frame": idle,
        "kernel_count_per_frame": prof["kernel_count_per_frame"],
        "kernel_us_median": prof["gpu_median_us"],
        "kernel_buckets": prof["buckets"],
        "cpu_aten_ms_per_frame": prof["cpu_aten_ms_per_frame"],
        "sync_calls_per_frame": sync_calls,
        "sync_ms_per_frame": sync_total,
        "gpu_util_estimate_pct": gpu_busy / wall_med * 100.0,
        "python_update_tottime_ms_per_frame": (update_tt / N_CPROF * 1000.0) if update_tt else None,
        "top_cpu_ops": [[n, us / 1000.0 / N_PROF] for n, us in prof["top_cpu_ops"]],
        "top_kernels_by_count": [[n, c / N_PROF] for n, c in prof["top_kernels_by_count"]],
    }
    dest = ROOT / "benchmarks" / f"diag_{PROFILE}_{POP}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {dest}")


if __name__ == "__main__":
    main()
