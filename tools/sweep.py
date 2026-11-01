"""多进程吞吐 / 参数扫描：回答"K 个并行世界里单世界延迟怎么变、总吞吐饱和在哪"。

父进程模式（推荐）：
    python tools/sweep.py --jobs-list 1,2,4,8 --population 10000 --frames 600
    python tools/sweep.py --population 1000000 --frames 300 --jobs-list 1,2,4 --json out.json
子进程模式（`--child`，父进程自动调用，勿手动用）：跑一个世界，用 SWEEP_READY /
SWEEP_RESULT 两行协议标记与父进程通信（父进程靠前者判断"全部 load+warm 完"才
开始 GPU 采样，否则加载段的低 util 会稀释均值）。
输出：控制台表 + `benchmarks/sweep_<profile>_<pop>.json`（逐 job 耗时、吞吐、加速比、
nvidia-smi 的 util/功耗/显存采样）。

口径陷阱：三个数不是一回事——`throughput_frames_per_s` 是稳态估计（Σ 1000/各 job
中位帧耗时），`end_to_end_frames_per_s` 含进程加载与 warm，`per_job_per_frame_ms`
才是单世界延迟。吞吐模式固定 `SIM_RENDER_EVERY=0`（纯计算），与 bench.py /
render_prof.py 的渲染口径不可混报（SCALING.md §6.4–§6.5）。
"""
from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# 子进程：跑一个世界，打印一行 JSON
# ---------------------------------------------------------------------------
def child_main(args: argparse.Namespace) -> int:
    """子进程：加载一个世界 → warm → 打印 SWEEP_READY → 逐帧计时 → 打印结果 JSON。"""
    import golden_check as gc  # noqa: E402  (延迟 import，避免父进程加载 torch)
    import torch  # noqa: E402

    # SIM_RENDER_EVERY=0 是吞吐口径的前提：不更新散点，省掉每帧两次显存→主机传输；
    # 其余 env 与 harness 一致（档案/人口/seed/Agg），保证与单进程口径同源。
    env = {
        "DISEASE_PROFILE": args.profile,
        "DISEASE_POPULATION": str(args.population),
        "DISEASE_SEED": str(args.seed),
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
        "SIM_RENDER_EVERY": str(getattr(args, "render_every", 0)),
    }
    module, load_s, _ = gc.load_module(ENTRY, env)

    # warm 默认 10 帧：让 CUDA kernel 首启与时钟爬坡过去；设 0 也能跑，但首帧会被
    # 算进延迟（首帧含 kernel 首次加载，明显偏慢）。
    warm = max(0, args.warm)
    for frame in range(warm):
        module.update(frame)
    torch.cuda.synchronize()
    # 协议标记 + flush：父进程按行读取，不 flush 就一直等不到 READY。
    print("SWEEP_READY", flush=True)

    # 逐帧计时且每帧 sync：这是"单世界延迟"口径（GPU 真跑完才记时间），
    # 与 ab_iter.py 的帧流吞吐口径不是一回事，报告里不要混用。
    durations: list[float] = []
    t0 = time.perf_counter()
    for frame in range(warm, warm + args.frames):
        ft0 = time.perf_counter()
        module.update(frame)
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - ft0) * 1000.0)
    wall = time.perf_counter() - t0

    # 末帧 counts 只作"各 job 同帧号状态是否同源"的参考；入口没暴露 status 时留 None，别中断。
    try:
        counts = [int(v) for v in torch.bincount(module.status, minlength=8).cpu().tolist()]
    except Exception:
        counts = None

    # 结果行是父进程唯一的数据来源：字段名与标记串改动等于改协议（父进程按 key 取值）。
    result = {
        "seed": args.seed,
        "profile": args.profile,
        "population": args.population,
        "frames": args.frames,
        "load_s": round(load_s, 3),
        "wall_s": round(wall, 3),
        "per_frame_ms_mean": round(statistics.fmean(durations), 3),
        "per_frame_ms_median": round(statistics.median(durations), 3),
        "per_frame_ms_p95": round(
            sorted(durations)[min(len(durations) - 1, int(0.95 * len(durations)))], 3
        ),
        "per_frame_ms_min": round(min(durations), 3),
        "per_frame_ms_max": round(max(durations), 3),
        "counts_end": counts,
        "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2)
        if torch.cuda.is_available()
        else 0.0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    print("SWEEP_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------
# GPU 采样（nvidia-smi，独立线程）
# ---------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """后台线程轮询 nvidia-smi，采 GPU util/功耗/显存；父进程只读 summary()。

    精度限制：默认 0.5 s 一采，且 nvidia-smi 自身启动要几十毫秒；短批次（<2 s）
    只落几个样本，util 只能当"量级"看，别拿它算效率百分比。
    """

    def __init__(self, interval: float) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop_evt = threading.Event()

    @staticmethod
    def _query() -> dict[str, float] | None:
        """问一次 nvidia-smi；None 表示没装/无 GPU/输出格式变了（不致命，跳过该样本）。"""
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            util, power, mem = [float(part.strip()) for part in out.split(",")]
            return {"util_pct": util, "power_w": power, "mem_mib": mem}
        except Exception:
            return None

    def run(self) -> None:  # pragma: no cover - 依赖外部进程
        while not self._stop_evt.is_set():
            sample = self._query()
            if sample:
                self.samples.append(sample)
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        utils = [s["util_pct"] for s in self.samples]
        powers = [s["power_w"] for s in self.samples]
        mems = [s["mem_mib"] for s in self.samples]
        return {
            "util_mean_pct": round(statistics.fmean(utils), 1),
            "util_peak_pct": round(max(utils), 1),
            "power_mean_w": round(statistics.fmean(powers), 1),
            "power_peak_w": round(max(powers), 1),
            "mem_peak_mib": round(max(mems), 1),
            "n_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# 父进程：跑一个 K，再汇总
# ---------------------------------------------------------------------------
def run_one_batch(args: argparse.Namespace, jobs: int, seed_base: int) -> dict:
    """启动 jobs 个独立进程跑同一档案（seed = seed_base+index），汇总成一批结果。

    全部子进程 READY 后才开 GPU 采样、全部退出后停采；返回本批吞吐/延迟/采样摘要。
    """
    procs: list[subprocess.Popen] = []
    lines: dict[int, list[str]] = {}
    ready = threading.Event()
    lock = threading.Lock()

    def reader(index: int, proc: subprocess.Popen) -> None:
        """线程内逐行收一个子进程的输出；全部 job 都 READY 时置位采样开关。"""
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            with lock:
                lines.setdefault(index, []).append(line)
                # 必须等所有 job 都 READY 才开采样：加载/预热段的低 util 会稀释均值。
                if sum(1 for v in lines.values() if any("SWEEP_READY" in x for x in v)) >= jobs:
                    ready.set()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    t_start = time.perf_counter()
    for index in range(jobs):
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--child",
            "--population",
            str(args.population),
            "--frames",
            str(args.frames),
            "--profile",
            args.profile,
            "--seed",
            str(seed_base + index),
            "--warm",
            str(args.warm),
            "--render-every",
            str(args.render_every),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        procs.append(proc)
        threading.Thread(target=reader, args=(index, proc), daemon=True).start()

    ready.wait(timeout=max(60.0, args.load_timeout))
    sampler = GpuSampler(args.interval)
    if not args.no_monitor:
        sampler.start()

    for proc in procs:
        proc.wait()
    wall = time.perf_counter() - t_start
    if not args.no_monitor:
        sampler.stop()
        sampler.join(timeout=2)

    results: list[dict] = []
    errors: list[str] = []
    for index, proc in enumerate(procs):
        payload = None
        for line in lines.get(index, []):
            if line.startswith("SWEEP_RESULT "):
                payload = json.loads(line[len("SWEEP_RESULT ") :])
            elif "Error" in line or "Traceback" in line:
                errors.append(line)
        if payload is None:
            errors.append(f"job {index} (exit {proc.returncode}) 未返回结果")
        else:
            payload["job_index"] = index
            results.append(payload)

    per_frame = [r["per_frame_ms_median"] for r in results]
    # 稳态吞吐 = Σ(1/单世界帧耗时)：K 个世界各自跑完自己那帧的速率估计，不含加载/预热，
    # 所以总是高于 end_to_end（后者含进程启动 + warm，短跑时差距明显）。
    steady_throughput = sum(1000.0 / v for v in per_frame) if per_frame else 0.0
    gpu = sampler.summary() if not args.no_monitor else {}
    return {
        "jobs": jobs,
        "frames_per_job": args.frames,
        "wall_s": round(wall, 3),
        "throughput_frames_per_s": round(steady_throughput, 2),
        "end_to_end_frames_per_s": round(len(results) * args.frames / wall, 2) if wall > 0 else 0.0,
        "per_job_per_frame_ms": [round(v, 2) for v in per_frame],
        "per_job_per_frame_ms_median": round(statistics.median(per_frame), 2) if per_frame else None,
        "jobs_reporting": len(results),
        "gpu": gpu,
        "counts_end": [r.get("counts_end") for r in results],
        "errors": errors,
    }


def parent_main(args: argparse.Namespace) -> int:
    """按 --jobs-list 逐个并发度跑一批，打印表格并落盘 benchmarks/sweep_*.json。"""
    jobs_list = [int(x) for x in str(args.jobs_list).split(",") if x.strip()]
    batches: list[dict] = []
    baseline = None
    for jobs in jobs_list:
        print(f"\n=== jobs={jobs}  population={args.population}  frames={args.frames} ===", flush=True)
        batch = run_one_batch(args, jobs, args.seed)
        if baseline is None:
            # 加速比的基准必须是 jobs=1 那批（--jobs-list 第一项得是 1），否则数字没意义。
            baseline = batch["throughput_frames_per_s"]
        batch["speedup_vs_single"] = round(
            (batch["throughput_frames_per_s"] / baseline) if baseline else 0.0, 3
        )
        batches.append(batch)
        gpu = batch.get("gpu", {})
        print(
            f"  per-job {batch['per_job_per_frame_ms_median']} ms | "
            f"吞吐 {batch['throughput_frames_per_s']:.1f} 帧/s "
            f"({batch['speedup_vs_single']:.2f}x) | util {gpu.get('util_mean_pct', '-')}% "
            f"| 功耗 {gpu.get('power_mean_w', '-')} W | 显存峰值 {gpu.get('mem_peak_mib', '-')} MiB",
            flush=True,
        )
        for err in batch["errors"][:3]:
            print(f"  ! {err}", flush=True)

    # K 建议（SCALING.md §6.4 实测）：10k 用 K≈4–8；1M 用 K≈2–4（每进程约 1 GB，
    # 上限受显存/功耗约束而不是算力）。
    payload = {
        "tool": "tools/sweep.py",
        "profile": args.profile,
        "population": args.population,
        "frames": args.frames,
        "warm": args.warm,
        "seed_base": args.seed,
        "batches": batches,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = (
        pathlib.Path(args.json)
        if args.json
        else ROOT / "benchmarks" / f"sweep_{args.profile}_{args.population}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{out_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多进程吞吐/参数扫描")
    parser.add_argument("--population", type=int, default=10000)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--profile", default="covid19")
    parser.add_argument("--seed", type=int, default=42, help="基准种子，第 i 个 job 用 seed+i")
    parser.add_argument("--warm", type=int, default=10)
    parser.add_argument("--jobs-list", default="1,2,4", help="要测的并发度列表，逗号分隔")
    parser.add_argument("--interval", type=float, default=0.5, help="GPU 采样间隔（秒）")
    parser.add_argument("--render-every", type=int, default=0,
                        help="子进程的 SIM_RENDER_EVERY：0=不更新散点（吞吐模式默认），1=每帧更新")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--load-timeout", type=float, default=600.0)
    parser.add_argument("--json", default=None)
    # --child 是内部协议（父进程自动传），帮助里不展示，避免被当正常用法调用。
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
