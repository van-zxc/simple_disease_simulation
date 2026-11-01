#!/usr/bin/env python
"""性能基准：同一会话内测 update / draw 的单帧成本，输出可 diff 的 JSON（只作 A/B 对比，不作绝对门禁）。

输入：仓库根 simple_disease_simulation.py（--module 可换；--profile/--population/--seed 走环境变量）。
输出：控制台一行汇总；--json 落盘；--compare 并排比较多份 JSON，并算首份 vs 末份的百分比变化。

口径（与其它工具不同，别混用）：
- update 计时含 torch.cuda.synchronize()（CUDA 下单帧真实耗时），不含绘制。
- draw 是每帧 canvas.draw() 全量重绘；Agg 后端的 canvas.blit() 是空实现，所以这里没有逐 artist blit，
  比真实 GUI 的 FuncAnimation(blit=True) 路径贵得多。
绝对值跨时段/跨机器不可比（热漂移可达 ±30%），只适合同机器同会话 A/B；结论看 per_frame_ms 的相对变化。
JSON 字段：update/draw_blit={n,mean_ms,median_ms,min_ms,max_ms,total_s}；first_update_ms 含懒初始化，
warm_update_ms 是第二次 update(0)；per_frame_ms=update.mean+draw.mean（--no-draw 时只有 update）。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODULE = ROOT / "simple_disease_simulation.py"


def patch_matplotlib() -> None:
    """打桩 matplotlib：必须先于入口模块 import 执行，否则入口在 import 期创建动画/调 show() 会卡死或弹窗。

    只做三件事：切 Agg、把 FuncAnimation 换成只存构造参数的假类、把 plt.show 置空。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt

    class _DummyFuncAnimation:  # pragma: no cover - 仅打桩
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def save(self, *args, **kwargs):
            return None

    mpl_animation.FuncAnimation = _DummyFuncAnimation  # type: ignore[assignment]
    plt.show = lambda *a, **k: None  # type: ignore[assignment]


def load_module(path: Path, env: dict[str, str]):
    """设好环境变量→打桩→用 spec_from_file_location 加载入口，返回 (module, 加载秒数, import 期 stdout)。

    env 必须在 import 前写入：入口按 DISEASE_PROFILE/DISEASE_POPULATION/DISEASE_SEED 决定规模与档案。
    加载期临时关字节码写入，避免仓库里出现 __pycache__；stdout 被吞掉，尾部留给 JSON 诊断。
    """
    for key, value in env.items():
        os.environ[key] = value
    patch_matplotlib()

    spec = importlib.util.spec_from_file_location("sim_bench", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim_bench"] = module

    captured = io.StringIO()
    t0 = time.perf_counter()
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with contextlib.redirect_stdout(captured):
            spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = old
    return module, time.perf_counter() - t0, captured.getvalue()


def snapshot_memory() -> dict:
    """取进程内存快照：CPU 下 cuda_peak_mb/cuda_reserved_mb 记 None；CUDA 下取峰值 allocated/reserved（MB）。"""
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda_peak_mb": None, "cuda_reserved_mb": None}
    return {
        "device": "cuda",
        "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "cuda_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        "cuda_device": torch.cuda.get_device_name(0),
    }


def summarize(samples: list[float]) -> dict:
    """把每帧秒数列表汇总成 {n,mean_ms,median_ms,min_ms,max_ms,total_s}；空列表返回 {}。"""
    if not samples:
        return {}
    return {
        "n": len(samples),
        "mean_ms": round(statistics.fmean(samples) * 1000, 3),
        "median_ms": round(statistics.median(samples) * 1000, 3),
        "min_ms": round(min(samples) * 1000, 3),
        "max_ms": round(max(samples) * 1000, 3),
        "total_s": round(sum(samples), 3),
    }


def bench(module, frames: int, measure_draw: bool, label: str) -> dict:
    """跑计时循环，返回一次基准的完整 JSON 结构。

    frames 是总 update 次数：先两次 update(0)（首帧含懒初始化、热帧代表稳定态），主循环从 1 跑到
    frames-1，与分析工具的帧序一致。update 每帧后 synchronize（CUDA）；measure_draw 时再全量
    canvas.draw()（Agg 下 blit 为空实现）。返回值字段见模块 docstring。
    """
    import torch

    canvas = module.fig.canvas
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 两次 update(0)：第一次含懒初始化/缓存构建，第二次代表稳定态；差值说明首次更新开销。
    t0 = time.perf_counter()
    artists = module.update(0)
    if cuda:
        torch.cuda.synchronize()  # 与主循环同口径：sync 后计时才包含 GPU 工作
    first_update = time.perf_counter() - t0
    t0 = time.perf_counter()
    artists = module.update(0)
    if cuda:
        torch.cuda.synchronize()
    warm_update = time.perf_counter() - t0

    update_samples: list[float] = []
    draw_samples: list[float] = []
    wall0 = time.perf_counter()  # wall 只覆盖主循环，不含加载/首帧/热帧
    for frame in range(1, frames):
        t0 = time.perf_counter()
        module.update(frame)
        if cuda:
            torch.cuda.synchronize()  # 内核异步执行，sync 后才算本帧真实耗时
        update_samples.append(time.perf_counter() - t0)
        if measure_draw:
            t0 = time.perf_counter()
            canvas.draw()  # 全量重绘（口径见模块 docstring，比 GUI blit 路径贵）
            if artists:
                canvas.blit(artists)  # Agg 下是空实现；保留调用以贴合 GUI 的调用结构
            draw_samples.append(time.perf_counter() - t0)
    wall = time.perf_counter() - wall0

    update_s = summarize(update_samples)
    draw_s = summarize(draw_samples)
    per_frame = update_s["mean_ms"] if update_s else 0.0
    if measure_draw and draw_s:
        per_frame += draw_s["mean_ms"]
    result = {
        "label": label,
        "profile": getattr(module, "ACTIVE_PROFILE_NAME", None),
        "population": int(module.CONFIG.get("N_AGENTS", -1)),
        "frames": frames,
        "first_update_ms": round(first_update * 1000, 3),
        "warm_update_ms": round(warm_update * 1000, 3),
        "update": update_s,
        "draw_blit": draw_s,
        "per_frame_ms": round(per_frame, 3),
        "wall_s": round(wall, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result.update(snapshot_memory())
    return result


def compare_rows(paths: list[str]) -> None:
    """并排打印多份基准 JSON，并在 >=2 份时算"首份 -> 末份"的变化率（正数=变慢，基准是首份）。"""
    runs = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            print(f"文件不存在：{path}", file=sys.stderr)
            return
        runs.append(json.loads(path.read_text(encoding="utf-8")))

    keys = ["update", "draw_blit"]
    header = f"{'run':<12}{'profile':<12}{'N':>8}{'frames':>7}"
    for key in keys:
        header += f"{key + '.mean_ms':>18}"
    header += f"{'per_frame_ms':>15}{'cuda_peak_mb':>14}{'wall_s':>9}"
    print(header)
    for run in runs:
        row = f"{str(run.get('label'))[:11]:<12}{str(run.get('profile'))[:11]:<12}{run.get('population', 0):>8}{run.get('frames', 0):>7}"
        for key in keys:
            row += f"{run.get(key, {}).get('mean_ms', 0.0):>18.3f}"
        row += f"{run.get('per_frame_ms', 0.0):>15.3f}"
        row += f"{str(run.get('cuda_peak_mb')):>14}{run.get('wall_s', 0.0):>9.2f}"
        print(row)

    if len(runs) >= 2:
        base, new = runs[0], runs[-1]
        for key in keys + ["per_frame_ms"]:
            b = base.get(key, {}).get("mean_ms") if isinstance(base.get(key), dict) else base.get(key)
            n = new.get(key, {}).get("mean_ms") if isinstance(new.get(key), dict) else new.get(key)
            if b and n:
                print(f"  {key}: {b:.3f} -> {n:.3f} ms ({(n / b - 1) * 100:+.1f}%)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """解析 CLI；默认值（--frames 400、draw 默认开启）是基准口径的一部分，改动会让历史 JSON 不可比。"""
    parser = argparse.ArgumentParser(description="性能基准")
    parser.add_argument("--module", default=str(DEFAULT_MODULE))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--label", default=None)
    parser.add_argument("--no-draw", action="store_true", help="只测 update，不重绘 canvas")
    parser.add_argument("--json", default=None, help="结果写入 JSON")
    parser.add_argument("--compare", nargs="+", default=None, help="对比多个 JSON 结果")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：返回 0=正常（含 --compare 走完）、2=入口文件不存在；运行期异常按未捕获异常抛出。"""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.compare:
        compare_rows(args.compare)
        return 0

    module_path = Path(args.module).expanduser().resolve()
    if not module_path.is_file():
        print(f"入口文件不存在：{module_path}", file=sys.stderr)
        return 2

    env: dict[str, str] = {"MPLBACKEND": "Agg"}
    if args.profile:
        env["DISEASE_PROFILE"] = args.profile
    if args.population:
        env["DISEASE_POPULATION"] = str(args.population)
    if args.seed is not None:
        env["DISEASE_SEED"] = str(args.seed)

    module, load_s, import_stdout = load_module(module_path, env)
    print(
        f"已加载：profile={getattr(module, 'ACTIVE_PROFILE_NAME', '?')} "
        f"N={module.CONFIG.get('N_AGENTS')} device={getattr(module, 'device', '?')} "
        f"加载耗时={load_s:.2f}s",
        flush=True,
    )
    label = args.label or f"{(args.profile or 'default')}_{module.CONFIG.get('N_AGENTS')}_{args.frames}"
    result = bench(module, args.frames, not args.no_draw, label)
    result["load_s"] = round(load_s, 3)
    result["import_stdout_tail"] = import_stdout[-1000:]
    result["python"] = sys.version.split()[0]
    try:
        result["torch"] = __import__("torch").__version__
    except Exception:  # pragma: no cover
        pass

    print(
        f"update mean={result['update']['mean_ms']}ms median={result['update']['median_ms']}ms "
        f"| draw+blit mean={result['draw_blit'].get('mean_ms')}ms | per_frame={result['per_frame_ms']}ms "
        f"| CUDA peak={result.get('cuda_peak_mb')}MB | wall={result['wall_s']}s",
        flush=True,
    )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
