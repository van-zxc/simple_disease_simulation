"""逐段归因探针：把 update() 内部拆成 mobility / contacts / rest，看时间花在哪。

用法（位置参数，均可省）：python tools/phase_probe.py [entry] [profile] [population] [frames]
    entry 默认 simple_disease_simulation.py，档案 covid19，人口 10000，frames 60。
输出：控制台每帧 ms 表（total / mobility / contacts / rest(inline)）与每帧调用次数；
`--json` 落盘。rest(inline) = total - mobility - contacts 是 update 内联段（含派发开销）。

已知偏差：
- 每段前后 synchronize()，会把上一次入队、尚未跑完的 GPU 工作算进当前段，并破坏
  pipelining——只能做同一版本内的相对归因，不能当验收数字（验收用 tools/ab_iter.py）。
- CUPTI/nsys 观测同样会放大绝对耗时（SCALING.md §6.7）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def main(argv: list[str] | None = None) -> int:
    """跑 warm 帧后逐帧计时，打印 total/mobility/contacts/rest 四段归因。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("entry", nargs="?", default=str(ROOT / "simple_disease_simulation.py"))
    p.add_argument("profile", nargs="?", default="covid19")
    p.add_argument("population", nargs="?", default="10000")
    p.add_argument("frames", nargs="?", type=int, default=60)
    p.add_argument("--warm", type=int, default=40)
    p.add_argument("--json", dest="json_path", default="")
    args = p.parse_args(argv)

    import golden_check as gc

    entry = pathlib.Path(args.entry).resolve()
    env = {
        "DISEASE_PROFILE": args.profile,
        "DISEASE_POPULATION": str(args.population),
        "MPLBACKEND": "Agg",
    }
    mod, load_s, _ = gc.load_module(entry, env)
    import torch

    # The entry point put its own directory at sys.path[0], so this resolves to
    # the package that belongs to `entry` (snapshot or live tree).
    import disease_sim.simulation as simmod

    acc: dict[str, float] = {"mobility": 0.0, "contacts": 0.0}
    calls: dict[str, int] = {"mobility": 0, "contacts": 0}

    def _wrap(fn, key):
        """包一层计时：前后各 sync 一次，把该段真实 GPU 耗时累加到 acc[key]。"""
        def inner(*a, **k):
            # 已知限制：CPU-only 环境会在这里抛异常（本探针只面向 GPU 归因，不做兜底）。
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            acc[key] += (time.perf_counter() - t0) * 1000.0
            calls[key] += 1
            return out

        return inner

    # 必须改包内模块的属性：update() 是按模块全局名查找这些函数的，改入口里的同名
    # 引用不会生效；contacts 的两条路径（dense/grid）都计入同一项。
    simmod.advance_mobility = _wrap(simmod.advance_mobility, "mobility")
    simmod._contact_pairs_dense = _wrap(simmod._contact_pairs_dense, "contacts")
    simmod._contact_pairs_grid = _wrap(simmod._contact_pairs_grid, "contacts")

    for f in range(args.warm):
        mod.update(f)
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # 热身段的调用必须清零，否则会摊进 per-frame 分母：warm=40/frames=60 时虚高约 33%。
    for key in acc:
        acc[key] = 0.0
        calls[key] = 0

    total = 0.0
    t_all = time.perf_counter()
    for f in range(args.warm, args.warm + args.frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mod.update(f)
        torch.cuda.synchronize()
        total += (time.perf_counter() - t0) * 1000.0

    n = args.frames
    # rest(inline) = update 内部除 mobility/contacts 外的段（含 torch 派发开销），是残差
    # 而非独立测量，会吸收前两段的同步误差（见模块 docstring 的偏差说明）。
    rows = {
        "total": total / n,
        "mobility": acc["mobility"] / n,
        "contacts": acc["contacts"] / n,
        "rest(inline)": (total - acc["mobility"] - acc["contacts"]) / n,
    }
    # 已知限制：只支持仓库内入口——外部路径 relative_to(ROOT) 会抛 ValueError。
    print(f"entry={entry.relative_to(ROOT)} profile={args.profile} "
          f"N={args.population} frames={n}")
    for k, v in rows.items():
        print(f"  {k:<14}: {v:8.3f} ms/frame")
    print(f"  calls/frame    : mobility {calls['mobility'] / n:.2f}, "
          f"contacts {calls['contacts'] / n:.2f}")
    print(f"  wall for loop  : {(time.perf_counter() - t_all):.2f} s")

    if args.json_path:
        out = pathlib.Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "entry": str(entry),
            "profile": args.profile,
            "population": int(args.population),
            "frames": n,
            "per_frame_ms": rows,
            "calls_per_frame": {k: v / n for k, v in calls.items()},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  json -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
