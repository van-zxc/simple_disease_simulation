"""两进程交替 A/B：比较旧快照与当前入口的**单世界每帧耗时**（相对收益的唯一验收口径）。

为什么必须两进程：两侧都要 import `disease_sim` 包，同进程里 `sys.modules` 命中
会让第二侧直接复用第一侧已导入的模块，结果是"两边跑同一份代码"。子进程各自从零
import 才是真 A/B（同进程版本 `tools/_ab.py` 因此只对单文件入口有效）。

用法（父进程；子进程由父进程用 `--child` 自动拉起）：
    python tools/ab_iter.py --profile covid19 --population 10000 --frames 150 --rounds 4
    python tools/ab_iter.py --before _w2_base/simple_disease_simulation.py --json out.json

每轮两侧交替（A→B、B→A、…）让热/时钟漂移均摊；丢弃前 `--drop-rounds` 轮（GPU 时钟
爬坡）后，报"中位数比"与"配对逐轮比值中位数"两个数；输出控制台表格 + `--json` 落盘。

**噪声地板约 ±3%**（docs/CONTRIBUTING.md §5）：低于 3% 的差不能当收益。绝对
ms 跨会话不可比（笔记本热漂移可达 ±30%），只读同会话内的相对变化。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 默认 before 指向第二轮的快照目录；换对比对象用 --before（必须是 .py，原因见
# golden_check.load_module 对 .bak 后缀的说明）。
DEFAULT_BEFORE = ROOT / "_w2_base" / "simple_disease_simulation.py"
DEFAULT_AFTER = ROOT / "simple_disease_simulation.py"


def _run_child(argv: list[str]) -> dict:
    """子进程模式：load → warm → 计时 frames 帧 → 返回一行结果 dict。

    帧号从 0 连续编号（warm 段就承担了"首帧推进"），不额外插 harness 的
    update(0) 探针——两侧协议完全相同即可比，counts 只作同帧号参考、不是验收。
    """
    # 复用 golden_check 的加载器（同 harness 环境注入 + 动画打桩），别另抄一份：
    # 两份加载语义一旦漂移，A/B 测到的差异就混进了工具本身。
    sys.path.insert(0, str(ROOT / "tools"))
    import golden_check as gc  # noqa: E402  (patch matplotlib before torch/mpl use)

    entry = pathlib.Path(argv[0]).resolve()
    env = {
        "DISEASE_PROFILE": argv[1],
        "DISEASE_POPULATION": argv[2],
        "DISEASE_SEED": argv[3],
        "MPLBACKEND": "Agg",
    }
    mod, load_s, _stdout = gc.load_module(entry, env)
    import torch

    warm = int(argv[4])
    frames = int(argv[5])
    # warm 帧不进计时：首帧含 kernel 首次加载与显存分配，算进去会把两侧一起抬高。
    for f in range(warm):
        mod.update(f)
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 整段只同步一次再取 wall（帧流吞吐口径）；逐帧 sync 测的是"单帧延迟"，两者别混。
    t0 = time.perf_counter()
    for f in range(warm, warm + frames):
        mod.update(f)
    torch.cuda.synchronize()
    per_frame_ms = (time.perf_counter() - t0) * 1000.0 / frames

    # counts/title/peak 只作"两侧跑到同一帧号后状态是否同源"的参考（非逐位验收）；
    # title 取第一行：后面几行是 7 日窗口等统计，跨版本本来就会变。
    counts = [int(x) for x in torch.bincount(mod.status, minlength=8).cpu().tolist()]
    title = mod.title_obj.get_text().splitlines()[0] if hasattr(mod, "title_obj") else ""
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
    return {
        "per_frame_ms": per_frame_ms,
        "counts": counts,
        "title": title,
        "peak_mb": peak_mb,
        "load_s": load_s,
    }


def _spawn(entry: pathlib.Path, args) -> dict:
    """父进程拉起一个子进程跑 entry，解析它 stdout 里的 @@AB@@ 结果行。

    子进程失败时只回显 stderr 尾部 12 行（完整 traceback 太长，而父进程要
    一次跑几十个进程）；没有结果行同样按失败处理，避免空数据混进中位数。
    """
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--child",
        str(entry),
        args.profile,
        str(args.population),
        str(args.seed),
        str(args.warm),
        str(args.frames),
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-12:])
        raise SystemExit(f"[ab_iter] child failed ({proc.returncode}) for {entry}:\n{tail}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("@@AB@@"):
            return json.loads(line[len("@@AB@@"):])
    raise SystemExit(f"[ab_iter] no result from {entry}:\n{proc.stdout[-2000:]}")


def main(argv: list[str] | None = None) -> int:
    """父进程：交替跑 rounds 轮 → 丢弃预热轮 → 中位数/配对比 → 打印并落盘。"""
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "--child":
        result = _run_child(argv[1:])
        print("@@AB@@" + json.dumps(result, ensure_ascii=False))
        return 0

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", default=str(DEFAULT_BEFORE))
    p.add_argument("--after", default=str(DEFAULT_AFTER))
    p.add_argument("--profile", default="covid19")
    p.add_argument("--population", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=200, help="frames timed per round")
    # 默认 8 轮 × 200 帧 ≈ 1 分钟量级，够让中位数稳定；warm 40 帧消化 kernel 首启。
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--warm", type=int, default=40)
    # 默认丢前 2 轮：前两轮常撞上 GPU 时钟从 idle 爬到 boost（可差 30%）。
    p.add_argument("--drop-rounds", type=int, default=2,
                   help="warm-up rounds excluded from the median (GPU clock ramp)")
    p.add_argument("--json", dest="json_path", default="")
    args = p.parse_args(argv)

    before = pathlib.Path(args.before)
    after = pathlib.Path(args.after)
    for side in (before, after):
        if not side.is_file():
            raise SystemExit(f"[ab_iter] entry not found: {side}")

    print(f"profile={args.profile} N={args.population} seed={args.seed} "
          f"frames/round={args.frames} rounds={args.rounds} warm={args.warm}")
    print(f"before: {before}\nafter : {after}")

    rounds: list[dict] = []
    for r in range(args.rounds):
        record: dict = {}
        # 奇数轮把前后两侧对调：热/时钟漂移在两个方向上各出现一次，配对时抵消。
        order = [("a", before), ("b", after)]
        if r % 2:
            order.reverse()
        for tag, entry in order:
            res = _spawn(entry, args)
            record[tag] = res
            print(f"  round {r + 1} {'before' if tag == 'a' else 'after ':>6}: "
                  f"{res['per_frame_ms']:.3f} ms/frame", flush=True)
        rounds.append(record)

    kept = rounds[max(0, args.drop_rounds):]
    # 丢完只剩 0/1 轮就没有中位数的意义了：宁可带着热漂移用全量，也不要单点结论。
    if len(kept) < 2:
        kept = rounds
    times_a = [r["a"]["per_frame_ms"] for r in kept]
    times_b = [r["b"]["per_frame_ms"] for r in kept]
    ratios = [r["b"]["per_frame_ms"] / r["a"]["per_frame_ms"] for r in kept]
    last_a = rounds[-1]["a"]
    last_b = rounds[-1]["b"]

    ma, mb = statistics.median(times_a), statistics.median(times_b)
    change = (mb / ma - 1.0) * 100.0
    paired = (statistics.median(ratios) - 1.0) * 100.0
    print()
    print(f"kept rounds   = {len(kept)} (dropped first {args.drop_rounds} warm-up rounds)")
    print(f"before median = {ma:.3f} ms/frame  {[round(v, 2) for v in times_a]}")
    print(f"after  median = {mb:.3f} ms/frame  {[round(v, 2) for v in times_b]}")
    print(f"change        = {change:+.1f}% (medians)   "
          f"{paired:+.1f}% (paired per-round ratio median)  "
          f"({mb - ma:+.3f} ms/frame)")
    if last_a and last_b:
        print(f"counts before = {last_a['counts']}")
        print(f"counts after  = {last_b['counts']}")
        print(f"peak before/after = {last_a['peak_mb']:.2f} / {last_b['peak_mb']:.2f} MB")

    if args.json_path:
        out = pathlib.Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # change_pct/paired_ratio_pct 均只基于 kept 轮；绝对毫秒只标同会话可比。
        payload = {
            "profile": args.profile,
            "population": args.population,
            "seed": args.seed,
            "frames_per_round": args.frames,
            "rounds": args.rounds,
            "warm": args.warm,
            "before": {"entry": str(before), "times": times_a, "median": ma,
                       "counts": last_a["counts"] if last_a else None},
            "after": {"entry": str(after), "times": times_b, "median": mb,
                      "counts": last_b["counts"] if last_b else None},
            "change_pct": change,
            "paired_ratio_pct": paired,
            "dropped_rounds": args.drop_rounds,
            "peak_mb": {"before": last_a["peak_mb"] if last_a else None,
                        "after": last_b["peak_mb"] if last_b else None},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
