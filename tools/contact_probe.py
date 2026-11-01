"""接触率探针：实测活动模型每帧"每人有效接触数"（C），并按场所分段。

用途：城市结构（区/邻里/场馆位置与尺度）改动后，验证接触率有没有被改坏。
``disease_sim/profiles/schema.py`` 的 ``CONTACT_RATE_PER_FRAME = 24.6`` 就是本探针
在 10k 基准人口上的日均值（夜间≈17、工作时段 85–99、日均≈25）；R0 的解析式标定
依赖它，所以改结构后必须重测。

接触定义与 ``disease_sim/contacts.py`` 一致：两活体位置欧氏距离 < ``INFECT_RADIUS``
即算一次接触（对称，按无向对计数；平均每人接触数 = 2·对数 / 存活人数）。

用法::

    MPLBACKEND=Agg python tools/contact_probe.py --profile covid19 --population 10000 \
        --frames 1200 --stride 5 --json benchmarks/contact_before.json

注意：本脚本只读模型状态、不改任何仿真参数；采样帧会推进同一个 sim，所以
``--frames`` 越大越慢（10k 约每秒 3–5 帧，含距离矩阵计算）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import _loader

ROOT = _loader.ROOT


def _per_person_degree(pos: torch.Tensor, radius: float, chunk: int = 512) -> torch.Tensor:
    """逐人接触度数（对称、排除自身），在 GPU 上分块算距离矩阵。

    返回 ``deg_i`` = 与 i 距离 < radius 的其他个体数；总对数 = ``deg.sum()/2``。
    分块只为控制显存峰值（``chunk × N`` 的距离矩阵），不影响结果。
    """
    n = int(pos.shape[0])
    # 分块宽度按人口自适应：单块距离矩阵控制在 ~2e8 元素（约 800 MB fp32）以内
    chunk = max(32, min(int(chunk), int(2.0e8 / max(n, 1))))
    deg = torch.zeros(n, dtype=torch.int64, device=pos.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(pos[start:stop], pos)
        # 对角线上是"自己与自己"，每个 i 恰好一次，最后统一减 1
        deg[start:stop] = (d < radius).sum(dim=1) - 1
    return deg


def _measure(sim, frame: int, radius: float) -> dict:
    status = sim.status
    alive_mask = status != 4
    n_alive = int(alive_mask.sum())
    # 只对存活个体算度数：死人位置仍在，但不算接触
    pos_alive = sim.pos[alive_mask]
    deg_alive = _per_person_degree(pos_alive, radius).cpu().numpy().astype(np.float64)
    deg = np.zeros(int(status.shape[0]), dtype=np.float64)
    deg[alive_mask.cpu().numpy()] = deg_alive
    pairs = float(deg.sum() / 2.0)
    mean_all = float(deg.sum() / max(n_alive, 1))

    alive = alive_mask.cpu().numpy()
    place = sim.place.cpu().numpy()
    traveling = sim.traveling.cpu().numpy()

    # 分段：只在同一场所类别内看平均度数（在途另算）
    seg = {}
    for code, name in ((0, "home"), (1, "work"), (2, "school"), (3, "leisure")):
        sel = alive & (~traveling) & (place == code)
        seg[name] = float(deg[sel].mean()) if sel.any() else 0.0
    sel = alive & traveling
    seg["transit"] = float(deg[sel].mean()) if sel.any() else 0.0
    seg["home_share"] = float((alive & (~traveling) & (place == 0)).sum() / max(n_alive, 1))
    return {"frame": frame, "n_alive": n_alive, "pairs": pairs,
            "mean_contacts": mean_all, "segments": seg}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="接触率探针（只读）")
    ap.add_argument("--profile", default=None, help="疾病档案名（默认走 DEFAULT_PROFILE=covid19）")
    ap.add_argument("--population", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=1200, help="总帧数（10k 下 ~3–5 帧/秒）")
    ap.add_argument("--warmup", type=int, default=0, help="跳过前 N 帧再采样")
    ap.add_argument("--stride", type=int, default=5, help="每 N 帧采样一次")
    ap.add_argument("--json", default=None, help="把汇总写到该 JSON 路径")
    args = ap.parse_args(argv)

    env = {
        "DISEASE_POPULATION": str(args.population),
        "DISEASE_SEED": str(args.seed),
        "SIM_FRAMES": str(args.frames),
    }
    if args.profile:
        env["DISEASE_PROFILE"] = args.profile
    module, load_s, init_out = _loader.load_module(
        ROOT / "simple_disease_simulation.py", env)
    sim = module._SIM
    cfg = module.CONFIG
    radius = float(cfg["INFECT_RADIUS"])
    min_per_frame = float(sim.MIN_PER_FRAME)
    clock_start = float(cfg["CLOCK_START_MIN"])

    samples = []
    for frame in range(args.frames):
        module.update(frame)
        if frame < args.warmup or frame % max(1, args.stride):
            continue
        rec = _measure(sim, frame, radius)
        clock = (clock_start + frame * min_per_frame) % 1440.0
        rec["hour"] = float(clock // 60)
        samples.append(rec)

    if not samples:
        print("[contact_probe] 没有采样（frames/warmup/stride 设置太小）")
        return 1

    means = np.array([s["mean_contacts"] for s in samples])
    hours = np.array([s["hour"] for s in samples])
    night = means[(hours >= 1) & (hours < 4)]
    daytime_work = means[(hours >= 9) & (hours < 17)]
    seg_names = ("home", "work", "school", "leisure", "transit")
    seg_mean = {name: float(np.mean([s["segments"][name] for s in samples]))
                for name in seg_names}

    print(f"[contact_probe] profile={module.ACTIVE_PROFILE_NAME} N={sim.N} "
          f"radius={radius} load={load_s:.2f}s samples={len(samples)}")
    print(f"[contact_probe] C(日均每帧每人接触数) = {means.mean():.3f} "
          f"(p10 {np.percentile(means, 10):.1f} / p90 {np.percentile(means, 90):.1f})")
    print(f"[contact_probe] 夜间(01–04 点) = {night.mean() if night.size else float('nan'):.3f} | "
          f"工作时段(09–17 点) = {daytime_work.mean() if daytime_work.size else float('nan'):.3f}")
    print("[contact_probe] 分场所平均每人接触数："
          + " ".join(f"{k}={v:.2f}" for k, v in seg_mean.items()))

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile": module.ACTIVE_PROFILE_NAME, "population": int(sim.N),
            "seed": args.seed, "frames": args.frames, "stride": args.stride,
            "infect_radius": radius, "samples": len(samples),
            "C_mean": float(means.mean()),
            "C_p10": float(np.percentile(means, 10)),
            "C_p90": float(np.percentile(means, 90)),
            "C_night": float(night.mean()) if night.size else None,
            "C_workhours": float(daytime_work.mean()) if daytime_work.size else None,
            "segments": seg_mean,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[contact_probe] 写出 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
