"""帧率统计：除平均帧率外，给出游戏显卡评测里常用的 **1% low / 0.1% low** 卡顿指标。

定义（与 Fraps / CapFrameX 等工具一致）：

- **帧时（frame time）**：生成一帧占用的墙钟时间（本仓库里是单帧 ``update`` 的耗时，
  含 ``torch.cuda.synchronize()``；开启绘制时还可以是 ``update + canvas.draw()``）。
- **平均帧率**：``1000 / mean(帧时 ms)``。
- **1% low 帧率**：把帧时从慢到快排序，取最慢的 1%（至少 1 帧）求平均，再取倒数：
  ``1000 / mean(最慢 1% 帧时)``。它回答"最卡的那一小撮帧有多卡"，比平均值更能暴露
  尖峰（内存分配、GC、同步、绘制抖动、GPU 唤醒）。
- **0.1% low**：同上，取最慢 0.1%（至少 1 帧）。

判读：``1% low ≈ 平均帧率`` 说明帧时稳定；``1% low 明显低于平均`` 说明存在偶发长帧。
对仿真来说"帧"就是一次模型步进，所以这里的 fps 也可以读成"每秒能推进多少个模拟帧"。
"""

from __future__ import annotations

import statistics

# 低于该帧数时 low 指标只由 1–2 帧决定，噪声大；文档里会提示样本不足。
MIN_SAMPLES_FOR_STABLE_LOW = 100


def _percentile(sorted_ms: list[float], fraction: float) -> float:
    """最近秩百分位：返回不小于 fraction 比例样本的那个值（样本已升序）。"""
    if not sorted_ms:
        return 0.0
    index = min(len(sorted_ms) - 1, max(0, int(round(fraction * len(sorted_ms))) - 1))
    return sorted_ms[index]


def _worst_mean_ms(sorted_ms: list[float], fraction: float) -> tuple[float, int]:
    """最慢 fraction 比例帧的平均帧时与帧数（至少 1 帧）。"""
    count = max(1, int(round(len(sorted_ms) * fraction)))
    worst = sorted_ms[-count:]
    return statistics.fmean(worst), count


def frame_rate_stats(durations_ms: list[float]) -> dict:
    """把逐帧耗时（ms）汇总成帧率分布 + 1% low / 0.1% low；空输入返回 ``{}``。

    字段：``n``、``avg_fps``、``p50_ms``/``p95_ms``/``p99_ms``/``max_ms``、
    ``one_percent_low``、``point_one_percent_low``（各含 window_frames/mean_ms/fps）。
    """
    samples = [float(v) for v in durations_ms if v is not None]
    if not samples:
        return {}
    ordered = sorted(samples)
    mean_ms = statistics.fmean(samples)
    one_ms, one_n = _worst_mean_ms(ordered, 0.01)
    tenth_ms, tenth_n = _worst_mean_ms(ordered, 0.001)
    return {
        "n": len(samples),
        "avg_fps": round(1000.0 / mean_ms, 1) if mean_ms > 0 else None,
        "p50_ms": round(_percentile(ordered, 0.50), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "p99_ms": round(_percentile(ordered, 0.99), 3),
        "max_ms": round(ordered[-1], 3),
        "one_percent_low": {
            "window_frames": one_n,
            "mean_ms": round(one_ms, 3),
            "fps": round(1000.0 / one_ms, 1) if one_ms > 0 else None,
        },
        "point_one_percent_low": {
            "window_frames": tenth_n,
            "mean_ms": round(tenth_ms, 3),
            "fps": round(1000.0 / tenth_ms, 1) if tenth_ms > 0 else None,
        },
        "stable_sample": len(samples) >= MIN_SAMPLES_FOR_STABLE_LOW,
    }
