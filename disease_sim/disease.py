"""分段线性传染力曲线及其求值器（LUT 加速版）。

曲线是"距发病天数 → 相对传染力"的分段线性函数，每帧对 E/I/A 三类人各求值一次。
原实现每次 bucketize + 4 次二维索引 + 5 次算术，约 20 个 aten 事件；Windows/WDDM
下 kernel launch 本身就是主要成本，故改为"0.25 天均匀网格 LUT + 2 次一维 gather"。

负责：曲线张量 → 网格 LUT 的离散化，以及逐帧的插值求值。
不负责：曲线数值本身（来自 CONFIG/档案）、感染事件的抽样（在 simulation 里）。

不变式：折点全在整数天、网格与折点对齐，线性插值与原分段线性函数数学等价
（仅浮点末位差异，最大偏差 ~1.2e-7，`python （脚本已删）` 可复现）。
惰性构建：CONFIG 在 import 之后才被档案写入，import 期构建会拿到 _DEFAULTS 默认曲线。
"""

import torch

from .config import CONFIG, device

# 惰性单例：首次求值时才按当前 CONFIG 建曲线（import 期建会拿到 _DEFAULTS 默认值）
_CURVE_SYMP: torch.Tensor | None = None
_CURVE_ASYM: torch.Tensor | None = None

# 曲线张量 → (lut, step, x_min, x_max, n)。键用 data_ptr+shape 而不是张量本身：
# 两条进程级曲线各 1 个张量且不重建，键稳定；代价是张量被替换/回收后 data_ptr 可能
# 被复用而误命中——若将来支持按档案重建曲线，必须同时清空本缓存
_LUT_CACHE: dict[tuple, tuple] = {}
# 网格步长 0.25 天：折点都是整数天，0.25 整除 1，网格点与折点重合 → 插值精确；
# 更细没有收益（折点之间本来就是直线），更粗会切进折点造成偏差
_LUT_STEP = 0.25


def _curve_symp() -> torch.Tensor:
    """有症状者传染力曲线：CONFIG["INFECTIVITY_CURVE"] 的 (K,2) 张量（懒构建）。"""
    global _CURVE_SYMP
    if _CURVE_SYMP is None:
        _CURVE_SYMP = torch.tensor(CONFIG["INFECTIVITY_CURVE"],
                                   dtype=torch.float32, device=device)
    return _CURVE_SYMP


def _curve_asym() -> torch.Tensor:
    """无症状者传染力曲线：CONFIG["ASYMPTOMATIC_INFECTIVITY_CURVE"] 的 (K,2) 张量（懒构建）。"""
    global _CURVE_ASYM
    if _CURVE_ASYM is None:
        _CURVE_ASYM = torch.tensor(CONFIG["ASYMPTOMATIC_INFECTIVITY_CURVE"],
                                   dtype=torch.float32, device=device)
    return _CURVE_ASYM


def _curve_lut(curve):
    """把分段线性曲线离散成 0.25 天均匀网格 LUT，返回 (lut, step, x_min, x_max, n)。

    每进程每曲线只建一次（缓存在 _LUT_CACHE，键 data_ptr+shape）；建表是 Python 循环，
    但只跑一次、不在热路径。不变式：n 覆盖两个端点，故 lut 长度 = 步数 + 1。
    """
    key = (curve.data_ptr(), tuple(curve.shape))
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    xs = [float(v) for v in curve[:, 0].tolist()]
    ys = [float(v) for v in curve[:, 1].tolist()]
    x_min, x_max = xs[0], xs[-1]
    n = int(round((x_max - x_min) / _LUT_STEP)) + 1   # +1：网格含两个端点
    grid = [x_min + i * _LUT_STEP for i in range(n)]
    values = []
    seg = 0
    # 性能：grid 单调递增，单指针顺序扫过折点即可（不必每点二分）；整表只建一次
    for gx in grid:
        while seg + 2 < len(xs) and xs[seg + 1] < gx:
            seg += 1
        x0, x1 = xs[seg], xs[seg + 1]
        y0, y1 = ys[seg], ys[seg + 1]
        values.append(y0 + (y1 - y0) * (gx - x0) / (x1 - x0))
    lut = torch.tensor(values, dtype=torch.float32, device=curve.device)
    cached = (lut, _LUT_STEP, x_min, x_max, n)
    _LUT_CACHE[key] = cached
    return cached


def _curve_eval(x, curve):
    """分段线性插值（LUT 版）：x 为"距发病天数"张量，返回同形状的相对传染力。

    不变式：x 先 clamp 到 [x_min, x_max]，越界输入按端点值处理；用 gather 取相邻两格
    做线性插值，与逐段 bucketize 版数学等价（仅浮点末位差异）。
    """
    lut, step, x_min, x_max, n = _curve_lut(curve)
    pos = (torch.clamp(x, x_min, x_max) - x_min) * (1.0 / step)
    # idx 夹到 [0, n-2] 保证 idx+1 不越界；pos 最大为 n-1，恰在最后一格的右端点
    idx = pos.to(torch.int64).clamp_(0, n - 2)
    frac = pos - idx.to(pos.dtype)
    y0 = torch.gather(lut, 0, idx)
    y1 = torch.gather(lut, 0, idx + 1)
    return y0 + (y1 - y0) * frac
