"""验证 disease_sim.disease._curve_eval（0.25 天步长 LUT）与解析式等价。

做法：把原分段线性实现（old_eval，纯 torch 逐点）当 oracle，在采样网格上比对
（含折点）。实测 covid19 档案 max|old-new| ≈ 1.19e-07（symp）/ 5.96e-08（asym），
折点处 ~1e-8 —— 这个量级就是 float32 舍入，属正常。

判读阈值：1e-7 量级 = 通过；若某条曲线 >1e-4，说明 LUT 缓存/步长/折点对齐错了
（LUT 步长 0.25 天见 disease_sim/disease.py:19，所有折点都落在整数天=网格点）。

依赖 CUDA（采样网格直接建在 device="cuda"），CPU-only 环境会失败；末尾打印 OK。
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISEASE_PROFILE", "covid19")
os.environ.setdefault("DISEASE_POPULATION", "10000")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from disease_sim.config import CONFIG, init_session  # noqa: E402
from disease_sim.disease import _curve_asym, _curve_eval, _curve_symp  # noqa: E402

init_session([])


def old_eval(x, curve):
    """原分段线性实现（已被 _curve_eval 的 LUT 取代），复刻来当 oracle：先 clamp 再 bucketize。"""
    xs = torch.clamp(x, float(curve[0, 0]), float(curve[-1, 0]))
    idx = torch.bucketize(xs, curve[:, 0].contiguous()).clamp(1, curve.shape[0] - 1)
    x0, y0 = curve[idx - 1, 0], curve[idx - 1, 1]
    x1, y1 = curve[idx, 0], curve[idx, 1]
    return y0 + (y1 - y0) * (xs - x0) / (x1 - x0)


for name, curve in (("symp", _curve_symp()), ("asym", _curve_asym())):
    lo, hi = float(curve[0, 0]), float(curve[-1, 0])
    # 两边各多探 2 天以覆盖 clamp 外的点；4001 点（步长 ~0.005 天）远细于 LUT 的 0.25 天，
    # 插值误差也会被比到。device="cuda"：LUT 缓存与 _curve_eval 都在 CUDA 上。
    xs = torch.linspace(lo - 2.0, hi + 2.0, 4001, device="cuda")
    a = old_eval(xs, curve)
    b = _curve_eval(xs, curve)
    diff = (a - b).abs().max().item()
    print(f"{name}: curve={CONFIG['INFECTIVITY_CURVE'] if name == 'symp' else CONFIG['ASYMPTOMATIC_INFECTIVITY_CURVE']}")
    print(f"{name}: max|old-new| = {diff:.3e}  (range {lo}..{hi})")

# 逐点核对折点处值
for name, curve in (("symp", _curve_symp()), ("asym", _curve_asym())):
    pts = curve[:, 0].contiguous()
    a = old_eval(pts, curve)
    b = _curve_eval(pts, curve)
    print(f"{name}: breakpoint max diff = {(a - b).abs().max().item():.3e}")
print("OK")
