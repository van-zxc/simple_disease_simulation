"""量"视图负载打包"（``render.pack_frame``）的每帧成本，并按 ``render_every`` K 摊薄。

背景：视图栈已从 matplotlib 换成浏览器仪表盘（disease_sim/webui + server.py）。
服务端每帧只做两次拷贝：统计 dict → JSON 字节、抽样点 → 二进制块（Float32 坐标 + Uint8 状态码）；
绘制在浏览器里。所以"渲染成本"在这里指 **pack 成本**（旧口径的 draw+blit 已不存在，不能混报）。

用法（位置参数，均可省）：
    python tools/render_prof.py [profile] [population] [frames] [render_every] [points]
默认 covid19 / 10000 / 40 帧 / 每帧打包 / 不设 SIM_RENDER_POINTS。
输出：update+sync、单次 pack（ms + 字节数）、摊薄后的 amortized = per_upd + t_pack/frames。
"""
import pathlib
import sys
import time

# 复用 _loader 的无界面加载器；tools 是相对路径，因此必须从仓库根运行。
sys.path.insert(0, str(pathlib.Path("tools").resolve()))
import _loader as gc  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

profile = sys.argv[1] if len(sys.argv) > 1 else "covid19"
pop = sys.argv[2] if len(sys.argv) > 2 else "10000"
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 40
every = max(1, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
points = sys.argv[5] if len(sys.argv) > 5 else None

env = {"DISEASE_PROFILE": profile, "DISEASE_POPULATION": pop, "DISEASE_VIEW": "headless"}
if points:
    env["SIM_RENDER_POINTS"] = points
mod, _, _ = gc.load_module(ROOT / "simple_disease_simulation.py", env)

import torch  # noqa: E402

from disease_sim import config as sim_config  # noqa: E402
from disease_sim import render as render_mod  # noqa: E402

sim = mod._SIM

# 15 帧 warm：让懒初始化（接触网格/缓存）落在计时区外。
for f in range(0, 15):
    mod.update(f)
torch.cuda.synchronize()

t_upd = t_pack = 0.0
n_packed = 0
header_bytes = 0
points_bytes = 0
for f in range(15, 15 + frames):
    t0 = time.perf_counter()
    mod.update(f)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if f % every == 0:
        header, blob = render_mod.pack_frame(sim, f)
        header_bytes += len(header)
        points_bytes += len(blob)
        t_pack += time.perf_counter() - t1
        n_packed += 1
    t_upd += t1 - t0

per_upd = t_upd / frames * 1000.0
per_pack = t_pack / max(1, n_packed) * 1000.0
# 摊薄：每帧都要 update，但打包成本只在 K 帧里付一次 → 分母是 frames（不是 n_packed）。
amortized = per_upd + (t_pack / frames) * 1000.0
per_frame_kb = (header_bytes + points_bytes) / max(1, n_packed) / 1024.0
print(f"profile={profile} pop={pop} points={sim_config.CONFIG['RENDER_MAX_POINTS']} "
      f"render_every={every} packed={n_packed}")
print(f"update+sync           : {per_upd:7.2f} ms/frame")
print(f"pack (packed frames)  : {per_pack:7.2f} ms/pack "
      f"({per_frame_kb:6.1f} KB/pack = {header_bytes / max(1, n_packed) / 1024.0:.1f} KB JSON "
      f"+ {points_bytes / max(1, n_packed) / 1024.0:.1f} KB 点云)")
print(f"=> amortized/frame    : {amortized:7.2f} ms/frame "
      f"({1000.0 / amortized:5.1f} fps)")
