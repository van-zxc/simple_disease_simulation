"""量交互渲染（FuncAnimation blit=True）的每帧成本，并按 render_every K 摊薄。

复刻的是 matplotlib blit 协议：每 axes 缓存一次背景 → 每帧 restore_region →
draw_artist 每个 artist → 每 axes 一次 blit；模型每帧都 update，但只有第 K 帧重绘。
这与 bench.py 的全量 canvas.draw()（最坏口径）不是一回事，不能混报（SCALING.md §6.2d/§6.5）。

用法（位置参数，均可省）：
    python tools/render_prof.py [profile] [population] [frames] [render_every] [points]
默认 covid19 / 10000 / 40 帧 / 每帧重绘 / 不设 SIM_RENDER_POINTS。
输出 update+sync、单次 draw、以及摊薄后的 amortized = per_upd + t_render/frames。
"""
import os
import pathlib
import sys
import time

# 无头 Agg + 复用 golden_check 的加载器；tools 是相对路径，因此必须从仓库根运行。
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(pathlib.Path("tools").resolve()))
import golden_check as gc  # noqa: E402

profile = sys.argv[1] if len(sys.argv) > 1 else "covid19"
pop = sys.argv[2] if len(sys.argv) > 2 else "10000"
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 40
every = max(1, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
points = sys.argv[5] if len(sys.argv) > 5 else None

env = {"DISEASE_PROFILE": profile, "DISEASE_POPULATION": pop}
if points:
    env["SIM_RENDER_POINTS"] = points
mod, _, _ = gc.load_module(pathlib.Path("simple_disease_simulation.py"), env)
import torch  # noqa: E402

from disease_sim import config as sim_config  # noqa: E402

fig = mod.fig
canvas = fig.canvas
# update(0) 先跑一次探针拿本帧的 artist 列表；axes 按 id 去重（多个 artist 共享同一
# axes，重复 restore/blit 等于把同一块区域画两遍）。
artists = list(mod.update(0))
axes, seen = [], set()
for a in artists:
    ax = a.axes
    if ax is not None and id(ax) not in seen:
        seen.add(id(ax))
        axes.append(ax)

# 全量 draw 一次 + 15 帧 warm：让字体/布局/bbox 等一次性开销落在计时区外。
canvas.draw()
for f in range(1, 15):
    list(mod.update(f))
torch.cuda.synchronize()
# blit 协议第一步：每 axes 缓存一次背景（之后每帧只还原这块区域，不重画整图）。
backgrounds = {ax: canvas.copy_from_bbox(ax.bbox) for ax in axes}

t_upd = t_render = 0.0
n_rendered = 0
for f in range(15, 15 + frames):
    t0 = time.perf_counter()
    artists = list(mod.update(f))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if f % every == 0:
        # 复刻 FuncAnimation(blit=True) 的每帧动作：还原背景 → 只画 artist → 提交 blit。
        # 顺序不能换：先 draw_artist 再 blit，否则提交上去的还是上一帧。
        for ax in axes:
            canvas.restore_region(backgrounds[ax])
        for a in artists:
            a.axes.draw_artist(a)
        for ax in axes:
            canvas.blit(ax.bbox)
        n_rendered += 1
        t_render += time.perf_counter() - t1
    t_upd += t1 - t0

per_upd = t_upd / frames * 1000.0
per_render = t_render / max(1, n_rendered) * 1000.0
# 摊薄：每帧都要 update，但渲染成本只在 K 帧里付一次 → 分母是 frames（不是 n_rendered）。
amortized = per_upd + (t_render / frames) * 1000.0
# 回显 RENDER_MAX_POINTS：10k 点以上会被抽样，先确认本机实际点数才能对上报告里的数字。
print(f"profile={profile} pop={pop} points={sim_config.CONFIG['RENDER_MAX_POINTS']} "
      f"render_every={every} artists={len(artists)} axes={len(axes)}")
print(f"update+sync           : {per_upd:7.2f} ms/frame")
print(f"render (drawn frames) : {per_render:7.2f} ms/draw ({n_rendered} draws)")
print(f"=> amortized/frame    : {amortized:7.2f} ms/frame "
      f"({1000.0 / amortized:5.1f} fps)")
