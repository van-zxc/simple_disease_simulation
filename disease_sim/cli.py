"""交互入口：打印运行横幅并启动 matplotlib 动画（不参与模型语义）。

负责：命令行 flag 解析（--render-every / --async-render）、帧回调（模型每帧推进、
按 K 帧重绘）、FuncAnimation 装配。
不负责：档案/配置解析（config.init_session）、单帧模拟（simulation.update）。

优先级：显式 --render-every > --async-render（仅当没有显式 --render-every 且
SIM_RENDER_EVERY 未被环境变量改动时生效）> sim.render_every（来自 SIM_RENDER_EVERY）。
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import config, profiles
from .simulation import update

CONFIG = config.CONFIG


def _cli_render_every(default: int) -> int:
    """解析 ``--render-every K`` / ``--render-every=K``，返回重绘间隔（默认沿用 default）。

    后出现的 flag 覆盖先出现的；非法值打印警告后回退 default。``default`` 通常来自
    ``SIM_RENDER_EVERY`` 环境变量（sim.render_every），故显式 flag 优先级更高。
    """
    args = sys.argv[1:]
    raw = None
    for index, arg in enumerate(args):
        if arg == "--render-every" and index + 1 < len(args):
            raw = args[index + 1]
        elif arg.startswith("--render-every="):
            raw = arg.split("=", 1)[1]
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        print(f"[warn] invalid --render-every {raw!r}, using {default}")
        return default


def _cli_async_render() -> int | None:
    """解析 ``--async-render`` / ``--async-render=K``：返回重绘间隔；未出现该 flag 返回 None。

    默认 K=2（模型每帧推进、每 2 帧重绘一次）。收益来自绘制帧耗时约 2.6×更新帧耗时，
    隔帧重绘后**有效帧率提升约 1.6×**（实测：``python tools/render_prof.py covid19 10000 40 2``）。
    返回 None 让调用方区分"未请求异步"与"请求了但用默认值"。
    """
    args = sys.argv[1:]
    raw = None
    found = False
    for index, arg in enumerate(args):
        if arg == "--async-render":
            found = True
        elif arg.startswith("--async-render="):
            found = True
            raw = arg.split("=", 1)[1]
    if not found:
        return None
    if raw is None:
        return 2
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        print(f"[warn] invalid --async-render {raw!r}, using 2")
        return 2


def main(sim) -> None:
    """打印运行横幅、确定重绘节流参数并启动动画（阻塞到窗口关闭）。"""
    print(f"Initialization complete, {sim.N} agents")
    # 惰性读 config.PROFILE_INFO：init_session() 会重新绑定这个字典，模块导入期拿到的是空 dict
    print(profiles.format_summary(config.PROFILE_INFO))
    print(f"\nImmunity System Parameters:")
    print(f"  - Immunity gain per infection: {CONFIG['IMMUNITY_GAIN_PER_INFECTION']}")
    print(f"  - Immunity decay rate: {CONFIG['IMMUNITY_DECAY_RATE']} per step")
    print(f"  - Max immunity: {CONFIG['MAX_IMMUNITY']}")
    print(f"  - Immunity effectiveness: {CONFIG['IMMUNITY_EFFECTIVENESS']}x")
    print(f"\nMechanism: Recovered individuals gain immunity that decays over time")
    print(f"Reinfection probability = base_prob × (1 - immunity)^{CONFIG['IMMUNITY_EFFECTIVENESS']}")

    # 优先级：显式 --render-every > SIM_RENDER_EVERY（sim.render_every）> 1
    render_every = _cli_render_every(max(1, int(getattr(sim, "render_every", 1) or 1)))
    async_k = _cli_async_render()
    if async_k is not None and render_every == max(1, int(getattr(sim, "render_every", 1) or 1)):
        # 相等说明 --render-every 没生效（未给或非法回退），此时 async 开关才覆盖
        render_every = async_k
        print(f"Async render: model steps every frame, redraw every {render_every} frames "
              f"(effective fps ≈ 1.6x at render_every=2)")
    # 写回 sim，供 F 段与诊断复用
    sim.render_every = render_every
    if render_every > 1:
        print(f"Render throttle: draw every {render_every} frames "
              f"(set SIM_RENDER_EVERY=1 for every frame)")

    def _frame(frame):
        # 模型每帧都推进；但只用每 K 帧重绘一次（返回空列表 = 本帧不重绘，
        # blit=True 下 FuncAnimation 不会画任何 artist，画面保持上一帧）。
        # 调用方契约：非绘制帧必须返回 []（不是 artists），否则 blit 会重画全部 artist
        artists = update(sim, frame)
        if render_every > 1 and frame % render_every:
            return []
        return artists

    # interval=1：节流由 _frame 自己控制，mpl 定时器只负责尽快调度；
    # cache_frame_data=False：帧数据由 update() 现算，缓存只会白占内存（长跑无界增长）
    ani = FuncAnimation(sim.fig, _frame, frames=CONFIG["FRAMES"], interval=1,
                        blit=True, repeat=False, cache_frame_data=False)
    plt.show()
    print("Simulation ended")
