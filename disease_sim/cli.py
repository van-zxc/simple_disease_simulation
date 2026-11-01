"""交互入口：打印运行横幅，把模拟跑起来并把每帧推给本地浏览器仪表盘（不参与模型语义）。

负责：命令行 flag 解析（窗口/节流/浏览器）、仿真循环（模型每帧推进、按 K 帧打包推送）、
      aiohttp 服务的启停与 Ctrl+C 收尾。
不负责：档案/配置解析（config.init_session）、单帧模拟（simulation.update）、
        负载序列化（render.pack_frame）、传输（server.py）。

工作方式：仿真在**工作线程**里同步跑（PyTorch/GIL 友好、不会阻塞事件循环），HTTP/WS 在
主线程的 asyncio 事件循环里；两者通过 ``render.ViewState`` 的"最新一帧"槽位通信。

优先级：显式 --render-every > --async-render（仅当没有显式 --render-every 且
SIM_RENDER_EVERY 未被环境变量改动时生效）> sim.render_every（来自 SIM_RENDER_EVERY）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import webbrowser

from . import config, profiles
from . import render as render_mod
from . import server as server_mod
from .simulation import update

CONFIG = config.CONFIG

# 默认端口：可用 --port 或 DISEASE_VIEW_PORT 覆盖；被占用时服务端自动 +1 重试
DEFAULT_PORT = 8765
# 默认推送帧率上限（模型本身就慢于这个值时不会 sleep）
DEFAULT_MAX_FPS = 60


def _flag_value(name: str) -> str | None:
    """解析 ``--name V`` / ``--name=V``，后出现的覆盖先出现的；未出现返回 None。"""
    args = sys.argv[1:]
    value = None
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def _flag_present(name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def _cli_render_every(default: int) -> int:
    """解析 ``--render-every K``，返回推送间隔（默认沿用 default）。

    后出现的 flag 覆盖先出现的；非法值打印警告后回退 default。``default`` 通常来自
    ``SIM_RENDER_EVERY`` 环境变量（sim.render_every），故显式 flag 优先级更高。
    """
    raw = _flag_value("--render-every")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        print(f"[warn] invalid --render-every {raw!r}, using {default}")
        return default


def _cli_async_render() -> int | None:
    """解析 ``--async-render`` / ``--async-render=K``：返回推送间隔；未出现该 flag 返回 None。

    默认 K=2（模型每帧推进、每 2 帧推一帧）。返回 None 让调用方区分"未请求异步"与
    "请求了但用默认值"。
    """
    if not _flag_present("--async-render"):
        return None
    raw = _flag_value("--async-render")
    if raw is None:
        return 2
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        print(f"[warn] invalid --async-render {raw!r}, using 2")
        return 2


def _is_headless() -> bool:
    """``--headless`` 或 ``DISEASE_VIEW=headless``：不启服务、直接跑完帧数（供工具/CI 用）。"""
    if _flag_present("--headless"):
        return True
    return os.environ.get("DISEASE_VIEW", "").strip().lower() in {"headless", "off", "none"}


def _sim_loop(sim, view, render_every: int, max_fps: float, frames: int) -> None:
    """仿真工作线程：每帧 update；每 render_every 帧打包发布一次；按 max_fps 节流。"""
    started = time.perf_counter()
    for frame in range(frames):
        update(sim, frame)
        if view is not None and render_every > 0 and frame % render_every == 0:
            header, points = render_mod.pack_frame(sim, frame)
            view.publish(frame, header, points)
        if max_fps > 0:
            # 只在模型比 max_fps 快时 sleep；正常档案（10k≈40fps）不会触发
            target = started + (frame + 1) / max_fps
            gap = target - time.perf_counter()
            if gap > 0:
                time.sleep(gap)
    if view is not None:
        view.done = True


def _wait_first_client(view, timeout: float = 180.0) -> None:
    """等待第一个浏览器连接（命令行打印一次提示，超时后照常开跑，避免无人值守时卡死）。"""
    print(f"Waiting for browser to connect … ({timeout:.0f}s timeout, --no-wait to skip)")
    deadline = time.perf_counter() + timeout
    while not view.started.wait(0.05):
        if time.perf_counter() >= deadline:
            print("[warn] no browser connected; starting anyway")
            view.started.set()
            return


async def _run_viewer(sim, render_every: int, host: str, port: int,
                      open_browser: bool, wait: bool, max_fps: float,
                      exit_when_done: bool) -> None:
    """启动服务 + 仿真线程，等模拟跑完（默认继续服务最后一帧，Ctrl+C 退出）。"""
    runner, actual_port = await server_mod.start(sim, render_mod.config.ACTIVE_PROFILE_NAME,
                                                 host=host, port=port)
    url = f"http://{host}:{actual_port}"
    print(f"Dashboard: {url}  (Ctrl+C to quit)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - 无默认浏览器时静默降级
            print(f"[warn] could not open browser automatically; open {url} manually")
    if wait:
        await asyncio.get_running_loop().run_in_executor(None, _wait_first_client, sim.view)
    else:
        sim.view.started.set()

    worker = threading.Thread(
        target=_sim_loop,
        args=(sim, sim.view, render_every, max_fps, CONFIG["FRAMES"]),
        name="sim-loop",
        daemon=True,
    )
    worker.start()
    try:
        while worker.is_alive():
            await asyncio.sleep(0.1)
        worker.join()
        print("Simulation ended")
        if exit_when_done:
            return
        print("Dashboard still serving the final frame; press Ctrl+C to quit.")
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover - 交互退出
        print("\nStopping …")
    finally:
        await runner.cleanup()


def main(sim) -> None:
    """打印运行横幅、确定推送节流参数并启动仪表盘（阻塞到模拟结束/Ctrl+C）。"""
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
    base_every = max(1, int(getattr(sim, "render_every", 1) or 1))
    render_every = _cli_render_every(base_every)
    async_k = _cli_async_render()
    if async_k is not None and render_every == base_every:
        # 相等说明 --render-every 没生效（未给或非法回退），此时 async 开关才覆盖
        render_every = async_k
        print(f"Async push: model steps every frame, dashboard updates every {render_every} frames")
    sim.render_every = render_every
    if render_every > 1:
        print(f"Push throttle: one dashboard frame every {render_every} frames "
              f"(set SIM_RENDER_EVERY=1 for every frame)")

    if _is_headless():
        print("Headless mode: no dashboard, running frames synchronously.")
        _sim_loop(sim, None, render_every, 0.0, CONFIG["FRAMES"])
        print("Simulation ended")
        return

    raw_port = _flag_value("--port") or os.environ.get("DISEASE_VIEW_PORT") or str(DEFAULT_PORT)
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        print(f"[warn] invalid port {raw_port!r}, using {DEFAULT_PORT}")
        port = DEFAULT_PORT
    host = (_flag_value("--host") or os.environ.get("DISEASE_VIEW_HOST") or "127.0.0.1").strip()
    raw_fps = _flag_value("--max-fps")
    try:
        max_fps = float(raw_fps) if raw_fps is not None else DEFAULT_MAX_FPS
    except (TypeError, ValueError):
        print(f"[warn] invalid --max-fps {raw_fps!r}, using {DEFAULT_MAX_FPS}")
        max_fps = DEFAULT_MAX_FPS
    open_browser = not _flag_present("--no-browser")
    wait = not _flag_present("--no-wait")
    exit_when_done = _flag_present("--exit-when-done")

    try:
        asyncio.run(_run_viewer(sim, render_every, host, port, open_browser, wait,
                                max_fps, exit_when_done))
    except KeyboardInterrupt:  # pragma: no cover - Ctrl+C 打断事件循环
        print("\nStopping …")
