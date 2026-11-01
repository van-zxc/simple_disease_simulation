"""本地 Web 仪表盘服务：静态页 + WebSocket 推帧 + 封控开关（基于 aiohttp，无前端依赖）。

职责边界：本模块只管"传输"——把 ``render.pack_frame`` 的字节送到浏览器、把按钮点击翻译成
``sim.toggle()``。仿真循环在 ``cli.py`` 的工作线程里跑，两者通过 ``render.ViewState`` 的
"最新一帧"槽位解耦：慢客户端只会掉帧，不会让仿真变慢或让内存无界增长。

路由：
    GET  /            仪表盘页面（``webui/index.html``）
    GET  /app.js      前端脚本
    GET  /style.css   样式
    GET  /ws          WebSocket：先 meta（JSON）+ 场所层（二进制），再逐帧 "JSON 头 + 点云"
    POST /api/toggle  翻转封控开关，返回 ``{"lockdown": bool}``
    GET  /api/health  健康检查（``{"ok": true, "frame": 最新帧号}``），供脚本冒烟用

绑定默认 ``127.0.0.1``：回环地址不需要管理员权限、也不会触发 Windows 防火墙弹窗。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import WSMsgType, web

from . import render as render_mod

WEBUI_DIR = Path(__file__).resolve().parent / "webui"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

# 背压阈值：单个连接待写信缓冲超过这个大小就丢帧（浏览器卡住时保护内存）
_MAX_WRITE_BUFFER = 4 * 1024 * 1024
# WS 循环的轮询间隔（秒）：仿真线程每帧 publish 一次，这里按 ~150Hz 检查新帧
_POLL_INTERVAL = 0.006


def _static_handler(filename: str):
    """按文件名返回 webui 目录下的静态资源（禁缓存：改前端后刷新即生效）。"""

    async def handler(request: web.Request) -> web.StreamResponse:
        path = WEBUI_DIR / filename
        if not path.is_file():
            raise web.HTTPNotFound(text=f"缺少前端文件：{filename}")
        ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        # 注意：charset 必须完整写在 Content-Type 头里，不能走 aiohttp 的 content_type 参数
        return web.Response(body=path.read_bytes(),
                            headers={"Content-Type": ctype, "Cache-Control": "no-store"})

    return handler


async def _ws_handler(request: web.Request) -> web.StreamResponse:
    """推送 meta + 场所层，然后按帧号去重推逐帧负载。"""
    sim = request.app["sim"]
    view = sim.view
    ws = web.WebSocketResponse(heartbeat=15, max_msg_size=0)
    await ws.prepare(request)

    meta = render_mod.meta(sim)
    await ws.send_str(json.dumps(meta, ensure_ascii=False))
    _layers, venue_blob = sim._venue_blob
    if venue_blob:
        await ws.send_bytes(venue_blob)
    # 第一个连接建立 → 放开仿真线程（cli 在 --no-wait 时已预先置位）
    view.started.set()

    # 客户端可能发来 {"cmd": "toggle"}：与 POST 等价，方便纯 WS 客户端
    async def _drain() -> None:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("cmd") == "toggle":
                    state = bool(sim.toggle())
                    await ws.send_str(json.dumps({"type": "ack", "lockdown": state}))
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break

    drain_task = asyncio.create_task(_drain())
    last_sent = -1
    try:
        while not ws.closed:
            payload = view.take_new()
            if payload is None:
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            frame, header, points = payload
            if frame <= last_sent:
                continue
            try:
                transport = request.transport
                if transport is not None and transport.get_write_buffer_size() > _MAX_WRITE_BUFFER:
                    last_sent = frame  # 丢帧：直接跳过，等客户端追上来
                    continue
                await ws.send_str(header.decode("utf-8"))
                await ws.send_bytes(points)
                last_sent = frame
            except (ConnectionResetError, RuntimeError):
                break
            await asyncio.sleep(0)
    except asyncio.CancelledError:  # pragma: no cover - 服务关闭时
        raise
    finally:
        drain_task.cancel()
    # 连接结束前正常关闭（客户端可能已先关，忽略二次关闭的异常）
    return ws


async def _toggle_handler(request: web.Request) -> web.StreamResponse:
    sim = request.app["sim"]
    state = bool(sim.toggle())
    return web.json_response({"lockdown": state})


async def _health_handler(request: web.Request) -> web.StreamResponse:
    sim = request.app["sim"]
    return web.json_response({
        "ok": True,
        "frame": sim.view.frame,
        "population": int(sim.N),
        "profile": request.app["profile"],
        "locked": bool(getattr(sim, "lockdown_active", False)),
    })


def _make_app(sim, profile_name: str) -> web.Application:
    app = web.Application()
    app["sim"] = sim
    app["profile"] = profile_name
    app.router.add_get("/", _static_handler("index.html"))
    app.router.add_get("/app.js", _static_handler("app.js"))
    app.router.add_get("/style.css", _static_handler("style.css"))
    app.router.add_get("/ws", _ws_handler)
    app.router.add_post("/api/toggle", _toggle_handler)
    app.router.add_get("/api/health", _health_handler)
    return app


async def start(sim, profile_name: str, host: str = "127.0.0.1", port: int = 8765,
                max_tries: int = 25):
    """启动服务；端口被占用时逐个 +1 重试。返回 (runner, 实际端口)。"""
    app = _make_app(sim, profile_name)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    last_error: Exception | None = None
    for candidate in range(port, port + max_tries):
        site = web.TCPSite(runner, host, candidate)
        try:
            await site.start()
        except OSError as exc:  # 端口被占用（Windows 上是 WinError 10048）
            last_error = exc
            continue
        return runner, candidate
    await runner.cleanup()
    raise SystemExit(f"端口 {port}-{port + max_tries - 1} 全部被占用：{last_error}")
