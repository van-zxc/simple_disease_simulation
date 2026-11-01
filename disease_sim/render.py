"""视图层：把每帧的仿真状态打包成浏览器可直接绘制的负载（不依赖任何绘图库）。

分工（模块边界）：
- ``build_render(sim)``：装配期一次性初始化（抽样索引、场所静态层、历史容器、封控开关）；
- ``meta(sim)``：连接建立时发送一次静态描述（人口/世界/帧数/床位/颜色/场所层清单）；
- ``venue_blob(sim)``：场所层二进制点集（Float32 x,y,size + Uint8 layer），只发一次；
- ``pack_frame(sim, frame)``：一帧的动态负载（JSON 头 + 点云二进制）。

调用方契约（``server.py``）：每帧先发 JSON（含 ``t`` 与 ``n``），紧跟点云二进制；
两段用帧号配对。``sim.last_stats`` 由 ``simulation.update`` 的 F 段写入，本模块只做序列化。

为什么是"打包"而不是"渲染"：曲线自动缩放、指标排版、字体渲染全部交给浏览器（Canvas 2D），
服务端每帧只做「统计 dict → JSON」+「抽样点 → 连续内存块」两次拷贝，没有图元/坐标轴概念。
"""

from __future__ import annotations

import json
import threading

import numpy as np
import torch

from . import config
from . import profiles as disease_profiles
from .config import CONFIG, device

# 状态 → 颜色：下标即 status 码 0..7（顺序必须与 state 里的状态定义一致）
COLOR_MAP = ["#B0B0B0", '#00BFFF', '#FFD700', '#FF4500', '#000000', '#32CD32',
             '#FF8C00', '#8A2BE2']
# 与 COLOR_MAP 同序的英文全称：服务端只给标签，中文注释在前端 i18n 表里
STATUS_LABELS = ["Healthy", "Immune", "Incubating", "Symptomatic", "Dead", "Recovered",
                 "Asymptomatic", "Hospitalized"]

# 场所静态层：(层名, sim 上的坐标属性, 占用人数属性, 展示标签, 颜色, 形状名)
# 形状名由前端映射到 Canvas 绘制路径（square/triangle/circle/cross）。
VENUE_STYLES = [
    ("workplace", "work_venue_np", "work", "Workplace 工作场所", '#4C72B0', "square"),
    ("school", "school_venue_np", "school", "School 学校", '#8172B2', "triangle"),
    ("shop", "shop_venue_np", "shop", "Shop 商店", '#DD8452', "circle"),
    ("park", "park_venue_np", "park", "Park 公园", '#55A868', "cross"),
]
# 点大小基准：10k 基准下每个工作场所的平均在岗人数（与旧 s=64 的基准一致）
VENUE_REFERENCE_OCCUPANCY = 208.0
# 前端把 size_norm = sqrt(occ/208) 映射成屏幕像素直径时的钳位（像素）
VENUE_SIZE_MIN_PX = 1.6
VENUE_SIZE_MAX_PX = 26.0


class ViewState:
    """跨线程的"最新一帧"槽位：仿真线程写、HTTP/WS 线程读。

    ``latest`` 只保留最新一帧（旧帧被覆盖），所以慢客户端不会积压内存，最坏情况是掉帧。
    ``started`` 供仿真线程等待第一个浏览器连接（``--no-wait`` 时预先置位）。
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.done = False
        self._lock = threading.Lock()
        self._latest: tuple[int, bytes, bytes] | None = None
        self._sent = -1

    def publish(self, frame: int, header: bytes, points: bytes) -> None:
        with self._lock:
            self._latest = (int(frame), header, points)

    def take_new(self) -> tuple[int, bytes, bytes] | None:
        """取出比上次取出的更新的一帧；没有新帧返回 None（同一帧不会重复发送）。"""
        with self._lock:
            if self._latest is None or self._latest[0] <= self._sent:
                return None
            self._sent = self._latest[0]
            return self._latest

    @property
    def frame(self) -> int:
        """最新已发布的帧号（从未发布返回 -1）。"""
        with self._lock:
            return -1 if self._latest is None else self._latest[0]


def _make_toggle(sim):
    """封控开关：浏览器点按钮 → POST /api/toggle → 这里翻转并返回新状态。

    参数 event 可选（保留旧 mpl 回调签名，headless/工具可直接 ``sim.toggle()``）。
    """

    def toggle(event=None) -> bool:
        sim.lockdown_active = not bool(getattr(sim, "lockdown_active", False))
        return sim.lockdown_active

    return toggle


def build_render(sim) -> None:
    """装配期初始化视图状态（取代旧的 matplotlib 图形搭建）。"""
    sim._RENDER_MAX = int(CONFIG["RENDER_MAX_POINTS"])
    if sim._RENDER_MAX and sim.N > sim._RENDER_MAX:
        # 大人口下只传固定抽样点（完整状态仍全部参与模拟）；固定抽样保证画面结构稳定
        sim.render_idx = torch.randperm(sim.N, device=device)[:sim._RENDER_MAX]
    else:
        sim.render_idx = None
    # 历史曲线（数据侧保留全历史；浏览器自己裁剪到 500 点窗口）
    sim.history = {s: [] for s in range(8)}
    sim.virus_history = {'beta': [], 'mu': [], 'gen': [], 'immunity': []}
    sim.last_stats = None
    if not hasattr(sim, "lockdown_active"):
        sim.lockdown_active = False
    sim.toggle = _make_toggle(sim)
    sim.view = ViewState()
    # 场所层二进制（静态，连接时只发一次）
    sim._venue_blob = _build_venue_blob(sim)


def _venue_size_norm(occupancy) -> np.ndarray:
    """占用人数 → 归一化尺寸因子（sqrt 缩放：面积 ∝ 人数，人越多占屏幕越大）。"""
    occ = np.maximum(np.asarray(occupancy, dtype=np.float32), 1.0)
    return np.sqrt(occ / VENUE_REFERENCE_OCCUPANCY)


def _build_venue_blob(sim) -> tuple[list[dict], bytes]:
    """构造场所层：返回 (层清单, 二进制) ；二进制 = 每层连续 Float32 (x, y, size_norm)。"""
    occ_map = getattr(sim, "venue_occupancy", {}) or {}
    layers: list[dict] = []
    chunks: list[np.ndarray] = []
    cursor = 0
    for kind, attr, occ_key, label, color, shape in VENUE_STYLES:
        arr = getattr(sim, attr, None)
        if arr is None:
            continue
        coords = np.asarray(arr, dtype=np.float32)
        if coords.size == 0:
            continue
        count = int(coords.shape[0])
        occ = occ_map.get(occ_key)
        if occ is None:
            sizes = np.ones(count, dtype=np.float32)
        else:
            sizes = _venue_size_norm(occ).astype(np.float32)
            if sizes.shape[0] != count:  # 防御：占用数组与坐标长度不一致时退化为统一点
                sizes = np.ones(count, dtype=np.float32)
        block = np.empty((count, 3), dtype=np.float32)
        block[:, :2] = coords[:, :2]
        block[:, 2] = sizes
        chunks.append(block)
        layers.append({
            "kind": kind, "label": label, "color": color, "shape": shape,
            "count": count, "start": cursor,
        })
        cursor += count
    if chunks:
        blob = np.concatenate(chunks, axis=0).tobytes()
    else:
        blob = b""
    return layers, blob


def meta(sim) -> dict:
    """连接建立时的静态描述（只发一次，不含逐帧数据）。"""
    layers, _ = sim._venue_blob
    return {
        "type": "meta",
        "profile": config.ACTIVE_PROFILE_NAME,
        "profile_summary": disease_profiles.format_summary(config.PROFILE_INFO),
        "population": int(sim.N),
        "world": [float(sim.W), float(sim.H)],
        "frames": int(CONFIG["FRAMES"]),
        "frames_per_day": int(CONFIG["DAY_FRAMES"]),
        "min_per_frame": float(sim.MIN_PER_FRAME),
        "clock_start_min": float(CONFIG["CLOCK_START_MIN"]),
        "beds": int(getattr(sim, "beds_total", 0)),
        "medical_threshold": int(CONFIG["MEDICAL_THRESHOLD"]),
        "seed": int(config.SEED),
        "sample": sim.render_idx is not None,
        "render_points": int(sim._RENDER_MAX) if sim.render_idx is not None else int(sim.N),
        "colors": list(COLOR_MAP),
        "status_labels": list(STATUS_LABELS),
        "natural_history": bool(CONFIG["NATURAL_HISTORY_ENABLED"]),
        "interventions": bool(getattr(sim, "IV_ENABLED", False)),
        "lockdown": bool(getattr(sim, "lockdown_active", False)),
        "venue_layers": layers,
        "venue_size_px": [VENUE_SIZE_MIN_PX, VENUE_SIZE_MAX_PX],
    }


def points_blob(sim) -> bytes:
    """当前帧的抽样点云：Float32 (n,2) 坐标 紧跟 Uint8 (n) 状态码。"""
    if sim.render_idx is None:
        pos = sim.pos
        status = sim.status
    else:
        pos = sim.pos[sim.render_idx]
        status = sim.status[sim.render_idx]
    xy = pos.detach().to(torch.float32).contiguous().cpu().numpy()
    st = status.detach().to(torch.uint8).contiguous().cpu().numpy()
    return xy.tobytes() + st.tobytes()


def pack_frame(sim, frame: int) -> tuple[bytes, bytes]:
    """打包一帧：返回 (JSON 头字节, 点云字节)。``sim.last_stats`` 由 simulation F 段写入。"""
    stats = sim.last_stats if sim.last_stats is not None else {}
    n = int(sim.render_idx.numel()) if sim.render_idx is not None else int(sim.N)
    header = {"type": "frame", "t": int(frame), "n": n}
    header.update(stats)
    blob = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return blob, points_blob(sim)
