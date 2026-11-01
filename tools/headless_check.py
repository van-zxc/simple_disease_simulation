#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""无头契约门禁：不开 GUI，用 Agg 后端加载入口模块逐帧驱动，断言 38 项用户侧契约（a/b/c/d/e/f/g/h/i/j/k/m 组）。

输入：仓库根的 simple_disease_simulation.py（--module 可换入口）；输出：控制台"断言明细 + 结论: PASS"、
--json 落盘；--bench N 只计时不写快照，--equiv 另跑稠密/稀疏传播等价测试。
退出码：0 = 启用断言全过（含 SKIP）；1 = 任一 FAIL；2 = 参数非法 / 加载或运行异常。
不负责：性能归因（bench.py / diag.py）与历史逐位金标（golden_check.py 自第二轮优化起只作诊断）。
口径差异：update 计时不含全量重绘——draw 只对 update 返回的 artist 做 draw_artist+blit；
  bench.py 每帧 canvas.draw() 全量重绘、update 后还带 torch.cuda.synchronize()，两者绝对值不可互换。
已知偏差：手工 blit 不刷新轴外标题像素（断言读 get_text() 不受影响）；缺张量属性时会静默 SKIP（见下）。
"""

# ==================== 契约与协议（改断言 / 加载顺序 / 关节代码前必读） ====================
# 1) 38 项记录是用户侧硬门禁，按组：a1-a3 标题与计数一致、b 标题 Reinf 与窗口比率、c1-c4 曲线、
#    d1-d2 轴限、e/f/m1 状态迁移、g1/g2 按钮与封控标题、h1/h2 artist 契约、i CUDA 峰值、
#    j1/j2 状态量与坐标、k1-k6 日常节律与出行、m1-m10 v2 自然史 + v3 社会响应；
#    其中 g1i/i1/i2 仅 INFO 观察、不断言。LABELS 文本、"结论: PASS/FAIL"、JSON 键名、CLI 参数
#    都属输出契约，只改注释时一律不动。
# 2) 加载协议：matplotlib.use("Agg")（import pyplot 之前）→ patch_matplotlib() 把
#    FuncAnimation 换成 _DummyFuncAnimation、plt.show 置空 → 然后 load_simulation()。
#    入口可能在 import 期就创建动画并阻塞在 show()，顺序颠倒会直接卡死。
#    入口用 spec_from_file_location("sim_under_test", ...) 加载：模块 __package__ 为空、
#    __name__ 为 "sim_under_test"；入口的 CLI 守卫必须看 sys.argv[0] 而不是 __package__。
# 3) 探针协议：canvas.draw() → mod.update(0) 探针（取 artist 集合/标题/按钮状态，计时单列不计入统计）
#    → 主循环 for i in range(1, frames)。合计 frames 次 update，末帧 index = frames-1。
#    任何"重放/复刻"都要按这个顺序：draw 一次 → update(0) → 逐帧 restore_region / update / draw_artist+blit。
# 4) 静默 SKIP 风险：多数检查前有 isinstance(属性, torch.Tensor) 守卫；模型缺 pos/quarantined/trip_*/
#    history 等属性时，该组会 SKIP 或"run 过但没查"，退出码仍是 0。改契约名后别只看 PASS，
#    要核对"共 38 项"和每项状态（SKIP 表示没真正执行；frames=1 时末帧观察项 i1/i2 也不产生）。
# 5) 不可动的契约：38 条记录与标签文本、退出码语义、JSON 键名；模型侧模块级名字与形状
#    （status/pos/history/update/fig/CONFIG/...）以及 update() 返回 15 个 artist
#    （2 散点+标题 + 8 lines_pop + 5 统计曲线，见 N_ARTISTS）。
# 已知偏差：旧版入口标题在轴外（y=1.05）时手工 blit 画布上没有标题像素，但断言读 title_obj.get_text()，
#    不受影响；--render 的 PNG 走 fig.savefig 全量重绘，与手工 blit 路径的像素口径不同。
# ====================

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as _mpl_animation  # noqa: E402
import matplotlib.colors as _mpl_colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# ==================== 常量 ====================
EPS = 1e-9                # 浮点比较容差：状态/坐标是 float32，不能按 == 判等
GREEN_RGBA = tuple(_mpl_colors.to_rgba("#32CD32"))   # 契约色：toggle 后按钮必须变成该绿色
ORANGE_RGBA = tuple(_mpl_colors.to_rgba("#FF4500"))  # 契约色：默认/解锁按钮橙色（当前只定义，未参与断言）
BTN_TOL = 0.02            # 按钮颜色容差：Agg 下 facecolor 往返量化误差约 1e-3，0.02 留足余量
REINF_TOL = 0.15          # 标题 Reinf 与窗口比率的允差（百分点），覆盖标题两位小数+窗口取整
CUDA_PEAK_LIMIT_MB = 300.0  # 峰值显存基线；N_AGENTS 超约 25 万后按 1.2 MB/千 agent 放宽（见 run_sim）
N_ARTISTS = 15            # h1 契约：2 (scat,title) + 8 (lines_pop) + 5 (曲线)
SNAPSHOT_EVERY = 100      # 快照与"贵检查"的采样间隔；bench 模式也靠它给 pos 全量扫描降采样
REINF_WINDOW = 50         # 与模型标题再感染率窗口同值；改这里等于换口径，必须同步模型
ROLLING_POINTS = 500      # 模型曲线滚动窗口点数（i1 观察项据此判断是否已滚动）

# 标题字段解析：数字容忍千分位/正负号/科学计数法；字段名容忍缩写，分隔符 = 与 : 均可，大小写不敏感。
# 负向后顾 (?<![A-Za-z]) 防止把 "Reinfected" 里的 "Inf" 误认成 Inf 字段。
_NUM = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
TITLE_PATTERNS = {
    "inf": re.compile(r"(?<![A-Za-z])Inf(?:ected)?\s*[:=]\s*(" + _NUM + r")", re.I),
    "symp": re.compile(r"(?<![A-Za-z])Symp(?:tomatic)?\s*[:=]\s*(" + _NUM + r")", re.I),
    "med": re.compile(r"(?<![A-Za-z])Med(?:ical)?\s*[:=]\s*([A-Za-z_]+)", re.I),
    "immunity": re.compile(r"(?<![A-Za-z])Immunity\s*[:=]\s*(" + _NUM + r")", re.I),
    "reinf": re.compile(r"(?<![A-Za-z])Reinf(?:ection)?\s*[:=]\s*(" + _NUM + r")", re.I),
    "lock": re.compile(r"(?<![A-Za-z])Lock(?:down)?\s*[:=]\s*([A-Za-z_]+)", re.I),
    "t": re.compile(r"(?<![A-Za-z])t\s*[:=]\s*([-+]?\d+)"),
    "hosp": re.compile(r"(?<![A-Za-z])Hosp(?:italized)?\s*[:=]\s*(" + _NUM + r")", re.I),
    "unmet": re.compile(r"(?<![A-Za-z])Unmet\s*[:=]\s*(" + _NUM + r")", re.I),
}
# Lock 字段的文本取值（标题里是词而非布尔）；识别不了时 parse_bool_word 返回 None，g2 记 FAIL。
_TRUE_WORDS = {"ON", "TRUE", "YES", "1", "ACTIVE", "LOCKDOWN", "LOCKED", "COLLAPSE"}
_FALSE_WORDS = {"OFF", "FALSE", "NO", "0", "INACTIVE", "RESUME", "UNLOCKED"}

# 断言键 -> 人类可读标签。分组：a 计数一致 / b 窗口再感染率 / c 曲线 / d 轴限 / e·f·m1 状态迁移 /
# g 按钮与封控 / h artist 契约 / i 资源观察 / j 状态量合法性 / k 行为与出行 / m v2 自然史与 v3 社会响应。
# 文本属于输出契约（用户按标签逐条核对 38 项），只改注释时不要动这里。
LABELS = {
    "a1": "标题 Inf/Symp 与 status 计数一致",
    "a2": "counts 总和 == N_AGENTS",
    "a3": "history[s][-1] == counts[s]",
    "b": "标题 Reinf 与最近 50 帧窗口比率一致",
    "c1": "line_reinf 范围/有限/末值与标题一致",
    "c2": "line_gen 数据在 [0, 100]",
    "c3": "line_mu 数据 <= 1.5",
    "c4": "所有曲线数据有限",
    "d1": "轴限全程恒定（首帧 vs 末帧）",
    "d2": "轴限等于预期值",
    "e": "无同帧 潜伏 -> 死亡 迁移",
    "f": "存在 免疫/康复 -> 健康 迁移（frames>=2000）",
    "g1": "toggle() 后按钮变为 #32CD32",
    "g2": "标题 Lock 字段与 lockdown_active 一致",
    "g1i": "（观察）再 toggle 回来的按钮颜色",
    "h1": "update() 返回 15 个 artist",
    "h2": "每个 artist 都带 .axes",
    "i": "CUDA 峰值显存 < 300 MB",
    "j1": "status 取值均在 0..7",
    "j2": "pos 无 NaN/Inf",
    "m1": "无 E/A 同帧直接死亡（2/6 -> 4）",
    "m2": "住院/无人照护计数与床位机制一致",
    "m3": "隔离者被限制在家（非在途）",
    "m4": "死亡年龄分布右偏（死亡>=20 人时）",
    "m5": "v2 自然史出现无症状状态（frames>=2000）",
    "m6": "旧路径/toy 不出现 A/H 状态（6/7）",
    "m7": "检测量不超过当日能力，累计检测单调不减",
    "m8": "密接隔离者被限制在家（非在途）",
    "m9": "行为乘子范围合法，累计报告不超过累计发病",
    "m10": "社会响应关闭时不产生检测/隔离/报告",
    "k1": "日常节律：03:00 居家>=95%；工作日 12:00 在岗/在校>=80%",
    "k2": "出行频率：成人 1.5-5 次/日，学生 0.5-4 次/日",
    "k3": "单次出行 <= TRIP_MAX_FRAMES，且无卡住/死亡在途",
    "k4": "03:00 多成员家庭成员与家庭锚点距离 <= 3.0 的比例 >= 95%",
    "k5": "place 取值在 0..3（家/工作/学校/休闲）",
    "k6": "周末 12:00 在岗比例 < 5%",
    "i1": "（观察）曲线 artist 为最近 500 点滚动窗口",
    "i2": "（观察）标题 t= 字段",
}


# ==================== matplotlib 打桩 ====================
class _DummyFuncAnimation:
    """替代 FuncAnimation：只记录构造参数，不注册定时器、不触发绘制。

    instances 保留全部实例供旧版代码/测试查询；_draw_was_started() 恒 False，
    让"动画从未启动"这一假设对入口模块成立，import 期不会产生副作用。
    """

    instances: list = []

    def __init__(self, fig=None, func=None, frames=None, init_func=None, fargs=None,
                 save_count=None, cache_frame_data=True, **kwargs):
        self.fig = fig
        self.func = func
        self.frames = frames
        self.init_func = init_func
        self.fargs = fargs
        self.save_count = save_count
        self.kwargs = dict(kwargs)
        _DummyFuncAnimation.instances.append(self)

    def _draw_was_started(self):  # 供旧代码/测试调用
        return False

    def save(self, *args, **kwargs):
        return None


def patch_matplotlib() -> None:
    """必须在 load_simulation 之前调用：让入口 import 时不会真的创建动画、阻塞在 show()。

    打桩是进程级全局的，只影响本进程后续加载的入口模块。
    """
    _mpl_animation.FuncAnimation = _DummyFuncAnimation
    plt.show = lambda *a, **k: None  # type: ignore[assignment]


# ==================== 小工具 ====================
def _jsonable(obj):
    """把 torch/numpy 标量与容器转成 json.dump 能接受的原生类型；未知对象退回 repr。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return repr(obj)


def _fmt_ms(x):
    return "n/a" if x is None else f"{x * 1000.0:.2f} ms"


def _fmt_rgba(c):
    return "(" + ", ".join(f"{float(v):.3f}" for v in c) + ")"


def _close(a, b, tol=EPS):
    return abs(float(a) - float(b)) <= tol


def _stats(vals):
    """汇总计时样本为 n/mean/median/max/min/total（秒）；空样本返回全 None / total=0。"""
    if not vals:
        return {"n": 0, "mean": None, "median": None, "max": None, "min": None, "total": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "total": float(arr.sum()),
    }


def _to_number(text):
    if text is None:
        return None
    return float(str(text).replace(",", ""))


def parse_title(text):
    """从标题文本里容错解析字段；缺失的键不出现在返回值里。"""
    out = {}
    if not text:
        return out
    for key, pat in TITLE_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        raw = m.group(1)
        if key in ("med", "lock"):
            out[key] = raw
        elif key == "t":
            out[key] = int(raw)
        else:
            out[key] = _to_number(raw)
    return out


def parse_bool_word(word):
    """把标题里的 Lock 词映射成 bool；无法识别返回 None（调用方当作解析失败处理）。"""
    if word is None:
        return None
    w = str(word).strip().upper()
    if w in _TRUE_WORDS:
        return True
    if w in _FALSE_WORDS:
        return False
    return None


def status_to_numpy(mod):
    """把模型 status（tensor 或数组）转成 int64 numpy；属性缺失返回 None。

    a1/a2/a3/j1/m1-e 等逐帧计数都基于它，所以每次取的是"本帧刚 update 完"的状态快照。
    """
    st = getattr(mod, "status", None)
    if st is None:
        return None
    if isinstance(st, torch.Tensor):
        return st.detach().to("cpu").numpy().astype(np.int64, copy=False)
    return np.asarray(st, dtype=np.int64)


def counts8(st):
    """返回长度 8 的计数；越界值单独统计（由 j1 断言负责）。"""
    out = np.zeros(8, dtype=np.int64)
    if st is None or st.size == 0:
        return out
    valid = (st >= 0) & (st < 8)
    if valid.any():
        out = np.bincount(st[valid], minlength=8)[:8].astype(np.int64)
    return out


def out_of_range_count(st):
    """统计 status 不在 0..7 的个体数，j1 用；状态缺失返回 0。"""
    if st is None or st.size == 0:
        return 0
    return int(np.count_nonzero((st < 0) | (st > 7)))


def lims_of(ax):
    """取轴限的纯 Python 副本 ((x0,x1),(y0,y1))，供 d1/d2 跨帧比较与 JSON 落盘。"""
    x = ax.get_xlim()
    y = ax.get_ylim()
    return ((float(x[0]), float(x[1])), (float(y[0]), float(y[1])))


def line_ydata(line):
    """读曲线 y 数据为 float 数组；artist 尚未就绪时返回 None（调用方记 c4 违例）。"""
    try:
        return np.asarray(line.get_ydata(), dtype=float)
    except Exception:
        return None


def title_text_of(mod):
    """读 title_obj 文本；旧版入口没有该属性时返回 None，走 SKIP/FAIL 分支而不是崩溃。"""
    obj = getattr(mod, "title_obj", None)
    if obj is None or not hasattr(obj, "get_text"):
        return None
    try:
        return str(obj.get_text())
    except Exception:
        return None


def parse_render_frames(spec):
    """解析 --render 的 "150,500"（逗号/空白分隔）：去重排序后返回帧号列表。"""
    if not spec:
        return []
    out = []
    for part in re.split(r"[,\s]+", str(spec).strip()):
        if part:
            out.append(int(part))
    return sorted(set(out))


# ==================== 断言记录 ====================
class Checks:
    """断言记录容器：add() 追加一条（key 决定展示标签），failed()/exit_code() 供汇总与退出码。

    enforce=False（--no-assert）时只记录不拦截，exit_code() 恒 0——旧版兼容冒烟靠它。
    """

    def __init__(self, enforce=True):
        self.enforce = bool(enforce)
        self.items = []

    def add(self, key, status, detail=""):
        self.items.append({"key": key, "label": LABELS.get(key, key),
                           "status": status, "detail": detail})
        return status

    def info(self, key, detail=""):
        return self.add(key, "INFO", detail)

    def failed(self):
        return [i for i in self.items if i["status"] == "FAIL"]

    def exit_code(self):
        if not self.enforce:
            return 0
        return 1 if self.failed() else 0


class Tracker:
    """逐帧累积型断言：run() 记检查次数，hit() 记违例（含首个违例帧与细节）。

    result() 在 runs==0 时返回 SKIP"未执行"——这是"张量属性缺失 → 静默跳过"的出口，
    读明细时要把 SKIP 与 PASS 区分开。
    """

    def __init__(self):
        self.hits = {}
        self.runs = {}

    def run(self, key, n=1):
        self.runs[key] = self.runs.get(key, 0) + int(n)

    def hit(self, key, frame, detail):
        d = self.hits.setdefault(key, {"count": 0, "first": None, "detail": ""})
        d["count"] += 1
        if d["first"] is None:
            d["first"] = frame
            d["detail"] = detail

    def result(self, key):
        runs = self.runs.get(key, 0)
        if runs == 0:
            return "SKIP", "未执行"
        d = self.hits.get(key)
        if not d:
            return "PASS", f"检查 {runs} 次，无违例"
        return "FAIL", f"{d['count']}/{runs} 次违例；首个 frame={d['first']}: {d['detail']}"


# ==================== 模型加载 ====================
def load_simulation(path: Path, quiet=True):
    """用 spec_from_file_location 从文件路径加载入口模块，返回 (module, import 期 stdout)。

    模块名固定 "sim_under_test"、__package__ 为空（入口的 CLI 守卫必须按 sys.argv[0] 判断）。
    加载期间临时关闭字节码写入，避免在仓库里留 __pycache__ 副作用；import 期 sys.exit 转成
    RuntimeError（main 按加载失败返回退出码 2）。
    """
    spec = importlib.util.spec_from_file_location("sim_under_test", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法为 {path} 构造 import spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_under_test"] = mod
    buf = io.StringIO()
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if quiet:
            with contextlib.redirect_stdout(buf):
                spec.loader.exec_module(mod)
        else:
            spec.loader.exec_module(mod)
    except SystemExit as exc:
        captured = buf.getvalue().strip() or "(模型未输出诊断信息)"
        raise RuntimeError(
            f"模型导入时调用 sys.exit({exc.code})；模型输出：\n{captured}") from None
    finally:
        sys.dont_write_bytecode = old_flag
    return mod, buf.getvalue()


def resolve_module_path(args):
    """解析被测入口：--module 优先，否则仓库根 simple_disease_simulation.py。"""
    if args.module:
        return Path(args.module).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "simple_disease_simulation.py")


# ==================== 算法等价测试 ====================
def run_equiv(trials=20, n_inf=30, n_tar=40, radius=1.8, seed=0):
    """比较稠密 (any+argmax) 与稀疏 (nonzero+scatter_reduce_ amin) 的等价性。

    每轮按 seed*1000+t 重播种（同参数可在 CPU/CUDA 上复现）；返回 0/1，--equiv 直接拿它当退出码。
    参数是固定的密集小场景（radius 取模型默认感染半径 1.8），只验证"每个目标选中哪个源"一致，
    不对传播概率做统计检验。
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    print("=" * 68)
    print("算法等价测试：稠密 any/argmax  vs  稀疏 nonzero/scatter_reduce_(amin)")
    print(f"n_inf={n_inf}, n_tar={n_tar}, INFECT_RADIUS={radius}, 随机种子 {trials} 组, 设备 {devices}")
    print("=" * 68)
    ok_all = True
    total_success = 0
    for dev in devices:
        dev_ok = True
        for t in range(trials):
            torch.manual_seed(seed * 1000 + t)
            dists = torch.rand(n_inf, n_tar) * 3.0
            contacts = dists < radius
            prob = torch.rand(n_inf, n_tar) * 0.25
            success = contacts & (torch.rand(n_inf, n_tar) < prob)
            success = success.to(dev)
            total_success += int(success.sum().item())

            # 稠密实现：any 判位、argmax 取每目标的最小源索引（0/1 掩码的 argmax 即首个 1）
            was_inf_d = success.any(dim=0)
            src_map = success.float().argmax(dim=0)
            src_d = src_map[was_inf_d]

            # 稀疏实现：只在命中对上 scatter_reduce_(amin)，语义与"每目标最小源索引"一致
            nz = success.nonzero(as_tuple=False)
            if nz.numel():
                rows = nz[:, 0]
                cols = nz[:, 1]
                src = torch.full((n_tar,), n_inf, dtype=torch.long, device=dev)
                src.scatter_reduce_(0, cols, rows, reduce="amin")
                was_inf_s = src < n_inf
                src_s = src[was_inf_s]
            else:
                was_inf_s = torch.zeros(n_tar, dtype=torch.bool, device=dev)
                src_s = torch.empty(0, dtype=torch.long, device=dev)

            same_flag = bool(torch.equal(was_inf_d, was_inf_s))
            same_src = bool(torch.equal(src_d.to(dev), src_s))
            if not (same_flag and same_src):
                dev_ok = False
                ok_all = False
                print(f"  [FAIL] dev={dev} trial={t}: was_inf 相同={same_flag}, "
                      f"sources 相同={same_src}, |was_inf|={int(was_inf_d.sum())}")
        print(f"  [{'PASS' if dev_ok else 'FAIL'}] 设备 {dev}: {trials} 组随机用例"
              f"{'全部一致' if dev_ok else '存在不一致'}")
    print(f"累计成功感染事件（非零项）: {total_success}")
    print(f"等价测试结论: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


# ==================== 派生断言（b/c）====================
def derived_checks(mod, frame, parsed, tracker, ctx):
    """窗口再感染率与曲线范围检查；可在快照帧/末帧调用。

    b：标题 Reinf 必须等于最近 REINF_WINDOW 帧的窗口比率（失败=标题口径与模型窗口分叉或窗口不同步）。
    c1-c4：line_reinf 范围/末值与标题一致，line_gen∈[0,100]，line_mu<=1.5，所有曲线数据有限。
    strain_stats 缺键时置 ctx["strain_missing"]，由收尾阶段把 b 记为 SKIP/FAIL。
    """
    ss = getattr(mod, "strain_stats", None)
    if isinstance(ss, dict) and ("new_infections" in ss) and ("reinfections" in ss):
        new_l = list(ss.get("new_infections") or [])
        reinf_l = list(ss.get("reinfections") or [])
        tracker.run("b")
        if new_l and reinf_l:
            win = min(REINF_WINDOW, len(new_l), len(reinf_l))
            denom = float(sum(new_l[-win:]))
            num = float(sum(reinf_l[-win:]))
            ratio = 100.0 * num / denom if denom > 0 else 0.0
            ctx["last_window"] = {"frame": frame, "window": win, "new": denom,
                                  "reinf": num, "ratio": ratio}
            val = parsed.get("reinf")
            if val is None:
                tracker.hit("b", frame, "标题缺少 Reinf 字段")
            elif abs(float(val) - ratio) >= REINF_TOL:
                tracker.hit("b", frame,
                            f"标题 Reinf={float(val):.3f} vs 窗口比率={ratio:.3f}% "
                            f"(窗口 {win} 帧, new={denom:g}, reinf={num:g})")
    else:
        ctx["strain_missing"] = True

    curves = ctx["curves"]
    for name, line in curves.items():
        tracker.run("c4")
        arr = line_ydata(line)
        if arr is None:
            tracker.hit("c4", frame, f"{name}.get_ydata() 读取失败")
        elif arr.size and not bool(np.all(np.isfinite(arr))):
            tracker.hit("c4", frame, f"{name} 含 NaN/Inf")

    lr = curves.get("line_reinf")
    if lr is not None:
        tracker.run("c1")
        arr = line_ydata(lr)
        if arr is None or arr.size == 0:
            tracker.hit("c1", frame, "line_reinf 无数据")
        else:
            if arr.min() < -EPS or arr.max() > 100.0 + EPS:
                tracker.hit("c1", frame, f"line_reinf 越界 [{arr.min():.4f}, {arr.max():.4f}]")
            val = parsed.get("reinf")
            if val is not None and abs(float(arr[-1]) - float(val)) >= REINF_TOL:
                tracker.hit("c1", frame,
                            f"line_reinf 末值 {float(arr[-1]):.4f} vs 标题 Reinf {float(val):.3f}")

    lg = curves.get("line_gen")
    if lg is not None:
        tracker.run("c2")
        arr = line_ydata(lg)
        if arr is None:
            tracker.hit("c2", frame, "line_gen.get_ydata() 读取失败")
        elif arr.size and (arr.min() < -EPS or arr.max() > 100.0 + EPS):
            tracker.hit("c2", frame, f"line_gen 越界 [{arr.min():.4f}, {arr.max():.4f}]")

    lm = curves.get("line_mu")
    if lm is not None:
        tracker.run("c3")
        arr = line_ydata(lm)
        if arr is None:
            tracker.hit("c3", frame, "line_mu.get_ydata() 读取失败")
        elif arr.size and arr.max() > 1.5 + EPS:
            tracker.hit("c3", frame, f"line_mu 最大值 {arr.max():.4f} > 1.5")


# ==================== 活动/轨迹抽查（k1/k4/k5/k6）====================
def _activity_checks(mod, frame, clock, day_index, tracker):
    """在行为时钟 03:00/12:00 抽查日常节律：k1 居家或在岗在校、k4 家庭锚点、k5 地点取值、k6 周末在岗。

    clock 是"距 0 点分钟数"（180=03:00、720=12:00），由 CLOCK_START_MIN + i*1440/DAY_FRAMES 推出，
    只在命中整点的帧执行；缺 status/place/traveling 时返回 False，但调用方忽略返回值（见 run_sim），
    所以张量契约缺失时这些检查表现为"没有执行"，而不是失败。
    阈值 95%/80%/5%/3.0 是行为日程的设计值，固化成门禁阈值；改 activity.py 的日程表时要同步复核。
    """
    status_t = getattr(mod, "status", None)
    place_t = getattr(mod, "place", None)
    traveling_t = getattr(mod, "traveling", None)
    if not all(isinstance(t, torch.Tensor) for t in (status_t, place_t, traveling_t)):
        return False
    status = status_t.detach().cpu().numpy()
    place = place_t.detach().cpu().numpy()
    traveling = traveling_t.detach().cpu().numpy().astype(bool)
    alive = status != 4
    alive_n = int(alive.sum())
    if alive_n == 0:
        return True
    # 两个抽查时刻：03:00 深夜（居家/家庭锚点）、12:00 中午（在岗/在校/周末在岗）
    is_night = abs(clock - 180.0) < 1e-6
    is_noon = abs(clock - 720.0) < 1e-6
    if not (is_night or is_noon):
        return True

    # k5: place 取值合法（0=家, 1=工作, 2=学校, 3=休闲；越界说明行程状态机写坏）
    tracker.run("k5")
    if place.min() < 0 or place.max() > 3:
        tracker.hit("k5", frame, f"place 取值范围 [{place.min()}, {place.max()}]，应为 0..3")

    if is_night:
        # k1: 深夜居家比例
        tracker.run("k1")
        at_home = alive & (~traveling) & (place == 0)
        frac = float(at_home.sum()) / float(alive_n)
        if frac < 0.95:
            tracker.hit("k1", frame, f"03:00 居家比例={frac:.3f} < 0.95")
        # k4: 多成员家庭深夜与家庭锚点的距离
        anchor = getattr(mod, "home_anchor", None)
        hh = getattr(mod, "household_id", None)
        pos_t = getattr(mod, "pos", None)
        if all(isinstance(t, torch.Tensor) for t in (anchor, hh, pos_t)):
            anc = anchor.detach().cpu().numpy()
            hh_np = hh.detach().cpu().numpy()
            pos = pos_t.detach().cpu().numpy()
            sizes = np.bincount(hh_np, minlength=int(hh_np.max()) + 1)
            multi = sizes[hh_np] >= 2
            masked = alive & multi
            tracker.run("k4")
            if int(masked.sum()) > 0:
                dist = np.linalg.norm(pos - anc, axis=1)
                frac = float((dist[masked] <= 3.0).sum()) / float(masked.sum())
                if frac < 0.95:
                    tracker.hit("k4", frame, f"03:00 家庭成员在锚点 3.0 内比例={frac:.3f} < 0.95")

    if is_noon:
        employed = getattr(mod, "employed", None)
        student = getattr(mod, "student", None)
        weekday = (day_index % 7) not in (5, 6)
        tracker.run("k1")
        if weekday and isinstance(employed, torch.Tensor):
            emp = employed.detach().cpu().numpy().astype(bool) & alive
            if int(emp.sum()) > 0:
                frac = float((emp & (~traveling) & (place == 1)).sum()) / float(emp.sum())
                if frac < 0.80:
                    tracker.hit("k1", frame, f"12:00 就业者在岗比例={frac:.3f} < 0.80")
        if weekday and isinstance(student, torch.Tensor):
            stu = student.detach().cpu().numpy().astype(bool) & alive
            if int(stu.sum()) > 0:
                frac = float((stu & (~traveling) & (place == 2)).sum()) / float(stu.sum())
                if frac < 0.80:
                    tracker.hit("k1", frame, f"12:00 在册学生在校比例={frac:.3f} < 0.80")
        # k6: 周末 12:00 在岗比例
        if (day_index % 7) in (5, 6):
            tracker.run("k6")
            if isinstance(employed, torch.Tensor):
                emp = employed.detach().cpu().numpy().astype(bool) & alive
                if int(emp.sum()) > 0:
                    frac = float((emp & (~traveling) & (place == 1)).sum()) / float(emp.sum())
                    if frac >= 0.05:
                        tracker.hit("k6", frame, f"周末 12:00 在岗比例={frac:.3f} >= 0.05")
    return True


def _trip_checks(mod, frame, tracker):
    """k3 门禁：单次出行时长不超过 CONFIG["TRIP_MAX_FRAMES"]，无 trip_left<=0 的卡住者、无死者仍在途。

    缺 trip_total/trip_left/traveling/status 时直接返回（k3 未 run → 明细记为 SKIP 而不是 PASS）。
    """
    trip_total = getattr(mod, "trip_total", None)
    trip_left = getattr(mod, "trip_left", None)
    traveling = getattr(mod, "traveling", None)
    status_t = getattr(mod, "status", None)
    if not all(isinstance(t, torch.Tensor) for t in (trip_total, trip_left, traveling, status_t)):
        return
    cfg = getattr(mod, "CONFIG", None) or {}
    limit = int(cfg.get("TRIP_MAX_FRAMES", 12))
    tt = trip_total.detach().cpu().numpy()
    tl = trip_left.detach().cpu().numpy()
    tr = traveling.detach().cpu().numpy().astype(bool)
    dead = status_t.detach().cpu().numpy() == 4
    tracker.run("k3")
    if tt.size and int(tt.max()) > limit:
        tracker.hit("k3", frame, f"trip_total 最大值={int(tt.max())} > {limit}")
    stuck = int((tr & (tl <= 0)).sum())
    if stuck:
        tracker.hit("k3", frame, f"{stuck} 个在途者 trip_left<=0（卡住）")
    dead_travel = int((tr & dead).sum())
    if dead_travel:
        tracker.hit("k3", frame, f"{dead_travel} 个死者仍标记在途")


# ==================== 主流程 ====================
def run_sim(args, frames, checks, out):
    """跑完整无头会话：加载→打桩→首帧 draw→探针 update(0)→主循环 update/blit→收尾断言。

    返回 None；断言结果写进 checks（main 用 checks.exit_code() 当进程退出码），采集数据写进 out
    （--json 的结构契约）。frames 是总 update 次数：探针占第 0 帧、主循环 i=1..frames-1，
    所以任何"重放"都必须保持 draw → update(0) → for i in range(1, frames) 这个顺序。
    """
    wall_start = time.perf_counter()
    mod_path = resolve_module_path(args)
    out["module"] = str(mod_path)
    if not mod_path.is_file():
        raise FileNotFoundError(f"找不到模型文件: {mod_path}")

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed) % (2 ** 32))

    patch_matplotlib()
    t_load = time.perf_counter()
    mod, import_stdout = load_simulation(mod_path, quiet=not args.verbose)
    out["load_s"] = time.perf_counter() - t_load
    out["import_stdout_tail"] = import_stdout[-4000:]

    cfg = getattr(mod, "CONFIG", None)
    cfg = cfg if isinstance(cfg, dict) else {}
    status_t = getattr(mod, "status", None)
    n_agents = cfg.get("N_AGENTS")
    if n_agents is None and status_t is not None:
        try:
            n_agents = int(status_t.shape[0])
        except Exception:
            n_agents = None
    n_agents = int(n_agents) if n_agents is not None else None
    width = cfg.get("WIDTH")
    height = cfg.get("HEIGHT")

    out["config"] = {k: _jsonable(cfg.get(k)) for k in
                     ("N_AGENTS", "MEDICAL_THRESHOLD", "INFECT_RADIUS", "SEED",
                      "IMMUNITY_WANING_THRESHOLD")}
    out["config_full"] = _jsonable(cfg)
    out["n_agents"] = n_agents
    out["device"] = str(getattr(mod, "device", "?"))
    out["seed"] = args.seed
    out["mode"] = "bench" if args.bench is not None else "full"
    out["profile"] = getattr(mod, "ACTIVE_PROFILE_NAME", None)

    fig = getattr(mod, "fig", None)
    if fig is None or not hasattr(fig, "canvas"):
        raise RuntimeError("模型缺少 fig / fig.canvas，无法无头驱动")
    canvas = fig.canvas

    # ---------- g1: 按钮切换 ----------
    toggle = getattr(mod, "toggle", None)
    btn = getattr(mod, "btn", None)
    if callable(toggle) and btn is not None and hasattr(btn, "ax"):
        try:
            toggle(None)  # 传 None = 模拟点击按钮，不依赖 GUI 事件循环
            fc = tuple(float(v) for v in btn.ax.get_facecolor())
            ok = len(fc) >= 4 and all(abs(fc[i] - GREEN_RGBA[i]) <= BTN_TOL for i in range(4))
            checks.add("g1", "PASS" if ok else "FAIL",
                       f"toggle() 后 facecolor={_fmt_rgba(fc)}，期望 #32CD32≈{_fmt_rgba(GREEN_RGBA)}（容差 {BTN_TOL}）")
            toggle(None)  # 切回原状态：g1i 只观察颜色是否随状态可逆变化
            fc2 = tuple(float(v) for v in btn.ax.get_facecolor())
            checks.info("g1i", f"再 toggle 回来 facecolor={_fmt_rgba(fc2)}（仅观察，不作断言）")
        except Exception as exc:
            checks.add("g1", "FAIL", f"toggle() 调用异常: {exc!r}")
    else:
        checks.add("g1", "SKIP" if not checks.enforce else "FAIL", "模型缺少 toggle()/btn 接口")

    # ---------- 首帧 draw ----------
    # 先全量画一次：建立 Agg 画布缓冲（后续循环用 restore_region + 逐 artist blit 复用），
    # 这段耗时单独记 first_draw_s，对应 GUI 首帧的真实开销。
    t0 = time.perf_counter()
    canvas.draw()
    first_draw = time.perf_counter() - t0

    # ---------- 探针 update(0)：推进到 frame 0，仅用于取 artist 集合 + g2 ----------
    # 主循环从 i=1 开始，所以整场恰好 frames 次 update（探针 1 次 + 主循环 frames-1 次）。
    t0 = time.perf_counter()
    probe_artists = mod.update(0)
    probe_update = time.perf_counter() - t0
    if not isinstance(probe_artists, (list, tuple)):
        probe_artists = list(probe_artists or [])

    # ---------- g2: 标题 Lock 与 lockdown_active ----------
    probe_title = title_text_of(mod)
    probe_parsed = parse_title(probe_title)
    if probe_parsed.get("lock") is None:
        checks.add("g2", "SKIP" if not checks.enforce else "FAIL",
                   f"标题缺少 Lock 字段（旧版接口）: {probe_title!r}")
    else:
        locked = parse_bool_word(probe_parsed["lock"])
        actual = bool(getattr(mod, "lockdown_active", False))
        if locked is None:
            checks.add("g2", "FAIL", f"无法解析 Lock 值: {probe_parsed['lock']!r}")
        else:
            checks.add("g2", "PASS" if locked == actual else "FAIL",
                       f"标题 Lock={probe_parsed['lock']} vs lockdown_active={actual}")

    # ---------- blit 背景缓存 ----------
    # 按 update() 返回的 artist 所在 axes 分组缓存背景；主循环每帧先 restore 再逐 artist blit，
    # 与 GUI 侧 FuncAnimation(blit=True) 的绘制路径同构（只重画变化 artist，不整图重绘）。
    blit_axes = []
    seen = set()
    for a in probe_artists:
        ax = getattr(a, "axes", None)
        if ax is not None and id(ax) not in seen:
            seen.add(id(ax))
            blit_axes.append(ax)
    if not blit_axes:
        raise RuntimeError("update() 返回的 artist 没有关联 axes，无法 blit")
    backgrounds = [(ax, canvas.copy_from_bbox(ax.bbox)) for ax in blit_axes]

    axis_list = []
    for name in ("ax_main", "ax_pop", "ax_virus", "ax_gen", "ax_btn"):
        ax = getattr(mod, name, None)
        if ax is not None and hasattr(ax, "get_xlim"):
            axis_list.append((name, ax))
    if not axis_list:
        axis_list = [(f"ax_{i}", ax) for i, ax in enumerate(blit_axes)]

    expected_limits = {}
    if width is not None and height is not None:
        expected_limits["ax_main"] = ((0.0, float(width)), (0.0, float(height)))
    if n_agents is not None:
        expected_limits["ax_pop"] = ((0.0, 500.0), (0.0, float(n_agents)))
    # 轴限是"契约 + 视觉留白"：ax_virus 上界 1.6、ax_gen 上界 105 都高于数据上界（1.0/100.0），
    # 让 100% 再感染率线不贴边；d2 断言的就是这些写死的值，模型侧一改就失败。
    expected_limits["ax_virus"] = ((0.0, 500.0), (0.0, 1.6))
    expected_limits["ax_gen"] = ((0.0, 500.0), (0.0, 105.0))

    # 曲线 artist 契约：5 条命名的统计曲线 + 8 条 lines_pop（h1 断言 artist 总数 N_ARTISTS=15）；
    # 按名字收集供 c1-c4 检查，缺某条就少一项。
    curves = {}
    for name in ("line_beta", "line_mu", "line_immunity", "line_gen", "line_reinf"):
        obj = getattr(mod, name, None)
        if obj is not None:
            curves[name] = obj
    for i, obj in enumerate(getattr(mod, "lines_pop", None) or []):
        curves[f"lines_pop[{i}]"] = obj

    ctx = {"curves": curves, "strain_missing": False, "last_window": None}
    tracker = Tracker()

    render_frames = set(parse_render_frames(args.render))
    render_dir = None
    if render_frames:
        render_dir = Path(args.render_dir) if args.render_dir else Path(
            tempfile.mkdtemp(prefix="headless_render_"))
        render_dir.mkdir(parents=True, exist_ok=True)
        out["render_dir"] = str(render_dir)

    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    # full 模式每帧都查位置（j1/j2 等），bench 模式按 SNAPSHOT_EVERY 降采样以压低计时开销
    check_pos_every = 1 if out["mode"] == "full" else SNAPSHOT_EVERY

    update_times, draw_times, restore_times = [], [], []
    snapshots = []
    rendered = []
    lims_first = None
    wan_total = 0
    prev_status = None
    last_snapshot = None
    nh_on = bool(cfg.get("NATURAL_HISTORY_ENABLED", False))
    iv_on = bool(cfg.get("INTERVENTIONS_ENABLED", False))
    seen_nh = {"max_asym": 0, "max_hosp": 0}

    for i in range(1, frames):  # 探针已推进 frame 0，循环从 1 开始，共 frames 次 update
        t = time.perf_counter()
        for _ax, bg in backgrounds:
            canvas.restore_region(bg)
        restore_times.append(time.perf_counter() - t)

        # 先取"上一帧"状态：m3 隔离标记延迟、e/f/m1 的迁移计数都要和 update 后的新状态做差分，
        # 所以在 mod.update(i) 之前捕获（含 isolated/quarantined/tests_total 供 v2/v3 断言）。
        prev_status = status_to_numpy(mod)
        _iso_prev_t = getattr(mod, "isolated", None)
        prev_iso = (_iso_prev_t.detach().cpu().numpy().astype(bool)
                    if isinstance(_iso_prev_t, torch.Tensor) else None)
        _q_prev_t = getattr(mod, "quarantined", None)
        prev_quar = (_q_prev_t.detach().cpu().numpy().astype(bool)
                     if isinstance(_q_prev_t, torch.Tensor) else None)
        prev_tests_total = getattr(mod, "tests_total", None)

        t = time.perf_counter()
        artists = mod.update(i)
        update_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        for a in artists:
            ax = getattr(a, "axes", None)
            if ax is not None:
                ax.draw_artist(a)
        for ax, _bg in backgrounds:
            canvas.blit(ax.bbox)
        draw_times.append(time.perf_counter() - t)

        # --render 的帧额外 savefig（全量重绘）：渲染只用于人眼核对，失败仅告警、不影响门禁
        if i in render_frames and render_dir is not None:
            path = render_dir / f"frame_{i:03d}.png"
            try:
                fig.savefig(path, dpi=100)
                rendered.append(str(path))
            except Exception as exc:  # 渲染失败不影响计时
                print(f"[warn] 渲染 frame {i} 失败: {exc!r}")

        # ==================== 逐帧记账（不计时）====================
        # 上面的 update/blit 已计过时；断言只读状态，开销不能算进性能口径，所以全放在计时块之后。
        now_status = status_to_numpy(mod)
        cnt = counts8(now_status)
        text = title_text_of(mod)
        parsed = parse_title(text)
        if int(cnt[6]) > seen_nh["max_asym"]:
            seen_nh["max_asym"] = int(cnt[6])
        if int(cnt[7]) > seen_nh["max_hosp"]:
            seen_nh["max_hosp"] = int(cnt[7])

        # a1
        if now_status is not None:
            tracker.run("a1")
            exp_inf = int(cnt[2] + cnt[3])
            if nh_on:
                exp_inf += int(cnt[6] + cnt[7])
            exp_symp = int(cnt[3])
            if parsed.get("inf") is None or parsed.get("symp") is None:
                tracker.hit("a1", i, f"标题缺少 Inf/Symp 字段: {text!r}")
            else:
                got_inf = int(round(float(parsed["inf"])))
                got_symp = int(round(float(parsed["symp"])))
                if got_inf != exp_inf or got_symp != exp_symp:
                    tracker.hit("a1", i,
                                f"标题 Inf={got_inf}, Symp={got_symp}；"
                                f"实测 Inf={exp_inf}, Symp={exp_symp}")

            # a2
            tracker.run("a2")
            if n_agents is not None and int(cnt.sum()) != n_agents:
                tracker.hit("a2", i, f"counts 总和={int(cnt.sum())} != N_AGENTS={n_agents}")

            # a3
            hist = getattr(mod, "history", None)
            tracker.run("a3")
            if not isinstance(hist, dict):
                tracker.hit("a3", i, "history 不是 dict")
            else:
                for s in range(8):
                    lst = hist.get(s)
                    if lst is None:
                        lst = hist.get(str(s))
                    if not lst:
                        tracker.hit("a3", i, f"history[{s}] 为空")
                        break
                    try:
                        tail = int(round(float(lst[-1])))
                    except Exception:
                        tracker.hit("a3", i, f"history[{s}][-1] 无法转成数值: {lst[-1]!r}")
                        break
                    if tail != int(cnt[s]):
                        tracker.hit("a3", i, f"history[{s}][-1]={tail} != counts[{s}]={int(cnt[s])}")
                        break

            # j1
            tracker.run("j1")
            bad = out_of_range_count(now_status)
            if bad:
                tracker.hit("j1", i, f"{bad} 个 status 越界（应在 0..7）")

            # m2: v2 住院/无人照护与床位机制一致
            if nh_on:
                tracker.run("m2")
                got_h = parsed.get("hosp")
                if got_h is None:
                    tracker.hit("m2", i, f"标题缺少 Hosp 字段: {text!r}")
                elif int(round(float(got_h))) != int(cnt[7]):
                    tracker.hit("m2", i,
                                f"标题 Hosp={int(round(float(got_h)))} != counts[7]={int(cnt[7])}")
                un_t = getattr(mod, "untreated", None)
                if isinstance(un_t, torch.Tensor):
                    exp_u = int(un_t.sum().item())
                    got_u = parsed.get("unmet")
                    if got_u is None:
                        tracker.hit("m2", i, "标题缺少 Unmet 字段")
                    elif int(round(float(got_u))) != exp_u:
                        tracker.hit("m2", i,
                                    f"标题 Unmet={int(round(float(got_u)))} != untreated={exp_u}")
                beds_t = getattr(mod, "beds_total", None)
                if beds_t is not None and int(cnt[7]) > int(beds_t):
                    tracker.hit("m2", i, f"住院人数 {int(cnt[7])} 超过床位数 {int(beds_t)}")

            # m3: 隔离者必须居家（在途者除外）
            if nh_on:
                iso_t = getattr(mod, "isolated", None)
                place_t = getattr(mod, "place", None)
                trav_t = getattr(mod, "traveling", None)
                if isinstance(iso_t, torch.Tensor) and isinstance(place_t, torch.Tensor):
                    tracker.run("m3")
                    iso_np = iso_t.detach().cpu().numpy().astype(bool)
                    place_np = place_t.detach().cpu().numpy()
                    bad_mask = iso_np & (place_np != 0)
                    # 隔离标记在本帧刚置位时允许还逗留在外（下一帧 mobility 才拉回家）
                    if prev_iso is not None:
                        bad_mask &= prev_iso
                    if isinstance(trav_t, torch.Tensor):
                        bad_mask &= ~trav_t.detach().cpu().numpy().astype(bool)
                    n_bad_iso = int(bad_mask.sum())
                    if n_bad_iso:
                        tracker.hit("m3", i, f"{n_bad_iso} 个隔离者不在家（非在途）")

            # m7/m8/m9/m10: v3 社会响应层
            # 顺序约束：先确认契约属性存在，再 tracker.run()。否则"属性缺失 → 守卫跳过"
            # 会被记成"检查了 N 次、无违例"（审计 obs-3 的假绿根因）。
            if iv_on:
                t_today = getattr(mod, "tests_today", None)
                t_cap = getattr(mod, "test_capacity_day", None)
                t_total = getattr(mod, "tests_total", None)
                if all(isinstance(v, (int, float)) for v in (t_today, t_cap, t_total)):
                    tracker.run("m7")
                    if int(t_today) > int(t_cap):
                        tracker.hit("m7", i, f"tests_today={t_today} > capacity={t_cap}")
                    if prev_tests_total is not None and int(t_total) < int(prev_tests_total):
                        tracker.hit("m7", i, "tests_total 递减")
                    prev_tests_total = int(t_total)
                else:
                    tracker.run("m7")
                    tracker.hit("m7", i, "契约缺失：tests_today/test_capacity_day/tests_total")

                q_t = getattr(mod, "quarantined", None)
                p_t = getattr(mod, "place", None)
                tr_t = getattr(mod, "traveling", None)
                if isinstance(q_t, torch.Tensor) and isinstance(p_t, torch.Tensor):
                    tracker.run("m8")
                    q_np = q_t.detach().cpu().numpy().astype(bool)
                    bad_q = q_np & (p_t.detach().cpu().numpy() != 0)
                    if prev_quar is not None:
                        bad_q &= prev_quar
                    if isinstance(tr_t, torch.Tensor):
                        bad_q &= ~tr_t.detach().cpu().numpy().astype(bool)
                    n_bad_q = int(bad_q.sum())
                    if n_bad_q:
                        tracker.hit("m8", i, f"{n_bad_q} 个密接隔离者不在家（非在途）")
                else:
                    tracker.run("m8")
                    tracker.hit("m8", i, "契约缺失：quarantined/place")

                mult = getattr(mod, "behavior_contact_mult", None)
                rep_t = getattr(mod, "reported_total", None)
                ons_t = getattr(mod, "onset_total", None)
                if all(isinstance(v, (int, float)) for v in (mult, rep_t, ons_t)):
                    tracker.run("m9")
                    lo = float(cfg.get("BEHAVIOR_MIN_CONTACT_MULT", 0.0))
                    if not (lo - 1e-9 <= float(mult) <= 1.0 + 1e-9):
                        tracker.hit("m9", i, f"behavior_contact_mult={mult} 越界")
                    if int(rep_t) > int(ons_t):
                        tracker.hit("m9", i,
                                    f"reported_total={rep_t} > onset_total={ons_t}")
                else:
                    tracker.run("m9")
                    tracker.hit("m9", i, "契约缺失：behavior_contact_mult/reported_total/onset_total")
            else:
                q_t = getattr(mod, "quarantined", None)
                counters = [getattr(mod, nm, None) for nm in
                            ("tests_total", "reported_total", "quarantine_total")]
                if isinstance(q_t, torch.Tensor) and all(
                        isinstance(v, (int, float)) for v in counters):
                    tracker.run("m10")
                    if int(q_t.sum().item()) > 0:
                        tracker.hit("m10", i, "关闭时 quarantined > 0")
                    for nm, v in zip(("tests_total", "reported_total", "quarantine_total"), counters):
                        if int(v) > 0:
                            tracker.hit("m10", i, f"关闭时 {nm}={v} > 0")
                else:
                    tracker.run("m10")
                    tracker.hit("m10", i, "契约缺失：quarantined/tests_total/reported_total/quarantine_total")

        # e / f / m1
        if prev_status is not None and now_status is not None and prev_status.shape == now_status.shape:
            tracker.run("e")
            n_bad = int(np.count_nonzero((prev_status == 2) & (now_status == 4)))
            if n_bad:
                tracker.hit("e", i, f"{n_bad} 个潜伏者同帧直接死亡（2 -> 4）")
            tracker.run("m1")
            n_bad_a = int(np.count_nonzero(
                ((prev_status == 2) | (prev_status == 6)) & (now_status == 4)))
            if n_bad_a:
                tracker.hit("m1", i, f"{n_bad_a} 个 E/A 同帧直接死亡（2/6 -> 4）")
            wan_total += int(np.count_nonzero(
                ((prev_status == 1) | (prev_status == 5)) & (now_status == 0)))

        # h
        tracker.run("h1")
        if not isinstance(artists, (list, tuple)) or len(artists) != N_ARTISTS:
            n_art = len(artists) if hasattr(artists, "__len__") else "?"
            tracker.hit("h1", i, f"artist 数量={n_art}，期望 {N_ARTISTS}")
        tracker.run("h2")
        for idx, a in enumerate(artists):
            if getattr(a, "axes", None) is None:
                tracker.hit("h2", i, f"第 {idx} 个 artist（{type(a).__name__}）没有 .axes")
                break

        # d1 / d2
        cur_lims = {name: lims_of(ax) for name, ax in axis_list}
        if lims_first is None:
            lims_first = cur_lims
        for name, (xlim, ylim) in cur_lims.items():
            tracker.run("d1")
            fx, fy = lims_first[name]
            if not (_close(xlim[0], fx[0]) and _close(xlim[1], fx[1])
                    and _close(ylim[0], fy[0]) and _close(ylim[1], fy[1])):
                tracker.hit("d1", i, f"{name} 轴限漂移: xlim={xlim} vs {fx}; ylim={ylim} vs {fy}")
            if name in expected_limits:
                tracker.run("d2")
                ex, ey = expected_limits[name]
                if not (_close(xlim[0], ex[0]) and _close(xlim[1], ex[1])
                        and _close(ylim[0], ey[0]) and _close(ylim[1], ey[1])):
                    tracker.hit("d2", i,
                                f"{name} 轴限={xlim}/{ylim}，期望={ex}/{ey}")

        # j2
        if i % check_pos_every == 0 or i == frames - 1:
            pos_t = getattr(mod, "pos", None)
            if isinstance(pos_t, torch.Tensor):
                tracker.run("j2")
                if not bool(torch.isfinite(pos_t).all().item()):
                    tracker.hit("j2", i, "pos 含 NaN/Inf")

        # k1/k4/k5/k6：按行为时钟在 03:00 与 12:00 抽查日常节律
        _clock_start = cfg.get("CLOCK_START_MIN")
        _day_frames = cfg.get("DAY_FRAMES")
        if _clock_start is not None and _day_frames:
            _min_per_frame = 1440.0 / float(_day_frames)
            _clock = (float(_clock_start) + i * _min_per_frame) % 1440.0
            _clock_day = int((float(_clock_start) + i * _min_per_frame) // 1440)
            if min(abs(_clock - 180.0), abs(_clock - 720.0)) < 1e-6:
                _activity_checks(mod, i, _clock, _clock_day, tracker)

        # 快照 + 派生检查
        is_snapshot = (i % SNAPSHOT_EVERY == 0) or (i == frames - 1)
        if is_snapshot:
            _trip_checks(mod, i, tracker)
            derived_checks(mod, i, parsed, tracker, ctx)
            snap = {
                "frame": i,
                "counts": [int(v) for v in cnt],
                "counts_total": int(cnt.sum()),
                "title": text,
                "title_parsed": _jsonable(parsed),
                "lock": getattr(mod, "lockdown_active", None),
                "iv": {
                    "enabled": iv_on,
                    "quarantined": (int(getattr(mod, "quarantined").sum().item())
                                    if isinstance(getattr(mod, "quarantined", None), torch.Tensor)
                                    else None),
                    "tests_today": getattr(mod, "tests_today", None),
                    "tests_total": getattr(mod, "tests_total", None),
                    "test_capacity_day": getattr(mod, "test_capacity_day", None),
                    "reported_total": getattr(mod, "reported_total", None),
                    "onset_total": getattr(mod, "onset_total", None),
                    "contact_mult": getattr(mod, "behavior_contact_mult", None),
                    "quarantine_total": getattr(mod, "quarantine_total", None),
                },
                "axes_lims": {name: {"xlim": list(v[0]), "ylim": list(v[1])}
                              for name, v in cur_lims.items()},
                "window_reinf": _jsonable(ctx["last_window"]),
                "n_artists": len(artists) if hasattr(artists, "__len__") else None,
                "update_ms": update_times[-1] * 1000.0,
                "draw_blit_ms": draw_times[-1] * 1000.0,
            }
            last_snapshot = snap
            if out["mode"] == "full":
                snapshots.append(snap)

    # ==================== 收尾检查 ====================
    lims_last = {name: lims_of(ax) for name, ax in axis_list}
    out["axes_lims_first"] = {k: {"xlim": list(v[0]), "ylim": list(v[1])} for k, v in (lims_first or {}).items()}
    out["axes_lims_last"] = {k: {"xlim": list(v[0]), "ylim": list(v[1])} for k, v in lims_last.items()}

    # a1/a2/a3/b/c*/d*/e/h*/j*
    checks.add("a1", *tracker.result("a1"))
    checks.add("a2", *tracker.result("a2"))
    checks.add("a3", *tracker.result("a3"))
    if ctx["strain_missing"]:
        checks.add("b", "SKIP" if not checks.enforce else "FAIL",
                   "strain_stats 缺少 new_infections/reinfections 键（旧版接口）")
    else:
        checks.add("b", *tracker.result("b"))
    checks.add("c1", *tracker.result("c1"))
    checks.add("c2", *tracker.result("c2"))
    checks.add("c3", *tracker.result("c3"))
    checks.add("c4", *tracker.result("c4"))
    checks.add("d1", *tracker.result("d1"))
    checks.add("d2", *tracker.result("d2"))
    checks.add("e", *tracker.result("e"))
    checks.add("h1", *tracker.result("h1"))
    checks.add("h2", *tracker.result("h2"))
    checks.add("j1", *tracker.result("j1"))
    checks.add("j2", *tracker.result("j2"))
    checks.add("m1", *tracker.result("m1"))
    checks.add("m2", *tracker.result("m2"))
    checks.add("m3", *tracker.result("m3"))
    if iv_on:
        checks.add("m7", *tracker.result("m7"))
        checks.add("m8", *tracker.result("m8"))
        checks.add("m9", *tracker.result("m9"))
        checks.add("m10", "PASS", "社会响应已启用，不适用")
    else:
        checks.add("m7", "PASS", "社会响应关闭，不适用")
        checks.add("m8", "PASS", "社会响应关闭，不适用")
        checks.add("m9", "PASS", "社会响应关闭，不适用")
        checks.add("m10", *tracker.result("m10"))

    # m4: 死亡年龄右偏（v2 年龄别严重性机制的定性检查）
    # 判据 = 死亡者平均年龄比全人群高 2.0 岁以上；20 人是最小样本下限，不足时只记 SKIP（样本太小分辨不出右偏）。
    st_end = status_to_numpy(mod)
    ages_t = getattr(mod, "agent_age", None)
    if nh_on and st_end is not None and isinstance(ages_t, torch.Tensor):
        ages = ages_t.detach().cpu().numpy()
        dead = st_end == 4
        n_dead = int(dead.sum())
        if n_dead < 20:
            checks.add("m4", "SKIP", f"死亡人数 {n_dead} < 20，不足以判断年龄分布")
        else:
            mean_dead = float(ages[dead].mean())
            mean_all = float(ages.mean())
            ok = mean_dead >= mean_all + 2.0
            checks.add("m4", "PASS" if ok else "FAIL",
                       f"死亡平均年龄={mean_dead:.1f}，全人群={mean_all:.1f}（需高 2 岁以上）")
    else:
        checks.add("m4", "SKIP", "旧路径或无 agent_age")

    # m5: v2 自然史应在足够长运行里出现无症状者
    # 2000 帧是门禁选定的观察窗口：更短的运行直接 SKIP，避免"A 状态还没出现"被误判成 FAIL。
    if not nh_on:
        checks.add("m5", "SKIP", "旧路径（toy/legacy）不产生 A 状态")
    elif frames >= 2000:
        ok = seen_nh["max_asym"] > 0
        checks.add("m5", "PASS" if ok else "FAIL",
                   f"无症状峰值={seen_nh['max_asym']}，住院峰值={seen_nh['max_hosp']}")
    else:
        checks.add("m5", "SKIP", f"帧数 {frames} < 2000")

    # m6: 旧路径不应出现 A/H 状态
    # counts[6]/[7]（无症状/住院）是 v2 专有状态；legacy 路径出现非零说明自然史开关泄漏到了旧路径。
    _last_cnt = (last_snapshot or {}).get("counts")
    if nh_on:
        checks.add("m6", "SKIP", "v2 路径允许 6/7 状态")
    elif _last_cnt is not None and len(_last_cnt) >= 8:
        ok = int(_last_cnt[6]) == 0 and int(_last_cnt[7]) == 0
        checks.add("m6", "PASS" if ok else "FAIL",
                   f"末帧 counts[6]={int(_last_cnt[6])}, counts[7]={int(_last_cnt[7])}（旧路径应为 0）")
    else:
        checks.add("m6", "SKIP", "无末帧快照")

    # k1/k3/k4/k5/k6：逐帧累积型行为断言
    for _key in ("k1", "k3", "k4", "k5", "k6"):
        checks.add(_key, *tracker.result(_key))

    # k2: 平均出行频率（按在世人群与运行天数折算）
    # 区间 成人 1.5-5.0 次/日、学生 0.5-4.0 次/日 是行为日程的设计值窗口；折算分母是"运行天数"，
    # 所以帧数不足一天或模型缺 trip_count 时只能 SKIP。
    _day_frames = cfg.get("DAY_FRAMES")
    _trip_cnt = getattr(mod, "trip_count", None)
    if not _day_frames:
        checks.add("k2", "SKIP", "模型缺少 DAY_FRAMES（旧版接口）")
    elif frames < int(_day_frames):
        checks.add("k2", "SKIP", f"帧数 {frames} < DAY_FRAMES={_day_frames}")
    elif not isinstance(_trip_cnt, torch.Tensor):
        checks.add("k2", "SKIP", "模型缺少 trip_count")
    else:
        _days = float(frames) / float(_day_frames)
        _cnt_np = _trip_cnt.detach().cpu().numpy()
        _st_np = status_to_numpy(mod)
        _alive = (_st_np != 4) if _st_np is not None else np.ones(_cnt_np.shape[0], bool)
        _parts = []
        _ok = True
        for _name, _mask_t, _lo, _hi in (
            ("成人", getattr(mod, "employed", None), 1.5, 5.0),
            ("学生", getattr(mod, "student", None), 0.5, 4.0),
        ):
            if not isinstance(_mask_t, torch.Tensor):
                continue
            _m = _mask_t.detach().cpu().numpy().astype(bool) & _alive
            if int(_m.sum()) == 0:
                continue
            _rate = float(_cnt_np[_m].mean()) / _days
            _parts.append(f"{_name} {_rate:.2f}/日")
            if not (_lo <= _rate <= _hi):
                _ok = False
        checks.add("k2", "PASS" if _ok else "FAIL",
                   f"{'；'.join(_parts)}（期望 成人 1.5-5.0，学生 0.5-4.0，days={_days:.2f}）")

    # f: 免疫/康复 -> 健康 的消退迁移必须出现（wan_total 由主循环逐帧差分累加）
    # 只在 2000 帧以上、且 toy/None 档案下断言；其它档案的时间尺度下 2000 帧走不完一个消退周期。
    _profile = getattr(mod, "ACTIVE_PROFILE_NAME", None)
    if frames < 2000:
        checks.add("f", "SKIP", f"帧数 {frames} < 2000，不检查")
    elif _profile not in (None, "toy"):
        checks.add("f", "SKIP",
                   f"档案 {_profile} 的时间尺度下 {frames} 帧不足以观察免疫消退，跳过")
    else:
        checks.add("f", "PASS" if wan_total > 0 else "FAIL",
                   f"免疫/康复 -> 健康 迁移总数={wan_total}（要求 > 0）")

    # i: CUDA 峰值显存门禁。300 MB 是基线；超过约 25 万 agent 后按 1.2 MB/千 agent 线性放宽，
    # 否则规模本身就会顶满阈值，断言变成"种群上限"而不是显存泄漏检测。只在 >=1000 帧时检查（峰值采样需要足够长的运行）。
    cuda_peak_mb = None
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
            cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    out["cuda_peak_mb"] = cuda_peak_mb
    peak_limit_mb = max(CUDA_PEAK_LIMIT_MB, 1.2 * (n_agents or 0) / 1000.0)
    out["cuda_peak_limit_mb"] = peak_limit_mb
    if torch.cuda.is_available() and frames >= 1000:
        ok = cuda_peak_mb is not None and cuda_peak_mb < peak_limit_mb
        checks.add("i", "PASS" if ok else "FAIL",
                   f"peak allocated={cuda_peak_mb:.1f} MB（阈值 {peak_limit_mb:.0f} MB，device={out['device']}）"
                   if cuda_peak_mb is not None else "无法读取 CUDA 峰值显存")
    else:
        checks.add("i", "SKIP",
                   "CUDA 不可用" if not torch.cuda.is_available() else f"帧数 {frames} < 1000")

    # 观察项（INFO，不计入 FAIL）：曲线滚动窗口与标题 t= 只记录不断言，用来发现窗口/时间轴口径漂移
    if last_snapshot is not None:
        xlen = None
        for name, line in curves.items():
            try:
                xd = np.asarray(line.get_xdata(), dtype=float)
            except Exception:
                continue
            xlen = int(xd.size)
            if xlen > ROLLING_POINTS:
                checks.info("i1", f"{name} 曲线点数={xlen} > {ROLLING_POINTS}（未断言）")
                break
        if xlen is not None and xlen <= ROLLING_POINTS:
            hist0 = getattr(mod, "history", None)
            hist_len = len(hist0.get(0)) if isinstance(hist0, dict) and hist0.get(0) else None
            checks.info("i1", f"曲线点数={xlen}（<= {ROLLING_POINTS}），history[0] 长度={hist_len}")
        t_val = (last_snapshot.get("title_parsed") or {}).get("t")
        if t_val is not None:
            checks.info("i2", f"末帧标题 t={t_val}，循环末帧 index={frames - 1}")
        else:
            checks.info("i2", "标题无 t= 字段（旧版接口，未断言）")

    # ==================== 汇总数据 ====================
    # loop_s 只累加 update/draw_blit/restore 三段计时；wall_s 还含加载、断言与收尾，
    # 所以 wall_s 更大；两者都只适合同会话 A/B 比较，跨机器绝对值不可比。
    loop_s = float(sum(update_times) + sum(draw_times) + sum(restore_times))
    out["frames"] = frames
    out["timings"] = {
        "first_draw_s": first_draw,
        "probe_update_s": probe_update,
        "load_s": out.pop("load_s", None),
        "update": _stats(update_times),
        "draw_blit": _stats(draw_times),
        "restore": _stats(restore_times),
        "loop_s": loop_s,
    }
    out["wall_s"] = time.perf_counter() - wall_start
    out["snapshots"] = snapshots
    out["last_snapshot"] = last_snapshot
    out["rendered"] = rendered
    out["wan_total"] = wan_total
    return mod


# ==================== 输出 ====================
def print_summary(out, checks, frames):
    """打印人读汇总：计时、快照表、断言明细；末行 `结论: PASS|FAIL` 是用户门禁的输出标记。

    "无头验收汇总""断言明细""失败断言 N 项 / 共 M 项"这些文案是外部依赖的文本契约，改文案等于改门禁输出。
    退出码不在这里决定（见 main）。
    """
    tim = out.get("timings", {})
    u = tim.get("update", {})
    d = tim.get("draw_blit", {})
    print()
    print("=" * 68)
    print("无头验收汇总")
    print("=" * 68)
    print(f"模块      : {out.get('module')}")
    if out.get("profile"):
        print(f"疾病档案  : {out['profile']}")
    print(f"设备      : {out.get('device')}    帧数: {frames}    模式: {out.get('mode')}"
          f"    种子: {out.get('seed')}")
    print(f"首帧 draw : {_fmt_ms(tim.get('first_draw_s'))}    "
          f"探针 update(0): {_fmt_ms(tim.get('probe_update_s'))}（不计入统计）")
    print(f"update    : mean={_fmt_ms(u.get('mean'))}  median={_fmt_ms(u.get('median'))}  "
          f"max={_fmt_ms(u.get('max'))}  (n={u.get('n')})")
    print(f"draw+blit : mean={_fmt_ms(d.get('mean'))}  median={_fmt_ms(d.get('median'))}  "
          f"max={_fmt_ms(d.get('max'))}  (n={d.get('n')})")
    print(f"总耗时    : {out.get('wall_s', 0.0):.2f} s（计时循环 {tim.get('loop_s', 0.0):.2f} s）")
    if out.get("cuda_peak_mb") is not None:
        print(f"CUDA peak : {out['cuda_peak_mb']:.1f} MB"
              f"（阈值 {out.get('cuda_peak_limit_mb', CUDA_PEAK_LIMIT_MB):.0f} MB）")
    else:
        print("CUDA peak : 不可用")

    if out.get("snapshots"):
        print("-" * 68)
        print(f"快照（每 {SNAPSHOT_EVERY} 帧 + 末帧，共 {len(out['snapshots'])} 条）:")
        print("   帧    healthy immune incub symp dead recov   标题Reinf  窗口Reinf  upd/draw(ms)")
        for s in out["snapshots"]:
            c = s["counts"]
            tr = (s.get("title_parsed") or {}).get("reinf")
            wr = (s.get("window_reinf") or {}).get("ratio")
            print(f"  {s['frame']:>4}  {c[0]:>7} {c[1]:>6} {c[2]:>6} {c[3]:>5} {c[4]:>5} {c[5]:>6}"
                  f"   {('n/a' if tr is None else f'{tr:>7.2f}')}"
                  f"   {('n/a' if wr is None else f'{wr:>7.2f}')}"
                  f"   {s['update_ms']:>6.2f}/{s['draw_blit_ms']:.2f}")
    if out.get("last_snapshot"):
        ls = out["last_snapshot"]
        print("-" * 68)
        print(f"末帧标题 : {ls.get('title')!r}")
        print(f"末帧计数 : counts={ls.get('counts')} 总和={ls.get('counts_total')}")

    print("-" * 68)
    print("断言明细:")
    for item in checks.items:
        tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "INFO": "[INFO]"}[item["status"]]
        print(f"  {tag} {item['key']:<3} {item['label']}")
        if item["detail"]:
            print(f"          {item['detail']}")

    fails = checks.failed()
    print("-" * 68)
    if checks.enforce:
        verdict = "PASS" if not fails else "FAIL"
        # SKIP 数显式披露（obs-1）：SKIP 不计入退出码是既有政策，但不能让"共 N 项"暗示
        # 全部真跑过；无 SKIP 时保持原输出格式不变。
        skip_n = sum(1 for item in checks.items if item["status"] == "SKIP")
        skip_txt = f" / SKIP {skip_n} 项" if skip_n else ""
        print(f"结论: {verdict}（失败断言 {len(fails)} 项 / 共 {len(checks.items)} 项{skip_txt}）")
    else:
        print(f"结论: 未启用断言（--no-assert）；记录到 {len(fails)} 项不满足项，退出码 0")
    if out.get("rendered"):
        print(f"渲染帧: {len(out['rendered'])} 张 -> {out.get('render_dir')}")
    if out.get("render_dir") and not out.get("rendered"):
        print(f"[warn] 指定了 --render 但未命中任何帧（帧数 {frames} 太小？）-> {out.get('render_dir')}")
    print("=" * 68)


def build_parser():
    """构造 CLI：参数名、默认值、help 文案都是对外契约（脚本/CI/文档都引用它们）。"""
    p = argparse.ArgumentParser(
        prog="headless_check.py",
        description="simple_disease_simulation 的无头验收 / 性能基线脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--module", default=None,
                   help="待测模型文件路径（默认仓库根的 simple_disease_simulation.py）")
    p.add_argument("--profile", default=None,
                   help="疾病参数档案名（写入 DISEASE_PROFILE；默认 toy）")
    p.add_argument("--frames", type=int, default=300, help="驱动帧数（默认 300）")
    p.add_argument("--json", dest="json_path", default=None, help="把结果写入 JSON 文件")
    p.add_argument("--no-assert", action="store_true",
                   help="关闭断言拦截，只记录（旧版兼容冒烟用）")
    p.add_argument("--bench", type=int, default=None, metavar="N",
                   help="基准模式：只跑 N 帧计时循环，不写每帧快照")
    p.add_argument("--equiv", action="store_true",
                   help="只跑独立的算法等价测试（稠密 any/argmax vs 稀疏 scatter_reduce）")
    p.add_argument("--render", default=None, metavar='"150,500"',
                   help="在这些帧上 fig.savefig 出 PNG（逗号分隔）")
    p.add_argument("--render-dir", default=None, help="PNG 输出目录（默认临时目录）")
    p.add_argument("--seed", type=int, default=1234,
                   help="import 模型前设置 torch/numpy 随机种子（默认 1234，-1 表示不设）")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="显示模型 import 期的 stdout")
    return p


def _tune_streams():
    """重定向到管道/文件时统一用 UTF-8，避免 GBK 控制台下中文乱码；TTY 保留原生编码。"""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")


def main(argv=None):
    """CLI 入口，返回退出码：0=启用断言全过（SKIP 不算失败）或 --no-assert；1=存在 FAIL；2=参数非法/运行异常。

    --equiv 只跑算法等价测试并直接返回其 0/1；--json 落盘 out 的契约结构（键名不可随意增删改）。
    """
    _tune_streams()

    args = build_parser().parse_args(argv)

    if args.equiv:
        return run_equiv()

    frames = args.bench if args.bench is not None else args.frames
    if frames is None or frames <= 0:
        print("错误: --frames / --bench 必须为正整数", file=sys.stderr)
        return 2
    if args.seed is not None and args.seed < 0:
        args.seed = None
    # 疾病档案：默认 toy 以保持回归与 waning 检查有效；显式 --profile 优先
    if args.profile:
        os.environ["DISEASE_PROFILE"] = str(args.profile)
    else:
        os.environ.setdefault("DISEASE_PROFILE", "toy")

    checks = Checks(enforce=not args.no_assert)
    out = {}

    try:
        run_sim(args, frames, checks, out)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("运行异常终止：", file=sys.stderr)
        traceback.print_exc()
        return 2

    print_summary(out, checks, frames)

    out["checks"] = checks.items
    out["failed"] = [i["key"] for i in checks.failed()]
    out["verdict"] = ("PASS" if not checks.failed() else "FAIL") if checks.enforce else "NO_ASSERT"
    out["versions"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.json_path:
        path = Path(args.json_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_jsonable(out), fh, ensure_ascii=False, indent=2)
        print(f"JSON 已写入: {path}")

    return checks.exit_code()


if __name__ == "__main__":
    sys.exit(main())
