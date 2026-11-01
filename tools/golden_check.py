#!/usr/bin/env python
"""逐位金标（golden）**历史快照**工具：固化初始化指纹 + 逐帧 counts/标题，定位第一处分歧。

它不是门禁：第二轮优化（S1）之后官方验收口径是"统计区间 + harness 38 项"
（tests/run_all.py、tools/headless_check.py），跨实现逐位相等已不再要求。它只在
怀疑"某次改动动了 RNG 抽取顺序或初始化"时用来二分定位。

用法（--dump/--compare 可叠加）：
    python tools/golden_check.py --profile toy --frames 300 --dump  benchmarks/golden_toy300.json
    python tools/golden_check.py --profile toy --frames 300 --compare benchmarks/golden_toy300.json
    python tools/golden_check.py --profile toy --frames 1   --contract   # 只查公开名字契约
退出码：0 = 一致/成功；1 = 出现分歧（打印首次分歧）；2 = 加载失败/金标文件缺失。

口径陷阱：主循环前先跑一次 update(0) 探针（同 headless_check.py:669），随后
range(1, frames)（同 :758）。探针会推进一帧仿真与 RNG，所以 --frames 是"含探针
共 frames 次 update"：少算/多算一帧，整条轨迹平移，比对必然 FAIL。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODULE = ROOT / "simple_disease_simulation.py"

# ---------------------------------------------------------------------------
# 名字契约清单：既用于"指纹完整性自检"，也用于包化重构后的类型/存在性校验。
# 张量按内容哈希（tensor_digest），标量/容器按稳定短表示（stable_repr）。
# 漏登记的张量不致命，但会被 collect_init() 以 unclassified_tensors 提示出来。
# ---------------------------------------------------------------------------
CONTRACT_TENSORS = [
    # 个体状态
    "status", "pos", "place", "traveling", "dest",
    "trip_left", "trip_total", "trip_place", "trip_count",
    "home_pos", "home_anchor", "household_id", "work_pos", "school_pos",
    "work_venue_idx", "school_venue_idx",
    "employed", "student", "agent_age",
    "virus_beta", "virus_gamma", "virus_mu", "virus_sigma", "virus_gen",
    "timer", "max_incubation", "ever_infected", "immunity_level",
    # v2 自然史
    "inf_elapsed", "symp_elapsed", "a_timer", "iso_timer",
    "isolated", "admission_attempted", "untreated",
    "severity_mult", "hosp_mult", "suscept_mult", "age_band",
    # v3 社会响应
    "wants_test", "test_timer", "tested", "trace_timer",
    "quarantined", "quarantine_timer",
    # 每日计划 / 日程
    "errand_flag", "errand_start_min", "errand_kind", "errand_pos",
    "work_start_min", "work_end_min", "school_start_min", "school_end_min",
    # 场所表
    "shop_venue", "park_venue",
]

CONTRACT_SCALARS = [
    "beds_total", "lockdown_active",
    "tests_today", "tests_total", "test_capacity_day",
    "reported_total", "onset_total", "quarantine_total",
    "behavior_contact_mult",
]

CONTRACT_MODULES = ["disease_profiles", "torch"]

# harness 依赖的可调用 / matplotlib 对象（只校验存在性，不做指纹）。
# update() 必须返回 15 个带 .axes 的 artist、lines_pop 8 条、history 整数键 0..7：
# 这三条是 headless_check 的硬契约，少一个 harness 会静默 SKIP 部分断言。
CONTRACT_CALLABLES = ["update", "toggle"]
CONTRACT_ARTISTS = [
    "fig", "ax_main", "ax_pop", "ax_virus", "ax_gen", "ax_btn",
    "title_obj", "btn",
    "line_beta", "line_mu", "line_immunity", "line_gen", "line_reinf",
]


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def patch_matplotlib() -> None:
    """在 exec_module 之前打桩：禁止 import 期创建真实动画或阻塞在 plt.show。

    与 headless_check 同语义，且顺序不能反——入口在 import 期就会建 fig/animation，
    先加载再打桩等于已经弹窗/卡死（不报错，只是静默变慢或挂住）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt

    class _DummyFuncAnimation:  # pragma: no cover - 仅打桩
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def save(self, *args, **kwargs):
            return None

    mpl_animation.FuncAnimation = _DummyFuncAnimation  # type: ignore[assignment]
    plt.show = lambda *a, **k: None  # type: ignore[assignment]


def load_module(path: Path, env: dict[str, str], verbose: bool = False):
    """以最接近 harness 的方式加载入口：注入 env → 打桩动画 → exec_module。

    path 必须以 ``.py`` 结尾：``spec_from_file_location`` 对未知后缀返回 None（`.bak`
    会走到 SystemExit → exit 2），A/B 前先把 .bak 复制成 ``_orig_sim.py`` 再传入；
    同进程先后 load 两入口不安全（第二个命中 sys.modules 的包），比对请用 ab_iter.py。
    """
    for key, value in env.items():
        os.environ[key] = value

    patch_matplotlib()

    spec = importlib.util.spec_from_file_location("sim_golden", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim_golden"] = module

    captured = io.StringIO()
    t0 = time.perf_counter()
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with contextlib.redirect_stdout(captured):
            spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = old
    load_s = time.perf_counter() - t0

    if verbose:
        print(captured.getvalue())
    return module, load_s, captured.getvalue()


# ---------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------
def tensor_digest(value) -> str:
    """张量指纹：sha256("(shape)|dtype|" + 原始字节) 取前 32 个 hex。

    先 detach→cpu→contiguous，保证同内容跨设备同哈希；非张量返回 "NOT_A_TENSOR"
    由调用方记入 missing_tensors。只取 32 hex（128 bit）够用，全 64 位会让长跑
    金标 JSON（v3 10800 帧那份已 6 MB）体积翻倍。
    """
    import torch

    if not isinstance(value, torch.Tensor):
        return "NOT_A_TENSOR"
    t = value.detach()
    if t.is_cuda:
        t = t.cpu()
    t = t.contiguous()
    h = hashlib.sha256()
    h.update(f"{tuple(t.shape)}|{t.dtype}|".encode("utf-8"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:32]


def stable_repr(value) -> str:
    """标量/容器的稳定短表示，用于初始化标量指纹。

    dict 只记键名（值可能是大结构），list/tuple 只记类型+长度——指纹要的是
    "结构没变"，不是逐元素快照；float 用 repr 保留全精度。
    """
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (bool, int, str)) or value is None:
        return repr(value)
    if isinstance(value, dict):
        return "dict[" + ",".join(sorted(str(k) for k in value)) + "]"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[len={len(value)}]"
    return f"<{type(value).__name__}>"


def rng_digest() -> dict[str, str]:
    """CPU/CUDA RNG 状态的 sha256 前 32 hex；CUDA 不可用时记 "unavailable"。"""
    import torch

    out = {"cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:32]}
    if torch.cuda.is_available():
        try:
            out["cuda"] = hashlib.sha256(
                torch.cuda.get_rng_state().cpu().numpy().tobytes()
            ).hexdigest()[:32]
        except Exception:  # pragma: no cover
            out["cuda"] = "unavailable"
    return out


def config_digest(config) -> tuple[str, int]:
    """CONFIG dict 的排序 JSON 哈希 + 键数；键数能立刻暴露"悄悄增删了开关"。"""
    payload = json.dumps(config, sort_keys=True, default=repr, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32], len(config)


def collect_init(module) -> dict:
    """初始化后立刻抓一份指纹：登记张量/标量/RNG/CONFIG/PROFILE_INFO。

    unclassified_tensors 列出"模块里有 shape+dtype 但没登记进 CONTRACT_TENSORS"的
    名字——漏登记不致命，但新增状态张量若不在此露头，就会悄悄逃过金标。
    """
    tensors: dict[str, str] = {}
    missing_tensors: list[str] = []
    for name in CONTRACT_TENSORS:
        value = getattr(module, name, None)
        digest = tensor_digest(value)
        if digest == "NOT_A_TENSOR":
            missing_tensors.append(name)
        else:
            tensors[name] = digest

    scalars: dict[str, str] = {}
    missing_scalars: list[str] = []
    for name in CONTRACT_SCALARS:
        if not hasattr(module, name):
            missing_scalars.append(name)
            continue
        scalars[name] = stable_repr(getattr(module, name))

    cfg = getattr(module, "CONFIG", None)
    cfg_hash, cfg_len = config_digest(cfg) if isinstance(cfg, dict) else ("NOT_A_DICT", 0)

    info = getattr(module, "PROFILE_INFO", None)
    info_hash = (
        hashlib.sha256(
            json.dumps(info, sort_keys=True, default=repr, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:32]
        if isinstance(info, dict)
        else "NOT_A_DICT"
    )

    discovered = sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("__")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
    )
    undiscovered = [n for n in discovered if n not in CONTRACT_TENSORS]

    return {
        "tensor_hashes": tensors,
        "scalars": scalars,
        "rng": rng_digest(),
        "config_sha": cfg_hash,
        "config_len": cfg_len,
        "profile_info_sha": info_hash,
        "profile_name": getattr(module, "ACTIVE_PROFILE_NAME", None),
        "device": str(getattr(module, "device", "?")),
        "n_agents": int(getattr(module, "CONFIG", {}).get("N_AGENTS", -1)),
        "missing_tensors": missing_tensors,
        "missing_scalars": missing_scalars,
        "unclassified_tensors": undiscovered,
    }


def verify_contract(module, dynamic: bool = True) -> tuple[bool, list[str]]:
    """校验模块级公开名字是否齐全（headless_check 全部按名字 getattr）。

    除存在性外还查三条 harness 硬契约：update() 返回 15 个 artist 且都带 .axes、
    lines_pop 8 条、history 有整数键 0..7——缺一条 harness 会静默 SKIP 部分断言。
    dynamic=True 时的 update(0) 会推进一帧仿真与 RNG，**只有 --contract 模式允许**；
    正常路径必须传 False（金标轨迹本身就从 update(0) 起），改由 run_frames() 探针校验。
    """
    import torch

    problems: list[str] = []
    for name in CONTRACT_TENSORS:
        value = getattr(module, name, None)
        if not isinstance(value, torch.Tensor):
            problems.append(f"{name}: 不是 torch.Tensor（{type(value).__name__}）")
    for name in CONTRACT_SCALARS:
        if not hasattr(module, name):
            problems.append(f"{name}: 缺失")
    for name in CONTRACT_MODULES + CONTRACT_ARTISTS:
        if not hasattr(module, name):
            problems.append(f"{name}: 缺失")
    for name in CONTRACT_CALLABLES:
        if not callable(getattr(module, name, None)):
            problems.append(f"{name}: 不可调用")
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        problems.append("CONFIG: 必须是 plain dict")
    artists = getattr(module, "update", None)
    if dynamic and callable(artists):
        try:
            produced = list(artists(0))
        except Exception as exc:  # pragma: no cover
            problems.append(f"update(0) 抛异常：{exc!r}")
            produced = []
        if len(produced) != 15:
            problems.append(f"update(0) 返回 {len(produced)} 个 artist（应为 15）")
        for artist in produced:
            if getattr(artist, "axes", None) is None:
                problems.append(f"artist {type(artist).__name__} 没有 .axes")
                break
    lines_pop = getattr(module, "lines_pop", None)
    if lines_pop is not None and len(list(lines_pop)) != 8:
        problems.append(f"lines_pop 长度 {len(list(lines_pop))}（应为 8）")
    history = getattr(module, "history", None)
    if isinstance(history, dict):
        for key in range(8):
            if key not in history:
                problems.append(f"history 缺整数键 {key}")
                break
    return (not problems), problems


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def run_frames(module, frames: int, stride: int) -> dict:
    """跑 frames 次 update（1 次探针 + frames-1 次主循环），按 stride 抽 counts/标题。

    timings 是逐帧 sync 的 wall，只能和同会话/同机器比（跨会话不可比）。
    """
    counts_all: list[list[int]] = []
    titles: list[str] = []
    iv_rows: list[dict] = []
    timings: list[float] = []

    import torch

    # 与 tools/headless_check.py 对齐：主循环前先做一次 update(0) 探针（不计入统计）。
    # 探针会推进一帧仿真与 RNG，因此"是否探针"会改变后续轨迹——用户给出的
    # 验收 counts（toy 300 / v2 400 / v3 300·500）都是 harness 口径（含探针）。
    t_probe = time.perf_counter()
    probe_artists = list(module.update(0))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    probe_s = time.perf_counter() - t_probe
    probe_title = getattr(module, "title_obj").get_text()
    probe_all_axes = all(getattr(a, "axes", None) is not None for a in probe_artists)

    def snapshot(frame_no: int) -> None:
        """记一帧 counts/标题/v3 计数器；INTERVENTIONS_ENABLED=0 时 iv 段为空 dict。"""
        status = getattr(module, "status")
        iv_enabled = bool(getattr(module, "CONFIG", {}).get("INTERVENTIONS_ENABLED", False))
        iv_row = {
            "Q": int(torch.count_nonzero(getattr(module, "quarantined")).item())
            if isinstance(getattr(module, "quarantined", None), torch.Tensor)
            else 0,
            "tests_total": int(getattr(module, "tests_total", 0)),
            "reported_total": int(getattr(module, "reported_total", 0)),
            "onset_total": int(getattr(module, "onset_total", 0)),
            "quarantine_total": int(getattr(module, "quarantine_total", 0)),
            "mult": float(getattr(module, "behavior_contact_mult", 1.0)),
        } if iv_enabled else {}
        counts_all.append(torch.bincount(status, minlength=8).cpu().tolist())
        title = getattr(module, "title_obj").get_text()
        titles.append(title)
        iv_rows.append(iv_row)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 探针已推进 frame 0，主循环从 1 开始（与 headless_check.py:758 完全一致），
    # 总计恰好 frames 次 update，末帧 index = frames-1。
    for frame in range(1, frames):
        t0 = time.perf_counter()
        module.update(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        # 按 stride 抽样只为压 JSON 体积（v3 10800 帧全采约 6 MB），末帧必采。
        if frame % stride == 0 or frame == frames - 1:
            snapshot(frame)

    peak_mb = (
        round(torch.cuda.max_memory_allocated() / 2**20, 2)
        if torch.cuda.is_available()
        else None
    )
    return {
        "probe": {
            "update_ms": round(probe_s * 1000, 3),
            "n_artists": len(probe_artists),
            "all_have_axes": probe_all_axes,
            "title": probe_title,
        },
        "frames": {
            "counts": counts_all,
            "titles": titles,
            "iv": iv_rows,
        },
        "timings": {
            "update_mean_ms": round(statistics.fmean(timings) * 1000, 3),
            "update_median_ms": round(statistics.median(timings) * 1000, 3),
            "update_max_ms": round(max(timings) * 1000, 3),
            "n": len(timings),
        },
        "cuda_peak_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------
def compare(golden: dict, current: dict) -> list[str]:
    """逐项比对金标与当前结果，返回人类可读的分歧列表（空 = 一致）。

    初始化与探针先比（更接近根因），逐帧只报每类第一处，避免一次改动画刷屏。
    """
    diffs: list[str] = []

    g_init, c_init = golden["init"], current["init"]
    for name in sorted(set(g_init["tensor_hashes"]) | set(c_init["tensor_hashes"])):
        a = g_init["tensor_hashes"].get(name, "<missing>")
        b = c_init["tensor_hashes"].get(name, "<missing>")
        if a != b:
            diffs.append(f"初始化张量 {name}: {a} != {b}")
    for name in sorted(set(g_init["scalars"]) | set(c_init["scalars"])):
        a = g_init["scalars"].get(name, "<missing>")
        b = c_init["scalars"].get(name, "<missing>")
        if a != b:
            diffs.append(f"初始化标量 {name}: {a} != {b}")
    if g_init["rng"] != c_init["rng"]:
        diffs.append(f"初始化后 RNG 状态: {g_init['rng']} != {c_init['rng']}")
    if g_init["config_sha"] != c_init["config_sha"]:
        diffs.append(
            f"应用档案后的 CONFIG 指纹: {g_init['config_sha']}(len={g_init['config_len']}) "
            f"!= {c_init['config_sha']}(len={c_init['config_len']})"
        )
    if g_init["profile_info_sha"] != c_init["profile_info_sha"]:
        diffs.append("PROFILE_INFO 指纹不一致")

    # 公开名字契约完整性也参与比对（obs-8）：任一方的契约不完整都判分歧——否则用残缺
    # 契约 dump 一次就能把缺失的 60 个张量/标量永久洗白成"新基线"。
    g_contract = golden.get("contract") or {}
    c_contract = current.get("contract") or {}
    if not c_contract.get("ok", True):
        diffs.append(
            f"当前运行的公开名字契约不完整（缺失 {len(c_contract.get('problems') or [])} 处）")
    if not g_contract.get("ok", True):
        diffs.append("金标文件的公开名字契约不完整（该金标不可作为比对基准，请重新采集）")

    g_frames, c_frames = golden["frames"], current["frames"]
    g_probe, c_probe = golden.get("probe", {}), current.get("probe", {})
    if g_probe.get("n_artists") != c_probe.get("n_artists"):
        diffs.append(f"探针 update(0) artist 数：{g_probe.get('n_artists')} != {c_probe.get('n_artists')}")
    if g_probe.get("all_have_axes") != c_probe.get("all_have_axes"):
        diffs.append(f"探针 artist 的 .axes 完整性：{g_probe.get('all_have_axes')} != {c_probe.get('all_have_axes')}")
    if g_probe.get("title") != c_probe.get("title"):
        diffs.append(
            f"探针 update(0) 标题分歧：\n  golden: {g_probe.get('title')!r}\n"
            f"  current: {c_probe.get('title')!r}"
        )
    if len(g_frames["counts"]) != len(c_frames["counts"]):
        diffs.append(
            f"记录帧数不同：{len(g_frames['counts'])} vs {len(c_frames['counts'])}"
        )
    for i, (a, b) in enumerate(zip(g_frames["counts"], c_frames["counts"])):
        if a != b:
            diffs.append(f"第 {i} 个记录帧 counts 分歧：{a} != {b}")
            break
    for i, (a, b) in enumerate(zip(g_frames["titles"], c_frames["titles"])):
        if a != b:
            diffs.append(f"第 {i} 个记录帧标题分歧：\n  golden: {a!r}\n  current: {b!r}")
            break
    for i, (a, b) in enumerate(zip(g_frames["iv"], c_frames["iv"])):
        if a != b:
            diffs.append(f"第 {i} 个记录帧 v3 计数器分歧：{a} != {b}")
            break
    return diffs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐位金标回归工具")
    parser.add_argument("--module", default=str(DEFAULT_MODULE), help="入口文件路径")
    parser.add_argument("--profile", default=None, help="疾病档案名（默认沿用 DISEASE_PROFILE 或 toy）")
    parser.add_argument("--population", type=int, default=None, help="人口（默认沿用 DISEASE_POPULATION 或 10000）")
    parser.add_argument("--seed", type=int, default=None, help="设置 DISEASE_SEED（默认不设置，沿用 CONFIG）")
    parser.add_argument("--frames", type=int, default=300, help="运行帧数")
    parser.add_argument("--stride", type=int, default=1, help="每多少帧记录一次（末帧总会记录）")
    parser.add_argument("--dump", default=None, help="把本次结果写入该 JSON")
    parser.add_argument("--compare", default=None, help="与给定 JSON 逐项比对")
    parser.add_argument("--contract", action="store_true", help="只做公开名字契约自检")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """按模式执行：--dump 落盘 / --compare 比对 / --contract 只查名字契约。

    返回值即退出码：0 = 一致或成功，1 = 有分歧，2 = 入口/金标文件加载失败。
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    module_path = Path(args.module).expanduser().resolve()
    if not module_path.is_file():
        print(f"入口文件不存在：{module_path}", file=sys.stderr)
        return 2

    env: dict[str, str] = {"MPLBACKEND": "Agg"}
    if args.profile:
        env["DISEASE_PROFILE"] = args.profile
    if args.population:
        env["DISEASE_POPULATION"] = str(args.population)
    if args.seed is not None:
        env["DISEASE_SEED"] = str(args.seed)

    try:
        module, load_s, import_stdout = load_module(module_path, env, verbose=args.verbose)
    except SystemExit as exc:
        print(f"模型导入时 sys.exit({exc.code})", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"模型导入失败：{exc!r}", file=sys.stderr)
        return 2

    ok, problems = verify_contract(module, dynamic=bool(args.contract))
    if args.contract:
        print(f"契约名字自检：{'PASS' if ok else 'FAIL'}")
        for line in problems:
            print(f"  - {line}")
        return 0 if ok else 1

    init = collect_init(module)
    if init["unclassified_tensors"]:
        print(
            "提示：以下张量未纳入指纹清单（请补进 CONTRACT_TENSORS）："
            + ", ".join(init["unclassified_tensors"]),
            file=sys.stderr,
        )
    if not ok:
        print("警告：公开名字契约不完整：", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)

    result = run_frames(module, args.frames, max(1, args.stride))
    result["init"] = init
    result["contract"] = {"ok": ok, "problems": problems}
    result["meta"] = {
        "module": str(module_path),
        "profile": init.get("profile_name"),
        "population": init.get("n_agents"),
        "frames": args.frames,
        "stride": max(1, args.stride),
        "device": init.get("device"),
        "load_s": round(load_s, 3),
        # 入口 import 期会打印档案摘要；截断到 2 KB 只为留证据，别让金标 JSON 膨胀。
        "import_stdout_tail": import_stdout[-2000:],
        "python": sys.version.split()[0],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(
        f"运行完成：{args.frames} 帧，profile={result['meta']['profile']}，"
        f"N={result['meta']['population']}，update mean={result['timings']['update_mean_ms']}ms，"
        f"CUDA peak={result['cuda_peak_mb']}MB"
    )
    final_counts = result["frames"]["counts"][-1]
    print(f"末帧 counts = {final_counts}")
    print(f"末帧标题 = {result['frames']['titles'][-1]!r}")

    if args.dump:
        # 拒绝把契约不完整的结果写成新基线（obs-8）：此时 rc=1，且不落盘。
        if not ok:
            print("拒绝 dump：公开名字契约不完整，先用 --contract 查缺失项再采集", file=sys.stderr)
            for line in problems:
                print(f"  - {line}", file=sys.stderr)
            return 1
        path = Path(args.dump)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"金标已写入：{path}")

    if args.compare:
        path = Path(args.compare)
        if not path.is_file():
            print(f"金标文件不存在：{path}", file=sys.stderr)
            return 2
        golden = json.loads(path.read_text(encoding="utf-8"))
        diffs = compare(golden, result)
        if not diffs:
            print(f"金标比对：PASS（{args.frames} 帧逐位一致）")
            return 0
        print(f"金标比对：FAIL（{len(diffs)} 处分歧）")
        for line in diffs:
            print(f"  - {line}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
