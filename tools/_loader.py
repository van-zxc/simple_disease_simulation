"""分析工具共用的加载器：无头打桩 + 按环境变量加载入口脚本。

研究脚本（bench/diag/ensemble/ab_iter/sweep…）需要在不启仪表盘的情况下驱动模拟。
入口在 import 期只做 ``build_simulation()``（不启动服务），所以这里只需设好
``DISEASE_VIEW=headless`` 让 ``cli`` 走无 UI 路径；不再需要打桩 matplotlib。

用法：
    import _loader
    module, load_s, _ = _loader.load_module(ROOT / "simple_disease_simulation.py", env)
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_headless() -> None:
    """无头模式：让入口的 ``cli.main`` 走"不启仪表盘、跑完即退"的分支。

    在 import 前设置环境变量（入口加载期不启动服务，这里主要是给可能调用 cli.main 的脚本兜底）。
    """
    os.environ["DISEASE_VIEW"] = "headless"


# 兼容旧名字：以前这个函数用来给 matplotlib 打桩，界面换成本地 Web 后只剩无头开关。
patch_matplotlib = patch_headless


def load_module(path: Path, env: dict[str, str], verbose: bool = False):
    """注入环境变量 → 无头打桩 → exec_module，返回 (module, 加载秒数, stdout)。

    ``path`` 必须以 .py 结尾（``spec_from_file_location`` 对 .bak 返回 None）。
    同一进程不要先后加载两个入口（第二个会命中 sys.modules 里的包），
    需要对比两个版本请用 ab_iter.py 起子进程。
    """
    for key, value in env.items():
        os.environ[key] = value

    patch_headless()

    spec = importlib.util.spec_from_file_location("sim_runtime", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim_runtime"] = module

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
