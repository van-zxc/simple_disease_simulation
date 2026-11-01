"""分析工具共用的加载器：Agg 打桩 + 按环境变量加载入口脚本。

研究脚本（bench/diag/ensemble/ab_iter/sweep…）需要在不弹窗的情况下驱动模拟。
入口在 import 期就会建 fig / FuncAnimation，所以必须先打桩再加载。

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


def patch_matplotlib() -> None:
    """禁止 import 期创建真实动画或阻塞在 plt.show（Agg + 动画打桩）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt

    class _DummyFuncAnimation:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def save(self, *args, **kwargs):
            return None

    mpl_animation.FuncAnimation = _DummyFuncAnimation  # type: ignore[assignment]
    plt.show = lambda *a, **k: None  # type: ignore[assignment]


def load_module(path: Path, env: dict[str, str], verbose: bool = False):
    """注入环境变量 → 打桩 → exec_module，返回 (module, 加载秒数, stdout)。

    ``path`` 必须以 .py 结尾（``spec_from_file_location`` 对 .bak 返回 None）。
    同一进程不要先后加载两个入口（第二个会命中 sys.modules 里的包），
    需要对比两个版本请用 ab_iter.py 起子进程。
    """
    for key, value in env.items():
        os.environ[key] = value

    patch_matplotlib()

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
