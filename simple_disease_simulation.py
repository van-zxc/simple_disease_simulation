"""疾病模拟入口：实现位于 :mod:`disease_sim`，本文件只负责装配与启动。

用法::

    python simple_disease_simulation.py
    python simple_disease_simulation.py --profile covid19 --population 100000 --seed 42
    python simple_disease_simulation.py --population 100000 --initial-seeds 200

分析工具（``tools/``）会加载本文件并读取模块级名字（``update`` / ``status`` /
``CONFIG`` 等），所以这里保留一层很薄的属性转发；仿真逻辑一律在包里。
"""

import os
import sys
from pathlib import Path

# 工具用 spec_from_file_location 加载本文件时 __package__ 为空，
# 需要仓库根在 sys.path 里才能 ``from disease_sim import ...``。
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _apply_cli_overrides() -> None:
    """把 CLI 参数翻译成包内环境变量；只在直接执行本文件时生效。

    支持 ``--population`` / ``--seed`` / ``--frames`` / ``--initial-seeds``（含 ``=`` 写法）。
    ``--initial-seeds`` 是**绝对人数**（不随人口缩放，``0`` = 零初始感染），对应
    ``CONFIG["INITIAL_SEEDS_ABSOLUTE"]``。
    只有 ``sys.argv[0]`` 是本文件时才解析——工具加载本模块时 argv 属于工具，
    把它的 flag 当模拟参数会污染配置。
    """
    argv = sys.argv
    if not argv or os.path.basename(argv[0]) != os.path.basename(__file__):
        return
    mapping = {
        "--population": "DISEASE_POPULATION",
        "--seed": "DISEASE_SEED",
        "--frames": "SIM_FRAMES",
        "--initial-seeds": "DISEASE_INITIAL_SEEDS",
    }
    args = argv[1:]
    for index, arg in enumerate(args):
        for flag, env_name in mapping.items():
            value = None
            if arg == flag and index + 1 < len(args):
                value = args[index + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            if value is None:
                continue
            value = str(value).strip()
            if value:
                os.environ[env_name] = value
            break


_apply_cli_overrides()

import torch  # noqa: E402,F401  分析工具习惯从入口取 torch

from disease_sim import build_simulation, config, profiles, simulation  # noqa: E402

_SIM = build_simulation()


def update(frame):
    """推进一帧，返回本帧统计 dict（浏览器仪表盘/分析工具消费）。"""
    return simulation.update(_SIM, frame)


def toggle(event=None):
    """封控开关（浏览器按钮 → POST /api/toggle）；返回新的封控状态。"""
    return _SIM.toggle(event)


# 分析工具读取的模块级名字；用 lambda 实时取值，避免标量快照过期。
_MODULE_EXPORTS = {
    "CONFIG": lambda: config.CONFIG,
    "device": lambda: config.device,
    "SEED": lambda: config.SEED,
    "PROFILE_INFO": lambda: config.PROFILE_INFO,
    "ACTIVE_PROFILE_NAME": lambda: config.ACTIVE_PROFILE_NAME,
    "profiles": lambda: profiles,
    "update": lambda: update,
    "toggle": lambda: toggle,
}


def __getattr__(name):
    """先查 ``_MODULE_EXPORTS``，再转发到 ``_SIM`` 的同名属性。"""
    export = _MODULE_EXPORTS.get(name)
    if export is not None:
        return export()
    sim = globals().get("_SIM")
    if sim is not None:
        try:
            return getattr(sim, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from disease_sim import cli

    cli.main(_SIM)
