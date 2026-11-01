"""疾病模拟的历史入口（thin wrapper）：实现已迁到 :mod:`disease_sim`。

职责：保留旧的模块级接口与 ``python simple_disease_simulation.py`` 用法，让
tools/headless_check.py、tools/golden_check.py 仍能直接加载本文件并读取
``status`` / ``fig`` / ``update`` 等属性；实际逻辑在 :class:`disease_sim.state.SimState`。
非职责：本文件不含仿真逻辑，也不复制状态；名字经 ``__getattr__`` 转发到 ``_SIM``。
入口：``python simple_disease_simulation.py [--profile NAME] [--population N] [--seed N]``。
不变式：``_MODULE_EXPORTS`` 用 lambda 实时取值——标量（如 ``tests_today``、``SEED``）
会被重新赋值，静态副本会过期；未登记的属性兜底取 ``_SIM`` 上的同名属性。
"""

import os
import sys
from pathlib import Path

# harness 契约：headless_check/golden_check 用 spec_from_file_location 加载本文件时
# ``__package__`` 为空，``from disease_sim import ...`` 需要仓库根在 sys.path 里。
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _apply_cli_overrides() -> None:
    """把用户面 CLI 参数翻译成包内环境变量；仅在直接执行本文件时生效。

    支持 ``--population`` / ``--seed`` / ``--frames``（含 ``=`` 写法），映射到
    ``DISEASE_POPULATION`` / ``DISEASE_SEED`` / ``SIM_FRAMES``，CLI 覆盖环境变量。
    不变式：只有 ``sys.argv[0]`` 的 basename 等于本文件 basename 才解析——工具加载
    本模块时 argv 属于工具，把它的 flag 当模拟参数会污染配置。
    """
    argv = sys.argv
    if not argv or os.path.basename(argv[0]) != os.path.basename(__file__):
        return
    mapping = {
        "--population": "DISEASE_POPULATION",
        "--seed": "DISEASE_SEED",
        "--frames": "SIM_FRAMES",
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

import torch  # noqa: E402  harness 契约：torch 保留为模块级属性，供工具读取

from disease_sim import build_simulation, config, profiles, simulation  # noqa: E402

_SIM = build_simulation()


def update(frame):
    """推进一帧（matplotlib FuncAnimation 回调），返回需要重绘的 artists。"""
    return simulation.update(_SIM, frame)


def toggle(event):
    """锁定按钮回调；保留历史的一参签名，兼容 matplotlib 的 event 传参。"""
    return _SIM.toggle(event)


# harness 契约：工具读取这些模块级属性；用 lambda 实时取值而非快照，因为标量
# （SEED / ACTIVE_PROFILE_NAME 等）运行期会被改写；"disease_profiles" 是旧名，
# 指向 disease_sim.profiles 子包。
_MODULE_EXPORTS = {
    "CONFIG": lambda: config.CONFIG,
    "device": lambda: config.device,
    "SEED": lambda: config.SEED,
    "PROFILE_INFO": lambda: config.PROFILE_INFO,
    "ACTIVE_PROFILE_NAME": lambda: config.ACTIVE_PROFILE_NAME,
    "disease_profiles": lambda: profiles,
    "update": lambda: update,
    "toggle": lambda: toggle,
}


def __getattr__(name):
    """按需转发模块级属性：先查 ``_MODULE_EXPORTS``，再取 ``_SIM`` 的同名属性。

    两处都取不到时抛 ``AttributeError``（保持模块属性语义，兼容 hasattr 等探测）。
    """
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
    # 注意：cli 在 __main__ 内才导入，模块被工具加载时不解析命令行、不建窗口。
    from disease_sim import cli

    cli.main(_SIM)
