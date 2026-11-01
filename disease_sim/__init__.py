"""仿真包：围绕显式 :class:`SimState` 容器装配整个模拟。

装配顺序（顺序即语义）：:func:`build_simulation` 先 ``config.init_session``（应用档案 +
人口缩放，必须在建 state 之前，否则会用默认参数建人群），再 ``build_state``（建人群/张量），
最后 ``build_render``（建 matplotlib 图形）。

失败路径不留下全局注册：档案非法时 init_session 打印原因后 ``sys.exit(1)``；
build_state/build_render 抛异常时 sim 直接丢弃即可，无需额外清理。

``CONFIG`` 是原地变更的普通 dict，外部代码（tools、headless harness）始终看到同一个对象；
本模块重导出常用符号，``from disease_sim import CONFIG, build_simulation`` 是工具层标准入口。
"""

from . import activity, cli, config, contacts, disease, interventions, profiles, render, simulation, state
from .config import ACTIVE_PROFILE_NAME, CONFIG, PROFILE_INFO, SEED, device, init_session
from .render import build_render
from .state import SimState, build_state

__all__ = [
    "ACTIVE_PROFILE_NAME",
    "CONFIG",
    "PROFILE_INFO",
    "SEED",
    "SimState",
    "activity",
    "build_render",
    "build_simulation",
    "build_state",
    "cli",
    "config",
    "contacts",
    "device",
    "disease",
    "init_session",
    "interventions",
    "profiles",
    "render",
    "simulation",
    "state",
]


def build_simulation(argv: list[str] | None = None) -> SimState:
    """装配一次完整会话：init_session → build_state → build_render，返回 SimState。

    入口脚本与工具（headless_check / render_prof）复用的唯一装配路径；argv 交给档案
    解析器（None 时读 sys.argv[1:]）。同进程重复调用可切换档案，但进程级单例（如
    disease 的曲线 LUT 缓存）不会重建，见 disease._LUT_CACHE 的说明。
    """
    init_session(argv)
    sim = build_state()
    build_render(sim)
    return sim
