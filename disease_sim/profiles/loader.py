"""档案定位与读取：扫描 diseases/、解析档案名、载入并校验 JSON。

职责：发现可用档案（文件名即 profile_id）、按优先级选名、读文件、校验结构与字段。
非职责：不派生参数、不写 CONFIG（见 derive.py / apply.py）；不做字段级校验细节。
入口：``load_profile(name)`` 读单个档案；``load_and_apply``（apply.py）串起完整链路。
不变式：选名优先级固定为 CLI ``--profile`` > 环境变量 ``DISEASE_PROFILE`` > ``DEFAULT_PROFILE``；
返回的是通过 ``validate_profile`` 的"原始 JSON"，不是派生后的 CONFIG。
"""

from __future__ import annotations

import json
import os

from .schema import DEFAULT_PROFILE, PROFILE_DIR, ProfileError
from .validate import validate_profile


def available_profiles() -> list[str]:
    """列出 ``diseases/*.json`` 的 profile_id（文件名即档案名），按字母排序。

    目录不存在时返回空列表，把"没有任何档案"的报错留给调用方。
    """
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def resolve_name(argv, env=None) -> str:
    """解析本次要加载的档案名：CLI ``--profile`` > 环境变量 > 默认档案。

    ``argv`` 应为不含程序名的参数列表；``env`` 默认取 ``os.environ``。
    返回去空白后的档案名，``--profile`` 缺值或为空时抛 ``ProfileError``。
    不变式：只扫描 ``--profile`` / ``--profile=``，绝不把 argv 里其它 flag 的值
    当成档案名——调用方常把工具自己的 ``sys.argv[1:]`` 原样传进来。
    """
    env = os.environ if env is None else env
    cli_name = None
    args = list(argv)
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--profile":
            if i + 1 >= len(args):
                raise ProfileError("--profile 需要一个档案名，例如 --profile covid19")
            cli_name = args[i + 1]
            break
        if arg.startswith("--profile="):
            cli_name = arg.split("=", 1)[1]
            break
        i += 1

    if cli_name is not None and not str(cli_name).strip():
        raise ProfileError("--profile 的档案名不能为空")

    env_name = env.get("DISEASE_PROFILE", "")
    name = cli_name if cli_name is not None else (env_name or "")
    name = str(name).strip()
    return name or DEFAULT_PROFILE




def load_profile(name: str) -> dict:
    """按 profile_id 读取并校验一个档案，返回原始 dict。

    未知/空档案名抛 ``ProfileError`` 并列出可用档案；JSON 或 IO 错误包装为
    ``ProfileError``；字段校验交给 ``validate_profile``（失败同样抛它）。
    不变式：``profile_id`` 必须与文件名一致（``expected_id=name``），防止档案内容
    与文件名错配；本函数不修改文件、不做缓存，也不返回派生后的参数。
    """
    name = str(name).strip()
    available = available_profiles()
    if not name:
        raise ProfileError("未指定疾病档案")
    if name not in available:
        hint = ", ".join(available) if available else "(diseases/ 目录为空)"
        raise ProfileError(f"未知疾病档案 '{name}'；可选档案: {hint}")

    path = PROFILE_DIR / f"{name}.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"档案 {path.name} 不是合法 JSON: {exc}") from exc
    except OSError as exc:
        raise ProfileError(f"无法读取档案 {path}: {exc}") from exc

    if not isinstance(profile, dict):
        raise ProfileError(f"档案 {path.name} 的顶层必须是 JSON 对象")
    validate_profile(profile, expected_id=name)
    return profile
