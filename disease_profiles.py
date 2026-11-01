"""兼容 shim（旧名，已不推荐新代码使用）：把旧的模块级 API 全部转发到 :mod:`disease_sim.profiles`。

职责：让 ``import disease_profiles`` / ``from disease_profiles import ...`` 的老工具
（tools/profile_check.py 等）和旧脚本无需改动即可继续工作；新代码请直接用
``from disease_sim import profiles``。
非职责：本文件不含任何逻辑；新增或删除公共名字时要与
``disease_sim/profiles/__init__.py`` 的 ``__all__`` 两边同步（本文件无 ``__all__``）。
不变式：仓库根被插到 ``sys.path`` 最前，保证在任意 CWD 下都能导入子包。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from disease_sim.profiles import (  # noqa: E402,F401
    PROFILE_DIR,
    DEFAULT_PROFILE,
    SCHEMA_VERSION,
    SCHEMA_VERSIONS,
    FRAMES_PER_DAY_KEY,
    CONTACT_RATE_PER_FRAME,
    TRANSMIT_BETA_EXPONENT,
    DEATH_HAZARD_FRACTION,
    ALLOWED_OVERRIDES,
    AGE_BANDS,
    LEGACY_INFECTIVITY_CURVE,
    LEGACY_ASYMPTOMATIC_CURVE,
    NATURAL_HISTORY_DEFAULTS,
    INTERVENTIONS_DEFAULTS,
    ProfileError,
    available_profiles,
    resolve_name,
    load_profile,
    validate_profile,
    derive_model_parameters,
    apply_profile,
    load_and_apply,
    format_summary,
    _INTERVENTION_KEYS,
    _field,
    _number,
    _value_entry,
    _validate_epidemiology,
    _validate_bounds,
    _validate_model_parameters,
    _validate_curve,
    _validate_age_table,
    _validate_natural_history,
    _validate_medical,
    _validate_interventions,
    _validate_overrides,
    _interp_curve_value,
    _integrate_curve,
    _age_table_or_default,
    _derive_natural_history,
    _derive_interventions,
    _apply_interventions,
    _validate_applied,
    _apply_derived,
)
