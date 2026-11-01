"""疾病参数档案子包：schema -> loader/validate -> derive -> apply 的公共门面。

职责：re-export 子模块的常量、异常与函数；外部只需
``from disease_sim import profiles``（或旧的 ``import disease_profiles``）。
非职责：本文件不含逻辑——校验在 validate.py、派生在 derive.py、写 CONFIG 在 apply.py。
入口：``load_and_apply(name, config)`` 一次完成"读 JSON -> 校验 -> 派生 -> 写 CONFIG"。
不变式（re-export 契约）：``__all__`` 中的名字是对外 API，其中以下划线开头的私有名是
历史遗留（旧 shim 与 tests 直接引用），内部子模块重构不得删除/改名/改签名；
子模块新增对外名字时，必须同步本文件、``__all__`` 与根级 ``disease_profiles.py``。
"""
from __future__ import annotations

from .schema import (  # noqa: F401
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
    _INTERVENTION_KEYS,
    ProfileError,
)
from .loader import (  # noqa: F401
    available_profiles,
    resolve_name,
    load_profile,
)
from .validate import (  # noqa: F401
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
    validate_profile,
)
from .derive import (  # noqa: F401
    _interp_curve_value,
    _integrate_curve,
    _age_table_or_default,
    _derive_natural_history,
    _derive_interventions,
    derive_model_parameters,
)
from .apply import (  # noqa: F401
    _apply_interventions,
    _validate_applied,
    _apply_derived,
    apply_profile,
    load_and_apply,
    format_summary,
)

__all__ = [
    "PROFILE_DIR",
    "DEFAULT_PROFILE",
    "SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "FRAMES_PER_DAY_KEY",
    "CONTACT_RATE_PER_FRAME",
    "TRANSMIT_BETA_EXPONENT",
    "DEATH_HAZARD_FRACTION",
    "ALLOWED_OVERRIDES",
    "AGE_BANDS",
    "LEGACY_INFECTIVITY_CURVE",
    "LEGACY_ASYMPTOMATIC_CURVE",
    "NATURAL_HISTORY_DEFAULTS",
    "INTERVENTIONS_DEFAULTS",
    "ProfileError",
    "available_profiles",
    "resolve_name",
    "load_profile",
    "validate_profile",
    "derive_model_parameters",
    "apply_profile",
    "load_and_apply",
    "format_summary",
    "_INTERVENTION_KEYS",
    "_field",
    "_number",
    "_value_entry",
    "_validate_epidemiology",
    "_validate_bounds",
    "_validate_model_parameters",
    "_validate_curve",
    "_validate_age_table",
    "_validate_natural_history",
    "_validate_medical",
    "_validate_interventions",
    "_validate_overrides",
    "_interp_curve_value",
    "_integrate_curve",
    "_age_table_or_default",
    "_derive_natural_history",
    "_derive_interventions",
    "_apply_interventions",
    "_validate_applied",
    "_apply_derived",
]
