"""Dataset-agnostic validation helpers for configuration and cache contracts."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


def require_int(value: Any, name: str) -> int:
    """Validate an integer without lossy coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def require_positive_int(value: Any, name: str) -> int:
    """Validate a strictly positive integer without lossy coercion."""
    resolved = require_int(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return resolved


def require_bool(value: Any, name: str) -> bool:
    """Reject truthy strings/numbers instead of using lossy ``bool(value)``."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return bool(value)
