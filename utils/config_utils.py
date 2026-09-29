"""Configuration loading, composition, and object construction helpers."""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .dist_utils import is_main_process


_INCLUDE_KEY = "include"
_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


def load_json(pth):
    with open(pth, "r") as file:
        return json.load(file)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge mappings; later values and complete lists win."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment_string(value: str, *, location: str) -> str:
    """Expand ${NAME} and ${NAME:-default}, failing on missing variables."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name in os.environ:
            return os.environ[name]
        default = match.group("default")
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable {name!r} required by config {location} is not set"
        )

    return _ENV_PATTERN.sub(replace, value)


def _expand_environment(value: Any, *, location: str) -> Any:
    if isinstance(value, str):
        return _expand_environment_string(value, location=location)
    if isinstance(value, list):
        return [
            _expand_environment(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            key: _expand_environment(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    return value


def _normalize_includes(value: Any, *, path: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(
        f"Top-level {_INCLUDE_KEY!r} in {path} must be a path or list of paths"
    )


def _load_yaml_composed(path: Path, *, stack: tuple[Path, ...]) -> dict:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Config include cycle detected: {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Top-level YAML config must be a mapping: {resolved}")

    local = dict(loaded)
    include_entries = _normalize_includes(local.pop(_INCLUDE_KEY, None), path=resolved)
    composed: dict[str, Any] = {}
    next_stack = (*stack, resolved)
    for include_entry in include_entries:
        include_text = _expand_environment_string(
            include_entry, location=f"{resolved}:{_INCLUDE_KEY}"
        )
        include_path = Path(include_text).expanduser()
        if not include_path.is_absolute():
            include_path = resolved.parent / include_path
        composed = _deep_merge(
            composed,
            _load_yaml_composed(include_path, stack=next_stack),
        )
    return _deep_merge(composed, local)


def load_yaml_config(path, overrides: Iterable[str | os.PathLike[str]] | None = None):
    """Load a composed YAML configuration.

    A config may declare a top-level ``include`` path or ordered path list.
    Paths are resolved relative to the declaring YAML file. Included mappings
    are deep-merged in order, then the declaring file wins. Optional override
    files are merged last in command-line order.

    String values support ``${NAME}`` and ``${NAME:-default}``. Expansion occurs
    after composition so a local override can replace a machine-specific value
    without requiring the original environment variable.
    """
    root = Path(path)
    config = _load_yaml_composed(root, stack=())
    for override_path in overrides or ():
        config = _deep_merge(
            config,
            _load_yaml_composed(Path(override_path), stack=()),
        )
    return _expand_environment(config, location=str(root.expanduser().resolve()))


def dump_yaml_config(config, path):
    # Check if the file already exists; if so its resolved content must match.
    if os.path.exists(path):
        existing_config = load_yaml_config(path)
        if existing_config == config:
            return
        raise ValueError(
            f"Config at {path} already exists and is different. "
            "Please check the file or choose a different path."
        )
    if is_main_process():
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        if config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(obj_or_str, reload=False):
    if not isinstance(obj_or_str, str):
        return obj_or_str

    module, name = obj_or_str.rsplit(".", 1)
    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)
    return getattr(module_imp, name)
