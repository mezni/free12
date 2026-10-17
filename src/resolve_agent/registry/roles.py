"""Role configuration: which model, parameters and pinned prompt version each role uses."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RolesError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RoleConfig:
    name: str
    model: str
    prompt: str
    prompt_version: int
    max_tokens: int
    temperature: float
    timeout_s: float


_FIELDS = {"model", "prompt", "prompt_version", "max_tokens", "temperature", "timeout_s"}


def _number(role: str, key: str, value: Any, *, integer: bool) -> float:
    kind = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kind):
        raise RolesError(f"role {role!r}: {key} must be {'an integer' if integer else 'a number'}")
    return float(value)


def parse_roles(data: Mapping[str, Any]) -> dict[str, RoleConfig]:
    raw_roles = data.get("roles")
    if not isinstance(raw_roles, Mapping) or not raw_roles:
        raise RolesError("roles file must define at least one [roles.<name>] table")
    roles: dict[str, RoleConfig] = {}
    for name, raw in raw_roles.items():
        if not isinstance(raw, Mapping):
            raise RolesError(f"role {name!r} must be a table")
        missing, unknown = _FIELDS - raw.keys(), raw.keys() - _FIELDS
        if missing or unknown:
            raise RolesError(f"role {name!r}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        for key in ("model", "prompt"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise RolesError(f"role {name!r}: {key} must be a non-empty string")
        version = int(_number(name, "prompt_version", raw["prompt_version"], integer=True))
        max_tokens = int(_number(name, "max_tokens", raw["max_tokens"], integer=True))
        temperature = _number(name, "temperature", raw["temperature"], integer=False)
        timeout = _number(name, "timeout_s", raw["timeout_s"], integer=False)
        if version < 1:
            raise RolesError(f"role {name!r}: prompt_version must be >= 1")
        if max_tokens < 1:
            raise RolesError(f"role {name!r}: max_tokens must be >= 1")
        if not 0.0 <= temperature <= 1.0:
            raise RolesError(f"role {name!r}: temperature must be between 0 and 1")
        if timeout <= 0:
            raise RolesError(f"role {name!r}: timeout_s must be > 0")
        roles[name] = RoleConfig(
            name, raw["model"], raw["prompt"], version, max_tokens, temperature, timeout
        )
    return roles


def load_roles(path: Path) -> dict[str, RoleConfig]:
    try:
        with path.open("rb") as handle:
            return parse_roles(tomllib.load(handle))
    except tomllib.TOMLDecodeError as exc:
        raise RolesError(f"{path}: invalid TOML: {exc}") from exc


def packaged_roles_path() -> Path:
    return Path(__file__).parent / "roles.toml"
