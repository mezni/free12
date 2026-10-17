import copy
from pathlib import Path
from typing import Any

import pytest

from resolve_agent.registry.roles import RolesError, load_roles, parse_roles

VALID: dict[str, Any] = {
    "roles": {
        "router": {
            "model": "m",
            "prompt": "router",
            "prompt_version": 1,
            "max_tokens": 100,
            "temperature": 0.0,
            "timeout_s": 2,
        }
    }
}


def _with(**changes: object) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    data["roles"]["router"].update(changes)
    return data


def test_valid_roles_parse() -> None:
    role = parse_roles(VALID)["router"]
    assert (role.model, role.prompt, role.prompt_version) == ("m", "router", 1)
    assert (role.max_tokens, role.temperature, role.timeout_s) == (100, 0.0, 2.0)


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"prompt_version": 0}, "prompt_version must be >= 1"),
        ({"prompt_version": "1"}, "prompt_version must be an integer"),
        ({"prompt_version": True}, "prompt_version must be an integer"),
        ({"max_tokens": 0}, "max_tokens must be >= 1"),
        ({"temperature": 1.5}, "temperature must be between 0 and 1"),
        ({"temperature": "hot"}, "temperature must be a number"),
        ({"timeout_s": 0}, "timeout_s must be > 0"),
        ({"model": ""}, "model must be a non-empty string"),
        ({"surprise": 1}, "unknown \\['surprise'\\]"),
    ],
)
def test_invalid_values_are_rejected(changes: dict[str, object], fragment: str) -> None:
    with pytest.raises(RolesError, match=fragment):
        parse_roles(_with(**changes))


def test_missing_fields_and_empty_files_are_rejected() -> None:
    data = _with()
    del data["roles"]["router"]["model"]
    with pytest.raises(RolesError, match="missing \\['model'\\]"):
        parse_roles(data)
    with pytest.raises(RolesError, match="at least one"):
        parse_roles({})


def test_invalid_toml_is_reported_with_the_path(tmp_path: Path) -> None:
    path = tmp_path / "roles.toml"
    path.write_text("[roles.router\nmodel = ")
    with pytest.raises(RolesError, match="invalid TOML"):
        load_roles(path)
