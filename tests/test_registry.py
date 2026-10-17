import re
from pathlib import Path

import pytest

from resolve_agent.config import Settings
from resolve_agent.llm import FakeLLM, Message, ObservedLLMClient, text_response
from resolve_agent.obs import InMemoryExporter
from resolve_agent.registry import PromptLibrary, Registry, RolesError
from resolve_agent.registry.lock import check, packaged_lock_path, read_lock
from resolve_agent.registry.prompts import packaged_library_dir


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def test_packaged_registry_has_the_expected_roles(registry: Registry) -> None:
    assert [r.name for r in registry.roles()] == ["agent", "judge", "router"]


def test_request_is_built_from_the_role_and_stamps_the_prompt_version(registry: Registry) -> None:
    request = registry.request(
        "agent",
        messages=[Message.user("where is my order?")],
        variables={"store_name": "Acme", "today": "2026-10-16"},
        tools=[{"name": "t"}],
    )
    role = registry.role("agent")
    prompt = registry.prompt_for("agent")
    assert request.model == role.model
    assert (request.max_tokens, request.temperature, request.timeout_s) == (1024, 0.2, 8.0)
    assert request.purpose == "agent"
    assert request.prompt_version == prompt.full_id
    assert re.fullmatch(r"agent@v1\+[0-9a-f]{8}", request.prompt_version or "")
    assert "Acme" in request.system and "2026-10-16" in request.system
    assert "$store_name" not in request.system
    assert list(request.tools) == [{"name": "t"}]


def test_per_call_overrides(registry: Registry) -> None:
    request = registry.request(
        "router", messages=[Message.user("hi")], max_tokens=50, timeout_s=1.0
    )
    assert (request.max_tokens, request.timeout_s) == (50, 1.0)


def test_unknown_role_lists_the_known_ones(registry: Registry) -> None:
    with pytest.raises(RolesError, match="known roles"):
        registry.role("nope")


def test_missing_variables_fail_before_any_llm_call(registry: Registry) -> None:
    with pytest.raises(Exception, match="missing variables"):
        registry.request("agent", messages=[Message.user("hi")])


def test_a_role_pointing_at_a_missing_prompt_fails_at_load(tmp_path: Path) -> None:
    roles = tmp_path / "roles.toml"
    roles.write_text(
        "[roles.router]\nmodel='m'\nprompt='router'\nprompt_version=9\n"
        "max_tokens=10\ntemperature=0.0\ntimeout_s=1\n"
    )
    with pytest.raises(RolesError, match="no version 9"):
        Registry.load(roles_path=roles)


def test_roles_path_from_settings_selects_another_environment(tmp_path: Path) -> None:
    roles = tmp_path / "staging.toml"
    roles.write_text(
        "[roles.router]\nmodel='tiny-model'\nprompt='router'\nprompt_version=1\n"
        "max_tokens=10\ntemperature=0.0\ntimeout_s=1\n"
    )
    registry = Registry.load(Settings.from_env({"RESOLVE_ROLES_PATH": str(roles)}))
    assert registry.role("router").model == "tiny-model"
    assert [r.name for r in registry.roles()] == ["router"]


# --- packaged content ----------------------------------------------------------------------


def test_every_packaged_prompt_renders_with_dummy_values() -> None:
    for prompt in PromptLibrary.from_directory(packaged_library_dir()).all():
        text = prompt.render(**{v: "x" for v in prompt.variables}).text
        assert text and "$" not in text.replace("x", ""), prompt.ref


def test_policy_numbers_live_in_code_not_in_prompts() -> None:
    for prompt in PromptLibrary.from_directory(packaged_library_dir()).all():
        assert not re.search(r"\b(30|50)\b", prompt.template), prompt.ref


def test_prompts_do_not_put_customer_text_in_placeholders() -> None:
    forbidden = {"message", "customer_message", "user_message", "input", "query"}
    for prompt in PromptLibrary.from_directory(packaged_library_dir()).all():
        assert not prompt.variables & forbidden, prompt.ref


def test_release_lock_is_current() -> None:
    library = PromptLibrary.from_directory(packaged_library_dir())
    assert check(library, read_lock(packaged_lock_path())) == []


def test_production_roles_only_use_released_prompts(registry: Registry) -> None:
    lock = read_lock(packaged_lock_path())
    for role in registry.roles():
        assert f"{role.prompt}@v{role.prompt_version}" in lock, role.name


def test_the_prompt_version_reaches_the_llm_span(
    registry: Registry, spans: InMemoryExporter
) -> None:
    client = ObservedLLMClient(FakeLLM([text_response('{"intent": "return"}')]))
    request = registry.request("router", messages=[Message.user("I want to return my shoes")])
    client.complete(request)
    attrs = spans.spans[0].attributes
    assert attrs["llm.prompt_version"] == registry.prompt_for("router").full_id
    assert attrs["llm.purpose"] == "router"
