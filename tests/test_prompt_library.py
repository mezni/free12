from pathlib import Path

import pytest

from resolve_agent.registry import PromptError, PromptLibrary, PromptVersion


def _write(root: Path, name: str, version: int, text: str) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"v{version}.md").write_text(text, encoding="utf-8")


def test_loads_versions_and_defaults_to_the_latest(tmp_path: Path) -> None:
    _write(tmp_path, "agent", 1, "old")
    _write(tmp_path, "agent", 3, "new")
    _write(tmp_path, "router", 1, "r")
    library = PromptLibrary.from_directory(tmp_path)
    assert library.get("agent").version == 3
    assert library.get("agent", 1).template == "old"
    assert [p.ref for p in library.all()] == ["agent@v1", "agent@v3", "router@v1"]


def test_unknown_prompt_or_version_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path, "agent", 1, "x")
    library = PromptLibrary.from_directory(tmp_path)
    with pytest.raises(PromptError, match="unknown prompt"):
        library.get("nope")
    with pytest.raises(PromptError, match="no version 2"):
        library.get("agent", 2)


def test_hash_tracks_content_and_ignores_line_endings() -> None:
    a = PromptVersion.from_text("p", 1, "line one\nline two\n")
    b = PromptVersion.from_text("p", 1, "line one\r\nline two\r\n")
    c = PromptVersion.from_text("p", 1, "line one\nline TWO\n")
    assert a.sha256 == b.sha256 != c.sha256
    assert a.full_id == f"p@v1+{a.sha256[:8]}"


def test_variables_are_discovered_and_dollar_escapes_are_literal() -> None:
    prompt = PromptVersion.from_text("p", 1, "Hi $name, you owe $${amount} and ${other}.")
    assert prompt.variables == {"name", "other"}
    rendered = prompt.render(name="Ada", other="x")
    assert rendered.text == "Hi Ada, you owe ${amount} and x."


def test_render_is_strict_about_variables() -> None:
    prompt = PromptVersion.from_text("p", 1, "Hello $name")
    with pytest.raises(PromptError, match="missing variables.*name"):
        prompt.render()
    with pytest.raises(PromptError, match="unexpected.*typo"):
        prompt.render(name="Ada", typo="x")
    assert prompt.render(name=42).text == "Hello 42"


def test_stray_dollar_signs_are_rejected_at_load() -> None:
    with pytest.raises(PromptError, match="invalid placeholder"):
        PromptVersion.from_text("p", 1, "Refunds over $50 need approval")


def test_bad_file_and_directory_names_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "agent", 1, "x")
    (tmp_path / "agent" / "v2.txt").write_text("x")
    with pytest.raises(PromptError, match="must be named"):
        PromptLibrary.from_directory(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _write(other, "Bad-Name", 1, "x")
    with pytest.raises(PromptError, match="invalid prompt name"):
        PromptLibrary.from_directory(other)
