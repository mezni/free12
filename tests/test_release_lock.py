from pathlib import Path

from resolve_agent.registry import PromptLibrary
from resolve_agent.registry.lock import check, main, read_lock, update


def _library(root: Path, **files: str) -> PromptLibrary:
    for key, text in files.items():
        name, version = key.rsplit("_v", 1)
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"v{version}.md").write_text(text)
    return PromptLibrary.from_directory(root)


def test_unreleased_versions_are_flagged_until_locked(tmp_path: Path) -> None:
    library = _library(tmp_path, agent_v1="a")
    assert "not released" in check(library, {})[0]
    assert check(library, update(library, {})) == []


def test_editing_a_released_prompt_is_flagged(tmp_path: Path) -> None:
    lock = update(_library(tmp_path, agent_v1="original"), {})
    edited = _library(tmp_path, agent_v1="edited")
    assert "changed after release" in check(edited, lock)[0]


def test_deleting_a_released_prompt_is_flagged(tmp_path: Path) -> None:
    lock = update(_library(tmp_path, agent_v1="a", router_v1="r"), {})
    (tmp_path / "router" / "v1.md").unlink()
    assert any("file is missing" in p for p in check(PromptLibrary.from_directory(tmp_path), lock))


def test_a_new_version_is_added_without_touching_released_ones(tmp_path: Path) -> None:
    lock = update(_library(tmp_path, agent_v1="a"), {})
    library = _library(tmp_path, agent_v2="b")
    new_lock = update(library, lock)
    assert new_lock["agent@v1"] == lock["agent@v1"] and "agent@v2" in new_lock


def test_cli_check_and_update(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    lib, lock_path = tmp_path / "lib", tmp_path / "lock.json"
    _library(lib, agent_v1="a")
    assert main(["check"], library_dir=lib, lock_path=lock_path) == 1
    assert "not released" in capsys.readouterr().err
    assert main(["update"], library_dir=lib, lock_path=lock_path) == 0
    assert main(["check"], library_dir=lib, lock_path=lock_path) == 0
    assert read_lock(lock_path).keys() == {"agent@v1"}

    (lib / "agent" / "v1.md").write_text("tampered")
    assert main(["check"], library_dir=lib, lock_path=lock_path) == 1
    assert main(["update"], library_dir=lib, lock_path=lock_path) == 1
    assert "refusing to update" in capsys.readouterr().err
    assert read_lock(lock_path)["agent@v1"] != ""  # lock untouched
    assert main([], library_dir=lib, lock_path=lock_path) == 2
