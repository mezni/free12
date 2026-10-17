"""Release lock: once a prompt version is locked, its content can never change.

    uv run python -m resolve_agent.registry.lock check    # CI: fails on edits or unlocked files
    uv run python -m resolve_agent.registry.lock update   # release new versions (never rewrites)

Changing a released prompt means creating v<N+1> and pointing roles.toml at it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from resolve_agent.registry.prompts import PromptLibrary, packaged_library_dir


def packaged_lock_path() -> Path:
    return Path(__file__).parent / "prompts.lock.json"


def compute(library: PromptLibrary) -> dict[str, str]:
    return {p.ref: p.sha256 for p in library.all()}


def read_lock(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def check(library: PromptLibrary, lock: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    current = compute(library)
    for ref, sha in current.items():
        if ref not in lock:
            problems.append(f"{ref} is not released yet (run `lock update` to release it)")
        elif lock[ref] != sha:
            problems.append(f"{ref} changed after release: create a new version instead")
    for ref in lock:
        if ref not in current:
            problems.append(f"{ref} was released but its file is missing")
    return problems


def update(library: PromptLibrary, lock: Mapping[str, str]) -> dict[str, str]:
    """Add unreleased versions. Refuses to touch anything that is already released."""
    broken = [p for p in check(library, lock) if "changed after release" in p or "missing" in p]
    if broken:
        raise ValueError("; ".join(broken))
    return {**lock, **{k: v for k, v in compute(library).items() if k not in lock}}


def write_lock(path: Path, lock: Mapping[str, str]) -> None:
    path.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    library_dir: Path | None = None,
    lock_path: Path | None = None,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in ("check", "update"):
        print("usage: python -m resolve_agent.registry.lock check|update", file=sys.stderr)
        return 2
    library = PromptLibrary.from_directory(library_dir or packaged_library_dir())
    path = lock_path or packaged_lock_path()
    lock = read_lock(path)
    if args[0] == "check":
        problems = check(library, lock)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    try:
        write_lock(path, update(library, lock))
    except ValueError as exc:
        print(f"refusing to update: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
