"""Versioned prompts: files on disk, immutable once released, rendered strictly.

Layout: library/<name>/v<N>.md. Templates use `string.Template` placeholders ($name or
${name}); write a literal dollar sign as `$$`. Only trusted, system-side values belong in
placeholders: customer text goes in user messages, never into a system prompt.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from string import Template

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_FILE = re.compile(r"^v(\d+)\.md$")


class PromptError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    full_id: str


@dataclass(frozen=True, slots=True)
class PromptVersion:
    name: str
    version: int
    template: str
    sha256: str
    variables: frozenset[str]

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"

    @property
    def full_id(self) -> str:
        """Stamped on every LLM span: version for humans, hash to catch silent edits."""
        return f"{self.ref}+{self.sha256[:8]}"

    @classmethod
    def from_text(cls, name: str, version: int, raw: str) -> PromptVersion:
        text = raw.replace("\r\n", "\n")  # same hash on every OS
        template = Template(text.strip("\n"))
        if not template.is_valid():
            raise PromptError(f"{name}@v{version}: invalid placeholder (use $$ for a literal $)")
        return cls(
            name=name,
            version=version,
            template=template.template,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            variables=frozenset(template.get_identifiers()),
        )

    def render(self, **values: object) -> RenderedPrompt:
        missing = self.variables - values.keys()
        extra = values.keys() - self.variables
        if missing or extra:
            raise PromptError(
                f"{self.ref}: missing variables {sorted(missing)}, unexpected {sorted(extra)}"
            )
        text = Template(self.template).substitute({k: str(v) for k, v in values.items()})
        return RenderedPrompt(text, self.full_id)


class PromptLibrary:
    def __init__(self, versions: Iterable[PromptVersion]) -> None:
        self._by_name: dict[str, dict[int, PromptVersion]] = {}
        for prompt in versions:
            self._by_name.setdefault(prompt.name, {})[prompt.version] = prompt

    @classmethod
    def from_directory(cls, root: Path) -> PromptLibrary:
        versions: list[PromptVersion] = []
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            if not _NAME.match(directory.name):
                raise PromptError(f"invalid prompt name: {directory.name!r}")
            for path in sorted(directory.iterdir()):
                match = _FILE.match(path.name)
                if match is None:
                    raise PromptError(f"{path}: prompt files must be named v<N>.md")
                versions.append(
                    PromptVersion.from_text(
                        directory.name, int(match.group(1)), path.read_text(encoding="utf-8")
                    )
                )
        return cls(versions)

    def get(self, name: str, version: int | None = None) -> PromptVersion:
        versions = self._by_name.get(name)
        if not versions:
            raise PromptError(f"unknown prompt {name!r}")
        if version is None:
            return versions[max(versions)]
        if version not in versions:
            raise PromptError(f"prompt {name!r} has no version {version}")
        return versions[version]

    def all(self) -> list[PromptVersion]:
        return [p for name in sorted(self._by_name) for _, p in sorted(self._by_name[name].items())]


def packaged_library_dir() -> Path:
    return Path(__file__).parent / "library"
