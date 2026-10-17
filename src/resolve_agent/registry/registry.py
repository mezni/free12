"""The registry: roles + prompts -> ready-to-send LLMRequests, validated at startup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from resolve_agent.config import Settings
from resolve_agent.llm.types import LLMRequest, Message
from resolve_agent.registry.prompts import (
    PromptError,
    PromptLibrary,
    PromptVersion,
    packaged_library_dir,
)
from resolve_agent.registry.roles import RoleConfig, RolesError, load_roles, packaged_roles_path


class Registry:
    def __init__(self, library: PromptLibrary, roles: Mapping[str, RoleConfig]) -> None:
        self._library = library
        self._roles = dict(roles)
        for role in self._roles.values():  # fail at startup, not on the first customer message
            try:
                library.get(role.prompt, role.prompt_version)
            except PromptError as exc:
                raise RolesError(f"role {role.name!r}: {exc}") from exc

    @classmethod
    def load(
        cls,
        settings: Settings | None = None,
        *,
        library_dir: Path | None = None,
        roles_path: Path | None = None,
    ) -> Registry:
        configured = Path(settings.roles_path) if settings and settings.roles_path else None
        library = PromptLibrary.from_directory(library_dir or packaged_library_dir())
        return cls(library, load_roles(roles_path or configured or packaged_roles_path()))

    def role(self, name: str) -> RoleConfig:
        try:
            return self._roles[name]
        except KeyError:
            raise RolesError(f"unknown role {name!r}; known roles: {sorted(self._roles)}") from None

    def roles(self) -> list[RoleConfig]:
        return [self._roles[name] for name in sorted(self._roles)]

    def prompt_for(self, role: str) -> PromptVersion:
        config = self.role(role)
        return self._library.get(config.prompt, config.prompt_version)

    def request(
        self,
        role: str,
        *,
        messages: Sequence[Message],
        variables: Mapping[str, object] | None = None,
        tools: Sequence[Mapping[str, object]] = (),
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMRequest:
        """Build a request for a role. `variables` fill the system prompt's placeholders."""
        config = self.role(role)
        rendered = self.prompt_for(role).render(**dict(variables or {}))
        return LLMRequest(
            model=config.model,
            messages=messages,
            system=rendered.text,
            tools=tools,
            max_tokens=max_tokens or config.max_tokens,
            temperature=config.temperature,
            timeout_s=timeout_s or config.timeout_s,
            purpose=config.name,
            prompt_version=rendered.full_id,
        )
