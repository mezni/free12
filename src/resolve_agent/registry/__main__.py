"""Print the active roles and prompt versions: uv run python -m resolve_agent.registry"""

from __future__ import annotations

from resolve_agent.config import Settings
from resolve_agent.registry.registry import Registry


def main() -> int:
    registry = Registry.load(Settings.from_env())
    for role in registry.roles():
        prompt = registry.prompt_for(role.name)
        variables = ", ".join(sorted(prompt.variables)) or "-"
        print(
            f"{role.name:<8} {role.model:<20} {prompt.full_id:<20} "
            f"max_tokens={role.max_tokens} temp={role.temperature} timeout={role.timeout_s}s "
            f"vars=[{variables}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
