"""Tool results: always a value, never an exception, so the agent loop can't crash on a tool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    data: Mapping[str, object] = field(default_factory=dict)
    error_code: str | None = None
    message: str | None = None

    def to_llm(self) -> str:
        """JSON string handed back to the model as the tool result."""
        payload: dict[str, object] = {"ok": self.ok}
        if self.ok:
            payload.update(self.data)
        else:
            payload.update({"error": self.error_code, "message": self.message, **self.data})
        return json.dumps(payload, default=str, sort_keys=True)


class ToolError(Exception):
    """Raised inside a tool to fail cleanly. Rolls back the tool's transaction."""

    def __init__(self, code: str, message: str, **data: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def success(**data: object) -> ToolResult:
    return ToolResult(True, data)
