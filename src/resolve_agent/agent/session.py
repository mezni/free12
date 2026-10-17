"""Conversation state for one customer session. One request at a time per session."""

from __future__ import annotations

from dataclasses import dataclass, field

from resolve_agent.domain.repositories import new_id
from resolve_agent.llm.types import Message
from resolve_agent.tools.result import ToolResult

MAX_EVIDENCE_RESULTS = 50


@dataclass(slots=True)
class Session:
    customer_id: str
    session_id: str = field(default_factory=lambda: new_id("ses"))
    history: list[Message] = field(default_factory=list)  # what the model sees (masked text only)
    tool_results: list[ToolResult] = field(default_factory=list)  # evidence for the output guard
    customer_messages: list[str] = field(default_factory=list)  # masked, for handoff summaries
    tools_used: list[str] = field(default_factory=list)
    turns: int = 0
    frustrated_turns: int = 0  # consecutive
    tool_failures: int = 0  # consecutive internal errors
    escalated: bool = False
    escalation_reference: str | None = None

    def record_result(self, tool_name: str, result: ToolResult) -> None:
        self.tool_results.append(result)
        del self.tool_results[:-MAX_EVIDENCE_RESULTS]
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
