"""Provider-neutral request, response and message types."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str

    def to_api(self) -> dict[str, object]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: Mapping[str, object]

    def to_api(self) -> dict[str, object]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": dict(self.input)}


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_api(self) -> dict[str, object]:
        block: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            block["is_error"] = True
        return block


Block = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: tuple[Block, ...]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", (TextBlock(text),))

    @classmethod
    def assistant(cls, blocks: Sequence[Block]) -> Message:
        return cls("assistant", tuple(blocks))

    @classmethod
    def tool_results(cls, results: Sequence[ToolResultBlock]) -> Message:
        return cls("user", tuple(results))

    def to_api(self) -> dict[str, object]:
        return {"role": self.role, "content": [b.to_api() for b in self.content]}


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMRequest:
    model: str
    messages: Sequence[Message]
    system: str = ""
    tools: Sequence[Mapping[str, object]] = ()
    max_tokens: int = 1024
    temperature: float | None = None
    timeout_s: float | None = None  # per attempt; falls back to settings
    purpose: str = "agent"  # for traces and metrics: "router", "agent", "judge", ...
    prompt_version: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: tuple[TextBlock | ToolUseBlock, ...]
    stop_reason: str
    usage: Usage
    model: str
    request_id: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> tuple[ToolUseBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))

    def assistant_message(self) -> Message:
        return Message.assistant(self.content)


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUseStarted:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class StreamDone:
    response: LLMResponse


StreamEvent = TextDelta | ToolUseStarted | StreamDone


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> Iterator[StreamEvent]: ...
