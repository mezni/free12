"""A scripted LLM for tests and evals: deterministic, offline, records every request."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence

from resolve_agent.llm.types import (
    LLMRequest,
    LLMResponse,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    ToolUseStarted,
    Usage,
)

ScriptItem = LLMResponse | Exception | Callable[[LLMRequest], LLMResponse]


def text_response(
    text: str, *, input_tokens: int = 10, output_tokens: int = 5, model: str = "fake-model"
) -> LLMResponse:
    return LLMResponse(
        content=(TextBlock(text),),
        stop_reason="end_turn",
        usage=Usage(input_tokens, output_tokens),
        model=model,
    )


def tool_call_response(
    name: str,
    tool_input: Mapping[str, object],
    *,
    tool_id: str = "toolu_1",
    text: str | None = None,
    model: str = "fake-model",
) -> LLMResponse:
    blocks: list[TextBlock | ToolUseBlock] = [TextBlock(text)] if text else []
    blocks.append(ToolUseBlock(tool_id, name, dict(tool_input)))
    return LLMResponse(
        content=tuple(blocks), stop_reason="tool_use", usage=Usage(20, 8), model=model
    )


class FakeLLM:
    """Plays back a script in order. Exceptions in the script are raised, like real failures."""

    def __init__(self, script: Sequence[ScriptItem] = ()) -> None:
        self._script: deque[ScriptItem] = deque(script)
        self.requests: list[LLMRequest] = []

    def enqueue(self, *items: ScriptItem) -> None:
        self._script.extend(items)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError(f"FakeLLM script exhausted at request #{len(self.requests)}")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item(request) if callable(item) else item

    def stream(self, request: LLMRequest) -> Iterator[StreamEvent]:
        response = self.complete(request)  # failures surface before any event, like a real retry
        for block in response.content:
            if isinstance(block, TextBlock):
                for chunk in re.findall(r"\S+\s*", block.text):
                    yield TextDelta(chunk)
            else:
                yield ToolUseStarted(block.id, block.name)
        yield StreamDone(response)
