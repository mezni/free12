"""LLM access: provider-neutral types, an httpx Anthropic client, a fake, and observability."""

from resolve_agent.llm.anthropic_client import AnthropicClient
from resolve_agent.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMOverloadedError,
    LLMRateLimitError,
    LLMServerError,
    LLMStreamInterruptedError,
    LLMTimeoutError,
)
from resolve_agent.llm.fake import FakeLLM, text_response, tool_call_response
from resolve_agent.llm.observed import ObservedLLMClient
from resolve_agent.llm.retry import RetryPolicy
from resolve_agent.llm.types import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseStarted,
    Usage,
)

__all__ = [
    "AnthropicClient",
    "FakeLLM",
    "LLMAuthError",
    "LLMBadRequestError",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMOverloadedError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMServerError",
    "LLMStreamInterruptedError",
    "LLMTimeoutError",
    "Message",
    "ObservedLLMClient",
    "RetryPolicy",
    "StreamDone",
    "StreamEvent",
    "TextBlock",
    "TextDelta",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUseStarted",
    "Usage",
    "text_response",
    "tool_call_response",
]
