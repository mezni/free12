import random

import pytest

from resolve_agent.llm import (
    Message,
    RetryPolicy,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    text_response,
    tool_call_response,
)


class FixedRandom(random.Random):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def random(self) -> float:
        return self.value


def test_message_serialisation() -> None:
    assert Message.user("hi").to_api() == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }
    assistant = Message.assistant(
        [TextBlock("ok"), ToolUseBlock("t1", "get_order_status", {"a": 1})]
    )
    assert assistant.to_api()["content"] == [
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "t1", "name": "get_order_status", "input": {"a": 1}},
    ]
    results = Message.tool_results([ToolResultBlock("t1", "{}"), ToolResultBlock("t2", "x", True)])
    assert results.role == "user"
    assert results.to_api()["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "{}"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "x", "is_error": True},
    ]


def test_response_helpers() -> None:
    response = tool_call_response("issue_refund", {"return_id": "r1"}, text="One moment.")
    assert response.text == "One moment."
    assert [c.name for c in response.tool_calls] == ["issue_refund"]
    assert response.assistant_message().role == "assistant"
    assert text_response("done").tool_calls == ()


@pytest.mark.parametrize(
    ("attempt", "jitter", "expected"),
    [
        (1, 0.0, 0.25),  # base 0.5: [0.25, 0.5]
        (1, 1.0, 0.5),
        (2, 0.0, 0.5),  # 1.0: [0.5, 1.0]
        (2, 1.0, 1.0),
        (3, 1.0, 2.0),
        (4, 1.0, 4.0),  # capped at max_delay
        (9, 0.0, 2.0),  # still capped: [2.0, 4.0]
    ],
)
def test_exponential_backoff_with_equal_jitter(
    attempt: int, jitter: float, expected: float
) -> None:
    assert RetryPolicy().delay(attempt, None, FixedRandom(jitter)) == pytest.approx(expected)


def test_retry_after_raises_the_delay_but_is_capped() -> None:
    policy = RetryPolicy()
    assert policy.delay(1, 3.0, FixedRandom(0.0)) == 3.0
    assert policy.delay(1, 60.0, FixedRandom(0.0)) == policy.max_delay_s
    assert policy.delay(2, 0.1, FixedRandom(1.0)) == 1.0  # never shortens the backoff
