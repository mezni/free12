import pytest

from resolve_agent.agent import Intent, Sentiment
from resolve_agent.agent.rules import (
    classify_by_rules,
    explicit_human_request,
    extract_order_id,
    safety_signals,
)


@pytest.mark.parametrize(
    "text",
    [
        "I want to talk to a human",
        "Can I speak with someone please?",
        "let me speak to a manager",
        "Transfer me to a real person",
        "connect me with an agent",
        "I need a representative",
        "I want a human",
        "get me a live agent",
        "please escalate this",
        "supervisor now",
        "chat with a team member",
    ],
)
def test_explicit_human_requests_are_detected(text: str) -> None:
    assert explicit_human_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "Is the delivery agent coming today?",
        "I'm only human, I make mistakes",
        "I talked to my manager and he said to order these",
        "I spoke to a person yesterday and they sent the wrong size",
        "my order was handled by a human error apparently",
        "Where is my order ord_1001?",
        "the person at the store was nice",
        "I need a refund for these shoes",
    ],
)
def test_ordinary_mentions_of_people_are_not_human_requests(text: str) -> None:
    assert not explicit_human_request(text)


@pytest.mark.parametrize(
    ("text", "sentiment", "legal"),
    [
        ("where is my order", Sentiment.CALM, False),
        ("I'm still waiting for my parcel", Sentiment.FRUSTRATED, False),
        ("THIS IS THE THIRD TIME I HAVE WRITTEN", Sentiment.FRUSTRATED, False),
        ("where is it!!!", Sentiment.FRUSTRATED, False),
        ("this is unacceptable", Sentiment.ANGRY, False),
        ("I'm fed up with this", Sentiment.ANGRY, False),
        ("I will contact my lawyer", Sentiment.ANGRY, True),
        ("this is a scam", Sentiment.ANGRY, True),
        ("I'll do a chargeback", Sentiment.ANGRY, True),
    ],
)
def test_safety_signals(text: str, sentiment: Sentiment, legal: bool) -> None:
    signals = safety_signals(text)
    assert (signals.sentiment, signals.legal_or_fraud) == (sentiment, legal)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Where is my order ord_1001?", Intent.ORDER_STATUS),
        ("the package hasn't arrived", Intent.ORDER_STATUS),
        ("can you give me the tracking number", Intent.ORDER_STATUS),
        ("I want to return these shoes", Intent.RETURN),
        ("I'd like my money back", Intent.RETURN),
        ("please send it back and refund me", Intent.RETURN),
        ("can I exchange this for a bigger size", Intent.EXCHANGE),
        ("I want to swap the jacket", Intent.EXCHANGE),
        ("wrong size, can I exchange or return it?", Intent.EXCHANGE),
        ("what's your favourite colour", Intent.OUT_OF_SCOPE),
        ("hello", Intent.OUT_OF_SCOPE),
    ],
)
def test_rule_based_intents(text: str, intent: Intent) -> None:
    assert classify_by_rules(text).intent is intent


def test_rule_confidence_is_modest_and_lowest_for_unknown() -> None:
    assert classify_by_rules("return this").confidence == 0.6
    assert classify_by_rules("blah").confidence == 0.3


def test_order_id_extraction() -> None:
    assert extract_order_id("Where is ORD_1001?") == "ord_1001"
    assert extract_order_id("order number 1001") is None
    assert classify_by_rules("status of ord_1002").order_id == "ord_1002"
