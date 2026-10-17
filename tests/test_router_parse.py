import json

import pytest

from resolve_agent.agent import Intent, RouterOutputError, Sentiment, parse_router_output

GOOD = {
    "intent": "return",
    "confidence": 0.92,
    "sentiment": "calm",
    "escalate": False,
    "order_id": None,
}


def _dump(**changes: object) -> str:
    return json.dumps({**GOOD, **changes})


def test_plain_json() -> None:
    out = parse_router_output(_dump())
    assert (out.intent, out.confidence, out.sentiment) == (Intent.RETURN, 0.92, Sentiment.CALM)
    assert out.escalate is False and out.order_id is None


@pytest.mark.parametrize(
    "wrap",
    [
        "```json\n{}\n```",
        "```\n{}\n```",
        "Here is the classification:\n{}",
        "{}\nHope that helps!",
        "  \n{}  ",
    ],
)
def test_fences_and_surrounding_prose_are_tolerated(wrap: str) -> None:
    assert parse_router_output(wrap.replace("{}", _dump())).intent is Intent.RETURN


def test_nested_braces_inside_strings_do_not_confuse_extraction() -> None:
    text = _dump(order_id="ord_1001") + ' {"note": "}"}'
    assert parse_router_output(text).order_id == "ord_1001"


def test_values_are_normalised_and_extras_ignored() -> None:
    out = parse_router_output(_dump(intent=" Order_Status ", sentiment="ANGRY", extra=[1, 2]))
    assert (out.intent, out.sentiment) == (Intent.ORDER_STATUS, Sentiment.ANGRY)


def test_integer_confidence_is_accepted() -> None:
    assert parse_router_output(_dump(confidence=1)).confidence == 1.0


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("I think it is a return", "no JSON object"),
        ("{not json}", "invalid JSON"),
        ("[1, 2]", "no JSON object"),
        (_dump(intent="refund_me"), "intent: unknown value"),
        (_dump(intent=5), "intent: missing or not a string"),
        (_dump(sentiment="furious"), "sentiment: unknown value"),
        (_dump(confidence=1.5), "confidence: out of range"),
        (_dump(confidence=-0.1), "confidence: out of range"),
        (_dump(confidence="high"), "confidence: missing or not a number"),
        (_dump(confidence=True), "confidence: missing or not a number"),
        (_dump(escalate="yes"), "escalate: missing or not a boolean"),
        (_dump(order_id=1001), "order_id: not a string"),
    ],
)
def test_invalid_outputs_are_rejected(text: str, fragment: str) -> None:
    with pytest.raises(RouterOutputError, match=fragment):
        parse_router_output(text)


def test_missing_fields_are_rejected() -> None:
    data = dict(GOOD)
    del data["escalate"]
    with pytest.raises(RouterOutputError, match="escalate"):
        parse_router_output(json.dumps(data))


def test_error_messages_never_echo_model_output() -> None:
    with pytest.raises(RouterOutputError) as excinfo:
        parse_router_output(_dump(intent="my card is 4111111111111111"))
    assert "4111" not in str(excinfo.value)
