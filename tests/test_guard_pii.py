import pytest

from resolve_agent.guardrails import pii


@pytest.mark.parametrize(
    "text",
    [
        "my card is 4111 1111 1111 1111 thanks",
        "4111-1111-1111-1111",
        "4111111111111111.",
        "card:5500005555555559 exp 12/29",
        "amex 378282246310005",
    ],
)
def test_valid_card_numbers_are_masked(text: str) -> None:
    result = pii.mask(text)
    assert result.counts == {"card": 1}
    assert "[CARD]" in result.text
    assert not any(ch.isdigit() for ch in result.text.split("[CARD]")[0][-3:])


@pytest.mark.parametrize(
    "text",
    [
        "order ord_1001 is late",
        "ord_1001_i1 please",
        "tracking 1Z999AA10123456784",
        "4111 1111 1111 1112",  # fails the Luhn check
        "call me about order 1234",
        "total was 120.00 on 2026-10-16",
        "ref_ab12cd34ef56",
    ],
)
def test_ids_and_non_card_numbers_are_left_alone(text: str) -> None:
    result = pii.mask(text)
    assert result.text == text and result.counts == {}


def test_sin_and_ssn_are_masked_only_when_valid() -> None:
    assert pii.mask("SIN 046 454 286").text == "SIN [SIN]"
    assert pii.mask("SIN 046 454 287").counts == {}  # bad Luhn
    assert pii.mask("ssn 123-45-6789").text == "ssn [SSN]"
    assert pii.mask("ssn 000-12-3456").counts == {}  # invalid area number


def test_email_and_phone_are_masked() -> None:
    result = pii.mask("write to ada.lovelace+shop@example.co.uk or call (416) 555-0199")
    assert result.text == "write to [EMAIL] or call [PHONE]"
    assert result.counts == {"email": 1, "phone": 1}
    assert pii.mask("+1 416.555.0199").text == "[PHONE]"
    assert pii.mask("4165550199").text == "[PHONE]"


def test_multiple_findings_are_counted() -> None:
    result = pii.mask("a@x.io b@y.io 4111111111111111")
    assert result.counts == {"email": 2, "card": 1}


def test_masking_is_idempotent_and_can_be_limited_to_kinds() -> None:
    once = pii.mask("a@x.io 4111111111111111").text
    assert pii.mask(once).text == once
    assert pii.mask("a@x.io 4111111111111111", {"card"}).text == "a@x.io [CARD]"


def test_find_secrets_ignores_email_and_phone() -> None:
    assert pii.find_secrets("a@x.io (416) 555-0199") == {}
    assert pii.find_secrets("card 4111111111111111") == {"card": 1}


def test_results_never_contain_the_matched_values() -> None:
    result = pii.mask("my card is 4111111111111111")
    assert "4111" not in repr(result.counts)
