from decimal import Decimal
from pathlib import Path

import resolve_agent.policy
from resolve_agent.policy import Decision, Reason, explain
from resolve_agent.policy.decisions import REASON_TEMPLATES


def test_every_reason_has_a_template() -> None:
    assert set(REASON_TEMPLATES) == set(Reason)


def test_explain_uses_only_the_facts() -> None:
    decision = Decision.deny(
        Reason.RETURN_WINDOW_EXPIRED, {"window_days": 30, "days_since_delivery": 31}
    )
    text = explain(decision)
    assert "30 days" in text and "31 days ago" in text


def test_explain_formats_money() -> None:
    decision = Decision.allow(
        Reason.REFUND_NEEDS_APPROVAL,
        {"amount": Decimal("50.00"), "auto_refund_limit": Decimal("50")},
        requires_approval=True,
    )
    assert (
        explain(decision) == "A refund of $50.00 needs manager approval because it is $50 or more."
    )


def test_policy_package_is_pure() -> None:
    forbidden = ("sqlite3", "datetime.now", "date.today", "os.environ", "import random", "open(")
    package_dir = Path(resolve_agent.policy.__file__).parent
    for path in package_dir.glob("*.py"):
        source = path.read_text()
        for needle in forbidden:
            assert needle not in source, f"{path.name} uses {needle}"
