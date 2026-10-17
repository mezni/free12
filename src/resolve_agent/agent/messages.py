"""Fixed customer-facing messages for the agent loop. No model text, no internals."""

HANDOFF = {
    "requested": "Of course. I'm bringing in a team member who can help you from here.",
    "upset": (
        "I'm sorry for the trouble this has caused. "
        "I'm bringing in a team member who can help you from here."
    ),
    "technical": (
        "I'm having trouble completing this myself, "
        "so I'm bringing in a team member who can help you."
    ),
    "accuracy": (
        "I want to be sure I give you accurate information, "
        "so I'm bringing in a team member who can confirm this for you."
    ),
}
REFERENCE = "Your reference is {reference}."
ALREADY_ESCALATED = (
    "A team member has taken over this conversation and will be in touch. "
    "Anything you add here will be passed along to them."
)
CORRECTION = (
    "Your previous reply was not sent because it contained statements that the tool results "
    "do not support ({codes}). Write a new reply that only states amounts, dates, ids and "
    "outcomes shown in tool results, or call a tool to check. If you cannot confirm "
    "something, say so plainly."
)
