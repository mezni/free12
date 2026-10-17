"""Fixed, customer-safe messages. Guardrail failures never show internals to the customer."""

BLOCKED_INPUT_REPLY = (
    "I couldn't process that message. Could you tell me in your own words what you need "
    "help with, for example an order, a return or an exchange?"
)
TOO_LONG_REPLY = (
    "That message is a bit too long for me to handle. Could you send a shorter version "
    "with the key details?"
)
EMPTY_REPLY = "I didn't catch that. What can I help you with today?"
SENSITIVE_NOTICE = (
    "For your security, please don't share card or ID numbers here. "
    "I've removed them from your message."
)
OUTPUT_FALLBACK_REPLY = (
    "I want to be sure I give you accurate information, so I'm going to bring in a team "
    "member who can confirm this for you."
)
