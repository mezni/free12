"""LLM errors, classified by whether retrying can help."""

from __future__ import annotations

from typing import ClassVar


class LLMError(Exception):
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.request_id = request_id


class LLMRateLimitError(LLMError):
    retryable = True


class LLMOverloadedError(LLMError):
    retryable = True


class LLMServerError(LLMError):
    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


class LLMStreamInterruptedError(LLMError):
    retryable = True


class LLMBadRequestError(LLMError):
    """Our request is wrong. Retrying the same request cannot help."""


class LLMAuthError(LLMError):
    """Missing, invalid or unauthorised API key."""


class LLMInvalidResponseError(LLMError):
    """The server answered, but not in a shape we understand."""


def error_from_status(
    status: int, message: str, *, retry_after: float | None, request_id: str | None
) -> LLMError:
    kwargs = {"status": status, "retry_after": retry_after, "request_id": request_id}
    if status == 429:
        return LLMRateLimitError(message, **kwargs)  # type: ignore[arg-type]
    if status == 529:
        return LLMOverloadedError(message, **kwargs)  # type: ignore[arg-type]
    if status in (401, 403):
        return LLMAuthError(message, **kwargs)  # type: ignore[arg-type]
    if status == 408 or status >= 500:
        return LLMServerError(message, **kwargs)  # type: ignore[arg-type]
    return LLMBadRequestError(message, **kwargs)  # type: ignore[arg-type]


def error_from_type(error_type: str, message: str, *, request_id: str | None) -> LLMError:
    """Map an in-stream `error` event type to an exception."""
    if error_type == "overloaded_error":
        return LLMOverloadedError(message, request_id=request_id)
    if error_type == "rate_limit_error":
        return LLMRateLimitError(message, request_id=request_id)
    if error_type in ("authentication_error", "permission_error"):
        return LLMAuthError(message, request_id=request_id)
    if error_type in ("invalid_request_error", "not_found_error", "request_too_large"):
        return LLMBadRequestError(message, request_id=request_id)
    return LLMServerError(message, request_id=request_id)
