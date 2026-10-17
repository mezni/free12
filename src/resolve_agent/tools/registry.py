"""Tool registry: JSON schemas from type hints, strict argument validation, safe dispatch.

Every call returns a ToolResult (never raises), runs write tools in one transaction,
and leaves an audit entry. `ctx` is injected by the caller and is not part of the schema.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from resolve_agent.domain.db import transaction
from resolve_agent.domain.repositories import AuditLog
from resolve_agent.obs import get_metrics, get_tracer
from resolve_agent.tools.context import ToolContext
from resolve_agent.tools.result import ToolError, ToolResult

logger = logging.getLogger("resolve_agent.tools")

MAX_STRING_LENGTH = 500

ToolFn = Callable[..., ToolResult]


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: str  # "str" | "int" | "float" | "bool" | "choice"
    required: bool
    description: str | None
    choices: tuple[str, ...] = ()
    enum_cls: type[Enum] | None = None

    def json_schema(self) -> dict[str, object]:
        schema: dict[str, object]
        if self.kind == "choice":
            schema = {"type": "string", "enum": list(self.choices)}
        elif self.kind == "str":
            schema = {"type": "string", "maxLength": MAX_STRING_LENGTH}
        else:
            schema = {"type": {"int": "integer", "float": "number", "bool": "boolean"}[self.kind]}
        if self.description:
            schema["description"] = self.description
        return schema

    def coerce(self, value: object) -> object:
        name = self.name
        if self.kind == "str":
            if not isinstance(value, str) or not value.strip():
                raise ToolError("invalid_arguments", f"'{name}' must be a non-empty string")
            if len(value) > MAX_STRING_LENGTH:
                raise ToolError(
                    "invalid_arguments", f"'{name}' must be at most {MAX_STRING_LENGTH} characters"
                )
            return value.strip()
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolError("invalid_arguments", f"'{name}' must be an integer")
            return value
        if self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ToolError("invalid_arguments", f"'{name}' must be a number")
            return float(value)
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ToolError("invalid_arguments", f"'{name}' must be a boolean")
            return value
        if not isinstance(value, str) or value not in self.choices:
            raise ToolError("invalid_arguments", f"'{name}' must be one of {list(self.choices)}")
        return self.enum_cls(value) if self.enum_cls else value


def _build_param(name: str, hint: Any, default: object) -> Param:
    description: str | None = None
    if get_origin(hint) is Annotated:
        base, *meta = get_args(hint)
        description = next((m for m in meta if isinstance(m, str)), None)
        hint = base
    optional = False
    if get_origin(hint) in (Union, UnionType):
        members = [a for a in get_args(hint) if a is not type(None)]
        if len(members) != 1 or len(members) == len(get_args(hint)):
            raise TypeError(f"unsupported type for '{name}': only `X | None` unions are allowed")
        hint, optional = members[0], True
    required = default is inspect.Parameter.empty and not optional

    if get_origin(hint) is Literal:
        return Param(name, "choice", required, description, tuple(str(a) for a in get_args(hint)))
    if isinstance(hint, type) and issubclass(hint, Enum):
        return Param(name, "choice", required, description, tuple(str(m.value) for m in hint), hint)
    for py_type, kind in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if hint is py_type:
            return Param(name, kind, required, description)
    raise TypeError(f"unsupported type for '{name}': {hint!r}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    fn: ToolFn
    writes: bool
    params: Mapping[str, Param]

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {n: p.json_schema() for n, p in self.params.items()},
            "required": [n for n, p in self.params.items() if p.required],
            "additionalProperties": False,
        }

    def validate(self, raw: object) -> dict[str, object]:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ToolError("invalid_arguments", "arguments must be a JSON object")
        unknown = sorted(set(raw) - set(self.params))
        if unknown:
            raise ToolError("invalid_arguments", f"unexpected argument(s): {unknown}")
        out: dict[str, object] = {}
        for param in self.params.values():
            value = raw.get(param.name)
            if value is None:
                if param.required:
                    raise ToolError(
                        "invalid_arguments", f"missing required argument '{param.name}'"
                    )
                continue
            out[param.name] = param.coerce(value)
        return out


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def add(self, fn: ToolFn, *, writes: bool = False) -> None:
        if fn.__name__ in self._tools:
            raise ValueError(f"tool {fn.__name__!r} already registered")
        parameters = list(inspect.signature(fn).parameters.values())
        if not parameters or parameters[0].name != "ctx":
            raise TypeError(f"{fn.__name__}: first parameter must be 'ctx: ToolContext'")
        doc = inspect.getdoc(fn)
        if not doc:
            raise TypeError(f"{fn.__name__}: a docstring is required (it is the tool description)")
        hints = get_type_hints(fn, include_extras=True)
        params = {p.name: _build_param(p.name, hints[p.name], p.default) for p in parameters[1:]}
        self._tools[fn.__name__] = ToolSpec(fn.__name__, doc.split("\n\n")[0], fn, writes, params)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, object]]:
        """Tool definitions in the shape LLM APIs expect (name, description, input_schema)."""
        return [
            {"name": s.name, "description": s.description, "input_schema": s.input_schema()}
            for s in self._tools.values()
        ]

    def call(self, name: str, raw_args: object, ctx: ToolContext) -> ToolResult:
        spec = self._tools.get(name)
        label = name if spec else "unknown"  # never use model-supplied names as span/metric labels
        attributes = {
            "tool.name": label,
            "customer_id": ctx.customer_id,
            "request_id": ctx.request_id,
        }
        with get_tracer().span(f"tool.{label}", kind="tool", attributes=attributes) as span:
            if spec is None:
                result = _failure("unknown_tool", f"There is no tool named '{name}'.")
                validated: dict[str, object] = {}
            else:
                validated, result = self._execute(spec, raw_args, ctx)
                span.set_attribute("tool.writes", spec.writes)
            span.set_attribute("tool.ok", result.ok)
            if result.error_code:
                span.set_attribute("tool.error_code", result.error_code)
            if result.error_code == "internal_error":
                span.record_error("internal_error")
            outcome = "ok" if result.ok else (result.error_code or "error")
            get_metrics().inc("tool.calls", 1.0, tool=label, outcome=outcome)
            if spec is not None:
                self._audit(spec, ctx, validated, result, span.trace_id)
            return result

    @staticmethod
    def _execute(
        spec: ToolSpec, raw_args: object, ctx: ToolContext
    ) -> tuple[dict[str, object], ToolResult]:
        validated: dict[str, object] = {}
        try:
            validated = spec.validate(raw_args)
            with transaction(ctx.conn) if spec.writes else nullcontext():
                result = spec.fn(ctx, **validated)
        except ToolError as err:
            result = ToolResult(False, err.data, err.code, err.message)
        except Exception:
            logger.exception("tool.crashed", extra={"fields": {"tool": spec.name}})
            result = _failure(
                "internal_error", "The tool failed unexpectedly. Try again or escalate."
            )
        return validated, result

    @staticmethod
    def _audit(
        spec: ToolSpec, ctx: ToolContext, args: dict[str, object], result: ToolResult, trace_id: str
    ) -> None:
        try:
            AuditLog(ctx.conn, ctx.clock).record(
                actor=ctx.actor,
                action=f"tool.{spec.name}",
                entity="customer",
                entity_id=ctx.customer_id,
                details={
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "args": args,
                    "request_id": ctx.request_id,
                    "trace_id": trace_id,
                },
            )
        except Exception:
            logger.exception("audit.write_failed", extra={"fields": {"tool": spec.name}})


def _failure(code: str, message: str) -> ToolResult:
    return ToolResult(False, {}, code, message)
