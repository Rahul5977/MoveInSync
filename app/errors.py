"""One plain sentence for request validation errors (D42).

We never return Pydantic's error list or any part of the submitted input. A field name is echoed
only when it is one of ours, or a short identifier-like extra key (e.g. "company_id").
"""

import re
from collections.abc import Iterable, Sequence
from typing import Any

from app.schemas import QUESTION_MAX, QUESTION_MIN

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")


def _field(loc: Sequence[Any]) -> str | None:
    names = [x for x in loc if isinstance(x, str) and x not in ("body", "query", "path")]
    name = names[-1] if names else None
    return name if name and _SAFE_NAME.match(name) else None


def not_allowed(names: Iterable[str]) -> str:
    safe = sorted(n for n in names if _SAFE_NAME.match(n))
    if len(safe) == 1:
        return f"The '{safe[0]}' field is not allowed."
    return "This request contains fields that are not allowed."


def validation_message(errors: Sequence[dict[str, Any]]) -> str:
    if not errors:
        return "The request is invalid."
    err = errors[0]
    kind, field = err.get("type", ""), _field(err.get("loc", ()))

    if kind == "json_invalid":
        return "The request body is not valid JSON."
    if kind in ("model_attributes_type", "dict_type", "model_type"):
        return "The request body must be a JSON object."
    if kind == "extra_forbidden":
        return not_allowed([field] if field else [])
    if field is None:
        return "The request is invalid."
    if kind == "missing":
        return f"The '{field}' field is required."
    if field == "question" and kind in ("string_too_short", "string_too_long"):
        return f"The question must be between {QUESTION_MIN} and {QUESTION_MAX} characters."
    if kind == "string_type":
        return f"The '{field}' field must be text."
    return f"The '{field}' field is invalid."
