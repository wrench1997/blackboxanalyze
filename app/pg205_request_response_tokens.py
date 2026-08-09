"""PG-205 request/response structure tokens.

The tokeniser keeps the information the policy needs to choose the next
request without copying route values or response bodies into the model.  Field
names are reduced to bounded role/count slots; response text is represented by
transport and shape buckets only.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


FIELD_TOKEN_SCHEMA = "pg205-request-response-token-v1"

METHODS = ("GET", "POST")
PLACEMENTS = ("query", "form")
FIELD_COUNT_BUCKETS = ("zero", "one", "two", "many")
VALUE_COUNT_BUCKETS = ("zero", "one", "many")
STATUS_BUCKETS = ("2xx", "3xx", "4xx_5xx", "transport")
REDIRECT_BUCKETS = ("zero", "one", "many")
CONTENT_BUCKETS = ("html", "json", "text", "other")
BODY_BUCKETS = ("empty", "short", "medium", "long")

FORBIDDEN_FIELDS = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "file", "uploadfile"})
SUBMIT_FIELDS = frozenset({"submit"})
VALUE_FIELDS = frozenset({"message", "text", "content", "name", "username", "id", "title", "url", "filename"})

# method(2), placement(2), field_count(4), value_count(3), submit(1),
# forbidden(1), status(4), redirects(3), content(4), body(4), marker(1),
# location(1), typed(1)
FIELD_TOKEN_DIM = 31


def _one_hot(values: Sequence[str], value: str) -> list[float]:
    return [float(item == value) for item in values]


def _field_count_bucket(count: int) -> str:
    return "zero" if count == 0 else "one" if count == 1 else "two" if count == 2 else "many"


def _value_count_bucket(count: int) -> str:
    return "zero" if count == 0 else "one" if count == 1 else "many"


def _status_bucket(value: Any) -> str:
    status = str(value or "").casefold()
    if status in {"transport_error", "transport", "unknown"}:
        return "transport"
    if status == "2xx":
        return "2xx"
    if status == "3xx":
        return "3xx"
    if status in {"4xx", "5xx", "4xx_5xx"}:
        return "4xx_5xx"
    return "transport"


def _redirect_bucket(value: Any) -> str:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        count = 0
    return "zero" if count <= 0 else "one" if count == 1 else "many"


def _content_bucket(value: Any) -> str:
    content = str(value or "").casefold()
    if content == "html" or "html" in content:
        return "html"
    if content == "json" or "json" in content:
        return "json"
    if content in {"text", "plain"} or "text" in content:
        return "text"
    return "other"


def _body_bucket(value: Any) -> str:
    body = str(value or "").casefold()
    if body in {"", "0", "empty"}:
        return "empty"
    if body in {"1-255", "short"}:
        return "short"
    if body in {"256-1023", "1024-4095", "medium"}:
        return "medium"
    return "long"


def field_tokens_for_runtime(
    *,
    method: str,
    field_names: Sequence[str],
    projection: Mapping[str, Any] | None,
    typed_available: bool,
    redirect_hops: int | None = None,
) -> list[float]:
    """Encode request fields and a bounded response projection into 31 slots."""

    method = str(method).upper()
    if method not in METHODS:
        raise ValueError("PG-205 method token must be GET or POST")
    fields = sorted({str(item).casefold() for item in field_names if str(item)})
    if len(fields) > 16:
        raise ValueError("PG-205 field token set is too large")
    projection = dict(projection or {})
    placement = "query" if method == "GET" else "form"
    value_fields = [field for field in fields if field not in SUBMIT_FIELDS]
    sensitive_count = sum(int(field in FORBIDDEN_FIELDS) for field in fields)
    marker = dict(projection.get("marker") or {})
    redirect_chain = projection.get("redirect_chain")
    observed_hops = redirect_hops
    if observed_hops is None:
        observed_hops = projection.get("redirect_hop_count")
    if observed_hops is None and isinstance(redirect_chain, list):
        observed_hops = len(redirect_chain)
    location_present = bool(projection.get("location_origin_changed") or projection.get("location_present") or redirect_chain)
    parts: list[float] = []
    parts += _one_hot(METHODS, method)
    parts += _one_hot(PLACEMENTS, placement)
    parts += _one_hot(FIELD_COUNT_BUCKETS, _field_count_bucket(len(fields)))
    parts += _one_hot(VALUE_COUNT_BUCKETS, _value_count_bucket(len(value_fields)))
    parts += [float("submit" in fields)]
    parts += [float(sensitive_count > 0)]
    parts += _one_hot(STATUS_BUCKETS, _status_bucket(projection.get("status_class")))
    parts += _one_hot(REDIRECT_BUCKETS, _redirect_bucket(observed_hops))
    parts += _one_hot(CONTENT_BUCKETS, _content_bucket(projection.get("content_type_class")))
    parts += _one_hot(BODY_BUCKETS, _body_bucket(projection.get("body_length_bucket")))
    parts += [float(bool(marker.get("reflected") or marker.get("marker_reflected")))]
    parts += [float(location_present)]
    parts += [float(bool(typed_available))]
    if len(parts) != FIELD_TOKEN_DIM:
        raise AssertionError(f"PG-205 field token dimension drift: {len(parts)}")
    return parts


def field_tokens_for_row(row: Mapping[str, Any]) -> list[float]:
    projection = row.get("response_projection") or row.get("projection") or {}
    return field_tokens_for_runtime(
        method=str(row.get("method", "GET")),
        field_names=list(row.get("field_names") or row.get("fields") or []),
        projection=dict(projection),
        typed_available=bool(row.get("typed_available", False)),
        redirect_hops=row.get("redirect_hops"),
    )


__all__ = [
    "BODY_BUCKETS",
    "CONTENT_BUCKETS",
    "FIELD_TOKEN_DIM",
    "FIELD_TOKEN_SCHEMA",
    "FIELD_COUNT_BUCKETS",
    "METHODS",
    "PLACEMENTS",
    "STATUS_BUCKETS",
    "field_tokens_for_row",
    "field_tokens_for_runtime",
]
