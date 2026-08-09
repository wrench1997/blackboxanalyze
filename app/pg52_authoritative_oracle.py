"""Evaluator-only helpers for PG-52's real local Pikachu oracle lane.

PG-51 deliberately stopped at response-surface projections.  This module is
the next boundary: it turns a bounded, generated canary into an authoritative
local observation when (and only when) a disposable loopback target shows the
expected effect.  The raw canary and response text are kept in memory only;
the returned projections contain hashes and small typed fields.

The module is not a scanner.  Paths, destinations, browser actions and SQL
operator classes are all allow-listed by the PG-52 runner.  Browser execution
sets an in-page marker only; the redirect oracle never follows the destination;
the SQL oracle is read-only and records an AST fingerprint rather than a raw
query.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx


PG52_SCHEMA = "sift-pg52-authoritative-local-oracle-v1"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
ALLOWED_PORTS = frozenset({8767, 8768})
SQL_UNSAFE_TOKENS = frozenset({"insert", "update", "delete", "drop", "alter", "create", "sleep", "benchmark", "load_file", "outfile"})


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_loopback_url(value: str) -> str:
    parsed = urlsplit(str(value).rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError("PG-52 requires an HTTP loopback URL")
    if parsed.port not in ALLOWED_PORTS:
        raise ValueError("PG-52 port is not allow-listed")
    return str(value).rstrip("/")


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status <= 599 else "other"


def _length_bucket(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    if length <= 65535:
        return "4096-65535"
    return "65536+"


class _HtmlShape(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags = 0
        self.forms = 0
        self.inputs = 0
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = str(tag).casefold()
        self.tags += 1
        self.forms += int(lower == "form")
        self.inputs += int(lower in {"input", "textarea", "select", "button"})
        self.scripts += int(lower == "script")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def response_projection(response: httpx.Response, marker: str = "") -> dict[str, Any]:
    """Return a bounded HTTP projection; response text is discarded by caller."""

    body = bytes(response.content)
    text = body.decode("utf-8", errors="replace")
    parser = _HtmlShape()
    try:
        parser.feed(text)
        parser.close()
    except (TypeError, ValueError):
        pass
    location = str(response.headers.get("location", ""))
    marker_count = min(text.count(str(marker)), 8) if marker else 0
    result = {
        "status_code": int(response.status_code),
        "status_class": _status_class(int(response.status_code)),
        "content_type": str(response.headers.get("content-type", "")).split(";", 1)[0].casefold(),
        "body_length": len(body),
        "body_length_bucket": _length_bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": hashlib.sha256(body).hexdigest(),
        "marker_reflected": bool(marker and marker in text),
        "marker_count": marker_count,
        "html_tag_count": min(parser.tags, 512),
        "form_count": min(parser.forms, 64),
        "input_count": min(parser.inputs, 128),
        "script_count": min(parser.scripts, 64),
        "has_location": bool(location),
        "location_origin": _location_origin(location),
        "location_sha256": sha256_text(location) if location else "",
        "external_network": False,
    }
    result["projection_sha256"] = sha256_json(result)
    return result


def _location_origin(location: str) -> str:
    if not location:
        return "none"
    parsed = urlsplit(location)
    if parsed.hostname in LOOPBACK_HOSTS and parsed.port in ALLOWED_PORTS:
        return "loopback"
    if parsed.scheme or parsed.netloc:
        return "external"
    return "relative"


def build_payload_manifest(*, family: str, surface: str, method: str, field: str, payload: str, probe_ref: str, mode: str) -> dict[str, Any]:
    """Hash a runtime canary without persisting its raw value."""

    return {
        "family": str(family),
        "surface": str(surface),
        "method": str(method).upper(),
        "field": str(field),
        "probe_ref": str(probe_ref),
        "mode": str(mode),
        "payload_sha256": sha256_text(payload),
        "payload_length": len(payload),
        "raw_payload_stored": False,
    }


def browser_execution_oracle(*, marker: str, executed: bool, same_origin: bool, external_request_count: int, navigation_count: int, mode: str, execution_path: str = "automatic") -> dict[str, Any]:
    """Build the typed browser effect without exposing a page or payload."""

    positive = bool(executed and same_origin and external_request_count == 0)
    return {
        "oracle_id": f"pg52-browser-{mode}-v1",
        "modality": "browser_dom_execution",
        "positive": positive,
        "positive_authority": True,
        "confirmed_effect": "dom_structure" if positive else "none",
        "candidate_signal": bool(executed),
        "signals": {
            "marker_set_in_page": bool(executed),
            "same_origin": bool(same_origin),
            "external_request_count": int(external_request_count),
            "navigation_count": int(navigation_count),
            "mode": str(mode),
            "execution_path": str(execution_path),
        },
        "safety": {
            "external_network": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def redirect_oracle(*, location: str, expected_destination: str, response_status: int) -> dict[str, Any]:
    """Confirm an open-redirect effect without following the destination."""

    parsed = urlsplit(str(location))
    expected = urlsplit(str(expected_destination))
    exact = bool(location and location == expected_destination)
    loopback_destination = expected.hostname in LOOPBACK_HOSTS and expected.port in ALLOWED_PORTS
    positive = bool(300 <= int(response_status) < 400 and exact and loopback_destination)
    return {
        "oracle_id": "pg52-url-redirect-loopback-v1",
        "modality": "redirect_destination_controlled",
        "positive": positive,
        "positive_authority": True,
        "confirmed_effect": "redirect_origin" if positive else "none",
        "candidate_signal": bool(location),
        "signals": {
            "status_class": _status_class(int(response_status)),
            "location_present": bool(location),
            "location_exact": exact,
            "destination_origin": "loopback" if loopback_destination else "external",
            "location_sha256": sha256_text(location) if location else "",
            "parsed_scheme_present": bool(parsed.scheme),
        },
        "safety": {
            "external_network": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def sql_ast_projection(query: str) -> dict[str, Any]:
    """Project a query into a small AST-like fingerprint, never the raw SQL."""

    raw = str(query)
    lowered = raw.casefold()
    # Replace literals/comments before token counting.  The fingerprint is
    # deliberately coarse: it detects operator-boundary changes, not schema
    # contents or a general SQL grammar.
    no_comments = re.sub(r"/\*.*?\*/|--[^\r\n]*|#[^\r\n]*", " ", lowered, flags=re.DOTALL)
    normalized = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "<literal>", no_comments)
    normalized = re.sub(r"\b\d+\b", "<number>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = re.findall(r"[a-z_]+|<>|<=|>=|=|\(|\)|,", normalized)
    operators = sorted({token for token in tokens if token in {"and", "or", "union", "like", "=", "<>", "<", ">", "in"}})
    unsafe = sorted({token for token in re.findall(r"[a-z_]+", lowered) if token in SQL_UNSAFE_TOKENS})
    result = {
        "raw_query_sha256": sha256_text(raw),
        "query_length": len(raw),
        "ast_sha256": sha256_text(normalized),
        "statement_class": "select" if re.search(r"\bselect\b", lowered) else "other",
        "operator_set": operators,
        "predicate_count": min(sum(token in {"and", "or", "like", "="} for token in tokens), 32),
        "comment_present": bool(re.search(r"/\*|--|#", lowered)),
        "unsafe_operator_set": unsafe,
        "raw_query_stored": False,
    }
    result["projection_sha256"] = sha256_json(result)
    return result


def sql_ast_differential_oracle(*, control_query: str, candidate_query: str, control_response: dict[str, Any], candidate_response: dict[str, Any], expected_marker: str) -> dict[str, Any]:
    control_ast = sql_ast_projection(control_query)
    candidate_ast = sql_ast_projection(candidate_query)
    control_rows = int(control_response.get("result_row_count", 0))
    candidate_rows = int(candidate_response.get("result_row_count", 0))
    ast_changed = control_ast["ast_sha256"] != candidate_ast["ast_sha256"]
    semantic_differential = control_rows != candidate_rows or control_response.get("semantic_body_sha256") != candidate_response.get("semantic_body_sha256")
    safe = not candidate_ast["unsafe_operator_set"] and candidate_ast["statement_class"] == "select"
    positive = bool(ast_changed and semantic_differential and safe and expected_marker)
    return {
        "oracle_id": "pg52-sql-ast-differential-v1",
        "modality": "sql_ast_differential",
        "positive": positive,
        "positive_authority": True,
        "confirmed_effect": "interpreter_boundary" if positive else "none",
        "candidate_signal": bool(ast_changed or semantic_differential),
        "signals": {
            "control_ast_sha256": control_ast["ast_sha256"],
            "candidate_ast_sha256": candidate_ast["ast_sha256"],
            "ast_changed": ast_changed,
            "control_operator_set": control_ast["operator_set"],
            "candidate_operator_set": candidate_ast["operator_set"],
            "control_result_row_count": control_rows,
            "candidate_result_row_count": candidate_rows,
            "semantic_differential": semantic_differential,
            "safe_read_only_select": safe,
            "marker_binding": sha256_text(expected_marker),
        },
        "safety": {
            "external_network": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


__all__ = [
    "PG52_SCHEMA",
    "ALLOWED_PORTS",
    "build_payload_manifest",
    "browser_execution_oracle",
    "redirect_oracle",
    "response_projection",
    "sha256_json",
    "sha256_text",
    "sql_ast_differential_oracle",
    "sql_ast_projection",
    "validate_loopback_url",
]
