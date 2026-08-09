"""PG-208 crawl-derived parameter catalog and replay eligibility.

The older browser crawl already saw many GET query keys and POST form fields,
but its response rows were intentionally marked incomplete because the
parameterized request had not been replayed.  This module turns that
inventory into a bounded, report-safe plan.  It never invents a field value,
and it classifies missing/secret/stateful surfaces instead of silently
dropping them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = "pg208-pikachu-parameter-catalog-v1"
SAFE_FIELDS = frozenset({
    "message", "text", "content", "name", "username", "id", "title",
    "url", "filename", "submit",
})
SECRET_FIELDS = frozenset({
    "password", "passwd", "secret", "token", "csrf", "cookie",
    "session", "authorization", "upload", "uploadfile",
})
BLOCKED_PATH_PARTS = (
    "/rce/", "/xxe/", "/ssrf/", "/unsafeupload/", "/unserilization/",
    "/fileinclude/", "/unsafedownload/", "/csrf/", "/burteforce/",
    "/overpermission/", "/pkxss/", "sqli_del.php",
    "xss_stored.php", "/xssblind/",
)


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def family_for_path(path: str) -> str:
    lower = str(path).casefold()
    if "/xss/" in lower:
        return "xss"
    if "/sqli/" in lower:
        return "injection"
    if "urlredirect" in lower:
        return "url_redirect"
    if "/xxe/" in lower:
        return "xxe"
    if "/rce/" in lower:
        return "command"
    return "logic"


def typed_oracle_for(*, path: str, family: str, method: str, fields: list[str]) -> str:
    lower = str(path).casefold()
    if family == "xss" and "stored" not in lower and "blind" not in lower:
        return "dom_nojs_dual"
    if family == "url_redirect":
        return "same_origin_redirect"
    # A browser response cannot expose the Pikachu backend SQL AST.  SQL
    # routes stay unknown until an evaluator-only backend oracle is attached.
    if family == "injection":
        return "unknown_sql_backend"
    return "unknown_surface"


def _fields_for_row(row: Mapping[str, Any]) -> list[str]:
    method = str(row.get("method", "")).upper()
    schema = dict(row.get("request_schema") or {})
    raw = schema.get("query_params", []) if method == "GET" else schema.get("form_params", [])
    return sorted({str(item) for item in raw if str(item)})


def _route_key(row: Mapping[str, Any], fields: list[str]) -> tuple[str, str, tuple[str, ...]]:
    return (str(row.get("route_path", "")), str(row.get("method", "")).upper(), tuple(fields))


def build_parameter_catalog(crawl: Mapping[str, Any]) -> dict[str, Any]:
    """Build a complete route catalog with explicit eligibility reasons."""

    rows = list(crawl.get("request_response_rows") or [])
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for row in rows:
        method = str(row.get("method", "")).upper()
        path = str(row.get("route_path", ""))
        fields = _fields_for_row(row)
        key = _route_key(row, fields)
        if key in seen:
            continue
        seen.add(key)
        reasons: list[str] = []
        if method not in {"GET", "POST"}:
            reasons.append("method_not_supported")
        if not path.startswith("/"):
            reasons.append("path_not_origin_relative")
        if not fields:
            reasons.append("missing_parameter_context")
        if any(field.casefold() in SECRET_FIELDS for field in fields):
            reasons.append("secret_or_credential_field_present")
        if any(field.casefold() not in SAFE_FIELDS for field in fields):
            reasons.append("field_not_in_bounded_canary_grammar")
        if any(part in path.casefold() for part in BLOCKED_PATH_PARTS):
            reasons.append("stateful_or_interpreter_surface_requires_special_evaluator")
        family = family_for_path(path)
        entry = {
            "surface_id": str(row.get("surface_id", _digest(key)[:20])),
            "path": path,
            "method": method,
            "source": str(row.get("source", "unknown")),
            "fields": fields,
            "field_shapes": list((row.get("request_schema") or {}).get("field_shapes") or []),
            "enctype": (row.get("request_schema") or {}).get("enctype"),
            "family": family,
            "typed_oracle": typed_oracle_for(path=path, family=family, method=method, fields=fields),
            "crawl_status": str(row.get("status", "incomplete")),
            "crawl_evidence_sha256": str((row.get("response_schema") or {}).get("baseline_evidence_sha256", "")),
            "parameterized_response_observed_before_pg208": bool((row.get("response_schema") or {}).get("parameterized_response_observed", False)),
            "active_replay_eligible": not reasons,
            "eligibility_reasons": sorted(set(reasons)),
            "training_eligible_before_pg208": False,
            "vulnerability_claim_allowed": False,
        }
        entries.append(entry)
    entries.sort(key=lambda item: (item["path"], item["method"], item["fields"], item["surface_id"]))
    eligible = [item for item in entries if item["active_replay_eligible"]]
    excluded = [item for item in entries if not item["active_replay_eligible"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": "research/pg179_pikachu_browser_crawl_manifest_v1.json",
        "source_request_surface_count": len(rows),
        "unique_route_entry_count": len(entries),
        "active_replay_eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "entries": entries,
        "eligible_entries": eligible,
        "excluded_entries": excluded,
        "raw_request_values_stored": False,
        "raw_response_bodies_stored": False,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


__all__ = ["BLOCKED_PATH_PARTS", "SAFE_FIELDS", "SECRET_FIELDS", "SCHEMA_VERSION", "build_parameter_catalog", "family_for_path", "typed_oracle_for"]
