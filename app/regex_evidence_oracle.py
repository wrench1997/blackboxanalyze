"""Allow-listed regex evidence for local, evaluator-side acceptance.

Patterns are selected by the adapter, never supplied by the model.  Only a
bounded match summary and a hash of the canonical summary are returned; raw
response text and capture groups are discarded immediately.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


REGEX_EVIDENCE_SCHEMA = "sift-regex-evidence-v1"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_STATIC_PATTERNS = {
    "header_xfo_allowall": re.compile(r"^\s*allowall\s*$", re.IGNORECASE),
    "header_xfo_protected": re.compile(r"^\s*(sameorigin|deny)\s*$", re.IGNORECASE),
    "header_csp_ancestors_none": re.compile(r"(?:^|;)\s*frame-ancestors\s+['\"]?none['\"]?\s*(?:;|$)", re.IGNORECASE),
    "sql_error_shape": re.compile(r"(?:sql|database|query)\s*(?:error|exception|syntax)", re.IGNORECASE),
    "generic_server_error_shape": re.compile(r"(?:internal\s+server\s+error|stack\s+trace|exception)", re.IGNORECASE),
}


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _span_bucket(span: int) -> str:
    if span < 32:
        return "0-31"
    if span < 128:
        return "32-127"
    if span < 512:
        return "128-511"
    return "512+"


def evaluate_allowlisted_regex(
    *,
    text: str,
    pattern_id: str,
    marker: str | None = None,
    max_matches: int = 8,
) -> dict[str, Any]:
    """Evaluate one adapter-selected pattern and return bounded evidence."""

    if not _ID_RE.fullmatch(pattern_id):
        raise ValueError("regex evidence pattern id is not bounded")
    if len(text) > 2_000_000:
        raise ValueError("regex evidence input exceeds adapter bound")
    if pattern_id == "escaped_marker_reflection":
        if marker is None or not _ID_RE.fullmatch(marker):
            raise ValueError("marker reflection evidence requires a bounded marker")
        pattern = re.compile(re.escape(marker))
    else:
        pattern = _STATIC_PATTERNS.get(pattern_id)
        if pattern is None:
            raise ValueError("regex evidence pattern is not allow-listed")
    matches = list(pattern.finditer(text))[: max(1, min(int(max_matches), 8))]
    summary = {
        "schema_version": REGEX_EVIDENCE_SCHEMA,
        "pattern_id": pattern_id,
        "matched": bool(matches),
        "match_count": len(matches),
        "span_buckets": sorted({_span_bucket(max(0, match.end() - match.start())) for match in matches}),
        "input_length_bucket": "0" if not text else "1-255" if len(text) <= 255 else "256-4095" if len(text) <= 4095 else "4096+",
        "raw_text_stored": False,
        "capture_groups_stored": False,
    }
    summary["evidence_hash"] = _sha256_json(summary)
    return summary


__all__ = ["REGEX_EVIDENCE_SCHEMA", "evaluate_allowlisted_regex"]
