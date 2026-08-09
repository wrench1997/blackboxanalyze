"""PG-252 causal probe-gate records.

PG-251 deliberately exposed a target mismatch: it trained a pre-probe action
head against the *eventual grounded payload* label.  That label is not
observable before the first request, so a safe model should abstain.  PG-252
separates the two decisions:

* ``probe_send_eligible`` is an off-input supervision target saying that a
  bounded, non-destructive probe may be sent on a fresh route with a usable
  field and an available typed oracle;
* the later typed effect / negative control remains outside the prefix and is
  judged by the evaluator.

Only abstract route facts are copied into the prefix.  No route identity,
payload, response body, expected effect, lane, or evaluator key is tokenized.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .pg230_next_token_quality_funnel import REPAIR_INDEX, LANE_INDEX, digest
from .pg231_feedback_trajectory import prepare_feedback_record


SCHEMA_VERSION = "pg252-causal-probe-gate-record-v1"


def _token_value(row: Mapping[str, Any], prefix: str, default: str = "") -> str:
    for token in list(row.get("tokens") or []):
        text = str(token)
        if text.startswith(prefix):
            return text.split("=", 1)[1] if "=" in text else default
    return default


def _bool_token(row: Mapping[str, Any], prefix: str, default: bool) -> bool:
    value = _token_value(row, prefix, "")
    if value in {"0", "1"}:
        return value == "1"
    return default


def _bucket_count(row: Mapping[str, Any]) -> int:
    value = row.get("field_count")
    if value not in (None, ""):
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            pass
    bucket = _token_value(row, "field_bucket=", "0")
    return {"0": 0, "1": 1, "2": 2, "3+": 3}.get(bucket, 0)


def _surface_role(row: Mapping[str, Any]) -> str:
    value = str(row.get("surface_class") or row.get("surface_role") or row.get("surface") or "generic_surface").casefold()
    if "dom" in value or "xss" in value:
        # PG-230's canonicalizer maps xss_surface to dom_surface.
        return "xss_surface"
    if "sql" in value or "sqli" in value or "injection" in value:
        # Use sqli_surface because the canonicalizer recognizes the sqli
        # family spelling; plain sql_surface would collapse to generic.
        return "sqli_surface"
    if "redirect" in value:
        return "redirect_surface"
    return "generic_surface"


def _typed_oracle_available(row: Mapping[str, Any], surface_role: str) -> bool:
    explicit = row.get("oracle_available")
    if explicit is not None:
        return bool(explicit)
    typed_oracle = str(row.get("typed_oracle", "")).casefold()
    if typed_oracle and typed_oracle not in {"unknown", "unknown_surface", "none", "null"}:
        return True
    # The source projections intentionally omit evaluator labels.  A DOM/XSS
    # surface is the only currently wired typed oracle family; SQL remains
    # unknown until a backend AST oracle is attached.
    return surface_role == "xss_surface"


def _probe_candidate_sent(row: Mapping[str, Any]) -> bool:
    # This is an audited *workflow* target, not a vulnerability label.  It is
    # used only for diagnostics and never copied to tokens.
    return _bool_token(row, "candidate_sent=", bool(row.get("candidate_sent", False)))


def build_probe_gate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a causal prefix whose action target is safe-probe availability."""

    surface_role = _surface_role(row)
    method = str(row.get("method") or _token_value(row, "method=", "GET")).upper()
    status = str(row.get("status_class") or _token_value(row, "status=", "2xx"))
    if status not in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"}:
        status = "unknown"
    field_count = _bucket_count(row)
    fresh_reset_ok = _bool_token(row, "fresh_reset=", bool(row.get("fresh_reset_ok", True)))
    reset_completed = _bool_token(row, "reset_completed=", bool(row.get("reset_completed", True)))
    reset_not_attempted = bool(row.get("reset_not_attempted", False))
    # The source projections' ``binding_valid`` bit is a post-candidate
    # result (PG-246 uses 0 even on its gold replay rows).  It is not a
    # pre-probe field-schema fact.  At this causal position the safe proxy is
    # simply that the crawl/parser exposed at least one field; the live loop
    # supplies its own packet binding check before sending.
    binding_valid = bool(field_count > 0)
    typed_available = _typed_oracle_available(row, surface_role)
    probe_send_eligible = bool(typed_available and field_count > 0 and binding_valid and fresh_reset_ok and reset_completed and not reset_not_attempted)
    evidence_hash = str(row.get("source_evidence_hash") or row.get("evidence_hash") or "")
    if len(evidence_hash) != 64:
        evidence_hash = digest({"source": str(row.get("source", "unknown")), "seed": int(row.get("seed", 0) or 0), "surface": surface_role, "method": method})

    raw = {
        "source": str(row.get("source", "unknown")),
        "split_source": str(row.get("split_source", row.get("source", "unknown"))),
        "seed": int(row.get("seed", 0) or 0),
        "surface_role": surface_role,
        "method": method,
        "status_class": status,
        "field_count": field_count,
        "history_len": 0,
        "fresh_reset_ok": fresh_reset_ok,
        "reset_completed": reset_completed,
        "reset_not_attempted": reset_not_attempted,
        "candidate_sent": False,
        "oracle_available": typed_available,
        "typed_effect_observed": False,
        "typed_effect_confirmed": False,
        "result_fixture_verified": False,
        "candidate_reference_agreement": False,
        "negative_clean": False,
        "binding_valid": binding_valid,
        "transport_error": False,
        "result_mismatch_observed": False,
        "next_step": "send_candidate" if probe_send_eligible else "abstain",
        "previous_feedback": "none",
        "candidate_result_present": False,
        "model_claimed_positive": False,
        "model_abstained": not probe_send_eligible,
        "backend_observed": _bool_token(row, "backend_observed=", bool(row.get("backend_observed", False))),
        "database_health_ok": _bool_token(row, "database_health=", bool(row.get("database_health_ok", False))),
        "reference_sent": False,
        "negative_sent": False,
        "candidate_sql_error_shape": False,
        "boolean_differential": False,
        "negative_result_absent": False,
        "hard_gate_observed": False,
        "model_self_error_detected": False,
        "evidence_hash": evidence_hash,
        # This is intentionally false: PG-252's action target is the
        # safe-probe gate, never a claim that the route is vulnerable.
        "payload_grounded_eligible": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    record = prepare_feedback_record(raw)
    tokens = list(record["tokens"])
    diagnose_index = next(index for index, token in enumerate(tokens) if token == "phase=diagnose")
    # Make the route's *configured* oracle availability explicit.  It is an
    # observable capability fact, not the result of the future probe.
    tokens.insert(diagnose_index, f"oracle_available={1 if typed_available else 0}")
    record["tokens"] = tokens
    record["classification_position"] = diagnose_index + 1
    record["lane"] = "silver"
    record["lane_index"] = LANE_INDEX["silver"]
    record["repair_action"] = "retry_candidate" if probe_send_eligible else "abstain"
    record["repair_index"] = REPAIR_INDEX[record["repair_action"]]
    record["payload_grounded_eligible"] = False
    record["probe_send_eligible"] = probe_send_eligible
    record["probe_target_source_action"] = _probe_candidate_sent(row)
    record["record_role"] = "probe_gate_action"
    record["split_source"] = raw["split_source"]
    record["parent_record_id"] = str(row.get("record_id", row.get("trajectory_hash", row.get("token_hash", ""))))
    record["quality_reasons"] = list(record.get("quality_reasons") or []) + ["causal_probe_gate_from_observable_route_facts"]
    record["trajectory_hash"] = digest({"tokens": tokens, "record_role": "probe_gate_action", "parent": record["parent_record_id"]})
    record["raw_payload_strings_stored"] = False
    record["raw_response_bodies_stored"] = False
    return record


def build_probe_gate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_probe_gate_record(row) for row in rows if row.get("lane") not in {"quarantine", "reject"}]


__all__ = ["SCHEMA_VERSION", "build_probe_gate_record", "build_probe_gate_rows"]
