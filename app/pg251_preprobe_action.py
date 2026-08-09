"""PG-251 pre-probe action records derived from audited trajectories.

The record keeps only the observable prefix available before a candidate is
sent.  The original typed outcome is retained as an off-input action target,
never copied into the prefix tokens.  ``split_source`` preserves source
holdouts for derived records.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .pg231_feedback_trajectory import prepare_feedback_record
from .pg230_next_token_quality_funnel import REPAIR_INDEX, LANE_INDEX, digest


SCHEMA_VERSION = "pg251-preprobe-action-record-v1"


def build_preprobe_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Create one action-decision prefix without result/oracle labels in input."""

    positive = bool(row.get("payload_grounded_eligible", False))
    original_lane = str(row.get("lane", "silver"))
    # A complete positive reference becomes a send target; all non-positive
    # observations are abstain targets.  The target is metadata for the
    # action head, not an input token.
    target_lane = "gold" if positive else "hard_negative" if original_lane in {"gold", "hard_negative"} else "silver"
    target_repair = "abstain" if not positive else "retry_candidate"
    surface = str(row.get("surface_class", row.get("surface_role", "generic_surface"))).casefold()
    surface_role = "xss_surface" if "dom" in surface or "xss" in surface else "sql_surface" if "sql" in surface else surface
    raw = {
        "source": str(row.get("source", "unknown")),
        "split_source": str(row.get("split_source", row.get("source", "unknown"))),
        "seed": int(row.get("seed", 0) or 0),
        "surface_role": surface_role,
        "method": str(row.get("method", "GET")).upper(),
        "status_class": str(row.get("status_class", "2xx")),
        "field_count": int(row.get("field_count", len(row.get("fields") or [])) or 0),
        "history_len": 0,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "reset_not_attempted": False,
        "candidate_sent": False,
        "oracle_available": bool(row.get("oracle_available", row.get("typed_effect_confirmed", False))),
        "typed_effect_observed": False,
        "typed_effect_confirmed": False,
        "result_fixture_verified": False,
        "candidate_reference_agreement": False,
        "negative_clean": False,
        "binding_valid": bool(row.get("binding_valid", True)),
        "transport_error": False,
        "result_mismatch_observed": False,
        "next_step": "send_candidate" if positive else "abstain",
        "previous_feedback": "none",
        "candidate_result_present": False,
        "model_claimed_positive": False,
        "model_abstained": False,
        "backend_observed": bool(row.get("backend_observed", True)),
        "database_health_ok": bool(row.get("database_health_ok", False)),
        "reference_sent": False,
        "negative_sent": False,
        "candidate_sql_error_shape": False,
        "boolean_differential": False,
        "negative_result_absent": False,
        "hard_gate_observed": False,
        "model_self_error_detected": False,
        "evidence_hash": str(row.get("source_evidence_hash", row.get("evidence_hash", ""))),
        "payload_grounded_eligible": positive,
    }
    record = prepare_feedback_record(raw)
    tokens = list(record["tokens"])
    # The classifier reads the causal hidden state immediately after the
    # observe block.  It cannot see failure, lane, repair, or replay tokens.
    record["classification_position"] = next(index for index, token in enumerate(tokens) if token == "phase=diagnose")
    record["lane"] = target_lane
    record["lane_index"] = LANE_INDEX[target_lane]
    record["repair_action"] = target_repair
    record["repair_index"] = REPAIR_INDEX[target_repair]
    # ``prepare_feedback_record`` computes this field from the temporary
    # observable prefix lane.  Restore the off-input supervision target after
    # assigning the explicit preprobe lane; it is never part of ``tokens``.
    record["payload_grounded_eligible"] = positive
    record["record_role"] = "preprobe_action"
    record["split_source"] = raw["split_source"]
    record["parent_record_id"] = str(row.get("record_id", row.get("trajectory_hash", row.get("token_hash", ""))))
    record["quality_reasons"] = list(record.get("quality_reasons") or []) + ["preprobe_prefix_from_audited_reference"]
    record["trajectory_hash"] = digest({"tokens": tokens, "record_role": "preprobe_action", "parent": record["parent_record_id"]})
    record["raw_payload_strings_stored"] = False
    record["raw_response_bodies_stored"] = False
    return record


def build_preprobe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_preprobe_record(row) for row in rows if row.get("lane") not in {"quarantine", "reject"}]


__all__ = ["SCHEMA_VERSION", "build_preprobe_record", "build_preprobe_rows"]
