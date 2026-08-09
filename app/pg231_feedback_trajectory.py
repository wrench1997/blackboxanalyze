"""PG-231 enriched failure -> repair trajectories for the next-token funnel.

PG-230 showed that the bottleneck was not the frozen body or the token head;
it was that many records collapsed to the same short event.  This module adds
only bounded, observable process states (request sent, backend/DB health,
reference/negative controls, result shape and prior feedback).  It deliberately
does not add routes, payload values, response bodies, evaluator keys or
diagnosis targets to the input tokens.
"""

from __future__ import annotations

from typing import Any, Mapping

from .pg230_next_token_quality_funnel import (
    LANES,
    LANE_INDEX,
    REPAIR_ACTIONS,
    REPAIR_INDEX,
    _bucket,
    _failure_kind,
    _repair_action,
    _surface_class,
    digest,
    quality_lane,
)


PG231_SCHEMA = "pg231-feedback-trajectory-funnel-v1"


def _flag(value: Any) -> str:
    return "1" if bool(value) else "0"


def _finite_feedback(value: Any) -> str:
    value = str(value or "none").casefold()
    if value in {"none", "result_verified", "recheck_oracle", "failure_adjusted", "mismatch", "abstain"}:
        return value
    return "other"


def _replay_expected(row: Mapping[str, Any]) -> str:
    if bool(row.get("typed_effect_confirmed") or row.get("typed_effect_observed") or row.get("result_fixture_verified")):
        return "typed"
    if bool(row.get("oracle_available")):
        return "reference"
    if bool(row.get("reset_not_attempted")) or not bool(row.get("candidate_sent", row.get("ai_sent", True))):
        return "incomplete"
    return "oracle_gap"


def feedback_tokens(row: Mapping[str, Any], lane: str | None = None) -> list[str]:
    """Encode a two-phase bounded trajectory without label/route leakage."""

    lane = lane or quality_lane(row)[0]
    method = str(row.get("method", "GET")).upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "HEAD", "OPTIONS"}:
        method = "OTHER"
    status = str(row.get("status_class", "unknown"))
    if status not in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"}:
        status = "unknown"
    tokens = [
        "[BOS]",
        "phase=observe",
        f"surface={_surface_class(row)}",
        f"method={method}",
        f"status={status}",
        f"candidate_sent={_flag(row.get('candidate_sent', row.get('ai_sent', True)))}",
        f"field_bucket={_bucket(row.get('field_count'))}",
        f"history_bucket={_bucket(row.get('history_len'))}",
        f"backend_observed={_flag(row.get('backend_observed'))}",
        f"database_health={_flag(row.get('database_health_ok'))}",
        f"binding_valid={_flag(row.get('binding_valid'))}",
        f"reference_sent={_flag(row.get('reference_sent'))}",
        f"negative_sent={_flag(row.get('negative_sent'))}",
        f"candidate_present={_flag(row.get('candidate_result_present'))}",
        f"candidate_error_shape={_flag(row.get('candidate_sql_error_shape'))}",
        f"boolean_differential={_flag(row.get('boolean_differential'))}",
        f"negative_result_absent={_flag(row.get('negative_result_absent'))}",
        f"result_verified={_flag(row.get('result_fixture_verified'))}",
        f"hard_gate={_flag(row.get('hard_gate_observed'))}",
        f"transport_error={_flag(row.get('transport_error'))}",
        f"result_mismatch={_flag(row.get('result_mismatch_observed'))}",
        f"fresh_reset={_flag(row.get('fresh_reset_ok'))}",
        f"reset_completed={_flag(row.get('reset_completed'))}",
        f"model_abstained={_flag(row.get('model_abstained'))}",
        f"model_claimed_positive={_flag(row.get('model_claimed_positive'))}",
        f"feedback={_finite_feedback(row.get('previous_feedback'))}",
        "phase=diagnose",
        f"failure={_failure_kind(row)}",
        "phase=repair",
        f"lane={lane}",
        f"repair={_repair_action(row, lane)}",
        f"self_error={_flag(row.get('model_self_error_detected') or row.get('model_self_error_kind'))}",
        "phase=replay",
        f"replay_expected={_replay_expected(row)}",
        "[EOS]",
    ]
    return tokens


def prepare_feedback_record(row: Mapping[str, Any]) -> dict[str, Any]:
    lane, reasons = quality_lane(row)
    tokens = feedback_tokens(row, lane)
    failure_index = next((index for index, token in enumerate(tokens) if token.startswith("failure=")), 0)
    repair_action = _repair_action(row, lane)
    return {
        "source": str(row.get("source", "unknown")),
        "seed": int(row.get("seed", 0) or 0),
        "surface_class": _surface_class(row),
        "method": str(row.get("method", "GET")).upper(),
        "lane": lane,
        "lane_index": LANE_INDEX[lane],
        "repair_action": repair_action,
        "repair_index": REPAIR_INDEX[repair_action],
        "failure_kind": _failure_kind(row),
        "replay_expected": _replay_expected(row),
        "classification_position": failure_index,
        "tokens": tokens,
        "trajectory_hash": digest(tokens),
        "quality_reasons": reasons,
        "source_evidence_hash": str(row.get("evidence_hash", "")),
        "model_self_error_detected": bool(row.get("model_self_error_detected") or row.get("model_self_error_kind")),
        "payload_grounded_eligible": bool(lane == "gold" and row.get("payload_grounded_eligible", False)),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


__all__ = [
    "PG231_SCHEMA",
    "feedback_tokens",
    "prepare_feedback_record",
]

