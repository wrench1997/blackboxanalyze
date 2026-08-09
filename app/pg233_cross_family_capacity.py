"""PG-233 cross-family token trajectories and capacity-sweep helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .pg230_next_token_quality_funnel import _surface_class, digest
from .pg231_feedback_trajectory import prepare_feedback_record


PG233_SCHEMA = "pg233-cross-family-capacity-v1"
FAMILIES = ("sql", "dom", "redirect", "logic", "generic")


def family_class(value: Any, surface: Any = None) -> str:
    text = str(value or surface or "generic").casefold()
    if any(needle in text for needle in ("sql", "sqli", "injection")):
        return "sql"
    if any(needle in text for needle in ("xss", "dom", "markup")):
        return "dom"
    if any(needle in text for needle in ("redirect", "url")):
        return "redirect"
    if any(needle in text for needle in ("logic", "auth", "access")):
        return "logic"
    return "generic"


def _safe_channel(value: Any) -> str:
    text = str(value or "unknown").casefold()
    if "query" in text or text == "get":
        return "query"
    if "form" in text or text == "post":
        return "form"
    return "unknown"


def add_family_context(record: Mapping[str, Any], *, family: Any = None, channel: Any = None, pair_role: Any = None, source_role: Any = None) -> dict[str, Any]:
    """Add bounded family/channel context before the failure target suffix."""

    output = dict(record)
    fam = family_class(family, record.get("surface_class"))
    channel_token = _safe_channel(channel or record.get("method"))
    pair = str(pair_role or "single").casefold()
    if pair not in {"candidate", "control", "single"}:
        pair = "single"
    role = str(source_role or "observed").casefold()
    if role not in {"observed", "evaluation_only", "derived"}:
        role = "observed"
    tokens = list(record.get("tokens", []))
    failure_index = next((index for index, token in enumerate(tokens) if str(token).startswith("failure=")), len(tokens) - 1)
    prefix = [f"family={fam}", f"channel={channel_token}", f"pair_role={pair}", f"source_role={role}"]
    tokens = tokens[:failure_index] + prefix + tokens[failure_index:]
    output.update({"family_class": fam, "channel_class": channel_token, "pair_role": pair, "source_role": role, "classification_position": failure_index + len(prefix), "tokens": tokens, "trajectory_hash": digest(tokens), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    return output


def prepare_pikachu_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Convert PG-51 safe projection-only samples into silver/abstention rows."""

    oracle = sample.get("oracle_projection") or {}
    response = sample.get("response_projection") or {}
    reset = sample.get("reset") or {}
    evidence = sample.get("evidence") or {}
    signals = oracle.get("signals") or {}
    family = family_class(sample.get("family"), sample.get("surface_id"))
    surface_role = {"sql": "sql_surface", "dom": "dom_surface", "redirect": "redirect_surface", "logic": "logic_surface"}.get(family, "generic_surface")
    row = {
        "source": "pg51_pikachu_docker_dual_channel",
        "seed": int(sample.get("sampling_seed", 51) or 51),
        "surface_role": surface_role,
        "method": str(sample.get("method", "GET")).upper(),
        "status_class": str(response.get("status_class", "unknown")),
        "field_count": int((response.get("shape") or {}).get("field_count", 0) or 0),
        "history_len": 0,
        "fresh_reset_ok": bool(reset.get("fresh_target", False)),
        "reset_completed": bool(reset.get("completed", False)),
        "reset_not_attempted": False,
        "candidate_sent": True,
        "oracle_available": False,
        "typed_effect_confirmed": False,
        "typed_effect_observed": False,
        "result_fixture_verified": False,
        "candidate_reference_agreement": False,
        "negative_clean": not bool(oracle.get("positive", False)),
        "binding_valid": True,
        "transport_error": bool(response.get("transport_error", False)),
        "result_mismatch_observed": bool(response.get("state_changed", False) or response.get("status_changed", False)),
        "next_step": "recheck_oracle",
        "previous_feedback": "none",
        "candidate_result_present": bool((signals.get("marker_reflected") or signals.get("sql_error_shape") or signals.get("external_redirect"))),
        "candidate_sql_error_shape": bool(signals.get("sql_error_shape", False)),
        "boolean_differential": False,
        "negative_result_absent": not bool(oracle.get("positive", False)),
        "hard_gate_observed": False,
        "backend_observed": True,
        "database_health_ok": True,
        "reference_sent": True,
        "negative_sent": sample.get("pair_role") == "control",
        "model_abstained": True,
        "model_claimed_positive": False,
        "evidence_hash": str(evidence.get("evidence_hash") or sample.get("source_sha256", "")),
        "payload_grounded_eligible": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    if len(row["evidence_hash"]) != 64:
        row["evidence_hash"] = digest({"source": row["source"], "sample_id": sample.get("sample_id", "")})
    return add_family_context(prepare_feedback_record(row), family=family, channel=sample.get("channel_reference"), pair_role=sample.get("pair_role"), source_role="evaluation_only")


__all__ = ["FAMILIES", "PG233_SCHEMA", "add_family_context", "family_class", "prepare_pikachu_sample"]

