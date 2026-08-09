"""PG-286 live observation collection boundary.

The target-side runner supplies bounded projections for a single authorized
GET/POST surface.  This module combines the existing PG-284 typed replay gate
with the shared PG-286 observation-token projection.  It never sends a
request, never stores a literal probe or response body, and never treats a
shape delta as a vulnerability label.

The resulting record is deliberately collection-only.  A later promotion
step may use a reviewed, cross-seed catalog, but one live record cannot become
training gold or long-term memory on its own.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .pg284_evaluator_contract import evaluate_typed_replay
from .pg286_observation_tokens import build_observation_tokens


SCHEMA_VERSION = "pg286-live-observation-collection-v1"
RAW_KEYS = frozenset(
    {
        "payload",
        "raw_payload",
        "payload_value",
        "probe_value",
        "request_body",
        "response_body",
        "raw_response",
        "body_text",
        "html",
        "query_value",
        "form_value",
        "cookie",
        "authorization_header",
        "credential",
        "location",
        "url",
    }
)
MODALITY_KEYS: dict[str, frozenset[str]] = {
    "dom_effect": frozenset(
        {
            "browser_dom_observed",
            "marker_hits",
            "body_text_hits",
            "element_count",
            "script_tag_count",
            "network_access",
            "navigation",
        }
    ),
    "sql_ast_shape": frozenset(
        {
            "kind",
            "interpreter_boundary",
            "boundary",
            "timing_differential",
            "timeout_observed",
            "row_shape_differential",
        }
    ),
    "redirect_hop": frozenset({"hop_count", "same_origin", "terminal_status", "chain_shape"}),
    "logic_transition": frozenset(
        {"transition_delta", "scope_changed", "authorization_changed", "visibility_changed", "state_changed"}
    ),
    "result_shape": frozenset({"result_shape_observed", "row_count_bucket", "schema_changed"}),
}
MODALITY_TO_ARGUMENT = {
    "dom_effect": "dom",
    "sql_ast_shape": "sql_ast",
    "redirect_hop": "redirect",
    "logic_transition": "logic",
    "result_shape": "logic",
}
FIELD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _contains_raw(value: Any, key: str = "") -> bool:
    if key.casefold() in RAW_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw(child, key) for child in value)
    return False


def _bounded_modality(value: Mapping[str, Any] | None, modality: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-286 {modality} projection must be an object")
    if _contains_raw(value):
        raise ValueError(f"PG-286 {modality} projection contains raw material")
    allowed = MODALITY_KEYS[modality]
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"PG-286 {modality} projection contains unsupported fields: {', '.join(unknown)}")
    result = {str(key): value[key] for key in allowed if key in value}
    if not result:
        raise ValueError(f"PG-286 {modality} projection is empty")
    return result


def _observation_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    """Convert PG-284's bounded projection into the token projector shape."""

    hops = max(0, min(int(projection.get("redirect_hops", 0) or 0), 8))
    marker = str(projection.get("effect_marker", "none"))
    return {
        "status_class": projection.get("status_class", "other"),
        "content_type_class": "unknown",
        "body_length_bucket": "unknown",
        "transport_error": not bool(projection.get("backend_observed", False)),
        "status_changed": False,
        "state_changed": False,
        "location_origin_changed": hops > 0,
        "marker_reflected": marker != "none",
        "marker": {"reflected": marker != "none", "location": "bounded" if marker != "none" else "none"},
        "redirect_chain": ["bounded"] * hops,
        "shape": {"kind": "unknown", "field_count": 0, "scalar_count": 0},
    }


def _modality_for_tokens(modality: str, projection: dict[str, Any]) -> dict[str, Any] | None:
    if modality == "result_shape":
        # Result-shape records still require a typed modality, but their
        # coarse signal is represented as logic-neutral state geometry.
        return {
            "transition_delta": "metadata",
            "state_changed": bool(projection.get("schema_changed", False)),
            "visibility_changed": bool(projection.get("result_shape_observed", False)),
        }
    return projection


def collect_pg286_live_record(
    *,
    record_id: str,
    surface: Mapping[str, Any],
    reset: Mapping[str, Any],
    baseline: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replay: Mapping[str, Any],
    typed_evidence: Mapping[str, Any],
    remote_probe: Mapping[str, Any],
    fields: list[str] | None = None,
    modality_projection: Mapping[str, Any] | None = None,
    hard_negative: bool = False,
    operator_reviewed: bool = False,
) -> dict[str, Any]:
    """Validate and tokenize one target-side live observation.

    ``typed_evidence.effect_type`` selects which bounded modality is required,
    but that label is never copied into ``context_tokens``.  The caller must
    provide the matching projection separately; otherwise the record is
    incomplete and quarantined.
    """

    if not isinstance(record_id, str) or not record_id or len(record_id) > 160:
        raise ValueError("PG-286 record_id is invalid")
    if _contains_raw({"surface": surface, "reset": reset, "baseline": baseline, "reference": reference, "negative": negative, "candidate": candidate, "replay": replay, "typed_evidence": typed_evidence, "modality_projection": modality_projection}):
        raise ValueError("PG-286 live input contains raw material")
    if not isinstance(baseline, Mapping):
        raise ValueError("PG-286 baseline projection is required")
    if not isinstance(typed_evidence, Mapping):
        raise ValueError("PG-286 typed evidence is required")
    effect_type = str(typed_evidence.get("effect_type", ""))
    if effect_type not in MODALITY_KEYS:
        raise ValueError("PG-286 typed evidence effect_type is not allow-listed")
    bounded_modality = _bounded_modality(modality_projection, effect_type)
    observed_fields = [str(field) for field in (fields or [])]
    if len(observed_fields) > 32 or any(not FIELD_RE.fullmatch(field) for field in observed_fields):
        raise ValueError("PG-286 fields must be short parameter names")

    evaluator = evaluate_typed_replay(
        surface=surface,
        reset=reset,
        reference=reference,
        negative=negative,
        candidate=candidate,
        replay=replay,
        typed_evidence=typed_evidence,
        remote_probe=remote_probe,
        hard_negative=hard_negative,
    )
    modality_arg = MODALITY_TO_ARGUMENT[effect_type]
    kwargs: dict[str, Any] = {
        "method": evaluator["surface"]["method"],
        "fields": observed_fields,
        "baseline": _observation_projection(dict(baseline)),
        "candidate": _observation_projection(evaluator["projections"]["candidate"]),
        "negative": _observation_projection(evaluator["projections"]["negative"]),
    }
    if effect_type == "sql_ast_shape":
        kwargs["sql_response"] = kwargs["candidate"]
    if bounded_modality is not None:
        kwargs[modality_arg] = _modality_for_tokens(effect_type, bounded_modality)
    token_result = build_observation_tokens(**kwargs)
    token_complete = token_result["evidence_status"] == "complete"
    remote_available = remote_probe.get("status") == "available"
    evaluator_confirmed = evaluator["status"] == "confirmed_effect"
    collection_complete = bool(token_complete and remote_available and evaluator_confirmed and not hard_negative)
    reasons = list(evaluator.get("reasons") or [])
    if not token_complete:
        reasons.append("observation_modality_missing")
    if not operator_reviewed:
        reasons.append("operator_review_required")
    if collection_complete and operator_reviewed:
        decision = "eligible_for_cross_seed_review"
    else:
        decision = "quarantine"
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "surface": evaluator["surface"],
        "reset": evaluator["reset"],
        "context_tokens": token_result["context_tokens"],
        "token_evidence_status": token_result["evidence_status"],
        "missing_modalities": token_result["missing_modalities"],
        "sql_ast_available": token_result["sql_ast_available"],
        "typed_effect_type": effect_type,
        "field_roles": sorted({token.split("=", 1)[1] for token in token_result["context_tokens"] if token.startswith("field_role=")}),
        "evaluator_status": evaluator["status"],
        "checks": evaluator["checks"],
        "hard_negative": bool(hard_negative),
        "operator_reviewed": bool(operator_reviewed),
        "decision": decision,
        "reasons": list(dict.fromkeys(reasons)),
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "evidence_hash": evaluator["evidence_projection_sha256"],
        "promotion": {
            "collection_complete": collection_complete,
            "cross_seed_review_required": True,
            "reason": "单条 live record 不得直接训练；需要跨 seed/source、hard-negative 与独立审计。",
        },
    }
    record["record_sha256"] = sha256_json({key: value for key, value in record.items() if key != "record_sha256"})
    return record


__all__ = ["SCHEMA_VERSION", "collect_pg286_live_record", "sha256_json"]
