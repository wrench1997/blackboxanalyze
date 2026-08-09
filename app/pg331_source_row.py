"""Strict PG-331A source-row collection for whole-web token experiments.

The collector is deliberately narrower than a crawler.  It accepts an already
de-identified structural observation produced by an authorised local adapter,
turns it into the ontology tokenizer stream, and keeps evaluator/provenance
sidecars outside model context.  It does not send requests, start containers,
or retain payloads, response bodies, routes, or source code.

An incomplete row is still useful as an ``ASK`` diagnostic, but it can never be
marked training-eligible.  This is the first gate before PG-331 information
preservation and capacity audits.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pg331_web_tokenizer import ONTOLOGY_PATH, STATIC_AXIS_ORDER, tokenize_web_observation


SCHEMA_VERSION = "pg331-whole-web-source-row-v1"
SPLITS = frozenset({"train", "dev", "route_holdout", "family_holdout", "implementation_holdout", "unassigned"})
HEX_SHA256 = re.compile(r"^[a-f0-9]{64}$")
SYMBOL = re.compile(r"^[a-z0-9_.:-]{1,64}$")

SOURCE_META_KEYS = frozenset(
    {
        "source_id",
        "implementation",
        "family_id",
        "surface_id",
        "collector_id",
        "authorization_id",
        "image_digest",
        "source_digest",
    }
)
RESET_KEYS = frozenset(
    {
        "fresh_reset",
        "reset_id",
        "target_instance_digest",
        "network_mode",
        "external_network",
        "loopback_only",
        "state_clean",
        "database_health_gate",
    }
)
EVALUATOR_KEYS = frozenset(
    {
        "typed_available",
        "negative_control",
        "reference_present",
        "candidate_present",
        "fresh_reset",
        "evidence_hash",
        "confirmed_positive",
        "effect_class",
        "evaluator_version",
    }
)
TARGET_KEYS = (
    "question",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "probe_variant_ref",
    "safe_to_send",
)
# Optional append-only target slots.  Keeping them outside TARGET_KEYS
# preserves compatibility with earlier PG-331 rows while newer collectors can
# teach the last-hop binder which abstract probe shape/oracle/negative-control
# state to use.  None of these slots contains a concrete payload string.
# PG-361 adds an abstract grammar class.  It remains append-only so old
# PG-349/350 rows can still be audited, but new candidates must provide it
# before the evaluator binder can bind a concrete wire.
OPTIONAL_TARGET_KEYS = ("syntax_category_ref", "payload_shape_ref", "oracle_ref", "negative_control_presence_ref")
AXIS_PRESENCE_KEYS = frozenset(
    {
        "document_presence",
        "navigation_presence",
        "request_transport_presence",
        "response_transport_presence",
        "javascript_presence",
        "failure_feedback_presence",
        "belief_replay_presence",
    }
)
FIELD_STATUS = frozenset({"observed", "absent", "not_observed", "unknown"})


def _declared_fields() -> dict[str, tuple[str, ...]]:
    document = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8-sig"))
    return {str(axis): tuple(str(field) for field in list(spec.get("fields") or [])) for axis, spec in dict(document.get("axes") or {}).items()}
TARGET_ALLOWED = {
    "question": frozenset({"none", "ask_transport", "ask_parameter_role", "ask_encoding", "ask_response", "ask_failure", "ask_belief", "ask_typed", "ask_typed_oracle"}),
    "next_action": frozenset({"ask", "ask_typed", "assemble_rule_ir", "assemble_abstract_plan", "select_probe_variant", "send_probe", "repair", "replay", "abstain"}),
    "repair_action": frozenset({"none", "encoding", "channel", "field", "method", "observe", "reset", "unknown"}),
    # These are abstract, allow-listed payload-shape coordinates.  They are
    # references for the local evaluator adapter, never raw wire values.
    "transport_ref": frozenset({"none", "request_method", "request_placement", "surface_method", "get_surface", "post_surface", "get_query", "get_path", "get_fragment", "post_form", "post_json", "unknown"}),
    "field_role_ref": frozenset({"none", "parameter_role", "surface_field_role", "display_text", "display_preference", "query_term", "query_text", "attribute_value", "path_segment", "fragment_identifier", "json_value", "dom_text", "form_field", "filter_choice", "sort_direction", "record_cursor", "profile_key", "view_mode", "tab_name", "step_index", "metric_group", "note_text", "notice_state", "status_label", "list_item", "static_label", "unknown"}),
    "encoding_ref": frozenset({"none", "encoding_chain", "surface_encoding", "identity", "url_percent", "form_urlencode", "form_urlencoded", "json_string", "json_object", "fragment", "query_parameter", "utf8", "form_urlencoded_then_url_percent", "form_urlencoded_then_utf8", "fragment_then_utf8", "json_object_then_utf8", "query_parameter_then_url_percent", "query_parameter_then_url_percent_then_utf8", "query_parameter_then_utf8", "unknown"}),
    "probe_variant_ref": frozenset({"none", "source_attested_candidate", "negative_control", "reference", "runtime_canary", "unknown"}),
    "payload_shape_ref": frozenset({
        "none", "unknown", "html_text_marker", "html_attribute_marker", "html_dom_marker", "html_fragment_marker",
        "html_form_marker", "json_string_marker", "path_segment_marker", "query_marker", "fragment_marker",
        "script_context_marker", "style_context_marker", "xml_text_marker", "xml_attribute_marker",
        "sql_string_marker", "sql_numeric_marker", "header_marker", "state_transition_marker",
    }),
    "syntax_category_ref": frozenset({
        "none", "unknown", "marker", "delimiter_boundary", "structured_value",
        "expression_node", "boolean_branch", "parser_node", "state_transition", "redirect_control",
    }),
    "oracle_ref": frozenset({"none", "unknown", "reflection", "response_shape", "parser_shape", "dom_shape", "typed_state_delta", "typed_effect", "negative_no_effect"}),
    "negative_control_presence_ref": frozenset({"unknown", "not_observed", "not_required", "matched_triplet"}),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-331A {name} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str] | frozenset[str], name: str) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"PG-331A {name} contains unsupported fields: {', '.join(unknown)}")


def _required_text(value: Any, name: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length or "\n" in value or "\r" in value:
        raise ValueError(f"PG-331A {name} is required")
    return value


def _required_digest(value: Any, name: str) -> str:
    text = str(value or "")
    if not HEX_SHA256.fullmatch(text):
        raise ValueError(f"PG-331A {name} must be a lowercase SHA-256 digest")
    return text


def _validate_source_meta(value: Any) -> dict[str, Any]:
    meta = _require_mapping(value, "source_meta")
    _reject_unknown(meta, SOURCE_META_KEYS, "source_meta")
    result = {key: meta[key] for key in meta}
    for key in ("source_id", "implementation", "collector_id", "authorization_id"):
        result[key] = _required_text(meta.get(key), f"source_meta.{key}")
    for key in ("image_digest", "source_digest"):
        result[key] = _required_digest(meta.get(key), f"source_meta.{key}")
    for key in ("family_id", "surface_id"):
        if key in meta:
            result[key] = _required_text(meta[key], f"source_meta.{key}")
    return result


def _validate_reset(value: Any) -> dict[str, Any]:
    reset = _require_mapping(value, "reset")
    _reject_unknown(reset, RESET_KEYS, "reset")
    if reset.get("fresh_reset") is not True:
        raise ValueError("PG-331A reset.fresh_reset must be true")
    if reset.get("external_network") is not False:
        raise ValueError("PG-331A reset.external_network must be false")
    if reset.get("loopback_only") is not True:
        raise ValueError("PG-331A reset.loopback_only must be true")
    if reset.get("state_clean") is not True:
        raise ValueError("PG-331A reset.state_clean must be true")
    if str(reset.get("network_mode", "")) not in {"none", "loopback"}:
        raise ValueError("PG-331A reset.network_mode must be none or loopback")
    result = dict(reset)
    result["reset_id"] = _required_text(reset.get("reset_id"), "reset.reset_id", max_length=256)
    result["target_instance_digest"] = _required_digest(reset.get("target_instance_digest"), "reset.target_instance_digest")
    if "database_health_gate" in reset:
        result["database_health_gate"] = _required_text(reset.get("database_health_gate"), "reset.database_health_gate", max_length=64)
        if result["database_health_gate"] not in {"mysqli_root_pikachu_ok", "juice_shop_http_health_ok", "database_health_ok", "not_applicable"}:
            raise ValueError("PG-331A reset.database_health_gate is not allow-listed")
    return result


def _validate_evaluator(value: Any) -> dict[str, Any]:
    evaluator = _require_mapping(value, "evaluator")
    _reject_unknown(evaluator, EVALUATOR_KEYS, "evaluator")
    for key in ("typed_available", "negative_control", "reference_present", "candidate_present", "fresh_reset", "confirmed_positive"):
        if key in evaluator and not isinstance(evaluator[key], bool):
            raise ValueError(f"PG-331A evaluator.{key} must be boolean")
    result = dict(evaluator)
    result["evidence_hash"] = _required_digest(evaluator.get("evidence_hash"), "evaluator.evidence_hash")
    result["evaluator_version"] = _required_text(evaluator.get("evaluator_version"), "evaluator.evaluator_version")
    # The effect class is an evaluator-side projection.  It is allowed in the
    # sidecar but is never copied into context tokens.
    if "effect_class" in evaluator:
        result["effect_class"] = _required_text(evaluator["effect_class"], "evaluator.effect_class")
    return result


def _validate_field_capture_manifest(value: Any) -> dict[str, dict[str, str]]:
    """Require an explicit status for every ontology-declared field.

    ``absent`` means the adapter actually observed that the field does not
    exist (for example, a page has no form); ``not_observed``/``unknown`` means
    the collector lacks the observation and must drive an ASK target.
    """

    manifest = _require_mapping(value, "field_capture_manifest")
    declared = _declared_fields()
    if set(str(key) for key in manifest) != set(declared):
        raise ValueError("PG-331A field_capture_manifest must cover exactly every ontology axis")
    result: dict[str, dict[str, str]] = {}
    for axis, fields in declared.items():
        axis_manifest = _require_mapping(manifest.get(axis), f"field_capture_manifest.{axis}")
        if set(str(key) for key in axis_manifest) != set(fields):
            raise ValueError(f"PG-331A field_capture_manifest.{axis} must cover every declared field")
        normalized: dict[str, str] = {}
        for field in fields:
            status = str(axis_manifest.get(field, "")).casefold()
            if status not in FIELD_STATUS:
                raise ValueError(f"PG-331A field_capture_manifest.{axis}.{field} has invalid status")
            normalized[field] = status
        result[axis] = normalized
    return result


def _validate_target(value: Any) -> dict[str, Any]:
    target = _require_mapping(value, "target_projection")
    _reject_unknown(target, set(TARGET_KEYS) | set(OPTIONAL_TARGET_KEYS), "target_projection")
    result: dict[str, Any] = {}
    for key in TARGET_KEYS:
        if key not in target:
            result[key] = "none" if key in {"question", "repair_action", "transport_ref", "field_role_ref", "encoding_ref", "probe_variant_ref"} else (False if key == "safe_to_send" else "abstain")
            continue
        raw = target[key]
        if key == "safe_to_send":
            if not isinstance(raw, bool):
                raise ValueError("PG-331A target_projection.safe_to_send must be boolean")
            result[key] = raw
            continue
        text = _required_text(raw, f"target_projection.{key}", max_length=64).casefold().replace("-", "_")
        if not SYMBOL.fullmatch(text):
            raise ValueError(f"PG-331A target_projection.{key} must be an abstract symbol")
        allowed = TARGET_ALLOWED.get(key)
        if allowed is not None and text not in allowed:
            raise ValueError(f"PG-331A target_projection.{key} is not allow-listed")
        result[key] = text
    for key in OPTIONAL_TARGET_KEYS:
        if key not in target:
            continue
        raw = target[key]
        text = _required_text(raw, f"target_projection.{key}", max_length=64).casefold().replace("-", "_")
        if not SYMBOL.fullmatch(text):
            raise ValueError(f"PG-331A target_projection.{key} must be an abstract symbol")
        allowed = TARGET_ALLOWED.get(key)
        if allowed is not None and text not in allowed:
            raise ValueError(f"PG-331A target_projection.{key} is not allow-listed")
        result[key] = text
    return result


def _target_tokens(target: Mapping[str, Any]) -> list[str]:
    tokens = ["[TARGET_BOS]"]
    for key in TARGET_KEYS:
        value = target[key]
        if key == "safe_to_send":
            value = int(bool(value))
        tokens.append(f"{key}={value}")
    for key in OPTIONAL_TARGET_KEYS:
        if key in target:
            tokens.append(f"{key}={target[key]}")
    tokens.append("[TARGET_EOS]")
    return tokens


def _ask_target_for_missing(target: Mapping[str, Any], failures: Sequence[str]) -> dict[str, Any]:
    """Prevent an incomplete observation from being labelled as a solution."""

    missing_observation = any(
        item.startswith(
            (
                "missing_presence:",
                "axis_not_observed:",
                "field_not_observed:",
                "field_unknown:",
                "tokenizer_loss",
                "evaluator_missing:",
                "fresh_reset",
                "raw_fields_in_observation",
                "context_firewall",
            )
        )
        for item in failures
    )
    if not missing_observation and "failure_action_not_changed" not in failures:
        return dict(target)
    repaired = dict(target)
    if "failure_action_not_changed" in failures and not missing_observation:
        repaired.update({"question": "ask_failure", "next_action": "repair", "repair_action": "observe", "safe_to_send": False})
        return repaired
    repaired.update(
        {
            "question": "ask_typed",
            "next_action": "ask_typed",
            "repair_action": "observe",
            "transport_ref": "unknown",
            "field_role_ref": "unknown",
            "encoding_ref": "unknown",
            "probe_variant_ref": "none",
            "safe_to_send": False,
        }
    )
    # Do not leave a positive oracle/negative-control target attached to an
    # incomplete observation.  If a legacy row had these append-only slots,
    # make the missingness explicit so the model learns to ASK.
    if "syntax_category_ref" in repaired:
        repaired["syntax_category_ref"] = "unknown"
    if "oracle_ref" in repaired:
        repaired["oracle_ref"] = "unknown"
    if "negative_control_presence_ref" in repaired:
        repaired["negative_control_presence_ref"] = "unknown"
    return repaired


def _presence_tokens(context_tokens: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in context_tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in AXIS_PRESENCE_KEYS:
            result[key] = value
    return result


def _field_tokens(context_tokens: Sequence[str]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    declared = _declared_fields()
    prefixes = {f"{axis}_field_": axis for axis in declared}
    for token in context_tokens:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        for prefix, axis in prefixes.items():
            if key.startswith(prefix):
                result[(axis, key[len(prefix) :])] = value
                break
    return result


def _context_forbidden(context_tokens: Sequence[str]) -> list[str]:
    forbidden_fragments = (
        "raw_",
        "payload",
        # Structural tokens such as response_body_length/shape are allowed;
        # only a literal body side-channel key is forbidden.
        "response_body=",
        "response_body_text=",
        "family=",
        "family_label=",
        "route_literal=",
        "route_name=",
        "surface_id=",
        "oracle=",
        "evaluator=",
        "typed_effect=",
        "expected_answer=",
        "target_answer=",
    )
    return [str(token) for token in context_tokens if any(fragment in str(token).casefold() for fragment in forbidden_fragments)]


def collect_pg331_source_row(
    *,
    record_id: str,
    observation: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    reset: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    field_capture_manifest: Mapping[str, Any],
    target_projection: Mapping[str, Any],
    split: str = "unassigned",
    operator_reviewed: bool = False,
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Collect one strict row from a local adapter's abstract observation.

    The function is intentionally pure: the caller must already have obtained
    the observation from an authorised local/loopback adapter.  Missing axes
    remain visible as ``not_observed`` in tokenizer output, but block training.
    """

    record_id = _required_text(record_id, "record_id", max_length=256)
    if split not in SPLITS:
        raise ValueError("PG-331A split is not allow-listed")
    observation = _require_mapping(observation, "observation")
    unknown_axes = sorted(str(key) for key in observation if str(key) not in STATIC_AXIS_ORDER)
    if unknown_axes:
        raise ValueError(f"PG-331A observation contains unsupported axes: {', '.join(unknown_axes)}")
    meta = _validate_source_meta(source_meta)
    reset_projection = _validate_reset(reset)
    evaluator_projection = _validate_evaluator(evaluator)
    field_manifest = _validate_field_capture_manifest(field_capture_manifest)
    target = _validate_target(target_projection)

    tokenized = tokenize_web_observation(observation)
    context_tokens = [str(token) for token in tokenized["context_tokens"]]
    presence = _presence_tokens(context_tokens)
    expected_presence = AXIS_PRESENCE_KEYS
    failures: list[str] = []
    failures.extend(f"missing_presence:{key}" for key in sorted(expected_presence - set(presence)))
    failures.extend(f"axis_not_observed:{key}" for key in sorted(expected_presence & {key for key, value in presence.items() if value != "observed"}))
    if tokenized["loss_report"].get("raw_fields_omitted"):
        failures.append("raw_fields_in_observation")
    if tokenized["loss_report"].get("losses"):
        failures.append("tokenizer_loss")
    field_tokens = _field_tokens(context_tokens)
    for axis, fields in field_manifest.items():
        for field, status in fields.items():
            if status == "not_observed":
                failures.append(f"field_not_observed:{axis}.{field}")
            elif status == "unknown":
                failures.append(f"field_unknown:{axis}.{field}")
            token_value = field_tokens.get((axis, field))
            if token_value is None or (status in {"observed", "absent"} and token_value in {"not_observed", "unknown"}) or (status in {"not_observed", "unknown"} and token_value != status):
                failures.append(f"field_manifest_mismatch:{axis}.{field}")
    failure_observation = observation.get("failure_feedback")
    if isinstance(failure_observation, Mapping):
        failure_class = str(failure_observation.get("failure_class", "unknown")).casefold()
        if failure_class not in {"", "none", "unknown"}:
            previous_action = str(failure_observation.get("previous_action", ""))
            next_action = str(failure_observation.get("next_action", ""))
            if not previous_action or not next_action or previous_action == next_action:
                failures.append("failure_action_not_changed")
    forbidden = _context_forbidden(context_tokens)
    if forbidden:
        failures.append("context_firewall")
    # Evaluator completeness is a gate, not a target label.  A row may remain
    # in the diagnostic catalog if it is incomplete, but cannot train.
    for key in ("typed_available", "negative_control", "reference_present", "candidate_present", "fresh_reset"):
        if evaluator_projection.get(key) is not True:
            failures.append(f"evaluator_missing:{key}")
    if reset_projection.get("fresh_reset") is not True or evaluator_projection.get("fresh_reset") is not True:
        failures.append("fresh_reset")

    target = _ask_target_for_missing(target, failures)
    target_tokens = _target_tokens(target)
    training_eligible = bool(operator_reviewed and not hard_negative and not failures)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "split": split,
        "source_meta": meta,
        "reset": reset_projection,
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "target_projection": target,
        "field_capture_manifest": field_manifest,
        "evaluator_sidecar": evaluator_projection,
        "tokenizer": {
            "schema_version": str(tokenized.get("schema_version", "")),
            "ontology_sha256": str(tokenized.get("ontology_sha256", "")),
            "loss_report": tokenized.get("loss_report", {}),
        },
        "axis_presence": presence,
        "hard_negative": bool(hard_negative),
        "operator_reviewed": bool(operator_reviewed),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "context_firewall": {"forbidden_token_count": len(forbidden), "sidecars_off_context": True},
        "training_eligible": training_eligible,
        "promotion": {
            "training_eligible": training_eligible,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "cross_source_review_required": True,
        },
        "failures": sorted(set(failures)),
    }
    row["record_sha256"] = sha256_json(row)
    return row


def validate_pg331_source_row(row: Mapping[str, Any], *, require_training_eligible: bool = False) -> dict[str, Any]:
    """Validate a serialized row without contacting a target or evaluator."""

    if not isinstance(row, Mapping):
        return {"valid": False, "failures": ["row_not_mapping"]}
    failures: list[str] = []
    if str(row.get("schema_version", "")) != SCHEMA_VERSION:
        failures.append("schema_version")
    context = [str(token) for token in row.get("context_tokens") or []]
    expected_presence = AXIS_PRESENCE_KEYS
    presence = _presence_tokens(context)
    failures.extend(f"missing_presence:{key}" for key in sorted(expected_presence - set(presence)))
    failures.extend(f"axis_not_observed:{key}" for key in sorted(expected_presence & {key for key, value in presence.items() if value != "observed"}))
    failures.extend(["context_firewall"] if _context_forbidden(context) else [])
    if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False:
        failures.append("raw_storage_flag")
    if row.get("oracle_answer_in_context") is not False:
        failures.append("oracle_context_flag")
    row_failures = [str(item) for item in row.get("failures") or []]
    try:
        _validate_source_meta(row.get("source_meta"))
        reset_projection = _validate_reset(row.get("reset"))
        evaluator_projection = _validate_evaluator(row.get("evaluator_sidecar"))
        for key in ("typed_available", "negative_control", "reference_present", "candidate_present", "fresh_reset"):
            if evaluator_projection.get(key) is not True:
                failures.append(f"evaluator_missing:{key}")
        if reset_projection.get("fresh_reset") is not True or evaluator_projection.get("fresh_reset") is not True:
            failures.append("fresh_reset")
    except (TypeError, ValueError):
        failures.append("sidecar_schema")
    try:
        field_manifest = _validate_field_capture_manifest(row.get("field_capture_manifest"))
        field_tokens = _field_tokens(context)
        for axis, fields in field_manifest.items():
            for field, status in fields.items():
                if status in {"not_observed", "unknown"}:
                    failures.append(f"field_not_ready:{axis}.{field}")
                token_value = field_tokens.get((axis, field))
                if token_value is None or (status in {"observed", "absent"} and token_value in {"not_observed", "unknown"}) or (status in {"not_observed", "unknown"} and token_value != status):
                    failures.append(f"field_manifest_mismatch:{axis}.{field}")
    except (TypeError, ValueError):
        failures.append("field_capture_manifest")
    try:
        target = _validate_target(row.get("target_projection"))
        if [str(token) for token in row.get("target_tokens") or []] != _target_tokens(target):
            failures.append("target_token_projection_mismatch")
        expected_target = _ask_target_for_missing(target, [*failures, *row_failures])
        if expected_target != target:
            failures.append("unsafe_target_on_incomplete_row")
    except (TypeError, ValueError):
        failures.append("target_projection")
    if row.get("training_eligible") is True and (row.get("operator_reviewed") is not True or row.get("hard_negative") is True or row_failures or failures):
        failures.append("training_eligibility_mismatch")
    expected_hash = str(row.get("record_sha256", ""))
    if not HEX_SHA256.fullmatch(expected_hash):
        failures.append("record_sha256")
    else:
        body = dict(row)
        body.pop("record_sha256", None)
        if sha256_json(body) != expected_hash:
            failures.append("record_hash_mismatch")
    if require_training_eligible and row.get("training_eligible") is not True:
        failures.append("training_not_eligible")
    return {"valid": not failures, "failures": sorted(set(failures)), "training_eligible": bool(row.get("training_eligible")), "axis_presence": presence}


__all__ = ["SCHEMA_VERSION", "collect_pg331_source_row", "sha256_json", "validate_pg331_source_row"]
