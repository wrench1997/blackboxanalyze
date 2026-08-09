"""Evaluator-side typed evidence for PG-331 source rows.

This module is deliberately independent from the PG-331 source-row collector.
It consumes bounded evaluator projections from an authorised local adapter and
returns two explicitly separated views:

* ``evaluator_sidecar`` keeps the candidate/reference/negative evidence and
  the fresh-reset attestation for an operator/auditor;
* ``model_context`` contains only abstract availability/presence booleans.

Literal payloads, response bodies and oracle answers are never copied to
either output.  Passing a value under a raw/evaluator-only key is a hard
failure, which makes accidental context leakage fail closed.  No request,
container, browser or training operation is performed here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg331-evaluator-sidecar-v1"
RECORD_SCHEMA_VERSION = "pg331-evaluator-record-v1"
ROLES = ("candidate", "reference", "negative")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,256}$")
SYMBOL_RE = re.compile(r"^[a-z0-9_.:-]{1,80}$")

# These are evaluator-side input names.  They are intentionally rejected
# rather than copied into a result, including nested values and list items.
RAW_KEYS = frozenset(
    {
        "payload",
        "raw_payload",
        "payload_value",
        "probe_value",
        "request_body",
        "request_value",
        "query_value",
        "form_value",
        "response_body",
        "raw_response",
        "body_text",
        "html",
        "markup",
        "raw_markup",
        "javascript_source",
        "source_code",
        "oracle_answer",
        "evaluator_answer",
        "expected_answer",
        "target_answer",
        "credential",
        "cookie_value",
        "authorization_value",
        "secret",
    }
)
RAW_FRAGMENTS = ("raw_", "payload", "response_body", "body_text", "oracle_answer", "evaluator_answer")
EFFECT_CLASSES = frozenset({"none", "result_shape", "dom_effect", "sql_ast_shape", "redirect_hop", "logic_transition", "unknown"})
NETWORK_MODES = frozenset({"none", "loopback"})

# PG-324 (challenge-state/DOM) and PG-325 (response-shape/SQL) projections
# use this common bounded vocabulary.  Unknown keys are not silently copied;
# this prevents a route/family/oracle label from becoming model input.
PROJECTION_KEYS = frozenset(
    {
        "status_class",
        "content_type_class",
        "body_shape",
        "body_length_bucket",
        "redirect_hop_count",
        "redirect_location_class",
        "redirect_chain_shape",
        "connection_outcome",
        "shape_sha256",
        "effect_marker",
        "effect_shape",
        "state_delta_class",
        "challenge_state_available",
        "challenge_state_baseline_available",
        "challenge_state_baseline_solved",
        "challenge_state_delta",
        "challenge_solved",
        "sink_present",
        "dom_script_execution",
        "script_execution",
        "backend_observed",
        "row_shape_changed",
        "response_shape_changed",
        "network_request_count",
        "external_network_blocked",
        "navigation_allowed",
        "database_touched",
        "disposable_state_delta",
        "non_destructive",
        "error_class",
        "error_shape",
    }
)

ROLE_INPUT_KEYS = frozenset(
    {
        "sent",
        "available",
        "executed",
        "typed_effect_confirmed",
        "effect_class",
        "effect_type",
        "projection",
        "evidence_sha256",
        "evidence_hash",
        "negative_control_clean",
        "reference_agreement",
        "replay_consistent",
        "non_destructive",
    }
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _contains_raw(value: Any, key: str = "") -> bool:
    lowered = str(key).casefold()
    if lowered in RAW_KEYS or lowered.startswith("raw_"):
        return True
    if any(fragment in lowered for fragment in RAW_FRAGMENTS):
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_raw(child, key) for child in value)
    return False


def _required_id(value: Any, name: str) -> str:
    text = str(value or "")
    if not ID_RE.fullmatch(text):
        raise ValueError(f"PG-331 evaluator {name} must be a bounded abstract id")
    return text


def _digest(value: Any, name: str) -> str:
    text = str(value or "").casefold()
    if not HASH_RE.fullmatch(text):
        raise ValueError(f"PG-331 evaluator {name} must be a lowercase SHA-256 digest")
    return text


def _optional_digest(value: Any) -> tuple[str, bool]:
    if value in (None, ""):
        return "", False
    text = str(value).casefold()
    return text, bool(HASH_RE.fullmatch(text))


def _symbol(value: Any, name: str, *, default: str = "unknown") -> str:
    if value in (None, ""):
        return default
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if not SYMBOL_RE.fullmatch(text):
        raise ValueError(f"PG-331 evaluator {name} must be an abstract symbol")
    return text


def _bool(value: Any) -> bool:
    return value is True


def _abstract_value(value: Any, name: str) -> Any:
    """Bound projection values without preserving literals.

    Projection values are never source text: booleans and small integers are
    retained as bounded shape facts; strings are allow-listed symbols.  A
    caller that needs a literal response must keep it in the evaluator process
    and pass only its typed projection here.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value > 4096:
            raise ValueError(f"PG-331 evaluator {name} integer is out of bounds")
        return int(value)
    if isinstance(value, float):
        if value != value or value < 0 or value > 4096:
            raise ValueError(f"PG-331 evaluator {name} number is out of bounds")
        return round(float(value), 6)
    if value is None:
        return "unknown"
    if isinstance(value, str):
        folded = value.casefold()
        if any(fragment in folded for fragment in RAW_FRAGMENTS):
            raise ValueError(f"PG-331 evaluator {name} contains a raw/literal marker")
        return _symbol(value, name)
    raise ValueError(f"PG-331 evaluator {name} must be abstract")


def _normalize_projection(value: Any) -> dict[str, Any]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("PG-331 evaluator projection must be an object")
    if _contains_raw(value):
        raise ValueError("PG-331 evaluator projection contains raw material")
    unknown = sorted(str(key) for key in value if str(key) not in PROJECTION_KEYS)
    if unknown:
        raise ValueError(f"PG-331 evaluator projection contains unsupported fields: {', '.join(unknown)}")
    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key)
        normalized[name] = _abstract_value(raw, f"projection.{name}")
    return normalized


def _normalize_reset(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("PG-331 evaluator reset must be an object")
    if _contains_raw(value):
        raise ValueError("PG-331 evaluator reset contains raw material")
    reset_id = _required_id(value.get("reset_id"), "reset_id")
    # PG-331 source rows call this fresh_reset; PG-324/325 call it
    # fresh_target/completed.  Do not infer freshness from a container id.
    fresh = value.get("fresh_reset")
    if fresh is None:
        fresh = value.get("fresh_target")
    if fresh is None:
        fresh = value.get("completed")
    target_digest = value.get("target_instance_digest")
    if target_digest in (None, ""):
        target_digest = value.get("container_id_sha256")
    target_digest = _digest(target_digest, "target_instance_digest")
    network_mode = str(value.get("network_mode", ""))
    if network_mode not in NETWORK_MODES:
        raise ValueError("PG-331 evaluator reset.network_mode must be none or loopback")
    external_network = value.get("external_network")
    loopback = value.get("loopback_only")
    if loopback is None:
        loopback = value.get("relay_loopback_only")
    state_clean = value.get("state_clean")
    if state_clean is None:
        state_clean = value.get("domain_data_write_allowed") is False
    # Missing safety attestations remain unknown; never turn absence into a
    # passing ``False``/``True`` default.  PG-324/325 supply the aliases above,
    # while a PG-331 reset must explicitly carry these booleans.
    if not isinstance(external_network, bool):
        external_network = "unknown"
    if not isinstance(loopback, bool):
        loopback = "unknown"
    if not isinstance(state_clean, bool):
        state_clean = "unknown"
    volume_count = value.get("volume_mount_count")
    if volume_count is None:
        volume_count = value.get("bind_or_volume_mount_count", 0)
    try:
        volume_count = int(volume_count)
    except (TypeError, ValueError) as error:
        raise ValueError("PG-331 evaluator reset volume count is invalid") from error
    if volume_count < 0 or volume_count > 64:
        raise ValueError("PG-331 evaluator reset volume count is out of bounds")
    normalized = {
        "reset_id": reset_id,
        "fresh_reset": _bool(fresh),
        "target_instance_digest": target_digest,
        "network_mode": network_mode,
        "external_network": external_network,
        "loopback_only": loopback,
        "state_clean": state_clean,
        "volume_mount_count": volume_count,
        "container_restart_used": _bool(value.get("container_restart_used")),
    }
    # Keep a bounded database health class where present; no database/route
    # literal is retained.
    if "database_health_gate" in value:
        normalized["database_health_gate"] = _symbol(value.get("database_health_gate"), "database_health_gate")
    return normalized


def _normalize_role(role: str, value: Mapping[str, Any], *, record_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"PG-331 evaluator {role} role must be an object")
    if _contains_raw(value):
        raise ValueError(f"PG-331 evaluator {role} role contains raw material")
    unknown = sorted(str(key) for key in value if str(key) not in ROLE_INPUT_KEYS)
    if unknown:
        raise ValueError(f"PG-331 evaluator {role} role contains unsupported fields: {', '.join(unknown)}")
    source_hash, source_hash_valid = _optional_digest(value.get("evidence_sha256", value.get("evidence_hash")))
    effect_confirmed = _bool(value.get("typed_effect_confirmed"))
    effect_class = value.get("effect_class", value.get("effect_type"))
    if effect_class in (None, ""):
        effect_class = "unknown" if effect_confirmed else "none"
    effect_class = _symbol(effect_class, f"{role}.effect_class")
    if effect_class not in EFFECT_CLASSES:
        raise ValueError(f"PG-331 evaluator {role}.effect_class is not allow-listed")
    projection = _normalize_projection(value.get("projection"))
    # Role-bound evidence prevents byte-identical SQL projections from
    # collapsing candidate/reference/negative belief steps.
    role_unsigned = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "role": role,
        "sent": _bool(value.get("sent")),
        "available": _bool(value.get("available")),
        "executed": _bool(value.get("executed")),
        "typed_effect_confirmed": effect_confirmed,
        "effect_class": effect_class,
        "projection": projection,
        "source_evidence_sha256": source_hash,
    }
    bound_hash = sha256_json(role_unsigned)
    return {
        "role": role,
        "sent": role_unsigned["sent"],
        "available": role_unsigned["available"],
        "executed": role_unsigned["executed"],
        "typed_effect_confirmed": effect_confirmed,
        "effect_class": effect_class,
        "projection": projection,
        "source_evidence_sha256": source_hash,
        "source_evidence_hash_valid": source_hash_valid,
        "evidence_scope": "record_role_bound",
        "evidence_sha256": bound_hash,
        "raw_payload_stored": False,
        "raw_response_stored": False,
        "oracle_answer_in_context": False,
    }


def _fresh_reset_ok(reset: Mapping[str, Any]) -> bool:
    return bool(
        reset.get("fresh_reset") is True
        and reset.get("external_network") is False
        and reset.get("loopback_only") is True
        and reset.get("state_clean") is True
        and reset.get("network_mode") in NETWORK_MODES
        and int(reset.get("volume_mount_count", -1)) == 0
        and reset.get("container_restart_used") is False
    )


def _context_firewall(value: Any) -> None:
    if _contains_raw(value):
        raise ValueError("PG-331 evaluator model context contains raw material")
    if isinstance(value, str):
        folded = value.casefold()
        if any(fragment in folded for fragment in RAW_FRAGMENTS) or any(item in folded for item in ("family=", "route_literal=", "oracle=", "evaluator=")):
            raise ValueError("PG-331 evaluator model context contains a forbidden literal")
        return
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (bytes, bytearray)):
        if value is not None:
            return
    if isinstance(value, Mapping):
        for key in value:
            lowered = str(key).casefold()
            if lowered in {"family", "family_label", "route_literal", "route_name", "oracle", "evaluator", "typed_effect", "expected_answer", "target_answer"}:
                raise ValueError(f"PG-331 evaluator model context contains forbidden key: {key}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _context_firewall(item)


def build_pg331_evaluator_sidecar(
    *,
    record_id: str,
    reset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    replay_consistent: bool = False,
    reference_agreement: bool | None = None,
    negative_control_clean: bool | None = None,
    evaluator_id: str = "pg331-evaluator-unknown",
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Build a typed evaluator-side sidecar without emitting wire material.

    Missing evidence is represented as an invalid source hash and a safe,
    non-confirmed sidecar; callers can therefore retain diagnostics without
    treating an incomplete replay as a positive.
    """

    record_id = _required_id(record_id, "record_id")
    evaluator_id = _required_id(evaluator_id, "evaluator_id")
    normalized_reset = _normalize_reset(reset)
    roles = {name: _normalize_role(name, value, record_id=record_id) for name, value in (("candidate", candidate), ("reference", reference), ("negative", negative))}
    if reference_agreement is None:
        reference_agreement = bool(
            roles["candidate"]["typed_effect_confirmed"]
            and roles["reference"]["typed_effect_confirmed"]
            and roles["candidate"]["effect_class"] == roles["reference"]["effect_class"]
        )
    if negative_control_clean is None:
        negative_control_clean = bool(not roles["negative"]["typed_effect_confirmed"])
    checks = {
        "candidate_present": roles["candidate"]["sent"],
        "reference_present": roles["reference"]["sent"],
        "negative_present": roles["negative"]["sent"],
        "candidate_available": roles["candidate"]["available"],
        "reference_available": roles["reference"]["available"],
        "negative_available": roles["negative"]["available"],
        "typed_effect": roles["candidate"]["typed_effect_confirmed"],
        "negative_control_clean": _bool(negative_control_clean) and not roles["negative"]["typed_effect_confirmed"],
        "reference_agreement": _bool(reference_agreement),
        "replay_consistent": _bool(replay_consistent),
        "fresh_reset": _fresh_reset_ok(normalized_reset),
        "evidence_hashes": all(role["source_evidence_hash_valid"] for role in roles.values()),
        "non_destructive": all(
            not bool(role["projection"].get("database_touched"))
            or (
                role["projection"].get("disposable_state_delta") is True
                and role["projection"].get("state_delta_class") == "disposable_evaluator_state"
                and role["projection"].get("external_network_blocked") is True
            )
            for role in roles.values()
        ),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    if hard_negative:
        reasons.insert(0, "hard_negative")
    reasons = list(dict.fromkeys(reasons))
    typed_effect_confirmed = bool(checks["typed_effect"])
    confirmed_positive = bool(all(checks.values()) and not hard_negative)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "evaluator_id": evaluator_id,
        "roles": roles,
        "reset": normalized_reset,
        "typed_effect_confirmed": typed_effect_confirmed,
        "effect_class": roles["candidate"]["effect_class"],
        "negative_control_clean": bool(checks["negative_control_clean"]),
        "reference_agreement": bool(checks["reference_agreement"]),
        "replay_consistent": bool(checks["replay_consistent"]),
        "checks": checks,
        "reasons": reasons,
        "hard_negative": bool(hard_negative),
    }
    evidence_sha256 = sha256_json(unsigned)
    sidecar = dict(unsigned)
    sidecar.update(
        {
            "evidence_sha256": evidence_sha256,
            "evidence_hash": evidence_sha256,
            "evidence_hash_valid": True,
            "confirmed_positive": confirmed_positive,
            "raw_payload_stored": False,
            "raw_response_stored": False,
            "oracle_answer_in_context": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        }
    )
    return sidecar


def build_pg331_evaluator_record(
    *,
    record_id: str,
    reset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    replay_consistent: bool = False,
    reference_agreement: bool | None = None,
    negative_control_clean: bool | None = None,
    evaluator_id: str = "pg331-evaluator-unknown",
    hard_negative: bool = False,
    context_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the evaluator sidecar plus a model-safe abstract projection."""

    sidecar = build_pg331_evaluator_sidecar(
        record_id=record_id,
        reset=reset,
        candidate=candidate,
        reference=reference,
        negative=negative,
        replay_consistent=replay_consistent,
        reference_agreement=reference_agreement,
        negative_control_clean=negative_control_clean,
        evaluator_id=evaluator_id,
        hard_negative=hard_negative,
    )
    checks = sidecar["checks"]
    complete_triplet = all(
        bool(checks[key])
        for key in (
            "candidate_present",
            "reference_present",
            "negative_present",
            "candidate_available",
            "reference_available",
            "negative_available",
        )
    )
    context = {
        # A candidate effect alone is never enough to advertise a typed lane;
        # all three roles, fresh reset, replay and evidence must be present.
        "typed_available": bool(
            complete_triplet
            and checks["typed_effect"]
            and checks["evidence_hashes"]
            and checks["fresh_reset"]
            and checks["replay_consistent"]
            and checks["negative_control_clean"]
            and checks["reference_agreement"]
        ),
        "evidence_present": bool(checks["evidence_hashes"] and complete_triplet),
        # An unsent negative is unknown, not clean.  This prevents a missing
        # control from being learned as a safe negative observation.
        "negative_control": bool(
            checks["negative_present"]
            and checks["negative_available"]
            and checks["negative_control_clean"]
        ),
        "fresh_reset": bool(checks["fresh_reset"]),
        "reference_present": bool(checks["reference_present"]),
        "candidate_present": bool(checks["candidate_present"]),
        "replay_ready": bool(
            checks["replay_consistent"]
            and checks["candidate_present"]
            and checks["reference_present"]
            and checks["negative_present"]
        ),
        "step_budget": "present",
    }
    if context_projection is not None:
        _context_firewall(context_projection)
        allowed = set(context)
        unknown = sorted(str(key) for key in context_projection if str(key) not in allowed)
        if unknown:
            raise ValueError(f"PG-331 evaluator context contains unsupported fields: {', '.join(unknown)}")
        for key, value in context_projection.items():
            if not isinstance(value, (bool, str, int)) or isinstance(value, float):
                raise ValueError(f"PG-331 evaluator context.{key} must be scalar")
            context[str(key)] = value
    _context_firewall(context)
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_id": str(record_id),
        "model_context": context,
        "evaluator_sidecar": sidecar,
        "raw_payload_stored": False,
        "raw_response_stored": False,
        "oracle_answer_in_context": False,
        "context_firewall": {"sidecars_off_context": True, "forbidden_token_count": 0},
        "training_eligible": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    record["record_sha256"] = sha256_json(record)
    return record


def validate_pg331_evaluator_sidecar(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Validate hashes and safety flags without contacting an evaluator."""

    failures: list[str] = []
    if not isinstance(sidecar, Mapping):
        return {"valid": False, "failures": ["sidecar_not_mapping"]}
    if str(sidecar.get("schema_version", "")) != SCHEMA_VERSION:
        failures.append("schema_version")
    for key in ("raw_payload_stored", "raw_response_stored", "oracle_answer_in_context", "training_eligible", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        if sidecar.get(key) is not False:
            failures.append(key)
    evidence = str(sidecar.get("evidence_sha256", "")).casefold()
    if not HASH_RE.fullmatch(evidence):
        failures.append("evidence_sha256")
    else:
        unsigned = {str(key): value for key, value in sidecar.items() if str(key) not in {"evidence_sha256", "evidence_hash", "evidence_hash_valid", "confirmed_positive", "raw_payload_stored", "raw_response_stored", "oracle_answer_in_context", "training_eligible", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"}}
        if sha256_json(unsigned) != evidence:
            failures.append("evidence_hash_mismatch")
    if sidecar.get("evidence_hash") != evidence:
        failures.append("evidence_hash_alias")
    roles = sidecar.get("roles")
    if not isinstance(roles, Mapping) or set(str(key) for key in roles) != set(ROLES):
        failures.append("role_triplet")
    else:
        for role in ROLES:
            value = roles[role]
            if not isinstance(value, Mapping) or value.get("role") != role:
                failures.append(f"role:{role}")
                continue
            role_hash = str(value.get("evidence_sha256", "")).casefold()
            if not HASH_RE.fullmatch(role_hash):
                failures.append(f"role_evidence:{role}")
                continue
            role_unsigned = {
                "schema_version": SCHEMA_VERSION,
                "record_id": str(sidecar.get("record_id", "")),
                "role": role,
                "sent": value.get("sent") is True,
                "available": value.get("available") is True,
                "executed": value.get("executed") is True,
                "typed_effect_confirmed": value.get("typed_effect_confirmed") is True,
                "effect_class": str(value.get("effect_class", "unknown")),
                "projection": value.get("projection") if isinstance(value.get("projection"), Mapping) else {},
                "source_evidence_sha256": str(value.get("source_evidence_sha256", "")),
            }
            if sha256_json(role_unsigned) != role_hash:
                failures.append(f"role_evidence_mismatch:{role}")
    return {"valid": not failures, "failures": sorted(set(failures)), "confirmed_positive": bool(sidecar.get("confirmed_positive"))}


def validate_pg331_evaluator_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {"valid": False, "failures": ["record_not_mapping"]}
    failures: list[str] = []
    if str(record.get("schema_version", "")) != RECORD_SCHEMA_VERSION:
        failures.append("schema_version")
    sidecar_result = validate_pg331_evaluator_sidecar(record.get("evaluator_sidecar") if isinstance(record.get("evaluator_sidecar"), Mapping) else {})
    failures.extend(f"sidecar:{item}" for item in sidecar_result["failures"])
    context = record.get("model_context")
    if not isinstance(context, Mapping):
        failures.append("model_context")
    try:
        _context_firewall(context)
    except (TypeError, ValueError):
        failures.append("context_firewall")
    for key in ("raw_payload_stored", "raw_response_stored", "oracle_answer_in_context", "training_eligible"):
        if record.get(key) is not False:
            failures.append(key)
    expected_hash = str(record.get("record_sha256", "")).casefold()
    if not HASH_RE.fullmatch(expected_hash):
        failures.append("record_sha256")
    else:
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        if sha256_json(unsigned) != expected_hash:
            failures.append("record_hash_mismatch")
    return {"valid": not failures, "failures": sorted(set(failures)), "confirmed_positive": bool(sidecar_result.get("confirmed_positive"))}


# Short aliases make the helper convenient for evaluator adapters while the
# explicit PG-331 names remain the stable public contract.
build_evaluator_sidecar = build_pg331_evaluator_sidecar
build_evaluator_record = build_pg331_evaluator_record


__all__ = [
    "EFFECT_CLASSES",
    "RECORD_SCHEMA_VERSION",
    "ROLES",
    "SCHEMA_VERSION",
    "build_evaluator_record",
    "build_evaluator_sidecar",
    "build_pg331_evaluator_record",
    "build_pg331_evaluator_sidecar",
    "canonical",
    "sha256_json",
    "validate_pg331_evaluator_record",
    "validate_pg331_evaluator_sidecar",
]
