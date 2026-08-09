"""Validate step-by-step, projection-only learning traces.

The collector is intentionally a validator rather than a network client.  A
local adapter supplies bounded request/response/oracle projections; this
module binds the model's hypothesis and belief transition to that observation
without retaining a raw probe or response body.  It supports episode-local
learning and shadow replay, never direct online weight or memory promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from .failure_guided_scheduler import validate_failure_signature


TRACE_SCHEMA = "sift-trace-aligned-step-v1"
EPISODE_SCHEMA = "sift-trace-aligned-episode-report-v1"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_METHODS = frozenset({"GET", "POST"})
_DECISIONS = frozenset({"candidate", "confirmed_positive", "confirmed_negative", "abstain"})
_WEAK_MODALITIES = frozenset({"reflection", "syntax_error", "bounded_timing", "status_change", "transport_error"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _id(value: Any, label: str) -> str:
    text = str(value)
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{label} must be a bounded identifier")
    return text


def _hash(value: Any, label: str) -> str:
    text = str(value).casefold()
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _bounded_belief(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty probability map")
    result: dict[str, float] = {}
    for key, raw in value.items():
        name = _id(key, f"{label}.key")
        number = float(raw)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{label}.{name} must be in [0, 1]")
        result[name] = number
    if sum(result.values()) > 1.000001:
        raise ValueError(f"{label} probabilities may not sum above 1")
    return result


def _projection(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a bounded projection")
    encoded = canonical_json(value)
    if len(encoded) > 4096:
        raise ValueError(f"{label} is too large")
    forbidden = {"body", "raw_body", "body_preview", "request_body", "raw_probe", "password", "token", "cookie", "authorization"}

    def contains_forbidden(node: Any) -> bool:
        if isinstance(node, dict):
            return any(str(key).casefold() in forbidden or contains_forbidden(child) for key, child in node.items())
        if isinstance(node, list):
            return any(contains_forbidden(child) for child in node)
        return False

    if contains_forbidden(value):
        raise ValueError(f"{label} contains raw or secret fields")
    return json.loads(encoded)


def _action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("action_manifest must be an object")
    method = str(value.get("method", "")).upper()
    if method not in _METHODS:
        raise ValueError("trace actions require GET or POST")
    placement = _id(value.get("placement"), "action_manifest.placement")
    chain = [str(item) for item in value.get("encoding_chain", [])]
    if not 1 <= len(chain) <= 3 or any(not _ID_RE.fullmatch(item) for item in chain):
        raise ValueError("action_manifest.encoding_chain is invalid")
    safety = dict(value.get("safety") or {})
    required_safety = ("no_external_network", "does_not_execute", "no_database_write", "no_credential_access")
    if any(not bool(safety.get(key)) for key in required_safety):
        raise ValueError("action_manifest safety attestation is incomplete")
    form_fields = [_id(item, "action_manifest.form_field_names") for item in value.get("form_field_names", [])]
    if method == "POST" and not form_fields:
        raise ValueError("POST action_manifest requires form_field_names")
    return {
        "method": method,
        "route_template_id": _id(value.get("route_template_id"), "action_manifest.route_template_id"),
        "placement": placement,
        "encoding_chain": chain,
        "probe_ref": _id(value.get("probe_ref"), "action_manifest.probe_ref"),
        "probe_sha256": _hash(value.get("probe_sha256"), "action_manifest.probe_sha256"),
        "safety": {key: True for key in required_safety},
        **({"form_field_names": form_fields} if method == "POST" else {}),
    }


def _oracle(value: Any) -> dict[str, Any]:
    oracle = _projection(value, "oracle_projection")
    modality = str(oracle.get("modality", ""))
    positive = bool(oracle.get("positive", False))
    authority = bool(oracle.get("positive_authority", False))
    if positive and (not authority or modality in _WEAK_MODALITIES):
        raise ValueError("weak oracle signal cannot be confirmed_positive")
    return oracle


def validate_trace_step(step: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError("trace step must be an object")
    action = _action(step.get("action_manifest"))
    oracle = _oracle(step.get("oracle_projection"))
    decision = str(step.get("decision", ""))
    if decision not in _DECISIONS:
        raise ValueError("trace decision is not allow-listed")
    if decision == "confirmed_positive" and not (bool(oracle.get("positive")) and bool(oracle.get("positive_authority"))):
        raise ValueError("confirmed_positive requires a typed positive oracle")
    baseline = _projection(step.get("baseline_projection"), "baseline_projection")
    response = _projection(step.get("response_projection"), "response_projection")
    # Optional causal-triplet fields.  Older v1 traces remain valid, while a
    # triplet trace can retain a neutral projection, a typed-negative probe,
    # and its oracle without exposing raw requests or bodies.
    neutral = _projection(step["neutral_projection"], "neutral_projection") if "neutral_projection" in step else None
    negative_probe = _projection(step["negative_probe_projection"], "negative_probe_projection") if "negative_probe_projection" in step else None
    neutral_oracle = _oracle(step["neutral_oracle_projection"]) if "neutral_oracle_projection" in step else None
    negative_oracle = _oracle(step["negative_oracle_projection"]) if "negative_oracle_projection" in step else None
    failure = validate_failure_signature(step["failure_signature"]) if "failure_signature" in step else None
    belief_before = _bounded_belief(step.get("belief_before"), "belief_before")
    belief_after = _bounded_belief(step.get("belief_after"), "belief_after")
    fresh_reset = _projection(step.get("fresh_reset"), "fresh_reset")
    if not (bool(fresh_reset.get("fresh_target")) and bool(fresh_reset.get("completed")) and bool(fresh_reset.get("evaluator_state_hidden"))):
        raise ValueError("fresh_reset must attest a fresh hidden evaluator target")
    normalized = {
        "schema_version": TRACE_SCHEMA,
        "episode_id": _id(step.get("episode_id"), "episode_id"),
        "step_id": _id(step.get("step_id"), "step_id"),
        "parent_step_id": None if step.get("parent_step_id") in (None, "") else _id(step.get("parent_step_id"), "parent_step_id"),
        "sampling_seed": int(step.get("sampling_seed", -1)),
        "target_instance_id": _id(step.get("target_instance_id"), "target_instance_id"),
        "hypothesis": _id(step.get("hypothesis"), "hypothesis"),
        "belief_before": belief_before,
        "action_manifest": action,
        "baseline_projection": baseline,
        "response_projection": response,
        "oracle_projection": oracle,
        "belief_after": belief_after,
        "decision": decision,
        "next_action": _id(step.get("next_action"), "next_action"),
        "fresh_reset": fresh_reset,
        "evidence_sha256": _hash(step.get("evidence_sha256"), "evidence_sha256"),
        "dataset_stage": str(step.get("dataset_stage", "trace_only")),
        "online_weight_update": False,
        "long_term_memory_write": False,
    }
    if neutral is not None:
        normalized["neutral_projection"] = neutral
    if negative_probe is not None:
        normalized["negative_probe_projection"] = negative_probe
    if neutral_oracle is not None:
        normalized["neutral_oracle_projection"] = neutral_oracle
    if negative_oracle is not None:
        normalized["negative_oracle_projection"] = negative_oracle
    if failure is not None:
        normalized["failure_signature"] = failure
    if normalized["sampling_seed"] < 0:
        raise ValueError("sampling_seed must be non-negative")
    if action["method"] not in _METHODS:
        raise ValueError("trace action method is not GET or POST")
    echo_body = {
        key: normalized[key]
        for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action")
    }
    for key in ("neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection"):
        if key in normalized:
            echo_body[key] = normalized[key]
    if "failure_signature" in normalized:
        echo_body["failure_signature"] = normalized["failure_signature"]
    declared_echo = step.get("echo")
    if not isinstance(declared_echo, dict) or str(declared_echo.get("sha256", "")).casefold() != sha256_json(echo_body):
        raise ValueError("trace echo hash does not cover the complete decision step")
    normalized["echo"] = {"sha256": sha256_json(echo_body)}
    normalized["trace_sha256"] = sha256_json(normalized)
    return normalized


def evaluate_episode(steps: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = [validate_trace_step(dict(step)) for step in steps]
    reasons: list[str] = []
    if not normalized:
        reasons.append("no_trace_steps")
    episode_ids = {step["episode_id"] for step in normalized}
    if len(episode_ids) != 1:
        reasons.append("multiple_episode_ids")
    step_ids = [step["step_id"] for step in normalized]
    if len(step_ids) != len(set(step_ids)):
        reasons.append("duplicate_step_id")
    if normalized and normalized[0]["parent_step_id"] is not None:
        reasons.append("first_step_has_parent")
    if any(
        current["parent_step_id"] != previous["step_id"]
        for previous, current in zip(normalized, normalized[1:])
    ):
        reasons.append("broken_step_chain")
    methods = {step["action_manifest"]["method"] for step in normalized}
    if not _METHODS.issubset(methods):
        reasons.append("missing_get_or_post_step")
    if normalized and not normalized[0]["fresh_reset"].get("completed", False):
        reasons.append("first_step_not_fresh_reset")
    if any(step["decision"] == "confirmed_positive" for step in normalized):
        positives = [step for step in normalized if step["decision"] == "confirmed_positive"]
        if not all(bool(step["oracle_projection"].get("positive_authority")) for step in positives):
            reasons.append("positive_without_typed_oracle")
    has_negative_pair = any(bool(step["oracle_projection"].get("negative_control_pair_id")) for step in normalized)
    if not has_negative_pair:
        reasons.append("missing_negative_control_pair")
    status = "accepted_evaluation" if not reasons else "trace_only"
    return {
        "schema_version": EPISODE_SCHEMA,
        "episode_id": next(iter(episode_ids), "unknown"),
        "step_count": len(normalized),
        "methods": sorted(methods),
        "status": status,
        "training_candidate": False,
        "memory_promotion_allowed": False,
        "reasons": sorted(set(reasons)),
        "trace_sha256": sha256_json([step["trace_sha256"] for step in normalized]),
    }


__all__ = ["EPISODE_SCHEMA", "TRACE_SCHEMA", "evaluate_episode", "sha256_json", "validate_trace_step"]
