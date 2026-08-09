"""PG-140 evaluator-side information-complete catalog.

PG-139 showed that a bounded token projection can be safe while still losing
the metadata needed to audit a learning row.  This module creates a companion
catalog without retaining raw probes or response bodies.  Missing observations
are represented explicitly as ``unknown``/``not_observed``; they are never
silently converted to false, zero, or an empty list.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pg140-information-complete-catalog-v1"
MODEL_DATASET_SCHEMA = "pg140-information-complete-model-dataset-v1"
TOKENIZER_SCHEMA = "pg136-causal-next-token-gru-v1"
PARSER_SCHEMA = "pg139-independent-parser-variant-v1"
MAX_SEQUENCE_LENGTH = 384
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

PROJECTION_SENTINELS = {
    "baseline_projection.body_length_bucket": "unknown",
    "baseline_projection.shape_class": "unknown",
    "baseline_projection.status_class": "unknown",
    "response_projection.body_length_bucket": "unknown",
    "response_projection.shape_class": "unknown",
    "response_projection.status_class": "unknown",
    "response_projection.transition_delta": "unknown",
    "response_projection.scope_changed": "not_observed",
    "response_projection.visibility_changed": "not_observed",
}

SAFE_OMISSION_POLICY = {
    "raw_probe": "never_stored",
    "raw_response_body": "never_stored",
    "target_identity_in_model_input": "catalog_only",
    "evaluator_action_in_model_input": "catalog_only",
    "positive_authority_in_model_input": "catalog_only",
    "family_label_in_model_input": "catalog_only",
}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _hash_or_unknown(value: Any) -> str:
    text = str(value or "")
    return text if HASH_RE.fullmatch(text) else "unknown"


def _bounded_bool(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    return "not_observed"


def _bounded_float(value: Any) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if number != number or number in (float("inf"), float("-inf")):
        return "unknown"
    return round(max(0.0, min(1.0, number)), 8)


def _normalize_mapping(source: Mapping[str, Any] | None, fields: Sequence[str], sentinels: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = source or {}
    result: dict[str, Any] = {}
    missing_fields: list[str] = []
    for field in fields:
        value = source.get(field)
        if _missing(value):
            result[field] = sentinels.get(field, "unknown")
            missing_fields.append(field)
        else:
            result[field] = value
    return result, missing_fields


def normalize_projection(step: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Whitelisted, bounded projection with explicit unknown sentinels."""

    missing_fields: list[str] = []
    baseline, missing = _normalize_mapping(
        step.get("baseline_projection"),
        ("body_length_bucket", "shape_class", "status_class"),
        {key.rsplit(".", 1)[-1]: value for key, value in PROJECTION_SENTINELS.items() if key.startswith("baseline_projection.")},
    )
    missing_fields.extend(f"baseline_projection.{field}" for field in missing)
    response, missing = _normalize_mapping(
        step.get("response_projection"),
        (
            "body_length_bucket",
            "shape_class",
            "status_class",
            "transition_delta",
            "scope_changed",
            "visibility_changed",
            "candidate_signal",
            "location_changed",
            "metadata_changed",
            "policy_header_changed",
            "response_projection_sha256",
        ),
        {
            key.rsplit(".", 1)[-1]: value
            for key, value in PROJECTION_SENTINELS.items()
            if key.startswith("response_projection.")
        },
    )
    missing_fields.extend(f"response_projection.{field}" for field in missing)
    # Hashes are evidence references, never arbitrary strings.
    response["response_projection_sha256"] = _hash_or_unknown(response.get("response_projection_sha256"))
    return {"baseline": baseline, "response": response}, sorted(set(missing_fields))


def repaired_observation_tokens(base_tokens: Sequence[str], projection: Mapping[str, Mapping[str, Any]], *, oracle_availability: Any = None) -> list[str]:
    """Add only bounded observation buckets, including explicit unknowns.

    The existing causal source/Rule-IR tokens remain intact.  This compact
    suffix makes the distinction between an observed false and an unavailable
    field learnable without exposing oracle authority, target identity, or
    request/response content.
    """

    tokens = [str(token) for token in base_tokens]
    observation = ["[OBS]"]
    for layer_name, fields in (("baseline", ("body_length_bucket", "shape_class", "status_class")), ("response", ("body_length_bucket", "shape_class", "status_class", "transition_delta", "scope_changed", "visibility_changed"))):
        observation.append(f"[OBS_{layer_name.upper()}]")
        for field in fields:
            value = str((projection.get(layer_name) or {}).get(field, "unknown"))
            # Projection values are generated from bounded source buckets;
            # still sanitize defensively before putting them in a token.
            value = re.sub(r"[^A-Za-z0-9_+.-]", "_", value)[:32] or "unknown"
            observation.append(f"obs.{layer_name}.{field}={value}")
    availability = "typed" if oracle_availability is True else "unknown" if oracle_availability is False else "unknown"
    observation.extend(["[OBS_ORACLE]", f"obs.oracle.availability={availability}"])
    try:
        eos = tokens.index("[EOS]")
    except ValueError:
        eos = len(tokens)
    tokens[eos:eos] = observation
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        raise ValueError("PG-140 repaired observation sequence exceeds bounded length")
    return tokens


def _normalize_belief(value: Any) -> dict[str, float | str]:
    if not isinstance(value, Mapping):
        return {"effect": "unknown", "input_only": "unknown", "no_effect": "unknown", "unknown": "unknown"}
    keys = ("effect", "input_only", "no_effect", "unknown")
    return {key: _bounded_float(value.get(key)) for key in keys}


def _normalize_action_manifest(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, Mapping) else {}
    safety = value.get("safety") if isinstance(value.get("safety"), Mapping) else {}
    return {
        "method": str(value.get("method") or "unknown"),
        "route_template_id": str(value.get("route_template_id") or "unknown"),
        "placement": str(value.get("placement") or "unknown"),
        "encoding_chain": list(value.get("encoding_chain") or ["unknown"]),
        "probe_sha256": _hash_or_unknown(value.get("probe_sha256")),
        "safety": {
            "no_external_network": _bounded_bool(safety.get("no_external_network")),
            "does_not_execute": _bounded_bool(safety.get("does_not_execute")),
            "no_database_write": _bounded_bool(safety.get("no_database_write")),
            "no_credential_access": _bounded_bool(safety.get("no_credential_access")),
        },
    }


def _normalize_oracle(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, Mapping) else {}
    safety = value.get("safety") if isinstance(value.get("safety"), Mapping) else {}
    return {
        "candidate_signal": _bounded_bool(value.get("candidate_signal")),
        "modality": str(value.get("modality") or "unknown"),
        "observed_atoms": sorted(str(item) for item in (value.get("observed_atoms") or [])),
        "typed_available": _bounded_bool(value.get("typed_available")),
        "oracle_contract_sha256": _hash_or_unknown(value.get("oracle_contract_sha256")),
        "positive": _bounded_bool(value.get("positive")),
        "positive_authority": _bounded_bool(value.get("positive_authority")),
        "source_evidence_sha256": _hash_or_unknown(value.get("source_evidence_sha256")),
        "safety": {
            "credentials_accessed": _bounded_bool(safety.get("credentials_accessed")),
            "database_touched": _bounded_bool(safety.get("database_touched")),
            "external_network": _bounded_bool(safety.get("external_network")),
            "navigation": _bounded_bool(safety.get("navigation")),
            "real_sleep_performed": _bounded_bool(safety.get("real_sleep_performed")),
            "script_execution": _bounded_bool(safety.get("script_execution")),
        },
    }


def _normalize_reset(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, Mapping) else {}
    return {
        "completed": _bounded_bool(value.get("completed")),
        "fresh_target": _bounded_bool(value.get("fresh_target")),
        "reset_epoch": value.get("reset_epoch") if value.get("reset_epoch") is not None else "unknown",
        "reset_evidence_sha256": _hash_or_unknown(value.get("reset_evidence_sha256")),
        "evidence_hash": _hash_or_unknown(value.get("evidence_hash")),
        "state_change_allowed": _bounded_bool(value.get("state_change_allowed")),
    }


def _normalize_failure(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, Mapping) else {}
    weights = value.get("key_feature_weights") if isinstance(value.get("key_feature_weights"), Mapping) else {}
    return {
        "schema_version": str(value.get("schema_version") or "unknown"),
        "kind": str(value.get("kind") or "unknown"),
        "failed_gate": str(value.get("failed_gate") or "unknown"),
        "observed_method": str(value.get("observed_method") or "unknown"),
        "methods_seen": sorted(str(item) for item in (value.get("methods_seen") or [])),
        "candidate_signal": _bounded_bool(value.get("candidate_signal")),
        "typed_available": _bounded_bool(value.get("typed_available")),
        "positive_authority": _bounded_bool(value.get("positive_authority")),
        "probe_round": value.get("probe_round") if value.get("probe_round") is not None else "unknown",
        "remaining_probe_budget": value.get("remaining_probe_budget") if value.get("remaining_probe_budget") is not None else "unknown",
        "key_feature_weights": {str(key): _bounded_float(item) for key, item in sorted(weights.items())},
        "key_features_ranked": [str(item) for item in (value.get("key_features_ranked") or [])],
        "next_action": str(value.get("next_action") or "unknown"),
        "model_visible": _bounded_bool(value.get("model_visible")),
        "raw_probe_retained": _bounded_bool(value.get("raw_probe_retained")),
        "raw_response_retained": _bounded_bool(value.get("raw_response_retained")),
        "memory_promotion_allowed": _bounded_bool(value.get("memory_promotion_allowed")),
    }


def _context_for_row(row: Mapping[str, Any], context_by_step: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any] | None:
    candidates = context_by_step.get(str(row.get("step_id")), [])
    for candidate in candidates:
        step = candidate["step"]
        episode = candidate["episode"]
        if str(episode.get("episode_id")) == str(row.get("episode_id")):
            return candidate
        if str(step.get("target_instance_id")) == str(row.get("target_instance_id")):
            return candidate
    return candidates[0] if candidates else None


def make_context_index(targets: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_name, target_rows in targets.items():
        for target in target_rows:
            for episode in target.get("episodes") or []:
                for step in episode.get("steps") or []:
                    index[str(step.get("step_id"))].append(
                        {
                            "source_name": source_name,
                            "target": target,
                            "episode": episode,
                            "step": step,
                        }
                    )
    return index


def build_catalog_row(
    row: Mapping[str, Any],
    *,
    fold: str,
    role: str,
    tokens: Sequence[str],
    context: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    target = context.get("target", {}) if context else {}
    episode = context.get("episode", {}) if context else {}
    step = context.get("step", {}) if context else {}
    normalized_projection, projection_missing = normalize_projection(step)
    action_manifest = _normalize_action_manifest(step.get("action_manifest"))
    oracle = _normalize_oracle(step.get("oracle_projection"))
    reset = _normalize_reset(step.get("fresh_reset"))
    failure = _normalize_failure(step.get("failure_signature", row.get("failure_signature")))
    # typed availability is a failure-signature property in the source trace;
    # copy only the bounded boolean into the evaluator-side oracle index.
    oracle["typed_available"] = failure.get("typed_available", "unknown")
    catalog_row_id = f"{fold}::{row.get('row_id')}"
    evidence = {
        "evidence_sha256": _hash_or_unknown(step.get("evidence_sha256")),
        "trace_sha256": _hash_or_unknown(step.get("trace_sha256")),
        "echo_sha256": _hash_or_unknown((step.get("echo") or {}).get("sha256") if isinstance(step.get("echo"), Mapping) else None),
        "response_projection_sha256": _hash_or_unknown((step.get("response_projection") or {}).get("response_projection_sha256") if isinstance(step.get("response_projection"), Mapping) else None),
        "oracle_contract_sha256": oracle["oracle_contract_sha256"],
        "source_evidence_sha256": oracle["source_evidence_sha256"],
        "reset_evidence_sha256": reset["reset_evidence_sha256"],
    }
    original_missing = sorted(
        set(projection_missing)
        | ({"context.target_implementation"} if context is None or _missing(target.get("target_implementation")) else set())
        | ({"context.target_instance_id"} if context is None or _missing(target.get("target_instance_id")) else set())
        | ({"step.evidence_sha256"} if evidence["evidence_sha256"] == "unknown" else set())
        | ({"step.trace_sha256"} if evidence["trace_sha256"] == "unknown" else set())
        | ({"step.fresh_reset"} if reset["completed"] in ("not_observed", "unknown") else set())
    )
    repaired_tokens = repaired_observation_tokens(tokens, normalized_projection, oracle_availability=failure.get("typed_available"))
    model_row = {
        "schema_version": MODEL_DATASET_SCHEMA,
        "model_row_id": catalog_row_id,
        "row_id": str(row.get("row_id") or "unknown"),
        "fold": fold,
        "split": str(row.get("split") or role),
        "tokens": repaired_tokens,
        "token_count": len(repaired_tokens),
    }
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "catalog_row_id": catalog_row_id,
        "row_id": str(row.get("row_id") or "unknown"),
        "fold": fold,
        "role": role,
        "split": str(row.get("split") or role),
        "source": str(row.get("source") or "unknown"),
        "provenance": {
            "target_instance_id": str(target.get("target_instance_id") or "unknown"),
            "target_seed": target.get("target_seed") if target.get("target_seed") is not None else "unknown",
            "target_implementation": str(target.get("target_implementation") or "unknown"),
            "target_schema_version": str(target.get("target_schema_version") or "unknown"),
            "surface_kind": str(episode.get("surface_kind") or "unknown"),
            "episode_variant": str(episode.get("episode_variant") or "unknown"),
            "oracle_available": _bounded_bool(episode.get("oracle_available")),
            "episode_id": str(step.get("episode_id") or row.get("episode_id") or "unknown"),
            "step_id": str(step.get("step_id") or row.get("step_id") or "unknown"),
            "parent_step_id": str(step.get("parent_step_id")) if step.get("parent_step_id") is not None else "root",
            "sampling_seed": step.get("sampling_seed") if step.get("sampling_seed") is not None else "unknown",
        },
        "request_projection": action_manifest,
        "observation_projection": normalized_projection,
        "oracle_projection": oracle,
        "belief": {
            "before": _normalize_belief(step.get("belief_before")),
            "after": _normalize_belief(step.get("belief_after")),
        },
        "failure_signature": failure,
        "replay": {
            "decision": str(step.get("decision") or "unknown"),
            "next_action": str(step.get("next_action") or row.get("label") or "unknown"),
            "fresh_reset": reset,
            "matched_negative_control": bool(failure.get("failed_gate") == "matched_negative_control"),
            "get_post_method": action_manifest["method"],
        },
        "evidence": evidence,
        "evaluator_label": {
            "action_label": str(row.get("label") or "unknown"),
            "label_source": "typed_contract_evaluator_only",
            "training_eligible_source": bool(row.get("training_eligible", False)),
            "memory_promotion_allowed_source": bool(row.get("memory_promotion_allowed", False)),
        },
        "information_quality": {
            "original_missing_fields": original_missing,
            "explicit_unknown_fields": projection_missing,
            "raw_probe_stored": False,
            "raw_response_body_stored": False,
            "model_input_excludes_evaluator_label": True,
        },
    }
    replayable = (
        not original_missing
        and evidence["evidence_sha256"] != "unknown"
        and evidence["trace_sha256"] != "unknown"
        and reset["completed"] is True
        and reset["fresh_target"] is True
        and oracle["oracle_contract_sha256"] != "unknown"
    )
    catalog["information_quality"]["replayable_complete"] = replayable
    catalog["information_quality"]["capability_train_candidate"] = replayable
    catalog_hash_input = dict(catalog)
    catalog["catalog_row_sha256"] = sha256_json(catalog_hash_input)
    return model_row, catalog, replayable


def build_manifest(catalog_rows: Sequence[Mapping[str, Any]], *, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    provenance = [
        {
            "catalog_row_id": row["catalog_row_id"],
            "row_id": row["row_id"],
            "fold": row["fold"],
            "split": row["split"],
            "source": row["source"],
            "step_id": row["provenance"]["step_id"],
            "episode_id": row["provenance"]["episode_id"],
            "target_instance_id": row["provenance"]["target_instance_id"],
            "evidence_sha256": row["evidence"]["evidence_sha256"],
            "trace_sha256": row["evidence"]["trace_sha256"],
        }
        for row in catalog_rows
    ]
    implementation_manifest = sorted(
        {
            (
                row["provenance"]["target_implementation"],
                row["provenance"]["target_schema_version"],
                row["provenance"]["target_seed"],
            )
            for row in catalog_rows
        }
    )
    oracle_manifest = sorted(
        {
            (row["oracle_projection"]["oracle_contract_sha256"], row["oracle_projection"]["modality"])
            for row in catalog_rows
        }
    )
    split_manifest = sorted((row["catalog_row_id"], row["fold"], row["role"], row["split"]) for row in catalog_rows)
    tokenizer_config = {
        "tokenizer_schema_version": TOKENIZER_SCHEMA,
        "parser_schema": PARSER_SCHEMA,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "projection": "bounded_source_rule_ir_tokens_plus_explicit_observation_and_oracle_availability",
        "special_tokens": ["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "catalog_row_count": len(catalog_rows),
        "row_provenance_manifest_sha256": sha256_json(provenance),
        "tokenizer_schema_version": TOKENIZER_SCHEMA,
        "tokenizer_config_sha256": sha256_json(tokenizer_config),
        "source_implementation_manifest_sha256": sha256_json(implementation_manifest),
        "oracle_contract_manifest_sha256": sha256_json(oracle_manifest),
        "split_manifest_sha256": sha256_json(split_manifest),
        "omission_policy": dict(SAFE_OMISSION_POLICY),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "source_hashes": dict(source_hashes),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def quality_summary(catalog_rows: Sequence[Mapping[str, Any]], replayable: Mapping[str, bool]) -> dict[str, Any]:
    missing = Counter()
    explicit_unknown = Counter()
    methods = Counter()
    split_counts = Counter()
    hash_invalid = 0
    fresh_reset_count = 0
    matched_negative_count = 0
    typed_oracle_count = 0
    unknown_oracle_count = 0
    complete_evidence_count = 0
    unique_catalog_ids: set[str] = set()
    for row in catalog_rows:
        unique_catalog_ids.add(str(row["catalog_row_id"]))
        for field in row["information_quality"]["original_missing_fields"]:
            missing[field] += 1
        for field in row["information_quality"]["explicit_unknown_fields"]:
            explicit_unknown[field] += 1
        methods[row["request_projection"]["method"]] += 1
        split_counts[(row["fold"], row["role"])] += 1
        reset = row["replay"]["fresh_reset"]
        if reset.get("completed") is True and reset.get("fresh_target") is True:
            fresh_reset_count += 1
        if row["replay"]["matched_negative_control"]:
            matched_negative_count += 1
        if row["oracle_projection"]["typed_available"] is True:
            typed_oracle_count += 1
        else:
            unknown_oracle_count += 1
        evidence_values = row["evidence"].values()
        if all(value != "unknown" and HASH_RE.fullmatch(str(value)) for value in evidence_values):
            complete_evidence_count += 1
        for value in row["evidence"].values():
            if value != "unknown" and not HASH_RE.fullmatch(str(value)):
                hash_invalid += 1
    capability_candidates = sum(
        1
        for row in catalog_rows
        if row["role"] in {"train", "dev"} and replayable.get(row["catalog_row_id"], False)
    )
    return {
        "catalog_row_count": len(catalog_rows),
        "original_missing_field_counts": dict(sorted(missing.items())),
        "explicit_unknown_field_counts": dict(sorted(explicit_unknown.items())),
        "method_counts": dict(sorted(methods.items())),
        "split_counts": {f"{fold}:{role}": count for (fold, role), count in sorted(split_counts.items())},
        "fresh_reset_count": fresh_reset_count,
        "matched_negative_control_count": matched_negative_count,
        "typed_oracle_count": typed_oracle_count,
        "unknown_oracle_count": unknown_oracle_count,
        "complete_evidence_count": complete_evidence_count,
        "unique_catalog_row_count": len(unique_catalog_ids),
        "hash_invalid_count": hash_invalid,
        "raw_content_stored": False,
        "model_labels_stored": False,
        "capability_train_candidate_count": capability_candidates,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "reason": "PG-140 catalog repair is not a capability-training or memory-promotion result",
    }


__all__ = [
    "SCHEMA_VERSION",
    "MODEL_DATASET_SCHEMA",
    "build_catalog_row",
    "build_manifest",
    "make_context_index",
    "quality_summary",
    "repaired_observation_tokens",
    "sha256_json",
]
