"""Audit a multi-step PG-331A abstract trajectory.

The trajectory layer never sends a request and never copies a source row into
the model context.  It only checks that already-collected rows can be ordered
as an episode without losing the evidence needed for ASK, repair, GET/POST,
and role-bound evaluator replay.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .pg331_source_row import sha256_json, validate_pg331_source_row


SCHEMA_VERSION = "pg331-whole-web-trajectory-v1"
MAX_STEPS = 128
STEP_KEYS = frozenset({"step_index", "action_role", "row"})
ACTION_ROLES = frozenset(
    {
        "ask",
        "baseline_observe",
        "candidate_request",
        "reference_request",
        "negative_request",
        "repair",
        "replay",
        "abstain",
    }
)


def _token_values(tokens: Sequence[Any], key: str) -> list[str]:
    prefix = f"{key}="
    return [str(token)[len(prefix) :] for token in tokens if str(token).startswith(prefix)]


def _one_token(tokens: Sequence[Any], key: str, default: str = "unknown") -> str:
    values = _token_values(tokens, key)
    return values[-1] if values else default


def audit_pg331_trajectory(
    steps: Sequence[Mapping[str, Any]],
    *,
    require_get_post: bool = False,
    require_triplet: bool = False,
) -> dict[str, Any]:
    """Return a safe, abstract audit projection for one ordered episode."""

    failures: list[str] = []
    row_failures: dict[str, list[str]] = {}
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return {"schema_version": SCHEMA_VERSION, "valid": False, "failures": ["steps_not_sequence"], "promotion": _promotion()}
    if not steps:
        failures.append("empty_trajectory")
    if len(steps) > MAX_STEPS:
        failures.append("step_budget_exceeded")

    record_ids: list[str] = []
    source_ids: list[str] = []
    methods: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    evidence_hashes: list[str] = []
    typed_evidence_roles: dict[str, str] = {}
    action_changed_count = 0
    ask_count = 0
    previous_target_action: str | None = None

    for expected_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            failures.append(f"step_not_mapping:{expected_index}")
            continue
        unknown = sorted(str(key) for key in step if str(key) not in STEP_KEYS)
        if unknown:
            failures.append(f"step_unsupported_fields:{expected_index}")
        index = step.get("step_index")
        if not isinstance(index, int) or isinstance(index, bool) or index != expected_index:
            failures.append(f"step_index:{expected_index}")
        role = str(step.get("action_role", "")).casefold()
        if role not in ACTION_ROLES:
            failures.append(f"action_role:{expected_index}")
        else:
            roles[role] += 1
        row = step.get("row")
        if not isinstance(row, Mapping):
            failures.append(f"row_not_mapping:{expected_index}")
            continue
        validation = validate_pg331_source_row(row)
        if not validation.get("valid"):
            row_failures[str(expected_index)] = [str(item) for item in validation.get("failures") or []]
            failures.append(f"row_invalid:{expected_index}")
        record_id = str(row.get("record_id", ""))
        if not record_id:
            failures.append(f"record_id:{expected_index}")
        elif record_id in record_ids:
            failures.append(f"duplicate_record_id:{expected_index}")
        else:
            record_ids.append(record_id)
        source_meta = row.get("source_meta")
        if isinstance(source_meta, Mapping):
            source_id = str(source_meta.get("source_id", ""))
            if source_id:
                source_ids.append(source_id)
        context = [str(token) for token in row.get("context_tokens") or []]
        method_values = set(_token_values(context, "request_method"))
        methods.update(value.upper() for value in method_values)
        if not method_values:
            failures.append(f"request_method_missing:{expected_index}")

        target = row.get("target_projection")
        if isinstance(target, Mapping):
            target_action = str(target.get("next_action", "unknown"))
            if str(target.get("question", "")) in {"ask_typed", "ask_typed_oracle"} or target_action in {"ask", "ask_typed"}:
                ask_count += 1
            if previous_target_action is not None:
                previous_observed = _one_token(context, "failure_previous_action", "none")
                if previous_observed not in {"none", "unknown"} and previous_observed != previous_target_action:
                    failures.append(f"previous_action_mismatch:{expected_index}")
            previous_target_action = target_action

        failure_class = _one_token(context, "failure_failure_class", "none")
        previous_action = _one_token(context, "failure_previous_action", "none")
        next_action = _one_token(context, "failure_next_action", "none")
        if failure_class not in {"none", "unknown"}:
            if previous_action in {"none", "unknown"} or next_action in {"none", "unknown"} or previous_action == next_action:
                failures.append(f"failure_action_not_changed:{expected_index}")
            else:
                action_changed_count += 1

        evaluator = row.get("evaluator_sidecar")
        if isinstance(evaluator, Mapping) and evaluator.get("typed_available") is True:
            evidence = str(evaluator.get("evidence_hash", ""))
            if evidence:
                evidence_hashes.append(evidence)
                if evidence in typed_evidence_roles and typed_evidence_roles[evidence] != role:
                    failures.append(f"evidence_reused_across_roles:{expected_index}")
                elif evidence in typed_evidence_roles:
                    failures.append(f"evidence_reused_same_role:{expected_index}")
                else:
                    typed_evidence_roles[evidence] = role

    if len(set(evidence_hashes)) != len(evidence_hashes):
        failures.append("typed_evidence_not_unique")
    if source_ids and len(set(source_ids)) != 1:
        failures.append("source_id_changed_within_trajectory")
    if require_get_post and not {"GET", "POST"}.issubset(set(methods)):
        failures.append("get_post_pair_missing")
    if require_triplet and not {"candidate_request", "reference_request", "negative_request"}.issubset(set(roles)):
        failures.append("candidate_reference_negative_triplet_missing")

    row_count = len([step for step in steps if isinstance(step, Mapping) and isinstance(step.get("row"), Mapping)])
    all_rows_training = row_count == len(steps) and row_count > 0 and all(bool(step["row"].get("training_eligible")) for step in steps if isinstance(step, Mapping) and isinstance(step.get("row"), Mapping))
    valid = not failures
    projection: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "step_count": len(steps),
        "row_count": row_count,
        "record_id_count": len(record_ids),
        "source_id_digest": hashlib.sha256(source_ids[0].encode("utf-8")).hexdigest() if len(set(source_ids)) == 1 and source_ids else None,
        "method_counts": dict(methods),
        "action_role_counts": dict(roles),
        "ask_count": ask_count,
        "failure_action_changed_count": action_changed_count,
        "typed_evidence_count": len(evidence_hashes),
        "typed_evidence_unique": len(set(evidence_hashes)) == len(evidence_hashes),
        "get_post_pair": {"GET", "POST"}.issubset(set(methods)),
        "row_failures": row_failures,
        "failures": sorted(set(failures)),
        "trajectory_training_eligible": bool(valid and all_rows_training),
        "promotion": _promotion(),
    }
    projection["trajectory_sha256"] = sha256_json({key: value for key, value in projection.items() if key != "trajectory_sha256"})
    return projection


def _promotion() -> dict[str, bool]:
    return {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


__all__ = ["ACTION_ROLES", "MAX_STEPS", "SCHEMA_VERSION", "audit_pg331_trajectory"]
