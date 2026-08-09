"""Offline, deterministic Rule IR evaluation data for PG-31.

This module makes *abstract* dataset rows only.  It does not construct or
send requests, start containers, or retain probe/payload strings.  The rows
are useful as a held-out evaluation manifest for ``model_capability_gate``;
they are deliberately marked evaluation-only and contain no model outputs.

The generator is intentionally small and boring.  A reproducible synthetic
set is preferable to pretending that a hand-written list of attack strings is
evidence of generalisation.  A later evaluator may attach model metrics to
the per-role ``dataset_tests`` skeleton, but this module never grants
training or memory-promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


OOD_DATASET_SCHEMA = "pg-pk-31-ood-rule-ir-evaluation-manifest-v1"
DATASET_TEST_SCHEMA = "pg-pk-31-dataset-test-skeleton-v1"
DEFAULT_SEEDS = (701, 809, 907)
ROLES = ("train", "dev", "family_holdout", "ood_source", "negative_control")

# These are semantic families, not payload classes.  They describe which
# abstract rule slots should be exercised by an evaluator.
_ROLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "train": ("xss", "sqli", "logic_access"),
    "dev": ("xss", "sqli", "logic_access"),
    "family_holdout": ("command_injection", "header_policy"),
    "ood_source": ("state_machine", "template_binding", "transport_split"),
    "negative_control": ("ordinary_response", "not_found", "escaped_reflection"),
}
_SURFACES = (
    "html_attribute",
    "html_text",
    "json_value",
    "response_header",
    "sql_ast_shape",
    "authorization_boundary",
    "business_invariant",
)
_TRANSPORTS = ("query", "form", "header", "path_segment", "none")
_ENCODINGS = ("identity", "url_percent", "html_entity", "json_string")
_STATUS_CLASSES = ("2xx", "3xx", "4xx", "5xx")
_CONTENT_TYPES = ("html", "json", "text", "xml", "other")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_hash(role: str) -> str:
    # The source is a deterministic in-repository generator, not an external
    # target.  Distinct role hashes make train/evaluation separation visible.
    return hashlib.sha256(f"pg31-synthetic-source:{role}:v1".encode("ascii")).hexdigest()


def _target_instance(role: str, seed: int, index: int) -> str:
    # Three stable target identities are enough for the capability gate while
    # remaining clearly synthetic and non-networked.
    return f"pg31-{role.replace('_', '-')}-fixture-{(seed + index) % 3 + 1}"


def _label_for(role: str, family: str, rng: random.Random) -> tuple[int, str, bool]:
    """Return (expected label, oracle effect, typed positive).

    A negative control never becomes positive merely because a response
    shape looks interesting.  Holdout/OOD positives are typed effects in the
    abstract fixture contract, not execution results.
    """

    if role == "negative_control":
        return 0, "none", False
    positive = rng.random() >= 0.35
    if not positive:
        return 0, "none", False
    effects = {
        "xss": "dom_structure",
        "sqli": "sql_ast_shape",
        "logic_access": "authorization_boundary",
        "command_injection": "interpreter_boundary",
        "header_policy": "frame_protection",
        "state_machine": "history_binding",
        "template_binding": "business_invariant",
        "transport_split": "transport_boundary",
    }
    return 1, effects.get(family, "typed_surface_effect"), True


def _make_rule_ir(*, family: str, transport: str, surface: str, encoding: str, depth: int) -> dict[str, Any]:
    # Rule IR contains slots and operators only; there is no concrete input,
    # URL, query value, form value, or exploit syntax in this structure.
    operators = ["field_present", "surface_match", "oracle_match"]
    if depth >= 2:
        operators.append("encoding_invariant")
    if depth >= 3:
        operators.append("history_consistent")
    return {
        "grammar_version": "rule-ir-v1",
        "family_slot": family,
        "surface_slot": surface,
        "transport_slot": transport,
        "encoding_slot": encoding,
        "depth": int(depth),
        "operators": operators,
        "executable": False,
    }


def _make_response_projection(*, status: str, content_type: str, surface: str, positive: bool, depth: int) -> dict[str, Any]:
    shape = "typed_effect" if positive else "ordinary_response"
    if surface == "html_attribute":
        shape = "attribute_surface" if positive else "escaped_attribute"
    elif surface == "sql_ast_shape":
        shape = "ast_delta" if positive else "ast_stable"
    elif surface == "authorization_boundary":
        shape = "authorization_delta" if positive else "authorization_stable"
    return {
        "status_class": status,
        "content_type_class": content_type,
        "body_length_bucket": ("small" if not positive else "medium"),
        "surface_shape": shape,
        "delta_class": "typed_change" if positive else "no_typed_change",
        "header_class": "policy" if surface == "response_header" else "ordinary",
        "raw_body_stored": False,
        "raw_probe_stored": False,
        "encoding_depth": int(depth),
    }


def _make_row(*, role: str, seed: int, index: int, rng: random.Random) -> dict[str, Any]:
    family = rng.choice(_ROLE_FAMILIES[role])
    surface = rng.choice(_SURFACES)
    transport = rng.choice(_TRANSPORTS)
    encoding = rng.choice(_ENCODINGS)
    depth = rng.randint(0, 3)
    status = rng.choice(_STATUS_CLASSES)
    content_type = rng.choice(_CONTENT_TYPES)
    label, effect, positive = _label_for(role, family, rng)
    dataset_id = f"pg31-{role.replace('_', '-')}-v1"
    row: dict[str, Any] = {
        "sample_id": f"{dataset_id}-s{seed}-n{index:04d}",
        "dataset_id": dataset_id,
        "role": role,
        "source_hash": _source_hash(role),
        "target_instance_id": _target_instance(role, seed, index),
        "sampling_seed": int(seed),
        "family": family,
        "expected_label": int(label),
        "expected_decision": "typed_positive" if positive else "abstain_or_negative",
        "rule_ir": _make_rule_ir(
            family=family,
            transport=transport,
            surface=surface,
            encoding=encoding,
            depth=depth,
        ),
        "response_projection": _make_response_projection(
            status=status,
            content_type=content_type,
            surface=surface,
            positive=positive,
            depth=depth,
        ),
        "typed_oracle": {
            "oracle_id": "pg31-synthetic-typed-oracle-v1",
            "authority": "in_repo_fixture_contract",
            "effect": effect,
            "positive": bool(positive),
            "evidence_kind": "projection_only",
            "evaluator_state_visible": False,
        },
        "safety": {
            "evaluation_only": True,
            "training_eligible": False,
            "network_access": False,
            "container_started": False,
            "script_execution": False,
            "database_write": False,
            "raw_data_retained": False,
        },
    }
    row["evidence_hash"] = sha256_json(row)
    return row


def _metric_skeleton(*, evidence_hash: str) -> dict[str, Any]:
    # Metrics are intentionally zero placeholders.  They are not model
    # measurements and must be replaced by an evaluator before gate use.
    metrics = {
        "typed_recall": 0.0,
        "precision": 0.0,
        "false_positive_rate": 0.0,
        "abstain_precision": 0.0,
        "ece": 0.0,
        "median_queries": 0.0,
    }
    return {
        "schema_version": DATASET_TEST_SCHEMA,
        "evidence_hash": evidence_hash,
        "metrics_status": "pending_model_run",
        "metrics": dict(metrics),
        "baseline_metrics": dict(metrics),
        "candidate_metrics": dict(metrics),
    }


def generate_manifest(*, seeds: Sequence[int] = DEFAULT_SEEDS, samples_per_role: int = 12) -> dict[str, Any]:
    """Generate a reproducible, evaluation-only manifest.

    ``samples_per_role`` is deliberately bounded.  This function performs no
    filesystem or network I/O; callers choose whether and where to serialize
    the returned object.
    """

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if len(normalized_seeds) < 3 or any(seed < 0 for seed in normalized_seeds):
        raise ValueError("at least three non-negative sampling seeds are required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("sampling seeds must be independent")
    samples_per_role = int(samples_per_role)
    if not 1 <= samples_per_role <= 256:
        raise ValueError("samples_per_role must be between 1 and 256")

    samples: list[dict[str, Any]] = []
    for role in ROLES:
        for seed in normalized_seeds:
            rng = random.Random(f"pg31:{role}:{seed}")
            for index in range(samples_per_role):
                samples.append(_make_row(role=role, seed=seed, index=index, rng=rng))

    dataset_tests: list[dict[str, Any]] = []
    for role in ROLES:
        for seed in normalized_seeds:
            role_rows = [
                row for row in samples
                if row["role"] == role and row["sampling_seed"] == seed
            ]
            target_ids = sorted({str(row["target_instance_id"]) for row in role_rows})
            families = sorted(set(_ROLE_FAMILIES[role]))
            dataset_id = f"pg31-{role.replace('_', '-')}-s{seed}-v1"
            summary = {
                "sample_id": f"pg31-test-{role.replace('_', '-')}-s{seed}",
                "dataset_id": dataset_id,
                "source_id": f"pg31-source-{role.replace('_', '-')}-v1",
                "source_hash": _source_hash(role),
                "target_instance_id": target_ids[0],
                "target_instance_ids": target_ids,
                "family_set": families,
                "sampling_seed": int(seed),
                "role": role,
                "sample_count": len(role_rows),
                "unique_sample_count": len({row["sample_id"] for row in role_rows}),
                "denominator": len(role_rows),
                "positive_count": sum(int(row["typed_oracle"]["positive"]) for row in role_rows),
                "negative_count": sum(int(not row["typed_oracle"]["positive"]) for row in role_rows),
                "abstain_count": 0,
                "dataset_manifest_sha256": sha256_json({"dataset_id": dataset_id, "seed": seed}),
                "split_manifest_sha256": sha256_json({"role": role, "seed": seed, "families": families}),
                "probe_sha256": sha256_json({"samples": [row["sample_id"] for row in role_rows]}),
                "oracle_contract_sha256": sha256_json("pg31-synthetic-typed-oracle-v1"),
                "checkpoint_sha256": sha256_json("pending-model-run"),
            }
            summary_hash = sha256_json(summary)
            test_row = {
                **summary,
                "evidence_hash": summary_hash,
                **_metric_skeleton(evidence_hash=summary_hash),
            }
            dataset_tests.append(test_row)

    manifest = {
        "schema_version": OOD_DATASET_SCHEMA,
        "manifest_id": "pg31-ood-rule-ir-evaluation-v1",
        "purpose": "family-holdout-and-negative-control-evaluation",
        "evaluation_only": True,
        "training_eligible": False,
        "training_artifact_generated": False,
        "memory_promotion_allowed": False,
        "model_evaluation_completed": False,
        "metrics_status": "pending_model_run",
        "source": {
            "source_type": "in_repo_synthetic",
            "license": "in-repo-synthetic",
            "authorization": "workspace_local_only",
            "network_access": False,
            "container_started": False,
            "raw_payloads_present": False,
            "raw_responses_present": False,
            "source_hashes_by_role": {role: _source_hash(role) for role in ROLES},
        },
        "roles": list(ROLES),
        "sampling_seeds": list(normalized_seeds),
        "family_policy": {
            "train": list(_ROLE_FAMILIES["train"]),
            "dev": list(_ROLE_FAMILIES["dev"]),
            "family_holdout": list(_ROLE_FAMILIES["family_holdout"]),
            "ood_source": list(_ROLE_FAMILIES["ood_source"]),
            "negative_control": list(_ROLE_FAMILIES["negative_control"]),
        },
        "dataset_tests": dataset_tests,
        "samples": samples,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def write_manifest(path: str | Any, *, seeds: Sequence[int] = DEFAULT_SEEDS, samples_per_role: int = 12) -> dict[str, Any]:
    """Write a manifest at ``path`` and return the in-memory object."""

    manifest = generate_manifest(seeds=seeds, samples_per_role=samples_per_role)
    target = path if hasattr(path, "write_text") else Path(str(path))
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["DEFAULT_SEEDS", "OOD_DATASET_SCHEMA", "ROLES", "generate_manifest", "write_manifest"]
