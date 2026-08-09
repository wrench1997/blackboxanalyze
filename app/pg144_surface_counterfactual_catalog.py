"""PG-144 surface-counterfactual data for next-token representation pretraining.

The source catalog already contains bounded Rule-IR/observation tokens, but it
does not contain enough *different surfaces with the same oracle state*.  This
module creates a provenance-preserving augmentation set from those tokens only.
The variants are explicitly marked as counterfactual representation data and
are never eligible for action, safety, or memory promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pg144-surface-counterfactual-next-token-v1"
DATASET_SCHEMA = "pg144-surface-counterfactual-model-dataset-v1"
MAX_SEQUENCE_LENGTH = 384
# A representation catalog must contain mostly distinct model sequences before
# it can be used for pretraining.  This is intentionally a conservative,
# explicit gate: synthetic rows that collapse onto the same sequence remain
# useful for diagnostics, but cannot silently inflate the training corpus.
MIN_UNIQUE_SEQUENCE_DENSITY = 0.5

# These are abstract bounded tokens, not source code or executable payloads.
# Each transform changes a surface description while leaving IR/observation
# and oracle-availability tokens untouched.
SURFACE_VARIANTS: dict[str, dict[str, str]] = {
    "html_shell": {
        "src.html.tag=form": "src.html.tag=section",
        "src.html.attribute=method": "src.html.attribute=role",
        "src.html.attribute=name": "src.html.attribute=id",
    },
    "javascript_api": {
        "src.javascript.api=fetch": "src.javascript.api=xhr",
        "src.javascript.keyword=const": "src.javascript.keyword=let",
        "src.javascript.keyword=if": "src.javascript.keyword=branch",
    },
    "transport_surface": {
        "src.transport.placement=query": "src.transport.placement=path",
        "src.transport.placement=json": "src.transport.placement=form",
        "src.transport.route_template=hash_present": "src.transport.route_template=route_named",
    },
    "encoding_surface": {
        "src.transport.placement=query": "src.transport.placement=encoded_query",
        "src.transport.placement=json": "src.transport.placement=encoded_body",
        "src.html.text_length_bucket=1-4": "src.html.text_length_bucket=5-16",
    },
    "html_length": {
        "src.html.text_length_bucket=1-4": "src.html.text_length_bucket=5-16",
        "src.html.text_length_bucket=5-16": "src.html.text_length_bucket=17+",
    },
    "html_script_count": {
        "src.html.script_count=1-4": "src.html.script_count=5-16",
        "src.html.text_length_bucket=1-4": "src.html.text_length_bucket=5-16",
    },
    "transport_fields": {
        "src.transport.form_field_count=0": "src.transport.form_field_count=1-4",
        "src.transport.form_field_count=1-4": "src.transport.form_field_count=5-16",
    },
    "javascript_length": {
        "src.javascript.length_bucket=17+": "src.javascript.length_bucket=5-16",
        "src.javascript.keyword=const": "src.javascript.keyword=let",
    },
    "source_count_up": {
        "src_count=1-4": "src_count=5-16",
        "src_count=0": "src_count=1-4",
    },
    "source_count_down": {
        "src_count=1-4": "src_count=0",
        "src_count=5-16": "src_count=1-4",
    },
    "html_element_alias": {
        "src.html.tag=form": "src.html.tag=input",
        "src.html.attribute=name": "src.html.attribute=id",
    },
    "javascript_control_alias": {
        "src.javascript.keyword=if": "src.javascript.keyword=switch",
        "src.javascript.keyword=const": "src.javascript.keyword=var",
    },
    "route_alias": {
        "src.transport.route_template=hash_present": "src.transport.route_template=path_named",
        "src.transport.placement=query": "src.transport.placement=route",
    },
    "placement_alias": {
        "src.transport.placement=query": "src.transport.placement=header",
        "src.transport.placement=json": "src.transport.placement=multipart",
    },
    "attribute_alias": {
        "src.html.attribute=method": "src.html.attribute=action",
        "src.html.attribute=name": "src.html.attribute=data_key",
    },
    "mixed_surface": {
        "src.html.tag=form": "src.html.tag=fieldset",
        "src.javascript.api=fetch": "src.javascript.api=request",
        "src.transport.route_template=hash_present": "src.transport.route_template=path_named",
        "src.html.text_length_bucket=1-4": "src.html.text_length_bucket=17+",
    },
}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def oracle_availability(tokens: Sequence[str]) -> str:
    values = {
        match.group(1)
        for token in tokens
        for match in [re.fullmatch(r"obs\.oracle\.availability=(typed|unknown)", str(token))]
        if match
    }
    if len(values) != 1:
        return "unknown"
    return next(iter(values))


def _surface_tokens(tokens: Iterable[str]) -> list[str]:
    return [
        str(token)
        for token in tokens
        if str(token).startswith(("src.", "src_count="))
    ]


def _insert_counterfactual_marker(tokens: Sequence[str], variant: str, availability: str) -> list[str]:
    # Counterfactual provenance stays in the evaluator-side row metadata.  Do
    # not put a variant/availability marker in the model sequence: that would
    # provide an artificial shortcut and let next-token training memorize the
    # augmentation label instead of learning the surface transformation.
    del variant, availability
    return [str(token) for token in tokens]


def augment_tokens(tokens: Sequence[str], variant: str) -> tuple[list[str], dict[str, Any]]:
    if variant not in SURFACE_VARIANTS:
        raise KeyError(f"unknown PG-144 surface variant: {variant}")
    base = [str(token) for token in tokens]
    availability = oracle_availability(base)
    replacements = SURFACE_VARIANTS[variant]
    transformed = [replacements.get(token, token) for token in base]
    transformed = _insert_counterfactual_marker(transformed, variant, availability)
    if len(transformed) > MAX_SEQUENCE_LENGTH:
        raise ValueError("PG-144 counterfactual sequence exceeds bounded length")
    transformed_availability = oracle_availability(transformed)
    if transformed_availability != availability:
        raise ValueError("PG-144 changed oracle availability while perturbing surface")
    changed = _surface_tokens(base) != _surface_tokens(transformed)
    if not changed:
        raise ValueError("PG-144 variant did not change a bounded surface token")
    metadata = {
        "variant": variant,
        "oracle_availability": availability,
        "surface_changed": True,
        "surface_token_count_before": len(_surface_tokens(base)),
        "surface_token_count_after": len(_surface_tokens(transformed)),
        "surface_delta_sha256": sha256_json({"before": _surface_tokens(base), "after": _surface_tokens(transformed)}),
    }
    return transformed, metadata


def build_augmented_rows(base_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    for base in base_rows:
        parent_id = str(base.get("model_row_id") or base.get("row_id") or "unknown")
        base_tokens = [str(token) for token in base.get("tokens", [])]
        if not base_tokens:
            continue
        for variant in SURFACE_VARIANTS:
            augmented, metadata = augment_tokens(base_tokens, variant)
            row_id = f"{parent_id}::cf::{variant}"
            rows.append(
                {
                    "schema_version": DATASET_SCHEMA,
                    "model_row_id": row_id,
                    "parent_model_row_id": parent_id,
                    "fold": base.get("fold", "unknown"),
                    "split": base.get("split", "unknown"),
                    "tokens": augmented,
                    "token_count": len(augmented),
                    "augmentation": metadata,
                    "labels_in_model_row": False,
                    "action_supervision_allowed": False,
                    "safety_supervision_allowed": False,
                    "memory_promotion_allowed": False,
                }
            )
            pair_records.append(
                {
                    "parent_model_row_id": parent_id,
                    "augmented_model_row_id": row_id,
                    "split": base.get("split", "unknown"),
                    "variant": variant,
                    "oracle_availability": metadata["oracle_availability"],
                    "surface_delta_sha256": metadata["surface_delta_sha256"],
                }
            )
    counts = Counter(record["variant"] for record in pair_records)
    availability = Counter(record["oracle_availability"] for record in pair_records)
    parent_split = {(record["parent_model_row_id"], record["split"]) for record in pair_records}
    sequence_hashes = [sha256_json(row["tokens"]) for row in rows]
    unique_sequence_count = len(set(sequence_hashes))
    duplicate_sequence_count = len(sequence_hashes) - unique_sequence_count
    unique_sequence_density = unique_sequence_count / len(sequence_hashes) if sequence_hashes else 0.0
    surface_diversity_gate = unique_sequence_density >= MIN_UNIQUE_SEQUENCE_DENSITY
    surface_vocab = sorted({token for row in rows for token in _surface_tokens(row["tokens"])})
    return rows, {
        "base_row_count": len(base_rows),
        "augmented_row_count": len(rows),
        "variant_count": len(SURFACE_VARIANTS),
        "variant_row_counts": dict(sorted(counts.items())),
        "oracle_availability_counts": dict(sorted(availability.items())),
        "changed_surface_pair_count": len(pair_records),
        "unique_surface_delta_count": len({record["surface_delta_sha256"] for record in pair_records}),
        "unique_sequence_count": unique_sequence_count,
        "unique_sequence_density": unique_sequence_density,
        "duplicate_sequence_count": duplicate_sequence_count,
        "surface_diversity_gate": surface_diversity_gate,
        "surface_diversity_gate_min_density": MIN_UNIQUE_SEQUENCE_DENSITY,
        "surface_vocab_count": len(surface_vocab),
        "parent_split_binding_count": len(parent_split),
        "pair_records": pair_records,
        "all_action_supervision_forbidden": all(not row["action_supervision_allowed"] for row in rows),
        "all_safety_supervision_forbidden": all(not row["safety_supervision_allowed"] for row in rows),
        "all_memory_promotion_forbidden": all(not row["memory_promotion_allowed"] for row in rows),
    }
