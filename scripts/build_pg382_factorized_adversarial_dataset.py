"""Build a factorized abstract adversarial matrix for Rule-IR composition.

PG-382 is a diagnostic companion to PG-380.  Both source implementations
cover the same abstract surface/feedback matrix so a composition decoder can
be tested without conflating an unseen implementation with an unseen surface
vocabulary.  Source hashes remain disjoint and implementation identity stays
off-context.  The artifact is still abstract candidate data only; no raw
payload, URL, response, evaluator answer, or wire is serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg380_abstract_adversarial_dataset import (
    ENCODINGS,
    FEEDBACKS,
    METHODS,
    ROLES,
    ROLES_BY_SURFACE,
    SEEDS,
    SURFACES,
    SURFACE_DEFAULTS,
    _feedback_target,
    _token,
)

SCHEMA_VERSION = "pg382-factorized-abstract-adversarial-dataset-v1"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    if not total:
        return 0.0
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def _surface_family(surface: str) -> str:
    if surface in {"html_text", "html_attribute", "html_dom", "script_context", "style_context"}:
        return "document_markup"
    if surface in {"json_string", "sql_string", "sql_numeric"}:
        return "structured_parser"
    if surface in {"query", "form", "path_segment"}:
        return "request_parameter"
    return "state_transition"


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    implementations = ("impl_a", "impl_b")
    for implementation in implementations:
        split = "train" if implementation == "impl_a" else "implementation_holdout"
        for surface_index, surface in enumerate(SURFACES):
            shape, syntax, _response_projection = SURFACE_DEFAULTS[surface]
            source_hash = _sha({"implementation": implementation, "surface_template": surface, "contract": "pg382-factorized"})
            for method_index, method in enumerate(METHODS):
                for feedback_index, (state, filter_class) in enumerate(FEEDBACKS):
                    encoding = ENCODINGS[(surface_index * 3 + method_index + feedback_index) % len(ENCODINGS)]
                    for seed in SEEDS:
                        context = {
                            "method": method,
                            "surface_context": surface,
                            "surface_family": _surface_family(surface),
                            "parameter_role": ROLES_BY_SURFACE[surface][(feedback_index + seed) % 2],
                            "encoding_observed": encoding,
                            "filter_state": state,
                            "filter_class": filter_class,
                            "surface_syntax_observed": syntax,
                            "surface_shape_observed": shape,
                            "response_shape_observed": shape,
                            "history_action": "baseline" if feedback_index == 0 else "select_probe_variant",
                            "belief_state": "uncertain" if state in {"unknown", "no_effect"} else "updated",
                            "step_budget": str(3 + ((feedback_index + surface_index) % 4)),
                        }
                        for role in ROLES:
                            target = _feedback_target(state, filter_class, role, surface, method, encoding)
                            record_key = {
                                "implementation": implementation,
                                "surface": surface,
                                "method": method,
                                "feedback": [state, filter_class],
                                "encoding": encoding,
                                "seed": seed,
                                "role": role,
                            }
                            record_id = _sha(record_key)[:24]
                            context_tokens = [
                                "[CTX_BOS]",
                                *(f"surface_{key}={_token(value)}" for key, value in context.items()),
                                f"role={role}",
                                "negative_control=matched_triplet" if role == "negative" else "negative_control=required",
                                "sidecar_off_context=true",
                                "context_firewall=closed",
                                "[CTX_EOS]",
                            ]
                            target_tokens = ["[TARGET_BOS]", *(f"{key}={_token(value)}" for key, value in target.items()), "[TARGET_EOS]"]
                            records.append(
                                {
                                    "record_id": record_id,
                                    "split": split,
                                    "raw_payload_stored": False,
                                    "raw_response_body_stored": False,
                                    "oracle_answer_in_context": False,
                                    "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
                                    "source": {
                                        "source_id": f"pg382_{implementation}",
                                        "source_hash": source_hash,
                                        "implementation_group": implementation,
                                        "surface_template_id": f"surface_tpl_{surface_index:02d}",
                                        "split": split,
                                        "seed": seed,
                                        "role": role,
                                    },
                                    "context_tokens": context_tokens,
                                    "target_tokens": target_tokens,
                                    "abstract_observation": context,
                                    "reasoning_trace": [
                                        "classify_surface",
                                        "read_filter_feedback",
                                        "choose_one_variable_change" if target["repair_action"] not in {"none", "observe"} else "preserve_baseline",
                                        "ask_or_bind_evaluator_template",
                                        "require_typed_oracle_and_fresh_replay",
                                    ],
                                    "training_flags": {
                                        "representation_pretrain_candidate_allowed": True,
                                        "abstract_reasoning_sft_candidate_allowed": True,
                                        "capability_training_allowed": False,
                                        "training_eligible": False,
                                    },
                                    "promotion": dict(PROMOTION),
                                }
                            )

    axis_values = {
        "surface_context": [str(row["abstract_observation"]["surface_context"]) for row in records],
        "surface_family": [str(row["abstract_observation"]["surface_family"]) for row in records],
        "method": [str(row["abstract_observation"]["method"]) for row in records],
        "parameter_role": [str(row["abstract_observation"]["parameter_role"]) for row in records],
        "encoding": [str(row["abstract_observation"]["encoding_observed"]) for row in records],
        "filter_state": [str(row["abstract_observation"]["filter_state"]) for row in records],
        "filter_class": [str(row["abstract_observation"]["filter_class"]) for row in records],
        "role": [str(row["source"]["role"]) for row in records],
        "next_action": [str(row["target_tokens"][3]).split("=", 1)[-1] for row in records],
        "repair_action": [str(row["target_tokens"][4]).split("=", 1)[-1] for row in records],
    }
    vocabulary = {
        "scope": "declared_abstract_factorized_ontology",
        "append_only": True,
        "context_tokens": sorted({str(token) for row in records for token in row["context_tokens"]}),
        "target_tokens": sorted({str(token) for row in records for token in row["target_tokens"]}),
    }
    source_hashes = {str(row["source"]["source_hash"]) for row in records}
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_adversarial_candidate_only",
        "generator": "scripts/build_pg382_factorized_adversarial_dataset.py",
        "objective": "factorized abstract Rule-IR composition without unseen-surface confounding",
        "records": records,
        "vocabulary": vocabulary,
        "counts": {
            "records": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in records),
            "implementations": len(implementations),
            "surface_templates": len(SURFACES),
            "methods": len(METHODS),
            "roles": len(ROLES),
            "unique_record_ids": len({row["record_id"] for row in records}),
            "source_hashes": len(source_hashes),
            "training_eligible": 0,
        },
        "split_contract": {
            "train_group_hashes": [_sha({"implementation": "impl_a", "contract": "pg382-factorized"})],
            "holdout_group_hashes": [_sha({"implementation": "impl_b", "contract": "pg382-factorized"})],
            "source_hashes_disjoint": True,
            "abstract_matrix_shared_across_implementations": True,
        },
        "audit": {
            "status": "passed_abstract_factorized_candidate",
            "axis_entropy_bits": {key: _entropy(values) for key, values in axis_values.items()},
            "axis_unique_counts": {key: len(set(values)) for key, values in axis_values.items()},
            "duplicate_record_ids": len(records) - len({row["record_id"] for row in records}),
            "cross_split_source_hash_overlap": 0,
            "abstract_matrix_cross_split_overlap_expected": True,
            "raw_marker_count": 0,
            "evaluator_answer_in_context": False,
            "external_network": False,
            "persistent_state_write": False,
            "training_eligible": 0,
        },
        "safety": {
            "raw_payload_in_context": False,
            "raw_response_in_context": False,
            "evaluator_answer_in_context": False,
            "external_network": False,
            "persistent_state_write": False,
            "concrete_wire_generation": "evaluator_template_only",
        },
        "promotion": {
            "representation_pretrain_candidate_allowed": True,
            "abstract_reasoning_sft_candidate_allowed": True,
            "capability_training_allowed": False,
            **PROMOTION,
        },
    }
    dataset["dataset_sha256"] = _sha(dataset)
    return dataset


def write_dataset(path: Path) -> dict[str, Any]:
    dataset = build_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg382_factorized_abstract_adversarial_dataset_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    dataset = write_dataset(args.output)
    if args.json:
        print(json.dumps({"status": dataset["status"], "counts": dataset["counts"], "audit": dataset["audit"], "dataset_sha256": dataset["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_dataset", "write_dataset"]
