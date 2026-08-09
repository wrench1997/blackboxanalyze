"""Build an abstract filter-feedback/repair dataset for PG-385.

The records describe what was filtered and which single abstract axis should
change.  They intentionally contain no literal canary, URL, response body,
wire, evaluator answer, or route name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pg385-filter-repair-abstract-dataset-v1"
ROLES = ("candidate", "reference", "negative", "replay")
METHODS = ("GET", "POST")
SCENARIOS = (
    {
        "scenario_id": "encoding_canonicalization",
        "surface_context": "query",
        "parameter_role": "query_term",
        "filter_class": "encoding_filter",
        "initial_encoding": "identity",
        "repair_encoding": "double_layer_order_sensitive",
        "initial_syntax": "delimiter_boundary",
        "repair_syntax": "structured_value",
        "initial_shape": "query_marker",
        "repair_shape": "query_marker",
        "oracle": "response_shape",
        "repair_action": "encoding",
    },
    {
        "scenario_id": "delimiter_syntax_gate",
        "surface_context": "html_text",
        "parameter_role": "display_text",
        "filter_class": "delimiter_rejected",
        "initial_encoding": "identity",
        "repair_encoding": "url_percent",
        "initial_syntax": "delimiter_boundary",
        "repair_syntax": "structured_value",
        "initial_shape": "html_fragment_marker",
        "repair_shape": "html_text_marker",
        "oracle": "reflection",
        "repair_action": "syntax",
    },
    {
        "scenario_id": "shape_length_gate",
        "surface_context": "query",
        "parameter_role": "query_term",
        "filter_class": "shape_filter",
        "initial_encoding": "identity",
        "repair_encoding": "identity",
        "initial_syntax": "marker",
        "repair_syntax": "marker",
        "initial_shape": "html_fragment_marker",
        "repair_shape": "query_marker",
        "oracle": "response_shape",
        "repair_action": "shape",
    },
    {
        "scenario_id": "parser_boundary_recovery",
        "surface_context": "json_string",
        "parameter_role": "structured_value",
        "filter_class": "parser_boundary",
        "initial_encoding": "json_escape",
        "repair_encoding": "json_escape",
        "initial_syntax": "expression_node",
        "repair_syntax": "structured_value",
        "initial_shape": "json_string_marker",
        "repair_shape": "json_string_marker",
        "oracle": "parser_shape",
        "repair_action": "syntax",
    },
    {
        "scenario_id": "missing_filter_observation",
        "surface_context": "query",
        "parameter_role": "query_term",
        "filter_class": "unknown",
        "initial_encoding": "unknown",
        "repair_encoding": "unknown",
        "initial_syntax": "unknown",
        "repair_syntax": "unknown",
        "initial_shape": "query_marker",
        "repair_shape": "query_marker",
        "oracle": "unknown",
        "repair_action": "observe",
        "missing_observation": True,
    },
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _context_tokens(spec: dict[str, str], method: str, role: str) -> list[str]:
    missing = bool(spec.get("missing_observation"))
    return [
        "[CTX_BOS]",
        f"method={method}",
        f"surface_context={spec['surface_context']}",
        f"parameter_role={spec['parameter_role']}",
        f"filter_state={'unknown' if missing else 'filtered'}",
        f"filter_class={spec['filter_class']}",
        f"encoding_observed={spec['initial_encoding']}",
        f"syntax_observed={spec['initial_syntax']}",
        f"shape_observed={spec['initial_shape']}",
        f"response_shape={'unknown' if missing else 'bounded_projection'}",
        f"role={role}",
        f"history_action={'observe' if missing else 'baseline_send'}",
        "replay_state=fresh_reset_required",
        "[CTX_EOS]",
    ]


def _target_tokens(spec: dict[str, str], method: str, role: str) -> list[str]:
    if spec.get("missing_observation"):
        values = {
            "question": "ask_observation",
            "ask_reason": "missing_filter_feedback",
            "next_action": "ask",
            "repair_action": "observe",
            "encoding_ref": "unknown",
            "syntax_category_ref": "unknown",
            "payload_shape_ref": spec["initial_shape"],
            "oracle_ref": "unknown",
            "safe_to_send": "0",
        }
    elif role == "negative":
        values = {
            "question": "none",
            "ask_reason": "negative_control",
            "next_action": "abstain",
            "repair_action": "none",
            "encoding_ref": spec["initial_encoding"],
            "syntax_category_ref": spec["initial_syntax"],
            "payload_shape_ref": spec["initial_shape"],
            "oracle_ref": "negative_no_effect",
            "safe_to_send": "0",
        }
    else:
        values = {
            "question": "none",
            "ask_reason": "none",
            "next_action": "repair",
            "repair_action": spec["repair_action"],
            "encoding_ref": spec["repair_encoding"],
            "syntax_category_ref": spec["repair_syntax"],
            "payload_shape_ref": spec["repair_shape"],
            "oracle_ref": spec["oracle"],
            "safe_to_send": "1",
        }
    ordered = [
        ("question", values["question"]),
        ("ask_reason", values["ask_reason"]),
        ("next_action", values["next_action"]),
        ("repair_action", values["repair_action"]),
        ("transport_ref", "get_query" if method == "GET" else "post_form"),
        ("field_role_ref", spec["parameter_role"]),
        ("encoding_ref", values["encoding_ref"]),
        ("syntax_category_ref", values["syntax_category_ref"]),
        ("probe_variant_ref", "one_variable_repair" if role != "negative" else "unsupported_variant"),
        ("safe_to_send", values["safe_to_send"]),
        ("payload_shape_ref", values["payload_shape_ref"]),
        ("oracle_ref", values["oracle_ref"]),
        ("negative_control_presence_ref", "matched_triplet" if role == "negative" else "unknown"),
    ]
    return ["[TARGET_BOS]"] + [f"{key}={value}" for key, value in ordered] + ["[TARGET_EOS]"]


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for implementation, split in (("pg385_filter_impl_a", "train"), ("pg385_filter_impl_b", "implementation_holdout")):
        source_hash = _sha({"implementation": implementation, "fixture_contract": SCHEMA_VERSION})
        for seed in (38501, 38502):
            for method in METHODS:
                for spec in SCENARIOS:
                    for role in ROLES:
                        context_tokens = _context_tokens(spec, method, role)
                        target_tokens = _target_tokens(spec, method, role)
                        identity = {
                            "implementation": implementation,
                            "seed": seed,
                            "method": method,
                            "scenario_id": spec["scenario_id"],
                            "role": role,
                        }
                        row = {
                            "record_id": "pg385-" + _sha(identity)[:20],
                            "split": split,
                            "implementation_id": implementation,
                            "source_hash": source_hash,
                            "seed": seed,
                            "method": method,
                            "scenario_id": spec["scenario_id"],
                            "role": role,
                            "context_tokens": context_tokens,
                            "target_tokens": target_tokens,
                            "field_capture_manifest": {
                                "surface_context": "observed",
                                "parameter_role": "observed",
                                "filter_feedback": "observed",
                                "response_shape": "observed",
                                "raw_value": "not_observed",
                                "raw_response": "not_observed",
                            },
                            "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
                            "raw_payload_stored": False,
                            "raw_response_body_stored": False,
                            "oracle_answer_in_context": False,
                            "training_eligible": False,
                            "promotion": {
                                "training_allowed": False,
                                "memory_promotion_allowed": False,
                                "payload_catalog_promotion_allowed": False,
                                "vulnerability_claim_allowed": False,
                            },
                        }
                        row["record_sha256"] = _sha(row)
                        records.append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_adversarial_candidate_only",
        "objective": "filtered canary -> abstract feedback -> one-variable repair -> evaluator last-hop canary",
        "counts": {
            "records": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in records),
            "implementations": 2,
            "seeds": 2,
            "methods": 2,
            "scenarios": len(SCENARIOS),
            "roles": len(ROLES),
        },
        "records": records,
        "vocabulary": {
            "scope": "declared_abstract_ontology",
            "context_tokens": sorted({token for row in records for token in row["context_tokens"]}),
            "target_tokens": sorted({token for row in records for token in row["target_tokens"]}),
        },
        "safety": {
            "abstract_only": True,
            "raw_payload_in_context": False,
            "raw_value_in_context": False,
            "raw_response_in_context": False,
            "evaluator_answer_in_context": False,
            "wire_in_context": False,
            "external_network": False,
            "training_eligible": 0,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    payload["dataset_sha256"] = _sha(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-385 abstract filter repair dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg385_filter_repair_adversarial_dataset_v1.json")
    args = parser.parse_args()
    payload = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "counts": payload["counts"], "dataset_sha256": payload["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
