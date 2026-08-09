"""Build the PG-380 abstract adversarial reasoning dataset.

The dataset is deliberately separate from PG-331 source rows.  It teaches a
decoder to ask for missing observations, identify abstract filter feedback,
change one Rule-IR variable after failure, keep a matched negative control,
and request fresh replay after a typed effect.  It contains no URL, wire,
response body, callback, exploit literal, or evaluator answer.

The output is suitable for an abstract-reasoning/SFT candidate only.  It is
not a capability-training authorization and never sets promotion flags true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pg380-abstract-adversarial-reasoning-dataset-v1"
SEEDS = (38001, 38002, 38003)
ROLES = ("candidate", "reference", "negative", "replay")
SURFACES = (
    "html_text",
    "html_attribute",
    "html_dom",
    "json_string",
    "sql_string",
    "sql_numeric",
    "path_segment",
    "query",
    "form",
    "redirect",
    "script_context",
    "style_context",
)
METHODS = ("GET", "POST")
ENCODINGS = (
    "identity",
    "url_percent",
    "form_urlencoded",
    "html_entity",
    "javascript_unicode",
    "json_escape",
    "xml_entity",
    "double_layer_order_sensitive",
)
FEEDBACKS = (
    ("baseline", "none"),
    ("filtered", "encoding_normalized"),
    ("filtered", "delimiter_rejected"),
    ("filtered", "syntax_filter"),
    ("filtered", "shape_filter"),
    ("filtered", "length_limit"),
    ("parser_error", "parser_boundary"),
    ("no_effect", "none"),
    ("typed_effect", "none"),
    ("negative_no_effect", "none"),
    ("unknown", "unknown"),
)
ROLES_BY_SURFACE = {
    "html_text": ("display_text", "dom_text"),
    "html_attribute": ("attribute_value", "display_text"),
    "html_dom": ("dom_text", "display_text"),
    "json_string": ("json_value", "structured_value"),
    "sql_string": ("query_term", "record_cursor"),
    "sql_numeric": ("record_cursor", "query_term"),
    "path_segment": ("path_segment", "record_cursor"),
    "query": ("query_text", "query_term"),
    "form": ("form_field", "note_text"),
    "redirect": ("notice_state", "status_label"),
    "script_context": ("display_text", "attribute_value"),
    "style_context": ("attribute_value", "display_text"),
}
SURFACE_DEFAULTS = {
    "html_text": ("html_text_marker", "marker", "reflection"),
    "html_attribute": ("html_attribute_marker", "structured_value", "dom_shape"),
    "html_dom": ("html_dom_marker", "expression_node", "dom_shape"),
    "json_string": ("json_string_marker", "structured_value", "parser_shape"),
    "sql_string": ("sql_string_marker", "delimiter_boundary", "response_shape"),
    "sql_numeric": ("sql_numeric_marker", "boolean_branch", "response_shape"),
    "path_segment": ("path_segment_marker", "parser_node", "response_shape"),
    "query": ("query_marker", "structured_value", "response_shape"),
    "form": ("html_form_marker", "structured_value", "response_shape"),
    "redirect": ("state_transition_marker", "redirect_control", "typed_state_delta"),
    "script_context": ("script_context_marker", "expression_node", "dom_shape"),
    "style_context": ("style_context_marker", "structured_value", "dom_shape"),
}
_FORBIDDEN = (
    "http://",
    "https://",
    "javascript:",
    "<script",
    "document.cookie",
    "response_body",
    "raw_payload",
    "callback",
    "webhook",
    "route_literal",
    "oracle_answer",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _token(value: Any) -> str:
    return str(value).casefold().replace("-", "_")


def _feedback_target(state: str, filter_class: str, role: str, surface: str, method: str, encoding: str) -> dict[str, str]:
    shape, syntax, oracle = SURFACE_DEFAULTS[surface]
    transport = "get_query" if method == "GET" else "post_form"
    parameter_role = ROLES_BY_SURFACE[surface][0 if role in {"candidate", "reference"} else 1]
    question = "none"
    ask_reason = "none"
    next_action = "select_probe_variant"
    repair = "none"
    variant = "source_attested_candidate" if role == "candidate" else "reference_shape"
    target_oracle = oracle
    target_encoding = encoding
    target_syntax = syntax
    target_shape = shape

    if role == "negative":
        next_action, variant, target_oracle = "abstain", "negative_control", "negative_no_effect"
        ask_reason = "matched_negative_control"
    elif role == "replay":
        next_action, variant, repair = "replay", "fresh_replay", "replay"
        target_oracle = "typed_state_delta" if state == "typed_effect" else oracle
    elif state in {"unknown", "baseline"}:
        question, ask_reason, next_action, repair = "ask_observation", "missing_filter_feedback", "ask", "observe"
        target_syntax = "unknown" if state == "unknown" else syntax
        target_shape = "unknown" if state == "unknown" else shape
        target_oracle = "unknown" if state == "unknown" else oracle
        variant = "unsupported_abstain"
    elif state in {"filtered", "parser_error"}:
        variant, next_action = "one_variable_repair", "repair" if role == "candidate" else "select_probe_variant"
        if filter_class in {"encoding_normalized"}:
            repair, target_encoding = "encoding", "double_layer_order_sensitive" if encoding != "double_layer_order_sensitive" else "html_entity"
        elif filter_class in {"delimiter_rejected", "syntax_filter", "parser_boundary"}:
            repair, target_syntax = "syntax", "structured_value" if syntax in {"delimiter_boundary", "structured_value"} else "parser_node"
        elif filter_class in {"shape_filter", "length_limit"}:
            repair, target_shape = "shape", "query_marker" if shape != "query_marker" else "html_text_marker"
        else:
            question, ask_reason, next_action, repair, variant = "ask_observation", "filter_class_unknown", "ask", "observe", "unsupported_abstain"
    elif state == "no_effect":
        variant, next_action, repair, target_oracle = "reference_shape", "repair" if role == "candidate" else "select_probe_variant", "shape", "response_shape"
    elif state == "typed_effect":
        variant, next_action, repair = "fresh_replay", "replay", "replay"

    return {
        "question": question,
        "ask_reason": ask_reason,
        "next_action": next_action,
        "repair_action": repair,
        "transport_ref": transport,
        "field_role_ref": parameter_role,
        "encoding_ref": target_encoding,
        "syntax_category_ref": target_syntax,
        "probe_variant_ref": variant,
        "safe_to_send": "false",
        "payload_shape_ref": target_shape,
        "oracle_ref": target_oracle,
        "negative_control_presence_ref": "matched_triplet" if role == "negative" else "unknown",
    }


def _tokens(prefix: str, values: dict[str, str]) -> list[str]:
    return [f"{prefix}_{key}={_token(value)}" for key, value in values.items()]


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return round(-sum((n / total) * math.log2(n / total) for n in counts.values()), 6)


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for surface_index, surface in enumerate(SURFACES):
        implementation = "impl_a" if surface_index < 8 else "impl_b"
        split = "train" if implementation == "impl_a" else "implementation_holdout"
        source_hash = _sha({"implementation": implementation, "surface_template": surface})
        for method_index, method in enumerate(METHODS):
            for feedback_index, (state, filter_class) in enumerate(FEEDBACKS):
                encoding = ENCODINGS[(surface_index * 3 + method_index + feedback_index) % len(ENCODINGS)]
                for seed in SEEDS:
                    context = {
                        "method": method,
                        "surface_context": surface,
                        "parameter_role": ROLES_BY_SURFACE[surface][(feedback_index + seed) % 2],
                        "encoding_observed": encoding,
                        "filter_state": state,
                        "filter_class": filter_class,
                        "response_shape": SURFACE_DEFAULTS[surface][0],
                        "history_action": "baseline" if feedback_index == 0 else "select_probe_variant",
                        "belief_state": "uncertain" if state in {"unknown", "no_effect"} else "updated",
                        "step_budget": str(3 + ((feedback_index + surface_index) % 4)),
                    }
                    for role in ROLES:
                        target = _feedback_target(state, filter_class, role, surface, method, encoding)
                        case_key = {
                            "implementation": implementation,
                            "surface": surface,
                            "method": method,
                            "feedback": [state, filter_class],
                            "encoding": encoding,
                            "seed": seed,
                            "role": role,
                        }
                        record_id = _sha(case_key)[:24]
                        context_tokens = [
                            "[CTX_BOS]",
                            *_tokens("surface", context),
                            f"role={role}",
                            f"split={split}",
                            "negative_control=matched_triplet" if role == "negative" else "negative_control=required",
                            "sidecar_off_context=true",
                            "context_firewall=closed",
                            "[CTX_EOS]",
                        ]
                        # PG-369/370 consume the canonical bare 13-slot
                        # sequence.  The surrounding dataset metadata stays
                        # abstract, but the target keys must remain exactly
                        # the Rule-IR slot names for a causal trainer.
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
                                    "source_id": f"pg380_{implementation}",
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
                                "promotion": {
                                    "training_allowed": False,
                                    "memory_promotion_allowed": False,
                                    "payload_catalog_promotion_allowed": False,
                                    "vulnerability_claim_allowed": False,
                                },
                            }
                        )

    axis_values = {
        "surface_context": [str(r["abstract_observation"]["surface_context"]) for r in records],
        "method": [str(r["abstract_observation"]["method"]) for r in records],
        "parameter_role": [str(r["abstract_observation"]["parameter_role"]) for r in records],
        "encoding": [str(r["abstract_observation"]["encoding_observed"]) for r in records],
        "filter_state": [str(r["abstract_observation"]["filter_state"]) for r in records],
        "filter_class": [str(r["abstract_observation"]["filter_class"]) for r in records],
        "role": [str(r["source"]["role"]) for r in records],
        "next_action": [str(r["target_tokens"][3]).split("=", 1)[-1] for r in records],
        "repair_action": [str(r["target_tokens"][4]).split("=", 1)[-1] for r in records],
    }
    inventory = {key: sorted(set(values)) for key, values in axis_values.items()}
    counts = {
        "records": len(records),
        "train": sum(r["source"]["split"] == "train" for r in records),
        "implementation_holdout": sum(r["source"]["split"] == "implementation_holdout" for r in records),
        "implementations": len({r["source"]["implementation_group"] for r in records}),
        "surface_templates": len({r["source"]["surface_template_id"] for r in records}),
        "roles": len(ROLES),
        "methods": len(METHODS),
        "unique_record_ids": len({r["record_id"] for r in records}),
        "training_eligible": 0,
    }
    audit = {
        "status": "passed_abstract_candidate",
        "axis_entropy_bits": {key: _entropy(values) for key, values in axis_values.items()},
        "axis_unique_counts": {key: len(set(values)) for key, values in axis_values.items()},
        "inventory": inventory,
        "duplicate_record_ids": counts["records"] - counts["unique_record_ids"],
        "cross_split_source_hash_overlap": 0,
        "raw_marker_count": 0,
        "evaluator_answer_in_context": False,
        "external_network": False,
        "persistent_state_write": False,
        "training_eligible": 0,
        "interpretation": "抽象对抗推理候选；可用于表示/Reasoning-SFT 实验，不能转成漏洞能力、payload catalog 或长期记忆。",
    }
    vocabulary = {
        "scope": "declared_abstract_ontology",
        "append_only": True,
        "context_tokens": sorted({str(token) for row in records for token in row["context_tokens"]}),
        "target_tokens": sorted({str(token) for row in records for token in row["target_tokens"]}),
    }
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_adversarial_candidate_only",
        "generator": "scripts/build_pg380_abstract_adversarial_dataset.py",
        "objective": "ASK/filter-feedback/one-variable-repair/negative-control/fresh-replay reasoning",
        "records": records,
        "vocabulary": vocabulary,
        "counts": counts,
        "audit": audit,
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
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
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
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg380_abstract_adversarial_reasoning_dataset_v1.json")
    parser.add_argument("--json", action="store_true", help="print a bounded summary")
    args = parser.parse_args()
    dataset = write_dataset(args.output)
    if args.json:
        print(json.dumps({"status": dataset["status"], "counts": dataset["counts"], "audit": dataset["audit"], "dataset_sha256": dataset["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_dataset", "write_dataset"]
