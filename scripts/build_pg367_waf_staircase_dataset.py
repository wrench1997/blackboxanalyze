"""Build abstract WAF-process traces from the PG-367 local evaluator.

The records contain surface/WAF/failure tokens and Rule-IR targets only.  The
evaluator projection is bounded and sidecar-only; no raw probe, response body,
URL, route, family label or bypass literal is serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg367_waf_staircase import POLICIES, build_failure_transition, evaluate_waf_probe

SLOTS = (
    "question", "ask_reason", "next_action", "repair_action", "transport_ref",
    "field_role_ref", "encoding_ref", "syntax_category_ref", "probe_variant_ref",
    "safe_to_send", "payload_shape_ref", "oracle_ref", "negative_control_presence_ref",
)
ROLES = ("candidate", "reference", "negative", "replay")
METHODS = ("GET", "POST")
FIELD_ROLES = ("query_term", "form_field", "display_text", "structured_value")
TARGET_VARIANTS = {
    "candidate": "source_attested_candidate",
    "reference": "reference",
    "negative": "negative_control",
    "replay": "fresh_replay",
}
POLICY_HOLDOUT = {"decode_once_guard", "parser_boundary_guard"}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _target(probe: dict[str, Any], projection: dict[str, Any], role: str, *, failed: bool) -> list[str]:
    method = str(probe["method"]).upper()
    negative = role == "negative"
    success = bool(projection["typed_effect_confirmed"])
    question = "ask_failure" if failed else "none"
    next_action = "repair" if failed else "abstain" if negative else "replay" if role == "replay" else "select_probe_variant"
    repair = str(projection["repair_axis"]) if failed else "none"
    safe = int(not failed and not negative and success)
    oracle = "unknown" if failed else "negative_no_effect" if negative else "typed_effect" if success else "unknown"
    transport = "get_query" if method == "GET" else "post_form"
    shape = "query_marker" if method == "GET" else "html_form_marker"
    return [
        "[TARGET_BOS]",
        f"question={question}",
        f"ask_reason={'failure_feedback' if failed else 'none'}",
        f"next_action={next_action}",
        f"repair_action={repair}",
        f"transport_ref={transport}",
        f"field_role_ref={probe['field_role']}",
        f"encoding_ref={probe['encoding_chain']}",
        f"syntax_category_ref={probe['syntax_category']}",
        f"probe_variant_ref={'none' if failed else TARGET_VARIANTS[role]}",
        f"safe_to_send={safe}",
        f"payload_shape_ref={shape}",
        f"oracle_ref={oracle}",
        "negative_control_presence_ref=matched_triplet",
        "[TARGET_EOS]",
    ]


def _context(probe: dict[str, Any], projection: dict[str, Any], *, step: str, previous_action: str) -> list[str]:
    method = str(probe["method"]).casefold()
    return [
        "chunk_boundary=begin",
        "document_presence=observed",
        "navigation_presence=observed",
        f"request_method={method}",
        f"request_field_role={probe['field_role']}",
        f"request_encoding={probe['encoding_chain']}",
        "response_transport_presence=observed",
        f"response_shape={'effect' if projection['typed_effect_confirmed'] else 'blocked' if projection['failure_signature'] != 'none' else 'neutral'}",
        "javascript_presence=observed",
        "failure_feedback_presence=observed",
        f"failure_process_step={step}",
        f"failure_filter_stage={projection['filter_stage']}",
        f"failure_filter_action={projection['filter_action']}",
        f"failure_transform_class={projection['transform_class']}",
        f"failure_signature={projection['failure_signature']}",
        f"failure_repair_axis={projection['repair_axis']}",
        f"failure_previous_action={previous_action}",
        "belief_replay_presence=observed",
        "belief_typed_available=present",
        "belief_evidence_present=present",
        "belief_negative_control=present",
        "belief_fresh_reset=present",
        "belief_replay_ready=present",
        f"belief_history_action={previous_action}",
        "belief_step_budget=present",
        "waf_presence=observed",
        "waf_policy_stage=abstract",
        "waf_filter_observation=typed_projection",
        "chunk_boundary=end",
    ]


def build() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    policy_counts: dict[str, int] = {}
    for policy in POLICIES:
        split = "implementation_holdout" if policy.policy_id in POLICY_HOLDOUT else "train"
        policy_counts[split] = policy_counts.get(split, 0) + 1
        for method in METHODS:
            for field_role in FIELD_ROLES:
                for role in ROLES:
                    baseline_probe = {"role": role, "method": method, "field_role": field_role, "syntax_category": "marker", "encoding_chain": "identity"}
                    baseline = evaluate_waf_probe(policy, baseline_probe)
                    baseline_row = {
                        "schema_version": "pg367-waf-staircase-row-v1",
                        "record_id": _sha({"policy": policy.policy_id, "method": method, "field_role": field_role, "role": role, "step": "baseline"}),
                        "split": split,
                        "context_tokens": _context(baseline_probe, baseline, step="baseline", previous_action="observe_surface"),
                        "target_tokens": _target(baseline_probe, baseline, role, failed=baseline["failure_signature"] != "none"),
                        "evaluator_projection": baseline,
                        "source_meta": {"implementation": "pg367_local_waf_staircase", "policy_id_hash": baseline["policy_id_hash"], "role": role, "fresh_reset": True, "external_network": False},
                        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
                        "raw_payload_stored": False,
                        "raw_response_body_stored": False,
                        "oracle_answer_in_context": False,
                        "operator_reviewed": False,
                        "training_eligible": False,
                        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
                    }
                    baseline_row["record_sha256"] = _sha(baseline_row)
                    records.append(baseline_row)
                    if baseline["failure_signature"] != "none":
                        repair_probe = {"role": role, "method": method, "field_role": field_role, "syntax_category": policy.accepted_syntax, "encoding_chain": policy.accepted_encoding}
                        transition = build_failure_transition(policy, baseline, repair_probe)
                        after = transition["after_projection"]
                        repair_row = {
                            "schema_version": "pg367-waf-staircase-row-v1",
                            "record_id": _sha({"policy": policy.policy_id, "method": method, "field_role": field_role, "role": role, "step": "repair"}),
                            "split": split,
                            "context_tokens": _context(repair_probe, after, step="repair", previous_action="repair_one_axis"),
                            "target_tokens": _target(repair_probe, after, role, failed=after["failure_signature"] != "none"),
                            "evaluator_projection": after,
                            "failure_transition": transition,
                            "source_meta": {"implementation": "pg367_local_waf_staircase", "policy_id_hash": after["policy_id_hash"], "role": role, "fresh_reset": True, "external_network": False},
                            "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
                            "raw_payload_stored": False,
                            "raw_response_body_stored": False,
                            "oracle_answer_in_context": False,
                            "operator_reviewed": False,
                            "training_eligible": False,
                            "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
                        }
                        repair_row["record_sha256"] = _sha(repair_row)
                        records.append(repair_row)
    return {
        "schema_version": "pg367-waf-staircase-dataset-v1",
        "status": "diagnostic_candidate_only",
        "records": records,
        "counts": {
            "records": len(records),
            "policies": len(POLICIES),
            "train_rows": sum(row["split"] == "train" for row in records),
            "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in records),
            "get_rows": sum(row["context_tokens"][3] == "request_method=get" for row in records),
            "post_rows": sum(row["context_tokens"][3] == "request_method=post" for row in records),
            "failure_rows": sum(any(token.startswith("failure_signature=") and not token.endswith("=none") for token in row["context_tokens"]) for row in records),
            "repair_rows": sum("failure_transition" in row for row in records),
            "roles": len(ROLES),
        },
        "slot_order": list(SLOTS),
        "waf_policy_contract": {
            "filter_levels": [policy.policy_id for policy in POLICIES],
            "raw_payload_in_context": False,
            "evaluator_projection_only": True,
            "fresh_reset_per_role": True,
            "candidate_reference_negative_replay": True,
            "external_network": False,
        },
        "split_contract": {"policy_holdout_literals_in_context": False, "holdout_policy_count": len(POLICY_HOLDOUT), "target_value_coverage": "audit_required"},
        "vocabulary": {"context_tokens": sorted({token for row in records for token in row["context_tokens"]}), "target_tokens": sorted({token for row in records for token in row["target_tokens"]})},
        "failures": [],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-367 abstract WAF staircase data")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg367_waf_staircase_dataset_v1.json")
    args = parser.parse_args()
    document = build()
    document["dataset_sha256"] = _sha(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": document["status"], "counts": document["counts"], "dataset_sha256": document["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
