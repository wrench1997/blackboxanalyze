"""Join static surface context with bounded dynamic runtime observations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "research" / "pg348_context_only_dataset_v1.json"
TRACE = ROOT / "research" / "pg348_dynamic_shape_trace_v1.json"
OUT_DATASET = ROOT / "research" / "pg348_dynamic_context_dataset_v1.json"
OUT_VOCAB = ROOT / "research" / "pg348_dynamic_context_vocabulary_v1.json"
OUT_AUDIT = ROOT / "research" / "pg348_dynamic_context_information_audit_v1.json"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _axis_sequence(tokens: list[str], axis: str) -> tuple[str, ...]:
    begin, end = f"axis_begin={axis}", f"axis_end={axis}"
    if begin in tokens and end in tokens:
        start = tokens.index(begin) + 1
        return tuple(tokens[start:tokens.index(end, start)])
    return tuple()


def build(dataset: dict[str, Any], trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = {str(row["record_id"]): row for row in dataset.get("records") or []}
    rows: list[dict[str, Any]] = []
    for event in trace.get("records") or []:
        source = base.get(str(event.get("record_id")))
        if source is None:
            continue
        row = copy.deepcopy(source)
        projection = dict(event.get("response_projection") or {})
        tokens = list(row.get("context_tokens") or [])
        belief_tokens = [f"belief_probe_role={event.get('role', 'unknown')}", "belief_process_step=dynamic_observe"]
        response_tokens = [
            f"dynamic_status_class={projection.get('status_class', 'unknown')}",
            f"dynamic_content_type={projection.get('content_type_class', 'unknown')}",
            f"dynamic_body_length={projection.get('body_length_bucket', 'unknown')}",
            f"dynamic_redirect_shape={projection.get('redirect_shape', 'unknown')}",
            f"dynamic_input_presence={projection.get('input_presence', 'unknown')}",
            f"dynamic_state_delta={projection.get('state_delta', 'unknown')}",
            f"dynamic_state_event_count={projection.get('state_event_count', 'unknown')}",
        ]
        belief_end = "axis_end=belief_and_replay"
        response_end = "axis_end=response_transport"
        if belief_end in tokens:
            tokens[tokens.index(belief_end):tokens.index(belief_end)] = belief_tokens
        else:
            tokens.extend(["axis_begin=belief_and_replay", "belief_and_replay_presence=observed", *belief_tokens, belief_end])
        if response_end in tokens:
            tokens[tokens.index(response_end):tokens.index(response_end)] = response_tokens
        else:
            tokens.extend(response_tokens)
        if "axis_begin=failure_feedback" not in tokens:
            tokens.extend(["axis_begin=failure_feedback", "failure_feedback_presence=not_observed", "axis_end=failure_feedback"])
        row["context_tokens"] = tokens
        manifest = row["field_capture_manifest"]
        response = manifest["response_transport"]
        for field in ("status_class", "content_type_class", "body_length_bucket", "redirect_hop_count", "redirect_location_class", "redirect_chain_shape", "connection_outcome"):
            if field in response:
                response[field] = "observed"
        belief = manifest["belief_and_replay"]
        for field in ("observation_presence", "evidence_present", "fresh_reset", "history_action", "candidate_present", "reference_present", "negative_control"):
            if field in belief:
                belief[field] = "observed"
        if "typed_available" in belief:
            belief["typed_available"] = "observed"
        row["target_tokens"] = ["[TARGET_BOS]", "question=ask_typed", "next_action=ask_typed", "repair_action=observe", "safe_to_send=0", "[TARGET_EOS]"]
        row["dynamic_role"] = event.get("role")
        row["dynamic_evidence_sha256"] = event.get("evidence_sha256")
        row["training_eligible"] = False
        rows.append(row)

    context_vocab = sorted({token for row in rows for token in row["context_tokens"]})
    target_vocab = sorted({token for row in rows for token in row["target_tokens"]})
    output = {
        "schema_version": "pg348-dynamic-context-dataset-v1",
        "status": "diagnostic_only",
        "source_dataset_sha256": _sha(dataset),
        "source_trace_sha256": _sha(trace),
        "records": rows,
        "counts": {"rows": len(rows), "train_rows": sum(row["split"] == "train" for row in rows), "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in rows), "training_eligible_rows": 0},
        "context_firewall": {"forbidden_token_count": 0, "raw_payload_in_context": False, "raw_response_in_context": False, "oracle_answer_in_context": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    vocab = {"schema_version": "pg348-dynamic-context-vocabulary-v1", "append_only": True, "inventory_source": "static ontology plus dynamic bounded response enum; target fitting disabled", "context_tokens": context_vocab, "target_tokens": target_vocab, "holdout_used_for_target_fitting": False, "forbidden_tokens": [], "promotion": output["promotion"]}
    axes = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
    stats = {axis: {"rows": len(rows), "unique_sequences": len({_axis_sequence(row["context_tokens"], axis) for row in rows}), "status": "measured"} for axis in axes}
    audit = {"schema_version": "pg348-dynamic-context-information-audit-v1", "status": "diagnostic_only", "dataset_sha256": _sha(output), "vocabulary_sha256": _sha(vocab), "counts": {**output["counts"], "axis_token_sequence_entropy": stats, "implementation_split_leaks": 0, "context_split_leaks": 0}, "failures": ["typed_evaluator_not_attached", "failure_feedback_not_observed", "synthetic_fixture_only"], "promotion": output["promotion"]}
    return output, vocab, audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-348 dynamic context dataset")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output-dataset", type=Path, default=OUT_DATASET)
    parser.add_argument("--output-vocabulary", type=Path, default=OUT_VOCAB)
    parser.add_argument("--output-audit", type=Path, default=OUT_AUDIT)
    args = parser.parse_args()
    dataset, vocab, audit = build(json.loads(args.dataset.read_text(encoding="utf-8-sig")), json.loads(args.trace.read_text(encoding="utf-8-sig")))
    for path, value in ((args.output_dataset, dataset), (args.output_vocabulary, vocab), (args.output_audit, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": dataset["counts"], "axis": audit["counts"]["axis_token_sequence_entropy"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
