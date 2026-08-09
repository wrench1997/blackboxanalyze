"""Read-only audit for the PG-351 ASK + Rule-IR candidate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg351_ask_oracle_composition_audit_v1.json"
RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "response_body_text=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=", "implementation=", "image=", "source=")
TARGET_PREFIXES = ("[TARGET_BOS]", "[TARGET_EOS]", "question=", "ask_reason=", "next_action=", "repair_action=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=", "safe_to_send=", "payload_shape_ref=", "oracle_ref=", "negative_control_presence_ref=")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entropy(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    bits = -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.0
    return {"count": total, "unique": len(counts), "bits": round(bits, 6), "values": dict(sorted(counts.items()))}


def audit(dataset: Mapping[str, Any], *, dataset_sha256: str) -> dict[str, Any]:
    failures: list[str] = []
    records = [row for row in dataset.get("records", []) if isinstance(row, Mapping)]
    contexts: dict[str, set[str]] = defaultdict(set)
    split_contexts: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[str] = set()
    lanes = Counter()
    questions: list[str] = []
    actions: list[str] = []
    split_action: dict[str, Counter[str]] = defaultdict(Counter)
    axis_presence = Counter()
    vocab_context = {str(token) for token in (dataset.get("vocabulary") or {}).get("context_tokens", [])}
    vocab_target = {str(token) for token in (dataset.get("vocabulary") or {}).get("target_tokens", [])}
    for index, row in enumerate(records):
        prefix = f"row_{index}"
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        if row.get("training_eligible") is not False or row.get("candidate_training_allowed") is not True:
            failures.append(f"{prefix}:candidate_flag")
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"{prefix}:context_firewall")
        if any(row.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"{prefix}:raw_flag")
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in context):
            failures.append(f"{prefix}:raw_context_token")
        if any(not token.startswith(TARGET_PREFIXES) for token in target):
            failures.append(f"{prefix}:target_not_abstract")
        if not target or target[0] != "[TARGET_BOS]" or target[-1] != "[TARGET_EOS]":
            failures.append(f"{prefix}:target_stream")
        pair = _sha({"context_tokens": context, "target_tokens": target})
        if pair in seen_pairs:
            failures.append("duplicate_context_target_pair")
        seen_pairs.add(pair)
        context_hash = _sha(context)
        target_hash = _sha(target)
        contexts[context_hash].add(target_hash)
        split_contexts[str(row.get("split"))].add(context_hash)
        lanes[str(row.get("supervision_lane"))] += 1
        q = next((token.split("=", 1)[1] for token in target if token.startswith("question=")), "none")
        a = next((token.split("=", 1)[1] for token in target if token.startswith("next_action=")), "none")
        questions.append(q)
        actions.append(a)
        split_action[str(row.get("split"))][a] += 1
        safe_token = next((token for token in target if token.startswith("safe_to_send=")), None)
        expected_safe = safe_token == "safe_to_send=1"
        if row.get("safe_to_send") is not expected_safe:
            failures.append(f"{prefix}:safe_slot_mismatch")
        if str(row.get("supervision_lane")) == "ask_missing_observation" and safe_token != "safe_to_send=0":
            failures.append(f"{prefix}:ask_safe_to_send_not_zero")
        for axis in ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay"):
            if f"axis_begin={axis}" in context:
                axis_presence[axis] += 1
        if not set(context) <= vocab_context:
            failures.append(f"{prefix}:context_vocab")
        if not set(target) <= vocab_target:
            failures.append(f"{prefix}:target_vocab")
    context_conflicts = sum(len(values) > 1 for values in contexts.values())
    train_overlap = len(split_contexts.get("train", set()) & split_contexts.get("implementation_holdout", set()))
    if context_conflicts:
        failures.append("context_target_conflict")
    if train_overlap:
        failures.append("train_holdout_context_overlap")
    required = ("ask_typed", "select_probe_variant", "abstain", "repair", "replay")
    for split in ("train", "implementation_holdout"):
        if any(split_action[split].get(action, 0) == 0 for action in required):
            failures.append(f"{split}:target_coverage")
    if lanes.get("ask_missing_observation", 0) == 0:
        failures.append("ask_lane_missing")
    split_counts = Counter(str(row.get("split")) for row in records)
    result = {
        "schema_version": "pg351-ask-oracle-composition-audit-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "dataset": "research/pg351_ask_oracle_composition_dataset_v1.json",
        "dataset_sha256": dataset_sha256,
        "record_count": len(records),
        "split_counts": dict(split_counts),
        "supervision_lanes": dict(lanes),
        "question_entropy": _entropy(questions),
        "next_action_entropy": _entropy(actions),
        "split_action_coverage": {split: dict(counts) for split, counts in sorted(split_action.items())},
        "axis_presence_counts": dict(axis_presence),
        "unique_context_target_pairs": len(seen_pairs),
        "context_target_conflict_groups": context_conflicts,
        "train_holdout_context_overlap": train_overlap,
        "vocabulary_missing": {
            "context": sum(1 for row in records for token in row.get("context_tokens", []) if str(token) not in vocab_context),
            "target": sum(1 for row in records for token in row.get("target_tokens", []) if str(token) not in vocab_target),
        },
        "information_gate": {
            "status": "diagnostic_candidate_only",
            "ask_supervision_present": lanes.get("ask_missing_observation", 0) > 0,
            "typed_rule_ir_present": lanes.get("typed_replay", 0) > 0,
            "accepted_training_rows": 0,
            "candidate_target_conditioned_smoke_allowed": not failures,
        },
        "warnings": ["candidate_training_is_abstract_only", "missing_observation_lane_is_not_vulnerability_gold"],
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-351 ASK/oracle composition candidate dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    result = audit(dataset, dataset_sha256=_file_sha(args.dataset))
    result["dataset"] = str(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "records": result["record_count"], "failures": result["failures"], "output_sha256": _file_sha(args.output)}, ensure_ascii=False))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
