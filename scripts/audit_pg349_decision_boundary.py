"""Audit whether PG-349 abstract contexts identify their Rule-IR targets.

This is a read-only, fail-closed audit.  It hashes context/target streams and
reports bounded counts only; it never emits tokens, route names, payloads,
response bodies, evaluator answers, or source rows.  A context that maps to
more than one abstract decision is not trainable until the collector observes
the missing feature or the target is changed to ASK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg349_dynamic_typed_source_rows_v5.json"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}
FORBIDDEN = (
    "raw_",
    "payload=",
    "payload_",
    "response_body=",
    "response_body_text=",
    "oracle=",
    "evaluator=",
    "route_literal=",
    "family=",
    "url=",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _stream_hash(tokens: Any) -> str:
    if not isinstance(tokens, list):
        return ""
    return _sha([str(token) for token in tokens])


def _decision(row: Mapping[str, Any]) -> tuple[str, ...]:
    target = row.get("target_projection")
    if not isinstance(target, Mapping):
        target = {}
    return tuple(str(target.get(key, "unknown")) for key in (
        "question",
        "next_action",
        "repair_action",
        "transport_ref",
        "field_role_ref",
        "encoding_ref",
        "probe_variant_ref",
        "payload_shape_ref",
        "safe_to_send",
    ))


def _variant(row: Mapping[str, Any]) -> str:
    target = row.get("target_projection")
    return str(target.get("probe_variant_ref", "unknown")) if isinstance(target, Mapping) else "unknown"


def _valid_row(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    context = row.get("context_tokens")
    target = row.get("target_tokens")
    if not isinstance(context, list) or not context:
        failures.append("context_missing")
    if not isinstance(target, list) or not target:
        failures.append("target_missing")
    if str(row.get("split")) not in {"train", "implementation_holdout"}:
        failures.append("split_invalid")
    firewall = row.get("context_firewall")
    if not isinstance(firewall, Mapping) or firewall.get("forbidden_token_count") != 0 or firewall.get("sidecars_off_context") is not True:
        failures.append("context_firewall")
    for token in [*(context if isinstance(context, list) else []), *(target if isinstance(target, list) else [])]:
        text = str(token).casefold()
        if text.startswith("payload_shape_ref="):
            continue
        if any(fragment in text for fragment in FORBIDDEN):
            failures.append("forbidden_token")
            break
    return not failures, sorted(set(failures))


def audit_document(document: Mapping[str, Any]) -> dict[str, Any]:
    raw_rows = document.get("records")
    rows = [row for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
    failures: list[str] = []
    contexts: dict[str, set[str]] = defaultdict(set)
    decisions: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    contexts_by_split: dict[str, set[str]] = defaultdict(set)
    source_groups: dict[str, set[str]] = defaultdict(set)
    split_counts = {"train": 0, "implementation_holdout": 0}
    valid_count = 0
    variant_counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    safe_counts = {"safe": 0, "reject": 0}
    for row in rows:
        valid, row_failures = _valid_row(row)
        failures.extend(row_failures)
        split = str(row.get("split"))
        if split in split_counts:
            split_counts[split] += 1
        if not valid:
            continue
        valid_count += 1
        context_hash = _stream_hash(row.get("context_tokens"))
        target_hash = _stream_hash(row.get("target_tokens"))
        contexts[context_hash].add(target_hash)
        decisions[context_hash].add(_decision(row))
        contexts_by_split[split].add(context_hash)
        variant = _variant(row)
        variant_counts[variant] += 1
        target = row.get("target_projection")
        action = str(target.get("next_action", "unknown")) if isinstance(target, Mapping) else "unknown"
        action_counts[action] += 1
        safe_counts["safe" if isinstance(target, Mapping) and target.get("safe_to_send") is True else "reject"] += 1
        source = row.get("source_meta")
        surface = str(source.get("surface_id", "unknown")) if isinstance(source, Mapping) else "unknown"
        source_groups[surface].add(variant)
    conflict_contexts = sum(1 for values in contexts.values() if len(values) > 1)
    decision_conflicts = sum(1 for values in decisions.values() if len(values) > 1)
    leakage = len(contexts_by_split["train"] & contexts_by_split["implementation_holdout"])
    paired_complete = sum(1 for values in source_groups.values() if {"source_attested_candidate", "reference", "negative_control", "runtime_canary"} <= values)
    status = "passed_decision_boundary_diagnostic" if rows and valid_count == len(rows) and not failures and conflict_contexts == 0 and decision_conflicts == 0 and leakage == 0 else "blocked_context_target_ambiguity"
    result = {
        "schema_version": "pg349-decision-boundary-audit-v1",
        "status": status,
        "record_count": len(rows),
        "valid_record_count": valid_count,
        "split_counts": split_counts,
        "unique_context_count": len(contexts),
        "unique_context_target_pairs": sum(len(values) for values in contexts.values()),
        "context_target_conflict_count": conflict_contexts,
        "decision_conflict_count": decision_conflicts,
        "train_holdout_context_overlap": leakage,
        "source_surface_groups": len(source_groups),
        "complete_candidate_reference_negative_replay_groups": paired_complete,
        "variant_counts": dict(sorted(variant_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "safe_counts": safe_counts,
        "failures": sorted(set(failures)),
        "promotion": dict(PROMOTION),
    }
    result["audit_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-349 context/Rule-IR decision identifiability")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    if not isinstance(document, Mapping):
        raise ValueError("dataset must contain an object")
    report = audit_document(document)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if report["status"].startswith("passed_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
