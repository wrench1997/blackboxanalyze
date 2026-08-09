"""Read-only PG-350 audit for abstract oracle/negative-control target slots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from app.pg331_source_row import validate_pg331_source_row


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
DEFAULT_VOCAB = ROOT / "research" / "pg350_oracle_slot_vocabulary_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg350_oracle_slot_audit_v1.json"
SLOTS = ("payload_shape_ref", "oracle_ref", "negative_control_presence_ref")
FORBIDDEN_CONTEXT = ("raw_payload=", "payload=", "response_body=", "wire=", "oracle=", "evaluator=")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_map(tokens: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            result[key] = value
    return result


def _entropy(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    nats = -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0
    return {"count": total, "unique": len(counts), "bits": round(nats / math.log(2), 6) if total else 0.0}


def audit(dataset: dict[str, Any], vocabulary: dict[str, Any], *, dataset_sha256: str, vocabulary_sha256: str) -> dict[str, Any]:
    rows = [dict(row) for row in list(dataset.get("records") or []) if isinstance(row, dict)]
    failures: list[str] = []
    invalid = 0
    slot_values: dict[str, list[str]] = {slot: [] for slot in SLOTS}
    context_targets: defaultdict[str, set[str]] = defaultdict(set)
    split_contexts: defaultdict[str, set[str]] = defaultdict(set)
    target_tokens: set[str] = set()
    context_tokens: set[str] = set()
    for row in rows:
        check = validate_pg331_source_row(row)
        if not check["valid"]:
            invalid += 1
        context = [str(token) for token in row.get("context_tokens") or []]
        target = _target_map(list(row.get("target_tokens") or []))
        if any(any(fragment in token.casefold() for fragment in FORBIDDEN_CONTEXT) for token in context):
            failures.append("context_firewall")
        for slot in SLOTS:
            if slot not in target:
                failures.append(f"missing_target_slot:{slot}")
            else:
                slot_values[slot].append(target[slot])
        context_hash = _sha(context)
        target_hash = _sha(target)
        context_targets[context_hash].add(target_hash)
        split_contexts[str(row.get("split", "missing"))].add(context_hash)
        target_tokens.update(str(token) for token in row.get("target_tokens") or [])
        context_tokens.update(context)
    conflicts = sum(1 for values in context_targets.values() if len(values) > 1)
    train_overlap = len(split_contexts.get("train", set()) & split_contexts.get("implementation_holdout", set()))
    vocabulary_target = set(str(token) for token in vocabulary.get("target_tokens") or [])
    vocabulary_context = set(str(token) for token in vocabulary.get("context_tokens") or [])
    missing_target_vocab = sorted(target_tokens - vocabulary_target)
    missing_context_vocab = sorted(context_tokens - vocabulary_context)
    if invalid:
        failures.append("invalid_rows")
    if conflicts:
        failures.append("context_target_conflict")
    if train_overlap:
        failures.append("train_holdout_overlap")
    if missing_target_vocab:
        failures.append("target_vocabulary_missing")
    if missing_context_vocab:
        failures.append("context_vocabulary_missing")
    context_lengths = [len(row.get("context_tokens") or []) for row in rows]
    target_lengths = [len(row.get("target_tokens") or []) for row in rows]
    required_window = math.ceil((max(context_lengths, default=0) + max(target_lengths, default=0)) * 1.25)
    warnings: list[str] = []
    if slot_values["negative_control_presence_ref"] and len(set(slot_values["negative_control_presence_ref"])) == 1:
        warnings.append("negative_control_presence_ref_zero_entropy")
    return {
        "schema_version": "pg350-oracle-slot-audit-v1",
        "status": "diagnostic_only" if not failures else "blocked_incomplete",
        "dataset": "research/pg350_oracle_slot_source_rows_v1.json",
        "dataset_sha256": dataset_sha256,
        "vocabulary": "research/pg350_oracle_slot_vocabulary_v1.json",
        "vocabulary_sha256": vocabulary_sha256,
        "record_count": len(rows),
        "split_counts": dict(Counter(str(row.get("split", "missing")) for row in rows)),
        "target_slot_entropy": {slot: _entropy(values) for slot, values in slot_values.items()},
        "target_slot_coverage": {slot: len(values) == len(rows) for slot, values in slot_values.items()},
        "context_target_conflict_groups": conflicts,
        "train_holdout_context_overlap": train_overlap,
        "invalid_rows": invalid,
        "vocabulary_missing": {"context": len(missing_context_vocab), "target": len(missing_target_vocab)},
        "sequence_capacity": {"context_max": max(context_lengths, default=0), "target_max": max(target_lengths, default=0), "required_context_window": required_window, "balanced_max_length": max(1024, required_window), "legacy_max_length": 72, "legacy_truncation_allowed": False},
        "information_gate": {"status": "diagnostic_only", "slot_coverage": not failures, "predictive_entropy_holdout": "not_run", "accepted_training_rows": 0},
        "warnings": warnings,
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8-sig"))
    report = audit(dataset, vocabulary, dataset_sha256=_file_sha(args.dataset), vocabulary_sha256=_file_sha(args.vocabulary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "record_count": report["record_count"], "failures": report["failures"], "output_sha256": _file_sha(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "diagnostic_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
