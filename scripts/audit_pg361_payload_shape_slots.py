"""Read-only audit for PG-361 abstract payload-shape slots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import validate_pg331_source_row
from app.pg361_payload_shape_slots import SLOT_ORDER


DEFAULT_DATASET = ROOT / "research" / "pg361_payload_shape_slot_source_rows_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg361_payload_shape_slot_audit_v1.json"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _entropy(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    nats = -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0
    return {"count": total, "unique": len(counts), "bits": round(nats / math.log(2), 6) if total else 0.0, "counts": dict(sorted(counts.items()))}


def _target(tokens: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            result[key] = value
    return result


def audit(dataset: dict[str, Any], *, dataset_sha256: str) -> dict[str, Any]:
    rows = [dict(row) for row in list(dataset.get("records") or []) if isinstance(row, dict)]
    invalid = 0
    failures: list[str] = []
    values: dict[str, list[str]] = {slot: [] for slot in SLOT_ORDER}
    context_targets: defaultdict[str, set[str]] = defaultdict(set)
    split_contexts: defaultdict[str, set[str]] = defaultdict(set)
    raw_hits = 0
    for row in rows:
        check = validate_pg331_source_row(row)
        if not check["valid"]:
            invalid += 1
        context = [str(token) for token in row.get("context_tokens") or []]
        target = _target(list(row.get("target_tokens") or []))
        raw_fragments = ("raw_", "response_body=", "route_literal=", "evaluator_", "http://", "https://")
        if any(any(fragment in token.casefold() for fragment in raw_fragments) for token in [*context, *target.keys(), *target.values()]):
            raw_hits += 1
        for slot in SLOT_ORDER:
            if slot not in target:
                failures.append(f"missing_target_slot:{slot}")
            else:
                values[slot].append(target[slot])
        context_targets[_sha(context)].add(_sha(target))
        split_contexts[str(row.get("split", "missing"))].add(_sha(context))
    if invalid:
        failures.append("invalid_rows")
    if raw_hits:
        failures.append("raw_context_or_target")
    conflicts = sum(1 for target_hashes in context_targets.values() if len(target_hashes) > 1)
    if conflicts:
        failures.append("context_target_conflict")
    train_overlap = len(split_contexts.get("train", set()) & split_contexts.get("implementation_holdout", set()))
    if train_overlap:
        failures.append("train_holdout_context_overlap")
    if sum(bool(row.get("training_eligible")) for row in rows):
        failures.append("unexpected_training_eligible_rows")
    failures.append("predictive_entropy_not_run")
    return {
        "schema_version": "pg361-payload-shape-slot-audit-v1",
        "status": "diagnostic_only" if not invalid and not raw_hits and not conflicts and not train_overlap else "blocked",
        "dataset_sha256": dataset_sha256,
        "counts": {
            "records": len(rows),
            "invalid_rows": invalid,
            "raw_hits": raw_hits,
            "context_target_conflicts": conflicts,
            "train_holdout_context_overlap": train_overlap,
            "training_eligible_rows": sum(bool(row.get("training_eligible")) for row in rows),
        },
        "slot_entropy": {slot: _entropy(values[slot]) for slot in SLOT_ORDER},
        "gates": {
            "raw_context_target_firewall": raw_hits == 0,
            "row_schema": invalid == 0,
            "context_target_conflict_free": conflicts == 0,
            "split_isolation": train_overlap == 0,
            "predictive_entropy_holdout": "not_run",
        },
        "failures": sorted(set(failures)),
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-361 abstract payload-shape slots")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    report = audit(dataset, dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "failures": report["failures"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
