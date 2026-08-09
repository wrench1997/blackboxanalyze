"""Audit PG-364 split integrity and information coverage.

This audit is intentionally stricter than the generic PG-362 structural
check: a compositional implementation holdout is only interpretable when all
held-out slot values were observed in the training split and source groups
are disjoint.  It still reports diagnostic-only; it never grants training or
promotion permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg362_full_rule_ir_dataset import RAW_FRAGMENTS, SLOTS, _sha


def audit(dataset: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    records = dataset.get("records") or []
    split_counts: Counter[str] = Counter()
    split_values: dict[str, dict[str, set[str]]] = {
        "train": {slot: set() for slot in SLOTS},
        "implementation_holdout": {slot: set() for slot in SLOTS},
    }
    seen: set[tuple[str, str]] = set()
    target_lengths: list[int] = []
    raw_hits = 0
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        split = str(row.get("split", ""))
        split_counts[split] += 1
        if split not in split_values:
            failures.append(f"row_{index}:split")
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:firewall")
        if any(row.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}:raw_flag")
        if target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.append(f"row_{index}:target_boundary")
        values: dict[str, str] = {}
        for token in target[1:-1]:
            if "=" not in token:
                failures.append(f"row_{index}:target_token")
                continue
            key, value = token.split("=", 1)
            if key not in SLOTS or key in values:
                failures.append(f"row_{index}:slot_coverage")
                continue
            values[key] = value
        if set(values) != set(SLOTS):
            failures.append(f"row_{index}:slot_coverage")
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context, *target]):
            raw_hits += 1
            failures.append(f"row_{index}:raw_token")
        key = (str(row.get("source_record_digest", "")), split)
        if key in seen:
            failures.append(f"row_{index}:duplicate_source")
        seen.add(key)
        if split in split_values:
            for slot in SLOTS:
                if slot in values:
                    split_values[split][slot].add(values[slot])
        target_lengths.append(len(context) + len(target))
    missing_by_slot = {
        slot: sorted(split_values["implementation_holdout"][slot] - split_values["train"][slot])
        for slot in SLOTS
    }
    for slot, missing in missing_by_slot.items():
        if missing:
            failures.append(f"target_coverage:{slot}:{','.join(missing)}")
    contract = dataset.get("split_contract") if isinstance(dataset.get("split_contract"), Mapping) else {}
    train_groups = {str(value) for value in contract.get("train_group_hashes") or []}
    holdout_groups = {str(value) for value in contract.get("holdout_group_hashes") or []}
    group_disjoint = not (train_groups & holdout_groups) and bool(train_groups) and bool(holdout_groups)
    if not group_disjoint:
        failures.append("implementation_group_overlap_or_missing")
    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    vocab_tokens = {str(token) for token in [*(vocab.get("context_tokens") or []), *(vocab.get("target_tokens") or [])]}
    missing_vocab = sorted({str(token) for row in records if isinstance(row, Mapping) for token in [*(row.get("context_tokens") or []), *(row.get("target_tokens") or [])]} - vocab_tokens)
    failures.extend(f"missing_vocab:{token}" for token in missing_vocab)
    failures = sorted(set(failures))
    split_target_cardinality = {
        split: {slot: len(values) for slot, values in fields.items()}
        for split, fields in split_values.items()
    }
    return {
        "schema_version": "pg364-compositional-rule-ir-audit-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "counts": {
            "records": len(records),
            "train_rows": split_counts.get("train", 0),
            "implementation_holdout_rows": split_counts.get("implementation_holdout", 0),
            "raw_hits": raw_hits,
            "missing_vocabulary_tokens": len(missing_vocab),
            "training_eligible_rows": 0,
            "unique_target_sequences": len({tuple(row.get("target_tokens") or []) for row in records if isinstance(row, Mapping)}),
        },
        "split_contract": {
            "group_disjoint": group_disjoint,
            "train_group_count": len(train_groups),
            "holdout_group_count": len(holdout_groups),
            "target_value_coverage": not any(missing_by_slot.values()),
            "target_value_cardinality": split_target_cardinality,
            "missing_target_values_by_slot": missing_by_slot,
        },
        "capacity": {"min_context_target_length": min(target_lengths or [0]), "max_context_target_length": max(target_lengths or [0]), "required_context_window": max(target_lengths or [0]), "truncation_risk": False},
        "slot_order": list(SLOTS),
        "failures": failures,
        "predictive_entropy": "not_run",
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-364 compositional Rule-IR dataset")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg364_compositional_rule_ir_dataset_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg364_compositional_rule_ir_audit_v1.json")
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "blocked_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
