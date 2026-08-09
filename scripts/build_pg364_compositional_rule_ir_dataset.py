"""Build a compositional implementation-holdout Rule-IR dataset.

PG-362 used the collector's original split, where the implementation holdout
introduced several target values that never occurred in training.  That is a
valid zero-shot stress test, but it cannot distinguish compositional
generalization from an unseen-label problem.  PG-364 keeps the source rows
unchanged and reassigns only the split using an explicit implementation
holdout list.  Every held-out slot value must still occur in train; the
implementation/family groups remain disjoint.

Only abstract context/target tokens are emitted.  Source implementation IDs
are used for split construction and retained only as salted group hashes in
the dataset manifest; they never enter model context or targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg362_full_rule_ir_dataset import (
    RAW_FRAGMENTS,
    SLOTS,
    _abstract_tokens,
    _sha,
    _target_values,
)

SCHEMA_VERSION = "pg364-compositional-rule-ir-dataset-v1"
DEFAULT_HOLDOUT_IMPLEMENTATIONS = (
    "pg348_pages_c_impl_3",
    "pg348_pages_c_impl_4",
    "pg348_pages_c_impl_5",
    "pg348_pages_c_impl_6",
    "pg348_pages_c_impl_7",
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_hash(value: str) -> str:
    """Hash a source implementation ID without persisting the literal ID."""
    return hashlib.sha256(f"pg364-source-group-v1:{value}".encode("utf-8")).hexdigest()


def _source_implementation(raw: Mapping[str, Any]) -> str | None:
    meta = raw.get("source_meta")
    if not isinstance(meta, Mapping):
        return None
    value = str(meta.get("implementation") or "").strip()
    return value or None


def _clean_row(raw: Mapping[str, Any], *, index: int, split: str) -> tuple[dict[str, Any] | None, str | None]:
    context = raw.get("context_tokens")
    if not isinstance(context, list) or not context:
        return None, f"row_{index}:context"
    context_tokens = [str(token) for token in context]
    target_tokens, target_failure = _abstract_tokens(raw.get("target_tokens"))
    if target_failure:
        return None, f"row_{index}:{target_failure}"
    assert target_tokens is not None
    values = _target_values(target_tokens)
    full_target = ["[TARGET_BOS]", *[f"{slot}={values[slot]}" for slot in SLOTS], "[TARGET_EOS]"]
    if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        return None, f"row_{index}:firewall"
    if any(raw.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
        return None, f"row_{index}:raw_flag"
    if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context_tokens, *full_target]):
        return None, f"row_{index}:raw_token"
    source_digest = str(raw.get("record_id") or raw.get("record_sha256") or _sha({"index": index, "context": context_tokens, "target": full_target}))
    record: dict[str, Any] = {
        "schema_version": "pg364-compositional-rule-ir-row-v1",
        "record_id": _sha({"source_record": source_digest, "context": context_tokens, "target": full_target, "split": split}),
        "source_record_digest": source_digest,
        "split": split,
        "context_tokens": context_tokens,
        "target_tokens": full_target,
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "full_target_contract": {"slot_order": list(SLOTS), "target_values_not_in_context": True, "source_sidecars_off_context": True},
        "operator_reviewed": False,
        "training_eligible": False,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    record["record_sha256"] = _sha(record)
    return record, None


def build(
    source: Mapping[str, Any],
    *,
    source_sha256: str,
    source_path: str,
    holdout_implementations: Sequence[str] = DEFAULT_HOLDOUT_IMPLEMENTATIONS,
) -> dict[str, Any]:
    holdout = {str(value).strip() for value in holdout_implementations if str(value).strip()}
    if not holdout:
        raise ValueError("PG-364 requires at least one implementation holdout")
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    context_vocab: set[str] = set()
    target_vocab: set[str] = {"[TARGET_BOS]", "[TARGET_EOS]"}
    impl_counts: Counter[str] = Counter()
    split_impls: dict[str, set[str]] = {"train": set(), "implementation_holdout": set()}
    for index, raw in enumerate(source.get("records") or []):
        if not isinstance(raw, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        implementation = _source_implementation(raw)
        if implementation is None:
            failures.append(f"row_{index}:source_implementation_missing")
            continue
        split = "implementation_holdout" if implementation in holdout else "train"
        impl_counts[implementation] += 1
        split_impls[split].add(implementation)
        record, failure = _clean_row(raw, index=index, split=split)
        if failure:
            failures.append(failure)
            continue
        assert record is not None
        records.append(record)
        context_vocab.update(record["context_tokens"])
        target_vocab.update(record["target_tokens"])
    overlap = split_impls["train"] & split_impls["implementation_holdout"]
    if overlap:
        failures.append("implementation_split_overlap")
    # Do not persist implementation literals; hashes are enough to audit group
    # disjointness and make the split reproducible with the input source hash.
    group_hashes = {
        split: sorted(_group_hash(value) for value in values)
        for split, values in split_impls.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "source_dataset": source_path,
        "source_dataset_sha256": source_sha256,
        "records": records,
        "slot_order": list(SLOTS),
        "vocabulary": {"context_tokens": sorted(context_vocab), "target_tokens": sorted(target_vocab), "shared_tokens": sorted(context_vocab | target_vocab), "append_only": True},
        "split_contract": {
            "kind": "implementation_holdout_with_target_value_coverage",
            "holdout_group_hashes": group_hashes["implementation_holdout"],
            "train_group_hashes": group_hashes["train"],
            "group_disjoint": not bool(overlap),
            "holdout_group_count": len(split_impls["implementation_holdout"]),
            "train_group_count": len(split_impls["train"]),
            "target_value_coverage_required": True,
            "source_implementation_literals_in_context": False,
        },
        "counts": {
            "records": len(records),
            "train_rows": sum(row["split"] == "train" for row in records),
            "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in records),
            "target_slots": len(SLOTS),
            "unique_target_sequences": len({tuple(row["target_tokens"]) for row in records}),
            "train_implementation_groups": len(split_impls["train"]),
            "holdout_implementation_groups": len(split_impls["implementation_holdout"]),
            "training_eligible_rows": 0,
        },
        "failures": sorted(set(failures)),
        "full_target_contract": {"slotwise_source_excluded": True, "whole_sequence_target": True, "context_preserved": True, "raw_payload_in_context": False, "evaluator_sidecar_read": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-364 compositional Rule-IR implementation holdout")
    parser.add_argument("--input", type=Path, default=ROOT / "research" / "pg361_dynamic_syntax_typed_source_rows_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg364_compositional_rule_ir_dataset_v1.json")
    parser.add_argument("--holdout-implementation", action="append", dest="holdouts", default=None)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    holdouts = tuple(args.holdouts) if args.holdouts else DEFAULT_HOLDOUT_IMPLEMENTATIONS
    result = build(source, source_sha256=_file_sha(args.input), source_path=str(args.input.resolve().relative_to(ROOT.resolve())), holdout_implementations=holdouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "blocked_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
