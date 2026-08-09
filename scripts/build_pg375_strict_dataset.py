"""Build the PG-375 strict, filtered abstract dataset.

PG-364 intentionally exposed implementation-holdout leakage and unseen
categories as a diagnostic.  This builder does not repair those rows by
renaming them or by copying holdout labels into train.  It applies two
explicit, auditable operations instead:

* holdout precedence removes train rows whose abstract context or complete
  context+target sequence is also present in holdout; and
* holdout rows containing a token/slot value absent from the resulting
  train-only coordinate system are quarantined.

Active rows remain abstract Rule-IR only.  Excluded/quarantined entries keep
hash references and reason counts, never raw contexts, payloads, responses,
URLs or evaluator answers.  The output is still candidate-only; this tool
does not grant model, memory, payload-catalog or vulnerability promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK
from scripts.build_pg362_full_rule_ir_dataset import RAW_FRAGMENTS, SLOTS
from scripts.run_pg370_multitask_moe_candidate import _safe_abstract_row, _sha_file, _sha_json, _target_values

SCHEMA_VERSION = "pg375-strict-filtered-rule-ir-dataset-v1"
DEFAULT_SOURCE = ROOT / "research" / "pg364_compositional_rule_ir_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json"


def _path_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _record_ref(raw: Mapping[str, Any], index: int) -> str:
    value = str(raw.get("record_id") or raw.get("source_record_digest") or "")
    if not value:
        value = _sha_json({"index": index, "context": raw.get("context_tokens", []), "target": raw.get("target_tokens", [])})
    return hashlib.sha256(f"pg375-record-ref-v1:{value}".encode("utf-8")).hexdigest()


def _context(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("context_tokens", []))


def _target(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("target_tokens", []))


def _full(row: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return _context(row), _target(row)


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slot_values: dict[str, set[str]] = {slot: set() for slot in SLOTS}
    syntax_payload = Counter()
    target_sequences = Counter()
    token_counts: Counter[str] = Counter()
    context_sequences = set()
    for row in rows:
        target_sequences["\x1f".join(_target(row))] += 1
        context_sequences.add(_context(row))
        token_counts.update([*_context(row), *_target(row)])
        values = row.get("_target_values") or _target_values(row["target_tokens"])
        for slot in SLOTS:
            slot_values[slot].add(str(values[slot]))
        syntax_payload[(str(values["syntax_category_ref"]), str(values["payload_shape_ref"]))] += 1
    return {
        "rows": len(rows),
        "unique_context_sequences": len(context_sequences),
        "unique_context_ratio": round(len(context_sequences) / max(len(rows), 1), 6),
        "unique_target_sequences": len(target_sequences),
        "token_entropy_bits": round(_entropy(token_counts), 6),
        "target_sequence_entropy_bits": round(_entropy(target_sequences), 6),
        "slot_cardinality": {slot: len(values) for slot, values in slot_values.items()},
        "syntax_payload_cells": len(syntax_payload),
        "syntax_payload_min_cell": min(syntax_payload.values()) if syntax_payload else 0,
    }


def _abstract_record(raw: Mapping[str, Any], *, index: int, split: str) -> dict[str, Any]:
    # _safe_abstract_row validates the source firewall and returns only model
    # visible fields; the hash is retained solely for audit references.
    item = _safe_abstract_row(raw, source="pg364")
    return {
        "schema_version": "pg375-strict-rule-ir-row-v1",
        "record_ref_sha256": _record_ref(raw, index),
        "split": split,
        "context_tokens": list(item["context_tokens"]),
        "target_tokens": list(item["target_tokens"]),
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "full_target_contract": {"slot_order": list(SLOTS), "target_values_not_in_context": True, "source_sidecars_off_context": True},
        "source_record_sidecar": False,
        "operator_reviewed": bool(raw.get("operator_reviewed", False)),
        "training_eligible": False,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def _quarantine_ref(row: Mapping[str, Any], *, reasons: Sequence[str], unknown_tokens: Sequence[str], unknown_slots: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    unknown_token_list = sorted(set(str(token) for token in unknown_tokens))
    unknown_slot_hashes = {slot: _sha_json(sorted(set(str(value) for value in values))) for slot, values in sorted(unknown_slots.items())}
    return {
        "record_ref_sha256": str(row["record_ref_sha256"]),
        "original_split": "implementation_holdout",
        "reasons": sorted(set(str(reason) for reason in reasons)),
        "unknown_token_count": len(unknown_token_list),
        "unknown_tokens_sha256": _sha_json(unknown_token_list),
        "unknown_slot_value_count": sum(len(set(values)) for values in unknown_slots.values()),
        "unknown_slot_values_sha256": unknown_slot_hashes,
    }


def build(source: Mapping[str, Any], *, source_sha256: str, source_path: str) -> dict[str, Any]:
    raw_records = source.get("records") or []
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        split = str(raw.get("split", ""))
        if split not in {"train", "implementation_holdout"}:
            failures.append(f"row_{index}:split")
            continue
        try:
            row = _abstract_record(raw, index=index, split=split)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"row_{index}:abstract:{type(exc).__name__}")
            continue
        (train if split == "train" else holdout).append(row)

    holdout_full = {_full(row) for row in holdout}
    holdout_context = {_context(row) for row in holdout}
    filtered_train: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in train:
        reasons: list[str] = []
        if _full(row) in holdout_full:
            reasons.append("cross_split_exact")
        if _context(row) in holdout_context:
            reasons.append("cross_split_context")
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            excluded.append({"record_ref_sha256": row["record_ref_sha256"], "original_split": "train", "reasons": sorted(reasons)})
        else:
            filtered_train.append(row)

    train_tokens = {str(token) for row in filtered_train for token in [*row["context_tokens"], *row["target_tokens"]]}
    slot_values: dict[str, set[str]] = {slot: set() for slot in SLOTS}
    for row in filtered_train:
        values = _target_values(row["target_tokens"])
        for slot in SLOTS:
            slot_values[slot].add(str(values[slot]))

    active_holdout: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in holdout:
        unknown_tokens = sorted({str(token) for token in [*row["context_tokens"], *row["target_tokens"]] if str(token) not in train_tokens})
        values = _target_values(row["target_tokens"])
        unknown_slots = {slot: [str(values[slot])] for slot in SLOTS if str(values[slot]) not in slot_values[slot]}
        reasons: list[str] = []
        if unknown_tokens:
            reasons.append("holdout_unknown_token")
        if unknown_slots:
            reasons.append("holdout_unknown_slot_value")
        if reasons:
            quarantine.append(_quarantine_ref(row, reasons=reasons, unknown_tokens=unknown_tokens, unknown_slots=unknown_slots))
        else:
            active_holdout.append(row)

    context_vocab = sorted({str(token) for row in filtered_train for token in row["context_tokens"]})
    target_vocab = sorted({str(token) for row in filtered_train for token in row["target_tokens"]})
    train_groups = [str(value) for value in (source.get("split_contract") or {}).get("train_group_hashes", [])] if isinstance(source.get("split_contract"), Mapping) else []
    holdout_groups = [str(value) for value in (source.get("split_contract") or {}).get("holdout_group_hashes", [])] if isinstance(source.get("split_contract"), Mapping) else []
    group_disjoint = bool(train_groups and holdout_groups and not (set(train_groups) & set(holdout_groups)))
    active_full_overlap = len({_full(row) for row in filtered_train} & {_full(row) for row in active_holdout})
    active_context_overlap = len({_context(row) for row in filtered_train} & {_context(row) for row in active_holdout})
    strict_failures = list(failures)
    if not filtered_train:
        strict_failures.append("empty_filtered_train")
    if not active_holdout:
        strict_failures.append("empty_filtered_holdout")
    if active_full_overlap or active_context_overlap:
        strict_failures.append("active_cross_split_overlap")
    if not group_disjoint:
        strict_failures.append("implementation_group_overlap_or_missing")
    missing_slots = [slot for slot in SLOTS if not slot_values[slot]]
    if missing_slots:
        strict_failures.append("train_slot_coverage_missing")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_only" if not strict_failures else "blocked_incomplete",
        "source_dataset": source_path,
        "source_dataset_sha256": source_sha256,
        "records": [*filtered_train, *active_holdout],
        "quarantine": quarantine,
        "excluded": excluded,
        "slot_order": list(SLOTS),
        "vocabulary": {"context_tokens": context_vocab, "target_tokens": target_vocab, "shared_tokens": sorted(set(context_vocab) | set(target_vocab)), "append_only": True, "scope": "filtered_train_only"},
        "split_contract": {
            "kind": "holdout_precedence_context_exact_quarantine",
            "train_group_hashes": sorted(set(train_groups)),
            "holdout_group_hashes": sorted(set(holdout_groups)),
            "group_disjoint": group_disjoint,
            "holdout_precedence": True,
            "context_duplicates_removed_from_train": True,
            "exact_duplicates_removed_from_train": True,
            "unknown_holdout_rows_quarantined": True,
            "active_cross_split_exact_overlap": active_full_overlap,
            "active_cross_split_context_overlap": active_context_overlap,
            "source_implementation_literals_in_context": False,
        },
        "counts": {
            "source_records": len(raw_records),
            "source_train_rows": len(train),
            "source_holdout_rows": len(holdout),
            "filtered_train_rows": len(filtered_train),
            "active_holdout_rows": len(active_holdout),
            "excluded_train_rows": len(excluded),
            "quarantined_holdout_rows": len(quarantine),
            # These rows are structurally valid abstract candidates, but the
            # source dataset is diagnostic-only and lacks operator-reviewed
            # typed evaluator sidecars.  Keep capability training eligibility
            # separate so a clean split cannot be mistaken for authorization.
            "abstract_candidate_rows": len(filtered_train) if not strict_failures else 0,
            "training_eligible_rows": 0,
            "quarantine_reason_counts": dict(sorted(Counter(reason for item in quarantine for reason in item["reasons"]).items())),
            "excluded_reason_counts": dict(sorted(reason_counts.items())),
        },
        "coverage": {"source": _coverage([*train, *holdout]), "filtered_train": _coverage(filtered_train), "active_holdout": _coverage(active_holdout)},
        "train_only_gap": {"unknown_context_count": 0, "unknown_target_count": 0, "unknown_slot_value_count": 0, "quarantined_rows_not_relabelled": True},
        "failures": sorted(set(strict_failures)),
        "capability_training_allowed": False,
        "representation_pretrain_candidate_allowed": not bool(strict_failures),
        "source_contract": {
            "source_status": str(source.get("status", "")),
            "operator_reviewed": False,
            "typed_evaluator_complete": False,
            "fresh_reset_role_attested": False,
            "capability_training_eligible": False,
        },
        "raw_material_available": False,
        "full_target_contract": {"slotwise_source_excluded": True, "whole_sequence_target": True, "context_preserved": True, "raw_payload_in_context": False, "evaluator_sidecar_read": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-375 strict filtered Rule-IR dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build(source, source_sha256=_sha_file(args.input), source_path=_path_string(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "counts": result["counts"], "failures": result["failures"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] == "candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
