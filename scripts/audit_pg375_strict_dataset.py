"""Audit the PG-375 filtered abstract dataset without running a model.

The audit recomputes the train-only vocabulary, slot classes and cross-split
overlap from active rows.  Quarantine/exclusion records are required to be
hash-only references so the repair process cannot leak the excluded page or
wire into a future model context.  A clean audit is still candidate evidence;
all promotion flags remain false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK
from scripts.build_pg362_full_rule_ir_dataset import RAW_FRAGMENTS, SLOTS
from scripts.run_pg370_multitask_moe_candidate import _target_values

SCHEMA_VERSION = "pg375-strict-filtered-rule-ir-audit-v1"
DEFAULT_DATASET = ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg375_strict_filtered_rule_ir_audit_v1.json"


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _context(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("context_tokens", []))


def _target(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("target_tokens", []))


def audit(dataset: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    records = dataset.get("records") or []
    train = [row for row in records if isinstance(row, Mapping) and str(row.get("split", "")) == "train"]
    holdout = [row for row in records if isinstance(row, Mapping) and str(row.get("split", "")) == "implementation_holdout"]
    malformed = 0
    raw_hits = 0
    train_tokens: set[str] = {PAD, UNK}
    slot_values: dict[str, set[str]] = {slot: set() for slot in SLOTS}
    for row in records:
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        if not context or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            malformed += 1
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            malformed += 1
        if any(row.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            malformed += 1
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context, *target]):
            raw_hits += 1
        try:
            values = _target_values(target)
            if set(values) != set(SLOTS):
                malformed += 1
            if str(row.get("split", "")) == "train":
                train_tokens.update([*context, *target])
                for slot in SLOTS:
                    slot_values[slot].add(str(values[slot]))
        except (KeyError, TypeError, ValueError):
            malformed += 1
    if malformed:
        failures.append("malformed_rows")
    if raw_hits:
        failures.append("raw_material_or_firewall_violation")
    if not train or not holdout:
        failures.append("empty_active_split")
    unknown_context = sorted({token for row in holdout for token in _context(row) if token not in train_tokens})
    unknown_target = sorted({token for row in holdout for token in _target(row) if token not in train_tokens})
    unknown_slots: dict[str, list[str]] = {}
    for slot in SLOTS:
        values = sorted({str((_target_values(row["target_tokens"]))[slot]) for row in holdout if str((_target_values(row["target_tokens"]))[slot]) not in slot_values[slot]})
        if values:
            unknown_slots[slot] = values
    if unknown_context or unknown_target or unknown_slots:
        failures.append("train_only_vocabulary_gap")
    train_context = {_context(row) for row in train}
    holdout_context = {_context(row) for row in holdout}
    train_full = {(_context(row), _target(row)) for row in train}
    holdout_full = {(_context(row), _target(row)) for row in holdout}
    context_overlap = len(train_context & holdout_context)
    exact_overlap = len(train_full & holdout_full)
    if context_overlap or exact_overlap:
        failures.append("active_cross_split_overlap")
    split_contract = dataset.get("split_contract") if isinstance(dataset.get("split_contract"), Mapping) else {}
    train_groups = {str(value) for value in split_contract.get("train_group_hashes", [])}
    holdout_groups = {str(value) for value in split_contract.get("holdout_group_hashes", [])}
    group_disjoint = bool(train_groups and holdout_groups and not (train_groups & holdout_groups))
    if not group_disjoint:
        failures.append("implementation_group_overlap_or_missing")
    quarantine = dataset.get("quarantine") or []
    excluded = dataset.get("excluded") or []
    if not isinstance(quarantine, list) or not isinstance(excluded, list):
        failures.append("quarantine_or_excluded_not_list")
        quarantine = []
        excluded = []
    # Quarantine/exclusion entries must not carry context/target/wire fields.
    forbidden_ref_fields = {"context_tokens", "target_tokens", "payload", "wire", "response_body", "url", "evaluator_answer"}
    ref_violations = 0
    for item in [*quarantine, *excluded]:
        if not isinstance(item, Mapping) or not str(item.get("record_ref_sha256", "")):
            ref_violations += 1
            continue
        if forbidden_ref_fields & set(item):
            ref_violations += 1
    if ref_violations:
        failures.append("quarantine_raw_reference_leak")
    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    declared_context = {str(token) for token in vocab.get("context_tokens", [])}
    declared_target = {str(token) for token in vocab.get("target_tokens", [])}
    manifest_mismatch = sorted((train_tokens - {PAD, UNK}) - (declared_context | declared_target)) if (declared_context or declared_target) else []
    if manifest_mismatch:
        failures.append("train_token_missing_from_manifest")
    # A structurally clean abstract split is not the same thing as a
    # capability-training authorization.  The derived PG-375 artifact keeps
    # typed/fresh/operator gates explicitly closed; audit that distinction so
    # downstream runners cannot infer permission from row counts.
    if dataset.get("capability_training_allowed") is not False:
        failures.append("capability_training_flag_not_fail_closed")
    source_contract = dataset.get("source_contract") if isinstance(dataset.get("source_contract"), Mapping) else {}
    if any(source_contract.get(key) is not False for key in ("operator_reviewed", "typed_evaluator_complete", "fresh_reset_role_attested", "capability_training_eligible")):
        failures.append("source_capability_gate_not_closed")
    failures = sorted(set(failures))
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_candidate_audit" if not failures else "blocked_incomplete",
        "counts": {
            "active_records": len(records),
            "train_rows": len(train),
            "active_holdout_rows": len(holdout),
            "quarantined_holdout_rows": len(quarantine),
            "excluded_train_rows": len(excluded),
            "raw_hits": raw_hits,
            "malformed_rows": malformed,
            "unknown_context_tokens": len(unknown_context),
            "unknown_target_tokens": len(unknown_target),
            "unknown_slot_values": sum(len(values) for values in unknown_slots.values()),
            "active_cross_split_context_overlap": context_overlap,
            "active_cross_split_exact_overlap": exact_overlap,
        },
        "train_only_contract": {
            "vocabulary_scope": "filtered_train_only",
            "vocabulary_size": len(train_tokens),
            "unknown_context_sha256": _sha_json(unknown_context),
            "unknown_target_sha256": _sha_json(unknown_target),
            "unknown_slot_values_sha256": {slot: _sha_json(values) for slot, values in sorted(unknown_slots.items())},
            "blocked": bool(unknown_context or unknown_target or unknown_slots),
        },
        "split_contract": {"group_disjoint": group_disjoint, "holdout_precedence": bool(split_contract.get("holdout_precedence")), "active_context_overlap": context_overlap, "active_exact_overlap": exact_overlap},
        "quarantine_contract": {"hash_only_refs": ref_violations == 0, "rows_not_relabelled": True, "reasons_preserved": all(isinstance(item, Mapping) and isinstance(item.get("reasons"), list) for item in quarantine)},
        "capability_training_allowed": False,
        "representation_pretrain_candidate_allowed": bool(dataset.get("representation_pretrain_candidate_allowed")) and not bool(failures),
        "coverage": dataset.get("coverage") if isinstance(dataset.get("coverage"), Mapping) else {},
        "failures": failures,
        "promotion": promotion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-375 strict filtered Rule-IR dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    result["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "counts": result["counts"], "failures": result["failures"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] == "passed_candidate_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
