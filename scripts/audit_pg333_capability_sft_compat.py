"""Audit PG-333's merged three-implementation rows for Rule-IR SFT use.

This is a read-only contract audit.  It deliberately does not manufacture
missing target slots, relabel the historical split, copy evaluator answers or
launch a trainer.  The current PG-333 artifact is expected to be blocked: it
is a diagnostic merge with a nine-slot target projection, a non-closed
train-only context vocabulary and a context window larger than the current
PG-370/PG-375 candidate defaults.

The report contains only bounded counts, slot names, hashes and gate states;
raw context/target token values, wire material and response bodies are never
written to the report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "pg333-capability-sft-compatibility-audit-v1"
REQUIRED_TARGET_SLOTS = (
    "question",
    "ask_reason",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "probe_variant_ref",
    "safe_to_send",
    "payload_shape_ref",
    "oracle_ref",
    "negative_control_presence_ref",
)
RAW_TEXT_FRAGMENTS = ("http://", "https://", "/webgoat", "response_body=", "raw_payload=")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _safe_counts(values: Sequence[Any]) -> dict[str, int]:
    """Count abstract scalar classes without emitting their values."""

    result = Counter()
    for value in values:
        if value is None:
            result["missing"] += 1
        elif isinstance(value, bool):
            result["boolean"] += 1
        elif isinstance(value, (int, float)):
            result["numeric"] += 1
        else:
            result["abstract_symbol"] += 1
    return dict(sorted(result.items()))


def _target_slot_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slots: dict[str, dict[str, Any]] = {}
    token_slot_names: set[str] = set()
    target_lengths = Counter()
    for row in rows:
        target = row.get("target_projection")
        target = target if isinstance(target, Mapping) else {}
        tokens = row.get("target_tokens")
        if isinstance(tokens, list):
            target_lengths[len(tokens)] += 1
            for token in tokens:
                key = str(token).split("=", 1)[0]
                if key not in {"[TARGET_BOS]", "[TARGET_EOS]"}:
                    token_slot_names.add(key)
        for slot in REQUIRED_TARGET_SLOTS:
            value = target.get(slot)
            entry = slots.setdefault(slot, {"rows": 0, "present": 0, "missing": 0, "value_kinds": Counter()})
            entry["rows"] += 1
            if value is None:
                entry["missing"] += 1
            else:
                entry["present"] += 1
                entry["value_kinds"].update(_safe_counts([value]).keys())
    normalized: dict[str, dict[str, Any]] = {}
    for slot, entry in slots.items():
        normalized[slot] = {
            "rows": int(entry["rows"]),
            "present": int(entry["present"]),
            "missing": int(entry["missing"]),
            "value_kinds": dict(sorted(entry["value_kinds"].items())),
            "exact_complete": bool(entry["rows"] and entry["missing"] == 0),
        }
    return {
        "required_slot_count": len(REQUIRED_TARGET_SLOTS),
        "required_slots": list(REQUIRED_TARGET_SLOTS),
        "slots": normalized,
        "missing_required_slots": [slot for slot in REQUIRED_TARGET_SLOTS if not normalized.get(slot, {}).get("exact_complete", False)],
        "target_token_length_counts": dict(sorted((str(key), int(value)) for key, value in target_lengths.items())),
        "target_token_slot_names": sorted(token_slot_names),
        "expected_target_token_length": len(REQUIRED_TARGET_SLOTS) + 2,
    }


def _split_audit(rows: Sequence[Mapping[str, Any]], source_audit: Mapping[str, Any], info_audit: Mapping[str, Any]) -> dict[str, Any]:
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    train = [row for row in rows if str(row.get("split")) == "train"]
    holdout = [row for row in rows if str(row.get("split")) == "implementation_holdout"]
    train_vocab = {str(token) for row in train for token in list(row.get("context_tokens") or [])}
    holdout_tokens = {str(token) for row in holdout for token in list(row.get("context_tokens") or [])}
    unknown_holdout = sorted(holdout_tokens - train_vocab)
    train_signatures = {tuple(str(token) for token in list(row.get("context_tokens") or [])) for row in train}
    holdout_signatures = {tuple(str(token) for token in list(row.get("context_tokens") or [])) for row in holdout}
    implementations = {
        "train": len({str((row.get("source_meta") or {}).get("implementation", "missing")) for row in train}),
        "holdout": len({str((row.get("source_meta") or {}).get("implementation", "missing")) for row in holdout}),
    }
    families = {
        "train": len({str((row.get("source_meta") or {}).get("family_id", "missing")) for row in train}),
        "holdout": len({str((row.get("source_meta") or {}).get("family_id", "missing")) for row in holdout}),
    }
    claimed_train = sum(int(row.get("training_eligible") is True) for row in train)
    claimed_all = sum(int(row.get("training_eligible") is True) for row in rows)
    accepted = int(dict(info_audit.get("validation") or {}).get("accepted_training_eligible_count", 0) or 0)
    source_split = dict(source_audit.get("split_isolation") or {})
    return {
        "split_counts": dict(sorted(split_counts.items())),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_only_context_vocab_size": len(train_vocab),
        "holdout_unknown_context_count": len(unknown_holdout),
        "holdout_unknown_context_sha256": _sha_json(unknown_holdout),
        "active_context_overlap_count": len(train_signatures & holdout_signatures),
        "implementation_group_counts": implementations,
        "family_group_counts": families,
        "source_audit_split_status": str(source_split.get("status", "unknown")),
        "source_audit_group_disjoint": bool(
            not source_split.get("implementation_cross_split_groups")
            and not source_split.get("family_cross_split_groups")
            and not source_split.get("source_cross_split_groups")
        ),
        "claimed_training_eligible_rows": claimed_all,
        "claimed_train_rows": claimed_train,
        "accepted_training_eligible_rows": accepted,
        "train_only_vocab_closed": not unknown_holdout,
    }


def _compatibility(
    *,
    dataset: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    info_audit: Mapping[str, Any],
    pg370: Mapping[str, Any],
    pg375_plan: Mapping[str, Any],
    pg375_context: Mapping[str, Any],
    required_window: int,
) -> dict[str, Any]:
    pg370_training = dict(pg370.get("training") or {})
    pg370_config = dict(pg370_training.get("config") or {})
    pg370_max_length = int(pg370_config.get("max_length", 0) or 0)
    pg375_training = dict(pg375_context.get("training") or {})
    pg375_smoke_window = int(pg375_training.get("required_context_window", 0) or 0)
    reasons: list[str] = []
    if dataset.get("diagnostic_only") is True:
        reasons.append("dataset_diagnostic_only")
    if str(source_audit.get("status")) != "passed":
        reasons.append("source_audit_not_passed")
    if bool(dict(source_audit.get("promotion") or {}).get("training_allowed")) is not True:
        reasons.append("source_audit_training_closed")
    if str(info_audit.get("status")) != "passed":
        reasons.append("information_audit_not_passed")
    if str(dict(info_audit.get("vocabulary_coverage") or {}).get("manifest_status")) != "passed":
        reasons.append("vocabulary_manifest_diagnostic_or_blocked")
    if pg370_max_length < required_window:
        reasons.append("pg370_context_capacity_below_required_window")
    if pg375_smoke_window and pg375_smoke_window < required_window:
        reasons.append("pg375_context_smoke_window_below_required_window")
    if str(pg375_plan.get("status", "")) == "blocked_data_contract":
        reasons.append("pg375_plan_blocked_data_contract")
    return {
        "candidate_runner_compatible": not reasons,
        "blocked_reasons": reasons,
        "pg370": {"status": str(pg370.get("status", "missing")), "max_length": pg370_max_length, "required_window": required_window, "capacity_pass": bool(pg370_max_length >= required_window)},
        "pg375": {"plan_status": str(pg375_plan.get("status", "missing")), "context_smoke_status": str(pg375_context.get("status", "missing")), "context_smoke_window": pg375_smoke_window, "required_window": required_window, "capacity_pass": bool(pg375_smoke_window >= required_window) if pg375_smoke_window else False},
        "remote_candidate_allowed": False,
        "remote_command": None,
    }


def audit_pg333_capability_sft_compatibility(
    *,
    dataset_path: Path = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json",
    source_audit_path: Path = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_source_audit_v1.json",
    information_audit_path: Path = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_information_audit_v1.json",
    vocabulary_path: Path = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_vocabulary_v1.json",
    capacity_path: Path = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_capacity_v1.json",
    pg370_path: Path = ROOT / "research" / "pg370_multitask_moe_candidate_v1.json",
    pg375_plan_path: Path = ROOT / "research" / "pg375_strict_candidate_plan_v1.json",
    pg375_context_path: Path = ROOT / "research" / "pg375_context_representation_cpu_smoke_v1.json",
) -> dict[str, Any]:
    paths = {"dataset": dataset_path, "source_audit": source_audit_path, "information_audit": information_audit_path, "vocabulary": vocabulary_path, "capacity": capacity_path, "pg370": pg370_path, "pg375_plan": pg375_plan_path, "pg375_context": pg375_context_path}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_missing_artifact",
            "missing_artifacts": missing,
            "promotion": {key: False for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")},
            "candidate_runner": {"compatible": False, "remote_candidate_allowed": False, "remote_command": None},
        }
        report["report_sha256"] = _sha_json(report)
        return report
    dataset = _load(dataset_path)
    source_audit = _load(source_audit_path)
    info_audit = _load(information_audit_path)
    vocabulary = _load(vocabulary_path)
    capacity = _load(capacity_path)
    pg370 = _load(pg370_path)
    pg375_plan = _load(pg375_plan_path)
    pg375_context = _load(pg375_context_path)
    records = dataset.get("records")
    rows = [row for row in records if isinstance(row, Mapping)] if isinstance(records, list) else []
    target = _target_slot_audit(rows)
    split = _split_audit(rows, source_audit, info_audit)
    capacity_coverage = dict(info_audit.get("capacity_coverage") or {})
    required_window = int(capacity_coverage.get("required_context_window", 0) or 0)
    compatibility = _compatibility(dataset=dataset, source_audit=source_audit, info_audit=info_audit, pg370=pg370, pg375_plan=pg375_plan, pg375_context=pg375_context, required_window=required_window)
    blocked = list(compatibility["blocked_reasons"])
    if target["missing_required_slots"]:
        blocked.append("target_schema_missing_required_slots")
    if not split["train_only_vocab_closed"]:
        blocked.append("holdout_unknown_context_under_train_only_vocab")
    if int(split["accepted_training_eligible_rows"]) == 0:
        blocked.append("accepted_training_rows_zero")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_not_capability_sft_compatible",
        "scope": {"dataset_schema": str(dataset.get("schema_version", "")), "diagnostic_only": bool(dataset.get("diagnostic_only")), "raw_material_reported": False, "split_relabelled": False, "fields_fabricated": False},
        "artifacts": {name: {"sha256": _sha(path), "schema_version": str((_load(path)).get("schema_version", ""))} for name, path in paths.items()},
        "dataset": {"record_count": len(rows), "source_counts": dict(dataset.get("counts") or {}), "promotion": dict(dataset.get("promotion") or {})},
        "target_schema": target,
        "train_only_holdout": split,
        "source_audit": {"status": str(source_audit.get("status", "")), "promotion_training_allowed": bool(dict(source_audit.get("promotion") or {}).get("training_allowed")), "axis_presence_complete": bool(all(int(value) == len(rows) for value in dict(source_audit.get("axis_presence_counts") or {}).values())), "split_isolation": dict(source_audit.get("split_isolation") or {})},
        "information_audit": {"status": str(info_audit.get("status", "")), "manifest_status": str(dict(info_audit.get("vocabulary_coverage") or {}).get("manifest_status", "")), "context_target_alignment": dict(info_audit.get("context_target_alignment") or {}), "capacity_coverage": {"required_context_window": required_window, "dataset_context_length_max": int(capacity_coverage.get("dataset_context_length_max", 0) or 0), "capacity_report_status": str(capacity_coverage.get("capacity_report_status", ""))}},
        "compatibility": compatibility,
        "blocked_reasons": sorted(set(blocked)),
        "minimum_adaptation_plan": [
            "保留原始 split；重新采集或严格派生完整 13-slot target_projection，不为旧行补写缺失答案。",
            "为 train-only vocab 单独构建上下文词表；holdout 未知 token 必须 quarantine/hash-only，不得映射成已知标签。",
            "补齐 ask_reason、syntax_category_ref、payload_shape_ref、oracle_ref、negative_control_presence_ref，以及真实 failure→repair/ASK 轨迹。",
            "为完整网页序列提供 required_context_window 以上的显式容量；禁止静默截断，先做 next-token baseline 与信息熵审计。",
            "完成 source/implementation/family holdout、正负/复放/fresh/evidence 合同后，才申请远端 A800 GPU0 candidate。",
        ],
        "candidate_runner": {"compatible": False, "implementation_status": "not_created_data_contract_blocked", "remote_candidate_allowed": False, "remote_command": None},
        "promotion": {key: False for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")},
        "interpretation": "PG-333 merged rows are valuable diagnostic evidence and a schema-repair reference, but cannot be used as capability Rule-IR SFT/A800 data without relabelling or fabricating fields.",
    }
    report["report_sha256"] = _sha_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg333_capability_sft_compatibility_audit_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_pg333_capability_sft_compatibility()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "blocked_reasons": report.get("blocked_reasons", []), "report_sha256": report["report_sha256"]}
    print(json.dumps(report if args.json else summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUIRED_TARGET_SLOTS", "SCHEMA_VERSION", "audit_pg333_capability_sft_compatibility"]
