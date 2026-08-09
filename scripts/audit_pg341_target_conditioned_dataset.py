"""Audit PG-341 target-conditioned views without exposing row contents.

The audit is intentionally stricter than a next-token loss report.  It checks
that full-axis context is retained, that target supervision is actually in the
training split, and that ASK/repair/negative targets are not silently
inferred from a holdout.  It never changes a split and never marks a row as
promotion eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg341_target_conditioned_process_full_axis_dataset_v1.json"
SCHEMA_VERSION = "pg341-target-conditioned-audit-v1"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_replay")
FORBIDDEN_FRAGMENTS = ("payload=", "payload_", "response_body=", "response_body_text=", "raw_", "oracle=", "evaluator=", "family=", "route=", "route_literal=", "implementation=", "image=", "url=", "path=", "source=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def _target_map(tokens: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" in token:
            key, value = token.split("=", 1)
            if key in {"question", "next_action", "repair_action", "action_changed", "safe_to_send", "transport_ref", "field_role_ref", "encoding_ref", "probe_variant_ref"}:
                result[key] = value
    return result


def _view_stats(rows: Sequence[Mapping[str, Any]], view: str, split: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row.get("view") == view and (split is None or row.get("split") == split)]
    targets = [_target_map(row.get("target_tokens") or []) for row in selected]
    questions = Counter(item.get("question", "missing") for item in targets)
    actions = Counter(item.get("next_action", "missing") for item in targets)
    repairs = Counter(item.get("repair_action", "missing") for item in targets)
    negatives = [item for item in targets if item.get("safe_to_send") == "0"]
    return {
        "records": len(selected),
        "unique_context_sequences": len({tuple(str(token) for token in row.get("context_tokens") or []) for row in selected}),
        "unique_target_sequences": len({tuple(str(token) for token in row.get("target_tokens") or []) for row in selected}),
        "context_token_entropy_bits": _entropy([str(token) for row in selected for token in row.get("context_tokens") or []]),
        "target_token_entropy_bits": _entropy([str(token) for row in selected for token in row.get("target_tokens") or []]),
        "question_counts": dict(sorted(questions.items())),
        "next_action_counts": dict(sorted(actions.items())),
        "repair_action_counts": dict(sorted(repairs.items())),
        "negative_target_count": len(negatives),
        "ask_target_count": sum(int(item.get("question", "none") != "none") for item in targets),
        "repair_target_count": sum(int(item.get("next_action") in {"repair", "repair_abstract_plan"}) for item in targets),
        "abstain_target_count": sum(int(item.get("next_action") == "abstain") for item in targets),
    }


def _audit_rows(rows: Any) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    reasons: Counter[str] = Counter()
    if not isinstance(rows, list) or not rows:
        return ["records_missing_or_empty"], {"records_missing_or_empty": 1}
    seen_ids: set[str] = set()
    seen_source: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        prefix = f"row_{index}"
        if not isinstance(row, Mapping):
            failures.append(f"{prefix}:not_mapping"); reasons["not_mapping"] += 1; continue
        view = str(row.get("view", ""))
        if view not in {"coarse_process", "full_axis"}:
            failures.append(f"{prefix}:unknown_view"); reasons["unknown_view"] += 1
        if str(row.get("record_id", "")) in seen_ids:
            failures.append(f"{prefix}:duplicate_record_id"); reasons["duplicate_record_id"] += 1
        seen_ids.add(str(row.get("record_id", "")))
        source_key = (view, str(row.get("source_record_sha256", "")))
        if source_key in seen_source:
            # A source observation may legitimately have several role/step
            # targets.  Preserve those rows and report the multiplicity; do
            # not turn it into a silent data drop or a hard structural fail.
            reasons["duplicate_source_ref"] += 1
        seen_source.add(source_key)
        context = row.get("context_tokens")
        target = row.get("target_tokens")
        if not isinstance(context, list) or len(context) < 2:
            failures.append(f"{prefix}:context_missing"); reasons["context_missing"] += 1
        if not isinstance(target, list) or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.append(f"{prefix}:target_boundary"); reasons["target_boundary"] += 1
        for token in [*(context or []), *(target or [])]:
            token_value = str(token).casefold()
            if any(fragment in token_value for fragment in FORBIDDEN_FRAGMENTS):
                failures.append(f"{prefix}:forbidden_token"); reasons["forbidden_token"] += 1; break
        firewall = row.get("context_firewall")
        if not isinstance(firewall, Mapping) or firewall.get("forbidden_token_count") != 0 or firewall.get("sidecars_off_context") is not True:
            failures.append(f"{prefix}:firewall"); reasons["firewall"] += 1
        for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context"):
            if row.get(flag) is not False:
                failures.append(f"{prefix}:{flag}"); reasons[flag] += 1
        target_values = _target_map(target or [])
        if target_values.get("question") is None or target_values.get("next_action") is None or target_values.get("safe_to_send") not in {"0", "1"}:
            failures.append(f"{prefix}:target_projection_incomplete"); reasons["target_projection_incomplete"] += 1
        if row.get("training_eligible") is not False or row.get("memory_promotion_allowed") is not False:
            failures.append(f"{prefix}:promotion_flag_open"); reasons["promotion_flag_open"] += 1
        if view == "full_axis":
            presence = row.get("axis_presence")
            if not isinstance(presence, Mapping) or any(axis not in presence for axis in AXES):
                failures.append(f"{prefix}:full_axis_presence_missing"); reasons["full_axis_presence_missing"] += 1
    return failures, dict(reasons)


def audit(dataset: Mapping[str, Any]) -> dict[str, Any]:
    rows = dataset.get("records")
    row_failures, reasons = _audit_rows(rows)
    failures = list(row_failures)
    rows = [row for row in rows or [] if isinstance(row, Mapping)]
    full_train = _view_stats(rows, "full_axis", "train")
    full_holdout = _view_stats(rows, "full_axis")
    coarse_train = _view_stats(rows, "coarse_process", "train")
    coarse_holdout = _view_stats(rows, "coarse_process")
    full_axis_target_train_complete = bool(
        full_train["records"]
        and full_train["ask_target_count"] > 0
        and full_train["repair_target_count"] > 0
        and full_train["abstain_target_count"] > 0
    )
    coarse_target_train_complete = bool(
        coarse_train["records"]
        and coarse_train["ask_target_count"] > 0
        and coarse_train["repair_target_count"] > 0
        and coarse_train["abstain_target_count"] > 0
    )
    if not full_axis_target_train_complete:
        failures.append("full_axis_train_target_coverage_missing")
        reasons["full_axis_train_target_coverage_missing"] = reasons.get("full_axis_train_target_coverage_missing", 0) + 1
    if not coarse_target_train_complete:
        failures.append("coarse_train_target_coverage_missing")
        reasons["coarse_train_target_coverage_missing"] = reasons.get("coarse_train_target_coverage_missing", 0) + 1
    # The two views must remain auditable even when their source hashes happen
    # to be repeated in old artifacts.  A duplicate context+target sequence is
    # reported, never silently removed.
    duplicate_sequences = len(rows) - len({(str(row.get("view")), tuple(row.get("context_tokens") or []), tuple(row.get("target_tokens") or [])) for row in rows})
    implementation_hashes = defaultdict(lambda: {"train": 0, "holdout": 0})
    missing_implementation_hash = 0
    for row in rows:
        value = str(row.get("source_implementation_hash", ""))
        if len(value) != 64:
            missing_implementation_hash += 1
            continue
        implementation_hashes[str(row.get("view"))]["holdout" if row.get("split") != "train" else "train"] += 1
    audit_status = "passed_diagnostic_coarse_target_track" if not failures and coarse_target_train_complete else "blocked_full_axis_target_gap"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": audit_status,
        "dataset_sha256": dataset.get("dataset_sha256"),
        "counts": {
            "records": len(rows),
            "coarse_process": sum(row.get("view") == "coarse_process" for row in rows),
            "full_axis": sum(row.get("view") == "full_axis" for row in rows),
            "duplicate_context_target_sequences": duplicate_sequences,
            "missing_implementation_hash_rows": missing_implementation_hash,
        },
        # The full-axis gap is deliberately orthogonal to the coarse process
        # decoder smoke.  Keeping this flag separate prevents a missing
        # full-axis target label from being silently treated as a reason to
        # delete the coarse target evidence, while the top-level status stays
        # blocked for the eventual unified model.
        "coarse_process": {"train": coarse_train, "all": coarse_holdout, "diagnostic_training_allowed": coarse_target_train_complete and not row_failures},
        "full_axis": {"train": full_train, "all": full_holdout, "target_training_allowed": False, "target_train_coverage_complete": full_axis_target_train_complete},
        "target_coverage": {"coarse_train_complete": coarse_target_train_complete, "full_axis_train_complete": full_axis_target_train_complete, "full_axis_holdout_has_ask_or_repair": full_holdout["ask_target_count"] > 0 or full_holdout["repair_target_count"] > 0},
        "implementation_hash_groups": {key: dict(value) for key, value in implementation_hashes.items()},
        "failures": sorted(set(failures)),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "predictive_entropy": {"status": "not_run_until_full_axis_target_train_coverage", "relative_drop_limit": 0.25},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "audit_sha256": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-341 target-conditioned diagnostic corpus")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be an object")
    result = audit(dataset)
    result["audit_sha256"] = _sha(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "target_coverage": result["target_coverage"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
