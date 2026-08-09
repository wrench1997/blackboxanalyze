"""Audit PG-335 source-grounded process-token diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg335_real_process_token_diagnostic_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg335_real_process_token_diagnostic_audit_v1.json"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_replay")
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "record=", "path=", "url=", "payload=", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.0


def audit(data: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in list(data.get("records") or []) if isinstance(row, dict)]
    context = [str(token) for row in rows for token in list(row.get("context_tokens") or [])]
    forbidden = [token for token in context if token.casefold().startswith(FORBIDDEN)]
    ids = [str(row.get("record_id")) for row in rows]
    kinds = Counter(str(row.get("diagnostic_kind")) for row in rows)
    masks = Counter(str(row.get("axis_mask")) for row in rows if row.get("axis_mask"))
    pre = [row for row in rows if row.get("diagnostic_kind") == "ask"]
    failures = [row for row in rows if row.get("diagnostic_kind") == "failure"]
    negatives = [row for row in rows if row.get("diagnostic_kind") == "negative_review"]
    def axis_status(row: dict[str, Any], axis: str) -> str:
        values = list(dict(row.get("field_capture_manifest", {}).get(axis) or {}).values())
        if not values:
            return "unknown"
        statuses = {str(value) for value in values}
        if "not_observed" in statuses:
            return "not_observed"
        if "unknown" in statuses:
            return "unknown"
        if "observed" in statuses:
            return "observed"
        if "absent" in statuses:
            return "absent"
        return "unknown"

    axis_entropy = {axis: entropy([axis_status(row, axis) for row in rows]) for axis in AXES}
    checks = {
        "schema": data.get("schema_version") == "pg335-real-process-token-diagnostic-v1",
        "dataset_hash": data.get("dataset_sha256") == digest({key: value for key, value in data.items() if key != "dataset_sha256"}),
        "source_rows_present": int(dict(data.get("source") or {}).get("source_rows", 0) or 0) >= 45,
        "unique_ids": len(ids) == len(set(ids)),
        "seven_axis_manifest": all(isinstance(row.get("field_capture_manifest"), dict) and set(row["field_capture_manifest"]) == set(AXES) for row in rows),
        "context_firewall": not forbidden and all(row.get("context_firewall", {}).get("forbidden_token_count") == 0 and row.get("context_firewall", {}).get("sidecars_off_context") is True for row in rows),
        "all_axis_masks": all(masks.get(axis, 0) > 0 for axis in AXES),
        "ask_recall": bool(pre) and all(row.get("target_projection", {}).get("next_action") == "ask_typed" and row.get("target_projection", {}).get("safe_to_send") is False and row.get("target_projection", {}).get("action_changed") is True for row in pre),
        "failure_action_change": bool(failures) and all(row.get("target_projection", {}).get("next_action") == "repair" and row.get("target_projection", {}).get("action_changed") is True for row in failures),
        "negative_abstain": bool(negatives) and all(row.get("target_projection", {}).get("next_action") == "abstain" and row.get("target_projection", {}).get("safe_to_send") is False for row in negatives),
        "split_isolation": sum(row.get("split") == "train" for row in rows) > 0 and sum(row.get("split") == "implementation_holdout" for row in rows) > 0,
        "no_promotion": not any(row.get("training_eligible") or row.get("memory_promotion_allowed") or row.get("payload_catalog_promotion_allowed") or row.get("vulnerability_claim_allowed") for row in rows),
    }
    report = {
        "schema_version": "pg335-real-process-token-diagnostic-audit-v1",
        "status": "diagnostic_only" if all(checks.values()) else "blocked",
        "dataset_sha256": data.get("dataset_sha256"),
        "counts": {"records": len(rows), "source_rows": int(dict(data.get("source") or {}).get("source_rows", 0) or 0), "kinds": dict(kinds), "train": sum(row.get("split") == "train" for row in rows), "implementation_holdout": sum(row.get("split") == "implementation_holdout" for row in rows), "unique_context_sequences": len({tuple(row.get("context_tokens", [])) for row in rows}), "unique_target_sequences": len({tuple(row.get("target_tokens", [])) for row in rows}), "context_token_entropy_bits": round(entropy(context), 6)},
        "axis_mask_counts": dict(masks),
        "axis_presence_entropy_bits": axis_entropy,
        "checks": checks,
        "forbidden_context_tokens": forbidden[:8],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "conclusion": "Source-grounded diagnostic only: masks teach the ASK boundary and do not create real oracle positives; no payload or vulnerability claim.",
    }
    report["audit_sha256"] = digest(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-335 process-token diagnostics")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "failed": [key for key, value in report["checks"].items() if not value]}, ensure_ascii=False))
    return 0 if report["status"] == "diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
