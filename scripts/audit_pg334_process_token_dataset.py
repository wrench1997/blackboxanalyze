"""Read-only audit for PG-334 abstract process-token diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg334_process_token_diagnostic_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg334_process_token_diagnostic_audit_v1.json"
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "slot=", "unknown_slot=", "bound_slot=", "oracle=", "evaluator=", "payload=", "raw_", "response_body=", "response_body_text=", "expected_")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def audit(data: dict[str, Any]) -> dict[str, Any]:
    rows = list(data.get("records") or [])
    contexts = [str(token) for row in rows for token in row.get("context_tokens", [])]
    forbidden = [token for token in contexts if token.casefold().startswith(FORBIDDEN)]
    ids = [str(row.get("record_id")) for row in rows]
    paired = Counter(str(row.get("paired_id")) for row in rows)
    split_counts = Counter(str(row.get("split")) for row in rows)
    pre = [row for row in rows if row.get("stage") == "pre"]
    post = [row for row in rows if row.get("stage") == "post"]
    checks = {
        "schema": data.get("schema_version") == "pg334-process-token-diagnostic-v1",
        "dataset_hash": data.get("dataset_sha256") == digest({key: value for key, value in data.items() if key != "dataset_sha256"}),
        "records_nonempty": bool(rows),
        "unique_ids": len(ids) == len(set(ids)),
        "pair_completeness": bool(paired) and all(count == 2 for count in paired.values()),
        "train_holdout_present": split_counts.get("train", 0) > 0 and split_counts.get("implementation_holdout", 0) > 0,
        "seven_axis_manifest": all(isinstance(row.get("field_capture_manifest"), dict) and len(row["field_capture_manifest"]) == 7 for row in rows),
        "context_firewall": not forbidden and all(row.get("context_firewall", {}).get("forbidden_token_count") == 0 and row.get("context_firewall", {}).get("sidecars_off_context") is True for row in rows),
        "ask_recall": bool(pre) and all(row.get("target_projection", {}).get("question") == "ask_missing_observation" and row.get("target_projection", {}).get("safe_to_send") is False for row in pre),
        "negative_abstain": bool(post) and all(not row.get("process_metadata", {}).get("negative_control") or row.get("target_projection", {}).get("next_action") == "abstain" for row in post),
        "repair_action_change": bool(rows) and all(row.get("target_projection", {}).get("action_changed") is True for row in rows),
        "no_promotion": not any(row.get("training_eligible") or row.get("memory_promotion_allowed") or row.get("payload_catalog_promotion_allowed") or row.get("vulnerability_claim_allowed") for row in rows),
    }
    report = {
        "schema_version": "pg334-process-token-diagnostic-audit-v1",
        "status": "diagnostic_only" if all(checks.values()) else "blocked",
        "dataset_sha256": data.get("dataset_sha256"),
        "counts": {"records": len(rows), "pre": len(pre), "post": len(post), "train": split_counts.get("train", 0), "implementation_holdout": split_counts.get("implementation_holdout", 0), "unique_context_sequences": len({tuple(row.get("context_tokens", [])) for row in rows}), "unique_target_sequences": len({tuple(row.get("target_tokens", [])) for row in rows})},
        "checks": checks,
        "forbidden_context_tokens": forbidden[:8],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "conclusion": "Controlled process-token diagnostics only; not real vulnerability gold and not a capability/payload claim.",
    }
    report["audit_sha256"] = digest(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-334 process-token diagnostics")
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
