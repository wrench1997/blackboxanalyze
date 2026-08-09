"""Audit PG-336 real failure/ASK/negative process-token diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg336_real_failure_process_token_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg336_real_failure_process_token_audit_v1.json"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_replay")
SCHEMA_VERSION = "pg336-real-failure-process-token-v1"
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "record=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.0


def _axis_status(row: Mapping[str, Any], axis: str) -> str:
    fields = dict(row.get("field_capture_manifest") or {}).get(axis)
    values = {str(value) for value in dict(fields or {}).values()}
    if "observed" in values and "not_observed" not in values:
        return "observed"
    if "absent" in values:
        return "absent"
    if "unknown" in values:
        return "unknown"
    return "not_observed"


def audit(data: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in list(data.get("records") or []) if isinstance(row, Mapping)]
    contexts = [str(token) for row in rows for token in list(row.get("context_tokens") or [])]
    targets = [str(token) for row in rows for token in list(row.get("target_tokens") or [])]
    forbidden = [token for token in [*contexts, *targets] if token.casefold().startswith(FORBIDDEN)]
    ids = [str(row.get("record_id")) for row in rows]
    kinds = Counter(str(row.get("diagnostic_kind")) for row in rows)
    failures = [row for row in rows if row.get("diagnostic_kind") == "failure_repair"]
    asks = [row for row in rows if row.get("diagnostic_kind") == "ask_preflight"]
    negatives = [row for row in rows if row.get("diagnostic_kind") == "negative_review"]
    probes = [row for row in rows if row.get("diagnostic_kind") == "probe_observed"]

    def has(row: Mapping[str, Any], prefix: str) -> bool:
        return any(str(token).startswith(prefix) for token in list(row.get("context_tokens") or []))

    checks = {
        "schema": data.get("schema_version") == SCHEMA_VERSION,
        "dataset_hash": data.get("dataset_sha256") == digest({key: value for key, value in data.items() if key != "dataset_sha256"}),
        "source_trace_present": len(str(dict(data.get("source") or {}).get("trace_sha256", ""))) == 64,
        "real_failure_trace_count": int(dict(data.get("source") or {}).get("real_failure_trace_count", 0) or 0) == 9 and len(failures) == 9,
        "real_negative_trace_count": int(dict(data.get("source") or {}).get("real_negative_trace_count", 0) or 0) == 9 and len(negatives) == 9,
        "ask_preflight_count": int(dict(data.get("source") or {}).get("ask_preflight_count", 0) or 0) == 135 and len(asks) == 135,
        "unique_ids": len(ids) == len(set(ids)) and all(item.startswith("pg336:") for item in ids),
        "seven_axis_manifest": all(set(dict(row.get("field_capture_manifest") or {})) == set(AXES) for row in rows),
        "context_firewall": not forbidden and all(dict(row.get("context_firewall") or {}).get("forbidden_token_count") == 0 and dict(row.get("context_firewall") or {}).get("sidecars_off_context") is True for row in rows),
        "sidecar_off_context": all(dict(row.get("evaluator_sidecar_ref") or {}).get("off_context") is True for row in rows),
        "raw_flags_clear": all(row.get("raw_payload_stored") is False and row.get("raw_response_body_stored") is False and row.get("oracle_answer_in_context") is False for row in rows),
        "get_post_present": bool(probes) and any(has(row, "surface_method=GET") for row in rows) and any(has(row, "surface_method=POST") for row in rows),
        "failure_action_change": bool(failures) and all(row.get("process_metadata", {}).get("real_failure_trace") is True and "action_changed=1" in list(row.get("target_tokens") or []) and row.get("target_tokens", [])[2] == "next_action=repair_abstract_plan" for row in failures),
        "ask_safe": bool(asks) and all(row.get("process_metadata", {}).get("real_ask_preflight") is True and "next_action=ask_typed" in list(row.get("target_tokens") or []) and "safe_to_send=0" in list(row.get("target_tokens") or []) for row in asks),
        "negative_abstain": bool(negatives) and all(row.get("process_metadata", {}).get("real_negative_evaluator_trace") is True and "next_action=abstain" in list(row.get("target_tokens") or []) and "safe_to_send=0" in list(row.get("target_tokens") or []) for row in negatives),
        "split_isolation": sum(row.get("split") == "train" for row in rows) > 0 and sum(row.get("split") == "seed_holdout" for row in rows) > 0,
        "implementation_gate_explicit": dict(data.get("source") or {}).get("independent_implementation_holdout") is False and dict(data.get("process_policy") or {}).get("seed_holdout_is_not_independent_implementation") is True,
        "no_promotion": not any(row.get("training_eligible") or row.get("memory_promotion_allowed") or row.get("payload_catalog_promotion_allowed") or row.get("vulnerability_claim_allowed") for row in rows) and all(value is False for value in dict(data.get("promotion") or {}).values()),
    }
    axis_presence_entropy = {axis: round(entropy([_axis_status(row, axis) for row in rows]), 6) for axis in AXES}
    report = {
        "schema_version": "pg336-real-failure-process-token-audit-v1",
        "status": "diagnostic_only" if all(checks.values()) else "blocked",
        "dataset_sha256": data.get("dataset_sha256"),
        "counts": {"records": len(rows), "kinds": dict(kinds), "probe_observed": len(probes), "failure_repair": len(failures), "negative_review": len(negatives), "ask_preflight": len(asks), "train": sum(row.get("split") == "train" for row in rows), "seed_holdout": sum(row.get("split") == "seed_holdout" for row in rows), "unique_context_sequences": len({tuple(row.get("context_tokens") or []) for row in rows}), "unique_target_sequences": len({tuple(row.get("target_tokens") or []) for row in rows}), "context_token_entropy_bits": round(entropy(contexts), 6), "target_token_entropy_bits": round(entropy(targets), 6)},
        "axis_presence_entropy_bits": axis_presence_entropy,
        "checks": checks,
        "forbidden_tokens": forbidden[:12],
        "scientific_gate": {"status": "blocked", "reason": "single_implementation_seed_holdout_only", "accepted_training_rows": 0, "independent_implementation_holdout": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "conclusion": "Real failure/ASK/negative process evidence is now represented, but this remains a diagnostic process dataset; it is not a transferable vulnerability or payload model.",
    }
    report["audit_sha256"] = digest(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-336 real failure process-token diagnostics")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "failed": [key for key, value in report["checks"].items() if not value]}, ensure_ascii=False))
    return 0 if report["status"] == "diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
