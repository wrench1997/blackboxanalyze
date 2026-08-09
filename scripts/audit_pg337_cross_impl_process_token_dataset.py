"""Read-only audit for the PG-337 cross-implementation process dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "research" / "pg337_cross_impl_process_token_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg337_cross_impl_process_token_audit_v1.json"
SCHEMA_VERSION = "pg337-cross-impl-process-token-v1"
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return 0.0
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def audit(data: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    records = data.get("records") if isinstance(data, dict) else None
    if data.get("schema_version") != SCHEMA_VERSION or data.get("status") != "diagnostic_only_cross_implementation":
        failures.append("schema")
    if not isinstance(records, list) or not records:
        failures.append("records")
        records = []
    split_counts = Counter(str(row.get("split")) for row in records if isinstance(row, dict))
    if not split_counts.get("train") or not split_counts.get("implementation_holdout"):
        failures.append("cross_split")
    if data.get("source", {}).get("independent_implementation_holdout") is not True:
        failures.append("implementation_holdout")
    forbidden_tokens: list[str] = []
    row_failures: list[str] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            row_failures.append(f"row_{index}_mapping")
            continue
        context = row.get("context_tokens")
        if not isinstance(context, list) or len(context) < 2:
            row_failures.append(f"row_{index}_context")
            continue
        forbidden = [str(token) for token in context if any(fragment in str(token).casefold() for fragment in FORBIDDEN)]
        forbidden_tokens.extend(forbidden)
        if forbidden:
            row_failures.append(f"row_{index}_context_firewall")
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            row_failures.append(f"row_{index}_firewall_metadata")
        for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context", "training_eligible", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
            if row.get(key) is not False:
                row_failures.append(f"row_{index}_{key}")
        target = " ".join(str(token) for token in list(row.get("target_tokens") or []))
        kind = str(row.get("diagnostic_kind", ""))
        if kind == "failure_repair":
            if "next_action=repair" not in target or "action_changed=1" not in target:
                row_failures.append(f"row_{index}_failure_repair_target")
        if kind == "negative_review" and ("next_action=abstain" not in target or "safe_to_send=0" not in target):
            row_failures.append(f"row_{index}_negative_target")
        meta = row.get("process_metadata") if isinstance(row.get("process_metadata"), dict) else {}
        if str(meta.get("source_track")) == "pg337_dvwa" and meta.get("independent_implementation_holdout") is not True:
            row_failures.append(f"row_{index}_dvwa_holdout")
    counts = {"records": len(records), "train": int(split_counts.get("train", 0)), "implementation_holdout": int(split_counts.get("implementation_holdout", 0)), "failure_repair": sum(int(str(r.get("diagnostic_kind")) == "failure_repair") for r in records if isinstance(r, dict)), "negative_review": sum(int(str(r.get("diagnostic_kind")) == "negative_review") for r in records if isinstance(r, dict)), "ask_preflight": sum(int(str(r.get("diagnostic_kind")) == "ask_preflight") for r in records if isinstance(r, dict))}
    unsigned = dict(data)
    expected_sha = unsigned.pop("dataset_sha256", "")
    if _sha(unsigned) != expected_sha:
        failures.append("dataset_hash")
    failures.extend(row_failures)
    failures = sorted(set(failures))
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "schema_version": "pg337-cross-impl-process-token-audit-v1",
        "status": "diagnostic_only" if not failures else "blocked",
        "dataset_sha256": expected_sha,
        "counts": counts,
        "unique_context_sequences": len({" ".join(str(t) for t in r.get("context_tokens", [])) for r in records if isinstance(r, dict)}),
        "unique_target_sequences": len({" ".join(str(t) for t in r.get("target_tokens", [])) for r in records if isinstance(r, dict)}),
        "context_token_entropy_bits": _entropy([str(t) for r in records if isinstance(r, dict) for t in r.get("context_tokens", [])]),
        "target_token_entropy_bits": _entropy([str(t) for r in records if isinstance(r, dict) for t in r.get("target_tokens", [])]),
        "forbidden_tokens": sorted(set(forbidden_tokens)),
        "checks": {"cross_split": "cross_split" not in failures, "independent_implementation_holdout": "implementation_holdout" not in failures, "context_firewall": not any(item.endswith("context_firewall") for item in failures), "failure_repair_target": not any("failure_repair_target" in item for item in failures), "negative_abstain": not any("negative_target" in item for item in failures), "hash": "dataset_hash" not in failures},
        "scientific_gate": {"status": "blocked", "accepted_training_rows": 0, "reason": "diagnostic_process_track_requires_operator_review_and_information_ablation"},
        "promotion": promotion,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="audit PG-337 cross-implementation process tokens")
    parser.add_argument("--dataset", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["audit_sha256"] = _sha({key: value for key, value in result.items() if key != "audit_sha256"})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
