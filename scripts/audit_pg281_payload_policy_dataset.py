"""Independent audit for the PG-281 abstract payload-policy dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg281_payload_policy_dataset_v1.json"
HARD = ROOT / "research" / "pg281_payload_policy_hard_negative_v1.json"
AUDIT = ROOT / "research" / "pg281_payload_policy_dataset_audit_v1.json"
FORBIDDEN = ("payload", "oracle", "response_body", "echo_excerpt", "confirmed_positive", "body_sha256")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    hard = json.loads(HARD.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    records = [row for row in list(data.get("records") or []) if isinstance(row, dict)]
    hard_rows = [row for row in list(hard.get("records") or []) if isinstance(row, dict)]
    check("dataset_hash", data.get("dataset_sha256") == sha(without(data, "dataset_sha256")))
    check("hard_negative_hash", hard.get("dataset_sha256") == sha(without(hard, "dataset_sha256")))
    check("source_loopback", bool(data.get("source", {}).get("loopback_only")) and data.get("source", {}).get("external_network") is False)
    check("source_catalog_audited", bool(data.get("source", {}).get("pg266_audit_sha256")) and bool(data.get("source", {}).get("pg269_audit_sha256")))
    check("row_quota", len(records) == 52 and int(data.get("counts", {}).get("total", 0)) == len(records))
    check("split_quota", sum(row.get("split") == "train" for row in records) == 43 and sum(row.get("split") == "route_dev" for row in records) == 4 and sum(row.get("split") == "family_holdout" for row in records) == 5)
    check("hard_negative_quota", len(hard_rows) == 12 and int(data.get("counts", {}).get("hard_negative", 0)) == 12)
    check("hard_negative_holdout", hard.get("training_eligible") is False and hard.get("memory_promotion_allowed") is False and all(row.get("training_eligible") is False for row in hard_rows))
    check("context_firewall", all(not any(any(term in str(token).casefold() for term in FORBIDDEN) for token in row.get("context_tokens", [])) for row in records + hard_rows))
    check("raw_fields_firewall", all(row.get("raw_payload_strings_stored") is False and row.get("raw_response_bodies_stored") is False and row.get("oracle_in_context") is False for row in records + hard_rows))
    target_values = {"probe_class", "channel", "encoding", "final_action", "safe_to_send", "oracle_required"}
    check("target_shape", all(set(row.get("target", {})) == target_values for row in records + hard_rows))
    check("target_tokens_abstract", all(row.get("target_tokens", [""])[0] == "[TARGET_BOS]" and row.get("target_tokens", [""])[-1] == "[TARGET_EOS]" for row in records + hard_rows))
    check("hard_negative_safe", all(row.get("target", {}).get("final_action") == "abstain" and row.get("target", {}).get("safe_to_send") is False for row in hard_rows))
    check("hard_negative_reason", all("evidence" in str(row.get("reason", "")).casefold() for row in hard_rows))
    check("promotion_blocked", data.get("training_contract", {}).get("memory_promotion_allowed") is False and data.get("training_contract", {}).get("vulnerability_claim_allowed") is False)
    audit = {
        "audit_id": "pg281-payload-policy-dataset-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in ["dataset_hash", "hard_negative_hash", "source_loopback", "source_catalog_audited", "row_quota", "split_quota", "hard_negative_quota", "hard_negative_holdout", "context_firewall", "raw_fields_firewall", "target_shape", "target_tokens_abstract", "hard_negative_safe", "hard_negative_reason", "promotion_blocked"]},
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "hard_negative_dataset": HARD.relative_to(ROOT).as_posix(),
        "rows": len(records),
        "hard_negative_rows": len(hard_rows),
        "interpretation": "这是抽象 probe plan 与安全拒答训练的可审计数据，不含原始 payload；真实发送和漏洞确认仍必须经过授权 loopback evaluator。",
        "failures": failures,
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
