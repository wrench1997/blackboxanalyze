"""Audit the PG-388 row-bound Rule-IR source-row artifacts without I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Keep direct ``python scripts/...`` execution rooted at the repository.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import validate_pg331_source_row


RULE_IR_SLOTS = (
    "question",
    "ask_reason",
    "logic_invariant_ref",
    "state_transition_ref",
    "precondition_ref",
    "counterfactual_ref",
    "probe_variant_ref",
    "next_action",
    "repair_action",
    "oracle_ref",
    "safe_to_send",
)
FORBIDDEN = ("http://", "https://", "payload=", "wire=", "response_body=", "oracle_answer=", "evaluator_answer=")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_hits(value: Any) -> int:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    return sum(serialized.count(marker) for marker in FORBIDDEN)


def audit(report_path: Path, rows_path: Path, sidecars_path: Path) -> dict[str, Any]:
    report = _load(report_path)
    rows_doc = _load(rows_path)
    sidecars_doc = _load(sidecars_path)
    rows = rows_doc.get("rows")
    sidecars = sidecars_doc.get("sidecars")
    failures: list[str] = []
    if report.get("status") != "completed_logic_rule_ir_source_rows_candidate_only":
        failures.append("report_status")
    if not isinstance(rows, list):
        failures.append("rows_missing")
        rows = []
    if not isinstance(sidecars, list):
        failures.append("sidecars_missing")
        sidecars = []
    strict_valid = 0
    row_refs: set[str] = set()
    for index, wrapper in enumerate(rows):
        if not isinstance(wrapper, dict):
            failures.append(f"row_{index}_not_object")
            continue
        source_row = wrapper.get("source_row")
        if not isinstance(source_row, dict):
            failures.append(f"row_{index}_source_row_missing")
            continue
        validation = validate_pg331_source_row(source_row)
        if not validation.get("valid"):
            failures.append(f"row_{index}_strict:{','.join(validation.get('failures') or [])}")
        else:
            strict_valid += 1
        ref = str(wrapper.get("record_ref_sha256", ""))
        if not ref or ref != str(source_row.get("record_sha256", "")):
            failures.append(f"row_{index}_record_ref")
        row_refs.add(ref)
        target = wrapper.get("logic_rule_ir_target")
        tokens = wrapper.get("logic_rule_ir_target_tokens")
        if not isinstance(target, dict) or tuple(target.keys()) != RULE_IR_SLOTS:
            failures.append(f"row_{index}_rule_ir_slots")
        if not isinstance(tokens, list) or len(tokens) != len(RULE_IR_SLOTS) + 2 or tokens[0] != "[RULE_IR_BOS]" or tokens[-1] != "[RULE_IR_EOS]":
            failures.append(f"row_{index}_rule_ir_tokens")
        if isinstance(target, dict) and target.get("safe_to_send") is not False:
            failures.append(f"row_{index}_safe_to_send")
        if wrapper.get("training_eligible") is not False:
            failures.append(f"row_{index}_training_flag")
    sidecar_refs = {str(item.get("record_ref_sha256", "")) for item in sidecars if isinstance(item, dict)}
    if len(row_refs) != len(rows):
        failures.append("duplicate_record_refs")
    if sidecar_refs != row_refs:
        failures.append("sidecar_join")
    if _literal_hits(rows) or _literal_hits(sidecars):
        failures.append("raw_literal_marker")
    expected = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    for key, actual in (("source_rows", len(rows)), ("strict_valid", strict_valid), ("typed", len(sidecars))):
        if int(expected.get(key, -1)) != actual:
            failures.append(f"count_{key}")
    if report.get("training_eligible") != 0 or report.get("promotion", {}).get("training_allowed") is not False:
        failures.append("training_gate")
    return {
        "schema_version": "pg388-logic-rule-ir-source-row-audit-v1",
        "status": "passed_candidate_logic_rule_ir_source_row_audit" if not failures else "blocked_logic_rule_ir_source_row_audit",
        "report": report_path.name,
        "rows": rows_path.name,
        "sidecars": sidecars_path.name,
        "report_sha256": _sha(report_path),
        "rows_sha256": _sha(rows_path),
        "sidecars_sha256": _sha(sidecars_path),
        "counts": {"records": len(rows), "strict_valid": strict_valid, "sidecars": len(sidecars), "unique_record_refs": len(row_refs), "raw_literal_hits": _literal_hits(rows) + _literal_hits(sidecars)},
        "failures": failures,
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--sidecars", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.report, args.rows, args.sidecars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed_candidate_logic_rule_ir_source_row_audit":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
