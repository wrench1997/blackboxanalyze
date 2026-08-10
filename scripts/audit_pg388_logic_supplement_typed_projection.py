"""Audit the abstract Rule-IR view derived from the local canary report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg388_logic_supplement_typed_projection import (  # noqa: E402
    DEFAULT_OUTPUT,
    SCHEMA_VERSION,
    _sha,
)


AUDIT_SCHEMA_VERSION = "pg388-logic-supplement-typed-rule-ir-projection-audit-v1"
_FORBIDDEN_MARKERS = ("http://", "https://", "payload=", "wire=", "response_body=", "<script")
_FORBIDDEN_KEYS = {"payload", "wire", "response_body", "oracle_answer", "evaluator_answer", "effect_shape", "state_before", "state_after", "vulnerable_effect"}


def _walk_check(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS:
                return False
            if not _walk_check(item):
                return False
        return not any(marker in json.dumps(value, ensure_ascii=False).casefold() for marker in _FORBIDDEN_MARKERS)
    if isinstance(value, list):
        return all(_walk_check(item) for item in value)
    if isinstance(value, str):
        return not any(marker in value.casefold() for marker in _FORBIDDEN_MARKERS)
    return True


def audit(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    input_path = Path(path)
    failures: list[str] = []
    if not input_path.exists():
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_missing_projection", "failures": ["missing_projection"], "training_eligible": 0}
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_invalid_projection", "failures": ["invalid_projection"], "training_eligible": 0}
    if not isinstance(document, dict):
        return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "blocked_invalid_projection", "failures": ["object_required"], "training_eligible": 0}
    claimed_hash = document.get("dataset_sha256")
    without_hash = dict(document)
    without_hash.pop("dataset_sha256", None)
    if not isinstance(claimed_hash, str) or claimed_hash != _sha(without_hash):
        failures.append("dataset_hash_mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version_mismatch")
    if document.get("status") != "typed_rule_ir_diagnostic_candidate_only":
        failures.append("status_mismatch")
    if document.get("training_eligible") != 0 or document.get("representation_pretrain_candidate_allowed") is not False:
        failures.append("training_gate_open")
    if document.get("promotion") != {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }:
        failures.append("promotion_gate_open")
    if document.get("source_contract") != {
        "fresh_role_reset": True,
        "candidate_reference_negative_replay": True,
        "typed_evidence": True,
        "operator_reviewed": False,
        "train_split_present": False,
        "implementation_holdout_present": False,
        "live_rows_emitted": True,
    }:
        failures.append("source_contract_mismatch")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 120:
        failures.append("row_count_mismatch")
        rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, dict):
            failures.append("row_object_required")
            continue
        if not _walk_check(row):
            failures.append("raw_or_evaluator_material_in_row")
        if row.get("split") != "evaluator_diagnostic":
            failures.append("unexpected_train_split")
        if row.get("fresh_reset") is not True or row.get("typed_evaluator_observed") is not True or row.get("role_bound_evidence") is not True:
            failures.append("source_evidence_missing")
        core = dict(row)
        claimed_row_hash = core.pop("row_sha256", None)
        if not isinstance(claimed_row_hash, str) or claimed_row_hash != _sha(core):
            failures.append("row_hash_mismatch")
        if not isinstance(row.get("context_tokens"), list) or not isinstance(row.get("target_tokens"), list):
            failures.append("token_projection_missing")
    counts = document.get("counts") if isinstance(document.get("counts"), dict) else {}
    summary_counts = {
        "records": len(rows),
        "train": int(counts.get("train", 0) or 0),
        "implementation_holdout": int(counts.get("implementation_holdout", 0) or 0),
        "evaluator_diagnostic": int(counts.get("evaluator_diagnostic", 0) or 0),
        "typed_evaluator_observed": int(counts.get("typed_evaluator_observed", 0) or 0),
        "fresh_reset": int(counts.get("fresh_reset", 0) or 0),
        "role_bound_evidence": int(counts.get("role_bound_evidence", 0) or 0),
    }
    if summary_counts["train"] != 0 or summary_counts["implementation_holdout"] != 0 or summary_counts["evaluator_diagnostic"] != 120:
        failures.append("split_summary_mismatch")
    if summary_counts["typed_evaluator_observed"] != 120 or summary_counts["fresh_reset"] != 120 or summary_counts["role_bound_evidence"] != 120:
        failures.append("evidence_summary_mismatch")
    result: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed_diagnostic_only" if not failures else "blocked_typed_projection_contract",
        "input_file": input_path.name,
        "input_sha256": _sha(document),
        "counts": summary_counts,
        "failures": sorted(set(failures)),
        "context_firewall_passed": not any(code == "raw_or_evaluator_material_in_row" for code in failures),
        "training_eligible": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "scope": "typed abstract Rule-IR diagnostic;no_train_split;no_general_vulnerability_or_payload_claim",
    }
    result["audit_sha256"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result


def write(path: str | Path, result: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg388_logic_supplement_typed_rule_ir_projection_audit_v1.json")
    args = parser.parse_args()
    result = audit(args.input)
    write(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"], "counts": result["counts"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()

