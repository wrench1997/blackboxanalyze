"""Project the local supplemental canary into abstract Rule-IR diagnostics.

The source report is evaluator-side state-shape evidence.  This builder keeps
only the ontology context/target projection plus hashes and reset metadata; it
never copies effect buckets, response bodies, wire, payloads, or evaluator
answers into the model view.  The output intentionally has no train split and
is therefore not training data.
"""

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

from app.pg388_logic_invariant_projection import project_logic_case  # noqa: E402


SCHEMA_VERSION = "pg388-logic-supplement-typed-rule-ir-projection-v1"
DEFAULT_INPUT = ROOT / "research" / "pg388_logic_supplement_canary_local_v1.json"
DEFAULT_AUDIT = ROOT / "research" / "pg388_logic_supplement_canary_local_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg388_logic_supplement_typed_rule_ir_projection_v1.json"
_RAW_MARKERS = ("http://", "https://", "payload=", "wire=", "response_body=", "<script")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("typed_projection_object_required")
    return document


def _feedback_for_role(row: dict[str, Any]) -> str:
    role = row.get("role")
    if role == "replay":
        return "typed_effect"
    if role in {"candidate", "negative"}:
        return "invariant_mismatch"
    return "missing"


def _assert_abstract(value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False).casefold()
    if any(marker in serialized for marker in _RAW_MARKERS):
        raise ValueError("typed_projection_raw_marker")


def build(input_path: Path = DEFAULT_INPUT, audit_path: Path = DEFAULT_AUDIT) -> dict[str, Any]:
    report = _load(input_path)
    audit = _load(audit_path) if audit_path.exists() else {}
    if report.get("status") != "completed_local_supplemental_canary_diagnostic":
        raise ValueError("typed_projection_source_status_blocked")
    if audit.get("status") != "passed_candidate_only":
        raise ValueError("typed_projection_source_audit_blocked")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("typed_projection_source_rows_missing")

    projected_rows: list[dict[str, Any]] = []
    for source in rows:
        if not isinstance(source, dict):
            raise ValueError("typed_projection_source_row_object_required")
        case_ref = str(source.get("case_ref", ""))
        role = str(source.get("role", ""))
        feedback_state = _feedback_for_role(source)
        projection = project_logic_case(case_ref, role=role, feedback_state=feedback_state)
        core = {
            "record_ref_sha256": _sha({
                "input_report_sha256": report.get("report_sha256"),
                "case_ref": case_ref,
                "seed": source.get("seed"),
                "role": role,
                "phase": source.get("phase"),
            }),
            "split": "evaluator_diagnostic",
            "implementation_ref": str(report.get("implementation_id", "")),
            "seed_bucket": f"seed_{source.get('seed')}",
            "case_ref": case_ref,
            "feedback_state": feedback_state,
            "role": role,
            "context_tokens": projection["context_tokens"],
            "target_tokens": projection["target_tokens"],
            "logic_context": projection["logic_context"],
            "target_projection": projection["target_projection"],
            "evaluator_sidecar_ref_sha256": str(source.get("evidence_sha256", "")),
            "fresh_reset": source.get("fresh_reset_before") is True and source.get("fresh_reset_after") is True,
            "typed_evaluator_observed": source.get("typed_observation") is True,
            "role_bound_evidence": isinstance(source.get("evidence_sha256"), str) and len(source["evidence_sha256"]) == 64,
            "operator_reviewed": False,
            "raw_source_stored": False,
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "training_eligible": False,
        }
        _assert_abstract({"context_tokens": core["context_tokens"], "target_tokens": core["target_tokens"], "logic_context": core["logic_context"], "target_projection": core["target_projection"]})
        row = dict(core)
        row["row_sha256"] = _sha(core)
        projected_rows.append(row)

    source_hash = str(report.get("report_sha256", ""))
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "pg388_logic_supplement_typed_rule_ir_projection_v1",
        "status": "typed_rule_ir_diagnostic_candidate_only",
        "objective": "把 fresh-role local canary 的抽象反馈接入 Rule-IR 诊断视图，不生成训练金样本",
        "source_report": {"path": input_path.name, "report_sha256": source_hash, "audit_path": audit_path.name, "audit_sha256": _sha(audit)},
        "rows": projected_rows,
        "counts": {
            "records": len(projected_rows),
            "evaluator_diagnostic": len(projected_rows),
            "train": 0,
            "implementation_holdout": 0,
            "cases": len({row["case_ref"] for row in projected_rows}),
            "seeds": len({row["seed_bucket"] for row in projected_rows}),
            "roles": len({row["role"] for row in projected_rows}),
            "typed_evaluator_observed": sum(row["typed_evaluator_observed"] for row in projected_rows),
            "fresh_reset": sum(row["fresh_reset"] for row in projected_rows),
            "role_bound_evidence": sum(row["role_bound_evidence"] for row in projected_rows),
        },
        "context_firewall": {
            "raw_source": False,
            "raw_payload": False,
            "raw_response": False,
            "evaluator_answer": False,
            "external_network": False,
            "state_effect_buckets_in_context": False,
        },
        "source_contract": {
            "fresh_role_reset": True,
            "candidate_reference_negative_replay": True,
            "typed_evidence": True,
            "operator_reviewed": False,
            "train_split_present": False,
            "implementation_holdout_present": False,
            "live_rows_emitted": True,
        },
        "training_eligible": 0,
        "representation_pretrain_candidate_allowed": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    artifact["dataset_sha256"] = _sha({key: value for key, value in artifact.items() if key != "dataset_sha256"})
    return artifact


def write(path: str | Path = DEFAULT_OUTPUT, *, input_path: Path = DEFAULT_INPUT, audit_path: Path = DEFAULT_AUDIT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(input_path, audit_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = write(args.output, input_path=args.input, audit_path=args.audit)
    document = _load(output)
    print(json.dumps({"output": str(output), "status": document["status"], "counts": document["counts"], "training_eligible": document["training_eligible"]}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()

