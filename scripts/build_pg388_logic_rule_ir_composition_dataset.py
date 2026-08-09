"""Build a safe, structured Rule-IR view of the PG-388 logic trajectories.

The source trajectory artifact deliberately mixes diagnostic target fields with
the ordered Rule-IR slots.  This builder creates a second, model-facing view:
only the 11 ordered slots become ``target_tokens``; effect/state fields remain
an evaluator-side summary.  The local live canary report is joined only as an
aggregate coverage attestation because it has no row-bound implementation and
seed identity.  It therefore never grants training eligibility.

No Docker, network, payload, wire, response body, or evaluator literal is
read or emitted by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "research" / "pg388_logic_canary_trajectory_dataset_v1.json"
DEFAULT_LIVE = ROOT / "research" / "pg388_logic_canary_live_v1.json"
SCHEMA_VERSION = "pg388-logic-rule-ir-composition-dataset-v1"
SOURCE_STATUS = "abstract_canary_trajectory_candidate_only"
SLOT_ORDER = (
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
SUMMARY_FIELDS = ("effect_shape", "state_delta", "invariant_result")
FORBIDDEN_MARKERS = (
    "http://",
    "https://",
    "payload=",
    "wire=",
    "response_body=",
    "raw_",
    "evaluator=",
    "<script",
)
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("pg388_json_object_required")
    return value


def _text_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"pg388_{label}_missing")
    result = [str(item) for item in value]
    if any(any(marker in item.casefold() for marker in FORBIDDEN_MARKERS) for item in result):
        raise ValueError(f"pg388_{label}_firewall_open")
    return result


def _parse_target(tokens: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    slots: dict[str, str] = {}
    summary: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if any(marker in text.casefold() for marker in FORBIDDEN_MARKERS):
            raise ValueError("pg388_target_firewall_open")
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in SLOT_ORDER:
            if key in slots:
                raise ValueError(f"pg388_duplicate_slot:{key}")
            slots[key] = value
        elif key in SUMMARY_FIELDS:
            if key in summary:
                raise ValueError(f"pg388_duplicate_summary:{key}")
            summary[key] = value
    missing = [slot for slot in SLOT_ORDER if slot not in slots]
    if missing:
        raise ValueError("pg388_slot_missing:" + ",".join(missing))
    return slots, summary


def _safe_source_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    context = _text_list(raw.get("context_tokens"), label="context_tokens")
    target = _text_list(raw.get("target_tokens"), label="target_tokens")
    slots, summary = _parse_target(target)
    for key in ("raw_source_stored", "raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context"):
        if raw.get(key) is not False:
            raise ValueError(f"pg388_source_{key}_must_be_false")
    return {
        "context_tokens": context,
        "slots": slots,
        "summary": summary,
        "split": str(raw.get("split", "")),
        "implementation_ref": str(raw.get("implementation_ref", "")),
        "seed_bucket": str(raw.get("seed_bucket", "")),
        "case_ref": str(raw.get("case_ref", "")),
        "role": str(raw.get("role", "")),
        "phase": str(raw.get("phase", "")),
        "record_ref_sha256": str(raw.get("record_ref_sha256", "")),
    }


def _live_coverage(live: Mapping[str, Any]) -> dict[str, Any]:
    counts = live.get("counts") if isinstance(live.get("counts"), Mapping) else {}
    rows = live.get("rows") if isinstance(live.get("rows"), list) else []
    case_refs = sorted({str(row.get("case_ref")) for row in rows if isinstance(row, Mapping) and row.get("case_ref")})
    role_phases = sorted(
        {
            f"{row.get('role')}:{row.get('phase')}"
            for row in rows
            if isinstance(row, Mapping) and row.get("role") and row.get("phase")
        }
    )
    # Only bounded aggregates are copied.  Evidence hashes, route literals and
    # all evaluator fields stay in the source report and are never joined here.
    return {
        "status": str(live.get("status", "unknown")),
        "fresh_resets": int(counts.get("fresh_resets", 0) or 0),
        "typed_observations": int(counts.get("typed_observations", 0) or 0),
        "candidate_effects": int(counts.get("candidate_effects", 0) or 0),
        "negative_control_clean": int(counts.get("negative_control_clean", 0) or 0),
        "unsafe_allow": int(counts.get("unsafe_allow", 0) or 0),
        "case_count": len(case_refs),
        "role_phase_shape_count": len(role_phases),
        "row_bound": False,
    }


def _model_row(source: Mapping[str, Any]) -> dict[str, Any]:
    slots = source["slots"]
    safe_row = {
        "record_ref_sha256": source["record_ref_sha256"],
        "split": source["split"],
        "implementation_ref": source["implementation_ref"],
        "seed_bucket": source["seed_bucket"],
        "case_ref": source["case_ref"],
        "role": source["role"],
        "phase": source["phase"],
        "context_tokens": list(source["context_tokens"]),
        "target_tokens": [f"{slot}={slots[slot]}" for slot in SLOT_ORDER],
        "evaluator_summary": dict(source["summary"]),
        "source_row_bound_typed_evidence": False,
        "fresh_role_reset_attested": False,
        "operator_reviewed": False,
        "raw_source_stored": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
    }
    row = dict(safe_row)
    row["row_sha256"] = _sha(safe_row)
    return row


def build_dataset(
    *,
    source_path: Path = DEFAULT_SOURCE,
    live_path: Path = DEFAULT_LIVE,
) -> dict[str, Any]:
    source = _load(source_path)
    if source.get("status") != SOURCE_STATUS:
        raise ValueError("pg388_source_status_mismatch")
    raw_rows = source.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("pg388_source_rows_missing")
    live = _load(live_path)
    safe_rows = [_safe_source_row(row) for row in raw_rows if isinstance(row, Mapping)]
    model_rows = [_model_row(row) for row in safe_rows]
    counts = {
        "records": len(model_rows),
        "train": sum(row["split"] == "train" for row in model_rows),
        "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in model_rows),
        "slot_count": len(SLOT_ORDER),
        "summary_fields_off_context": len(SUMMARY_FIELDS),
    }
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "pg388_logic_rule_ir_composition_dataset_v1",
        "status": "abstract_rule_ir_composition_candidate_only",
        "objective": "按顺序组合不变量、状态转移、动作和修复槽；不学习原始攻击字符串",
        "source": {
            "trajectory_path": str(source_path),
            "trajectory_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "live_report_path": str(live_path),
            "live_report_sha256": hashlib.sha256(live_path.read_bytes()).hexdigest(),
            "live_binding": "aggregate_only_not_row_bound",
        },
        "rows": model_rows,
        "counts": counts,
        "slot_order": list(SLOT_ORDER),
        "summary_fields_off_context": list(SUMMARY_FIELDS),
        "live_coverage": _live_coverage(live),
        "context_firewall": {
            "raw_source": False,
            "raw_payload": False,
            "raw_response": False,
            "evaluator_answer": False,
            "external_network": False,
        },
        "source_contract": {
            "trajectory_is_synthetic_abstract": True,
            "row_bound_typed_evidence": False,
            "fresh_role_reset_attested": False,
            "operator_reviewed": False,
            "candidate_reference_negative_replay": True,
            "training_eligible": 0,
        },
        "training_eligible": 0,
        "promotion": dict(PROMOTION),
    }
    artifact["dataset_sha256"] = _sha({key: value for key, value in artifact.items() if key != "dataset_sha256"})
    return artifact


def audit_dataset(artifact: Mapping[str, Any]) -> dict[str, Any]:
    rows = artifact.get("rows")
    failures: list[str] = []
    invalid_rows = 0
    seen_hashes: set[str] = set()
    if artifact.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if artifact.get("status") != "abstract_rule_ir_composition_candidate_only":
        failures.append("status")
    if artifact.get("training_eligible") != 0:
        failures.append("training_flag")
    if artifact.get("promotion") != PROMOTION:
        failures.append("promotion")
    if artifact.get("source_contract", {}).get("row_bound_typed_evidence") is not False:
        failures.append("row_bound_typed_evidence")
    if not isinstance(rows, list) or not rows:
        failures.append("rows")
        rows = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            invalid_rows += 1
            continue
        try:
            context = _text_list(raw.get("context_tokens"), label="context_tokens")
            target = _text_list(raw.get("target_tokens"), label="target_tokens")
            slots, _ = _parse_target(target)
            if list(target) != [f"{slot}={slots[slot]}" for slot in SLOT_ORDER]:
                raise ValueError("target_slot_order")
            for key in ("raw_source_stored", "raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context"):
                if raw.get(key) is not False:
                    raise ValueError(key)
            payload = {key: raw[key] for key in raw if key != "row_sha256"}
            if _sha(payload) != raw.get("row_sha256"):
                raise ValueError("row_hash")
            if raw.get("row_sha256") in seen_hashes:
                raise ValueError("duplicate_row_hash")
            seen_hashes.add(str(raw.get("row_sha256")))
            if any(any(marker in token.casefold() for marker in FORBIDDEN_MARKERS) for token in context):
                raise ValueError("context_firewall")
        except (TypeError, ValueError, KeyError):
            invalid_rows += 1
    counts = artifact.get("counts") if isinstance(artifact.get("counts"), Mapping) else {}
    if counts.get("records") != len(rows):
        failures.append("count_records")
    if counts.get("slot_count") != len(SLOT_ORDER):
        failures.append("slot_count")
    report = {
        "schema_version": "pg388-logic-rule-ir-composition-audit-v1",
        "status": "passed_candidate_rule_ir_audit" if not failures and invalid_rows == 0 else "blocked_rule_ir_audit",
        "records": len(rows),
        "invalid_rows": invalid_rows,
        "failures": sorted(set(failures)),
        "slot_count": len(SLOT_ORDER),
        "unique_row_hashes": len(seen_hashes),
        "context_firewall_passed": invalid_rows == 0,
        "training_eligible": 0,
        "promotion": dict(PROMOTION),
    }
    report["report_sha256"] = _sha(report)
    return report


def write_dataset(path: str | Path, *, source_path: Path = DEFAULT_SOURCE, live_path: Path = DEFAULT_LIVE) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_dataset(source_path=source_path, live_path=live_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--live-report", default=str(DEFAULT_LIVE))
    parser.add_argument("--output", default="research/pg388_logic_rule_ir_composition_dataset_v1.json")
    args = parser.parse_args()
    output = write_dataset(args.output, source_path=Path(args.source), live_path=Path(args.live_report))
    artifact = _load(output)
    print(json.dumps({"output": str(output), "status": artifact["status"], "counts": artifact["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
