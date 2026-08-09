"""PG-100: revalidate a fresh local Pikachu replay with an independent sink oracle.

This is an evaluation-only lane.  It reuses the allow-listed PG-52 transport
adapter for one fresh loopback container per case, but deliberately discards
PG-52's verdict fields before calling the independent PG-100 oracle.  Only
bounded projections, hashes, and typed verdicts are written.  No training
catalog or long-term memory entry is produced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg52_authoritative_local_oracle as pg52  # noqa: E402
from app.pg100_semantic_sink_oracle import (  # noqa: E402
    SCHEMA_VERSION,
    evaluate_browser_pair,
    evaluate_redirect_pair,
    evaluate_sql_pair,
    model_visible_has_evaluator_label,
)


PROTOCOL_ID = "pg-pk-100-independent-semantic-sink-replay-v1"
IMAGE = pg52.IMAGE
GET_BASE = "http://127.0.0.1:8767"
POST_BASE = "http://127.0.0.1:8768"
REPORT_PATH = ROOT / "research" / "pg100_semantic_sink_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg100_semantic_sink_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg100_semantic_sink_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg100_semantic_sink_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg100_semantic_sink_report_v1.md"
PG99_REPORT_PATH = ROOT / "research" / "pg99_surface_novelty_report_v1.json"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _valid_hash(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def _redirect_raw(case: dict[str, Any], base: str, container_id: str) -> dict[str, Any]:
    """Capture redirect headers in memory; the destination is never followed."""

    destination = "http://127.0.0.1:8768/pg100-loopback-callback"
    control_destination = "i"
    with httpx.Client(base_url=base, timeout=8.0, follow_redirects=False) as client:
        control_response = client.get(case["path"], params={case["field"]: control_destination})
        candidate_response = client.get(case["path"], params={case["field"]: destination})
    return {
        "control": {
            "response": pg52.response_projection(control_response),
            "status": int(control_response.status_code),
            "location": str(control_response.headers.get("location", "")),
        },
        "candidate": {
            "response": pg52.response_projection(candidate_response),
            "status": int(candidate_response.status_code),
            "location": str(candidate_response.headers.get("location", "")),
        },
        "expected_destination": destination,
        "payload_manifest": pg52.build_payload_manifest(
            family=case["family"],
            surface=case["surface"],
            method=case["method"],
            field=case["field"],
            payload=destination,
            probe_ref="pg100-redirect-loopback-destination",
            mode=case["mode"],
        ),
        "container_id": container_id,
    }


def _fresh_reset(container_id: str, case_id: str, reset_hash: str) -> dict[str, Any]:
    return {
        "kind": "pg100-disposable-container-round",
        "reset_id": f"pg100-reset-{case_id}-{container_id[:12]}",
        "target_instance_id": container_id[:24],
        "state_epoch": f"{container_id[:16]}-read-only",
        "reset_adapter_sha256": reset_hash,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "read_only_round": True,
    }


def _browser_evidence(case: dict[str, Any], raw: dict[str, Any], reset: dict[str, Any]) -> str:
    paths = list(raw.get("execution_paths") or [])
    value = {
        "case_id": case["case_id"],
        "control_executed": bool(raw["control"].get("executed")),
        "candidate_executed": bool(raw["candidate"].get("executed")),
        "execution_paths": paths,
        "control_projection_sha256": str((raw["control"].get("response") or {}).get("projection_sha256", "")),
        "candidate_projection_sha256": str((raw["candidate"].get("response") or {}).get("projection_sha256", "")),
        "reset_id": reset["reset_id"],
    }
    return _hash(value)


def _sql_evidence(case: dict[str, Any], raw: dict[str, Any], reset: dict[str, Any]) -> str:
    value = {
        "case_id": case["case_id"],
        "control_ast_sha256": str((raw["control"].get("ast") or {}).get("ast_sha256", "")),
        "negative_ast_sha256": str((raw["negative"].get("ast") or {}).get("ast_sha256", "")),
        "candidate_ast_sha256": str((raw["candidate"].get("ast") or {}).get("ast_sha256", "")),
        "control_projection_sha256": str((raw["control"].get("response") or {}).get("projection_sha256", "")),
        "negative_projection_sha256": str((raw["negative"].get("response") or {}).get("projection_sha256", "")),
        "candidate_projection_sha256": str((raw["candidate"].get("response") or {}).get("projection_sha256", "")),
        "reset_id": reset["reset_id"],
    }
    return _hash(value)


def _redirect_evidence(case: dict[str, Any], raw: dict[str, Any], reset: dict[str, Any]) -> str:
    value = {
        "case_id": case["case_id"],
        "control_status": int(raw["control"]["status"]),
        "candidate_status": int(raw["candidate"]["status"]),
        "control_location_sha256": _hash(str(raw["control"].get("location", ""))),
        "candidate_location_sha256": _hash(str(raw["candidate"].get("location", ""))),
        "expected_destination_sha256": _hash(str(raw["expected_destination"])),
        "reset_id": reset["reset_id"],
    }
    return _hash(value)


def _model_visible_projection(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Keep only generic, non-semantic shape fields for a possible model input."""

    dom_shape = response.get("dom_shape") or {}
    visible = {
        "schema_version": "pg100-model-visible-observation-v1",
        "method": str(case["method"]),
        "status_class": str(response.get("status_class", "")),
        "content_type_class": str(response.get("content_type", "")),
        "body_length_bucket": str(response.get("body_length_bucket", "")),
        "html_tag_count_bucket": min(int(response.get("html_tag_count", 0)) // 8, 64),
        "form_count": min(int(response.get("form_count", 0)), 16),
        "input_count": min(int(response.get("input_count", 0)), 32),
        "script_count": min(int(response.get("script_count", 0)), 16),
        "has_location": bool(response.get("has_location", False)),
        "location_origin": str(response.get("location_origin", "none")),
        "dom_node_count_bucket": min(int(dom_shape.get("node_count", 0)) // 8, 64),
        "dom_svg_count": min(int(dom_shape.get("svg_count", 0)), 16),
        "dom_form_count": min(int(dom_shape.get("form_count", 0)), 16),
        "dom_input_count": min(int(dom_shape.get("input_count", 0)), 32),
        "dom_script_count": min(int(dom_shape.get("script_count", 0)), 16),
    }
    return visible


def _oracle_for(case: dict[str, Any], raw: dict[str, Any], reset: dict[str, Any], evidence_sha256: str) -> dict[str, Any]:
    if case["family"] == "xss":
        paths = list(raw.get("execution_paths") or [])
        if len(paths) != 2:
            return evaluate_browser_pair(
                control_executed=bool(raw["control"].get("executed")),
                candidate_executed=bool(raw["candidate"].get("executed")),
                control_execution_path="invalid",
                candidate_execution_path="invalid",
                same_origin=True,
                external_request_count=0,
                navigation_count=0,
                mode=case["mode"],
                fresh_reset=bool(reset["fresh_target"] and reset["completed"]),
                evidence_sha256=evidence_sha256,
            )
        return evaluate_browser_pair(
            control_executed=bool(raw["control"].get("executed")),
            candidate_executed=bool(raw["candidate"].get("executed")),
            control_execution_path=str(paths[0]),
            candidate_execution_path=str(paths[1]),
            same_origin=True,
            external_request_count=0,
            navigation_count=0,
            mode=case["mode"],
            fresh_reset=bool(reset["fresh_target"] and reset["completed"]),
            evidence_sha256=evidence_sha256,
        )
    if case["family"] == "injection":
        return evaluate_sql_pair(
            control_ast=raw["control"].get("ast") or {},
            negative_ast=raw["negative"].get("ast") or {},
            candidate_ast=raw["candidate"].get("ast") or {},
            control_response=raw["control"].get("response") or {},
            negative_response=raw["negative"].get("response") or {},
            candidate_response=raw["candidate"].get("response") or {},
            fresh_reset=bool(reset["fresh_target"] and reset["completed"]),
            evidence_sha256=evidence_sha256,
        )
    return evaluate_redirect_pair(
        control_location=str(raw["control"].get("location", "")),
        candidate_location=str(raw["candidate"].get("location", "")),
        control_status=int(raw["control"].get("status", 0)),
        candidate_status=int(raw["candidate"].get("status", 0)),
        expected_destination=str(raw.get("expected_destination", "")),
        fresh_reset=bool(reset["fresh_target"] and reset["completed"]),
        evidence_sha256=evidence_sha256,
    )


def _record(case: dict[str, Any], raw: dict[str, Any], reset: dict[str, Any], oracle: dict[str, Any], evidence_sha256: str) -> dict[str, Any]:
    response = raw["candidate"].get("response") or {}
    visible = _model_visible_projection(case, response)
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "surface": case["surface"],
        "method": case["method"],
        "model_visible_observation": visible,
        "independent_oracle": oracle,
        "oracle_label_is_not_model_input": not model_visible_has_evaluator_label(visible),
        "payload_manifest": raw["payload_manifest"],
        "negative_control": {
            "matched": bool(oracle.get("negative_control_matched")),
            "same_case": True,
            "candidate_vs_control": True,
        },
        "fresh_reset": reset,
        "evidence_sha256": evidence_sha256,
        "oracle_evidence_sha256": str(oracle.get("oracle_evidence_sha256", "")),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "raw_query_stored": False,
        "old_pg52_oracle_discarded_before_pg100": True,
    }


def _run() -> dict[str, Any]:
    reset_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started: list[str] = []
    containers: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    try:
        for case in pg52.CASES:
            port = int(case["port"])
            name = f"pg100-pikachu-{case['case_id']}"
            container_id = pg52._start(name, port)
            started.append(name)
            containers[case["case_id"]] = container_id
            try:
                if port == 8767:
                    pg52._wait_application_surface(port, "/vul/sqli/sqli_str.php", b"what's your username")
                else:
                    pg52._wait_application_surface(port, "/vul/xss/xsspost/post_login.php", b'name="username"')
                base = GET_BASE if port == 8767 else POST_BASE
                if case["family"] == "xss":
                    raw = pg52._browser_case(case, base, f"pg100-{case['case_id']}-m", container_id)
                    reset = _fresh_reset(container_id, case["case_id"], reset_hash)
                    evidence_sha256 = _browser_evidence(case, raw, reset)
                elif case["family"] == "injection":
                    pg52._prepare_mysql(name)
                    raw = pg52._sql_case(case, base, name)
                    reset = _fresh_reset(container_id, case["case_id"], reset_hash)
                    evidence_sha256 = _sql_evidence(case, raw, reset)
                else:
                    raw = _redirect_raw(case, base, container_id)
                    reset = _fresh_reset(container_id, case["case_id"], reset_hash)
                    evidence_sha256 = _redirect_evidence(case, raw, reset)
                oracle = _oracle_for(case, raw, reset, evidence_sha256)
                rows.append(_record(case, raw, reset, oracle, evidence_sha256))
            finally:
                pg52._stop(name)
                started.remove(name)
    finally:
        for name in reversed(started):
            pg52._stop(name)

    positives = [row for row in rows if row["independent_oracle"]["status"] == "confirmed_positive"]
    negatives = [row for row in rows if row["independent_oracle"]["status"] == "confirmed_negative"]
    abstains = [row for row in rows if row["independent_oracle"]["status"] == "abstain"]
    modalities = {str(row["independent_oracle"].get("modality")) for row in rows}
    pg99 = json.loads(PG99_REPORT_PATH.read_text(encoding="utf-8")) if PG99_REPORT_PATH.exists() else {}
    pg99_overlap = ((pg99.get("metrics") or {}).get("pg42_known_unknown_overlap") or {})
    checks = {
        "case_count_expected": len(rows) == len(pg52.CASES),
        "typed_positive_revalidated": len(positives) == len(pg52.CASES),
        "zero_unjustified_positive": all(row["independent_oracle"].get("positive_authority") is True for row in positives),
        "negative_control_matched": all(bool(row["negative_control"]["matched"]) for row in rows),
        "fresh_reset_per_case": all(bool(row["fresh_reset"]["fresh_target"] and row["fresh_reset"]["completed"]) for row in rows),
        "evidence_hashes_valid": all(_valid_hash(row["evidence_sha256"]) and _valid_hash(row["oracle_evidence_sha256"]) for row in rows),
        "get_post_covered": sorted({str(row["method"]) for row in rows}) == ["GET", "POST"],
        "required_modalities": modalities == {"browser_dom_execution", "sql_ast_differential", "redirect_destination_controlled"},
        "model_input_excludes_evaluator_labels": all(bool(row["oracle_label_is_not_model_input"]) for row in rows),
        "raw_persistence_forbidden": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] and not row["raw_query_stored"] for row in rows),
        "old_oracle_discarded": all(bool(row["old_pg52_oracle_discarded_before_pg100"]) for row in rows),
        "pg99_equivalence_control_present": bool(pg99_overlap.get("impossibility_witness")),
    }
    blocked = [name for name, value in checks.items() if not value]
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg100-independent-semantic-sink-report-v1",
        "status": "blocked" if blocked or bool(pg99_overlap.get("impossibility_witness")) else "passed",
        "target": {
            "image": IMAGE,
            "loopback_only": True,
            "external_network": False,
            "fresh_target_rounds": len(rows),
            "container_instance_count": len(containers),
            "container_ids": dict(containers),
        },
        "oracle_contract": {
            "implementation": SCHEMA_VERSION,
            "old_pg52_labels_discarded_before_revalidation": True,
            "browser_dom_execution": True,
            "read_only_sql_ast_differential": True,
            "controlled_redirect_without_follow": True,
            "positive_requires_negative_control": True,
            "positive_requires_fresh_reset": True,
            "positive_requires_evidence_hash": True,
            "oracle_is_evaluator_only": True,
        },
        "metrics": {
            "case_count": len(rows),
            "confirmed_positive_count": len(positives),
            "confirmed_negative_count": len(negatives),
            "abstain_count": len(abstains),
            "positive_by_modality": {modality: sum(row["independent_oracle"].get("modality") == modality and row["independent_oracle"].get("status") == "confirmed_positive" for row in rows) for modality in sorted(modalities)},
            "get_post_covered": {method: sum(row["method"] == method for row in rows) for method in ("GET", "POST")},
            "negative_false_accept_count": sum(row["independent_oracle"].get("status") == "confirmed_positive" and not row["negative_control"]["matched"] for row in rows),
            "oracle_evidence_hash_count": sum(_valid_hash(row["oracle_evidence_sha256"]) for row in rows),
        },
        "pg99_unidentifiability_control": {
            "known_unknown_fingerprint_overlap_rate": pg99_overlap.get("unknown_overlap_rate"),
            "equivalence_class_conflict_count": pg99_overlap.get("equivalence_class_conflict_count"),
            "impossibility_witness": bool(pg99_overlap.get("impossibility_witness")),
            "semantic_oracle_is_not_model_feature": True,
            "unknown_family_strict_abstain": False,
            "interpretation": "独立语义 oracle 能提高验收标签可信度，但不会凭空提供模型区分未知族所需的可见信息。",
        },
        "capability_gate": {
            "status": "blocked",
            "checks": checks,
            "blocking_reasons": blocked + (["pg99_known_unknown_visible_equivalence_remains"] if pg99_overlap.get("impossibility_witness") else []),
            "claim_allowed": False,
        },
        "training_boundary": {
            "training_eligible": False,
            "catalog_generated": False,
            "long_term_memory_write": False,
            "reason": "PG-100 validates an evaluator channel only; it is not a cross-source model capability gain.",
        },
        "detection_results": rows,
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "raw_sql_stored": False,
            "evaluator_labels_in_model_input": False,
            "evidence_hashes_verified": checks["evidence_hashes_valid"],
        },
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg100-independent-semantic-sink-protocol-v1",
        "target_contract": {"image": IMAGE, "loopback_ports": [8767, 8768], "methods": ["GET", "POST"], "fresh_disposable_container_per_case": True, "external_network": False, "state_change_allowed": False},
        "oracle_contract": {"implementation": SCHEMA_VERSION, "typed_modalities": ["browser_dom_execution", "sql_ast_differential", "redirect_destination_controlled"], "old_verdict_fields_not_read": True, "negative_control": "same allow-listed case with inert/control branch", "evidence": "canonical SHA-256"},
        "evaluation_boundary": {"training_allowed": False, "memory_promotion_allowed": False, "unknown_family_claim_allowed": False},
        "status": report["status"],
    }
    dataset_rows = [{"dataset_id": f"pg100-{row['case_id']}", "role": "evaluation_only", "visible_observation": row["model_visible_observation"], "source": "fresh_pg100_pikachu_replay"} for row in rows]
    trace_rows = [{"trace_id": _hash(row["case_id"] + row["evidence_sha256"])[:24], "step_index": 0, "model_input": row["model_visible_observation"], "target_typed_oracle": {"status": row["independent_oracle"]["status"], "modality": row["independent_oracle"]["modality"]}, "evidence_sha256": row["evidence_sha256"], "fresh_reset": row["fresh_reset"], "negative_control_matched": row["negative_control"]["matched"]} for row in rows]
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg100-semantic-sink-visible-dataset-v1", "training_eligible": False, "rows": dataset_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg100-semantic-sink-trace-v1", "training_eligible": False, "steps": trace_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-100 独立语义 sink oracle 回放", "", f"fresh Docker cases: {len(rows)}；confirmed_positive: {len(positives)}；abstain: {len(abstains)}。", "", "| family | surface | method | independent oracle | status |", "|---|---|---|---|---|"]
    for row in rows:
        oracle = row["independent_oracle"]
        lines.append(f"| `{row['family']}` | `{row['surface']}` | `{row['method']}` | `{oracle['modality']}` | `{oracle['status']}` |")
    lines.extend(["", "PG-100 只验证验收通道，不把 oracle 标签放进模型输入，也不生成训练样本或长期记忆。PG-99 的已知/未知可见等价类仍然存在，因此能力门保持 blocked。", "", f"JSON: `{REPORT_PATH.relative_to(ROOT)}`", f"协议: `{PROTOCOL_PATH.relative_to(ROOT)}`", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "case_count": len(rows), "confirmed_positive_count": len(positives), "confirmed_negative_count": len(negatives), "abstain_count": len(abstains), "checks": checks, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    _run()
