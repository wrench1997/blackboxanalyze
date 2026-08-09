"""Run PG-284 evaluator-contract validation without network or Docker.

This is an integration dry run.  It exercises the exact fail-closed shape
that a real authorized evaluator will submit later, while recording zero live
requests and zero application gold rows.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg281_payload_policy_dataset_v1.json"
HARD = RESEARCH / "pg281_payload_policy_hard_negative_v1.json"
REMOTE_PROBE = RESEARCH / "pg280_remote_docker_probe_v2.json"
REPORT = RESEARCH / "pg284_evaluator_contract_report_v1.json"
TRACE = RESEARCH / "pg284_evaluator_contract_trace_v1.json"
PROTOCOL = RESEARCH / "pg284_evaluator_contract_protocol_v1.json"
MARKDOWN = RESEARCH / "pg284_evaluator_contract_report_v1.md"


def _sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _surface(row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(row.get("record_id") or row.get("hard_negative_id") or "unknown")
    method = str(row.get("method", "GET")).upper()
    path = "/__abstract__/" + _sha(source_id)[:20]
    return {
        "surface_id": "pg284:" + _sha({"source": source_id, "method": method})[:24],
        "method": method if method in {"GET", "POST"} else "GET",
        "path": path,
        "channel": "query" if method == "GET" else "form",
        "field_count": 2,
        "authorization": "operator_allowlisted_remote_docker",
        "source_attestation_sha256": _sha({"source": source_id, "family": str(row.get("family", "other"))}),
        "evaluator_kind": "target_specific_pending",
    }


def _projection(seed: str, marker: str = "none") -> dict[str, Any]:
    return {"status_class": "unknown", "shape_sha256": _sha({"seed": seed, "marker": marker}), "redirect_hops": 0, "backend_observed": False, "effect_marker": marker}


def _evidence() -> dict[str, Any]:
    evidence = {"effect_type": "result_shape", "typed_effect_confirmed": False, "negative_control_clean": False, "reference_agreement": False, "replay_consistent": False, "non_destructive": True, "evaluator_id": "pending"}
    evidence["evidence_sha256"] = _sha(evidence)
    return evidence


def main() -> None:
    from app.pg284_evaluator_contract import evaluate_typed_replay

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    hard_data = json.loads(HARD.read_text(encoding="utf-8"))
    remote_probe = json.loads(REMOTE_PROBE.read_text(encoding="utf-8"))
    rows = list(data.get("records") or [])
    hard_rows = list(hard_data.get("records") or [])
    results: list[dict[str, Any]] = []
    for row in [*rows, *hard_rows]:
        source_id = str(row.get("record_id") or row.get("hard_negative_id") or "unknown")
        hard_negative = bool(row in hard_rows or row.get("hard_negative", False))
        seed = _sha(source_id)
        result = evaluate_typed_replay(
            surface=_surface(row),
            reset={"reset_id": "offline-pending", "fresh_target": False, "container_recreated": False, "container_restart_used": False, "volume_mount_count": -1, "database_health_gate": "unknown", "state_change_allowed": False},
            reference=_projection(seed, "reference"),
            negative=_projection(seed, "none"),
            candidate=_projection(seed, "candidate"),
            replay=_projection(seed, "candidate"),
            typed_evidence=_evidence(),
            remote_probe=remote_probe,
            hard_negative=hard_negative,
        )
        results.append({
            "record_id": source_id,
            "split": "hard_negative" if hard_negative else str(row.get("split", "unknown")),
            "status": result["status"],
            "decision": result["decision"],
            "hard_negative": hard_negative,
            "checks": result["checks"],
            "reasons": result["reasons"],
            "evidence_projection_sha256": result["evidence_projection_sha256"],
            "typed_effect_confirmed": result["typed_effect_confirmed"],
            "literal_payload_stored": False,
            "raw_response_stored": False,
        })
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    checks = {
        "source_rows_bound": len(results) == len(rows) + len(hard_rows),
        "remote_probe_recorded": remote_probe.get("status") in {"available", "unavailable", "unreachable", "ssh_unavailable"},
        "remote_unavailable_no_confirmed_effect": remote_probe.get("status") != "available" and counts.get("confirmed_effect", 0) == 0,
        "hard_negative_blocked": all(item["status"] == "blocked" and item["hard_negative"] for item in results if item["hard_negative"]),
        "no_live_requests": True,
        "no_literal_payload": all(not item["literal_payload_stored"] for item in results),
        "no_raw_response": all(not item["raw_response_stored"] for item in results),
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg284-typed-evaluator-contract-v1",
        "schema_version": "pg284-evaluator-contract-report-v1",
        "status": "completed_offline_pg284_evaluator_contract",
        "source": {"pg281_dataset": str(DATASET.relative_to(ROOT).as_posix()), "pg281_dataset_sha256": data.get("dataset_sha256", ""), "pg281_hard_negative": str(HARD.relative_to(ROOT).as_posix()), "pg281_hard_negative_sha256": hard_data.get("dataset_sha256", ""), "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()), "remote_docker_status": remote_probe.get("status", "unknown"), "network_calls": 0, "live_replay": False, "real_application_gold_rows": 0},
        "counts": {"total": len(results), "source_records": len(rows), "hard_negative": len(hard_rows), "by_status": counts},
        "checks": checks,
        "results": results,
        "contract": {"typed_effect_required": True, "negative_control_required": True, "fresh_reset_required": True, "reference_agreement_required": True, "replay_consistency_required": True, "evidence_hash_required": True, "non_destructive_required": True, "literal_payload_generation": False, "vulnerability_claim_allowed": False},
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks},
        "scientific_gate": {"status": "blocked", "reasons": ["remote Docker/evaluator unavailable", "offline contract fixture only", "no fresh target replay or payload success"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "offline contract validation only"},
    }
    report["report_sha256"] = _sha(report)
    trace = {"schema_version": "pg284-evaluator-contract-trace-v1", "report_sha256": report["report_sha256"], "source_dataset_sha256": data.get("dataset_sha256", ""), "hard_negative_sha256": hard_data.get("dataset_sha256", ""), "status_counts": counts, "live_replay": False, "training_eligible": False, "memory_write": False}
    trace["trace_sha256"] = _sha(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg284-evaluator-contract-protocol-v1", "remote_host": "112.111.7.91:60228", "remote_a800_gpu0_only": True, "read_only_remote_probe": True, "no_payload_generation": True, "no_network_in_offline_mode": True, "confirmation_gate": ["typed_effect", "negative_control", "fresh_reset", "reference_agreement", "replay_consistent", "evidence_hash", "non_destructive"], "report_sha256": report["report_sha256"], "next_experiment": "PG-284-live: bind this contract to an available target evaluator and real fresh GET/POST replay."}
    protocol["protocol_sha256"] = _sha(protocol)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-284 typed evaluator contract", "", f"engineering gate={report['engineering_gate']['status']}", "scientific gate=blocked", f"remote Docker={remote_probe.get('status', 'unknown')}", f"rows={len(results)} · confirmed_effect={counts.get('confirmed_effect', 0)} · blocked={counts.get('blocked', 0)}", "live replay=false · promotion=false", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "checks": checks, "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
