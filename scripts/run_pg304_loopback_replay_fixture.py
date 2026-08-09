"""Exercise PG-304 with bounded loopback evaluator projections only."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg284_evaluator_contract import sha256_json  # noqa: E402
from app.pg302_symbolic_assembly import symbolic_target_for_context  # noqa: E402
from app.pg304_loopback_replay import evaluate_loopback_batch  # noqa: E402


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg304_loopback_replay_fixture_report_v1.json"
TRACE = RESEARCH / "pg304_loopback_replay_fixture_trace_v1.json"
PROTOCOL = RESEARCH / "pg304_loopback_replay_fixture_protocol_v1.json"


def verify() -> datetime:
    if os.environ.get("PG304_LOCAL_EVAL") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-304 requires PG304_LOCAL_EVAL=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-304 loopback fixture is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    return now


def _context(method: str, field: str, encoding: str) -> list[str]:
    return [
        f"surface_method={method}", f"surface_field_role={field}", f"surface_encoding={encoding}",
        "typed_available=1", "feedback_state=observable_progress", "replay_ready=1", "evidence_present=1", "negative_control=1", "fresh_reset=1", "history_action=observe", "failure_class=none", "step_budget=present",
    ]


def _shape(name: str) -> str:
    return sha256_json({"fixture_shape": name})


def _surface(surface_id: str, method: str) -> dict:
    return {"surface_id": surface_id, "method": method, "path": f"/{surface_id}", "channel": "query" if method == "GET" else "form", "field_count": 1, "authorization": "operator_allowlisted_remote_docker", "source_attestation_sha256": sha256_json({"surface_id": surface_id, "fixture": True}), "evaluator_kind": "fixture_typed_projection"}


def _reset(reset_id: str) -> dict:
    return {"reset_id": reset_id, "fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "database_health_gate": "healthy", "state_change_allowed": False}


def _projection(name: str, *, backend: bool = False, marker: str = "none") -> dict:
    return {"status_class": "2xx", "shape_sha256": _shape(name), "redirect_hops": 0, "backend_observed": backend, "effect_marker": marker}


def _episode(surface_id: str, method: str, field: str, encoding: str, *, negative: bool = False, hard_negative: bool = False) -> dict:
    context = _context(method, field, encoding)
    model_plan = symbolic_target_for_context(context)
    unsigned_evidence = {"effect_type": "result_shape", "typed_effect_confirmed": not negative, "negative_control_clean": not negative, "reference_agreement": not negative, "replay_consistent": not negative, "non_destructive": True, "evaluator_id": "fixture-evaluator"}
    evidence = {**unsigned_evidence, "evidence_sha256": sha256_json(unsigned_evidence)}
    return {"context_tokens": context, "plan_tokens": model_plan, "surface": _surface(surface_id, method), "reset": _reset(f"reset-{surface_id}"), "reference": _projection(f"{surface_id}-reference", backend=False), "negative": _projection(f"{surface_id}-negative", backend=False), "candidate": _projection(f"{surface_id}-candidate", backend=not negative, marker="typed" if not negative else "none"), "replay": _projection(f"{surface_id}-candidate", backend=not negative, marker="typed" if not negative else "none"), "typed_evidence": evidence, "remote_probe": {"status": "available", "loopback_only": True, "external_network": False, "container_id": f"fixture-{surface_id}", "source_attested": True}, "hard_negative": hard_negative}


def main() -> None:
    verify()
    episodes = [_episode("pg304-get", "GET", "query_param", "url_percent"), _episode("pg304-post", "POST", "form_field", "form_urlencoded"), _episode("pg304-negative", "GET", "query_param", "url_percent", negative=True, hard_negative=True)]
    result = evaluate_loopback_batch(episodes)
    checks = dict(result.get("checks") or {})
    checks["positive_fixture_confirmed"] = result["metrics"]["typed_positive_count"] == 2
    checks["negative_fixture_blocked"] = result["episodes"][2]["typed_effect_confirmed"] is False
    checks["no_training_promotion"] = result["metrics"]["training_eligible_count"] == 0 and result["metrics"]["memory_promotion_allowed_count"] == 0
    result["checks"] = checks
    contract_passed = bool(checks.get("loopback_only") and checks.get("external_network_disabled") and checks.get("get_post_pair") and checks.get("positive_fixture_confirmed") and checks.get("negative_fixture_blocked") and checks.get("no_training_promotion") and checks.get("wire_emission") is False and checks.get("raw_material_stored") is False)
    result["engineering_gate"] = {"status": "passed" if contract_passed else "blocked", "claim_allowed": False}
    result["source"] = {"fixture_only": True, "actual_docker_contacted": False, "external_network": False, "literal_payload": False}
    result["report_sha256"] = sha256_json(result)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg304-loopback-replay-fixture-trace-v1", "report_sha256": result["report_sha256"], "fixture_only": True, "actual_docker_contacted": False, "training_eligible": False, "memory_write": False, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg304-loopback-replay-fixture-v1", "schema_version": "pg304-loopback-replay-fixture-protocol-v1", "execution_mode": "local_morning_fixture_only", "loopback_only": True, "actual_docker_contacted": False, "get_post_pair_required": True, "fresh_reset_required": True, "negative_control_required": True, "typed_evidence_sha256_required": True, "wire_emission": False, "promotion_blocked": True, "report_sha256": result["report_sha256"], "next_experiment": "PG-305: replace fixture projections with an explicitly authorized local Docker adapter, then re-run fresh reset/replay audit."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"], "checks": result["checks"], "engineering_gate": result["engineering_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
