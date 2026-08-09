"""Run the PG-282 abstract-plan/evaluator binding contract offline.

This runner consumes only the PG-281 abstract plan rows and the recorded
remote Docker availability projection.  It does not open a socket, start a
container, send a request, or manufacture evaluator evidence.  When the
remote daemon becomes available, the same binding function can be fed a
target-side manifest and a typed-evidence projection.
"""

from __future__ import annotations

import hashlib
import json
import re
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
REPORT = RESEARCH / "pg282_evaluator_binding_report_v1.json"
TRACE = RESEARCH / "pg282_evaluator_binding_trace_v1.json"
PROTOCOL = RESEARCH / "pg282_evaluator_binding_protocol_v1.json"
MARKDOWN = RESEARCH / "pg282_evaluator_binding_report_v1.md"


def _sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _surface_path(group_id: str, method: str) -> str:
    match = re.search(r"(/vul/[^:]+):" + re.escape(method) + r":", group_id)
    if match:
        return match.group(1)
    # Synthetic PG-266 surfaces have no public route.  Keep them visibly
    # abstract rather than pretending that a real endpoint was observed.
    return "/__abstract__/" + _sha(group_id)[:20]


def _surface_from_row(row: dict[str, Any]) -> dict[str, Any]:
    method = str(row.get("method", "GET")).upper()
    group_id = str(row.get("group_id", row.get("record_id", "unknown")))
    channel = "query" if method == "GET" else "form"
    family = str(row.get("family", "other"))
    return {
        "surface_id": "pg282:" + _sha({"group_id": group_id, "method": method})[:24],
        "path": _surface_path(group_id, method),
        "method": method,
        "channel": channel,
        "field_count": max(1, int(next((token.split("=", 1)[1] for token in row.get("context_tokens", []) if str(token).startswith("field_count=")), "1"))),
        "authorization": "operator_allowlisted_remote_docker",
        "typed_evaluator": "target_specific_evaluator_required",
        "fresh_reset_contract": False,
        "reference_contract": False,
        "negative_contract": False,
        "family_hint": family,
    }


def _abstract_plan(row: dict[str, Any]) -> dict[str, Any]:
    target = dict(row.get("target") or {})
    return {
        "probe_class": str(target.get("probe_class", "other")),
        "channel": str(target.get("channel", "unknown")),
        "encoding": str(target.get("encoding", "unknown")),
        "final_action": str(target.get("final_action", "abstain")),
        "safe_to_send": bool(target.get("safe_to_send", False)),
        "oracle_required": bool(target.get("oracle_required", True)),
    }


def _summarize(binding: dict[str, Any], row_id: str, split: str) -> dict[str, Any]:
    return {
        "record_id": row_id,
        "split": split,
        "status": binding["status"],
        "decision": binding["decision"],
        "surface_id": binding["surface"]["surface_id"],
        "method": binding["wire_shape"]["method"],
        "channel": binding["wire_shape"]["channel"],
        "encoding": binding["wire_shape"]["encoding"],
        "hard_negative": binding["hard_negative"],
        "reasons": binding["reasons"],
        "checks": binding["checks"],
        "binding_evidence_sha256": binding["binding_evidence_sha256"],
        "literal_payload_stored": False,
        "raw_response_stored": False,
    }


def main() -> None:
    from app.pg282_evaluator_binding import bind_abstract_plan

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    hard = json.loads(HARD.read_text(encoding="utf-8"))
    remote_probe = json.loads(REMOTE_PROBE.read_text(encoding="utf-8"))
    rows = list(data.get("records") or [])
    hard_rows = list(hard.get("records") or [])
    results: list[dict[str, Any]] = []
    for row in rows:
        binding = bind_abstract_plan(_abstract_plan(row), _surface_from_row(row), remote_probe=remote_probe, hard_negative=False)
        results.append(_summarize(binding, str(row.get("record_id", "")), str(row.get("split", "unknown"))))
    for row in hard_rows:
        binding = bind_abstract_plan(_abstract_plan(row), _surface_from_row(row), remote_probe=remote_probe, hard_negative=True)
        results.append(_summarize(binding, str(row.get("hard_negative_id", "")), "hard_negative"))

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    split_counts: dict[str, int] = {}
    for result in results:
        split_counts[result["split"]] = split_counts.get(result["split"], 0) + 1
    hard_negative_results = [result for result in results if result["hard_negative"]]
    checks = {
        "row_count_matches_source": len(results) == len(rows) + len(hard_rows),
        "remote_probe_recorded": remote_probe.get("status") in {"available", "unavailable", "unreachable", "ssh_unavailable"},
        "remote_unavailable_is_blocked": remote_probe.get("status") != "available" and counts.get("confirmed_positive", 0) == 0,
        "hard_negative_all_abstain": all(result["status"] == "abstain" for result in hard_negative_results),
        "no_literal_payload_stored": all(not result["literal_payload_stored"] for result in results),
        "no_raw_response_stored": all(not result["raw_response_stored"] for result in results),
        "training_promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg282-evaluator-only-binding-v1",
        "schema_version": "pg282-evaluator-binding-report-v1",
        "status": "completed_offline_pg282_binding_contract",
        "source": {
            "pg281_dataset": str(DATASET.relative_to(ROOT)),
            "pg281_dataset_sha256": data.get("dataset_sha256", ""),
            "pg281_hard_negative": str(HARD.relative_to(ROOT)),
            "pg281_hard_negative_sha256": hard.get("dataset_sha256", ""),
            "remote_probe": str(REMOTE_PROBE.relative_to(ROOT)),
            "remote_probe_evidence_sha256": remote_probe.get("evidence_sha256", ""),
            "remote_docker_status": remote_probe.get("status", "unknown"),
            "network_calls": 0,
            "live_replay": False,
        },
        "counts": {"total": len(results), "source_records": len(rows), "hard_negative": len(hard_rows), "by_status": counts, "by_split": split_counts},
        "checks": checks,
        "binding_contract": {
            "model_output": ["probe_class", "channel", "encoding", "final_action", "safe_to_send"],
            "runtime_binding": ["method", "origin_relative_path", "field_count", "authorized_surface_id"],
            "confirmation_requirements": ["typed_effect", "negative_control_clean", "fresh_reset", "reference_agreement", "replay_consistent", "evidence_hash", "non_destructive"],
            "literal_payload_generation": False,
            "raw_response_storage": False,
            "vulnerability_claim_allowed": False,
        },
        "results": results,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "offline binding contract only; no target evaluator or live Docker replay"},
        "formal_conclusion": "PG-282 验证了 abstract plan→authorized surface 的 fail-closed 绑定和族外 hard-negative 拒答；远程 Docker 不可用时不生成 real application gold，也不宣称 payload 成功。",
    }
    report["report_sha256"] = _sha(report)
    trace = {
        "schema_version": "pg282-evaluator-binding-trace-v1",
        "report_sha256": report["report_sha256"],
        "source_dataset_sha256": data.get("dataset_sha256", ""),
        "hard_negative_sha256": hard.get("dataset_sha256", ""),
        "remote_probe_status": remote_probe.get("status", "unknown"),
        "status_counts": counts,
        "training_eligible": False,
        "memory_write": False,
        "literal_payload_in_context": False,
        "live_send": False,
    }
    trace["trace_sha256"] = _sha(trace)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg282-evaluator-binding-protocol-v1",
        "scope": "authorized remote Docker only; fixed SSH host; private container tunnel; evaluator-only confirmation",
        "offline_mode": True,
        "remote_docker_required_for_live_replay": True,
        "commands_allowed": "inherited PG-280 read-only adapter; no docker run/restart/rm",
        "hard_negative_policy": "evaluation-only and forced abstain",
        "promotion_gate": "all typed evidence + fresh reset + reference/negative/replay agreement + evidence hash",
        "report_sha256": report["report_sha256"],
        "next_experiment": "当远程 Docker 与目标 evaluator 可用时，替换 offline projection 为真实 GET/POST evaluator evidence，再比较 AI/reference/negative wire 结构。",
    }
    protocol["protocol_sha256"] = _sha(protocol)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join([
        "# PG-282 evaluator-only binding",
        "",
        f"status={report['status']}",
        f"remote Docker={remote_probe.get('status', 'unknown')}",
        f"rows={len(results)} · confirmed_positive={counts.get('confirmed_positive', 0)} · hard-negative abstain={sum(result['status'] == 'abstain' for result in hard_negative_results)}/{len(hard_negative_results)}",
        "literal payload generation=false · live replay=false · promotion=false",
        "",
    ]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "remote_docker": remote_probe.get("status"), "counts": report["counts"], "checks": checks, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
