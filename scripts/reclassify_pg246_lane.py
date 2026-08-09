"""Repair PG-246 retention lanes from the already verified episode trace.

PG-246's first run had correct DOM/reference/replay evidence but marked the
positive replay rows as quarantine because ``negative_clean`` was incorrectly
used to mean ``negative_control_confirmed``.  This audit-only migration
reconstructs the row contract from the immutable episode lineage, recomputes
the lane with the canonical funnel, and appends a repair note.  It never
replays the network and never adds raw payload or response material.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg230_next_token_quality_funnel import quality_lane  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
TRACE = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_trace_v1.json"
REPORT = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_report_v1.json"
PROTOCOL = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_protocol_v1.json"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _row_from_episode(old: Mapping[str, Any], episode: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    expected = str(episode["expected"])
    method = str(episode["method"])
    positive = expected == "positive"
    replay = role == "replay"
    counterfactual = role == "counterfactual"
    if counterfactual:
        # POST-405 cases never sent an executable candidate; keep their
        # counterfactual as an explicit abstention rather than pretending a
        # candidate was sent.
        candidate_sent = method == "GET"
        typed_effect = False
        oracle_available = method == "GET"
        negative_control_confirmed = True
        failure_signature = "counterfactual_candidate_no_effect"
        failure_stage = "typed_oracle"
        attribution = "none"
        next_step = "retry_candidate" if positive else "abstain"
        parent_record_id = None
        history_len = 0
    elif replay:
        replay_info = dict(episode.get("replay") or {})
        replay_effect = bool(replay_info.get("marker_observed"))
        replay_match = bool(replay_info.get("expected_match"))
        candidate_sent = method == "GET"
        typed_effect = bool(positive and replay_effect)
        oracle_available = method == "GET"
        negative_control_confirmed = not positive
        failure_signature = "typed_effect" if typed_effect else "counterfactual_candidate_no_effect" if replay_match else "replay_mismatch"
        failure_stage = "replay"
        attribution = "none" if replay_match else "environment"
        next_step = "abstain" if replay_match else "inspect_environment"
        parent_record_id = str(episode["record_id"])
        history_len = 1
    else:
        typed = dict(episode.get("typed_oracle") or {})
        candidate_sent = bool((episode.get("ai") or {}).get("sent", False))
        ai_marker = bool(((episode.get("ai") or {}).get("response") or {}).get("marker_observed", False))
        ref_marker = bool(((episode.get("reference") or {}).get("response") or {}).get("marker_observed", False))
        typed_effect = bool(typed.get("confirmed_positive"))
        oracle_available = method == "GET"
        negative_control_confirmed = not positive
        failure_signature = str(episode.get("failure_signature", "candidate_no_effect"))
        failure_stage = str(episode.get("failure_stage", "typed_oracle"))
        attribution = str(episode.get("failure_is_model_or_environment", "none"))
        next_step = "abstain" if typed_effect or expected != "positive" else "recheck_oracle"
        parent_record_id = str(episode["counterfactual_record_id"])
        history_len = 1
        # Keep the independently observed agreement; it is not inferred from
        # the lane being selected.
        candidate_reference_agreement = bool(ai_marker == ref_marker)
    if counterfactual or replay:
        candidate_reference_agreement = True
    row = {
        "source": str(old.get("source", "pg246_vulnerableapp_source_independent")),
        "seed": int(old.get("seed", episode.get("seed", 0)) or 0),
        "surface_role": str(old.get("surface_role", episode.get("surface_role", "xss_dom_surface"))),
        "method": method,
        "field_count": 1,
        "status_class": "2xx" if method == "GET" else "4xx",
        "history_len": history_len,
        "candidate_sent": candidate_sent,
        "oracle_available": oracle_available,
        "typed_effect_confirmed": typed_effect,
        "typed_effect_observed": typed_effect,
        "candidate_reference_agreement": candidate_reference_agreement,
        "negative_clean": True,
        "negative_control_confirmed": negative_control_confirmed,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "reset_not_attempted": False,
        "failure_signature": failure_signature,
        "failure_stage": failure_stage,
        "failure_is_model_or_environment": attribution,
        "next_step": next_step,
        "previous_feedback": "result_verified" if typed_effect else "failure_adjusted",
        "candidate_result_present": typed_effect,
        "model_claimed_positive": bool(candidate_sent and typed_effect),
        "model_abstained": not candidate_sent,
        "abstention_required": not oracle_available or not candidate_sent,
        "evidence_hash": str(old.get("source_evidence_hash", "")),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "payload_grounded_eligible": typed_effect,
    }
    prepared = prepare_feedback_record(row)
    # Preserve the immutable lineage/implementation annotations from the first
    # run while replacing only the lane-dependent projection.
    for key in ("failure_stage", "failure_is_model_or_environment", "route_template_hash", "source_implementation", "generator_id", "implementation_holdout", "candidate_reference_replay_match", "repair_delta_projection", "repair_outcome", "repair_replay_count"):
        if key in old:
            prepared[key] = old[key]
    prepared["parent_record_id"] = parent_record_id
    prepared["record_id"] = str(old.get("record_id", ""))
    prepared["lane_hint"] = "gold" if typed_effect else "hard_negative"
    lane, reasons = quality_lane(row)
    prepared["lane"] = lane
    prepared["lane_index"] = {"gold": 0, "hard_negative": 1, "silver": 2, "quarantine": 3, "reject": 4}[lane]
    prepared["quality_reasons"] = reasons
    prepared["reclassification_parent_hash"] = _hash(old)
    prepared["record_hash"] = _hash(prepared)
    return prepared


def main() -> int:
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    old_records = {str(row["record_id"]): row for row in dataset.get("records", [])}
    rebuilt: dict[str, dict[str, Any]] = {}
    for episode in trace.get("episodes", []):
        counter_id = str(episode["counterfactual_record_id"])
        main_id = str(episode["record_id"])
        replay_id = str((episode.get("replay") or {}).get("record_id", ""))
        for record_id, role in ((counter_id, "counterfactual"), (main_id, "main"), (replay_id, "replay")):
            if record_id and record_id in old_records:
                rebuilt[record_id] = _row_from_episode(old_records[record_id], episode, role=role)
    if set(rebuilt) != set(old_records):
        missing = sorted(set(old_records) - set(rebuilt))
        raise RuntimeError(f"PG-246 lane repair refuses incomplete lineage: {missing[:5]}")
    records = [rebuilt[str(row["record_id"])] for row in dataset["records"]]
    counts = {
        "records": len(records),
        "gold": sum(int(row["lane"] == "gold") for row in records),
        "hard_negative": sum(int(row["lane"] == "hard_negative") for row in records),
        "silver": sum(int(row["lane"] == "silver") for row in records),
        "quarantine": sum(int(row["lane"] == "quarantine") for row in records),
        "reject": sum(int(row["lane"] == "reject") for row in records),
    }
    dataset["records"] = records
    dataset["counts"] = counts
    dataset["data_repair"] = {
        "repair_id": "pg246-lane-reclassification-v1",
        "reason": "positive fresh-replay rows were incorrectly treated as negative_control_confirmed",
        "network_replay_performed": False,
        "oracle_evidence_changed": False,
        "parent_dataset_sha256": str(dataset.get("dataset_sha256", "")),
        "lane_rule": "negative_control_confirmed is distinct from negative_clean; typed positive rows may be gold",
    }
    dataset.pop("dataset_sha256", None)
    dataset["dataset_sha256"] = _hash(dataset)
    trace["records"] = records
    trace["data_repair"] = dataset["data_repair"]
    trace["training_eligible"] = bool(counts["gold"] > 0 and counts["hard_negative"] > 0)
    report["counts"]["gold_record_count"] = counts["gold"]
    report["counts"]["hard_negative_record_count"] = counts["hard_negative"]
    report["counts"]["quarantine_record_count"] = counts["quarantine"]
    report["promotion"]["training_eligible"] = bool(counts["gold"] > 0 and counts["hard_negative"] > 0)
    report["data_repair"] = dataset["data_repair"]
    report.pop("report_sha256", None)
    report["report_sha256"] = _hash(report)
    protocol["lane_reclassification_audit"] = {
        "performed": True,
        "script": "scripts/reclassify_pg246_lane.py",
        "network_replay_performed": False,
        "oracle_evidence_changed": False,
        "training_rows_recomputed_after_repair": True,
    }
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = _hash(protocol)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(REPORT, report)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": "completed_pg246_lane_reclassification", "counts": counts, "dataset_sha256": dataset["dataset_sha256"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
