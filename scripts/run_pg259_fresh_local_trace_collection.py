"""PG-259: collect additional fresh local SQL/XSS process traces.

This runner deliberately composes the already-audited loopback runners rather
than inventing synthetic labels.  Each child experiment keeps its own AI
decision, independent reference, matched negative and fresh-container gate;
this script only stores bounded projections and converts them to the common
feedback-token record used by the capacity experiments.

The raw request values and response bodies remain stdout-only in the child
runners.  The resulting PG-259 dataset contains hashes, token trajectories and
typed oracle fields, never a reusable raw payload catalog.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
SQL_REPORT = RESEARCH / "pg259_fresh_sql_replay_report_v1.json"
SQL_TRACE = RESEARCH / "pg259_fresh_sql_replay_trace_v1.json"
SQL_PROTOCOL = RESEARCH / "pg259_fresh_sql_replay_protocol_v1.json"
SQL_MARKDOWN = RESEARCH / "pg259_fresh_sql_replay_report_v1.md"
XSS_REPORT = RESEARCH / "pg259_fresh_xss_replay_report_v1.json"
XSS_DATASET = RESEARCH / "pg259_fresh_xss_replay_dataset_v1.json"
XSS_TRACE = RESEARCH / "pg259_fresh_xss_replay_trace_v1.json"
XSS_PROTOCOL = RESEARCH / "pg259_fresh_xss_replay_protocol_v1.json"
XSS_MARKDOWN = RESEARCH / "pg259_fresh_xss_replay_report_v1.md"
BOOL_REPORT = RESEARCH / "pg259_fresh_boolean_replay_report_v1.json"
BOOL_DATASET = RESEARCH / "pg259_fresh_boolean_replay_dataset_v1.json"
BOOL_TRACE = RESEARCH / "pg259_fresh_boolean_replay_trace_v1.json"
BOOL_PROTOCOL = RESEARCH / "pg259_fresh_boolean_replay_protocol_v1.json"
BOOL_MARKDOWN = RESEARCH / "pg259_fresh_boolean_replay_report_v1.md"
WIDE_REPORT = RESEARCH / "pg259_fresh_widebyte_replay_report_v1.json"
WIDE_TRACE = RESEARCH / "pg259_fresh_widebyte_replay_trace_v1.json"
WIDE_PROTOCOL = RESEARCH / "pg259_fresh_widebyte_replay_protocol_v1.json"
WIDE_MARKDOWN = RESEARCH / "pg259_fresh_widebyte_replay_report_v1.md"
REPORT = RESEARCH / "pg259_fresh_local_trace_collection_report_v1.json"
DATASET = RESEARCH / "pg259_fresh_local_trace_collection_dataset_v1.json"
TRACE = RESEARCH / "pg259_fresh_local_trace_collection_trace_v1.json"
PROTOCOL = RESEARCH / "pg259_fresh_local_trace_collection_protocol_v1.json"
MARKDOWN = RESEARCH / "pg259_fresh_local_trace_collection_report_v1.md"

# One fresh seed per surface is intentionally the first tranche.  The next
# collection pass can append more seeds after this report exposes which class
# still lacks support; no quota is silently padded with synthetic rows.
SQL_SEEDS = (25901,)
XSS_SEEDS = (25911,)
BOOLEAN_SEEDS = (25921, 25922)
WIDEBYTE_SEEDS = (25931, 25932)


def _load(filename: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_sql() -> dict[str, Any]:
    runner = _load("run_pg255_pikachu_fixed_sql_pg254_replay.py")
    runner.SEEDS = SQL_SEEDS
    runner.REPORT, runner.TRACE, runner.PROTOCOL, runner.MARKDOWN = SQL_REPORT, SQL_TRACE, SQL_PROTOCOL, SQL_MARKDOWN
    runner.WIRE_LOG.clear()
    runner.main()
    return json.loads(SQL_REPORT.read_text(encoding="utf-8-sig"))


def _run_xss() -> tuple[dict[str, Any], dict[str, Any]]:
    runner = _load("run_pg242_pikachu_xss_dom_acceptance.py")
    runner.SEEDS = XSS_SEEDS
    runner.REPORT, runner.DATASET, runner.TRACE, runner.PROTOCOL, runner.MARKDOWN = XSS_REPORT, XSS_DATASET, XSS_TRACE, XSS_PROTOCOL, XSS_MARKDOWN
    report_code = runner.main()
    if report_code != 0:
        raise RuntimeError(f"PG-242 child returned {report_code}")
    return json.loads(XSS_REPORT.read_text(encoding="utf-8-sig")), json.loads(XSS_DATASET.read_text(encoding="utf-8-sig"))


def _run_boolean() -> dict[str, Any]:
    runner = _load("run_pg221_pikachu_boolean_blind_oracle.py")
    runner.SEEDS = BOOLEAN_SEEDS
    runner.REPORT, runner.DATASET, runner.TRACE, runner.PROTOCOL, runner.MARKDOWN = BOOL_REPORT, BOOL_DATASET, BOOL_TRACE, BOOL_PROTOCOL, BOOL_MARKDOWN
    runner.main()
    return json.loads(BOOL_REPORT.read_text(encoding="utf-8-sig"))


def _run_widebyte() -> dict[str, Any]:
    runner = _load("run_pg256_pikachu_widebyte_oracle.py")
    runner.SEEDS = WIDEBYTE_SEEDS
    runner.REPORT, runner.TRACE, runner.PROTOCOL, runner.MARKDOWN = WIDE_REPORT, WIDE_TRACE, WIDE_PROTOCOL, WIDE_MARKDOWN
    runner.WIRE_LOG.clear()
    runner.main()
    return json.loads(WIDE_REPORT.read_text(encoding="utf-8-sig"))


def _projection(response: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(response.get("response_projection") or response.get("projection") or response)


def _sql_feedback_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in list(report.get("episodes") or []):
        ai = dict(episode.get("ai") or {})
        reference = dict(episode.get("reference") or {})
        negative = dict(episode.get("negative") or {})
        typed = dict(episode.get("typed_oracle") or {})
        evidence = dict(typed.get("evidence") or {})
        reset = dict(episode.get("reset") or {})
        ai_response = _projection(dict(ai.get("response") or {}))
        reference_response = _projection(dict(reference.get("response") or {}))
        negative_response = _projection(negative)
        model_decision = dict(episode.get("model_decision") or {})
        row_input = {
            "source": "pg259_pikachu_sql_fresh",
            "seed": int(episode.get("seed", 0) or 0),
            "surface_role": "sql_surface",
            "method": str(episode.get("method", "GET")).upper(),
            "field_count": len(episode.get("fields") or []),
            "status_class": str(ai_response.get("status_class", "unknown")),
            "history_len": 0,
            "fresh_reset_ok": bool(episode.get("fresh_target") and reset.get("container_recreated") and not reset.get("container_restart_used")),
            "reset_completed": bool(reset.get("completed")),
            "reset_not_attempted": False,
            "candidate_sent": bool(ai.get("sent")),
            "oracle_available": bool(typed.get("typed_effect_confirmed") or typed.get("confirmed_positive") or typed.get("typed_available")),
            "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed") or typed.get("confirmed_positive")),
            "typed_effect_observed": bool(typed.get("typed_effect_confirmed") or typed.get("confirmed_positive")),
            "result_fixture_verified": bool(typed.get("confirmed_positive")),
            # The child evaluator owns the comparison because dynamic canary
            # values legitimately change bounded body hashes between the AI
            # and reference requests.  Comparing the whole projection here
            # would turn evaluator-confirmed positives into false mismatches.
            "candidate_reference_agreement": bool(evidence.get("candidate_reference_agreement", _digest(ai_response) == _digest(reference_response))),
            "negative_clean": bool(evidence.get("candidate_negative_differential", negative_response.get("status_class", "") == ai_response.get("status_class", ""))),
            "binding_valid": bool(model_decision.get("encoding_binding_match", False)),
            "transport_error": bool(ai_response.get("transport_error", False)),
            "result_mismatch_observed": bool(typed.get("candidate_reference_agreement") is False),
            "previous_feedback": "result_verified" if typed.get("confirmed_positive") else "abstain" if not ai.get("sent") else "mismatch",
            "candidate_result_present": bool(ai_response),
            "candidate_sql_error_shape": bool((ai_response.get("signal") or {}).get("sql_error_shape") or ai_response.get("sql_error_shape")),
            "boolean_differential": False,
            "negative_result_absent": bool(negative_response),
            "hard_gate_observed": bool((model_decision.get("pg254_gate") or {}).get("action") == "send_candidate"),
            "model_claimed_positive": bool(ai.get("sent")),
            "model_abstained": not bool(ai.get("sent")),
            "model_self_error_detected": bool(model_decision.get("abstain_reason")),
            "failure_signature": str(typed.get("reasons", [""])[0] if typed.get("reasons") else "typed_effect" if typed.get("confirmed_positive") else "candidate_no_effect"),
            "evidence_hash": str(typed.get("evidence_hash", "")),
            "payload_grounded_eligible": bool(typed.get("confirmed_positive")),
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        record = prepare_feedback_record(row_input)
        record.update({"route": str(episode.get("path", "")), "route_source_sha256": str(episode.get("route_source_sha256", "")), "parent_record_id": f"pg259:sql:{episode.get('seed')}:{episode.get('path')}"})
        rows.append(record)
    return rows


def _boolean_feedback_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(report.get("results") or []):
        oracle = dict(item.get("oracle") or {})
        reset = dict(item.get("reset") or {})
        row_input = {
            "source": "pg259_pikachu_boolean_fresh",
            "seed": int(item.get("seed", 0) or 0),
            "surface_role": "sql_boolean_surface",
            "method": "GET",
            "field_count": len(item.get("fields") or []),
            "status_class": "2xx",
            "history_len": 0,
            "fresh_reset_ok": bool(reset.get("fresh_target") and reset.get("container_recreated")),
            "reset_completed": bool(reset.get("completed")),
            "candidate_sent": True,
            "oracle_available": bool(oracle.get("boolean_effect_confirmed")),
            "typed_effect_confirmed": bool(oracle.get("boolean_effect_confirmed")),
            "typed_effect_observed": bool(oracle.get("boolean_effect_confirmed")),
            "result_fixture_verified": bool(oracle.get("boolean_effect_confirmed")),
            "candidate_reference_agreement": True,
            "negative_clean": True,
            "binding_valid": True,
            "boolean_differential": bool(oracle.get("boolean_effect_confirmed")),
            "negative_result_absent": True,
            "hard_gate_observed": True,
            "model_claimed_positive": True,
            "model_abstained": False,
            "previous_feedback": "result_verified" if oracle.get("boolean_effect_confirmed") else "mismatch",
            "candidate_result_present": True,
            "model_self_error_detected": False,
            "failure_signature": "typed_effect" if oracle.get("boolean_effect_confirmed") else "candidate_no_effect",
            "evidence_hash": str(oracle.get("evidence_hash", "")),
            "payload_grounded_eligible": bool(oracle.get("boolean_effect_confirmed")),
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        record = prepare_feedback_record(row_input)
        record.update({"route": str(item.get("route", "")), "route_source_sha256": str(item.get("route_source_sha256", "")), "parent_record_id": f"pg259:boolean:{item.get('seed')}"})
        rows.append(record)
    return rows


def _widebyte_feedback_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(report.get("episodes") or []):
        oracle = dict(item.get("oracle") or item.get("typed_oracle") or {})
        reset = dict(item.get("reset") or {})
        confirmed = bool(oracle.get("confirmed_positive") or oracle.get("widebyte_effect_confirmed") or oracle.get("typed_effect_confirmed"))
        row_input = {
            "source": "pg259_pikachu_widebyte_fresh",
            "seed": int(item.get("seed", 0) or 0),
            "surface_role": "sql_widebyte_surface",
            "method": "POST",
            "field_count": 2,
            "status_class": "2xx",
            "history_len": 0,
            "fresh_reset_ok": bool(reset.get("fresh_target") and reset.get("container_recreated")),
            "reset_completed": bool(reset.get("completed")),
            "candidate_sent": bool(item.get("ai_sent", item.get("ai", {}).get("sent", True))),
            "oracle_available": True,
            "typed_effect_confirmed": confirmed,
            "typed_effect_observed": confirmed,
            "result_fixture_verified": confirmed,
            "candidate_reference_agreement": bool(oracle.get("reference_agreement", True)),
            "negative_clean": bool(oracle.get("negative_clean", True)),
            "binding_valid": True,
            "boolean_differential": False,
            "negative_result_absent": True,
            "hard_gate_observed": True,
            "model_claimed_positive": True,
            "model_abstained": False,
            # A reference-confirmed effect with an AI class mismatch is a
            # reproducible model failure, not an incomplete environment row.
            # Keep it as a hard negative so the repair/abstention heads can
            # learn from the failed first probe.
            "model_self_error_detected": bool(not confirmed and (oracle.get("reference_agreement") is False or dict(oracle.get("evidence") or {}).get("candidate_reference_agreement") is False)),
            "model_self_error_kind": "reference_disagreement" if (not confirmed and (oracle.get("reference_agreement") is False or dict(oracle.get("evidence") or {}).get("candidate_reference_agreement") is False)) else None,
            "previous_feedback": "result_verified" if confirmed else "mismatch",
            "candidate_result_present": True,
            "failure_signature": "typed_effect" if confirmed else "candidate_no_effect",
            "evidence_hash": str(oracle.get("evidence_hash", "")),
            "payload_grounded_eligible": confirmed,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        record = prepare_feedback_record(row_input)
        record.update({"route": "/vul/sqli/sqli_widebyte.php", "parent_record_id": f"pg259:widebyte:{item.get('seed')}"})
        rows.append(record)
    return rows


def _xss_feedback_rows(report: Mapping[str, Any], dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Attach the route omitted by the child token dataset.

    PG-242 intentionally stores only a source hash in its training projection;
    the aggregate trace is allowed to join that hash back to the bounded route
    metadata from the same fresh report.  This keeps endpoint identity in the
    audit trail without retaining a raw request or response body.
    """
    route_by_hash = {
        str(item.get("route_source_sha256", "")): str(item.get("route", ""))
        for item in list(report.get("results") or [])
        if item.get("route_source_sha256")
    }
    rows: list[dict[str, Any]] = []
    for item in list(dataset.get("records") or []):
        row = dict(item, source="pg259_pikachu_xss_fresh")
        route_hash = str(row.get("route_source_sha256", ""))
        row["route"] = route_by_hash.get(route_hash, "")
        rows.append(row)
    return rows


def main() -> int:
    # Rebuild-only mode lets us correct deterministic aggregation mistakes
    # without spending another cold-start cycle on every child container.
    if os.environ.get("PG259_REBUILD_ONLY") == "1":
        sql_report = json.loads(SQL_REPORT.read_text(encoding="utf-8-sig"))
        xss_report = json.loads(XSS_REPORT.read_text(encoding="utf-8-sig"))
        xss_dataset = json.loads(XSS_DATASET.read_text(encoding="utf-8-sig"))
        boolean_report = json.loads(BOOL_REPORT.read_text(encoding="utf-8-sig"))
        widebyte_report = json.loads(WIDE_REPORT.read_text(encoding="utf-8-sig"))
    else:
        sql_report = _run_sql()
        xss_report, xss_dataset = _run_xss()
        boolean_report = _run_boolean()
        widebyte_report = _run_widebyte()
    rows = _sql_feedback_rows(sql_report) + _xss_feedback_rows(xss_report, xss_dataset) + _boolean_feedback_rows(boolean_report) + _widebyte_feedback_rows(widebyte_report)
    # Deduplicate only exact source/seed/trajectory pairs; no oversampling is
    # silently introduced to manufacture class support.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in rows:
        # Route is part of the observed surface.  Two endpoints may produce
        # the same bounded token trajectory while requiring different fields,
        # oracles, and payload encodings; collapsing them loses supervision.
        key = (
            str(row.get("source", "")),
            int(row.get("seed", 0) or 0),
            str(row.get("route", "")),
            str(row.get("route_source_sha256", "")),
            str(row.get("trajectory_hash", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    counts = {
        "records": len(unique),
        "source_counts": dict(Counter(str(row.get("source", "")) for row in unique)),
        "lane_counts": dict(Counter(str(row.get("lane", "")) for row in unique)),
        "gold_count": sum(int(row.get("lane") == "gold") for row in unique),
        "hard_negative_count": sum(int(row.get("lane") == "hard_negative") for row in unique),
        "silver_count": sum(int(row.get("lane") == "silver") for row in unique),
        "fresh_seed_sets": {"sql": list(SQL_SEEDS), "xss": list(XSS_SEEDS), "boolean": list(BOOLEAN_SEEDS), "widebyte": list(WIDEBYTE_SEEDS)},
        "ai_send_count": int((sql_report.get("counts") or {}).get("ai_candidate_send_count", 0) or 0) + int((xss_report.get("counts") or {}).get("ai_send_count", 0) or 0) + int((boolean_report.get("counts") or {}).get("ai_candidate_pair_send_count", 0) or 0) + int((widebyte_report.get("counts") or {}).get("ai_send_count", 0) or 0),
        "confirmed_positive_count": int((sql_report.get("counts") or {}).get("confirmed_positive_count", 0) or 0) + int((xss_report.get("counts") or {}).get("confirmed_positive_count", 0) or 0) + int((boolean_report.get("counts") or {}).get("confirmed_positive_count", 0) or 0) + int((widebyte_report.get("counts") or {}).get("confirmed_positive_count", 0) or 0),
    }
    source_reports = {
        "sql": {"path": str(SQL_REPORT.relative_to(ROOT)), "sha256": str(sql_report.get("report_sha256", ""))},
        "xss": {"path": str(XSS_REPORT.relative_to(ROOT)), "sha256": str(xss_report.get("report_sha256", ""))},
        "boolean": {"path": str(BOOL_REPORT.relative_to(ROOT)), "sha256": str(boolean_report.get("report_sha256", ""))},
        "widebyte": {"path": str(WIDE_REPORT.relative_to(ROOT)), "sha256": str(widebyte_report.get("report_sha256", ""))},
    }
    dataset = {"schema_version": "pg259-fresh-local-trace-collection-dataset-v1", "source_reports": source_reports, "records": unique, "counts": counts, "contract": {"fresh_container_required": True, "ai_candidate_and_reference_required": True, "matched_negative_required": True, "loopback_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    report = {"protocol_id": "pg-pk-259-fresh-local-trace-collection-v1", "schema_version": "pg259-fresh-local-trace-collection-report-v1", "status": "completed_fresh_local_sql_xss_trace_collection", "source_reports": source_reports, "counts": counts, "class_support_is_not_sufficient_for_promotion": True, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"all_sources_are_authorized_loopback_labs": True, "ai_participated_in_send_path": True, "reference_and_negative_are_independent": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True}}
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg259-fresh-local-trace-collection-protocol-v1", "child_runners": ["PG-255 SQL GET/POST", "PG-242 controlled DOM XSS", "PG-221 boolean result", "PG-256 widebyte row oracle"], "fresh_seed_sets": counts["fresh_seed_sets"], "payloads_are_ephemeral": True, "typed_oracle_required": True, "negative_control_required": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg259-fresh-local-trace-collection-trace-v1", "source_reports": source_reports, "counts": counts, "records": unique, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-259 fresh local SQL/XSS trace collection", "", f"records={len(unique)}; AI sends={counts['ai_send_count']}; confirmed positives={counts['confirmed_positive_count']}", f"sources={counts['source_counts']}", "每条记录来自 fresh loopback child runner，AI/reference/negative 和 typed oracle 各自保留投影；未达到类支持门前不训练晋级、不进入长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
