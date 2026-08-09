"""Bounded research-operations snapshot for the human/AI handoff UI.

The UI is deliberately fed from projections, not raw replay reports.  Runtime
payload values, response bodies and evaluator internals stay in the local
runner; the browser receives task state, hashes, route anatomy and bounded
metrics only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _read_json(name: str, default: Any) -> Any:
    path = RESEARCH / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return default


def _report_time(name: str) -> str | None:
    path = RESEARCH / name
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _training_marker_active(marker_name: str, report_name: str) -> bool:
    """Return True while a report-producing run has not written its final report.

    A marker is deliberately timestamp-based rather than process-based: the
    worker may run in another Python process, while a stale marker after a
    successful write must not hide the completed report forever.  The UI uses
    this only to avoid presenting the previous report as the current run.
    """
    marker = RESEARCH / marker_name
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8-sig"))
        started_at = datetime.fromisoformat(str(payload.get("started_at", "")).replace("Z", "+00:00"))
        report = RESEARCH / report_name
        return not report.exists() or report.stat().st_mtime < started_at.timestamp()
    except (OSError, ValueError, TypeError):
        # An unreadable marker is not evidence of a run; keep the last valid
        # report visible instead of inventing a running state.
        return False


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _route_label(row: dict[str, Any]) -> str:
    return f"{str(row.get('method', 'GET')).upper()} {str(row.get('path', row.get('route', '')))}"


def _episode_task(row: dict[str, Any], index: int, *, prefix: str = "pg255") -> dict[str, Any]:
    typed = dict(row.get("typed_oracle") or {})
    confirmed = bool(row.get("confirmed_positive") or typed.get("confirmed_positive"))
    reasons = [str(item) for item in list(typed.get("reasons") or [])]
    if confirmed:
        status = "ready_for_review"
        label = "复核已确认效果"
        escalation = False
        action = "复核 AI/reference/negative 对照，确认本地范围后归档。"
    elif reasons:
        status = "needs_oracle"
        label = "补齐 evaluator 证据"
        escalation = any("timing" in reason or "oracle" in reason or "escape" in reason for reason in reasons)
        action = "保持未确认；补齐类型化 oracle 或记录明确 abstain 原因。"
    else:
        status = "needs_review"
        label = "检查回放证据"
        escalation = True
        action = "检查 fresh reset、negative、reference 和证据哈希。"
    return {
        "id": f"{prefix}-episode-{index + 1}",
        "role": "reviewer" if confirmed else "collector",
        "owner": "AI" if not escalation else "AI → 人工",
        "human_required": escalation,
        "status": status,
        "label": label,
        "route": _route_label(row),
        "seed": int(row.get("seed", 0) or 0),
        "method": str(row.get("method", "GET")).upper(),
        "typed_effect": bool(typed.get("typed_effect_confirmed")),
        "confirmed_positive": confirmed,
        "reasons": reasons,
        "evidence_hash": str(typed.get("evidence_hash", ""))[:16],
        "instruction": action,
        "raw_material_available": False,
    }


def _process_trace(row: dict[str, Any], index: int, *, prefix: str, family: str) -> dict[str, Any]:
    """Project one bounded local episode into a human-readable timeline.

    The projection deliberately keeps response shapes and hashes, not raw
    request values or response bodies.  It is enough to show the AI's causal
    sequence and the independent judge without turning the UI into a scanner.
    """

    ai = dict(row.get("ai") or {})
    reference = dict(row.get("reference") or {})
    negative = dict(row.get("negative") or {})
    typed = dict(row.get("typed_oracle") or {})
    reset = dict(row.get("reset") or {})
    decision = dict(row.get("model_decision") or ai.get("model_decision") or {})
    route = str(row.get("path") or row.get("route") or "")
    method = str(row.get("method", "GET")).upper()
    confirmed = bool(row.get("confirmed_positive") or typed.get("confirmed_positive"))
    oracle_available = typed.get("oracle_available", typed.get("typed_effect_confirmed", False))
    reasons = [str(item) for item in list(typed.get("reasons") or [])]
    baseline = dict(row.get("baseline") or {})
    baseline_projection = dict(baseline.get("response_projection") or baseline.get("projection") or baseline)
    reference_sent = bool(reference.get("sent"))
    negative_sent = bool(negative.get("sent", negative))
    ai_sent = bool(ai.get("sent"))
    if confirmed:
        final_state, final_detail = "confirmed_local", "typed oracle 与 reference/negative 对照通过；仅限本地复核。"
    elif reasons or oracle_available is False:
        final_state, final_detail = "abstain", reasons[0] if reasons else "oracle 不完整，AI 保持 abstain。"
    else:
        final_state, final_detail = "needs_review", "证据链未形成 confirmed_positive。"

    observe_state = "pass" if baseline_projection or reset.get("fresh_target") else "warn"
    observe_detail = f"{method} {route} · baseline {baseline_projection.get('status_class', 'projection-only')}" if baseline_projection else f"{method} {route} · fresh surface loaded，baseline body 保持 runner 内部"
    stages = [
        {"id": "observe", "label": "观察页面", "state": observe_state, "detail": observe_detail},
        {"id": "decide", "label": "AI 选探针", "state": "pass" if decision or ai.get("selected") else "warn", "detail": str(decision.get("effective_action") or decision.get("action") or ai.get("abstract_probe_class") or ai.get("selected", {}).get("probe_kind") or "未输出抽象动作")},
        {"id": "candidate", "label": "AI candidate", "state": "pass" if ai_sent else "abstain", "detail": "已在 loopback 发送" if ai_sent else "AI abstain，未发送 candidate"},
        {"id": "reference", "label": "Reference", "state": "pass" if reference_sent else "warn", "detail": "独立参考已复放" if reference_sent else "reference 缺失"},
        {"id": "negative", "label": "Negative", "state": "pass" if negative_sent else "warn", "detail": "匹配阴性对照已复放" if negative_sent else "negative 缺失"},
        {"id": "oracle", "label": "Typed oracle", "state": "pass" if confirmed else "warn", "detail": "confirmed_positive" if confirmed else (reasons[0] if reasons else "未确认")},
        {"id": "next", "label": "下一步", "state": final_state, "detail": final_detail},
    ]
    return {
        "id": f"{prefix}-trace-{index + 1}",
        "family": family,
        "route": f"{method} {route}",
        "method": method,
        "seed": int(row.get("seed", 0) or 0),
        "fresh_reset": bool(row.get("fresh_target", row.get("fresh_reset", reset.get("fresh_target", False)))),
        "database_health": str(reset.get("database_health_gate", "unknown")),
        "target_hash": str(row.get("target_instance_hash", reset.get("container_id_sha256", "")))[:16],
        "evidence_hash": str(typed.get("evidence_hash", row.get("evidence_hash", "")))[:16],
        "ai_sent": ai_sent,
        "reference_sent": reference_sent,
        "negative_sent": negative_sent,
        "confirmed_positive": confirmed,
        "oracle_available": bool(oracle_available),
        "abstract_probe": str(ai.get("abstract_probe_class") or ai.get("runtime_probe_class") or (ai.get("selected") or {}).get("abstract_class") or (ai.get("selected") or {}).get("probe_kind") or "abstract_probe"),
        "stages": stages,
    }


def _surface_catalog_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project the browser crawl into a parameter/response completeness view."""

    rows: list[dict[str, Any]] = []
    for raw in list(manifest.get("route_catalog") or []):
        route = dict(raw) if isinstance(raw, dict) else {}
        request = dict(route.get("request_schema") or {})
        response = dict(route.get("response_schema") or {})
        # The browser manifest names the channels explicitly (`get_*`), while
        # older manifests used the shorter `query_params`/`form_params` keys.
        # Accept both so a discovered GET form is not silently reported as an
        # unparameterized page.
        query_params = [str(item) for item in list(request.get("query_params") or request.get("get_query_params") or [])]
        form_params = [str(item) for item in list(request.get("form_params") or request.get("get_form_params") or [])]
        post_params = [str(item) for item in list(request.get("post_form_params") or [])]
        methods = [str(item).upper() for item in list(request.get("methods") or route.get("methods_observed") or [])]
        parameterized_observed = bool(
            response.get("parameterized_response_observed")
            or (query_params and response.get("request_query_replayed"))
            or (form_params and response.get("request_query_replayed"))
            or (post_params and response.get("request_post_replayed"))
        )
        rows.append({
            "path": str(route.get("path") or ""),
            "methods": methods,
            "query_params": query_params,
            "form_params": form_params,
            "post_form_params": post_params,
            "has_parameter_context": bool(query_params or form_params or post_params),
            "parameterized_response_observed": parameterized_observed,
            "status": str(route.get("quality_status") or route.get("status") or "unknown"),
            "training_eligible": bool(route.get("training_eligible")),
            "evidence_sha256": str(response.get("evidence_sha256") or response.get("baseline_evidence_sha256") or "")[:16],
        })
    counts = {
        "routes": len(rows),
        "with_parameter_context": sum(int(row["has_parameter_context"]) for row in rows),
        "parameterized_response_observed": sum(int(row["parameterized_response_observed"]) for row in rows),
        "training_eligible": sum(int(row["training_eligible"]) for row in rows),
        "missing_parameter_context": sum(int(not row["has_parameter_context"]) for row in rows),
    }
    return {"manifest_id": str(manifest.get("manifest_id") or ""), "generated_at": _report_time("pg179_pikachu_browser_crawl_manifest_v1.json"), "counts": counts, "routes": rows}


def _pg324_contract_projection(
    report: dict[str, Any],
    catalog: dict[str, Any],
    trace: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Project PG-324 without turning a stale artifact into a capability claim.

    The runner writes v2 artifacts only after the fresh replay completes.  The
    previous v1 report intentionally remains on disk for audit history, so a
    UI projection must distinguish that report from a current result instead of
    displaying its counts as if the new typed-state oracle had passed.
    """

    expected = {
        "report": "pg324-juice-shop-source-heldout-report-v2",
        "catalog": "pg324-juice-shop-source-heldout-catalog-v2",
        "trace": "pg324-juice-shop-source-heldout-trace-v2",
        "protocol": "pg324-juice-shop-source-heldout-protocol-v2",
    }
    documents = {
        name: dict(value) if isinstance(value, dict) else {}
        for name, value in {
            "report": report,
            "catalog": catalog,
            "trace": trace,
            "protocol": protocol,
        }.items()
    }
    missing: list[str] = []
    schema_ok = True
    for name, value in documents.items():
        if not value:
            missing.append(f"{name}:document")
            schema_ok = False
        if value.get("schema_version") != expected[name]:
            schema_ok = False
            if value:
                missing.append(f"{name}:schema_v2")

    required_fields = {
        "report": ("counts", "checks", "hypothesis_gate", "promotion", "report_sha256"),
        "catalog": ("entries", "raw_payloads_human_review_only", "raw_response_bodies_stored", "catalog_sha256"),
        "trace": ("episodes", "training_eligible", "memory_promotion_allowed", "raw_response_bodies_stored", "trace_sha256"),
        "protocol": ("required_gates", "promotion", "protocol_sha256"),
    }
    for name, fields in required_fields.items():
        value = documents[name]
        for field in fields:
            if field not in value:
                missing.append(f"{name}:{field}")

    if not report and not catalog and not trace and not protocol:
        artifact_status = "awaiting_fresh_replay"
    elif not schema_ok:
        artifact_status = "stale_contract"
    elif missing:
        artifact_status = "incomplete"
    else:
        artifact_status = "completed_evaluation_only"

    counts = dict(documents["report"].get("counts") or {}) if artifact_status == "completed_evaluation_only" else {}
    worst = dict(documents["report"].get("worst_seed_metrics") or {}) if artifact_status == "completed_evaluation_only" else {}
    checks = dict(documents["report"].get("checks") or {}) if artifact_status == "completed_evaluation_only" else {}
    gate = dict(documents["report"].get("hypothesis_gate") or {}) if artifact_status == "completed_evaluation_only" else {}
    required_check_names = (
        "real_docker_contacted", "fresh_container_per_route_seed", "get_post_pair", "independent_implementation",
        "docker_network_none", "loopback_relay_only", "external_network_disabled", "zero_bind_volume_per_route",
        "source_attestation_per_route", "safety_mode_override_all", "typed_evidence_hash_per_route", "challenge_state_baseline_all",
        "belief_trace_complete", "failure_action_changed_all", "model_context_firewall", "raw_payload_in_model_context", "raw_response_bodies_stored", "public_target_contacted",
        "time_delay", "domain_data_write", "stateful_xss_write",
    )
    required_gates = dict(documents["protocol"].get("required_gates") or {}) if artifact_status == "completed_evaluation_only" else {}
    return {
        "artifact_status": artifact_status,
        "report_status": str(documents["report"].get("status", "not_run")),
        "schema_version": str(documents["report"].get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "counts": {key: int(value or 0) for key, value in counts.items() if isinstance(value, (int, float))},
        "worst_seed_metrics": {key: float(value) if isinstance(value, (int, float)) else value for key, value in worst.items()},
        "checks": {key: bool(checks.get(key)) for key in required_check_names if key in checks},
        "required_gates": {key: bool(required_gates.get(key)) for key in ("multi_missing_question", "get_post_pair", "typed_challenge_state_delta", "fresh_baseline_unsolved", "belief_update", "failure_action_changed", "model_context_firewall", "matched_negative", "fresh_reset", "evidence_hash", "safety_mode_override", "docker_network_none", "loopback_relay_only", "raw_payload_training_excluded") if key in required_gates},
        "hypothesis_gate_status": str(gate.get("status", "not_run")),
        "claim_allowed": bool(gate.get("claim_allowed")) if artifact_status == "completed_evaluation_only" else False,
        "report_evidence_hash": str(documents["report"].get("report_sha256", ""))[:16] if artifact_status == "completed_evaluation_only" else "",
        "promotion_blocked": True,
        "model_capability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg325_contract_projection(
    report: dict[str, Any],
    catalog: dict[str, Any],
    trace: dict[str, Any],
    protocol: dict[str, Any],
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project PG-325 SQL family-heldout replay into the research UI.

    PG-325 is deliberately an evaluation-only artifact.  Its SQL typed
    response-shape oracle can prove that the bounded local replay contract was
    exercised, but it cannot promote a checkpoint, a payload catalog, or a
    general vulnerability claim.  Missing/stale documents remain visible as
    such instead of being converted into a flattering zero or a capability
    claim.
    """

    expected = {
        "report": "pg325-sql-family-holdout-report-v1",
        "catalog": "pg325-sql-family-holdout-catalog-v1",
        "trace": "pg325-sql-family-holdout-trace-v1",
        "protocol": "pg325-sql-family-holdout-protocol-v1",
    }
    documents = {
        name: dict(value) if isinstance(value, dict) else {}
        for name, value in {
            "report": report,
            "catalog": catalog,
            "trace": trace,
            "protocol": protocol,
        }.items()
    }
    missing: list[str] = []
    schema_ok = True
    for name, value in documents.items():
        if not value:
            missing.append(f"{name}:document")
            schema_ok = False
        if value.get("schema_version") != expected[name]:
            schema_ok = False
            if value:
                missing.append(f"{name}:schema_v1")

    required_fields = {
        "report": ("counts", "worst_seed_metrics", "checks", "hypothesis_gate", "scientific_gate", "promotion", "report_sha256"),
        "catalog": ("entries", "raw_payloads_human_review_only", "raw_response_bodies_stored", "catalog_sha256"),
        "trace": ("episodes", "training_eligible", "memory_promotion_allowed", "raw_response_bodies_stored", "trace_sha256"),
        "protocol": ("required_gates", "promotion", "protocol_sha256"),
    }
    for name, fields in required_fields.items():
        for field in fields:
            if field not in documents[name]:
                missing.append(f"{name}:{field}")

    if not any(documents.values()):
        artifact_status = "awaiting_fresh_replay"
    elif not schema_ok:
        artifact_status = "stale_contract"
    elif missing:
        artifact_status = "incomplete"
    else:
        artifact_status = "completed_evaluation_only"

    counts = dict(documents["report"].get("counts") or {}) if artifact_status == "completed_evaluation_only" else {}
    worst = dict(documents["report"].get("worst_seed_metrics") or {}) if artifact_status == "completed_evaluation_only" else {}
    checks = dict(documents["report"].get("checks") or {}) if artifact_status == "completed_evaluation_only" else {}
    gate = dict(documents["report"].get("hypothesis_gate") or {}) if artifact_status == "completed_evaluation_only" else {}
    required_check_names = (
        "real_docker_contacted", "fresh_container_per_route_seed", "get_post_pair", "sql_family_holdout",
        "cross_implementation_replay_canaries_present", "docker_network_none", "external_network_disabled",
        "zero_volume_per_route", "database_health_per_route", "source_attestation_per_route",
        "typed_evidence_hash_per_route", "belief_trace_complete", "belief_role_bound_evidence", "failure_action_changed_all",
        "model_context_firewall", "raw_payload_in_model_context", "raw_response_bodies_stored",
        "public_target_contacted", "sql_time_delay", "sql_write",
    )
    required_gates = dict(documents["protocol"].get("required_gates") or {}) if artifact_status == "completed_evaluation_only" else {}
    audit_doc = dict(audit) if isinstance(audit, dict) else {}
    audit_status = str(audit_doc.get("status", "not_embedded"))
    audit_failures = [str(item) for item in list(audit_doc.get("failures") or [])]
    return {
        "artifact_status": artifact_status,
        "report_status": str(documents["report"].get("status", "not_run")),
        "schema_version": str(documents["report"].get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "counts": {key: int(value or 0) for key, value in counts.items() if isinstance(value, (int, float))},
        "worst_seed_metrics": {key: float(value) if isinstance(value, (int, float)) else value for key, value in worst.items()},
        "checks": {key: bool(checks.get(key)) for key in required_check_names if key in checks},
        "required_gates": {key: bool(required_gates.get(key)) for key in (
            "multi_missing_question", "get_post_pair", "typed_sql_effect", "matched_negative", "fresh_reset",
            "database_health", "source_attestation", "evidence_hash", "belief_update", "role_bound_belief_evidence", "failure_action_changed",
            "model_context_firewall", "docker_network_none", "raw_payload_training_excluded",
        ) if key in required_gates},
        "hypothesis_gate_status": str(gate.get("status", "not_run")),
        "claim_allowed": bool(gate.get("claim_allowed")) if artifact_status == "completed_evaluation_only" else False,
        "report_evidence_hash": str(documents["report"].get("report_sha256", ""))[:16] if artifact_status == "completed_evaluation_only" else "",
        "audit_status": audit_status,
        "audit_failures": audit_failures,
        "audit_promotion_allowed": bool(audit_doc.get("promotion_allowed")) if audit_doc else False,
        "audit_target_contacted": bool(audit_doc.get("target_contacted")) if audit_doc else False,
        "audit_evidence_hash": str(audit_doc.get("audit_sha256") or audit_doc.get("report_sha256", ""))[:16],
        "promotion_blocked": True,
        "model_capability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg326_contract_projection(
    report: dict[str, Any],
    protocol: dict[str, Any],
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the read-only cross-implementation/forgetting matrix.

    The matrix is useful even while blocked: it exposes perfect observed
    replay rates separately from missing uniform contracts.  PG-327B may now
    supply a real before/after pair, but missing per-source fields are never
    coerced to a pass or to a training claim.
    """

    expected = {
        "report": "pg326-cross-implementation-forgetting-matrix-report-v1",
        "protocol": "pg326-cross-implementation-forgetting-matrix-protocol-v1",
    }
    documents = {
        "report": dict(report) if isinstance(report, dict) else {},
        "protocol": dict(protocol) if isinstance(protocol, dict) else {},
    }
    missing: list[str] = []
    schema_ok = True
    for name, document in documents.items():
        if not document:
            missing.append(f"{name}:document")
            schema_ok = False
        if document.get("schema_version") != expected[name]:
            schema_ok = False
            if document:
                missing.append(f"{name}:schema")
    for name, fields in {
        "report": ("source_rows", "totals", "worst_seed_metrics", "uniform_checks", "forgetting", "hypothesis_gate", "promotion", "report_sha256"),
        "protocol": ("required_gates", "promotion", "protocol_sha256"),
    }.items():
        for field in fields:
            if field not in documents[name]:
                missing.append(f"{name}:{field}")
    if not any(documents.values()):
        artifact_status = "awaiting_matrix"
    elif not schema_ok:
        artifact_status = "stale_contract"
    elif missing:
        artifact_status = "incomplete"
    else:
        artifact_status = "completed_evaluation_matrix_blocked"
    report_doc = documents["report"] if artifact_status == "completed_evaluation_matrix_blocked" else {}
    gate = dict(report_doc.get("hypothesis_gate") or {})
    audit_doc = dict(audit) if isinstance(audit, dict) else {}
    return {
        "artifact_status": artifact_status,
        "report_status": str(report_doc.get("status", "not_run")),
        "schema_version": str(report_doc.get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "implementation_count": len(list(report_doc.get("implementation_digests") or [])),
        "families": [str(item) for item in list(report_doc.get("families") or [])],
        "counts": dict(report_doc.get("totals") or {}),
        "worst_seed_metrics": dict(report_doc.get("worst_seed_metrics") or {}),
        "uniform_checks": {str(key): bool(value) for key, value in dict(report_doc.get("uniform_checks") or {}).items()},
        "forgetting": dict(report_doc.get("forgetting") or {}),
        "matrix_gate_status": str(gate.get("status", "not_run")),
        "matrix_gate_checks": {str(key): bool(value) for key, value in dict(gate.get("checks") or {}).items()},
        "claim_allowed": bool(gate.get("claim_allowed")) if artifact_status == "completed_evaluation_matrix_blocked" else False,
        "report_evidence_hash": str(report_doc.get("report_sha256", ""))[:16],
        "audit_status": str(audit_doc.get("status", "not_embedded")),
        "audit_failures": [str(item) for item in list(audit_doc.get("failures") or [])],
        "audit_target_contacted": bool(audit_doc.get("target_contacted")) if audit_doc else False,
        "audit_evidence_hash": str(audit_doc.get("audit_sha256") or audit_doc.get("report_sha256", ""))[:16],
        "promotion_blocked": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg327_training_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Project the remote A800 candidate without turning it into capability.

    A training report is useful for provenance and resource accounting, but its
    offline metrics cannot satisfy the PG-326 strict schema or forgetting-pair
    gate.  Missing report fields therefore remain visible as incomplete.
    """

    expected_schema = "pg327-a800-replay-training-report-v1"
    document = dict(report) if isinstance(report, dict) else {}
    required = ("protocol_id", "schema_version", "status", "sources", "training", "metrics", "per_seed", "hypothesis_gate", "promotion", "provenance", "report_sha256")
    missing = sorted(field for field in required if field not in document)
    schema_ok = document.get("schema_version") == expected_schema
    if not document:
        artifact_status = "awaiting_training"
    elif not schema_ok:
        artifact_status = "stale_contract"
    elif missing:
        artifact_status = "incomplete"
    else:
        artifact_status = "completed_remote_a800_candidate"
    training = dict(document.get("training") or {}) if artifact_status == "completed_remote_a800_candidate" else {}
    metrics = dict(document.get("metrics") or {}) if artifact_status == "completed_remote_a800_candidate" else {}
    promotion = dict(document.get("promotion") or {}) if artifact_status == "completed_remote_a800_candidate" else {}
    return {
        "artifact_status": artifact_status,
        "report_status": str(document.get("status", "not_run")),
        "schema_version": str(document.get("schema_version", "")),
        "missing_fields": missing,
        "execution_mode": str(training.get("execution_mode", "")),
        "device": str(training.get("device", "")),
        "gpu_name": str(training.get("gpu_name", "")),
        "visible_cuda_devices": str(training.get("visible_cuda_devices", "")),
        "seed_count": len(list(training.get("seeds") or [])),
        "train_count": int(training.get("train_count", 0) or 0),
        "metrics": metrics,
        "per_seed": [dict(item) for item in list(document.get("per_seed") or []) if isinstance(item, dict)],
        "hypothesis_gate": dict(document.get("hypothesis_gate") or {}),
        "provenance": dict(document.get("provenance") or {}),
        "report_evidence_hash": str(document.get("report_sha256", ""))[:16],
        "training_allowed": bool(promotion.get("training_allowed")) if artifact_status == "completed_remote_a800_candidate" else False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg370_candidate_projection(
    report: dict[str, Any] | None,
    *,
    report_present: bool | None = None,
) -> dict[str, Any]:
    """Expose only bounded PG-370 candidate-training facts.

    The report contains large vocabularies, candidate rows and checkpoint
    paths.  None of those are UI data: only abstract vocabulary scope/gaps,
    worst-seed metrics and integrity hashes are projected.  This is a
    candidate-training projection, never a capability or promotion claim.
    """

    expected_schema = "pg370-multitask-moe-candidate-v1"
    document = dict(report) if isinstance(report, dict) else {}
    present = bool(document) if report_present is None else bool(report_present)
    required = ("schema_version", "status", "training", "candidates", "worst_seed", "locks", "promotion", "scientific_gate", "report_sha256")
    missing = [field for field in required if field not in document]
    schema_ok = document.get("schema_version") == expected_schema
    if not present:
        artifact_status = "pending"
        missing = ["report"]
    elif not schema_ok:
        artifact_status = "stale_contract"
    else:
        training = document.get("training") if isinstance(document.get("training"), dict) else {}
        worst = document.get("worst_seed") if isinstance(document.get("worst_seed"), dict) else {}
        for field in ("device", "vocabulary_scope", "vocabulary_gaps"):
            if field not in training:
                missing.append(f"training.{field}")
        for field in ("sequence_exact_min", "slot_accuracy_min", "ask_recall_min", "repair_recall_min", "positive_recall_min", "negative_false_allow_max", "entropy_relative_drop_max"):
            if field not in worst:
                missing.append(f"worst_seed.{field}")
        if not isinstance(document.get("candidates"), list) or not document.get("candidates"):
            missing.append("candidates")
        artifact_status = "incomplete" if missing else "candidate_only"

    training = dict(document.get("training") or {}) if isinstance(document.get("training"), dict) else {}
    gaps = dict(training.get("vocabulary_gaps") or {}) if isinstance(training.get("vocabulary_gaps"), dict) else {}
    worst = dict(document.get("worst_seed") or {}) if isinstance(document.get("worst_seed"), dict) else {}

    def as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def as_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if result == result and result not in (float("inf"), float("-inf")) else None

    def sha(value: Any) -> str:
        text = str(value or "")
        return text.lower() if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text) else ""

    checkpoints: list[str] = []
    checkpoint_seeds: list[int] = []
    for item in list(document.get("candidates") or []):
        if not isinstance(item, dict):
            continue
        digest = sha(dict(item.get("checkpoint") or {}).get("sha256"))
        if digest:
            checkpoints.append(digest)
            checkpoint_seeds.append(as_int(item.get("seed")))
    locks = dict(document.get("locks") or {}) if isinstance(document.get("locks"), dict) else {}
    lock_projection: dict[str, Any] = {}
    for category in ("datasets", "audits"):
        values = locks.get(category)
        if isinstance(values, dict):
            lock_projection[category] = {str(key): sha(value) for key, value in values.items() if sha(value)}
    declared = dict(locks.get("declared_vocabulary") or {}) if isinstance(locks.get("declared_vocabulary"), dict) else {}
    unknown_slot_values = gaps.get("unknown_slot_values") if isinstance(gaps.get("unknown_slot_values"), dict) else {}
    declared_slot_values = locks.get("declared_slot_values") if isinstance(locks.get("declared_slot_values"), dict) else {}
    lock_projection["declared_vocabulary_size"] = len(list(locks.get("declared_vocabulary") or []))
    lock_projection["declared_slot_count"] = len(declared_slot_values)
    for key in ("declared_vocabulary_sha256", "declared_slot_values_sha256", "rules_sha256"):
        lock_projection[key] = sha(locks.get(key) or declared.get(key))

    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "artifact_status": artifact_status,
        "report_status": str(document.get("status", "pending")) if present else "pending",
        "schema_version": str(document.get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "vocabulary": {
            "scope": str(training.get("vocabulary_scope", "unknown")),
            "size": as_int(training.get("vocabulary_size")) if "vocabulary_size" in training else None,
            "gaps": {
                "unknown_token_count": as_int(gaps.get("unknown_token_count")),
                "unknown_slot_value_count": as_int(gaps.get("unknown_slot_value_count")),
                "unknown_slot_value_axes": sorted(str(key) for key in unknown_slot_values)[:64],
                "blocked": bool(gaps.get("blocked")),
                "unknown_tokens_sha256": sha(gaps.get("unknown_tokens_sha256")),
            },
            "required_context_window": as_int(training.get("required_context_window")),
            "cpu_diagnostic_union": bool(training.get("cpu_diagnostic_union")),
        },
        "worst_seed_metrics": {key: (as_int(worst.get(key)) if key == "negative_false_allow_max" else as_float(worst.get(key))) for key in ("sequence_exact_min", "slot_accuracy_min", "ask_recall_min", "repair_recall_min", "positive_recall_min", "negative_false_allow_max", "entropy_relative_drop_max")},
        "a800": {
            "device": str(training.get("device", "unknown")),
            "gpu_name": str(training.get("gpu_name", "")),
            "visible_cuda_devices": str(training.get("visible_cuda_devices", "")),
            "seeds": [as_int(seed) for seed in list(training.get("seeds") or [])[:32]],
            "checkpoint_seeds": checkpoint_seeds[:32],
            "checkpoint_hashes": checkpoints[:32],
        },
        "locks": lock_projection,
        "report_evidence_hash": sha(document.get("report_sha256")),
        "promotion": promotion,
        "training_eligible": False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "claim_allowed": False,
        "raw_material_available": False,
    }


def _pg373_staged_candidate_projection(
    report: dict[str, Any] | None,
    *,
    report_present: bool | None = None,
) -> dict[str, Any]:
    """Project the staged PG-373 candidate without exposing model material.

    PG-373 is the corrected entropy experiment: a train-only next-token
    baseline is trained first, then structured heads are added with a KL
    anchor.  The dashboard may show bounded metrics and provenance hashes,
    but never candidate rows, checkpoint paths, vocab entries, wire or oracle
    material.  It is still a research candidate, not a capability claim.
    """

    expected_schema = "pg373-staged-pretrain-multitask-candidate-v1"
    document = dict(report) if isinstance(report, dict) else {}
    present = bool(document) if report_present is None else bool(report_present)
    required = (
        "schema_version",
        "status",
        "training",
        "candidates",
        "worst_seed",
        "promotion",
        "scientific_gate",
        "locks",
        "report_sha256",
    )
    missing = [field for field in required if field not in document]
    schema_ok = document.get("schema_version") == expected_schema
    if not present:
        artifact_status = "pending"
        missing = ["report"]
    elif not schema_ok:
        artifact_status = "stale_contract"
    else:
        training = document.get("training") if isinstance(document.get("training"), dict) else {}
        worst = document.get("worst_seed") if isinstance(document.get("worst_seed"), dict) else {}
        for field in (
            "device",
            "seeds",
            "baseline_kind",
            "pretrain_epochs",
            "posttrain_epochs",
            "required_context_window",
            "vocabulary_size",
        ):
            if field not in training:
                missing.append(f"training.{field}")
        for field in (
            "sequence_exact_min",
            "slot_accuracy_min",
            "ask_recall_min",
            "repair_recall_min",
            "positive_recall_min",
            "negative_false_allow_max",
            "entropy_relative_drop_max",
        ):
            if field not in worst:
                missing.append(f"worst_seed.{field}")
        if not isinstance(document.get("candidates"), list) or not document.get("candidates"):
            missing.append("candidates")
        artifact_status = "incomplete" if missing else "candidate_only"

    training = dict(document.get("training") or {}) if isinstance(document.get("training"), dict) else {}
    worst = dict(document.get("worst_seed") or {}) if isinstance(document.get("worst_seed"), dict) else {}
    gate = dict(document.get("scientific_gate") or {}) if isinstance(document.get("scientific_gate"), dict) else {}

    def as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def as_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if result == result and result not in (float("inf"), float("-inf")) else None

    def sha(value: Any) -> str:
        text = str(value or "")
        return text.lower() if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text) else ""

    def metric_block(value: Any) -> dict[str, Any]:
        block = dict(value) if isinstance(value, dict) else {}
        return {
            "rows": as_int(block.get("rows")),
            "token_accuracy": as_float(block.get("token_accuracy")),
            "sequence_exact": as_float(block.get("sequence_exact")),
            "slot_accuracy": as_float(block.get("slot_accuracy")),
            "ask_recall": as_float(block.get("ask_recall")),
            "repair_recall": as_float(block.get("repair_recall")),
            "positive_recall": as_float(block.get("positive_recall")),
            "negative_false_allow": as_int(block.get("negative_false_allow")),
            "predictive_entropy": as_float(block.get("predictive_entropy")),
        }

    per_seed: list[dict[str, Any]] = []
    for item in list(document.get("candidates") or [])[:32]:
        if not isinstance(item, dict):
            continue
        per_seed.append(
            {
                "seed": as_int(item.get("seed")),
                "baseline": metric_block(item.get("baseline")),
                "post": metric_block(item.get("post")),
                "entropy_relative_drop": as_float(item.get("entropy_relative_drop")),
                "checkpoint_sha256": sha(dict(item.get("checkpoint") or {}).get("sha256")),
            }
        )

    locks = dict(document.get("locks") or {}) if isinstance(document.get("locks"), dict) else {}
    lock_projection: dict[str, Any] = {}
    for category in ("datasets", "audits"):
        values = locks.get(category)
        if isinstance(values, dict):
            lock_projection[category] = {str(key): sha(value) for key, value in values.items() if sha(value)}
    for key in ("runner_sha256", "base_model_sha256", "rules_sha256"):
        lock_projection[key] = sha(locks.get(key))

    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "artifact_status": artifact_status,
        "report_status": str(document.get("status", "pending")) if present else "pending",
        "schema_version": str(document.get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "baseline_kind": str(training.get("baseline_kind", "unknown")),
        "training": {
            "device": str(training.get("device", "unknown")),
            "seeds": [as_int(seed) for seed in list(training.get("seeds") or [])[:32]],
            "pretrain_epochs": as_int(training.get("pretrain_epochs")),
            "posttrain_epochs": as_int(training.get("posttrain_epochs")),
            "microbatch": as_int(training.get("microbatch")),
            "required_context_window": as_int(training.get("required_context_window")),
            "vocabulary_size": as_int(training.get("vocabulary_size")),
            "config": {
                key: as_int(dict(training.get("config") or {}).get(key))
                for key in ("d_model", "n_layers", "experts", "expert_hidden", "max_length")
                if key in dict(training.get("config") or {})
            },
        },
        "worst_seed_metrics": {
            key: (as_int(worst.get(key)) if key == "negative_false_allow_max" else as_float(worst.get(key)))
            for key in (
                "sequence_exact_min",
                "slot_accuracy_min",
                "ask_recall_min",
                "repair_recall_min",
                "positive_recall_min",
                "negative_false_allow_max",
                "entropy_relative_drop_max",
            )
        },
        "per_seed": per_seed,
        "scientific_gate": {
            "trained_baseline_entropy_comparison": bool(gate.get("trained_baseline_entropy_comparison")),
            "typed_live_replay_with_model_selected_wire": bool(gate.get("typed_live_replay_with_model_selected_wire")),
            "independent_implementation": bool(gate.get("independent_implementation")),
            "claim_allowed": False,
        },
        "locks": lock_projection,
        "report_evidence_hash": sha(document.get("report_sha256")),
        "promotion": promotion,
        "training_eligible": False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "claim_allowed": False,
        "raw_material_available": False,
    }


def _pg374_model_selected_replay_projection(
    report: dict[str, Any] | None,
    *,
    report_present: bool | None = None,
) -> dict[str, Any]:
    """Project the PG-374 plan as bounded research-operations counts.

    PG-374 is planning-only: the PG-373 decoder output is not materialized,
    so no model selection, typed effect, wire creation, or target contact is
    claimed.  The source plan contains per-row route references and execution
    details; those are deliberately consumed only for consistency checks and
    never returned to the dashboard.  Promotion remains closed even if a
    malformed or optimistic source report says otherwise.
    """

    expected_schema = "pg374-model-selected-replay-plan-v1"
    document = dict(report) if isinstance(report, dict) else {}
    present = bool(document) if report_present is None else bool(report_present)
    required = (
        "schema_version",
        "status",
        "implementation",
        "rule_ir_schema",
        "execution",
        "fresh_typed_replay_contract",
        "rows",
        "counts",
        "promotion",
        "report_sha256",
    )
    missing = [field for field in required if field not in document]
    schema_ok = document.get("schema_version") == expected_schema
    if not present:
        artifact_status = "pending"
        missing = ["report"]
    elif not schema_ok:
        artifact_status = "stale_contract"
    else:
        counts = document.get("counts") if isinstance(document.get("counts"), dict) else {}
        if not isinstance(document.get("counts"), dict):
            missing.append("counts")
        for field in (
            "roles",
            "get_rows",
            "post_rows",
            "candidate_rows",
            "reference_rows",
            "negative_rows",
            "replay_rows",
            "model_selected",
            "typed_effect_confirmed",
            "wire_created",
            "target_contacted",
        ):
            if field not in counts:
                missing.append(f"counts.{field}")
        if document.get("status") != "planning_only_blocked":
            missing.append("status:planning_only_blocked")
        rows = document.get("rows")
        if not isinstance(rows, list):
            missing.append("rows:list")
        elif len(rows) != 24:
            missing.append("rows:expected_24")
        contract = document.get("fresh_typed_replay_contract")
        if not isinstance(contract, dict):
            missing.append("fresh_typed_replay_contract")
        else:
            for field in (
                "candidate_reference_negative_replay_required",
                "fresh_reset_per_seed_route_role",
                "typed_evidence_sha256_required",
                "negative_violation_zero_required",
                "model_selected_separate_from_typed_effect",
                "wire_creation_separate_from_model_selected",
            ):
                if contract.get(field) is not True:
                    missing.append(f"fresh_typed_replay_contract.{field}")
        implementation = document.get("implementation")
        if not isinstance(implementation, dict) or implementation.get("independent_implementation_required") is not True:
            missing.append("implementation.independent_implementation_required")
        execution = document.get("execution")
        if not isinstance(execution, dict):
            missing.append("execution")
        else:
            for field in ("docker_started", "gpu_started", "network_contacted"):
                if execution.get(field) is not False:
                    missing.append(f"execution.{field}=false")
        # The plan has a fixed 3-seed × 2-route × 4-role envelope.  Derive
        # method/role counts from rows and compare to the declared summary so
        # the UI cannot display an internally inconsistent optimistic count.
        rows = document.get("rows") if isinstance(document.get("rows"), list) else []
        derived = {
            "get_rows": 0,
            "post_rows": 0,
            "candidate_rows": 0,
            "reference_rows": 0,
            "negative_rows": 0,
            "replay_rows": 0,
        }
        valid_rows = True
        for row in rows:
            if not isinstance(row, dict):
                valid_rows = False
                continue
            method = str(row.get("method", "")).upper()
            role = str(row.get("role", ""))
            if method == "GET":
                derived["get_rows"] += 1
            elif method == "POST":
                derived["post_rows"] += 1
            else:
                valid_rows = False
            role_key = f"{role}_rows"
            if role_key in derived:
                derived[role_key] += 1
            else:
                valid_rows = False
        if not valid_rows:
            missing.append("rows:bounded_method_role_shape")
        for field, value in derived.items():
            try:
                declared = int(counts.get(field))
            except (TypeError, ValueError, OverflowError):
                declared = -1
            if declared != value:
                missing.append(f"counts.{field}:inconsistent")
        try:
            declared_roles = int(counts.get("roles"))
        except (TypeError, ValueError, OverflowError):
            declared_roles = -1
        if declared_roles != len(rows):
            missing.append("counts.roles:inconsistent")
        artifact_status = "incomplete" if missing else "planning_only_blocked"

    counts = document.get("counts") if isinstance(document.get("counts"), dict) else {}

    def as_nonnegative_int(value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return result if 0 <= result <= 100000 else 0

    def sha(value: Any) -> str:
        text = str(value or "")
        return text.lower() if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text) else ""

    count_projection = {
        "row_count": as_nonnegative_int(counts.get("roles")),
        "get_row_count": as_nonnegative_int(counts.get("get_rows")),
        "post_row_count": as_nonnegative_int(counts.get("post_rows")),
        "role_counts": {
            role: as_nonnegative_int(counts.get(f"{role}_rows"))
            for role in ("candidate", "reference", "negative", "replay")
        },
        "model_selected_count": as_nonnegative_int(counts.get("model_selected")),
        "typed_effect_confirmed_count": as_nonnegative_int(counts.get("typed_effect_confirmed")),
        "wire_created_count": as_nonnegative_int(counts.get("wire_created")),
        "target_contacted_count": as_nonnegative_int(counts.get("target_contacted")),
    }
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "artifact_status": artifact_status,
        "report_status": str(document.get("status", "pending")) if present else "pending",
        "schema_version": str(document.get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "counts": count_projection,
        "report_evidence_hash": sha(document.get("report_sha256")),
        "promotion": promotion,
        "promotion_blocked": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg374_replay_plan_projection(
    plan: dict[str, Any] | None,
    *,
    plan_present: bool | None = None,
) -> dict[str, Any]:
    """Expose only the bounded PG-374 model-selected replay contract."""

    expected_schema = "pg374-model-selected-replay-plan-v1"
    document = dict(plan) if isinstance(plan, dict) else {}
    present = bool(document) if plan_present is None else bool(plan_present)
    required = ("schema_version", "status", "implementation", "staged_candidate", "rule_ir_schema", "execution", "fresh_typed_replay_contract", "counts", "promotion", "report_sha256")
    missing = [field for field in required if field not in document]
    if not present:
        artifact_status = "pending"
        missing = ["plan"]
    elif document.get("schema_version") != expected_schema:
        artifact_status = "stale_contract"
    elif missing:
        artifact_status = "incomplete"
    else:
        artifact_status = "planning_only_blocked"

    def as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    def sha(value: Any) -> str:
        text = str(value or "")
        return text.lower() if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text) else ""

    counts = dict(document.get("counts") or {}) if isinstance(document.get("counts"), dict) else {}
    execution = dict(document.get("execution") or {}) if isinstance(document.get("execution"), dict) else {}
    contract = dict(document.get("fresh_typed_replay_contract") or {}) if isinstance(document.get("fresh_typed_replay_contract"), dict) else {}
    staged = dict(document.get("staged_candidate") or {}) if isinstance(document.get("staged_candidate"), dict) else {}
    implementation = dict(document.get("implementation") or {}) if isinstance(document.get("implementation"), dict) else {}
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "artifact_status": artifact_status,
        "report_status": str(document.get("status", "pending")) if present else "pending",
        "schema_version": str(document.get("schema_version", "")),
        "missing_fields": sorted(set(missing)),
        "implementation": {
            "implementation_id": str(implementation.get("implementation_id", "unknown")),
            "independent_implementation_required": bool(implementation.get("independent_implementation_required")),
        },
        "staged_candidate": {
            "candidate_seed_count": as_int(staged.get("candidate_seed_count")),
            "output_materialized": bool(staged.get("output_materialized")),
            "full_13_slot_output_materialized": bool(staged.get("full_13_slot_output_materialized")),
            "typed_live_replay_with_model_selected_wire": bool(staged.get("typed_live_replay_with_model_selected_wire")),
        },
        "execution": {
            "docker_started": bool(execution.get("docker_started")),
            "gpu_started": bool(execution.get("gpu_started")),
            "network_contacted": bool(execution.get("network_contacted")),
            "network_mode": str(execution.get("network_mode", "unknown")),
            "loopback_only_required": bool(execution.get("loopback_only_required")),
        },
        "counts": {key: as_int(counts.get(key)) for key in ("seeds", "routes", "episodes", "roles", "get_rows", "post_rows", "model_selected", "typed_effect_confirmed", "wire_created", "target_contacted")},
        "replay_contract": {key: bool(contract.get(key)) for key in ("candidate_reference_negative_replay_required", "fresh_reset_per_seed_route_role", "typed_evidence_sha256_required", "negative_violation_zero_required", "model_selected_separate_from_typed_effect", "wire_creation_separate_from_model_selected", "observed_in_this_plan")},
        "report_evidence_hash": sha(document.get("report_sha256")),
        "promotion": promotion,
        "training_eligible": False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg327b_replay_projection(report: dict[str, Any], audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project paired fresh replay evidence while keeping promotion closed."""

    expected_schema = "pg327b-paired-fresh-replay-report-v1"
    document = dict(report) if isinstance(report, dict) else {}
    required = ("schema_version", "status", "runtime", "counts", "checks", "forgetting", "hypothesis_gate", "promotion", "report_sha256")
    missing = sorted(field for field in required if field not in document)
    schema_ok = document.get("schema_version") == expected_schema
    audit_doc = dict(audit) if isinstance(audit, dict) else {}
    if not document:
        artifact_status = "awaiting_paired_replay"
    elif not schema_ok:
        artifact_status = "stale_contract"
    elif missing:
        artifact_status = "incomplete"
    elif audit_doc.get("status") != "passed":
        artifact_status = "audit_blocked"
    else:
        artifact_status = "completed_paired_fresh_replay"
    counts = dict(document.get("counts") or {}) if artifact_status == "completed_paired_fresh_replay" else {}
    checks = {str(key): bool(value) for key, value in dict(document.get("checks") or {}).items()} if artifact_status == "completed_paired_fresh_replay" else {}
    forgetting = dict(document.get("forgetting") or {}) if artifact_status == "completed_paired_fresh_replay" else {}
    promotion = dict(document.get("promotion") or {}) if artifact_status == "completed_paired_fresh_replay" else {}
    return {
        "artifact_status": artifact_status,
        "report_status": str(document.get("status", "not_run")),
        "schema_version": str(document.get("schema_version", "")),
        "missing_fields": missing,
        "counts": counts,
        "checks": checks,
        "forgetting": forgetting,
        "audit_status": str(audit_doc.get("status", "not_embedded")),
        "audit_failures": [str(item) for item in list(audit_doc.get("failures") or [])],
        "report_evidence_hash": str(document.get("report_sha256", ""))[:16],
        "audit_evidence_hash": str(audit_doc.get("audit_sha256", ""))[:16],
        "target_contacted": bool((document.get("runtime") or {}).get("target_contacted")) if document else False,
        "paired_replay_present": bool(forgetting.get("paired_replay_present")),
        "same_canary_route_set": bool(forgetting.get("same_canary_route_set")),
        "promotion_blocked": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg331_information_projection(audit: dict[str, Any], vocabulary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project the ontology/token-vocabulary audit without exposing tokens."""

    document = dict(audit) if isinstance(audit, dict) else {}
    vocab = dict(vocabulary) if isinstance(vocabulary, dict) else {}
    expected_schema = "pg331-information-preservation-audit-v1"
    if not document:
        status = "awaiting_audit"
    elif document.get("schema_version") != expected_schema:
        status = "stale_contract"
    elif document.get("status") != "passed":
        status = "blocked_missing_information"
    else:
        status = "passed_information_audit"
    axes = dict(document.get("token_axes") or {}) if document else {}
    missing_axes = [str(item) for item in list(document.get("failures") or []) if str(item).startswith("axis_missing:")]
    return {
        "artifact_status": status,
        "audit_status": str(document.get("status", "not_run")),
        "audit_schema": str(document.get("schema_version", "")),
        "record_count": int(document.get("record_count", 0) or 0),
        "unique_sequence_ratio": float(document.get("unique_sequence_ratio", 0.0) or 0.0),
        "axis_count": len(axes),
        "missing_axes": missing_axes,
        "axis_coverage": {str(key): float(dict(value).get("coverage", 0.0) or 0.0) for key, value in axes.items() if isinstance(value, dict)},
        "context_target_alignment": float(dict(document.get("context_target_alignment") or {}).get("rate", 0.0) or 0.0),
        "split_isolation_status": str(dict(document.get("split_isolation") or {}).get("status", "unknown")),
        "context_forbidden_literal_count": int(dict(document.get("context_firewall") or {}).get("forbidden_token_count", 0) or 0),
        "vocabulary_status": str(vocab.get("status", "missing")),
        "vocabulary_training_allowed": bool(dict(vocab.get("training_eligibility") or {}).get("allowed", False)),
        "audit_evidence_hash": str(document.get("audit_sha256", ""))[:16],
        "vocabulary_evidence_hash": str(vocab.get("vocabulary_sha256", ""))[:16],
        "promotion_blocked": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg331_readonly_source_projection(
    legacy_manifest_audit: dict[str, Any],
    remote_a800_preflight: dict[str, Any],
) -> dict[str, Any]:
    """Project PG-331 legacy/preflight evidence without promoting it.

    The historical browser manifest is a bounded coverage diagnostic, while
    the remote preflight only describes GPU0 availability.  Keep those facts
    visible as model fields, but force the training gate closed regardless of
    any permissive value in a future source report.
    """

    legacy = dict(legacy_manifest_audit) if isinstance(legacy_manifest_audit, dict) else {}
    preflight = dict(remote_a800_preflight) if isinstance(remote_a800_preflight, dict) else {}
    coverage = dict(legacy.get("coverage") or {})
    route_quality = dict(legacy.get("route_quality_counts") or {})
    missing_observations = [str(item) for item in list(legacy.get("missing_observations") or [])]

    raw_gpu0 = dict(preflight.get("gpu0") or {})
    gpu0_utilization = int(raw_gpu0.get("utilization_percent", 0) or 0)
    gpu0_compute_apps = int(raw_gpu0.get("compute_apps", 0) or 0)
    if not raw_gpu0:
        gpu0_state = "not_observed"
    elif gpu0_utilization == 0 and gpu0_compute_apps == 0:
        gpu0_state = "idle"
    else:
        gpu0_state = "busy"
    gpu0 = {
        "index": int(raw_gpu0.get("index", 0) or 0),
        "name": str(raw_gpu0.get("name", "")),
        "driver": str(raw_gpu0.get("driver", "")),
        "memory_total_mib": int(raw_gpu0.get("memory_total_mib", 0) or 0),
        "memory_used_mib": int(raw_gpu0.get("memory_used_mib", 0) or 0),
        "utilization_percent": gpu0_utilization,
        "compute_apps": gpu0_compute_apps,
        "mode": str(raw_gpu0.get("mode", "")),
        "persistence": str(raw_gpu0.get("persistence", "")),
        "power_state": str(raw_gpu0.get("power_state", "")),
        "resource_status": gpu0_state,
    }
    legacy_projection = {
        "status": str(legacy.get("status", "not_run")),
        "schema_version": str(legacy.get("schema_version", "")),
        "page_count": int(coverage.get("page_count", 0) or 0),
        "route_count": int(coverage.get("route_count", 0) or 0),
        "request_response_row_count": int(coverage.get("request_response_row_count", 0) or 0),
        "script_catalog_count": int(coverage.get("script_catalog_count", 0) or 0),
        "route_quality_counts": {str(key): int(value or 0) for key, value in route_quality.items() if isinstance(value, (int, float))},
        "missing_observations": missing_observations,
        "missing_observation_count": len(missing_observations),
        "audit_evidence_hash": str(legacy.get("audit_sha256", "")),
        "training_allowed": False,
    }
    preflight_projection = {
        "status": str(preflight.get("status", "not_run")),
        "schema_version": str(preflight.get("schema_version", "")),
        "gpu0": gpu0,
        "gpu0_resource_status": gpu0_state,
        "training_allowed": False,
        "training_allowed_now": False,
        "preflight_evidence_hash": str(preflight.get("report_sha256", "")),
    }
    return {
        "legacy_web_manifest": legacy_projection,
        "remote_a800_readonly_preflight": preflight_projection,
        "training_allowed": False,
    }


def _pg331_source_collection_projection(
    report: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Project the PG-331 Pikachu collector artifacts as diagnostics only.

    The collector intentionally writes incomplete/ASK rows because its typed
    evaluator is not present.  This projection accepts either artifact being
    absent, derives only bounded GET/POST shape counts, and never forwards the
    report's promotion values as an authorization to train.
    """

    report_doc = dict(report) if isinstance(report, dict) else {}
    dataset_doc = dict(dataset) if isinstance(dataset, dict) else {}
    report_counts = dict(report_doc.get("counts") or {})
    runtime = dict(report_doc.get("runtime") or {})
    dataset_counts = dict(dataset_doc.get("counts") or {})
    records = [dict(item) for item in list(dataset_doc.get("records") or []) if isinstance(item, dict)]

    def _count(*keys: str, default: int = 0) -> int:
        for source in (report_counts, runtime, dataset_counts, report_doc, dataset_doc):
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)
        return default

    def _token_values(row: dict[str, Any]) -> set[str]:
        return {str(item) for item in list(row.get("context_tokens") or []) if isinstance(item, str)}

    def _derived_method_count(method: str) -> int:
        marker = f"request_method={method}"
        return sum(int(str(row.get("method", "")).upper() == method or marker in _token_values(row)) for row in records)

    get_count = _count("get_count", "GET", "get_routes", default=_derived_method_count("GET"))
    post_count = _count("post_count", "POST", "post_routes", default=_derived_method_count("POST"))
    parameterized_get_count = _count(
        "parameterized_get",
        "parameterized_get_count",
        "parameterized_get_rows",
        default=sum(
            int(
                (
                    "request_method=GET" in _token_values(row)
                    or str(row.get("method", "")).upper() == "GET"
                )
                and (
                    bool(row.get("parameterized_get"))
                    or bool(row.get("parameterized"))
                    or any(
                        token.startswith("request_query_count=") and token.rsplit("=", 1)[-1] not in {"zero", "unknown"}
                        for token in _token_values(row)
                    )
                )
            )
            for row in records
        ),
    )
    route_count = _count("route_count", default=int(runtime.get("route_count", 0) or dataset_counts.get("input", 0) or len(records)))
    target_contacted = _count("target_contacted", "target_contacted_count")
    ask_count = _count("ask_rows", "ask_count", "ask")
    training_eligible_count = _count("training_eligible", "training_eligible_count")
    if not report_doc and not dataset_doc:
        artifact_status = "pending"
    elif not report_doc or not dataset_doc:
        artifact_status = "blocked_incomplete"
    else:
        artifact_status = str(report_doc.get("status", "blocked"))

    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": artifact_status,
        "artifact_status": artifact_status,
        "report_status": str(report_doc.get("status", "pending")),
        "source_collection_status": artifact_status,
        "dataset_status": "available" if dataset_doc else "pending",
        "route_count": route_count,
        "get_count": get_count,
        "post_count": post_count,
        "parameterized_get": parameterized_get_count,
        "parameterized_get_count": parameterized_get_count,
        "target_contacted": target_contacted,
        "target_contacted_count": target_contacted,
        "target_contacted_any": bool(target_contacted),
        "target_contacted_bool": bool(target_contacted),
        "ask": ask_count,
        "ask_count": ask_count,
        "ask_rows": ask_count,
        "training_eligible": False,
        "training_eligible_count": 0,
        "promotion": promotion,
        "training_allowed": False,
        "report_evidence_hash": str(report_doc.get("report_sha256", "")),
        "dataset_evidence_hash": str(dataset_doc.get("dataset_sha256", "")),
    }


def _pg331_typed_source_rows_projection(
    report: dict[str, Any],
    audit: dict[str, Any],
    evaluator_sidecars: dict[str, Any],
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project typed PG-331 rows as local evaluator evidence only.

    Typed positives are useful to show the evaluator/replay contract, but the
    source-row audit and operator review gates still govern training.  Missing
    report, audit, or sidecar artifacts therefore remain ``pending`` and every
    promotion flag is forced false.
    """

    report_doc = dict(report) if isinstance(report, dict) else {}
    audit_doc = dict(audit) if isinstance(audit, dict) else {}
    sidecar_doc = dict(evaluator_sidecars) if isinstance(evaluator_sidecars, dict) else {}
    dataset_doc = dict(dataset) if isinstance(dataset, dict) else {}
    report_counts = dict(report_doc.get("counts") or {})
    audit_counts = dict(audit_doc.get("counts") or {})
    dataset_counts = dict(dataset_doc.get("counts") or {})

    def _count(*keys: str, default: int = 0) -> int:
        for source in (report_counts, audit_counts, dataset_counts, report_doc, audit_doc, dataset_doc):
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)
        return default

    route_reports = [dict(item) for item in list(report_doc.get("route_reports") or sidecar_doc.get("route_reports") or []) if isinstance(item, dict)]
    route_count = _count("route_count", "routes", default=len(route_reports))
    row_count = _count("row_count", "rows", "record_count", default=len(list(dataset_doc.get("records") or [])))
    typed_positive = _count("typed_positive_routes", "typed_positive", "typed_effect_confirmed_count", default=sum(int(bool(item.get("typed_effect_confirmed"))) for item in route_reports))
    training_eligible_count = _count("training_eligible", "training_eligible_count", default=0)

    # Route reports are one entry per route (not one per candidate/reference /
    # negative role), so their suffixes provide a bounded GET/POST count when
    # the runner does not repeat those counts in its report.
    methods: dict[str, set[str]] = {"GET": set(), "POST": set()}
    for item in route_reports:
        route_id = str(item.get("route_id", ""))
        method = str(item.get("method", "")).upper()
        if method not in methods:
            if route_id.endswith("-get"):
                method = "GET"
            elif route_id.endswith("-post"):
                method = "POST"
        if method in methods:
            methods[method].add(route_id or f"route-{len(methods[method])}")
    records = [dict(item) for item in list(dataset_doc.get("records") or []) if isinstance(item, dict)]
    if not route_reports and records:
        for row in records:
            tokens = {str(token) for token in list(row.get("context_tokens") or []) if isinstance(token, str)}
            method = next((str(token).split("=", 1)[1].upper() for token in tokens if token.startswith("request_method=") and "=" in token), "")
            record_id = str(row.get("record_id", ""))
            if method in methods:
                methods[method].add(record_id.rsplit(":", 1)[0] if record_id else f"row-{len(methods[method])}")
    get_count = _count("get_count", "GET", default=len(methods["GET"]))
    post_count = _count("post_count", "POST", default=len(methods["POST"]))
    all_present = bool(report_doc and audit_doc and sidecar_doc)
    artifact_status = str(report_doc.get("status", "pending")) if all_present else "pending"
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": artifact_status,
        "artifact_status": artifact_status,
        "report_status": str(report_doc.get("status", "pending")),
        "audit_status": str(audit_doc.get("status", "pending")),
        "sidecar_status": "available" if sidecar_doc else "pending",
        "route_count": route_count,
        "row_count": row_count,
        "typed_positive": typed_positive,
        "typed_positive_routes": typed_positive,
        "training_eligible": False,
        "training_eligible_count": 0,
        "operator_reviewed": bool(report_doc.get("operator_reviewed")) if report_doc else False,
        "get_count": get_count,
        "post_count": post_count,
        "get": get_count,
        "post": post_count,
        "promotion": promotion,
        "training_allowed": False,
        "report_evidence_hash": str(report_doc.get("report_sha256", "")),
        "audit_evidence_hash": str(audit_doc.get("audit_sha256", "")),
        "sidecar_evidence_hash": str(sidecar_doc.get("artifact_sha256", sidecar_doc.get("report_sha256", ""))),
    }


def _pg332_extended_diagnostic_projection(
    get_report: dict[str, Any],
    get_audit: dict[str, Any],
    get_sidecars: dict[str, Any],
    get_dataset: dict[str, Any],
    post_report: dict[str, Any],
    post_audit: dict[str, Any],
    post_sidecars: dict[str, Any],
    post_dataset: dict[str, Any],
    merged_audit: dict[str, Any],
    information_audit: dict[str, Any],
    capacity_audit: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Project the PG-332 GET/POST and A800 evidence for the research UI.

    This is deliberately an evidence *projection*, not a source-row loader.
    It may inspect the local artifacts to derive bounded counts, but it never
    returns records, token arrays, wire values, response bodies, oracle
    answers, or evaluator-side payload material.  Promotion remains closed
    even when a lane's local source audit says that rows are technically
    valid: PG-332 is a two-implementation diagnostic and still needs a
    family/implementation holdout plus the information gate.
    """

    def _doc(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _count(document: dict[str, Any], *keys: str) -> int:
        counts = _doc(document.get("counts"))
        for source in (counts, document):
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)
        return 0

    def _lane(
        report: dict[str, Any],
        audit: dict[str, Any],
        sidecars: dict[str, Any],
        dataset: dict[str, Any],
        *,
        stateful: bool,
        method: str,
    ) -> dict[str, Any]:
        projection = _pg331_typed_source_rows_projection(report, audit, sidecars, dataset)
        report_doc, audit_doc = _doc(report), _doc(audit)
        runtime = _doc(report_doc.get("runtime"))
        hard_gate = _doc(report_doc.get("hard_gate"))
        validation = _doc(audit_doc.get("validation_counts"))
        split_counts = {
            str(key): int(value or 0)
            for key, value in _doc(audit_doc.get("split_counts")).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        # The GET/POST runners use the same report shape.  Keep only boolean
        # attestations and numeric aggregates; the route/family literals stay
        # evaluator-side.
        projection.update(
            {
                "route_count": _count(report_doc, "route_count"),
                "get_count": 1 if method == "GET" else 0,
                "post_count": 1 if method == "POST" else 0,
                "get": 1 if method == "GET" else 0,
                "post": 1 if method == "POST" else 0,
                "typed_positive": _count(report_doc, "typed_positive_seed_count", "typed_positive_routes"),
                "typed_positive_routes": _count(report_doc, "typed_positive_seed_count", "typed_positive_routes"),
                "sidecar_evidence_hash": str(sidecars.get("sidecars_sha256", sidecars.get("artifact_sha256", ""))),
                "seed_count": _count(report_doc, "seed_count"),
                "role_replay_count": _count(report_doc, "role_replay_count"),
                "source_row_count": _count(report_doc, "source_row_count", "rows"),
                "typed_positive_seed_count": _count(report_doc, "typed_positive_seed_count"),
                "negative_violation_count": _count(report_doc, "negative_violation_count"),
                "split_counts": split_counts,
                "target_contacted": bool(runtime.get("target_contacted")),
                "network_none": str(runtime.get("network_mode", "")) == "none",
                "loopback_only": bool(runtime.get("loopback_only")),
                "fresh_reset_per_role": bool(hard_gate.get("fresh_reset_per_role")),
                "negative_zero_violation": bool(hard_gate.get("negative_zero_violation")),
                "replay_consistent": bool(hard_gate.get("replay_consistent")),
                "role_bound_evidence": bool(hard_gate.get("role_bound_evidence")),
                "typed_candidate_reference": bool(hard_gate.get("typed_candidate_reference")),
                "stateful_disposable": bool(runtime.get("stateful_disposable")) if stateful else False,
                "database_clean_before": bool(hard_gate.get("database_clean_before")) if stateful else False,
                "state_delta_evaluator_only": bool(hard_gate.get("state_delta_evaluator_only")) if stateful else False,
                "teardown_observed": bool(hard_gate.get("teardown_observed")) if stateful else False,
                "claimed_training_eligible_count": int(audit_doc.get("training_eligible_count", 0) or 0),
                "validated_row_count": int(audit_doc.get("record_count", 0) or 0),
                "validation_counts": {
                    key: int(validation.get(key, 0) or 0)
                    for key in (
                        "valid_row_count",
                        "invalid_row_count",
                        "typed_complete_count",
                        "fresh_reset_complete_count",
                        "negative_control_complete_count",
                        "replay_state_complete_count",
                        "operator_reviewed_count",
                    )
                    if isinstance(validation.get(key), (int, float)) and not isinstance(validation.get(key), bool)
                },
                # The UI may show the lane's evaluator evidence, but never an
                # evaluator promotion bit or source-row training authorization.
                "training_eligible": False,
                "training_eligible_count": 0,
                "training_allowed": False,
            }
        )
        return projection

    get_projection = _lane(get_report, get_audit, get_sidecars, get_dataset, stateful=False, method="GET")
    post_projection = _lane(post_report, post_audit, post_sidecars, post_dataset, stateful=True, method="POST")

    merged = _doc(merged_audit)
    info = _doc(information_audit)
    capacity = _doc(capacity_audit)
    a800 = _doc(a800_report)
    merged_validation = _doc(merged.get("validation_counts")) or _doc(merged.get("validation"))
    info_validation = _doc(info.get("validation"))
    merged_split_counts = {
        str(key): int(value or 0)
        for key, value in _doc(merged.get("split_counts")).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    axes_document = _doc(info.get("axes")) or _doc(info.get("axis_quality"))
    axes: dict[str, dict[str, Any]] = {}
    for axis_name, axis_value in axes_document.items():
        axis = _doc(axis_value)
        entropy = _doc(axis.get("entropy"))
        axes[str(axis_name)] = {
            "status": str(axis.get("status", entropy.get("status", "unknown"))),
            "entropy_bits": float(entropy.get("bits", 0.0) or 0.0),
            "unique": int(entropy.get("unique", axis.get("unique", 0)) or 0),
            "unique_ratio": float(entropy.get("unique_ratio", axis.get("unique_ratio", 0.0)) or 0.0),
            "field_count": int(axis.get("field_count", 0) or 0),
        }
    variants: dict[str, int] = {}
    for item in list(capacity.get("variants") or []):
        if not isinstance(item, dict):
            continue
        config = _doc(item.get("config"))
        identifier = str(config.get("id", ""))
        if identifier:
            variants[identifier] = int(config.get("max_length", 0) or 0)

    loss_rows: list[dict[str, Any]] = []
    for item in list(a800.get("loss") or []):
        if not isinstance(item, dict):
            continue
        train = _doc(item.get("train"))
        heldout = _doc(item.get("heldout"))
        loss_rows.append(
            {
                "seed": int(item.get("seed", 0) or 0),
                "train_loss": float(train.get("mean_next_token_loss", 0.0) or 0.0),
                "holdout_loss": float(heldout.get("mean_next_token_loss", 0.0) or 0.0),
                "train_entropy_nats": float(train.get("mean_predictive_entropy_nats", 0.0) or 0.0),
                "holdout_entropy_nats": float(heldout.get("mean_predictive_entropy_nats", 0.0) or 0.0),
                "holdout_context_rows": int(heldout.get("context_row_count", 0) or 0),
                "holdout_unknown_tokens": int(heldout.get("unknown_context_token_count", 0) or 0),
            }
        )

    if not any((get_report, get_audit, get_sidecars, get_dataset, post_report, post_audit, post_sidecars, post_dataset, merged, info, capacity, a800)):
        status = "pending"
    elif not merged or not info or not capacity:
        status = "blocked_incomplete"
    elif str(info.get("status", "")) != "passed" or str(capacity.get("status", "")) != "passed":
        status = "diagnostic_blocked"
    else:
        status = "candidate_only"

    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": status,
        "artifact_status": status,
        "get": get_projection,
        "post": post_projection,
        "cross_impl": {
            "status": str(merged.get("status", "pending")),
            "record_count": int(merged.get("record_count", 0) or 0),
            "claimed_training_eligible_count": int(merged.get("training_eligible_count", 0) or 0),
            "split_counts": merged_split_counts,
            "unique_sequence_count": int(merged.get("unique_sequence_count", 0) or 0),
            "unique_sequence_ratio": float(merged.get("unique_sequence_ratio", 0.0) or 0.0),
            "implementation_count": int(merged_validation.get("implementation_count", info_validation.get("implementation_count", merged.get("implementation_count", 0))) or 0),
            "source_count": int(merged_validation.get("source_count", info_validation.get("source_count", merged.get("source_count", 0))) or 0),
            "valid_row_count": int(merged_validation.get("valid_row_count", info_validation.get("valid_row_count", 0)) or 0),
            "typed_complete_count": int(merged_validation.get("typed_complete_count", info_validation.get("typed_complete_count", 0)) or 0),
            "fresh_reset_complete_count": int(merged_validation.get("fresh_reset_complete_count", info_validation.get("fresh_reset_complete_count", 0)) or 0),
            "negative_control_complete_count": int(merged_validation.get("negative_control_complete_count", info_validation.get("negative_control_complete_count", 0)) or 0),
            "replay_state_complete_count": int(merged_validation.get("replay_state_complete_count", info_validation.get("replay_state_complete_count", 0)) or 0),
            "operator_reviewed_count": int(merged_validation.get("operator_reviewed_count", info_validation.get("operator_reviewed_count", 0)) or 0),
            "context_forbidden_literal_count": int(_doc(merged.get("context_firewall")).get("forbidden_token_count", 0) or 0),
            "context_target_alignment": float(_doc(merged.get("context_target_alignment")).get("rate", 0.0) or 0.0),
            "audit_evidence_hash": str(merged.get("audit_sha256", "")),
        },
        "information": {
            "status": str(info.get("status", "pending")),
            "record_count": int(info.get("record_count", 0) or 0),
            "unique_sequence_ratio": float(info.get("unique_sequence_ratio", 0.0) or 0.0),
            "split_counts": {
                str(key): int(value or 0)
                for key, value in _doc(info.get("split_counts")).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "axes": axes,
            "context_forbidden_literal_count": int(_doc(info.get("context_firewall")).get("forbidden_token_count", 0) or 0),
            "accepted_training_eligible_count": int(_doc(info.get("training_eligibility")).get("accepted_count", 0) or 0),
            "claimed_training_eligible_count": int(_doc(info.get("training_eligibility")).get("claimed_count", 0) or 0),
            "failures": [str(item) for item in list(info.get("failures") or [])][:20],
            "audit_evidence_hash": str(info.get("dataset_information_sha256", "")),
        },
        "capacity": {
            "status": str(capacity.get("status", "pending")),
            "information_audit_status": str(capacity.get("information_audit_status", "pending")),
            "input_vocabulary_size": int(capacity.get("input_vocabulary_size", 0) or 0),
            "target_vocabulary_size": int(capacity.get("target_vocabulary_size", 0) or 0),
            "model_vocabulary_size": int(capacity.get("model_vocabulary_size", 0) or 0),
            "required_context_window": int(capacity.get("required_context_window", 0) or 0),
            "variants": variants,
            "inventory_missing_count": int(capacity.get("inventory_missing_count", 0) or 0),
            "audit_evidence_hash": str(capacity.get("audit_sha256", "")),
        },
        "a800_representation": {
            "status": str(a800.get("status", "pending")),
            "context_only": bool(_doc(a800.get("training")).get("context_only")),
            "target_tokens_read": bool(_doc(a800.get("training")).get("target_tokens_read")),
            "device": str(_doc(a800.get("training")).get("device", "")),
            "epochs": int(_doc(a800.get("training")).get("epochs", 0) or 0),
            "learning_rate": float(_doc(a800.get("training")).get("learning_rate", 0.0) or 0.0),
            "seeds": [int(item.get("seed", 0) or 0) for item in loss_rows],
            "loss": loss_rows,
            "gate_status": str(_doc(a800.get("gate")).get("status", "pending")),
            "information_gate_status": str(_doc(a800.get("gate")).get("information_gate_status", "pending")),
            "information_promotion_gate_passed": bool(_doc(a800.get("gate")).get("information_promotion_gate_passed")),
            "source_implementation_holdout_recorded": bool(_doc(a800.get("gate")).get("source_implementation_holdout_recorded")),
            "report_evidence_hash": str(a800.get("report_sha256", "")),
        },
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg333_webgoat_projection(
    report: dict[str, Any],
    audit: dict[str, Any],
    sidecars: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Project the WebGoat third-implementation method-shape canary.

    The canary is deliberately structural (page vs redirect), so the UI may
    show its GET/POST and fresh/negative/replay evidence without presenting it
    as a vulnerability result.  Raw login values, response bodies and route
    literals remain outside this projection.
    """

    report_doc = dict(report) if isinstance(report, dict) else {}
    audit_doc = dict(audit) if isinstance(audit, dict) else {}
    sidecar_doc = dict(sidecars) if isinstance(sidecars, dict) else {}
    dataset_doc = dict(dataset) if isinstance(dataset, dict) else {}
    counts = dict(report_doc.get("counts") or {})
    hard_gate = dict(report_doc.get("hard_gate") or {})
    runtime = dict(report_doc.get("runtime") or {})
    audit_split = {
        str(key): int(value or 0)
        for key, value in dict(audit_doc.get("split_counts") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    methods = {
        str(key).upper(): int(value or 0)
        for key, value in dict(report_doc.get("methods") or {}).items()
        if str(key).upper() in {"GET", "POST"} and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not report_doc and not audit_doc and not sidecar_doc and not dataset_doc:
        status = "pending"
    elif not report_doc or not audit_doc or not sidecar_doc or not dataset_doc:
        status = "blocked_incomplete"
    elif audit_doc.get("status") != "passed":
        status = "diagnostic_blocked"
    else:
        status = "completed_diagnostic_only"
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": status,
        "artifact_status": status,
        "report_status": str(report_doc.get("status", "pending")),
        "audit_status": str(audit_doc.get("status", "pending")),
        "sidecar_status": str(sidecar_doc.get("status", "pending")),
        "seed_count": int(counts.get("seed_count", 0) or 0),
        "route_count": int(counts.get("route_count", 0) or 0),
        "role_replay_count": int(counts.get("role_replay_count", 0) or 0),
        "source_row_count": int(counts.get("source_row_count", 0) or 0),
        "typed_positive_route_seed_count": int(counts.get("typed_positive_route_seed_count", 0) or 0),
        "negative_violation_count": int(counts.get("negative_violation_count", 0) or 0),
        "claimed_training_eligible_count": int(counts.get("training_eligible_row_count", 0) or 0),
        "validated_row_count": int(audit_doc.get("record_count", 0) or 0),
        "audit_training_eligible_count": int(audit_doc.get("training_eligible_count", 0) or 0),
        "methods": methods,
        "split_counts": audit_split,
        "unique_sequence_count": int(audit_doc.get("unique_sequence_count", 0) or 0),
        "unique_sequence_ratio": float(audit_doc.get("unique_sequence_ratio", 0.0) or 0.0),
        "context_target_alignment": float(dict(audit_doc.get("context_target_alignment") or {}).get("rate", 0.0) or 0.0),
        "context_forbidden_literal_count": int(dict(audit_doc.get("context_firewall") or {}).get("forbidden_token_count", 0) or 0),
        "fresh_reset_per_role": bool(hard_gate.get("fresh_reset_per_role")),
        "typed_candidate_reference": bool(hard_gate.get("typed_candidate_reference")),
        "negative_zero_violation": bool(hard_gate.get("negative_zero_violation")),
        "replay_consistent": bool(hard_gate.get("replay_consistent")),
        "role_bound_evidence": bool(hard_gate.get("role_bound_evidence")),
        "network_none": bool(hard_gate.get("network_none")) or str(runtime.get("network_mode", "")) == "none",
        "no_bind_or_volume": bool(hard_gate.get("no_bind_or_volume")) or not bool(runtime.get("bind_or_volume_mounts")),
        "target_contacted": bool(runtime.get("target_contacted")),
        "internal_disposable_db": bool(runtime.get("disposable_internal_db")),
        "hard_gate_status": str(hard_gate.get("status", "pending")),
        "report_evidence_hash": str(report_doc.get("report_sha256", "")),
        "audit_evidence_hash": str(audit_doc.get("audit_sha256", "")),
        "dataset_evidence_hash": str(dataset_doc.get("dataset_sha256", "")),
        "sidecar_evidence_hash": str(sidecar_doc.get("sidecars_sha256", "")),
        "training_eligible": False,
        "training_eligible_count": 0,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg333_cross_impl_projection(
    dataset: dict[str, Any],
    source_audit: dict[str, Any],
    information_audit: dict[str, Any],
    vocabulary: dict[str, Any],
    capacity: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Project the three-implementation PG-333 diagnostic without raw rows.

    The merged dataset is useful for information/holdout diagnostics, but it
    is deliberately not a training manifest: this projection exposes only
    bounded counts, hashes, axis summaries and context-only smoke metrics.
    """

    docs = [item if isinstance(item, dict) else {} for item in (dataset, source_audit, information_audit, vocabulary, capacity, a800_report)]
    merged, source, info, vocab, cap, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:5]):
        status = "blocked_incomplete"
    elif str(info.get("status", "")) != "diagnostic":
        status = "diagnostic_blocked"
    else:
        status = "completed_diagnostic_only"

    counts = dict(merged.get("counts") or {})
    axis_quality = dict(info.get("axis_quality") or {})
    axis_entropy: dict[str, float] = {}
    axis_sequence_ratio: dict[str, float] = {}
    for axis, value in axis_quality.items():
        if not isinstance(value, dict):
            continue
        entropy = dict(value.get("entropy") or {})
        sequence = dict(value.get("sequence") or {})
        if isinstance(entropy.get("bits"), (int, float)) and not isinstance(entropy.get("bits"), bool):
            axis_entropy[str(axis)] = float(entropy.get("bits", 0.0) or 0.0)
        if isinstance(sequence.get("unique_ratio"), (int, float)) and not isinstance(sequence.get("unique_ratio"), bool):
            axis_sequence_ratio[str(axis)] = float(sequence.get("unique_ratio", 0.0) or 0.0)

    smoke_training = dict(smoke.get("training") or {})
    smoke_gate = dict(smoke.get("gate") or {})
    smoke_losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("heldout") or {}) for item in smoke_losses]
    heldout_loss = [float(item.get("mean_next_token_loss", 0.0) or 0.0) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item.get("mean_predictive_entropy_nats", 0.0) or 0.0) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": status,
        "artifact_status": status,
        "merged_record_count": int(counts.get("records", 0) or 0),
        "merged_training_claimed_count": int(counts.get("training_eligible", 0) or 0),
        "implementation_count": int(counts.get("implementations", 0) or 0),
        "family_count": int(counts.get("families", 0) or 0),
        "implementation_counts": {str(k): int(v or 0) for k, v in dict(merged.get("implementation_counts") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "family_counts": {str(k): int(v or 0) for k, v in dict(merged.get("family_counts") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "source_audit_status": str(source.get("status", "pending")),
        "source_valid_rows": int(source.get("record_count", 0) or 0),
        "source_training_eligible_rows": int(source.get("training_eligible_count", 0) or 0),
        "source_split_counts": {str(k): int(v or 0) for k, v in dict(source.get("split_counts") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "information_audit_status": str(info.get("status", "pending")),
        "information_failures": [str(item) for item in list(info.get("failures") or []) if str(item) in {"capacity_dataset_window", "vocabulary_context_coverage", "vocabulary_target_coverage", "context_firewall", "split_isolation", "empty:dataset", "empty:ontology_axes"}],
        "unique_sequence_ratio": float(info.get("unique_sequence_ratio", 0.0) or 0.0),
        "axis_entropy_bits": axis_entropy,
        "axis_sequence_ratio": axis_sequence_ratio,
        "vocabulary_status": str(vocab.get("status", "pending")),
        "context_vocabulary_size": int(dict(vocab.get("counts") or {}).get("context_total", 0) or 0),
        "target_vocabulary_size": int(dict(vocab.get("counts") or {}).get("target_total", 0) or 0),
        "ontology_inventory_size": int(dict(vocab.get("counts") or {}).get("ontology_inventory", 0) or 0),
        "capacity_status": str(cap.get("status", "pending")),
        "required_context_window": int(cap.get("required_context_window", 0) or 0),
        "model_vocabulary_size": int(cap.get("model_vocabulary_size", 0) or 0),
        "capacity_passing_variants": [str((item.get("config") or {}).get("id")) for item in list(cap.get("variants") or []) if isinstance(item, dict) and item.get("capacity_pass") is True],
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(smoke_gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(smoke_gate.get("context_row_count", 0) or 0),
        "a800_holdout_rows": max([int(item.get("context_row_count", 0) or 0) for item in heldout] or [0]),
        "a800_epochs": int(smoke_training.get("epochs", 0) or 0),
        "a800_heldout_loss_min": min(heldout_loss) if heldout_loss else 0.0,
        "a800_heldout_loss_max": max(heldout_loss) if heldout_loss else 0.0,
        "a800_heldout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_heldout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_target_tokens_read": bool(smoke_training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(smoke_gate.get("checks", {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg334_process_token_projection(
    dataset: dict[str, Any],
    information_audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Expose only bounded PG-334 process-token/ASK diagnostics.

    The dataset is a controlled fixture projection.  This deliberately never
    emits context/target tokens, family names, slot names, payloads, responses,
    or evaluator material into the research UI, and it can never promote.
    """
    docs = [item if isinstance(item, dict) else {} for item in (dataset, information_audit, vocabulary, a800_report)]
    data, audit, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif str(audit.get("status", "")) != "diagnostic_only":
        status = "diagnostic_blocked"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    audit_counts = dict(audit.get("counts") or {})
    smoke_gate = dict(smoke.get("gate") or {})
    smoke_training = dict(smoke.get("training") or {})
    smoke_losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("heldout") or {}) for item in smoke_losses]
    heldout_losses = [float(item.get("mean_next_token_loss")) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item.get("mean_predictive_entropy_nats")) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("total", 0) or 0),
        "train_count": int(counts.get("train", 0) or 0),
        "holdout_count": int(counts.get("implementation_holdout", 0) or 0),
        "pre_question_count": int(counts.get("pre", 0) or 0),
        "post_observation_count": int(counts.get("post", 0) or 0),
        "negative_count": int(counts.get("negative", 0) or 0),
        "unique_context_sequences": int(audit_counts.get("unique_context_sequences", 0) or 0),
        "unique_target_sequences": int(audit_counts.get("unique_target_sequences", 0) or 0),
        "information_audit_status": str(audit.get("status", "pending")),
        "audit_checks_passed": sum(bool(value) for value in dict(audit.get("checks") or {}).values()),
        "audit_checks_total": len(dict(audit.get("checks") or {})),
        "context_vocabulary_size": len(list(vocab.get("context_tokens") or [])),
        "target_vocabulary_size": len(list(vocab.get("target_tokens") or [])),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(smoke_gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(smoke_gate.get("context_row_count", 0) or 0),
        "a800_holdout_rows": max([int(item.get("context_row_count", 0) or 0) for item in heldout] or [0]),
        "a800_epochs": int(smoke_training.get("epochs", 0) or 0),
        "a800_heldout_loss_min": min(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_loss_max": max(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_heldout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_target_tokens_read": bool(smoke_training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(smoke_gate.get("checks", {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg335_real_process_projection(
    dataset: dict[str, Any],
    information_audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for source-grounded ASK/failure diagnostics."""
    docs = [item if isinstance(item, dict) else {} for item in (dataset, information_audit, vocabulary, a800_report)]
    data, audit, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif str(audit.get("status", "")) != "diagnostic_only":
        status = "diagnostic_blocked"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    audit_counts = dict(audit.get("counts") or {})
    smoke_gate = dict(smoke.get("gate") or {})
    smoke_training = dict(smoke.get("training") or {})
    smoke_losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("heldout") or {}) for item in smoke_losses]
    losses = [float(item.get("mean_next_token_loss")) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    entropy = [float(item.get("mean_predictive_entropy_nats")) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "source_row_count": int(counts.get("source_rows", 0) or 0),
        "record_count": int(counts.get("total", 0) or 0),
        "train_count": int(counts.get("train", 0) or 0),
        "holdout_count": int(counts.get("implementation_holdout", 0) or 0),
        "observed_count": int(counts.get("observed", 0) or 0),
        "ask_count": int(counts.get("ask", 0) or 0),
        "failure_count": int(counts.get("failure", 0) or 0),
        "negative_review_count": int(counts.get("negative_review", 0) or 0),
        "axis_mask_counts": {str(key): int(value or 0) for key, value in dict(counts.get("axis_masks") or {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)},
        "unique_context_sequences": int(audit_counts.get("unique_context_sequences", 0) or 0),
        "unique_target_sequences": int(audit_counts.get("unique_target_sequences", 0) or 0),
        "context_token_entropy_bits": float(audit_counts.get("context_token_entropy_bits", 0.0) or 0.0),
        "information_audit_status": str(audit.get("status", "pending")),
        "audit_checks_passed": sum(bool(value) for value in dict(audit.get("checks") or {}).values()),
        "audit_checks_total": len(dict(audit.get("checks") or {})),
        "context_vocabulary_size": len(list(vocab.get("context_tokens") or [])),
        "target_vocabulary_size": len(list(vocab.get("target_tokens") or [])),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(smoke_gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(smoke_gate.get("context_row_count", 0) or 0),
        "a800_holdout_rows": max([int(item.get("context_row_count", 0) or 0) for item in heldout] or [0]),
        "a800_epochs": int(smoke_training.get("epochs", 0) or 0),
        "a800_heldout_loss_min": min(losses) if losses else 0.0,
        "a800_heldout_loss_max": max(losses) if losses else 0.0,
        "a800_heldout_entropy_min": min(entropy) if entropy else 0.0,
        "a800_heldout_entropy_max": max(entropy) if entropy else 0.0,
        "a800_target_tokens_read": bool(smoke_training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(smoke_gate.get("checks", {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg336_real_failure_process_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for real failure/ASK/negative process evidence.

    PG-336 is intentionally a single-implementation diagnostic track.  The
    UI receives counts, entropy and smoke metrics only; no trace rows, route
    identities, tokens, wire, payload, response or oracle literals cross this
    boundary.
    """
    docs = [item if isinstance(item, dict) else {} for item in (dataset, audit, vocabulary, a800_report)]
    data, audit_doc, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif str(audit_doc.get("status", "")) != "diagnostic_only":
        status = "diagnostic_blocked"
    elif not smoke:
        status = "blocked_incomplete"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    audit_counts = dict(audit_doc.get("counts") or {})
    smoke_gate = dict(smoke.get("gate") or {})
    smoke_training = dict(smoke.get("training") or {})
    smoke_losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("heldout") or {}) for item in smoke_losses]
    heldout_losses = [float(item.get("mean_next_token_loss")) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item.get("mean_predictive_entropy_nats")) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("total", 0) or 0),
        "probe_observed_count": int(counts.get("probe_observed", 0) or 0),
        "failure_repair_count": int(counts.get("failure_repair", 0) or 0),
        "negative_review_count": int(counts.get("negative_review", 0) or 0),
        "ask_preflight_count": int(counts.get("ask_preflight", 0) or 0),
        "real_failure_trace_count": int(dict(data.get("source") or {}).get("real_failure_trace_count", 0) or 0),
        "real_negative_trace_count": int(dict(data.get("source") or {}).get("real_negative_trace_count", 0) or 0),
        "train_count": int(counts.get("train", 0) or 0),
        "seed_holdout_count": int(counts.get("seed_holdout", 0) or 0),
        "get_count": int(counts.get("get", 0) or 0),
        "post_count": int(counts.get("post", 0) or 0),
        "unique_context_sequences": int(audit_counts.get("unique_context_sequences", 0) or 0),
        "unique_target_sequences": int(audit_counts.get("unique_target_sequences", 0) or 0),
        "context_token_entropy_bits": float(audit_counts.get("context_token_entropy_bits", 0.0) or 0.0),
        "target_token_entropy_bits": float(audit_counts.get("target_token_entropy_bits", 0.0) or 0.0),
        "axis_presence_entropy_bits": {str(key): float(value or 0.0) for key, value in dict(audit_doc.get("axis_presence_entropy_bits") or {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)},
        "information_audit_status": str(audit_doc.get("status", "pending")),
        "audit_checks_passed": sum(bool(value) for value in dict(audit_doc.get("checks") or {}).values()),
        "audit_checks_total": len(dict(audit_doc.get("checks") or {})),
        "scientific_gate_status": str(dict(audit_doc.get("scientific_gate") or {}).get("status", "blocked")),
        "independent_implementation_holdout": bool(dict(audit_doc.get("scientific_gate") or {}).get("independent_implementation_holdout", False)),
        "context_vocabulary_size": len(list(vocab.get("context_tokens") or [])),
        "target_vocabulary_size": len(list(vocab.get("target_tokens") or [])),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(smoke_gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(smoke_gate.get("context_row_count", 0) or 0),
        "a800_holdout_rows": max([int(item.get("context_row_count", 0) or 0) for item in heldout] or [0]),
        "a800_epochs": int(smoke_training.get("epochs", 0) or 0),
        "a800_heldout_loss_min": min(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_loss_max": max(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_heldout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_target_tokens_read": bool(smoke_training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(dict(smoke_gate.get("checks") or {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg337_cross_impl_process_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for the PG-337 cross-implementation process track.

    Only aggregate counts, entropy, holdout and A800 metadata are exposed to
    the research UI.  Records, tokens, route identifiers and evaluator-side
    state remain outside this projection.
    """
    docs = [item if isinstance(item, dict) else {} for item in (dataset, audit, vocabulary, a800_report)]
    data, audit_doc, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif str(audit_doc.get("status", "")) != "diagnostic_only":
        status = "diagnostic_blocked"
    elif not smoke:
        status = "blocked_incomplete"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    audit_counts = dict(audit_doc.get("counts") or {})
    source = dict(data.get("source") or {})
    policy = dict(data.get("process_policy") or {})
    smoke_gate = dict(smoke.get("gate") or {})
    smoke_training = dict(smoke.get("training") or {})
    smoke_losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("heldout") or {}) for item in smoke_losses]
    heldout_losses = [float(item.get("mean_next_token_loss")) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item.get("mean_predictive_entropy_nats")) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("total", 0) or 0),
        "train_count": int(counts.get("train", 0) or 0),
        "implementation_holdout_count": int(counts.get("implementation_holdout", 0) or 0),
        "failure_repair_count": int(counts.get("failure_repair", 0) or 0),
        "negative_review_count": int(counts.get("negative_review", 0) or 0),
        "ask_preflight_count": int(counts.get("ask_preflight", 0) or 0),
        "real_dvwa_failure_rows": int(counts.get("real_dvwa_failure_rows", 0) or 0),
        "accepted_training_rows": int(policy.get("accepted_training_rows", 0) or 0),
        "unique_context_sequences": int(audit_doc.get("unique_context_sequences", 0) or 0),
        "unique_target_sequences": int(audit_doc.get("unique_target_sequences", 0) or 0),
        "context_token_entropy_bits": float(audit_doc.get("context_token_entropy_bits", 0.0) or 0.0),
        "target_token_entropy_bits": float(audit_doc.get("target_token_entropy_bits", 0.0) or 0.0),
        "information_audit_status": str(audit_doc.get("status", "pending")),
        "audit_checks_passed": sum(bool(value) for value in dict(audit_doc.get("checks") or {}).values()),
        "audit_checks_total": len(dict(audit_doc.get("checks") or {})),
        "independent_implementation_holdout": bool(source.get("independent_implementation_holdout")),
        "context_vocabulary_size": len(list(vocab.get("context_tokens") or [])),
        "target_vocabulary_size": len(list(vocab.get("target_tokens") or [])),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(smoke_gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(smoke_gate.get("context_row_count", 0) or 0),
        "a800_holdout_rows": max([int(item.get("context_row_count", 0) or 0) for item in heldout] or [0]),
        "a800_epochs": int(smoke_training.get("epochs", 0) or 0),
        "a800_heldout_loss_min": min(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_loss_max": max(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_heldout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_target_tokens_read": bool(smoke_training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(dict(smoke_gate.get("checks") or {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg338_information_preserving_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for the full-axis PG-338 diagnostic track."""

    docs = [item if isinstance(item, dict) else {} for item in (dataset, audit, vocabulary, a800_report)]
    data, audit_doc, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif str(audit_doc.get("status", "")) != "diagnostic_only":
        status = "diagnostic_blocked"
    elif not smoke:
        status = "blocked_incomplete"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    split_counts = dict(audit_doc.get("split_counts") or {})
    source = dict(data.get("source") or {})
    policy = dict(data.get("process_policy") or {})
    gate = dict(smoke.get("gate") or {})
    training = dict(smoke.get("training") or {})
    losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("heldout") or {}) for item in losses]
    heldout_losses = [float(item["mean_next_token_loss"]) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item["mean_predictive_entropy_nats"]) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    axis_entropy: dict[str, dict[str, Any]] = {}
    for axis, details in dict(audit_doc.get("axis_entropy") or {}).items():
        if not isinstance(details, dict):
            continue
        entropy = dict(details.get("entropy") or {})
        axis_entropy[str(axis)] = {
            "coverage": float(details.get("coverage", 0.0) or 0.0),
            "bits": float(entropy["bits"]) if isinstance(entropy.get("bits"), (int, float)) else None,
            "unique": int(entropy.get("unique", 0) or 0),
            "ablation_changed_rate": float(dict(details.get("field_ablation") or {}).get("changed_rate", 0.0) or 0.0),
        }
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("total", 0) or 0),
        "train_count": int(split_counts.get("train", counts.get("train", 0)) or 0),
        "implementation_holdout_count": int(split_counts.get("implementation_holdout", counts.get("implementation_holdout", 0)) or 0),
        "probe_observed_count": int(counts.get("probe_observed", 0) or 0),
        "failure_repair_count": int(counts.get("failure_repair", 0) or 0),
        "negative_review_count": int(counts.get("negative_review", 0) or 0),
        "full_axis_rows": int(counts.get("full_axis_rows", 0) or 0),
        "accepted_training_rows": int(policy.get("accepted_training_rows", 0) or 0),
        "context_length": dict(audit_doc.get("context_length") or {}),
        "unique_context_sequences": int(audit_doc.get("unique_context_sequences", 0) or 0),
        "unique_target_sequences": int(audit_doc.get("unique_target_sequences", 0) or 0),
        "context_token_entropy_bits": float(dict(audit_doc.get("context_token_entropy_bits") or {}).get("bits", 0.0) or 0.0),
        "target_token_entropy_bits": float(dict(audit_doc.get("target_token_entropy_bits") or {}).get("bits", 0.0) or 0.0),
        "axis_entropy": axis_entropy,
        "information_audit_status": str(audit_doc.get("status", "pending")),
        "independent_implementation_holdout": bool(source.get("independent_implementation_holdout")),
        "context_vocabulary_size": int(vocab.get("context_vocabulary_size", len(list(vocab.get("context_tokens") or []))) or 0),
        "target_vocabulary_size": int(vocab.get("target_vocabulary_size", len(list(vocab.get("target_tokens") or []))) or 0),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(gate.get("context_row_count", 0) or 0),
        "a800_holdout_rows": max([int(item.get("context_row_count", 0) or 0) for item in heldout] or [0]),
        "a800_epochs": int(training.get("epochs", 0) or 0),
        "a800_required_context_window": int(dict(smoke.get("context_capacity_requirement") or {}).get("required_max_length", 0) or 0),
        "a800_heldout_loss_min": min(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_loss_max": max(heldout_losses) if heldout_losses else 0.0,
        "a800_heldout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_heldout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_target_tokens_read": bool(training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(dict(gate.get("checks") or {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg339_multi_shape_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for PG-339 multi-shape context-only diagnostics."""

    docs = [item if isinstance(item, dict) else {} for item in (dataset, audit, vocabulary, a800_report)]
    data, audit_doc, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif not str(audit_doc.get("status", "")).startswith("diagnostic_only"):
        status = "diagnostic_blocked"
    elif not smoke:
        status = "blocked_incomplete"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    audit_counts = dict(audit_doc.get("counts") or {})
    gate = dict(smoke.get("gate") or {})
    training = dict(smoke.get("training") or {})
    losses: list[dict[str, Any]] = []
    for item in list(smoke.get("loss") or []):
        if isinstance(item, dict):
            losses.append(dict(item))
    heldout = [dict(item.get("shape_holdout") or {}) for item in losses]
    heldout_losses = [float(item["mean_next_token_loss"]) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item["mean_predictive_entropy_nats"]) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    axis_entropy: dict[str, dict[str, Any]] = {}
    for axis, details in dict(audit_doc.get("axis_entropy") or {}).items():
        if not isinstance(details, dict):
            continue
        presence = dict(details.get("presence_entropy") or {})
        field = dict(details.get("field_status_entropy") or {})
        ablation = dict(details.get("field_ablation") or {})
        axis_entropy[str(axis)] = {
            "presence_bits": float(presence["bits"]) if isinstance(presence.get("bits"), (int, float)) else None,
            "field_status_bits": float(field["bits"]) if isinstance(field.get("bits"), (int, float)) else None,
            "field_status_unique": int(field.get("unique", 0) or 0),
            "ablation_changed_rate": float(ablation.get("changed_rate", 0.0) or 0.0),
        }
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("accepted_rows", 0) or 0),
        "input_row_count": int(counts.get("input_rows", 0) or 0),
        "train_count": int(counts.get("train_rows", audit_counts.get("train", 0)) or 0),
        "shape_holdout_count": int(counts.get("shape_holdout_rows", audit_counts.get("shape_holdout", 0)) or 0),
        "duplicate_row_count": int(counts.get("duplicate_rows", 0) or 0),
        "accepted_training_rows": 0,
        "axis_entropy": axis_entropy,
        "split_implementation_isolation": dict(audit_doc.get("split_implementation_isolation") or {}),
        "information_audit_status": str(audit_doc.get("status", "pending")),
        "context_vocabulary_size": int(vocab.get("context_vocabulary_size", len(list(vocab.get("context_tokens") or []))) or 0),
        "unknown_context_token_count": int(gate.get("unknown_context_token_count", 0) or 0),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(dict(gate.get("split_counts") or {}).get("train", 0) or 0),
        "a800_shape_holdout_rows": int(dict(gate.get("split_counts") or {}).get("shape_holdout", 0) or 0),
        "a800_epochs": int(training.get("epochs", 0) or 0),
        "a800_required_context_window": int(dict(smoke.get("context_capacity_requirement") or {}).get("required_max_length", 0) or 0),
        "a800_shape_holdout_loss_min": min(heldout_losses) if heldout_losses else 0.0,
        "a800_shape_holdout_loss_max": max(heldout_losses) if heldout_losses else 0.0,
        "a800_shape_holdout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_shape_holdout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_target_tokens_read": bool(training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(dict(gate.get("checks") or {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg340_balanced_axis_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for the PG-340 balanced implementation split.

    Only aggregate counts, entropy summaries, capacity and hash-lock status are
    exposed.  This projection never emits records, token sequences, raw wire,
    payloads, responses, oracle answers or implementation labels.
    """

    docs = [item if isinstance(item, dict) else {} for item in (dataset, audit, vocabulary, a800_report)]
    data, audit_doc, vocab, smoke = docs
    if not any(docs):
        status = "pending"
    elif not all(docs[:3]):
        status = "blocked_incomplete"
    elif not str(audit_doc.get("status", "")).startswith("diagnostic_only"):
        status = "diagnostic_blocked"
    elif not smoke:
        status = "blocked_incomplete"
    else:
        status = "completed_diagnostic_only"
    counts = dict(data.get("counts") or {})
    audit_counts = dict(audit_doc.get("counts") or {})
    gate = dict(smoke.get("gate") or {})
    training = dict(smoke.get("training") or {})
    losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    heldout = [dict(item.get("shape_holdout") or {}) for item in losses]
    heldout_losses = [float(item["mean_next_token_loss"]) for item in heldout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    heldout_entropy = [float(item["mean_predictive_entropy_nats"]) for item in heldout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    axis_entropy: dict[str, dict[str, Any]] = {}
    for axis, details in dict(audit_doc.get("axis_entropy") or {}).items():
        if not isinstance(details, dict):
            continue
        train_seq = dict(details.get("train_sequence_entropy") or {})
        holdout_seq = dict(details.get("shape_holdout_sequence_entropy") or {})
        field = dict(details.get("field_status_entropy") or {})
        ablation = dict(details.get("field_ablation") or {})
        axis_entropy[str(axis)] = {
            "train_sequence_bits": float(train_seq["bits"]) if isinstance(train_seq.get("bits"), (int, float)) else None,
            "shape_holdout_sequence_bits": float(holdout_seq["bits"]) if isinstance(holdout_seq.get("bits"), (int, float)) else None,
            "train_sequence_unique": int(train_seq.get("unique", 0) or 0),
            "shape_holdout_sequence_unique": int(holdout_seq.get("unique", 0) or 0),
            "field_status_bits": float(field["bits"]) if isinstance(field.get("bits"), (int, float)) else None,
            "field_status_unique": int(field.get("unique", 0) or 0),
            "ablation_changed_rate": float(ablation.get("changed_rate", 0.0) or 0.0),
        }
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("accepted_rows", 0) or 0),
        "input_row_count": int(counts.get("input_rows", 0) or 0),
        "train_count": int(counts.get("train_rows", audit_counts.get("train", 0)) or 0),
        "shape_holdout_count": int(counts.get("shape_holdout_rows", audit_counts.get("shape_holdout", 0)) or 0),
        "duplicate_row_count": int(counts.get("duplicate_rows", 0) or 0),
        "train_implementation_count": int(counts.get("train_implementation_count", audit_counts.get("train_implementation_count", 0)) or 0),
        "holdout_implementation_count": int(counts.get("holdout_implementation_count", audit_counts.get("holdout_implementation_count", 0)) or 0),
        "accepted_training_rows": 0,
        "axis_entropy": axis_entropy,
        "split_implementation_isolation": dict(audit_doc.get("split_implementation_isolation") or {}),
        "information_audit_status": str(audit_doc.get("status", "pending")),
        "context_vocabulary_size": int(vocab.get("context_vocabulary_size", len(list(vocab.get("context_tokens") or []))) or 0),
        "unknown_context_token_count": int(gate.get("unknown_context_token_count", 0) or 0),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(dict(gate.get("split_counts") or {}).get("train", 0) or 0),
        "a800_shape_holdout_rows": int(dict(gate.get("split_counts") or {}).get("shape_holdout", 0) or 0),
        "a800_epochs": int(training.get("epochs", 0) or 0),
        "a800_required_context_window": int(dict(smoke.get("context_capacity_requirement") or {}).get("required_max_length", 0) or 0),
        "a800_shape_holdout_loss_min": min(heldout_losses) if heldout_losses else 0.0,
        "a800_shape_holdout_loss_max": max(heldout_losses) if heldout_losses else 0.0,
        "a800_shape_holdout_entropy_min": min(heldout_entropy) if heldout_entropy else 0.0,
        "a800_shape_holdout_entropy_max": max(heldout_entropy) if heldout_entropy else 0.0,
        "a800_entropy_drop_max": float(dict(smoke.get("entropy_gate") or {}).get("max_relative_entropy_drop", 0.0) or 0.0),
        "a800_target_tokens_read": bool(training.get("target_tokens_read", False)),
        "a800_hash_lock_passed": bool(dict(gate.get("checks") or {}).get("data_code_vocab_rules_hashes_locked")),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg341_target_conditioned_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bounded projection for the PG-341 two-view target decoder.

    The research UI may show target coverage and aggregate candidate metrics,
    but it must not expose context/target sequences, source rows, sidecars,
    routes, evaluator answers or raw wire material.
    """

    data = dataset if isinstance(dataset, dict) else {}
    audit_doc = audit if isinstance(audit, dict) else {}
    vocab = vocabulary if isinstance(vocabulary, dict) else {}
    reports = [item for item in a800_reports if isinstance(item, dict) and item]
    if not data and not audit_doc and not vocab and not reports:
        status = "pending"
    elif not data or not audit_doc or not vocab:
        status = "blocked_incomplete"
    else:
        status = str(audit_doc.get("status", "diagnostic_blocked"))
    counts = dict(data.get("counts") or {})
    coverage = dict(audit_doc.get("target_coverage") or {})
    coarse = dict(audit_doc.get("coarse_process") or {})
    full_axis = dict(audit_doc.get("full_axis") or {})
    latest = reports[0] if reports else {}
    candidates = [dict(item) for item in list(latest.get("candidates") or []) if isinstance(item, dict)]
    holdout = [dict(item.get("implementation_holdout") or {}) for item in candidates]
    def _numeric(name: str) -> list[float]:
        return [float(item[name]) for item in holdout if isinstance(item.get(name), (int, float))]
    promotion = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("total", 0) or 0),
        "coarse_process_count": int(counts.get("coarse_process", 0) or 0),
        "full_axis_count": int(counts.get("full_axis", 0) or 0),
        "coarse_train_count": int(counts.get("coarse_train", 0) or 0),
        "coarse_holdout_count": int(counts.get("coarse_holdout", 0) or 0),
        "full_axis_train_count": int(counts.get("full_axis_train", 0) or 0),
        "full_axis_holdout_count": int(counts.get("full_axis_holdout", 0) or 0),
        "coarse_diagnostic_training_allowed": bool(coarse.get("diagnostic_training_allowed", False)),
        "full_axis_target_training_allowed": bool(full_axis.get("target_training_allowed", False)),
        "coarse_train_target_coverage": dict(coverage.get("coarse_train_complete") if isinstance(coverage.get("coarse_train_complete"), dict) else {"complete": bool(coverage.get("coarse_train_complete", False))}),
        "full_axis_train_target_coverage": dict(coverage.get("full_axis_train_complete") if isinstance(coverage.get("full_axis_train_complete"), dict) else {"complete": bool(coverage.get("full_axis_train_complete", False))}),
        "context_vocabulary_size": len(list(vocab.get("context_tokens") or [])),
        "target_vocabulary_size": len(list(vocab.get("target_tokens") or [])),
        "a800_status": str(latest.get("status", "pending")),
        "a800_report_count": len(reports),
        "a800_epochs": int(dict(latest.get("training") or {}).get("epochs", 0) or 0),
        "a800_target_tokens_read": bool(dict(latest.get("training") or {}).get("target_tokens_read", False)),
        "a800_track": str(dict(latest.get("training") or {}).get("track", "pending")),
        "a800_holdout_ask_recall_min": min(_numeric("missing_question_recall")) if _numeric("missing_question_recall") else None,
        "a800_holdout_positive_recall_min": min(_numeric("positive_recall")) if _numeric("positive_recall") else None,
        "a800_holdout_false_allow_max": max(_numeric("hard_negative_false_allow")) if _numeric("hard_negative_false_allow") else 0,
        "a800_holdout_safe_reject_min": min(_numeric("safe_reject_rate")) if _numeric("safe_reject_rate") else None,
        "a800_hash_lock_passed": bool(dict(latest.get("gate") or {}).get("checks", {}).get("data_code_vocab_rules_hashes_locked")),
        "full_axis_gap": dict(latest.get("full_axis_target_gap") or {"status": "pending"}),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg342_full_axis_failure_repair_projection(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    vocabulary: dict[str, Any],
    a800_report: dict[str, Any],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    """Bounded projection for PG-342 full-axis failure/repair diagnostics.

    The UI receives only counts, gate states, aggregate losses/entropy and
    hashes.  It never emits context/target sequences, source identities,
    sidecars, routes, wire values, payloads, response bodies or oracle answers.
    """

    data = dataset if isinstance(dataset, dict) else {}
    audit_doc = audit if isinstance(audit, dict) else {}
    vocab = vocabulary if isinstance(vocabulary, dict) else {}
    smoke = a800_report if isinstance(a800_report, dict) else {}
    source = source_report if isinstance(source_report, dict) else {}
    if not any((data, audit_doc, vocab, smoke, source)):
        status = "pending"
    elif not all((data, audit_doc, vocab)):
        status = "blocked_incomplete"
    elif str(audit_doc.get("status", "")) != "diagnostic_only":
        status = "diagnostic_blocked"
    elif not smoke:
        status = "blocked_incomplete"
    else:
        status = "completed_diagnostic_only"

    counts = dict(data.get("counts") or {})
    audit_gate = dict(audit_doc.get("information_gate") or {})
    smoke_gate = dict(smoke.get("gate") or {})
    training = dict(smoke.get("training") or {})
    losses = [dict(item) for item in list(smoke.get("loss") or []) if isinstance(item, dict)]
    holdout = [dict(item.get("implementation_holdout") or {}) for item in losses]
    holdout_losses = [float(item["mean_next_token_loss"]) for item in holdout if isinstance(item.get("mean_next_token_loss"), (int, float))]
    holdout_entropy = [float(item["mean_predictive_entropy_nats"]) for item in holdout if isinstance(item.get("mean_predictive_entropy_nats"), (int, float))]
    source_counts = dict(source.get("counts") or {})
    source_methods = dict(source.get("methods") or {})
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": status,
        "artifact_status": status,
        "record_count": int(counts.get("total", 0) or 0),
        "full_axis_count": int(counts.get("full_axis_rows", 0) or 0),
        "train_count": int(counts.get("train", 0) or 0),
        "implementation_holdout_count": int(counts.get("implementation_holdout", 0) or 0),
        "failure_repair_count": int(counts.get("failure_repair", 0) or 0),
        "negative_review_count": int(counts.get("negative_review", 0) or 0),
        "get_count": int(source_methods.get("GET", 0) or 0),
        "post_count": int(source_methods.get("POST", 0) or 0),
        "source_row_count": int(source_counts.get("source_row_count", 0) or 0),
        "source_failure_action_changed_count": int(source_counts.get("failure_action_changed_count", 0) or 0),
        "source_typed_positive_route_count": int(source_counts.get("typed_positive_route_count", 0) or 0),
        "source_negative_violation_count": int(source_counts.get("negative_violation_count", 0) or 0),
        "accepted_training_rows": 0,
        "information_audit_status": str(audit_doc.get("status", "pending")),
        "all_seven_axes_present": bool(audit_gate.get("all_axes_present")),
        "context_target_alignment": float(dict(audit_doc.get("context_target_alignment") or {}).get("rate", 0.0) or 0.0),
        "context_firewall_forbidden_count": int(dict(audit_doc.get("context_firewall") or {}).get("forbidden_token_count", 0) or 0),
        "context_vocabulary_size": int(vocab.get("context_vocabulary_size", len(list(vocab.get("context_tokens") or []))) or 0),
        "target_vocabulary_size": int(vocab.get("target_vocabulary_size", len(list(vocab.get("target_tokens") or []))) or 0),
        "a800_status": str(smoke.get("status", "pending")),
        "a800_information_gate_status": str(smoke_gate.get("information_gate_status", "pending")),
        "a800_train_rows": int(dict(smoke_gate.get("split_counts") or {}).get("train", 0) or 0),
        "a800_implementation_holdout_rows": int(dict(smoke_gate.get("split_counts") or {}).get("implementation_holdout", 0) or 0),
        "a800_required_context_window": int(dict(smoke.get("context_capacity_requirement") or {}).get("required_max_length", 0) or 0),
        "a800_epochs": int(training.get("epochs", 0) or 0),
        "a800_target_tokens_read": bool(training.get("target_tokens_read", False)),
        "a800_entropy_gate_passed": bool(dict(smoke.get("entropy_gate") or {}).get("passed", False)),
        "a800_entropy_drop_max": float(dict(smoke.get("entropy_gate") or {}).get("max_relative_entropy_drop", 0.0) or 0.0),
        "a800_holdout_loss_min": min(holdout_losses) if holdout_losses else 0.0,
        "a800_holdout_loss_max": max(holdout_losses) if holdout_losses else 0.0,
        "a800_holdout_entropy_min": min(holdout_entropy) if holdout_entropy else 0.0,
        "a800_holdout_entropy_max": max(holdout_entropy) if holdout_entropy else 0.0,
        "a800_hash_lock_passed": bool(dict(smoke_gate.get("checks") or {}).get("data_code_vocab_rules_hashes_locked")),
        "implementation_isolation": dict(smoke_gate.get("implementation_isolation") or {}),
        "training_eligible": False,
        "training_allowed": False,
        "promotion": promotion,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }


def _pg331_typed_capacity_projection(capacity: dict[str, Any]) -> dict[str, Any]:
    """Project the typed source-row model-capacity audit without promotion."""

    document = dict(capacity) if isinstance(capacity, dict) else {}
    context = dict(document.get("dataset_context_length") or {})
    variants = [dict(item) for item in list(document.get("variants") or []) if isinstance(item, dict)]
    variant_max_lengths = {
        str((item.get("config") or {}).get("id")): int((item.get("config") or {}).get("max_length", 0) or 0)
        for item in variants
        if str((item.get("config") or {}).get("id", ""))
    }
    truncation_risk = bool(any(bool(item.get("truncation_risk")) for item in variants))
    status = str(document.get("status", "pending")) if document else "pending"
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": status,
        "artifact_status": status,
        "context_min": int(context.get("min", 0) or 0),
        "context_max": int(context.get("max", 0) or 0),
        "required_context_window": int(document.get("required_context_window", 0) or 0),
        "model_vocabulary_size": int(document.get("model_vocabulary_size", 0) or 0),
        "variant_max_length": variant_max_lengths,
        "variant_max_lengths": [
            {"id": str((item.get("config") or {}).get("id", "")), "max_length": int((item.get("config") or {}).get("max_length", 0) or 0), "truncation_risk": bool(item.get("truncation_risk"))}
            for item in variants
        ],
        "truncation_risk": truncation_risk,
        "promotion": promotion,
        "training_allowed": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "audit_evidence_hash": str(document.get("audit_sha256", "")),
    }


def _pg331_train_holdout_diagnostic_v2_projection(
    report: dict[str, Any],
    dataset: dict[str, Any],
    source_audit: dict[str, Any],
    vocabulary: dict[str, Any],
    information: dict[str, Any],
    capacity: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Return a bounded, read-only PG-331 train/holdout diagnostic summary.

    The merged diagnostic dataset can be large and contains token rows.  This
    UI projection intentionally reads only top-level status/count/hash and
    per-axis aggregate entropy values.  It never iterates or emits records,
    tokens, row IDs, probes, responses, or evaluator/oracle fields.
    """

    documents = {
        "report": dict(report) if isinstance(report, dict) else {},
        "dataset": dict(dataset) if isinstance(dataset, dict) else {},
        "source_audit": dict(source_audit) if isinstance(source_audit, dict) else {},
        "vocabulary": dict(vocabulary) if isinstance(vocabulary, dict) else {},
        "information": dict(information) if isinstance(information, dict) else {},
        "capacity": dict(capacity) if isinstance(capacity, dict) else {},
        "plan": dict(plan) if isinstance(plan, dict) else {},
    }
    missing = [name for name, document in documents.items() if not document]
    report_counts = dict(documents["report"].get("counts") or {})
    dataset_counts = dict(documents["dataset"].get("counts") or {})
    source_counts = dict(documents["source_audit"].get("validation_counts") or {})
    info_validation = dict(documents["information"].get("validation") or {})
    capacity_context = dict(documents["capacity"].get("dataset_context_length") or {})
    plan_counts = dict(documents["plan"].get("counts") or {})
    axes = dict(documents["information"].get("axes") or {})
    axis_entropy: dict[str, dict[str, Any]] = {}
    for axis, details in axes.items():
        if not isinstance(details, dict):
            continue
        entropy = dict(details.get("entropy") or {})
        axis_entropy[str(axis)] = {
            "status": str(entropy.get("status", details.get("status", "unknown"))),
            "bits": float(entropy.get("bits", 0.0) or 0.0) if isinstance(entropy.get("bits"), (int, float)) else None,
            "count": int(entropy.get("count", 0) or 0),
            "unique": int(entropy.get("unique", 0) or 0),
            "unique_ratio": float(entropy.get("unique_ratio", 0.0) or 0.0) if isinstance(entropy.get("unique_ratio"), (int, float)) else None,
        }
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "status": "pending" if missing else "diagnostic_blocked",
        "missing_artifacts": missing,
        "artifact_status": {name: str(document.get("status", "available")) if document else "pending" for name, document in documents.items()},
        "counts": {
            "dataset_records": int(report_counts.get("records", dataset_counts.get("records", 0)) or 0),
            "implementations": int(report_counts.get("implementations", 0) or 0),
            "families": int(report_counts.get("families", 0) or 0),
            "source_audit_records": int(documents["source_audit"].get("record_count", 0) or 0),
            "source_audit_valid_rows": int(source_counts.get("valid", 0) or 0),
            "information_valid_rows": int(info_validation.get("valid_row_count", 0) or 0),
            "plan_eligible_train_rows": int(plan_counts.get("eligible_train_rows", 0) or 0),
            "plan_holdout_rows": int(plan_counts.get("holdout_rows", 0) or 0),
        },
        "axis_entropy": axis_entropy,
        "capacity": {
            "status": str(documents["capacity"].get("status", "pending")),
            "required_context_window": int(documents["capacity"].get("required_context_window", 0) or 0),
            "context_min": int(capacity_context.get("min", 0) or 0),
            "context_max": int(capacity_context.get("max", 0) or 0),
            "model_vocabulary_size": int(documents["capacity"].get("model_vocabulary_size", 0) or 0),
            "inventory_missing_count": int(documents["capacity"].get("inventory_missing_count", 0) or 0),
        },
        "evidence_hashes": {
            "report": str(documents["report"].get("dataset_sha256", ""))[:16],
            "dataset": str(documents["dataset"].get("dataset_sha256", ""))[:16],
            "source_audit": str(documents["source_audit"].get("audit_sha256", ""))[:16],
            "vocabulary": str(documents["vocabulary"].get("vocabulary_sha256", ""))[:16],
            "information": str(documents["information"].get("dataset_information_sha256", ""))[:16],
            "capacity": str(documents["capacity"].get("audit_sha256", ""))[:16],
            "plan": str(documents["plan"].get("plan_sha256", ""))[:16],
        },
        "promotion": promotion,
        "training_allowed": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
def _metric_bound(
    report: dict[str, Any],
    variant: str,
    section: str,
    metric: str,
    bound: str,
) -> float:
    try:
        return float(report["aggregated"][variant][section][metric][bound])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _learning_requirements(
    report: dict[str, Any],
    audit: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Translate PG-277 evidence into an executable collection brief.

    The quotas are graduation targets for the next real multi-family tranche,
    not invented progress.  Controlled-fixture rows stay visible separately so
    nobody mistakes them for application-backed gold data.
    """

    collision = dict((dataset.get("projection_collision_audit") or {}).get("coarse") or {})
    counts = dict(dataset.get("counts") or {})
    report_ready = report.get("status") == "completed_question_composition_ablation"
    audit_pass = audit.get("status") == "passed"
    process_question_min = _metric_bound(
        report,
        "enriched_process_sft",
        "missing_observation",
        "question_recovery_rate",
        "min",
    )
    conservative_question_min = _metric_bound(
        report,
        "conservative_offline_update",
        "missing_observation",
        "question_recovery_rate",
        "min",
    )
    dpo_question_min = _metric_bound(
        report,
        "dpo_preference_update",
        "missing_observation",
        "question_recovery_rate",
        "min",
    )
    return {
        "schema_version": "sift-learning-requirements-v1",
        "title": "让模型学到真东西的数据任务书",
        "principle": "先保证模型看得到区分答案所需的信息，再训练它提出问题、吸收失败并选择下一步；任何漂亮平均分都不能覆盖最坏 seed。",
        "evidence": {
            "source_report": "pg277_question_composition_report_v1.json",
            "source_audit": "pg277_question_composition_audit_v1.json",
            "report_ready": report_ready,
            "audit_pass": audit_pass,
            "controlled_rows": int(counts.get("train", 0) or 0) + int(counts.get("holdout", 0) or 0),
            "real_multifamily_gold_rows": 0,
            "coarse_conflict_groups": int(collision.get("conflict_group_count", 0) or 0),
            "coarse_conflicting_rows": int(collision.get("conflicting_record_count", 0) or 0),
            "process_question_recovery_worst_seed": process_question_min,
            "conservative_question_recovery_worst_seed": conservative_question_min,
            "dpo_question_recovery_worst_seed": dpo_question_min,
            "claim": "受控表示/监督假设证据，不是实际多族漏洞能力证明",
        },
        "findings": [
            {
                "id": "projection-collision",
                "severity": "P0",
                "title": "缺观测时，增加训练量也无解",
                "evidence": f"{int(collision.get('conflicting_record_count', 0) or 0)} 条记录落入 {int(collision.get('conflict_group_count', 0) or 0)} 个相同输入却不同标签的冲突组；coarse 正例召回 0%。",
                "action": "先补 marker channel、编码传输、失败类型或 ASK 状态；冲突输入只能训练 ASK/不确定性，不能硬塞最终阳性/阴性。",
            },
            {
                "id": "final-only",
                "severity": "P0",
                "title": "只存答案会训练出被动分类器",
                "evidence": "final-only 在完整输入上判断 100%，但 pre-question=0%、ask-recovery=0%，missing-safety 最坏 seed=0%。",
                "action": "每条必须保存问题前状态、实际提问、返回观测、失败归因、下一问、下一动作和 belief 变化。",
            },
            {
                "id": "question-instability",
                "severity": "P1",
                "title": "会 ASK 不等于问对问题",
                "evidence": f"process SFT 精确问题恢复最坏 seed={process_question_min * 100:.0f}%；DPO={dpo_question_min * 100:.0f}%；保守更新={conservative_question_min * 100:.0f}%。",
                "action": "补同一状态下不同缺字段对应不同下一问的反事实对；所有选择按跨 seed 最小值验收。",
            },
            {
                "id": "scope-limit",
                "severity": "P0",
                "title": "单一受控族不能证明渗透泛化",
                "evidence": "PG-277 只有一个受控 surface family；训练、长期记忆和漏洞声明均冻结。",
                "action": "下一批必须跨漏洞族、独立实现、GET/POST、编码和页面表面隔离，并在 fresh 本地目标上由 typed oracle 复放。",
            },
        ],
        "queues": [
            {
                "id": "Q1-observation-counterfactuals",
                "priority": "P0",
                "owner": "采集员",
                "title": "补齐可区分的观测反事实对",
                "why": "消灭相同模型输入对应冲突目标的不可学样本，让模型知道还缺哪条信息。",
                "status": "ready",
                "current": "60 条受控 fixture 记录；真实多族 gold=0",
                "minimum_quota": [
                    "至少 4 个漏洞/行为族",
                    "每族至少 3 个独立实现或源码哈希",
                    "每实现至少 3 个 seed、2 种编码/传输表面",
                    "每个正例至少 1 个形状匹配阴性 + 1 个单字段反事实",
                    "目标：每族 ≥24 gold transition + ≥12 hard-negative repair transition",
                ],
                "collect": [
                    "问题前可见 token 与明确 missing-field mask",
                    "模型提出的下一问与目标观测槽位",
                    "baseline/reference/candidate/negative 的 bounded response projection",
                    "marker/DOM/SQL-result/redirect/auth-transition 等观测通道",
                    "编码链、方法、字段绑定、失败签名和 evidence hash",
                ],
                "acceptance": [
                    "模型可见 context 去重后不得出现冲突 target；若信息确实不足，统一 target=ASK",
                    "source/implementation/seed/generator split 完全隔离",
                    "fresh reset 下至少 2 次复放一致",
                    "reference agreement + matched negative clean + typed oracle 或显式 oracle_unavailable",
                ],
                "output_lane": "gold / hard_negative / diagnostic_collision",
                "prevents": ["projection collision", "模板捷径", "全 abstain 假成功"],
            },
            {
                "id": "Q2-failure-repair-trajectories",
                "priority": "P0",
                "owner": "采集员 + 复核员",
                "title": "保存失败如何变成下一问",
                "why": "让模型学习排查流程，而不是背一个最终 payload 或标签。",
                "status": "ready",
                "current": "PG-277 有 ASK/观察状态；真实多步 repair 仍不足",
                "minimum_quota": [
                    "每族至少 12 条真实失败→诊断→最小修复链",
                    "每条链至少 2 个失败分支和 1 个成功或可信 abstain 分支",
                    "环境失败与模型失败各至少 6 条，且单独归因",
                    "每个 question class 在 3 个模型 seed 中均有支持",
                ],
                "collect": [
                    "失败发生阶段、可观察签名和 model/environment 归因",
                    "失败前 belief、失败后 belief posterior 与被否定的假设",
                    "下一问候选、选择原因、未选择问题及拒绝原因",
                    "最小 repair delta、repair 后观测和 fresh replay 结果",
                    "成功、继续探索、oracle gap、safe abstain 四类终止状态",
                ],
                "acceptance": [
                    "失败可复现且不是单纯 HTTP 状态码猜测",
                    "repair 只改变已声明的一个抽象槽位",
                    "父子 record_id 和 before/after token hash 连续",
                    "复核员能从证据链重建为何问这一问",
                ],
                "output_lane": "hard_negative / gold_process / quarantine",
                "prevents": ["final-label classifier", "错误归因", "无意义长轨迹"],
            },
            {
                "id": "Q3-missing-question-recovery",
                "priority": "P0",
                "owner": "数据设计员",
                "title": "专门训练“缺什么就问什么”",
                "why": "当前模型能选择 ASK，但精确问题会随 seed 退化；这是最直接的短板。",
                "status": "ready",
                "current": f"process/DPO 最坏 seed={min(process_question_min, dpo_question_min) * 100:.0f}%；目标 ≥90%",
                "minimum_quota": [
                    "至少 8 类 missing observation slot",
                    "每类 16 个配对状态，其中 8 个单缺失、8 个多缺失/干扰",
                    "每个 context 配 1 个正确问题 + ≥2 个困难错误问题",
                    "至少 25% 为完全未知，应输出 ASK_GENERIC 或 abstain",
                ],
                "collect": [
                    "完整状态与逐槽位 mask/counterfactual 版本",
                    "信息增益最高的问题标签和次优问题排序",
                    "问后观测是否真正降低 belief entropy",
                    "无可用 oracle 时的停止/升级人工理由",
                ],
                "acceptance": [
                    "三模型 seed 的 exact question recovery 最小值 ≥90%",
                    "missing-safe、ask-recovery、positive recall、negative reject 均 ≥90%",
                    "错误问题不会因 route/name/template token 被猜中",
                ],
                "output_lane": "question_gold / preference_pair / abstain_silver",
                "prevents": ["问错问题", "DPO seed 方差", "信息缺失时乱报阳性"],
            },
            {
                "id": "Q4-ood-and-forgetting",
                "priority": "P1",
                "owner": "训练员 + 最终判官",
                "title": "实现外、族外与遗忘矩阵",
                "why": "随机换 seed 不能排除模板记忆，平均分也会掩盖某个实现或 seed 的崩溃。",
                "status": "blocked_on_Q1_Q2_Q3",
                "current": "alpha/beta→gamma 受控实现留出通过；真实多族矩阵未通过",
                "minimum_quota": [
                    "训练 ≥3 实现，留出 ≥1 全新实现；轮换 leave-one-implementation-out",
                    "至少 1 个完整漏洞族只做 family holdout",
                    "每次更新固定回放旧族 canary 和 unknown-family abstain",
                    "报告每个 family×implementation×seed 单元，禁止只报平均值",
                ],
                "collect": [
                    "镜像 digest、源码 hash、parser/checker/tokenizer/model 版本",
                    "split lineage 与 near-duplicate group hash",
                    "更新前后旧 canary 的 loss/action/belief/abstain 指标",
                    "每个最坏单元的失败记录和 repair backlog",
                ],
                "acceptance": [
                    "最坏 seed/实现的正例召回、负例拒绝、ASK 恢复均 ≥90%",
                    "matched-negative false accept=0",
                    "旧 canary 无 guardrail 回退，unknown 不产生 unsupported positive",
                ],
                "output_lane": "holdout_only / canary / promotion_evidence",
                "prevents": ["composition leakage", "灾难性遗忘", "平均分遮丑"],
            },
        ],
        "record_contract": [
            {
                "group": "来源与隔离",
                "fields": ["record_id", "parent_record_id", "source/license/authorization", "container_digest", "source_hash", "family", "implementation", "generator", "seed", "split"],
                "rule": "任一字段缺失只能 quarantine；同源近重复不得跨 train/holdout。",
            },
            {
                "group": "问题前状态",
                "fields": ["surface/Rule-IR/history tokens", "method", "parameter slots", "encoding chain", "known/unknown slots", "belief prior", "candidate questions"],
                "rule": "缺失信息用显式 mask 表达，不得静默填 0 或猜默认值。",
            },
            {
                "group": "动作与观测",
                "fields": ["chosen question", "abstract action", "target slot", "bounded response projection", "redirect chain", "failure signature", "failure attribution"],
                "rule": "保存模型真正看到的观测；oracle 结果只在 evaluator side。",
            },
            {
                "group": "学习目标与证据",
                "fields": ["next question/action/belief", "rejected alternatives", "repair delta", "reference/negative", "typed oracle", "fresh replay count", "evidence hashes", "retention lane"],
                "rule": "最终标签只是一个字段，不能替代过程 target 和可复放证据。",
            },
        ],
        "resources": [
            {
                "category": "授权目标",
                "required": True,
                "items": ["本地 Docker 靶场及 pinned image digest", "仓库内独立实现 fixture", "fresh reset/数据库健康脚本", "每个来源的授权与许可证记录"],
                "why": "提供跨实现、可重置、可追溯的真实响应，而不是纸面标签。",
            },
            {
                "category": "采集能力",
                "required": True,
                "items": ["浏览器 DOM/表单/链接发现", "GET/POST/form/multipart/302 trace", "bounded header/body-shape/encoding projection", "失败阶段与网络事件时间线"],
                "why": "把问题、动作和返回信息对齐，避免只保存页面外观或最终答案。",
            },
            {
                "category": "独立判官",
                "required": True,
                "items": ["DOM marker oracle", "SQL result/AST differential oracle", "redirect/auth-transition oracle", "reference + matched negative evaluator", "evidence hash index"],
                "why": "模型负责猜和问，判官只负责在发送后确认效果，防止标签泄漏。",
            },
            {
                "category": "训练与审计",
                "required": True,
                "items": ["版本化 tokenizer/encoder/checker", "append-only gold/hard-negative/silver/quarantine 存储", "碰撞与近重复扫描", "跨 seed/实现/family split builder", "旧能力 canary checkpoint"],
                "why": "能定位训练失败是数据、表示、奖励还是工程问题，并可回滚污染样本。",
            },
            {
                "category": "人工资源",
                "required": True,
                "items": ["采集员填写过程字段", "复核员检查证据链", "领域人员裁决 oracle disagreement", "最终判官签署 promotion report"],
                "why": "人工重点审核最有价值的歧义、失败和判官冲突，不做机械复制粘贴。",
            },
        ],
        "forbidden": [
            {"id": "final-label-only", "title": "只有最终阳性/阴性标签", "reason": "无法训练提问、失败归因和修复", "lane": "reject_or_rework"},
            {"id": "missing-as-zero", "title": "缺字段静默填零/空串", "reason": "制造伪规律和模型输入碰撞", "lane": "quarantine"},
            {"id": "status-is-oracle", "title": "把 HTTP 200/302/500/timeout 当漏洞成立", "reason": "状态码不是类型化效果证据", "lane": "diagnostic_only"},
            {"id": "template-duplicates", "title": "同模板只换 seed/文案大量复制", "reason": "扩大记忆权重但不增加信息", "lane": "deduplicate"},
            {"id": "oracle-leakage", "title": "把 typed oracle/答案 token 放进模型输入", "reason": "训练出读答案的分类器", "lane": "reject"},
            {"id": "split-leakage", "title": "同 source/implementation/duplicate group 跨训练留出", "reason": "不能证明组合泛化", "lane": "holdout_rebuild"},
            {"id": "unverified-raw", "title": "未复放的 payload、正文或猜测直接入库", "reason": "污染长期记忆且无法追责", "lane": "quarantine_or_reject"},
            {"id": "mean-only", "title": "只报平均准确率/reward/loss", "reason": "会隐藏掉到 0% 的 seed 或实现", "lane": "report_invalid"},
            {"id": "rl-on-incomplete", "title": "在缺观测/冲突标签上继续堆 RL", "reason": "优化器只会放大 abstain 或伪捷径", "lane": "training_blocked"},
        ],
        "promotion_gate": {
            "current_status": "BLOCKED",
            "conditions": [
                "Q1/Q2/Q3 最低配额和全部字段硬门通过",
                "模型可见 context-target collision=0；歧义样本正确输出 ASK",
                "三 seed 最坏单元 question/action/belief/recall/reject ≥90%",
                "family/implementation holdout 和旧 canary 通过",
                "matched-negative false accept=0，unsupported positive=0",
                "独立审计重算数据、报告、轨迹和协议哈希",
            ],
            "next_experiment": "PG-278：先补 missing-question 反事实和真实多族 failure→repair 轨迹，再做保守 process-SFT/update；DPO 只作对照，不按平均分晋级。",
        },
    }


def _augment_learning_requirements_with_pg278(
    brief: dict[str, Any],
    report: dict[str, Any],
    audit: dict[str, Any],
    dataset: dict[str, Any],
    model_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the latest filled-data experiment without widening its claim.

    PG-278 is a controlled four-fixture study.  The operations page must show
    its evidence and the resulting human work, but it must not silently turn
    controlled rows into real-application gold or a vulnerability claim.
    """

    counts = dict(dataset.get("counts") or {})
    collisions = dict(dataset.get("projection_collision_audit") or {})
    aggregate = dict(report.get("aggregated") or {})
    process = dict(aggregate.get("enriched_process_sft") or {})
    conservative = dict(aggregate.get("conservative_offline_update") or {})
    dpo = dict(aggregate.get("dpo_preference_update") or {})
    gate = dict(report.get("hypothesis_gate") or {})
    report_ready = report.get("status") == "completed_controlled_multifamily_question_policy_study"
    dataset_audit_pass = audit.get("status") == "passed"
    model_audit_pass = (model_audit or {}).get("status") == "passed"
    audit_pass = dataset_audit_pass and model_audit_pass

    def metric(variant: str, section: str, name: str, bound: str = "min") -> float:
        return _metric_bound(report, variant, section, name, bound)

    pre_transition = metric("enriched_process_sft", "implementation_holdout", "pre_transition_accuracy")
    post_transition = metric("enriched_process_sft", "implementation_holdout", "post_transition_accuracy")
    slot_accuracy = metric("enriched_process_sft", "implementation_holdout", "pre_slot_accuracy")
    pair_accuracy = metric("enriched_process_sft", "paired_counterfactual", "paired_counterfactual_transition_accuracy")
    safe_rate = metric("enriched_process_sft", "missing_observation", "safe_non_supported_rate")
    family_question = min((metric for metric in [
        float(dict(value).get("pre_question_accuracy", {}).get("min", 0.0) or 0.0)
        for value in dict(report.get("family_holdout_abstract_question") or {}).values()
    ]), default=0.0)
    coarse = dict(collisions.get("coarse") or {})
    enriched = dict(collisions.get("enriched") or {})
    post = dict(collisions.get("post") or {})
    real_gold = int((dataset.get("source") or {}).get("real_multifamily_gold_rows", 0) or 0)

    brief["schema_version"] = "sift-learning-requirements-v2"
    brief["title"] = "让模型学到真东西的数据任务书 / PG-278"
    brief["principle"] = "先把能区分答案的请求条件、响应通道和失败状态记录完整；再训练提问、证据吸收和修复。缺观测的样本只能教 ASK，不能硬教最终标签。"
    brief["evidence"].update({
        "source_report": "pg278_multifamily_question_policy_report_v1.json",
        "source_audit": "pg278_multifamily_question_policy_audit_v1.json",
        "dataset_audit": "pg278_multifamily_question_dataset_audit_v1.json",
        "report_ready": report_ready,
        "audit_pass": audit_pass,
        "dataset_audit_pass": dataset_audit_pass,
        "model_audit_pass": model_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "real_multifamily_gold_rows": real_gold,
        "coarse_conflict_groups": int(coarse.get("conflict_group_count", 0) or 0),
        "coarse_conflicting_rows": int(coarse.get("conflicting_record_count", 0) or 0),
        "pg278_enriched_conflict_groups": int(enriched.get("conflict_group_count", 0) or 0),
        "pg278_post_conflict_groups": int(post.get("conflict_group_count", 0) or 0),
        "pg278_pre_transition_worst_seed": pre_transition,
        "pg278_post_transition_worst_seed": post_transition,
        "pg278_slot_binding_worst_seed": slot_accuracy,
        "pg278_pair_flip_worst_seed": pair_accuracy,
        "pg278_missing_safe_worst_seed": safe_rate,
        "pg278_family_question_worst_seed": family_question,
        "pg278_gate_status": str(gate.get("status", "blocked")),
        "claim": "PG-278 只证明受控四族 Rule-IR slot binding 与失败修复流程；真实多族 gold=0，不能推出公网漏洞能力。",
    })
    for finding in brief.get("findings", []):
        if finding.get("id") == "projection-collision":
            finding["evidence"] = f"PG-278 coarse context 的 {int(coarse.get('conflicting_record_count', 0) or 0)} 条记录落入 {int(coarse.get('conflict_group_count', 0) or 0)} 个冲突组；补足 missing slot 后 pre/post collision 都为 {int(enriched.get('conflict_group_count', 0) or 0)}/{int(post.get('conflict_group_count', 0) or 0)}。"
            finding["action"] = "采集抽象请求条件、响应通道和失败状态；每次新增字段后同时重算 pre 与 post collision。"
        elif finding.get("id") == "question-instability":
            finding["evidence"] = f"PG-277 的 exact question 曾随 seed 退化；PG-278 加入 8 个 slot、4 个族、3 个实现后，process pre/post 最坏 seed={pre_transition * 100:.0f}%/{post_transition * 100:.0f}%，配对翻转={pair_accuracy * 100:.0f}%。"
            finding["action"] = "保留实现留出、slot 对照和 request-condition 字段；没有 post collision=0 不得晋级。"
        elif finding.get("id") == "scope-limit":
            finding["evidence"] = "PG-278 已覆盖 4 个受控行为族、288 条带两次复放的记录，但真实多族 gold 仍为 0；当前只能证明受控流程。"
            finding["action"] = "下一步把同一 record contract 接到授权的本地真实回放，保留 source/implementation/family holdout 与旧 canary。"
    brief["findings"].append({
        "id": "request-context-completeness",
        "severity": "P0",
        "title": "响应相同也可能需要请求条件才能判定",
        "evidence": "PG-278 的逻辑/授权对照显示，正常 owner grant 与非 owner grant 不能只用 HTTP 状态和响应形状区分；最终记录增加了抽象主体/条件字段，并把 post collision 纳入硬审计。",
        "action": "采集 GET query / POST form 的抽象条件、字段绑定和编码链；禁止只保存页面截图或状态码。",
    })
    for queue in brief.get("queues", []):
        if queue.get("id") == "Q1-observation-counterfactuals":
            queue["status"] = "in_progress"
            queue["current"] = f"PG-278 已填入 {int(counts.get('total', 0) or 0)} 条四族受控记录；pre/post collision={int(enriched.get('conflict_group_count', 0) or 0)}/{int(post.get('conflict_group_count', 0) or 0)}；真实 gold={real_gold}"
            queue["collect"].append("抽象请求条件（主体/边界/字段状态）与响应投影的配对，避免响应同形时不可判定")
            queue["acceptance"].append("post-observation context collision=0，不能只审问题前 context")
        elif queue.get("id") == "Q2-failure-repair-trajectories":
            queue["status"] = "in_progress"
            queue["current"] = f"PG-278 已提供 {int(counts.get('total', 0) or 0)} 条正/负 paired transition；真实目标 repair 仍待采集"
        elif queue.get("id") == "Q3-missing-question-recovery":
            queue["status"] = "review_ready" if report_ready and audit_pass and str(gate.get("status")) == "passed" else "in_progress"
            queue["current"] = f"PG-278 controlled gate={str(gate.get('status', 'blocked')).upper()}；slot={slot_accuracy * 100:.0f}%、family abstract question={family_question * 100:.0f}%；真实 gold={real_gold}"
        elif queue.get("id") == "Q4-ood-and-forgetting":
            queue["status"] = "blocked_on_real_gold"
            queue["current"] = "PG-278 实现留出通过；真实族外、旧能力遗忘 canary 和多靶场回放仍未通过"
    for contract in brief.get("record_contract", []):
        if contract.get("group") == "问题前状态":
            if "abstract request condition" not in contract["fields"]:
                contract["fields"].append("abstract request condition")
            contract["rule"] = "缺失信息显式 mask；请求条件、字段绑定和编码链必须可复放，不能静默猜默认值。"
        if contract.get("group") == "动作与观测":
            if "post-observation projection collision" not in contract["fields"]:
                contract["fields"].append("post-observation projection collision")
    if not any(item.get("category") == "PG-278 受控多族包" for item in brief.get("resources", [])):
        brief["resources"].append({"category": "PG-278 受控多族包", "required": True, "items": ["DOM/SQL/redirect/logic 四个 loopback fixture", "每族 3 implementation × 3 seed × 2 encoding", "288 条正负 paired transition", "两次 fresh replay evidence hash", "A800 GPU0 训练报告与独立数据审计"], "why": "先验证记录契约、slot binding 和失败修复的可学性；它不是实际应用 gold，必须与真实回放 lane 隔离。"})
    brief["promotion_gate"]["next_experiment"] = "PG-279：把同一 record contract 接到授权本地真实回放，先补带 GET/POST 请求条件的 failure→repair gold，再做族外与遗忘矩阵。"
    brief["promotion_gate"]["conditions"].append("PG-278 post-observation collision=0 且 gate 通过，但真实多族 gold>0、fresh replay 和旧 canary 仍是必须条件")
    brief["latest_experiment"] = {
        "id": "PG-278",
        "status": str(gate.get("status", "blocked")),
        "report": "pg278_multifamily_question_policy_report_v1.json",
        "audit": "pg278_multifamily_question_policy_audit_v1.json",
        "dataset_audit": "pg278_multifamily_question_dataset_audit_v1.json",
        "independent_audit_pass": model_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "families": 4,
        "implementations_per_family": 3,
        "seeds_per_implementation": 3,
        "encodings_per_seed": 2,
        "pre_transition_worst_seed": pre_transition,
        "post_transition_worst_seed": post_transition,
        "pair_flip_worst_seed": pair_accuracy,
        "real_multifamily_gold_rows": real_gold,
        "promotion_blocked": True,
    }
    return brief


def _augment_learning_requirements_with_pg279(
    brief: dict[str, Any],
    report: dict[str, Any],
    dataset_audit: dict[str, Any],
    dataset: dict[str, Any],
    policy_audit: dict[str, Any],
    training_mix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose PG-279 as the current task station without overclaiming.

    The operational replay/audit can pass while the family-heldout scientific
    hypothesis remains blocked.  The UI must show both states separately.
    """

    counts = dict(dataset.get("counts") or {})
    replay = dict(dataset.get("replay_contract") or {})
    source = dict(dataset.get("source") or {})
    collisions = dict(dataset.get("projection_collision_audit") or {})
    gate = dict(report.get("hypothesis_gate") or {})
    retention = dict(report.get("retention_matrix") or {})
    after_min = dict(retention.get("after_min") or {})
    family_values = [
        float(dict(value).get("pre_question_accuracy", {}).get("min", 0.0) or 0.0)
        for value in dict(report.get("family_holdout_abstract_question") or {}).values()
    ]
    family_question = min(family_values, default=0.0)
    dataset_audit_pass = dataset_audit.get("status") == "passed"
    policy_audit_pass = policy_audit.get("status") == "passed"
    report_ready = report.get("status") == "completed_remote_loopback_replay_policy_study"
    mix = dict(training_mix or {})
    mix_meta = dict(report.get("source", {}).get("training_replay_mix") or {})

    brief["schema_version"] = "sift-learning-requirements-v3"
    brief["title"] = "让模型学到真东西的数据任务书 / PG-279 远程回放"
    brief["principle"] = "先记录真实 GET/POST 传输、失败到修复和 typed/abstain 证据，再用 replay mix 保持旧能力；族外假设单独验收，不能用工程通过掩盖科学失败。"
    brief["evidence"].update({
        "source_report": "pg279_remote_replay_policy_report_v1.json",
        "source_audit": "pg279_remote_replay_policy_audit_v1.json",
        "dataset_audit": "pg279_remote_replay_dataset_audit_v1.json",
        "report_ready": report_ready,
        "audit_pass": policy_audit_pass,
        "dataset_audit_pass": dataset_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "coarse_conflict_groups": int(dict(collisions.get("coarse") or {}).get("conflict_group_count", 0) or 0),
        "coarse_conflicting_rows": int(dict(collisions.get("coarse") or {}).get("conflicting_record_count", 0) or 0),
        "pg279_enriched_conflict_groups": int(dict(collisions.get("enriched") or {}).get("conflict_group_count", 0) or 0),
        "pg279_post_conflict_groups": int(dict(collisions.get("post") or {}).get("conflict_group_count", 0) or 0),
        "pg279_get_rows": int(replay.get("get_rows", 0) or 0),
        "pg279_post_rows": int(replay.get("post_rows", 0) or 0),
        "pg279_failure_repair_rows": int(replay.get("failure_repair_rows", 0) or 0),
        "pg279_typed_effect_rows": int(replay.get("typed_effect_rows", 0) or 0),
        "pg279_abstain_rows": int(replay.get("abstain_rows", 0) or 0),
        "pg279_family_question_worst_seed": family_question,
        "pg279_gate_status": str(gate.get("status", "blocked")),
        "pg279_retention_pre_min": float(after_min.get("pre_transition_accuracy", 0.0) or 0.0),
        "pg279_retention_post_min": float(after_min.get("post_transition_accuracy", 0.0) or 0.0),
        "pg279_retention_missing_safe_min": float(after_min.get("missing_safe_rate", 0.0) or 0.0),
        "pg279_training_mix_sha256": str(mix_meta.get("dataset_sha256") or mix.get("dataset_sha256") or ""),
        "claim": "PG-279 证明远程受控回放、GET/POST failure→repair 和遗忘保持契约；族外科学 gate blocked，real application gold=0，不能推出 Pikachu 或公网漏洞能力。",
    })
    brief["findings"].append({
        "id": "pg279-family-heldout-blocked",
        "severity": "P0",
        "title": "族外问题恢复仍未通过",
        "evidence": f"PG-279 operational audit={str(policy_audit.get('status', 'missing'))}，PG-278 retention={str(retention.get('status', 'blocked'))}；family-heldout question 最坏={family_question * 100:.0f}%，科学 gate={str(gate.get('status', 'blocked'))}。",
        "action": "PG-280 增加 provenance-safe shared slot ontology 与跨族 hard-negative；在 real application gold 可用前保持 promotion/memory 冻结。",
    })
    for contract in brief.get("record_contract", []):
        if contract.get("group") == "请求与传输":
            for field in ("GET query / POST form projection", "two fresh replay evidence hashes"):
                if field not in contract["fields"]:
                    contract["fields"].append(field)
        if contract.get("group") == "动作与观测":
            for field in ("typed effect / explicit abstain", "failure→repair transition"):
                if field not in contract["fields"]:
                    contract["fields"].append(field)
    if not any(item.get("category") == "PG-279 远程回放包" for item in brief.get("resources", [])):
        brief["resources"].append({
            "category": "PG-279 远程回放包",
            "required": True,
            "items": ["4 families × 3 implementations × 3 seeds × 2 encodings", "GET 216 / POST 72", "每条 failure→repair + reference + matched-negative", "两次 fresh replay hash", "PG-278 replay mix 与冻结保持矩阵"],
            "why": "让采集员能看到真实传输和修复过程；同时把族外失败和遗忘回归分开记录。",
        })
    for queue in brief.get("queues", []):
        if queue.get("id") == "Q1-observation-counterfactuals":
            queue["status"] = "review_ready" if policy_audit_pass else "in_progress"
            queue["current"] = f"PG-279 GET/POST={int(replay.get('get_rows', 0) or 0)}/{int(replay.get('post_rows', 0) or 0)}；typed/abstain={int(replay.get('typed_effect_rows', 0) or 0)}/{int(replay.get('abstain_rows', 0) or 0)}"
        elif queue.get("id") == "Q2-failure-repair-trajectories":
            queue["status"] = "review_ready" if policy_audit_pass else "in_progress"
            queue["current"] = f"PG-279 已保留 {int(replay.get('failure_repair_rows', 0) or 0)} 条 failure→repair；真实 gold={int(source.get('real_application_gold_rows', 0) or 0)}"
        elif queue.get("id") == "Q4-ood-and-forgetting":
            queue["status"] = "blocked_on_family_holdout" if not bool(gate.get("checks", {}).get("family_holdout_question_min")) else "review_ready"
            queue["current"] = f"遗忘保持={str(retention.get('status', 'blocked'))}；族外 question={family_question * 100:.0f}%；real application gold=0"
    brief["promotion_gate"]["current_status"] = "PG-279 operational audit passed / scientific family gate blocked"
    brief["promotion_gate"]["next_experiment"] = "PG-280：授权远程 Docker/真实应用可用后，补 shared slot ontology 与族外 hard-negative，再做真实 gold 验收。"
    brief["promotion_gate"]["conditions"].extend([
        "PG-279 operational audit 必须保持通过；family-heldout scientific gate 不能用平均分替代",
        "PG-278 retention pre/post/missing-safe 最坏指标全部不回退；real_application_gold_rows>0 前仍不得晋级",
    ])
    brief["latest_experiment"] = {
        "id": "PG-279",
        "status": "operational_audit_passed_scientific_gate_blocked" if policy_audit_pass else "audit_pending",
        "report": "pg279_remote_replay_policy_report_v1.json",
        "audit": "pg279_remote_replay_policy_audit_v1.json",
        "dataset_audit": "pg279_remote_replay_dataset_audit_v1.json",
        "independent_audit_pass": policy_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "families": len(counts.get("families") or {}),
        "implementations_per_family": 3,
        "seeds_per_implementation": 3,
        "encodings_per_seed": 2,
        "pre_transition_worst_seed": float(dict(report.get("aggregated", {}).get("enriched_process_sft", {}).get("implementation_holdout", {})).get("pre_transition_accuracy", {}).get("min", 0.0) or 0.0),
        "post_transition_worst_seed": float(dict(report.get("aggregated", {}).get("enriched_process_sft", {}).get("implementation_holdout", {})).get("post_transition_accuracy", {}).get("min", 0.0) or 0.0),
        "pair_flip_worst_seed": float(dict(report.get("aggregated", {}).get("enriched_process_sft", {}).get("paired_counterfactual", {})).get("paired_counterfactual_transition_accuracy", {}).get("min", 0.0) or 0.0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
    }
    return brief


def _augment_learning_requirements_with_pg280(
    brief: dict[str, Any],
    report: dict[str, Any],
    dataset_audit: dict[str, Any],
    dataset: dict[str, Any],
    policy_audit: dict[str, Any],
    docker_probe: dict[str, Any],
    remote_adapter_probe: dict[str, Any] | None = None,
    remote_adapter_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose PG-280's shared ontology and identifiability result.

    PG-280 deliberately separates two claims: a final/post classifier can
    score well, while an active process must learn to ASK when a slot is not
    identifiable.  The operations projection keeps that distinction visible
    and never treats evaluation-only hard negatives or a missing Docker daemon
    as real application gold.
    """

    counts = dict(dataset.get("counts") or {})
    remote_adapter_probe = dict(remote_adapter_probe or {})
    remote_adapter_audit = dict(remote_adapter_audit or {})
    records = [item for item in list(dataset.get("records") or []) if isinstance(item, dict)]
    ident = dict(report.get("identifiability") or dataset.get("identifiability") or {})
    comparison = dict(report.get("comparison") or {})
    final_only = dict(comparison.get("final_only") or {})
    process = dict(comparison.get("process") or {})
    hard_negative = dict(report.get("family_ood_hard_negative") or {})
    source = dict(report.get("source") or dataset.get("source") or {})
    gate = dict(report.get("hypothesis_gate") or {})
    report_ready = report.get("status") == "completed_remote_pg280_ontology_policy_study"
    dataset_audit_pass = dataset_audit.get("status") == "passed"
    policy_audit_pass = policy_audit.get("status") == "passed"
    docker_status = str(docker_probe.get("status") or report.get("docker_probe", {}).get("status") or "unavailable")
    remote_adapter_status = str(remote_adapter_probe.get("status") or "not_run")
    remote_adapter_audit_pass = remote_adapter_audit.get("status") == "passed"
    audit_pass = dataset_audit_pass and policy_audit_pass and remote_adapter_audit_pass
    family_count = len({str(item.get("family")) for item in records if item.get("family")})
    implementation_count = len({str(item.get("implementation")) for item in records if item.get("implementation")})
    seed_count = len({item.get("collection_seed") for item in records if item.get("collection_seed") is not None})
    encoding_count = len({str(item.get("encoding")) for item in records if item.get("encoding")})

    brief["schema_version"] = "sift-learning-requirements-v4"
    brief["title"] = "让模型学到真东西的数据任务书 / PG-280 shared slot ontology"
    brief["principle"] = "如果可见 context 对应多个隐藏 slot，H(Y|X)>0 时精确答案不是训练少而是当前观测不可辨识；训练必须把 ASK/未决 belief 与最终分类分开验收。"
    brief["evidence"].update({
        "source_report": "pg280_ontology_policy_report_v1.json",
        "source_audit": "pg280_ontology_policy_audit_v1.json",
        "dataset_audit": "pg280_shared_ontology_dataset_audit_v1.json",
        "report_ready": report_ready,
        "audit_pass": audit_pass,
        "dataset_audit_pass": dataset_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "pg280_conditional_entropy_bits": float(ident.get("conditional_entropy_bits", 0.0) or 0.0),
        "pg280_bayes_error_lower_bound": float(ident.get("bayes_error_lower_bound", 0.0) or 0.0),
        "pg280_final_only_pre_supervision_rows": int(final_only.get("pre_supervision_rows", 0) or 0),
        "pg280_final_only_post_accuracy": float(final_only.get("post_transition_accuracy_min", 0.0) or 0.0),
        "pg280_final_only_ask_rate": float(final_only.get("missing_ask_rate_min", 0.0) or 0.0),
        "pg280_process_pre_supervision_rows": int(process.get("pre_supervision_rows", 0) or 0),
        "pg280_process_post_accuracy": float(process.get("post_transition_accuracy_min", 0.0) or 0.0),
        "pg280_process_ask_rate": float(process.get("missing_ask_rate_min", 0.0) or 0.0),
        "pg280_process_safe_rate": float(process.get("missing_safe_non_supported_min", 0.0) or 0.0),
        "pg280_hard_negative_rows": int(hard_negative.get("rows", counts.get("family_ood_hard_negative", 0)) or 0),
        "pg280_docker_status": docker_status,
        "pg280_remote_adapter_status": remote_adapter_status,
        "pg280_remote_adapter_audit_pass": remote_adapter_audit_pass,
        "pg280_remote_adapter_probe_hash": str(remote_adapter_probe.get("evidence_sha256", ""))[:16],
        "pg280_remote_adapter_mutations_allowed": bool((remote_adapter_probe.get("scope") or {}).get("mutating_docker_commands_allowed", True)),
        "pg280_gate_status": str(gate.get("status", "blocked")),
        "claim": "PG-280 支持‘缺观测时必须 ASK/保持未决’这一数学与流程结论；final-only 的 post 分数不证明主动排错，也不证明真实应用漏洞能力。",
    })
    brief["findings"].append({
        "id": "pg280-mathematical-non-identifiability",
        "severity": "P0",
        "title": "缺观测是不可辨识，不是训练轮数不足",
        "evidence": f"coarse context 条件熵={float(ident.get('conditional_entropy_bits', 0.0) or 0.0):.1f} bits，Bayes error 下界={float(ident.get('bayes_error_lower_bound', 0.0) or 0.0):.2f}。",
        "action": "记录 pre-state→ASK→observation→repair；在观测缺失时禁止硬猜最终 slot。",
    })
    brief["findings"].append({
        "id": "pg280-final-only-shortcut",
        "severity": "P0",
        "title": "final-only 高分不能替代主动提问",
        "evidence": f"final-only post 最坏={float(final_only.get('post_transition_accuracy_min', 0.0) or 0.0) * 100:.0f}%、pre supervision={int(final_only.get('pre_supervision_rows', 0) or 0)}、ASK={float(final_only.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%；process ASK={float(process.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%。",
        "action": "把 final-only 作为分类对照，把 process supervision 作为主动排错主轨；两者不能合并报一个分数。",
    })
    brief["findings"].append({
        "id": "pg280-docker-unavailable",
        "severity": "P0",
        "title": "授权远程真实 Docker 尚未接通",
        "evidence": f"{source.get('remote_host', 'remote')} legacy probe={docker_status}；adapter probe={remote_adapter_status}；adapter audit={remote_adapter_audit.get('status', 'missing')}。",
        "action": "保持 hard-negative evaluation-only、real gold=0 和 promotion/memory/vulnerability claim 冻结；Docker 可用且 evaluator/重置契约齐全后再做真实应用 gold。",
    })
    for contract in brief.get("record_contract", []):
        if contract.get("group") == "问题前状态":
            for field in ("shared slot ontology tokens", "coarse context collision / conditional entropy"):
                if field not in contract["fields"]:
                    contract["fields"].append(field)
            contract["rule"] = "slot 只从可见证据绑定；若 coarse context 不可辨识，目标必须是 ASK/未决而非虚构最终答案。"
        if contract.get("group") == "动作与观测":
            for field in ("pre-question supervision", "post-observation slot binding", "family-OOD hard-negative lane"):
                if field not in contract["fields"]:
                    contract["fields"].append(field)
    if not any(item.get("category") == "PG-280 shared slot ontology 包" for item in brief.get("resources", [])):
        brief["resources"].append({
            "category": "PG-280 shared slot ontology 包",
            "required": True,
            "items": ["288 条 PG-279 派生记录", "shared effect/control × surface × measure ontology", "16 coarse collision groups / H=1.0 bit", "48 条族外 hard-negative（evaluation-only）", "final-only vs process 对照", "远程 A800 GPU0 报告与独立审计"],
            "why": "验证模型是否学会在信息不足时提问，而不是用最终标签捷径伪装能力。",
        })
    for queue in brief.get("queues", []):
        if queue.get("id") == "Q1-observation-counterfactuals":
            queue["status"] = "review_ready" if audit_pass else "in_progress"
            queue["current"] = f"PG-280 shared ontology 已绑定 {int(counts.get('total', 0) or 0)} 条；coarse H={float(ident.get('conditional_entropy_bits', 0.0) or 0.0):.1f} bit；post collision=0"
        elif queue.get("id") == "Q2-failure-repair-trajectories":
            queue["status"] = "review_ready" if audit_pass else "in_progress"
            queue["current"] = f"process pre supervision={int(process.get('pre_supervision_rows', 0) or 0)}；ASK={float(process.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%；final-only ASK={float(final_only.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%"
        elif queue.get("id") == "Q3-missing-question-recovery":
            queue["status"] = "review_ready" if audit_pass else "in_progress"
            queue["current"] = f"不可辨识性审计={'通过' if audit_pass else '待审'}；safe unresolved={float(process.get('missing_safe_non_supported_min', 0.0) or 0.0) * 100:.0f}%"
        elif queue.get("id") == "Q4-ood-and-forgetting":
            queue["status"] = "blocked_on_real_gold"
            queue["current"] = f"family-OOD hard-negative={int(hard_negative.get('rows', 0) or 0)}（evaluation-only）；Docker={docker_status}；real gold=0"
    brief["promotion_gate"]["current_status"] = "PG-280 operational/math audit passed / scientific family gate blocked / remote Docker unavailable"
    brief["promotion_gate"]["next_experiment"] = "PG-282：远程 Docker 可用且 evaluator/重置契约齐全后，绑定 PG-281 abstract plan 到非破坏性 GET/POST 回放。"
    brief["promotion_gate"]["conditions"].extend([
        "PG-280 final-only 与 process 必须分开报告；final-only 不得声称 ASK/主动排错",
        "family-OOD hard-negative 只能 evaluation-only，不能提升训练或长期记忆",
        "remote Docker unavailable 或 real_application_gold_rows=0 时，training/memory/vulnerability promotion 全部冻结",
    ])
    brief["latest_experiment"] = {
        "id": "PG-280",
        "status": "operational_math_audit_passed_scientific_gate_blocked" if policy_audit_pass else "audit_pending",
        "report": "pg280_ontology_policy_report_v1.json",
        "audit": "pg280_ontology_policy_audit_v1.json",
        "dataset_audit": "pg280_shared_ontology_dataset_audit_v1.json",
        "independent_audit_pass": policy_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "families": family_count,
        "implementations_per_family": implementation_count // family_count if family_count else 0,
        "seeds_per_implementation": seed_count,
        "encodings_per_seed": encoding_count,
        "pre_transition_worst_seed": float(process.get("pre_action_accuracy_min", 0.0) or 0.0),
        "post_transition_worst_seed": float(process.get("post_transition_accuracy_min", 0.0) or 0.0),
        "pair_flip_worst_seed": float(process.get("missing_ask_rate_min", 0.0) or 0.0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "remote_adapter_status": remote_adapter_status,
        "remote_adapter_audit_pass": remote_adapter_audit_pass,
    }
    return brief


def _augment_learning_requirements_with_pg281(
    brief: dict[str, Any],
    report: dict[str, Any],
    dataset_audit: dict[str, Any],
    dataset: dict[str, Any],
    policy_audit: dict[str, Any],
    hard_negative: dict[str, Any],
) -> dict[str, Any]:
    """Expose PG-281's abstract payload-plan policy without claiming live use."""

    counts = dict(dataset.get("counts") or {})
    aggregate = dict(report.get("aggregated") or {})
    guarded = dict(aggregate.get("guarded_sft") or {})
    route = dict(guarded.get("route_dev") or {})
    family = dict(guarded.get("family_holdout") or {})
    hard = dict(guarded.get("hard_negative") or {})
    risk_sweep = dict(report.get("risk_weight_sweep") or {})
    source = dict(report.get("source") or {})
    gate = dict(report.get("hypothesis_gate") or {})
    report_ready = report.get("status") == "completed_remote_pg281_payload_policy_study"
    dataset_audit_pass = dataset_audit.get("status") == "passed"
    policy_audit_pass = policy_audit.get("status") == "passed"
    audit_pass = dataset_audit_pass and policy_audit_pass
    families = len({str(row.get("family")) for row in list(dataset.get("records") or []) if isinstance(row, dict) and row.get("family")})
    brief["schema_version"] = "sift-learning-requirements-v5"
    brief["title"] = "让模型学到真东西的数据任务书 / PG-281 abstract payload plan"
    brief["principle"] = "先让模型从已授权回放学会抽象 probe plan 和 safe gate；literal payload、wire 和 typed oracle 留在 evaluator/人工复核层，证据缺失时必须 abstain。"
    brief["evidence"].update({
        "source_report": "pg281_payload_policy_report_v1.json",
        "source_audit": "pg281_payload_policy_audit_v1.json",
        "dataset_audit": "pg281_payload_policy_dataset_audit_v1.json",
        "report_ready": report_ready,
        "audit_pass": audit_pass,
        "dataset_audit_pass": dataset_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "pg281_train_rows": int(counts.get("train", 0) or 0),
        "pg281_route_dev_rows": int(counts.get("route_dev", 0) or 0),
        "pg281_family_holdout_rows": int(counts.get("family_holdout", 0) or 0),
        "pg281_hard_negative_rows": int(counts.get("hard_negative", 0) or hard_negative.get("records", 0) or 0),
        "pg281_route_positive_recall_min": float(route.get("positive_replay_recall", {}).get("min", 0.0) or 0.0),
        "pg281_family_positive_recall_min": float(family.get("positive_replay_recall", {}).get("min", 0.0) or 0.0),
        "pg281_hard_negative_reject_min": float(hard.get("safe_reject_rate", {}).get("min", 0.0) or 0.0),
        "pg281_hard_negative_false_allow_max": int(hard.get("false_allow_count", {}).get("max", 0) or 0),
        "pg281_selected_variant": str(risk_sweep.get("selected_variant", "not_recorded")),
        "pg281_risk_weight_variant_count": len(dict(risk_sweep.get("variants") or {})),
        "pg281_gate_status": str(gate.get("status", "blocked")),
        "pg281_docker_status": "unavailable" if source.get("remote_docker_available") is False else "unknown",
        "claim": "PG-281 只证明抽象 probe plan 与证据缺失时的安全 gate 可训练；不证明模型已生成或发送真实 payload，也不证明真实应用漏洞能力。",
    })
    brief["findings"].append({
        "id": "pg281-abstract-payload-plan",
        "severity": "P1",
        "title": "先学会选择抽象探针计划，再绑定 wire",
        "evidence": f"guarded route positive replay recall 最坏={float(route.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%，family holdout={float(family.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%；输出是 probe class/channel/encoding/action。",
        "action": "PG-282 才把抽象 plan 绑定到授权 evaluator 的非破坏性 GET/POST wire；当前不输出可执行原始 payload。",
    })
    brief["findings"].append({
        "id": "pg281-safe-send-gate",
        "severity": "P0",
        "title": "证据缺失时拒绝发送",
        "evidence": f"hard-negative safe reject 最坏={float(hard.get('safe_reject_rate', {}).get('min', 0.0) or 0.0) * 100:.0f}%，false-allow 最大={int(hard.get('false_allow_count', {}).get('max', 0) or 0)}。",
        "action": "保持 hard-negative evaluation-only；不能用全 abstain 冒充能力，必须同时审正例 replay recall。",
    })
    if not any(item.get("category") == "PG-281 抽象 probe plan 包" for item in brief.get("resources", [])):
        brief["resources"].append({
            "category": "PG-281 抽象 probe plan 包",
            "required": True,
            "items": ["52 条 PG-266/PG-269 抽象过程记录", "43 train / 4 route-dev / 5 family-holdout", "12 条证据缺失 hard-negative", "plain vs guarded SFT × 3 seed", "safe gate、正例 replay recall、plan exact 和 Brier"],
            "why": "让模型先学会探测顺序、通道和拒答边界，再由真实 evaluator 负责绑定 payload 与判定。",
        })
    for queue in brief.get("queues", []):
        if queue.get("id") == "Q2-failure-repair-trajectories":
            queue["status"] = "review_ready" if audit_pass else "in_progress"
            queue["current"] = f"PG-281 abstract plan：route/family positive recall={float(route.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%/{float(family.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%；hard reject={float(hard.get('safe_reject_rate', {}).get('min', 0.0) or 0.0) * 100:.0f}%"
            queue["collect"].append("probe class / channel / encoding / safe_to_send 的抽象目标；literal payload 留在人审 evaluator")
        elif queue.get("id") == "Q4-ood-and-forgetting":
            queue["status"] = "blocked_on_live_evaluator"
            queue["current"] = f"family holdout={float(family.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%；hard-negative false-allow={int(hard.get('false_allow_count', {}).get('max', 0) or 0)}；Docker unavailable"
    brief["promotion_gate"]["current_status"] = "PG-281 abstract plan gate passed / live evaluator blocked"
    brief["promotion_gate"]["next_experiment"] = "PG-282：授权远程 Docker 可用后，将抽象 plan 绑定到一个非破坏性 GET/POST evaluator，比较 AI plan、reference wire、negative 与 typed oracle。"
    brief["promotion_gate"]["conditions"].extend([
        "PG-281 的 safe_to_send 不能替代 typed oracle；live evaluator 复放前不得形成 payload gold",
        "PG-281 hard-negative false-allow 必须保持 0，同时正例 replay recall 不能退化",
        "远程 Docker unavailable 或 real_application_gold_rows=0 时，训练/记忆/漏洞声明仍冻结",
    ])
    brief["latest_experiment"] = {
        "id": "PG-281",
        "status": "abstract_plan_audit_passed_live_evaluator_blocked" if policy_audit_pass else "audit_pending",
        "report": "pg281_payload_policy_report_v1.json",
        "audit": "pg281_payload_policy_audit_v1.json",
        "dataset_audit": "pg281_payload_policy_dataset_audit_v1.json",
        "independent_audit_pass": policy_audit_pass,
        "controlled_rows": int(counts.get("total", 0) or 0),
        "families": families,
        "implementations_per_family": 0,
        "seeds_per_implementation": 3,
        "encodings_per_seed": 2,
        "pre_transition_worst_seed": float(route.get("safe_accuracy", {}).get("min", 0.0) or 0.0),
        "post_transition_worst_seed": float(route.get("plan_exact_accuracy", {}).get("min", 0.0) or 0.0),
        "pair_flip_worst_seed": float(hard.get("safe_reject_rate", {}).get("min", 0.0) or 0.0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
    }
    return brief


def _augment_learning_requirements_with_pg287(
    brief: dict[str, Any],
    report: dict[str, Any],
    dataset_audit: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Expose PG-287's ambiguity/identifiability result without overstating it.

    PG-287 is a remote A800 diagnostic: it teaches the decoder to ask when an
    encoding slot is not observable, then checks whether an observed encoding
    can be decoded.  The family-held-out resolved score is deliberately kept
    visible because it is the main failure, while the missing live evaluator
    keeps training/memory/vulnerability promotion closed.
    """

    split = dict(report.get("split") or {})
    variants = dict(report.get("variants") or {})
    selected = str(report.get("selected_variant", "not_recorded"))
    metrics = dict(variants.get(selected) or variants.get("plain_sft") or {})
    source = dict(report.get("source") or {})
    engineering = dict(report.get("engineering_gate") or {})
    scientific = dict(report.get("scientific_gate") or {})
    promotion = dict(report.get("promotion") or {})
    def optional_metric(section: str, key: str) -> float | None:
        value = (metrics.get(section) or {}).get(key)
        return None if value is None else float(value)

    family_resolved = optional_metric("family_resolved_encoding_accuracy", "min")
    audit_pass = dataset_audit.get("status") == "passed"
    report_ready = report.get("status") == "completed_remote_pg287_identifiability_training"
    brief["schema_version"] = "sift-learning-requirements-v6"
    brief["title"] = "让模型学到真东西的数据任务书 / PG-287 identifiability"
    brief["principle"] = "先区分不可辨识与可辨识：encoding 观测缺失时目标必须是 ask_typed；只有真实观测 token 出现时才允许解码具体编码。"
    brief["evidence"].update({
        "source_report": "pg287_identifiability_training_report_v1.json",
        "source_audit": "pg287_identifiability_dataset_audit_v1.json",
        "dataset_audit": "pg287_identifiability_dataset_audit_v1.json",
        "report_ready": report_ready,
        "audit_pass": audit_pass,
        "pg287_train_rows": int(split.get("train", 0) or 0),
        "pg287_route_dev_rows": int(split.get("route_dev", 0) or 0),
        "pg287_family_holdout_rows": int(split.get("family_holdout", 0) or 0),
        "pg287_hard_negative_rows": int(split.get("hard_negative", 0) or 0),
        "pg287_route_ambiguous_ask_recall": float((metrics.get("route_ambiguous_ask_recall") or {}).get("min", 0.0) or 0.0),
        "pg287_route_resolved_encoding_accuracy": float((metrics.get("route_resolved_encoding_accuracy") or {}).get("min", 0.0) or 0.0),
        "pg287_family_ambiguous_ask_recall": float((metrics.get("family_ambiguous_ask_recall") or {}).get("min", 0.0) or 0.0),
        "pg287_family_resolved_encoding_accuracy": family_resolved,
        "pg287_hard_negative_ask_recall": float((metrics.get("hard_negative_ask_recall") or {}).get("min", 0.0) or 0.0),
        "pg287_hard_negative_false_allow_max": int(metrics.get("hard_negative_false_allow_max", 0) or 0),
        "pg287_engineering_gate_status": str(engineering.get("status", "blocked")),
        "pg287_scientific_gate_status": str(scientific.get("status", "blocked")),
        "pg287_remote_docker_status": str(source.get("remote_docker_status", "unavailable")),
        "pg287_real_application_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "claim": "PG-287 证明了 ask-on-ambiguity 可在模板化诊断集上学到，但族外 resolved encoding=0%，且没有真实应用 gold；不证明真实 payload 或漏洞能力。",
    })
    if not any(item.get("id") == "pg287-family-identifiability" for item in brief.get("findings", [])):
        brief["findings"].append({
            "id": "pg287-family-identifiability",
            "severity": "P0",
            "title": "族外具体编码仍未学会",
            "evidence": f"ambiguous ASK 最坏={float((metrics.get('family_ambiguous_ask_recall') or {}).get('min', 0.0) or 0.0) * 100:.0f}%，但 family-heldout resolved encoding 覆盖数={int((metrics.get('family_resolved_encoding_accuracy') or {}).get('available_count', 0) or 0)}，因此准确率为 N/A。",
            "action": "下一轮把真实 GET/POST evaluator 的 encoding_observed/field_role 投影接入 source-heldout replay；先补 resolved coverage，没有观测时继续 ask，不用模板标签补齐。",
        })
    if not any(item.get("category") == "PG-287 identifiability 包" for item in brief.get("resources", [])):
        brief["resources"].append({
            "category": "PG-287 identifiability 包",
            "required": True,
            "items": ["6678 train / 756 route-dev / 630 family holdout / 1512 hard-negative", "ambiguous→ask_typed 与 resolved→bounded_wire_plan 配对", "3 seeds，remote A800 GPU0", "dataset audit、training trace、protocol hash"],
            "why": "把编码碰撞从错误硬猜改成可审计的提问；族外 resolved=0% 提醒我们必须接入真实观测，而不是继续堆模板数据。",
        })
    for queue in brief.get("queues", []):
        if queue.get("id") == "Q2-failure-repair-trajectories":
            queue["status"] = "review_ready" if audit_pass else "in_progress"
            queue["current"] = f"PG-287 ambiguous ASK={float((metrics.get('family_ambiguous_ask_recall') or {}).get('min', 0.0) or 0.0) * 100:.0f}%；family resolved encoding coverage={int((metrics.get('family_resolved_encoding_accuracy') or {}).get('available_count', 0) or 0)}（N/A）"
            queue["collect"].append("真实 evaluator 的 observed encoding / field role / response projection；模板 counterfactual 不晋升 gold")
        elif queue.get("id") == "Q4-ood-and-forgetting":
            queue["status"] = "blocked_on_live_evaluator"
            queue["current"] = f"PG-287 family resolved coverage={int((metrics.get('family_resolved_encoding_accuracy') or {}).get('available_count', 0) or 0)}（N/A）；Docker={source.get('remote_docker_status', 'unavailable')}；real gold=0"
    brief["promotion_gate"]["current_status"] = "PG-287 safety metrics recorded / family resolved coverage missing / live evaluator blocked"
    brief["promotion_gate"]["next_experiment"] = "PG-287-live：把真实 GET/POST evaluator 的 observed encoding/field_role projection 接入 source-heldout replay，再重跑族外矩阵。"
    brief["promotion_gate"]["conditions"].extend([
        "family-heldout resolved encoding accuracy 必须从 0% 提升，且不能牺牲 ambiguous ASK recall",
        "hard-negative ask recall 保持 100%、false-allow 保持 0",
        "remote Docker/evaluator unavailable 或 real_application_gold_rows=0 时，training/memory/vulnerability promotion 仍冻结",
    ])
    brief["latest_experiment"] = {
        "id": "PG-287",
        "status": "identifiability_coverage_gap_live_evaluator_blocked" if report_ready and family_resolved is None else ("identifiability_engineering_passed_family_gate_failed" if report_ready else "report_pending"),
        "report": "pg287_identifiability_training_report_v1.json",
        "audit": "pg287_identifiability_dataset_audit_v1.json",
        "dataset_audit": "pg287_identifiability_dataset_audit_v1.json",
        "independent_audit_pass": audit_pass,
        "controlled_rows": int(sum(int(split.get(key, 0) or 0) for key in ("train", "route_dev", "family_holdout", "hard_negative"))),
        "families": 0,
        "implementations_per_family": 0,
        "seeds_per_implementation": 3,
        "encodings_per_seed": 2,
        "pre_transition_worst_seed": float((metrics.get("family_ambiguous_ask_recall") or {}).get("min", 0.0) or 0.0),
        "post_transition_worst_seed": family_resolved,
        "pair_flip_worst_seed": float((metrics.get("hard_negative_ask_recall") or {}).get("min", 0.0) or 0.0),
        "real_multifamily_gold_rows": int(source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "trace": "pg287_identifiability_training_trace_v1.json",
        "trace_sha256": str(trace.get("trace_sha256", "")),
        "promotion_reason": str(promotion.get("reason", "identifiability only; live gold absent")),
    }
    return brief


def build_research_ops_snapshot() -> dict[str, Any]:
    improvement_rules = _read_json("improvement_rules.json", {})
    research_goal = dict(improvement_rules.get("research_goal_v2") or {})
    report_name = "pg255_pikachu_fixed_sql_pg254_replay_report_v1.json"
    report = _read_json(report_name, {})
    pg256_name = "pg256_pikachu_widebyte_oracle_report_v1.json"
    pg256_report = _read_json(pg256_name, {})
    pg257_name = "pg257_widebyte_rule_ir_capacity_training_report_v1.json"
    pg257_report = _read_json(pg257_name, {})
    pg258_name = "pg258_unified_rule_ir_capacity_report_v1.json"
    pg258_report = _read_json(pg258_name, {})
    pg259_name = "pg259_active_belief_capacity_training_report_v1.json"
    pg259_report = _read_json(pg259_name, {})
    pg260_name = "pg260_active_belief_capacity_training_report_v1.json"
    pg260_report = _read_json(pg260_name, {})
    pg261_name = "pg261_masked_active_belief_capacity_training_report_v1.json"
    pg261_report = _read_json(pg261_name, {})
    pg262_name = "pg262_targeted_paired_trace_collection_report_v1.json"
    pg262_report = _read_json(pg262_name, {})
    pg263_name = "pg263_pg262_augmented_masked_capacity_training_report_v1.json"
    pg263_report = _read_json(pg263_name, {})
    pg264_name = "pg264_pikachu_growth_collection_report_v1.json"
    pg264_report = _read_json(pg264_name, {})
    pg264_audit = _read_json("pg264_pikachu_growth_collection_audit_v1.json", {})
    pg265_name = "pg265_growth_augmented_large_capacity_training_report_v1.json"
    pg265_report = _read_json(pg265_name, {})
    pg265_audit = _read_json("pg265_growth_augmented_large_capacity_training_audit_v1.json", {})
    pg265_stop = _read_json("pg265_training_stop_checkpoint_v1.json", {})
    pg265_remote = _read_json("pg265_remote_run_status_v1.json", {})
    pg266_name = "pg266_pikachu_payload_grounding_replay_report_v1.json"
    pg266_report = _read_json(pg266_name, {})
    pg266_catalog = _read_json("pg266_pikachu_payload_grounding_catalog_v1.json", {})
    pg267_name = "pg267_payload_grounding_capacity_training_report_v1.json"
    pg267_report = _read_json(pg267_name, {})
    pg267_audit = _read_json("pg267_payload_grounding_capacity_training_audit_v1.json", {})
    pg268_name = "pg268_pikachu_parameterized_replay_report_v1.json"
    pg268_report = _read_json(pg268_name, {})
    pg268_catalog = _read_json("pg268_pikachu_parameterized_replay_catalog_v1.json", {})
    pg268_manifest = _read_json("pg268_pikachu_browser_parameterized_crawl_manifest_v1.json", {})
    pg268_audit = _read_json("pg268_pikachu_parameterized_replay_audit_v1.json", {})
    pg269_name = "pg269_failure_guided_replay_report_v1.json"
    pg269_report = _read_json(pg269_name, {})
    pg269_catalog = _read_json("pg269_failure_guided_replay_catalog_v1.json", {})
    pg269_audit = _read_json("pg269_failure_guided_replay_audit_v1.json", {})
    pg270_name = "pg270_teacher_sft_ablation_report_v1.json"
    pg270_report = _read_json(pg270_name, {})
    pg270_dataset = _read_json("pg270_teacher_sft_dataset_v1.json", {})
    pg270_audit = _read_json("pg270_teacher_sft_ablation_audit_v1.json", {})
    pg271_name = "pg271_teacher_candidate_replay_report_v1.json"
    pg271_report = _read_json(pg271_name, {})
    pg271_audit = _read_json("pg271_teacher_candidate_replay_audit_v1.json", {})
    pg272_name = "pg272_independent_surface_probe_report_v1.json"
    pg272_report = _read_json(pg272_name, {})
    pg272_audit = _read_json("pg272_independent_surface_probe_audit_v1.json", {})
    pg274_name = "pg274_score_rl_report_v1.json"
    pg274_report = _read_json(pg274_name, {})
    pg274_audit = _read_json("pg274_score_rl_audit_v1.json", {})
    pg275_name = "pg275_hypothesis_ablation_report_v1.json"
    pg275_report = _read_json(pg275_name, {})
    pg275_audit = _read_json("pg275_hypothesis_ablation_audit_v1.json", {})
    pg276_name = "pg276_third_implementation_report_v1.json"
    pg276_report = _read_json(pg276_name, {})
    pg276_audit = _read_json("pg276_third_implementation_audit_v1.json", {})
    pg277_name = "pg277_question_composition_report_v1.json"
    pg277_report = _read_json(pg277_name, {})
    pg277_audit = _read_json("pg277_question_composition_audit_v1.json", {})
    pg277_dataset = _read_json("pg277_counterfactual_question_dataset_v1.json", {})
    pg278_name = "pg278_multifamily_question_policy_report_v1.json"
    pg278_report = _read_json(pg278_name, {})
    pg278_audit = _read_json("pg278_multifamily_question_dataset_audit_v1.json", {})
    pg278_model_audit = _read_json("pg278_multifamily_question_policy_audit_v1.json", {})
    pg278_dataset = _read_json("pg278_multifamily_question_dataset_v1.json", {})
    pg279_name = "pg279_remote_replay_policy_report_v1.json"
    pg279_report = _read_json(pg279_name, {})
    pg279_dataset_audit = _read_json("pg279_remote_replay_dataset_audit_v1.json", {})
    pg279_model_audit = _read_json("pg279_remote_replay_policy_audit_v1.json", {})
    pg279_dataset = _read_json("pg279_remote_replay_dataset_v1.json", {})
    pg279_training_mix = _read_json("pg279_remote_replay_training_mix_v1.json", {})
    pg280_name = "pg280_ontology_policy_report_v1.json"
    pg280_report = _read_json(pg280_name, {})
    pg280_dataset_audit = _read_json("pg280_shared_ontology_dataset_audit_v1.json", {})
    pg280_model_audit = _read_json("pg280_ontology_policy_audit_v1.json", {})
    pg280_dataset = _read_json("pg280_shared_ontology_dataset_v1.json", {})
    pg280_docker_probe = _read_json("pg280_remote_docker_probe_v1.json", {})
    pg280_remote_adapter_probe = _read_json("pg280_remote_docker_probe_v2.json", {})
    pg280_remote_adapter_audit = _read_json("pg280_remote_docker_probe_audit_v1.json", {})
    pg281_name = "pg281_payload_policy_report_v1.json"
    pg281_report = _read_json(pg281_name, {})
    pg281_dataset_audit = _read_json("pg281_payload_policy_dataset_audit_v1.json", {})
    pg281_model_audit = _read_json("pg281_payload_policy_audit_v1.json", {})
    pg281_dataset = _read_json("pg281_payload_policy_dataset_v1.json", {})
    pg281_hard_negative = _read_json("pg281_payload_policy_hard_negative_v1.json", {})
    pg282_name = "pg282_evaluator_binding_report_v1.json"
    pg282_report = _read_json(pg282_name, {})
    pg282_audit = _read_json("pg282_evaluator_binding_audit_v1.json", {})
    pg283_name = "pg283_feedback_policy_report_v1.json"
    pg283_report = _read_json(pg283_name, {})
    pg283_audit = _read_json("pg283_feedback_policy_audit_v1.json", {})
    pg283_dataset = _read_json("pg283_feedback_policy_dataset_v1.json", {})
    pg284_name = "pg284_evaluator_contract_report_v1.json"
    pg284_report = _read_json(pg284_name, {})
    pg284_audit = _read_json("pg284_evaluator_contract_audit_v1.json", {})
    pg285_name = "pg285_payload_grounding_report_v1.json"
    pg285_report = _read_json(pg285_name, {})
    pg285_audit = _read_json("pg285_payload_grounding_audit_v1.json", {})
    pg285_dataset = _read_json("pg285_payload_grounding_dataset_v1.json", {})
    pg286_name = "pg286_observation_token_catalog_v1.json"
    pg286_catalog = _read_json(pg286_name, {})
    pg286_hard_negative = _read_json("pg286_observation_token_hard_negative_v1.json", {})
    pg286_builder_audit = _read_json("pg286_observation_token_catalog_audit_v1.json", {})
    pg286_independent_audit = _read_json("pg286_observation_token_catalog_independent_audit_v1.json", {})
    pg286_protocol = _read_json("pg286_live_protocol_v1.json", {})
    pg286_batch_audit = _read_json("pg286_live_batch_audit_v1.json", {})
    pg287_name = "pg287_identifiability_training_report_v1.json"
    pg287_report = _read_json(pg287_name, {})
    pg287_dataset = _read_json("pg287_identifiability_dataset_v1.json", {})
    pg287_dataset_audit = _read_json("pg287_identifiability_dataset_audit_v1.json", {})
    pg287_trace = _read_json("pg287_identifiability_training_trace_v1.json", {})
    pg287_protocol = _read_json("pg287_identifiability_training_protocol_v1.json", {})
    pg287_live_protocol = _read_json("pg287_live_protocol_v1.json", {})
    pg287_live_batch_audit = _read_json("pg287_live_batch_audit_v1.json", {})
    pg288_name = "pg288_rule_ir_verifier_report_v1.json"
    pg288_report = _read_json(pg288_name, {})
    pg289_name = "pg289_safe_abstain_report_v1.json"
    pg289_report = _read_json(pg289_name, {})
    pg290_name = "pg290_balanced_abstain_report_v1.json"
    pg290_report = _read_json(pg290_name, {})
    pg291_name = "pg291_abstain_gate_report_v1.json"
    pg291_report = _read_json(pg291_name, {})
    pg292_name = "pg292_feature_gate_report_v1.json"
    pg292_report = _read_json(pg292_name, {})
    pg293_name = "pg293_failure_next_action_training_report_v1.json"
    pg293_report = _read_json(pg293_name, {})
    pg293_local_name = "pg293_failure_next_action_training_report_v1_local_morning.json"
    pg293_local_report = _read_json(pg293_local_name, {})
    pg295_name = "pg295_causal_moe_training_report_v1_local_morning.json"
    pg295_report = _read_json(pg295_name, {})
    pg300_name = "pg300_question_policy_training_report_v1_local_morning.json"
    pg300_report = _read_json(pg300_name, {})
    pg300_dataset = _read_json("pg300_question_policy_dataset_v1.json", {})
    pg300_audit = _read_json("pg300_question_policy_audit_v1.json", {})
    pg301_name = "pg301_payload_assembly_training_report_v1_local_morning.json"
    pg301_report = _read_json(pg301_name, {})
    pg302_name = "pg302_symbolic_assembly_training_report_v1_local_morning.json"
    pg302_report = _read_json(pg302_name, {})
    pg302b_name = "pg302b_symbolic_curriculum_training_report_v1_local_morning.json"
    pg302b_report = _read_json(pg302b_name, {})
    pg303_name = "pg303_guarded_composer_eval_report_v1.json"
    pg303_report = _read_json(pg303_name, {})
    pg304_name = "pg304_loopback_replay_fixture_report_v1.json"
    pg304_report = _read_json(pg304_name, {})
    pg305_name = "pg305_live_loopback_replay_report_v1.json"
    pg305_report = _read_json(pg305_name, {})
    pg306_name = "pg306_real_process_moe_training_report_v1_local_morning.json"
    pg306_report = _read_json(pg306_name, {})
    pg306b_name = "pg306b_real_process_curriculum_training_report_v1_local_morning.json"
    pg306b_report = _read_json(pg306b_name, {})
    pg306c_name = "pg306c_balanced_curriculum_training_report_v1_local_morning.json"
    pg306c_report = _read_json(pg306c_name, {})
    pg307_dataset_name = "pg307_symbolic_real_process_dataset_v1.json"
    pg307_dataset = _read_json(pg307_dataset_name, {})
    pg307_audit_name = "pg307_symbolic_real_process_dataset_audit_v1.json"
    pg307_audit = _read_json(pg307_audit_name, {})
    pg307_name = "pg307_symbolic_real_process_moe_training_report_v1_local_morning.json"
    pg307_report = _read_json(pg307_name, {})
    pg308_dataset_name = "pg308_multisource_slot_dataset_v1.json"
    pg308_dataset = _read_json(pg308_dataset_name, {})
    pg308_audit_name = "pg308_multisource_slot_dataset_audit_v1.json"
    pg308_audit = _read_json(pg308_audit_name, {})
    pg308_name = "pg308_multisource_slot_moe_training_report_v1_local_morning.json"
    pg308_report = _read_json(pg308_name, {})
    pg309_dataset_name = "pg309_balanced_counterfactual_dataset_v1.json"
    pg309_dataset = _read_json(pg309_dataset_name, {})
    pg309_audit_name = "pg309_balanced_counterfactual_dataset_audit_v1.json"
    pg309_audit = _read_json(pg309_audit_name, {})
    pg309_name = "pg309_balanced_counterfactual_moe_training_report_v1_local_morning.json"
    pg309_report = _read_json(pg309_name, {})
    pg310_name = "pg310_optimization_ablation_report_v1_local_morning.json"
    pg310_report = _read_json(pg310_name, {})
    pg311_name = "pg311_wide_question_anchor_report_v1_local_morning.json"
    pg311_report = _read_json(pg311_name, {})
    pg312_name = "pg312_live_wide_checkpoint_replay_report_v1.json"
    pg312_report = _read_json(pg312_name, {})
    pg313_dataset_name = "pg313_probe_variant_dataset_v1.json"
    pg313_dataset = _read_json(pg313_dataset_name, {})
    pg313_audit_name = "pg313_probe_variant_dataset_audit_v1.json"
    pg313_audit = _read_json(pg313_audit_name, {})
    pg313_name = "pg313_probe_variant_moe_training_report_v1_local_morning.json"
    pg313_report = _read_json(pg313_name, {})
    pg314_dataset_name = "pg314_independent_variant_training_dataset_v1.json"
    pg314_dataset = _read_json(pg314_dataset_name, {})
    pg314_name = "pg314_independent_variant_replay_report_v1.json"
    pg314_report = _read_json(pg314_name, {})
    pg315_dataset_name = "pg315_worst_seed_training_dataset_v1.json"
    pg315_dataset = _read_json(pg315_dataset_name, {})
    pg315_name = "pg315_worst_seed_replay_report_v1.json"
    pg315_report = _read_json(pg315_name, {})
    pg316_dataset_name = "pg316_failure_repair_dataset_v1.json"
    pg316_dataset = _read_json(pg316_dataset_name, {})
    pg316_name = "pg316_failure_repair_moe_training_report_v1_local_morning.json"
    pg316_report = _read_json(pg316_name, {})
    pg316_live_name = "pg316_live_independent_variant_replay_report_v1.json"
    pg316_live_report = _read_json(pg316_live_name, {})
    pg317_dataset_name = "pg317_question_anchor_dataset_v1.json"
    pg317_dataset = _read_json(pg317_dataset_name, {})
    pg317_audit_name = "pg317_question_anchor_dataset_audit_v1.json"
    pg317_audit = _read_json(pg317_audit_name, {})
    pg317_name = "pg317_question_anchor_moe_training_report_v1_local_morning.json"
    pg317_report = _read_json(pg317_name, {})
    pg317_live_name = "pg317_live_independent_variant_replay_report_v1.json"
    pg317_live_report = _read_json(pg317_live_name, {})
    pg318_name = "pg318_family_holdout_replay_report_v1.json"
    pg318_report = _read_json(pg318_name, {})
    pg318_catalog_name = "pg318_family_holdout_human_catalog_v1.json"
    pg318_trace_name = "pg318_family_holdout_trace_v1.json"
    pg318_protocol_name = "pg318_family_holdout_protocol_v1.json"
    pg319_dataset_name = "pg319_cross_impl_rule_ir_dataset_v1.json"
    pg319_dataset = _read_json(pg319_dataset_name, {})
    pg319_audit_name = "pg319_cross_impl_rule_ir_dataset_audit_v1.json"
    pg319_audit = _read_json(pg319_audit_name, {})
    pg319_name = "pg319_cross_impl_moe_training_report_v1_local_morning.json"
    pg319_report = _read_json(pg319_name, {})
    pg320_dataset_name = "pg320_observation_lattice_dataset_v1.json"
    pg320_dataset = _read_json(pg320_dataset_name, {})
    pg320_audit_name = "pg320_observation_lattice_dataset_audit_v1.json"
    pg320_audit = _read_json(pg320_audit_name, {})
    pg320_name = "pg320_observation_lattice_finetune_report_v1_local_morning.json"
    pg320_report = _read_json(pg320_name, {})
    pg320_live_name = "pg320_family_holdout_replay_report_v1.json"
    pg320_live_report = _read_json(pg320_live_name, {})
    pg321_dataset_name = "pg321_variant_role_lattice_dataset_v1.json"
    pg321_dataset = _read_json(pg321_dataset_name, {})
    pg321_audit_name = "pg321_variant_role_lattice_dataset_audit_v1.json"
    pg321_audit = _read_json(pg321_audit_name, {})
    pg321_name = "pg321_variant_role_finetune_report_v1_local_morning.json"
    pg321_report = _read_json(pg321_name, {})
    pg321_live_name = "pg321_family_holdout_replay_report_v1.json"
    pg321_live_report = _read_json(pg321_live_name, {})
    pg321_catalog_name = "pg321_family_holdout_human_catalog_v1.json"
    pg321_trace_name = "pg321_family_holdout_trace_v1.json"
    pg321_protocol_name = "pg321_family_holdout_protocol_v1.json"
    pg322_dataset_name = "pg322_cross_impl_decoy_dataset_v1.json"
    pg322_dataset = _read_json(pg322_dataset_name, {})
    pg322_audit_name = "pg322_cross_impl_decoy_dataset_audit_v1.json"
    pg322_audit = _read_json(pg322_audit_name, {})
    pg322_name = "pg322_cross_impl_decoy_moe_training_report_v1_local_morning.json"
    pg322_report = _read_json(pg322_name, {})
    pg323_dataset_name = "pg323_decoy_ask_anchor_dataset_v1.json"
    pg323_dataset = _read_json(pg323_dataset_name, {})
    pg323_audit_name = "pg323_decoy_ask_anchor_dataset_audit_v1.json"
    pg323_audit = _read_json(pg323_audit_name, {})
    pg323_name = "pg323_decoy_ask_anchor_moe_training_report_v1_local_morning.json"
    pg323_report = _read_json(pg323_name, {})
    pg323_live_name = "pg323_vulnerableapp_role_replay_report_v1.json"
    pg323_live_report = _read_json(pg323_live_name, {})
    pg323_catalog_name = "pg323_vulnerableapp_role_catalog_v1.json"
    pg323_trace_name = "pg323_vulnerableapp_role_trace_v1.json"
    pg323_protocol_name = "pg323_vulnerableapp_role_protocol_v1.json"
    pg324_name = "pg324_juice_shop_source_heldout_report_v1.json"
    pg324_report = _read_json(pg324_name, {})
    pg324_catalog_name = "pg324_juice_shop_source_heldout_catalog_v1.json"
    pg324_catalog = _read_json(pg324_catalog_name, {})
    pg324_trace_name = "pg324_juice_shop_source_heldout_trace_v1.json"
    pg324_trace = _read_json(pg324_trace_name, {})
    pg324_protocol_name = "pg324_juice_shop_source_heldout_protocol_v1.json"
    pg324_protocol = _read_json(pg324_protocol_name, {})
    pg324_contract = _pg324_contract_projection(pg324_report, pg324_catalog, pg324_trace, pg324_protocol)
    pg325_name = "pg325_sql_family_holdout_report_v1.json"
    pg325_report = _read_json(pg325_name, {})
    pg325_catalog_name = "pg325_sql_family_holdout_catalog_v1.json"
    pg325_catalog = _read_json(pg325_catalog_name, {})
    pg325_trace_name = "pg325_sql_family_holdout_trace_v1.json"
    pg325_trace = _read_json(pg325_trace_name, {})
    pg325_protocol_name = "pg325_sql_family_holdout_protocol_v1.json"
    pg325_protocol = _read_json(pg325_protocol_name, {})
    pg325_audit_name = "pg325_sql_family_holdout_audit_v1.json"
    pg325_audit = _read_json(pg325_audit_name, {})
    pg325_contract = _pg325_contract_projection(pg325_report, pg325_catalog, pg325_trace, pg325_protocol, pg325_audit)
    pg326_name = "pg326_cross_impl_forgetting_matrix_v1.json"
    pg326_report = _read_json(pg326_name, {})
    pg326_protocol_name = "pg326_cross_impl_forgetting_matrix_protocol_v1.json"
    pg326_protocol = _read_json(pg326_protocol_name, {})
    pg326_audit_name = "pg326_cross_impl_forgetting_matrix_audit_v1.json"
    pg326_audit = _read_json(pg326_audit_name, {})
    pg326_contract = _pg326_contract_projection(pg326_report, pg326_protocol, pg326_audit)
    pg327_name = "pg327_a800_replay_training_report_v1.json"
    pg327_report = _read_json(pg327_name, {})
    pg327_contract = _pg327_training_projection(pg327_report)
    pg327b_name = "pg327b_paired_fresh_replay_report_v1.json"
    pg327b_report = _read_json(pg327b_name, {})
    pg327b_audit_name = "pg327b_paired_fresh_replay_audit_v1.json"
    pg327b_audit = _read_json(pg327b_audit_name, {})
    pg327b_contract = _pg327b_replay_projection(pg327b_report, pg327b_audit)
    pg370_name = "pg370_multitask_moe_candidate_v1.json"
    pg370_report = _read_json(pg370_name, {})
    pg370_contract = _pg370_candidate_projection(pg370_report, report_present=(RESEARCH / pg370_name).exists())
    pg373_name = "pg373_staged_pretrain_candidate_v1.json"
    pg373_report = _read_json(pg373_name, {})
    pg373_contract = _pg373_staged_candidate_projection(pg373_report, report_present=(RESEARCH / pg373_name).exists())
    pg374_name = "pg374_model_selected_replay_plan_v1.json"
    pg374_plan = _read_json(pg374_name, {})
    pg374_contract = _pg374_replay_plan_projection(pg374_plan, plan_present=(RESEARCH / pg374_name).exists())
    pg331_audit_name = "pg331_information_preservation_audit_v1.json"
    pg331_audit = _read_json(pg331_audit_name, {})
    pg331_vocab_name = "pg331_web_token_vocabulary_v1.json"
    pg331_vocab = _read_json(pg331_vocab_name, {})
    pg331_contract = _pg331_information_projection(pg331_audit, pg331_vocab)
    pg331_capacity_name = "pg331_model_capacity_audit_v1.json"
    pg331_capacity = _read_json(pg331_capacity_name, {})
    pg331_source_row_audit_name = "pg331_source_row_audit_v1.json"
    pg331_source_row_audit = _read_json(pg331_source_row_audit_name, {})
    pg331_loopback_smoke_name = "pg331_loopback_adapter_smoke_v1.json"
    pg331_loopback_smoke = _read_json(pg331_loopback_smoke_name, {})
    pg331_legacy_manifest_name = "pg331_legacy_web_manifest_audit_v1.json"
    pg331_legacy_manifest = _read_json(pg331_legacy_manifest_name, {})
    pg331_remote_preflight_name = "pg331_remote_a800_readonly_preflight_v1.json"
    pg331_remote_preflight = _read_json(pg331_remote_preflight_name, {})
    pg331_readonly_sources = _pg331_readonly_source_projection(pg331_legacy_manifest, pg331_remote_preflight)
    pg331_source_collection_name = "pg331_pikachu_source_collection_report_v1.json"
    pg331_source_collection = _read_json(pg331_source_collection_name, {})
    pg331_source_dataset_name = "pg331_pikachu_source_row_collection_v1.json"
    pg331_source_dataset = _read_json(pg331_source_dataset_name, {})
    pg331_source_collection_projection = _pg331_source_collection_projection(pg331_source_collection, pg331_source_dataset)
    pg331_typed_source_rows_report_name = "pg331_pikachu_typed_source_rows_report_v1.json"
    pg331_typed_source_rows_report = _read_json(pg331_typed_source_rows_report_name, {})
    pg331_typed_source_rows_audit_name = "pg331_pikachu_typed_source_rows_audit_v1.json"
    pg331_typed_source_rows_audit = _read_json(pg331_typed_source_rows_audit_name, {})
    pg331_typed_sidecars_name = "pg331_pikachu_typed_evaluator_sidecars_v1.json"
    pg331_typed_sidecars = _read_json(pg331_typed_sidecars_name, {})
    pg331_typed_source_rows_projection = _pg331_typed_source_rows_projection(
        pg331_typed_source_rows_report,
        pg331_typed_source_rows_audit,
        pg331_typed_sidecars,
    )
    pg331_typed_capacity_name = "pg331_pikachu_typed_model_capacity_audit_v1.json"
    pg331_typed_capacity = _read_json(pg331_typed_capacity_name, {})
    pg331_typed_capacity_projection = _pg331_typed_capacity_projection(pg331_typed_capacity)
    pg331_holdout_dataset_name = "pg331_train_holdout_diagnostic_v2.json"
    pg331_holdout_dataset = _read_json(pg331_holdout_dataset_name, {})
    pg331_holdout_source_audit_name = "pg331_train_holdout_diagnostic_source_audit_v2.json"
    pg331_holdout_source_audit = _read_json(pg331_holdout_source_audit_name, {})
    pg331_holdout_vocab_name = "pg331_train_holdout_diagnostic_vocab_v2.json"
    pg331_holdout_vocab = _read_json(pg331_holdout_vocab_name, {})
    pg331_holdout_information_name = "pg331_train_holdout_diagnostic_information_v2.json"
    pg331_holdout_information = _read_json(pg331_holdout_information_name, {})
    pg331_holdout_capacity_name = "pg331_train_holdout_diagnostic_capacity_v2.json"
    pg331_holdout_capacity = _read_json(pg331_holdout_capacity_name, {})
    pg331_holdout_plan_name = "pg331_train_holdout_diagnostic_plan_v2.json"
    pg331_holdout_plan = _read_json(pg331_holdout_plan_name, {})
    pg331_holdout_diagnostic = _pg331_train_holdout_diagnostic_v2_projection(
        pg331_holdout_dataset,
        pg331_holdout_dataset,
        pg331_holdout_source_audit,
        pg331_holdout_vocab,
        pg331_holdout_information,
        pg331_holdout_capacity,
        pg331_holdout_plan,
    )
    pg332_dvwa_get_report_name = "pg332_dvwa_typed_get_report_v1.json"
    pg332_dvwa_get_report = _read_json(pg332_dvwa_get_report_name, {})
    pg332_dvwa_get_audit_name = "pg332_dvwa_typed_get_source_rows_audit_v1.json"
    pg332_dvwa_get_audit = _read_json(pg332_dvwa_get_audit_name, {})
    pg332_dvwa_get_sidecars_name = "pg332_dvwa_typed_get_sidecars_v1.json"
    pg332_dvwa_get_sidecars = _read_json(pg332_dvwa_get_sidecars_name, {})
    pg332_dvwa_get_dataset_name = "pg332_dvwa_typed_get_source_rows_v1.json"
    pg332_dvwa_get_dataset = _read_json(pg332_dvwa_get_dataset_name, {})
    pg332_dvwa_post_report_name = "pg332_dvwa_typed_stored_post_report_v1.json"
    pg332_dvwa_post_report = _read_json(pg332_dvwa_post_report_name, {})
    pg332_dvwa_post_audit_name = "pg332_dvwa_typed_stored_post_source_audit_v1.json"
    pg332_dvwa_post_audit = _read_json(pg332_dvwa_post_audit_name, {})
    pg332_dvwa_post_sidecars_name = "pg332_dvwa_typed_stored_post_sidecars_v1.json"
    pg332_dvwa_post_sidecars = _read_json(pg332_dvwa_post_sidecars_name, {})
    pg332_dvwa_post_dataset_name = "pg332_dvwa_typed_stored_post_source_rows_v1.json"
    pg332_dvwa_post_dataset = _read_json(pg332_dvwa_post_dataset_name, {})
    pg332_cross_audit_name = "pg332_dvwa_pikachu_get_post_cross_impl_source_audit_v1.json"
    pg332_cross_audit = _read_json(pg332_cross_audit_name, {})
    pg332_information_name = "pg332_dvwa_pikachu_get_post_cross_impl_information_audit_v3.json"
    pg332_information = _read_json(pg332_information_name, {})
    pg332_capacity_name = "pg332_dvwa_pikachu_get_post_cross_impl_capacity_v2.json"
    pg332_capacity = _read_json(pg332_capacity_name, {})
    pg332_a800_name = "pg332_dvwa_pikachu_get_post_cross_impl_a800_representation_e4_v1.json"
    if not (RESEARCH / pg332_a800_name).exists():
        pg332_a800_name = "pg332_dvwa_pikachu_get_post_cross_impl_a800_representation_smoke_v1.json"
    pg332_a800 = _read_json(pg332_a800_name, {})
    pg332_extended = _pg332_extended_diagnostic_projection(
        pg332_dvwa_get_report,
        pg332_dvwa_get_audit,
        pg332_dvwa_get_sidecars,
        pg332_dvwa_get_dataset,
        pg332_dvwa_post_report,
        pg332_dvwa_post_audit,
        pg332_dvwa_post_sidecars,
        pg332_dvwa_post_dataset,
        pg332_cross_audit,
        pg332_information,
        pg332_capacity,
        pg332_a800,
    )
    pg333_webgoat_report_name = "pg333_webgoat_typed_method_shape_report_v1.json"
    pg333_webgoat_report = _read_json(pg333_webgoat_report_name, {})
    pg333_webgoat_audit_name = "pg333_webgoat_typed_method_shape_source_audit_v1.json"
    pg333_webgoat_audit = _read_json(pg333_webgoat_audit_name, {})
    pg333_webgoat_sidecars_name = "pg333_webgoat_typed_method_shape_sidecars_v1.json"
    pg333_webgoat_sidecars = _read_json(pg333_webgoat_sidecars_name, {})
    pg333_webgoat_dataset_name = "pg333_webgoat_typed_method_shape_source_rows_v1.json"
    pg333_webgoat_dataset = _read_json(pg333_webgoat_dataset_name, {})
    pg333_webgoat_projection = _pg333_webgoat_projection(
        pg333_webgoat_report,
        pg333_webgoat_audit,
        pg333_webgoat_sidecars,
        pg333_webgoat_dataset,
    )
    pg333_cross_dataset_name = "pg333_three_impl_get_post_diagnostic_source_rows_v1.json"
    pg333_cross_dataset = _read_json(pg333_cross_dataset_name, {})
    pg333_cross_source_audit_name = "pg333_three_impl_get_post_diagnostic_source_audit_v1.json"
    pg333_cross_source_audit = _read_json(pg333_cross_source_audit_name, {})
    pg333_cross_information_name = "pg333_three_impl_get_post_diagnostic_information_audit_v1.json"
    pg333_cross_information = _read_json(pg333_cross_information_name, {})
    pg333_cross_vocabulary_name = "pg333_three_impl_get_post_diagnostic_vocabulary_v1.json"
    pg333_cross_vocabulary = _read_json(pg333_cross_vocabulary_name, {})
    pg333_cross_capacity_name = "pg333_three_impl_get_post_diagnostic_capacity_v1.json"
    pg333_cross_capacity = _read_json(pg333_cross_capacity_name, {})
    pg333_cross_a800_name = "pg333_three_impl_a800_representation_e1_v1.json"
    pg333_cross_a800 = _read_json(pg333_cross_a800_name, {})
    pg333_cross_projection = _pg333_cross_impl_projection(
        pg333_cross_dataset,
        pg333_cross_source_audit,
        pg333_cross_information,
        pg333_cross_vocabulary,
        pg333_cross_capacity,
        pg333_cross_a800,
    )
    pg334_process_dataset_name = "pg334_process_token_diagnostic_v1.json"
    pg334_process_dataset = _read_json(pg334_process_dataset_name, {})
    pg334_process_audit_name = "pg334_process_token_diagnostic_audit_v1.json"
    pg334_process_audit = _read_json(pg334_process_audit_name, {})
    pg334_process_vocab_name = "pg334_process_token_vocabulary_v1.json"
    pg334_process_vocab = _read_json(pg334_process_vocab_name, {})
    pg334_process_a800_name = "pg334_a800_process_representation_e1_v1.json"
    pg334_process_a800 = _read_json(pg334_process_a800_name, {})
    pg334_process_projection = _pg334_process_token_projection(
        pg334_process_dataset,
        pg334_process_audit,
        pg334_process_vocab,
        pg334_process_a800,
    )
    pg335_process_dataset_name = "pg335_real_process_token_diagnostic_v1.json"
    pg335_process_dataset = _read_json(pg335_process_dataset_name, {})
    pg335_process_audit_name = "pg335_real_process_token_diagnostic_audit_v1.json"
    pg335_process_audit = _read_json(pg335_process_audit_name, {})
    pg335_process_vocab_name = "pg335_real_process_token_vocabulary_v1.json"
    pg335_process_vocab = _read_json(pg335_process_vocab_name, {})
    pg335_process_a800_name = "pg335_a800_process_representation_e1_v1.json"
    pg335_process_a800 = _read_json(pg335_process_a800_name, {})
    pg335_process_projection = _pg335_real_process_projection(
        pg335_process_dataset,
        pg335_process_audit,
        pg335_process_vocab,
        pg335_process_a800,
    )
    pg336_process_dataset_name = "pg336_real_failure_process_token_v1.json"
    pg336_process_dataset = _read_json(pg336_process_dataset_name, {})
    pg336_process_audit_name = "pg336_real_failure_process_token_audit_v1.json"
    pg336_process_audit = _read_json(pg336_process_audit_name, {})
    pg336_process_vocab_name = "pg336_real_failure_process_vocabulary_v1.json"
    pg336_process_vocab = _read_json(pg336_process_vocab_name, {})
    pg336_process_a800_name = "pg336_a800_real_failure_representation_e1_v1.json"
    pg336_process_a800 = _read_json(pg336_process_a800_name, {})
    pg336_process_projection = _pg336_real_failure_process_projection(
        pg336_process_dataset,
        pg336_process_audit,
        pg336_process_vocab,
        pg336_process_a800,
    )
    pg337_process_dataset_name = "pg337_cross_impl_process_token_v1.json"
    pg337_process_dataset = _read_json(pg337_process_dataset_name, {})
    pg337_process_audit_name = "pg337_cross_impl_process_token_audit_v1.json"
    pg337_process_audit = _read_json(pg337_process_audit_name, {})
    pg337_process_vocab_name = "pg337_cross_impl_process_vocabulary_v1.json"
    pg337_process_vocab = _read_json(pg337_process_vocab_name, {})
    # Prefer the newest completed representation candidate while retaining the
    # earlier e1 smoke as an artifact when e2 is absent.  The projection stays
    # aggregate-only and never promotes either run.
    pg337_process_a800_candidates = (
        "pg337_a800_cross_impl_representation_e2_v1.json",
        "pg337_a800_cross_impl_representation_e1_v1.json",
    )
    pg337_process_a800_name = next(
        (name for name in pg337_process_a800_candidates if (RESEARCH / name).exists()),
        pg337_process_a800_candidates[-1],
    )
    pg337_process_a800 = _read_json(pg337_process_a800_name, {})
    pg337_process_projection = _pg337_cross_impl_process_projection(
        pg337_process_dataset,
        pg337_process_audit,
        pg337_process_vocab,
        pg337_process_a800,
    )
    pg338_process_dataset_name = "pg338_information_preserving_process_token_v1.json"
    pg338_process_dataset = _read_json(pg338_process_dataset_name, {})
    pg338_process_audit_name = "pg338_information_preserving_process_audit_v1.json"
    pg338_process_audit = _read_json(pg338_process_audit_name, {})
    pg338_process_vocab_name = "pg338_information_preserving_vocabulary_v1.json"
    pg338_process_vocab = _read_json(pg338_process_vocab_name, {})
    pg338_process_a800_name = "pg338_a800_information_preserving_representation_e1_v1.json"
    pg338_process_a800 = _read_json(pg338_process_a800_name, {})
    pg338_process_projection = _pg338_information_preserving_projection(
        pg338_process_dataset,
        pg338_process_audit,
        pg338_process_vocab,
        pg338_process_a800,
    )
    pg339_shape_dataset_name = "pg339_multi_shape_diagnostic_dataset_v1.json"
    pg339_shape_dataset = _read_json(pg339_shape_dataset_name, {})
    pg339_shape_audit_name = "pg339_multi_shape_diagnostic_audit_v1.json"
    pg339_shape_audit = _read_json(pg339_shape_audit_name, {})
    pg339_shape_vocab_name = "pg339_multi_shape_vocabulary_v1.json"
    pg339_shape_vocab = _read_json(pg339_shape_vocab_name, {})
    pg339_shape_a800_candidates = (
        "pg339_a800_multi_shape_representation_e6_v1.json",
        "pg339_a800_multi_shape_representation_e5_v1.json",
        "pg339_a800_multi_shape_representation_e4_v1.json",
        "pg339_a800_multi_shape_representation_e3_v1.json",
        "pg339_a800_multi_shape_representation_e2_v1.json",
        "pg339_a800_multi_shape_representation_v1.json",
    )
    pg339_shape_a800_name = next(
        (name for name in pg339_shape_a800_candidates if (RESEARCH / name).exists()),
        pg339_shape_a800_candidates[-1],
    )
    pg339_shape_a800 = _read_json(pg339_shape_a800_name, {})
    pg339_shape_projection = _pg339_multi_shape_projection(
        pg339_shape_dataset,
        pg339_shape_audit,
        pg339_shape_vocab,
        pg339_shape_a800,
    )
    pg340_axis_dataset_name = "pg340_balanced_axis_representation_dataset_v1.json"
    pg340_axis_dataset = _read_json(pg340_axis_dataset_name, {})
    pg340_axis_audit_name = "pg340_balanced_axis_representation_audit_v1.json"
    pg340_axis_audit = _read_json(pg340_axis_audit_name, {})
    pg340_axis_vocab_name = "pg340_balanced_axis_vocabulary_v1.json"
    pg340_axis_vocab = _read_json(pg340_axis_vocab_name, {})
    pg340_axis_a800_candidates = ("pg340_a800_balanced_axis_representation_e1_v1.json",)
    pg340_axis_a800_name = next(
        (name for name in pg340_axis_a800_candidates if (RESEARCH / name).exists()),
        pg340_axis_a800_candidates[-1],
    )
    pg340_axis_a800 = _read_json(pg340_axis_a800_name, {})
    pg340_axis_projection = _pg340_balanced_axis_projection(
        pg340_axis_dataset,
        pg340_axis_audit,
        pg340_axis_vocab,
        pg340_axis_a800,
    )
    pg341_target_dataset_name = "pg341_target_conditioned_process_full_axis_dataset_v1.json"
    pg341_target_dataset = _read_json(pg341_target_dataset_name, {})
    pg341_target_audit_name = "pg341_target_conditioned_audit_v1.json"
    pg341_target_audit = _read_json(pg341_target_audit_name, {})
    pg341_target_vocab_name = "pg341_target_conditioned_vocabulary_v1.json"
    pg341_target_vocab = _read_json(pg341_target_vocab_name, {})
    pg341_target_a800_candidates = (
        "pg341_a800_target_conditioned_smoke_e4_v1.json",
        "pg341_a800_target_conditioned_smoke_e3_v1.json",
        "pg341_a800_target_conditioned_smoke_e2_v1.json",
        "pg341_a800_target_conditioned_smoke_v1.json",
    )
    pg341_target_a800_name = next(
        (name for name in pg341_target_a800_candidates if (RESEARCH / name).exists()),
        pg341_target_a800_candidates[-1],
    )
    pg341_target_a800_reports = [
        _read_json(name, {}) for name in pg341_target_a800_candidates if (RESEARCH / name).exists()
    ]
    pg341_target_projection = _pg341_target_conditioned_projection(
        pg341_target_dataset,
        pg341_target_audit,
        pg341_target_vocab,
        pg341_target_a800_reports,
    )
    pg342_failure_dataset_name = "pg342_full_axis_failure_repair_dataset_v1.json"
    pg342_failure_dataset = _read_json(pg342_failure_dataset_name, {})
    pg342_failure_audit_name = "pg342_full_axis_failure_repair_audit_v1.json"
    pg342_failure_audit = _read_json(pg342_failure_audit_name, {})
    pg342_failure_vocab_name = "pg342_full_axis_failure_repair_vocabulary_v1.json"
    pg342_failure_vocab = _read_json(pg342_failure_vocab_name, {})
    pg342_failure_source_name = "pg342_webgoat_failure_repair_report_v1.json"
    pg342_failure_source = _read_json(pg342_failure_source_name, {})
    pg342_failure_a800_candidates = (
        "pg342_a800_full_axis_representation_smoke_v2.json",
        "pg342_a800_full_axis_representation_smoke_v1.json",
    )
    pg342_failure_a800_name = next(
        (name for name in pg342_failure_a800_candidates if (RESEARCH / name).exists()),
        pg342_failure_a800_candidates[-1],
    )
    pg342_failure_a800 = _read_json(pg342_failure_a800_name, {})
    pg342_failure_projection = _pg342_full_axis_failure_repair_projection(
        pg342_failure_dataset,
        pg342_failure_audit,
        pg342_failure_vocab,
        pg342_failure_a800,
        pg342_failure_source,
    )
    pg221_name = "pg221_pikachu_boolean_blind_oracle_report_v1.json"
    pg221_report = _read_json(pg221_name, {})
    pg242_name = "pg242_pikachu_xss_dom_acceptance_report_v1.json"
    pg242_report = _read_json(pg242_name, {})
    crawl_name = "pg179_pikachu_browser_crawl_manifest_v1.json"
    crawl_manifest = _read_json(crawl_name, {})
    surface_catalog = _surface_catalog_projection(crawl_manifest)
    episodes = [dict(item) for item in list(report.get("episodes") or []) if isinstance(item, dict)]
    counts = dict(report.get("counts") or {})
    pg256_episodes = [dict(item) for item in list(pg256_report.get("episodes") or []) if isinstance(item, dict)]
    pg242_results = [dict(item) for item in list(pg242_report.get("results") or []) if isinstance(item, dict)]
    pg259_sql_report = _read_json("pg259_fresh_sql_replay_report_v1.json", {})
    pg259_xss_report = _read_json("pg259_fresh_xss_replay_report_v1.json", {})
    pg259_boolean_report = _read_json("pg259_fresh_boolean_replay_report_v1.json", {})
    pg259_widebyte_report = _read_json("pg259_fresh_widebyte_replay_report_v1.json", {})
    pg259_sql_episodes = [dict(item) for item in list(pg259_sql_report.get("episodes") or []) if isinstance(item, dict)]
    pg259_xss_results = [dict(item) for item in list(pg259_xss_report.get("results") or []) if isinstance(item, dict)]
    pg259_boolean_results = [dict(item) for item in list(pg259_boolean_report.get("results") or []) if isinstance(item, dict)]
    pg259_widebyte_episodes = [dict(item) for item in list(pg259_widebyte_report.get("episodes") or []) if isinstance(item, dict)]
    tasks = [_episode_task(row, index) for index, row in enumerate(episodes)]
    tasks.extend(_episode_task(row, index, prefix="pg256") for index, row in enumerate(pg256_episodes))
    tasks.extend(_episode_task(row, index, prefix="pg242") for index, row in enumerate(pg242_results))
    tasks.extend(_episode_task(row, index, prefix="pg259-sql") for index, row in enumerate(pg259_sql_episodes))
    tasks.extend(_episode_task(row, index, prefix="pg259-xss") for index, row in enumerate(pg259_xss_results))
    process_traces = [
        *[_process_trace(row, index, prefix="pg255", family="sql") for index, row in enumerate(episodes)],
        *[_process_trace(row, index, prefix="pg256", family="sql_widebyte") for index, row in enumerate(pg256_episodes)],
        *[_process_trace(row, index, prefix="pg242", family="xss_dom") for index, row in enumerate(pg242_results)],
        *[_process_trace(row, index, prefix="pg259-sql", family="sql") for index, row in enumerate(pg259_sql_episodes)],
        *[_process_trace(row, index, prefix="pg259-xss", family="xss_dom") for index, row in enumerate(pg259_xss_results)],
    ]
    # Boolean and widebyte child reports use a compact ``oracle`` object;
    # normalize only the bounded status flags needed by the replay timeline.
    for prefix, family, source_rows in (
        ("pg259-boolean", "sql_boolean", pg259_boolean_results),
        ("pg259-widebyte", "sql_widebyte", pg259_widebyte_episodes),
    ):
        for index, raw in enumerate(source_rows):
            row = dict(raw)
            oracle = dict(row.get("oracle") or {})
            row["typed_oracle"] = {"confirmed_positive": bool(oracle.get("confirmed_positive") or oracle.get("boolean_effect_confirmed") or oracle.get("widebyte_effect_confirmed") or oracle.get("typed_effect_confirmed")), "typed_effect_confirmed": bool(oracle.get("confirmed_positive") or oracle.get("boolean_effect_confirmed") or oracle.get("widebyte_effect_confirmed") or oracle.get("typed_effect_confirmed")), "evidence_hash": str(oracle.get("evidence_hash", "")), "reasons": list(oracle.get("reasons") or [])}
            row.setdefault("ai", {"sent": True})
            process_traces.append(_process_trace(row, index, prefix=prefix, family=family))
    confirmed = [task for task in tasks if task["confirmed_positive"]]
    collector = [task for task in tasks if task["role"] == "collector"]
    reviewer = [task for task in tasks if task["role"] == "reviewer"]
    training_report = _read_json("pg254_pikachu_payload_catalog_capacity_training_report_v1.json", {})
    training_judge = dict(training_report.get("independent_final_judge") or {})
    selected = dict(training_report.get("selected") or {})
    architecture = [
        {"id": "surface", "title": "Surface tokens", "subtitle": "HTML / GET / POST / JS", "detail": "把页面结构、方法、字段和重定向压成可跨语言复用的表面 token。", "owner": "collector"},
        {"id": "rule-ir", "title": "Rule IR", "subtitle": "抽象槽位与证据绑定", "detail": "候选只表达边界、编码、失败类型和下一步，不把路由名称当捷径。", "owner": "model"},
        {"id": "belief", "title": "Belief update", "subtitle": "失败 → 诊断 → 修复", "detail": "失败信息调整历史 token 权重，再回到正向推理；不可复现的失败进入 quarantine。", "owner": "model"},
        {"id": "probe-gate", "title": "Probe gate", "subtitle": "安全探针发送决策", "detail": "冻结的 legacy policy 与 active-belief adapter 共同决定是否允许 loopback probe；不能跳过 reference/negative。", "owner": "model"},
        {"id": "capacity", "title": "Capacity sweep", "subtitle": "2048 / 4096 / 8192 + abstain", "detail": "PG-260 比较三档 adapter 容量，并单独训练 unknown-family abstain；选择依据是留出 Rule-IR、family、abstain 和 token loss，而不是训练集分数。", "owner": "trainer"},
        {"id": "oracle", "title": "Typed oracle", "subtitle": "独立最终判官", "detail": "reference、negative、fresh reset、源码哈希和证据哈希决定效果，结果不回流为输入。", "owner": "reviewer"},
        {"id": "memory", "title": "Dataset / memory", "subtitle": "gold / hard-negative / quarantine", "detail": "只有完整、可复放、跨 seed/route 留出的记录才允许进入训练候选。", "owner": "trainer"},
    ]
    architecture.append({"id": "fresh-augmentation", "title": "Fresh augmentation", "subtitle": "PG-262 → PG-263", "detail": "把新的 Pikachu GET/POST paired traces 作为全新偶数 seed 留出；先审计数据完整性，再做 mask-aware 容量复跑。", "owner": "trainer"})
    architecture.append({"id": "growth-tranche", "title": "Growth tranche", "subtitle": "PG-264 fresh seeds", "detail": "先增加全新 SQL/XSS/boolean/widebyte seed，再把完整审计的抽象 token 交给更大容量训练；缺字段的 episode 不得进入模型。", "owner": "collector"})
    architecture.append({"id": "large-capacity", "title": "Large adapter", "subtitle": "PG-265 4096 / 8192 / 12288", "detail": "把数据增长与容量增长分开验收；12288 只在 PG-264 完整审计后运行，仍保留族外、遗忘 canary 和晋级冻结。", "owner": "trainer"})
    architecture.append({"id": "payload-grounding", "title": "Payload grounding", "subtitle": "PG-266 wire → echo → oracle", "detail": "将 Rule-IR 候选绑定到已观测 GET/POST 字段，在全新 Pikachu 容器中回放；人可看到 wire 和有限回显，训练只接收抽象结果 token。", "owner": "reviewer"})
    architecture.append({"id": "payload-grounding-capacity", "title": "Grounded token adapter", "subtitle": "PG-267 8192 / 12288 / 16384", "detail": "把 PG-266 的真实回放压成抽象 token，payload 与 oracle 仍在模型输入外；偶数 seed 做 fresh holdout，route-seed 门未过则保持候选。", "owner": "trainer"})
    architecture.append({"id": "parameterized-replay", "title": "Parameterized replay", "subtitle": "PG-268B GET / POST / reset / oracle", "detail": "浏览器发现的参数上下文逐路由进入 AI candidate、reference、negative 与 fresh reset 回放；oracle gap 和 multipart 缺口显式 abstain，不产生训练标签。", "owner": "reviewer"})
    architecture.append({"id": "shared-slot-ontology", "title": "Shared slot ontology", "subtitle": "PG-280 H(Y|X) / ASK gate", "detail": "把 effect/control、surface、measure 压成跨族 Rule IR slot；若 coarse context 条件熵为正，训练 ASK/未决而不是凭最终标签硬猜。族外 hard-negative 只进 evaluation lane。", "owner": "model"})
    architecture.append({"id": "abstract-payload-policy", "title": "Abstract payload policy", "subtitle": "PG-281 plan → safe gate", "detail": "先预测 probe class、GET/POST channel、encoding 与 final action；safe_to_send 只表示进入授权 evaluator 的候选资格，不能替代 typed oracle，也不输出 literal payload。", "owner": "model"})
    architecture.append({"id": "identifiability", "title": "Identifiability / ASK", "subtitle": "PG-287 ambiguous → ask_typed", "detail": "把同一可见 context 对应多个编码的碰撞显式标成不可辨识：缺少 observed encoding 时输出 ask_typed，只有真实 GET/POST evaluator 提供 field-role/encoding 投影后才解码具体 plan。族外 resolved=0% 时优先补观测，不继续堆模板轮数。", "owner": "model"})
    architecture.append({"id": "feature-safety-gate", "title": "Feature safety gate", "subtitle": "PG-292 key/value OOD abstain", "detail": "把 typed_available、feedback、replay 等观测槽拆成可组合 key/value 特征；未知 evaluator 名称只作为未知值，不得让模型把缺证据当成可发送。route/family 正例召回与族外 false-allow 分开验收。", "owner": "model"})
    architecture.append({"id": "whole-web-token-ontology", "title": "Whole-web token ontology", "subtitle": "PG-331 document → request → response → JS → belief", "detail": "不把网页压成少数 surface 标签；用版本化 ontology tokenizer 与 strict source-row collector 保留 DOM/导航、GET/POST 参数、响应/302、JavaScript AST/source/sink、失败转移和 replay/belief 轴，缺观测显式 not_observed，sidecar 不进入模型 context。", "owner": "collector"})
    # Keep the headline process metric aligned with every trace rendered in
    # the replay panel, including the newly collected PG-259 lanes.
    total_episodes = len(process_traces)
    total_sends = sum(int(trace["ai_sent"]) for trace in process_traces)
    total_typed = sum(int(trace["confirmed_positive"]) for trace in process_traces)
    total_fresh = sum(int(trace["fresh_reset"]) for trace in process_traces)
    total_false = int(counts.get("false_positive_count", 0) or 0) + int(pg256_report.get("counts", {}).get("false_positive_count", 0) or 0) + int(pg242_report.get("counts", {}).get("false_positive_count", 0) or 0) + sum(int(dict(item.get("counts") or {}).get("false_positive_count", 0) or 0) for item in (pg259_sql_report, pg259_xss_report, pg259_boolean_report, pg259_widebyte_report))
    metrics = [
        {"id": "send", "label": "AI 预探针发送", "value": f"{total_sends}/{total_episodes}", "status": "pass", "note": "AI 在真实 GET/POST send path 决策"},
        {"id": "typed", "label": "typed effect", "value": f"{total_typed}/{total_episodes}", "status": "partial", "note": "结果/DOM 类型化 oracle；不等于公网漏洞"},
        {"id": "widebyte", "label": "PG-256 宽字节", "value": f"{int(pg256_report.get('counts', {}).get('confirmed_positive_count', 0) or 0)}/{int(pg256_report.get('counts', {}).get('episode_count', 0) or 0)}", "status": "pass" if int(pg256_report.get("counts", {}).get("confirmed_positive_count", 0) or 0) else "partial", "note": "失败反馈后 Rule-IR class 跨 seed 迁移"},
        {"id": "boolean", "label": "PG-221 布尔 oracle", "value": f"{int(pg221_report.get('counts', {}).get('confirmed_positive_count', 0) or 0)}/{int(pg221_report.get('counts', {}).get('fresh_container_count', 0) or 0)}", "status": "pass" if int(pg221_report.get("counts", {}).get("confirmed_positive_count", 0) or 0) == int(pg221_report.get("counts", {}).get("fresh_container_count", 0) or 0) and int(pg221_report.get("counts", {}).get("fresh_container_count", 0) or 0) > 0 else "partial", "note": "独立 fresh seed true/false 行差分；与 PG-255 任务分开计"},
        {"id": "xss", "label": "PG-242 XSS DOM oracle", "value": f"{int(pg242_report.get('counts', {}).get('confirmed_positive_count', 0) or 0)}/{int(pg242_report.get('counts', {}).get('ai_send_count', 0) or 0)}", "status": "pass" if int(pg242_report.get("counts", {}).get("confirmed_positive_count", 0) or 0) > 0 and int(pg242_report.get("counts", {}).get("false_positive_count", 0) or 0) == 0 else "partial", "note": "受控浏览器 marker；2 条 oracle gap 保持 abstain"},
        {"id": "pg257", "label": "PG-257 Rule-IR 留出", "value": f"{float(((pg257_report.get('selected') or {}).get('metrics') or {}).get('seed_holdout', {}).get('rule_accuracy', 0.0) or 0.0) * 100:.0f}%", "status": "pass" if str(pg257_report.get("status", "")) == "completed_rule_ir_class_capacity_training" else "partial", "note": "偶数 seed 留出；oracle 标签不进入模型输入"},
        {"id": "pg258", "label": "PG-258 SQL/XSS Rule-IR", "value": f"{float((((pg258_report.get('selected') or {}).get('metrics') or {}).get('seed_route_family_holdout') or {}).get('rule_accuracy', 0.0) or 0.0) * 100:.0f}%", "status": "pass" if bool((pg258_report.get("independent_final_judge") or {}).get("pass")) else "blocked", "note": "留出 rule/family + VulnerableApp OOD；当前晋级冻结"},
        {"id": "pg259", "label": "PG-259 active-belief", "value": f"{float((((pg259_report.get('selected') or {}).get('metrics') or {}).get('fresh_route_holdout') or {}).get('rule_accuracy', 0.0) or 0.0) * 100:.0f}%", "status": "pass" if bool((pg259_report.get("independent_final_judge") or {}).get("pass")) else "blocked", "note": "fresh 跨路由 Rule-IR；belief/probe 头与 OOD 独立计分"},
        {"id": "false", "label": "误报", "value": str(total_false), "status": "pass", "note": "matched negative 未出现误报"},
        {"id": "fresh", "label": "fresh reset", "value": f"{total_fresh}/{total_episodes}", "status": "pass", "note": "无卷容器 + 数据库健康门"},
    ]
    learning_requirements = _augment_learning_requirements_with_pg278(
        _learning_requirements(pg277_report, pg277_audit, pg277_dataset),
        pg278_report,
        pg278_audit,
        pg278_dataset,
        pg278_model_audit,
    )
    learning_requirements = _augment_learning_requirements_with_pg279(
        learning_requirements,
        pg279_report,
        pg279_dataset_audit,
        pg279_dataset,
        pg279_model_audit,
        pg279_training_mix,
    )
    learning_requirements = _augment_learning_requirements_with_pg280(
        learning_requirements,
        pg280_report,
        pg280_dataset_audit,
        pg280_dataset,
        pg280_model_audit,
        pg280_docker_probe,
        pg280_remote_adapter_probe,
        pg280_remote_adapter_audit,
    )
    learning_requirements = _augment_learning_requirements_with_pg281(
        learning_requirements,
        pg281_report,
        pg281_dataset_audit,
        pg281_dataset,
        pg281_model_audit,
        pg281_hard_negative,
    )
    learning_requirements = _augment_learning_requirements_with_pg287(
        learning_requirements,
        pg287_report,
        pg287_dataset_audit,
        pg287_trace,
    )
    snapshot = {
        "schema_version": "sift-research-ops-snapshot-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_location_policy": dict(improvement_rules.get("execution_location_policy") or {}),
        "research_goal": {"title": str(research_goal.get("title", "")), "objective": str(research_goal.get("objective", "")), "priority_order": [str(item) for item in list(research_goal.get("priority_order") or [])], "training_stack": [dict(item) for item in list(research_goal.get("training_stack") or []) if isinstance(item, dict)], "mentor_judge_loop": dict(research_goal.get("mentor_judge_loop") or {}), "next_experiment": str(research_goal.get("next_experiment", "")), "non_goal": [str(item) for item in list(research_goal.get("non_goal") or [])]},
        "source_reports": [{"name": report_name, "updated_at": _report_time(report_name), "sha256": str(report.get("report_sha256", ""))}, {"name": pg221_name, "updated_at": _report_time(pg221_name), "sha256": str(pg221_report.get("report_sha256", ""))}, {"name": pg242_name, "updated_at": _report_time(pg242_name), "sha256": str(pg242_report.get("report_sha256", ""))}, {"name": pg256_name, "updated_at": _report_time(pg256_name), "sha256": str(pg256_report.get("report_sha256", ""))}, {"name": pg257_name, "updated_at": _report_time(pg257_name), "sha256": str(pg257_report.get("report_sha256", ""))}, {"name": pg258_name, "updated_at": _report_time(pg258_name), "sha256": str(pg258_report.get("report_sha256", ""))}, {"name": pg259_name, "updated_at": _report_time(pg259_name), "sha256": str(pg259_report.get("report_sha256", ""))}, {"name": pg279_name, "updated_at": _report_time(pg279_name), "sha256": str(pg279_report.get("report_sha256", ""))}, {"name": pg280_name, "updated_at": _report_time(pg280_name), "sha256": str(pg280_report.get("report_sha256", ""))}, {"name": "pg280_ontology_policy_audit_v1.json", "updated_at": _report_time("pg280_ontology_policy_audit_v1.json"), "sha256": str(pg280_model_audit.get("audit_sha256", ""))}, {"name": "pg280_remote_docker_probe_v2.json", "updated_at": _report_time("pg280_remote_docker_probe_v2.json"), "sha256": str(pg280_remote_adapter_probe.get("evidence_sha256", ""))}, {"name": "pg280_remote_docker_probe_audit_v1.json", "updated_at": _report_time("pg280_remote_docker_probe_audit_v1.json"), "sha256": str(pg280_remote_adapter_audit.get("audit_sha256", ""))}, {"name": pg281_name, "updated_at": _report_time(pg281_name), "sha256": str(pg281_report.get("report_sha256", ""))}, {"name": "pg281_payload_policy_audit_v1.json", "updated_at": _report_time("pg281_payload_policy_audit_v1.json"), "sha256": str(pg281_model_audit.get("audit_sha256", ""))}, {"name": pg282_name, "updated_at": _report_time(pg282_name), "sha256": str(pg282_report.get("report_sha256", ""))}, {"name": "pg282_evaluator_binding_audit_v1.json", "updated_at": _report_time("pg282_evaluator_binding_audit_v1.json"), "sha256": str(pg282_audit.get("audit_sha256", ""))}, {"name": pg283_name, "updated_at": _report_time(pg283_name), "sha256": str(pg283_report.get("report_sha256", ""))}, {"name": "pg283_feedback_policy_audit_v1.json", "updated_at": _report_time("pg283_feedback_policy_audit_v1.json"), "sha256": str(pg283_audit.get("audit_sha256", ""))}, {"name": pg327_name, "updated_at": _report_time(pg327_name), "sha256": str(pg327_report.get("report_sha256", ""))}, {"name": pg327b_name, "updated_at": _report_time(pg327b_name), "sha256": str(pg327b_audit.get("audit_sha256", ""))}, {"name": pg327b_audit_name, "updated_at": _report_time(pg327b_audit_name), "sha256": str(pg327b_audit.get("audit_sha256", ""))}, {"name": pg331_audit_name, "updated_at": _report_time(pg331_audit_name), "sha256": str(pg331_audit.get("audit_sha256", ""))}, {"name": pg331_vocab_name, "updated_at": _report_time(pg331_vocab_name), "sha256": str(pg331_vocab.get("vocabulary_sha256", ""))}, {"name": pg331_capacity_name, "updated_at": _report_time(pg331_capacity_name), "sha256": str(pg331_capacity.get("audit_sha256", ""))}, {"name": crawl_name, "updated_at": _report_time(crawl_name), "sha256": str(crawl_manifest.get("manifest_id", ""))}],
        "judge": {"name": "SIFT final judge", "scope": "loopback-only Pikachu typed effect", "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "tasks": {"all": tasks, "collector": collector, "reviewer": reviewer, "trainer": [{"id": "pg254-train-promotion", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked", "label": "训练晋级门未开放", "route": "PG-254 route-family holdout", "seed": 25402, "typed_effect": False, "confirmed_positive": False, "reasons": ["training promotion is intentionally disabled", "payload catalog promotion is blocked"], "evidence_hash": str(training_report.get("report_sha256", ""))[:16], "instruction": "AI 先生成隔离训练候选；人工只需在新增数据通过最终判官后批准晋级。", "raw_material_available": False}, {"id": "pg257-rule-ir-capacity", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked", "label": "PG-257 Rule-IR 留出训练", "route": "PG-257 odd/even seed holdout", "seed": 25702, "typed_effect": False, "confirmed_positive": False, "reasons": ["16 records only", "training and memory promotion are intentionally disabled", "general web capability is not established"], "evidence_hash": str(pg257_report.get("report_sha256", ""))[:16], "instruction": "人工查看留出矩阵、类别召回和遗忘 canary 后，才能决定是否扩展数据；当前不晋级。", "raw_material_available": False}, {"id": "pg258-unified-rule-ir", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked", "label": "PG-258 SQL/XSS Rule-IR 族外训练", "route": "PG-258 seed/route + VulnerableApp OOD", "seed": 25802, "typed_effect": False, "confirmed_positive": False, "reasons": list((pg258_report.get("independent_final_judge") or {}).get("reasons") or ["report unavailable"]), "evidence_hash": str(pg258_report.get("report_sha256", ""))[:16], "instruction": "人工查看每类留出支持数、rule/family 留出、实现族外分数和旧策略 canary；本轮未通过，不晋级。", "raw_material_available": False}, {"id": "pg259-active-belief", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked", "label": "PG-259 fresh active-belief 训练", "route": "PG-259 fresh route holdout + VulnerableApp OOD", "seed": 25902, "typed_effect": False, "confirmed_positive": False, "reasons": list((pg259_report.get("independent_final_judge") or {}).get("reasons") or ["report unavailable"]), "evidence_hash": str(pg259_report.get("report_sha256", ""))[:16], "instruction": "人工查看 fresh route Rule-IR、belief/probe、实现 OOD 与遗忘 canary；未通过前不晋级。", "raw_material_available": False}, {"id": "pg327-a800-replay", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked", "label": "PG-327 A800 replay-mix 候选训练", "route": "远程 A800 GPU0 · 三 seed · PG-323 抽象 replay", "seed": 31902, "typed_effect": False, "confirmed_positive": False, "reasons": ["offline candidate only", "PG-326 strict schema incomplete", "paired forgetting replay missing"], "evidence_hash": str(pg327_report.get("report_sha256", ""))[:16], "instruction": "查看每个 seed 的 ASK/variant/hard-negative/old-retention 与 provenance 哈希；允许继续候选训练，但不批准长期记忆、payload catalog 或漏洞能力声明。", "raw_material_available": False}]},
        "process_traces": process_traces,
        "surface_catalog": surface_catalog,
        "learning_requirements": learning_requirements,
        "architecture": architecture,
        "capability": {"metrics": metrics, "model": {"selected_hidden_dim": selected.get("hidden_dim", 4096), "adapter_parameter_count": selected.get("adapter_parameter_count", 4745349), "route_holdout_pass": bool(training_judge.get("pass")), "model_input_uses_oracle": False, "pg256_policy": str(pg256_report.get("model", {}).get("policy", "not_run")), "pg256_failure_feedback": bool(pg256_report.get("model", {}).get("failure_feedback_updates_selection", False)), "pg257": {"status": str(pg257_report.get("status", "not_run")), "selected_hidden_dim": int((pg257_report.get("selected") or {}).get("hidden_dim", 0) or 0), "seed_holdout_rule_accuracy": float(((pg257_report.get("selected") or {}).get("metrics") or {}).get("seed_holdout", {}).get("rule_accuracy", 0.0) or 0.0), "seed_holdout_widebyte_recall": float(((pg257_report.get("selected") or {}).get("metrics") or {}).get("seed_holdout", {}).get("widebyte_escape_boundary_recall", 0.0) or 0.0), "seed_holdout_next_token_accuracy": float(((pg257_report.get("selected") or {}).get("metrics") or {}).get("next_token_accuracy", 0.0) or 0.0), "record_count": int((pg257_report.get("counts") or {}).get("records", 0) or 0), "promotion_blocked": bool((pg257_report.get("promotion") or {}).get("training_promotion_allowed") is False)}, "pg258": {"status": str(pg258_report.get("independent_final_judge", {}).get("decision", "not_run")), "selected_hidden_dim": int((pg258_report.get("selected") or {}).get("hidden_dim", 0) or 0), "holdout_rule_accuracy": float((((pg258_report.get("selected") or {}).get("metrics") or {}).get("seed_route_family_holdout") or {}).get("rule_accuracy", 0.0) or 0.0), "holdout_family_accuracy": float((((pg258_report.get("selected") or {}).get("metrics") or {}).get("seed_route_family_holdout") or {}).get("family_accuracy", 0.0) or 0.0), "implementation_ood_family_accuracy": float((((pg258_report.get("selected") or {}).get("metrics") or {}).get("implementation_ood") or {}).get("family_accuracy", 0.0) or 0.0), "record_count": int((pg258_report.get("counts") or {}).get("records", 0) or 0), "canary_pass": bool((pg258_report.get("catastrophic_forgetting_canary") or {}).get("pass")), "promotion_blocked": bool((pg258_report.get("promotion") or {}).get("training_promotion_allowed") is False)}, "pg259": {"status": str(pg259_report.get("independent_final_judge", {}).get("decision", "not_run")), "selected_hidden_dim": int((pg259_report.get("selected") or {}).get("hidden_dim", 0) or 0), "fresh_route_rule_accuracy": float((((pg259_report.get("selected") or {}).get("metrics") or {}).get("fresh_route_holdout") or {}).get("rule_accuracy", 0.0) or 0.0), "fresh_route_family_accuracy": float((((pg259_report.get("selected") or {}).get("metrics") or {}).get("fresh_route_holdout") or {}).get("family_accuracy", 0.0) or 0.0), "fresh_route_belief_accuracy": float((((pg259_report.get("selected") or {}).get("metrics") or {}).get("fresh_route_holdout") or {}).get("belief_accuracy", 0.0) or 0.0), "fresh_route_probe_accuracy": float((((pg259_report.get("selected") or {}).get("metrics") or {}).get("fresh_route_holdout") or {}).get("probe_accuracy", 0.0) or 0.0), "implementation_ood_family_accuracy": float((((pg259_report.get("selected") or {}).get("metrics") or {}).get("implementation_ood") or {}).get("family_accuracy", 0.0) or 0.0), "record_count": int((pg259_report.get("counts") or {}).get("records", 0) or 0), "canary_pass": bool((pg259_report.get("catastrophic_forgetting_canary") or {}).get("pass")), "promotion_blocked": bool((pg259_report.get("promotion") or {}).get("training_promotion_allowed") is False)}}, "limits": ["不支持公网目标", "timing channel 未启用", "PG-256 是只读结果差分，SQL AST evaluator 仍在建设", "PG-257 只有 16 条本地记录，不能代表公网或跨实现能力", "PG-258 留出 rule=61%、family=65%、VulnerableApp OOD family=0%，所以晋级冻结", "PG-259 fresh 路由 Rule-IR=50%、VulnerableApp OOD family=14%，belief/probe=100% 但整体晋级冻结", "发送探针或行差分都不自动等于漏洞确认"], "next": "PG-260：增加 fresh seed 与 SQL/XSS/boolean/widebyte 各类配对样本，先提高 Rule-IR family 留出，再复跑 active-belief。"},
        "instructions": {"collector": "采集路由、GET/POST 字段、baseline/control、AI candidate、reference、negative 和失败原因；缺字段就 quarantine。", "reviewer": "检查 fresh reset、source hash、reference agreement、negative clean 与 evidence hash；只把 typed effect 标成本地确认。", "trainer": "按 gold/hard-negative/silver/quarantine 分层；不把原始 payload/response 写入训练集，不以 next-token loss 单独晋级。"},
    }
    pg257_metrics = dict((pg257_report.get("selected") or {}).get("metrics") or {})
    snapshot["capability"]["model"]["pg257"]["seed_holdout_next_token_accuracy"] = float((pg257_metrics.get("seed_holdout") or {}).get("next_token_accuracy", 0.0) or 0.0)
    # PG-260 is loaded as a bounded, report-backed card.  An absent report is
    # shown as pending; it never gets converted into a success by fallback
    # values.  This lets the UI update automatically when the GPU run writes
    # its final report without leaking raw traces or oracle inputs.
    pg260_selected = dict(pg260_report.get("selected") or {})
    pg260_metrics = dict(pg260_selected.get("metrics") or {})
    pg260_fresh = dict(pg260_metrics.get("fresh_route_holdout") or {})
    pg260_judge = dict(pg260_report.get("independent_final_judge") or {})
    pg260_done = bool(pg260_selected and pg260_report.get("status"))
    pg260_rule = float(pg260_fresh.get("rule_accuracy", 0.0) or 0.0)
    pg260_family = float(pg260_fresh.get("family_accuracy", 0.0) or 0.0)
    pg260_unknown = float(pg260_fresh.get("unknown_abstain_accuracy", 0.0) or 0.0)
    snapshot["source_reports"].append({"name": pg260_name, "updated_at": _report_time(pg260_name), "sha256": str(pg260_report.get("report_sha256", ""))})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg260", "label": "PG-260 paired active-belief", "value": f"{pg260_rule * 100:.0f}%" if pg260_done else "PENDING", "status": "pass" if bool(pg260_judge.get("pass")) else ("blocked" if pg260_done else "partial"), "note": "fresh paired SQL/XSS/boolean/widebyte；unknown-family abstain 独立计分"})
    snapshot["capability"]["model"]["pg260"] = {"status": str(pg260_report.get("status", "not_run")), "selected_hidden_dim": int(pg260_selected.get("hidden_dim", 0) or 0), "adapter_parameter_count": int(pg260_selected.get("adapter_parameter_count", 0) or 0), "fresh_route_rule_accuracy": pg260_rule, "fresh_route_family_accuracy": pg260_family, "fresh_route_unknown_abstain_accuracy": pg260_unknown, "implementation_ood_family_accuracy": float(dict(pg260_metrics.get("implementation_ood") or {}).get("family_accuracy", 0.0) or 0.0), "record_count": int((pg260_report.get("counts") or {}).get("records", 0) or 0), "canary_pass": bool((pg260_report.get("catastrophic_forgetting_canary") or {}).get("pass")), "judge_pass": bool(pg260_judge.get("pass")), "promotion_blocked": bool((pg260_report.get("promotion") or {}).get("training_promotion_allowed") is False)}
    snapshot["tasks"]["trainer"].append({"id": "pg260-active-belief", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked", "label": "PG-260 paired active-belief 容量训练", "route": "PG-260 fresh paired route/seed + unknown abstain", "seed": 26001, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": list(pg260_judge.get("reasons") or (["training report unavailable"] if not pg260_done else ["training promotion is intentionally disabled"])), "evidence_hash": str(pg260_report.get("report_sha256", ""))[:16], "instruction": "查看 2048/4096/8192 容量对比、fresh route Rule-IR/family、unknown-family abstain、OOD 与遗忘 canary；未通过前不晋级。", "raw_material_available": False})
    if pg260_done:
        snapshot["capability"]["limits"].append(f"PG-260 fresh route Rule-IR={pg260_rule * 100:.0f}%、family={pg260_family * 100:.0f}%、unknown-abstain={pg260_unknown * 100:.0f}%；最终晋级={bool(pg260_judge.get('pass'))}")
        snapshot["capability"]["next"] = "PG-261：mask-aware pooling 修复 padding 表示漂移，复跑同一容量矩阵。"
    pg261_running = _training_marker_active("pg261_training_running.json", pg261_name)
    # Do not leak the previous completed run while a fresh GPU run is writing
    # the same report path.  Once the report mtime advances beyond the marker,
    # the complete metrics become visible again.
    pg261_selected = {} if pg261_running else dict(pg261_report.get("selected") or {})
    pg261_metrics = dict(pg261_selected.get("metrics") or {})
    pg261_fresh = dict(pg261_metrics.get("fresh_route_holdout") or {})
    pg261_judge = dict(pg261_report.get("independent_final_judge") or {})
    pg261_done = bool(pg261_selected and pg261_report.get("status")) and not pg261_running
    pg261_rule = float(pg261_fresh.get("rule_accuracy", 0.0) or 0.0)
    pg261_family = float(pg261_fresh.get("family_accuracy", 0.0) or 0.0)
    pg261_unknown = float(pg261_fresh.get("unknown_abstain_accuracy", 0.0) or 0.0)
    snapshot["source_reports"].append({"name": pg261_name, "updated_at": _report_time(pg261_name), "sha256": str(pg261_report.get("report_sha256", ""))})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg261", "label": "PG-261 mask-aware capacity", "value": f"{pg261_rule * 100:.0f}%" if pg261_done else "RUNNING", "status": "pass" if bool(pg261_judge.get("pass")) else ("blocked" if pg261_done else "partial"), "note": "mask-aware pooling；检查 batch padding 不变性与实现 OOD"})
    snapshot["capability"]["model"]["pg261"] = {"status": "training_running" if pg261_running else str(pg261_report.get("status", "not_run")), "selected_hidden_dim": int(pg261_selected.get("hidden_dim", 0) or 0), "adapter_parameter_count": int(pg261_selected.get("adapter_parameter_count", 0) or 0), "fresh_route_rule_accuracy": pg261_rule, "fresh_route_family_accuracy": pg261_family, "fresh_route_unknown_abstain_accuracy": pg261_unknown, "implementation_ood_family_accuracy": float(dict(pg261_metrics.get("implementation_ood") or {}).get("family_accuracy", 0.0) or 0.0), "record_count": int((pg261_report.get("counts") or {}).get("records", 0) or 0), "canary_pass": bool((pg261_report.get("catastrophic_forgetting_canary") or {}).get("pass")), "judge_pass": bool(pg261_judge.get("pass")) if not pg261_running else False, "promotion_blocked": True}
    snapshot["tasks"]["trainer"].append({"id": "pg261-masked-active-belief", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "training_running" if not pg261_done else "promotion_blocked", "label": "PG-261 mask-aware pooling 容量复跑", "route": "PG-260 paired traces + padding-invariance adapter", "seed": 26101, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": (["GPU training still running; previous report is hidden until the new report is finalized"] if pg261_running else list(pg261_judge.get("reasons") or (["training promotion is intentionally disabled"] if not pg261_done else ["training promotion is intentionally disabled"]))), "evidence_hash": "" if pg261_running else str(pg261_report.get("report_sha256", ""))[:16], "instruction": "等待 2048/4096/8192 mask-aware sweep；完成后核对 split-padding invariance、OOD 与遗忘 canary，未过硬门不晋级。", "raw_material_available": False})
    if pg261_done:
        snapshot["capability"]["limits"].append(f"PG-261 fresh route Rule-IR={pg261_rule * 100:.0f}%、family={pg261_family * 100:.0f}%、OOD family={float(dict(pg261_metrics.get('implementation_ood') or {}).get('family_accuracy', 0.0) or 0.0) * 100:.0f}%；最终晋级={bool(pg261_judge.get('pass'))}")
        snapshot["capability"]["next"] = "PG-262：若 OOD 通过仍需补 route/seed Rule-IR 失败类别；若未通过，按混淆矩阵扩充跨实现样本。"
    pg262_running = _training_marker_active("pg262_training_running.json", pg262_name)
    pg262_done = bool(pg262_report.get("status")) and not pg262_running
    pg262_judge = dict(pg262_report.get("independent_final_judge") or {})
    snapshot["source_reports"].append({"name": pg262_name, "updated_at": _report_time(pg262_name), "sha256": str(pg262_report.get("report_sha256", ""))})
    snapshot["tasks"]["collector"].append({"id": "pg262-targeted-paired-traces", "role": "collector", "owner": "AI → 人工", "human_required": True, "status": "training_running" if pg262_running else ("collection_complete_review" if pg262_done else "ready_to_collect"), "label": "PG-262 失败导向 fresh GET/POST 轨迹", "route": "Pikachu 20 new route/seed pairs", "seed": 26201, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": (["collector running on loopback Pikachu; final report not written"] if pg262_running else list(pg262_judge.get("reasons") or (["collection report complete; await PG-263 judge"] if pg262_done else ["ready after PG-261 audit"]))), "evidence_hash": "" if pg262_running else str(pg262_report.get("report_sha256", ""))[:16], "instruction": "只采新 seed/route；每条必须有 AI candidate、reference、negative、typed oracle、fresh reset 和证据哈希。采集完成前不进入训练。", "raw_material_available": False})
    if pg262_running:
        snapshot["capability"]["next"] = "PG-262 正在本地 Pikachu 采集 20 条失败导向 GET/POST paired traces；报告完成后先复核完整性，再进入 PG-263 训练判官。"
    elif pg262_done:
        snapshot["capability"]["next"] = "PG-263：将 PG-262 新鲜 paired traces 与 PG-261 mask-aware 数据合并，复跑容量/route/implementation OOD。"
    pg263_running = _training_marker_active("pg263_training_running.json", pg263_name)
    pg263_selected = {} if pg263_running else dict(pg263_report.get("selected") or {})
    pg263_judge = dict(pg263_report.get("independent_final_judge") or {})
    pg263_done = bool(pg263_selected and pg263_report.get("status")) and not pg263_running
    snapshot["source_reports"].append({"name": pg263_name, "updated_at": _report_time(pg263_name), "sha256": str(pg263_report.get("report_sha256", ""))})
    snapshot["tasks"]["trainer"].append({"id": "pg263-pg262-augmented-training", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "training_running" if pg263_running else ("promotion_blocked" if pg263_done else "ready_to_train"), "label": "PG-263 PG-262 增量 mask-aware 容量训练", "route": "PG-261 base + PG-262 fresh even-seed holdout", "seed": 26301, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": (["GPU training running; final report not written"] if pg263_running else list(pg263_judge.get("reasons") or (["training promotion is intentionally disabled"] if pg263_done else ["awaiting GPU start"]))), "evidence_hash": "" if pg263_running else str(pg263_report.get("report_sha256", ""))[:16], "instruction": "检查 PG-262 audit 完整性、2048/4096/8192 容量、PG-262 fresh holdout、实现 OOD 与遗忘 canary；所有硬门通过前不晋级。", "raw_material_available": False})
    if pg263_running:
        snapshot["capability"]["next"] = "PG-263 正在训练；等待 2048/4096/8192 及 PG-262 fresh holdout 报告后再做独立审计。"
    elif pg263_done:
        snapshot["capability"]["next"] = "PG-264：若 PG-263 仍有失败类别，继续按新混淆矩阵采集；通过后才考虑跨靶场。"
    pg264_running = _training_marker_active("pg264_collection_running.json", pg264_name)
    pg264_done = bool(pg264_report.get("status")) and not pg264_running
    pg264_audit_complete = bool(pg264_audit.get("all_required_fields_complete")) and not pg264_running
    pg264_audit_complete = pg264_audit_complete and int(pg264_audit.get("audited_record_count", 0) or 0) == 32
    pg264_counts = dict(pg264_report.get("counts") or {})
    pg264_sources = dict(pg264_counts.get("source_counts") or {})
    pg264_family_counts = dict(pg264_audit.get("family_counts") or {})
    snapshot["source_reports"].append({"name": pg264_name, "updated_at": _report_time(pg264_name), "sha256": str(pg264_report.get("report_sha256", ""))})
    snapshot["tasks"]["collector"].append({"id": "pg264-pikachu-growth-collection", "role": "collector", "owner": "AI → 人工", "human_required": True, "status": "collecting" if pg264_running else ("collection_complete_review" if pg264_done else "ready_to_collect"), "label": "PG-264 Pikachu 数据增长采集", "route": "32 fresh SQL/XSS/boolean/widebyte seed cells", "seed": 26401, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": (["fresh local collection running; audit not written"] if pg264_running else (["collection complete; independent audit required"] if not pg264_audit_complete else ["audit complete; training remains separately gated"])), "evidence_hash": "" if pg264_running else str(pg264_audit.get("audit_sha256") or pg264_report.get("report_sha256", ""))[:16], "instruction": "等待 32 个全新 seed 单元完整回放；逐条核对 AI/reference/negative、fresh reset、typed oracle 和证据哈希，缺字段保持 quarantine。", "raw_material_available": False})
    if pg264_running:
        snapshot["capability"]["next"] = "PG-264 正在本地后台采集 32 个 Pikachu fresh seed；完成后先跑独立完整性审计，再决定 PG-265 大容量训练。"
    elif pg264_done and not pg264_audit_complete:
        snapshot["capability"]["next"] = "PG-264 采集报告已写入，但独立审计未通过/未完成；不得训练，先修复缺失字段。"
    elif pg264_audit_complete:
        snapshot["capability"]["next"] = "PG-265：把 PG-264 审计通过的抽象 token 加入训练，比较 8192 与更大 adapter；晋级仍关闭。"
    # A remote run is deliberately represented by a small local status
    # contract.  It lets the ops UI show the authorized A800 job while the
    # report/artifacts remain remote; the final report is still the only
    # source of metrics and must be copied back and independently audited.
    pg265_remote_running = str(pg265_remote.get("status", "")) == "running"
    pg265_running = _training_marker_active("pg265_training_running.json", pg265_name) or pg265_remote_running
    pg265_done = bool(pg265_report.get("status")) and not pg265_running
    pg265_stopped = str(pg265_stop.get("status", "")) == "stopped_user_request"
    pg265_judge = dict(pg265_report.get("independent_final_judge") or {})
    if pg265_remote_running:
        pg265_reasons = ["remote A800 GPU0 running; final report pending"]
    elif pg265_running:
        pg265_reasons = ["12288 capacity sweep running"]
    elif pg265_stopped and not pg265_done:
        pg265_reasons = ["user requested stop", "only one local CUDA device detected", "partial weights are not promotable"]
    elif pg265_done:
        pg265_reasons = list(pg265_judge.get("reasons") or ["independent audit complete; promotion remains intentionally disabled"])
    else:
        pg265_reasons = ["PG-264 audit required before training"]
    snapshot["source_reports"].append({"name": pg265_name, "updated_at": _report_time(pg265_name), "sha256": str(pg265_report.get("report_sha256", ""))})
    snapshot["tasks"]["trainer"].append({"id": "pg265-growth-augmented-large-capacity", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "training_running" if pg265_running else ("stopped_waiting_external_device" if pg265_stopped and not pg265_done else ("promotion_blocked" if pg265_done else "waiting_pg264_audit")), "label": "PG-265 增量大容量训练", "route": "PG-263 base + PG-264 audited growth", "seed": 26501, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": pg265_reasons, "evidence_hash": "" if pg265_running else str(pg265_report.get("report_sha256", ""))[:16], "instruction": "仅在 PG-264 32 条 fresh seed 审计通过后比较 4096/8192/12288；核对族外、遗忘 canary 和报告哈希，任何通过都不自动晋级。", "raw_material_available": False})
    if pg265_remote_running:
        snapshot["capability"]["next"] = "PG-265 正在授权远端 A800 的 GPU0 顺序比较 4096/8192/12288；其他 GPU 未触碰，等待报告回传后独立审计。"
    elif pg265_running:
        snapshot["capability"]["next"] = "PG-265 正在单卡顺序比较 4096/8192/12288；等待独立报告审计。"
    elif pg265_stopped and not pg265_done:
        snapshot["capability"]["next"] = "PG-265 已按要求停止；本机只检测到 RTX 3060，等待配置其他授权设备后从 PG-264 审计点重新开始。"
    elif pg265_done:
        snapshot["capability"]["next"] = "PG-265 已完成；根据独立审计和族外混淆矩阵决定是否补采或接入第二个本地靶场。"
    # Expose the two newest stages as bounded capability facts as well as
    # task cards.  The frontend should never make the operator open a raw
    # report to tell whether collection is complete or the GPU run is alive.
    pg262_counts = dict(pg262_report.get("counts") or {})
    pg262_audit = dict(pg262_report.get("collection_audit") or {})
    pg262_source_counts = dict(pg262_counts.get("source_counts") or {})
    snapshot["capability"]["model"]["pg262"] = {
        "status": "collection_complete_review" if pg262_done else ("collecting" if pg262_running else "not_run"),
        "record_count": int(pg262_counts.get("records", 0) or 0),
        "sql_count": int(pg262_source_counts.get("pg262_pikachu_sql_paired", 0) or 0),
        "xss_count": int(pg262_source_counts.get("pg262_pikachu_xss_paired", 0) or 0),
        "audit_complete": bool(pg262_audit.get("all_required_fields_complete")),
        "training_eligible": bool(pg262_audit.get("training_eligible")),
        "evidence_hash": str(pg262_audit.get("audit_sha256") or pg262_report.get("report_sha256") or "")[:16],
    }
    pg263_metrics = dict((pg263_selected.get("metrics") or {}))
    pg263_fresh = dict(pg263_metrics.get("fresh_route_holdout") or {})
    pg263_ood = dict(pg263_metrics.get("implementation_ood") or {})
    snapshot["capability"]["model"]["pg263"] = {
        "status": "training_running" if pg263_running else (str(pg263_report.get("status", "not_run")) if pg263_done else "not_run"),
        "selected_hidden_dim": int(pg263_selected.get("hidden_dim", 0) or 0),
        "adapter_parameter_count": int(pg263_selected.get("adapter_parameter_count", 0) or 0),
        "record_count": int((pg263_report.get("counts") or {}).get("records", 0) or 0),
        "fresh_route_rule_accuracy": float(pg263_fresh.get("rule_accuracy", 0.0) or 0.0),
        "fresh_route_family_accuracy": float(pg263_fresh.get("family_accuracy", 0.0) or 0.0),
        "implementation_ood_family_accuracy": float(pg263_ood.get("family_accuracy", 0.0) or 0.0),
        "canary_pass": bool((pg263_report.get("catastrophic_forgetting_canary") or {}).get("pass")),
        "judge_pass": bool(pg263_judge.get("pass")) if not pg263_running else False,
        "promotion_blocked": True,
        "evidence_hash": "" if pg263_running else str(pg263_report.get("report_sha256", ""))[:16],
        "resource_profile": dict(pg263_report.get("resource_profile") or {}),
    }
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg262",
        "label": "PG-262 fresh paired traces",
        "value": f"{int(pg262_counts.get('records', 0) or 0)} 条" if pg262_done else ("RUNNING" if pg262_running else "PENDING"),
        "status": "partial" if pg262_done else "blocked",
        "note": "Pikachu GET/POST；完整性通过但训练资格保持关闭",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg263",
        "label": "PG-263 augmented capacity",
        "value": f"{float(pg263_fresh.get('rule_accuracy', 0.0) or 0.0) * 100:.0f}%" if pg263_done else ("RUNNING" if pg263_running else "PENDING"),
        "status": "pass" if bool(pg263_judge.get("pass")) else ("blocked" if pg263_done else "partial"),
        "note": "PG-262 fresh holdout + 2048/4096/8192；独立审计后才可解释",
    })
    snapshot["capability"]["model"]["pg264"] = {
        "status": "collecting" if pg264_running else ("collection_complete_review" if pg264_done else "not_run"),
        "record_count": int(pg264_counts.get("records", 0) or 0),
        "audit_record_count": int(pg264_audit.get("audited_record_count", 0) or 0),
        "sql_count": int(pg264_family_counts.get("sql", 0) or pg264_sources.get("pg264_pikachu_sql_paired", 0) or 0),
        "xss_count": int(pg264_family_counts.get("xss", 0) or pg264_sources.get("pg264_pikachu_xss_paired", 0) or 0),
        "boolean_count": int(pg264_family_counts.get("boolean", 0) or pg264_sources.get("pg264_pikachu_boolean_paired", 0) or 0),
        "widebyte_count": int(pg264_family_counts.get("widebyte", 0) or pg264_sources.get("pg264_pikachu_widebyte_paired", 0) or 0),
        "audit_complete": pg264_audit_complete,
        "training_eligible": False,
        "evidence_hash": "" if pg264_running else str(pg264_audit.get("audit_sha256") or pg264_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg264",
        "label": "PG-264 fresh data growth",
        "value": f"{int(pg264_counts.get('records', 0) or 0)} 条" if pg264_done else ("RUNNING" if pg264_running else "PENDING"),
        "status": "pass" if pg264_audit_complete else ("partial" if pg264_done else "blocked"),
        "note": "32 个新 seed；完整性审计通过前不进入 PG-265 训练",
    })
    pg265_selected = dict(pg265_report.get("selected") or {})
    pg265_metrics = dict(pg265_selected.get("metrics") or {})
    pg265_fresh = dict(pg265_metrics.get("fresh_route_holdout") or {})
    pg265_ood = dict(pg265_metrics.get("implementation_ood") or {})
    snapshot["capability"]["model"]["pg265"] = {
        "status": "remote_training_running" if pg265_remote_running else ("training_running" if pg265_running else (str(pg265_report.get("status", "not_run")) if pg265_done else ("stopped_waiting_external_device" if pg265_stopped else "waiting_pg264_audit"))),
        "selected_hidden_dim": int(pg265_selected.get("hidden_dim", 0) or 0),
        "adapter_parameter_count": int(pg265_selected.get("adapter_parameter_count", 0) or 0),
        "capacity_variants": [int(item.get("hidden_dim", 0) or 0) for item in list(pg265_report.get("capacity_variant_metrics") or []) if isinstance(item, dict)],
        "train_steps": int(pg265_report.get("train_steps", 0) or 0),
        "micro_batch_size": int((pg265_report.get("resource_profile") or {}).get("micro_batch_size", 0) or 0),
        "elapsed_seconds_to_report": float(pg265_remote.get("elapsed_seconds_to_report", 0.0) or 0.0),
        "device": str(pg265_remote.get("device", {}).get("gpu_model", "")) + (" GPU" + str(pg265_remote.get("device", {}).get("gpu_visible_to_process", "")) if pg265_remote.get("device") else ""),
        "record_count": int((pg265_report.get("counts") or {}).get("records", 0) or 0),
        "fresh_route_rule_accuracy": float(pg265_fresh.get("rule_accuracy", 0.0) or 0.0),
        "fresh_route_family_accuracy": float(pg265_fresh.get("family_accuracy", 0.0) or 0.0),
        "implementation_ood_family_accuracy": float(pg265_ood.get("family_accuracy", 0.0) or 0.0),
        "judge_pass": bool(pg265_judge.get("pass")) if not pg265_running else False,
        "promotion_blocked": True,
        "audit_pass": bool(pg265_audit.get("pass")) if pg265_done else False,
        "evidence_hash": "" if pg265_running else str(pg265_report.get("report_sha256", ""))[:16],
        "stop_checkpoint": str(pg265_stop.get("status", "")) if pg265_stopped and not pg265_done else "",
    }
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg265",
        "label": "PG-265 large adapter",
        "value": f"{float(pg265_fresh.get('rule_accuracy', 0.0) or 0.0) * 100:.0f}%" if pg265_done else ("RUNNING" if pg265_running else ("STOPPED" if pg265_stopped else "WAITING")),
        "status": "pass" if bool(pg265_audit.get("pass")) else ("blocked" if pg265_done or pg265_stopped else "partial"),
        "note": "4096/8192/12288；PG-264 审计与独立判官均完成后才可解释",
    })
    pg266_done = str(pg266_report.get("status", "")) == "completed_local_payload_grounding_replay"
    pg266_counts = dict(pg266_report.get("counts") or pg266_catalog.get("counts") or {})
    pg266_judge = bool(pg266_counts.get("false_positive_count", 1) == 0 and pg266_counts.get("fresh_reset_count", 0) == pg266_counts.get("route_count", 0))
    snapshot["source_reports"].append({"name": pg266_name, "updated_at": _report_time(pg266_name), "sha256": str(pg266_report.get("report_sha256", ""))})
    pg266_task = {"id": "pg266-payload-grounding-replay", "role": "reviewer", "owner": "AI → 人工", "human_required": True, "status": "review_ready" if pg266_done else "ready_to_collect", "label": "PG-266 Pikachu 实际 payload 回放", "route": f"{int(pg266_counts.get('route_count', 0) or 0)} 个 fresh GET/POST 路由", "seed": 26601, "method": "GET/POST", "typed_effect": bool(pg266_counts.get("confirmed_positive_count", 0)), "confirmed_positive": bool(pg266_judge), "reasons": ([f"本地 confirmed effect {int(pg266_counts.get('confirmed_positive_count', 0) or 0)} 条；误报 {int(pg266_counts.get('false_positive_count', 0) or 0)} 条"] if pg266_done else ["等待 PG-266 fresh replay report"]), "evidence_hash": str(pg266_report.get("report_sha256", ""))[:16], "instruction": "逐条查看 AI/reference/negative 的真实 wire、有限 echo、DOM/Location oracle 和证据哈希；只把摘要 token 送训练，raw payload 只能留在本地人工审查目录。", "raw_material_available": False, "human_catalog_available": bool(pg266_done)}
    snapshot["tasks"]["reviewer"].append(pg266_task)
    snapshot["tasks"]["all"].append(pg266_task)
    snapshot["capability"]["model"]["pg266"] = {"status": str(pg266_report.get("status", "not_run")), "route_count": int(pg266_counts.get("route_count", 0) or 0), "get_count": int(pg266_counts.get("get_count", 0) or 0), "post_count": int(pg266_counts.get("post_count", 0) or 0), "confirmed_local_effect_count": int(pg266_counts.get("confirmed_positive_count", 0) or 0), "false_positive_count": int(pg266_counts.get("false_positive_count", 0) or 0), "abstain_count": int(pg266_counts.get("abstain_count", 0) or 0), "fresh_reset_count": int(pg266_counts.get("fresh_reset_count", 0) or 0), "elapsed_seconds": float(pg266_counts.get("elapsed_seconds", 0.0) or 0.0), "judge_pass": pg266_judge, "training_promotion_blocked": True, "raw_payloads_human_review_only": True}
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg266", "label": "PG-266 grounded payload replay", "value": f"{int(pg266_counts.get('confirmed_positive_count', 0) or 0)}/{int(pg266_counts.get('route_count', 0) or 0)}" if pg266_done else "PENDING", "status": "pass" if pg266_judge else ("blocked" if pg266_done else "partial"), "note": "真实 GET/POST wire + 有限回显 + DOM/Location oracle；仅 loopback 人工复核"})
    if pg266_done:
        snapshot["capability"]["next"] = "PG-267：把 PG-266 的抽象 payload-grounding token 加入更大 decoder 训练，并在第二个全新本地靶场复放；raw payload 继续不进模型输入。"
    pg267_selected = dict(pg267_report.get("selected") or {})
    pg267_metrics = dict(pg267_selected.get("metrics") or {})
    pg267_fresh = dict(pg267_metrics.get("fresh_route_holdout") or {})
    pg267_holdout = dict(pg267_metrics.get("route_seed_holdout") or {})
    pg267_ood = dict(pg267_metrics.get("implementation_ood") or {})
    pg267_judge = dict(pg267_report.get("independent_final_judge") or {})
    pg267_done = bool(pg267_selected and pg267_report.get("status"))
    pg267_audit_pass = bool(pg267_audit.get("all_required_fields_complete"))
    snapshot["source_reports"].append({"name": pg267_name, "updated_at": _report_time(pg267_name), "sha256": str(pg267_report.get("report_sha256", ""))})
    pg267_reasons = list(pg267_judge.get("reasons") or [])
    if pg267_done and pg267_audit_pass:
        pg267_reasons = pg267_reasons or ["structural audit passed; model gate remains blocked or candidate-only"]
    snapshot["tasks"]["trainer"].append({"id": "pg267-payload-grounding-capacity", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked" if pg267_done else "ready_to_train", "label": "PG-267 payload-grounding 抽象 token 容量训练", "route": "PG-265 base + PG-266 abstract 12 records; even seed holdout", "seed": 26701, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": pg267_reasons or ["training report unavailable"], "evidence_hash": str(pg267_audit.get("audit_sha256") or pg267_report.get("report_sha256", ""))[:16], "instruction": "查看 8192/12288/16384 容量、fresh payload-grounding Rule-IR、route-seed 留出、实现 OOD 与遗忘 canary；结构审计通过也不等于模型泛化通过。", "raw_material_available": False})
    snapshot["capability"]["model"]["pg267"] = {"status": str(pg267_report.get("status", "not_run")), "selected_hidden_dim": int(pg267_selected.get("hidden_dim", 0) or 0), "adapter_parameter_count": int(pg267_selected.get("adapter_parameter_count", 0) or 0), "capacity_variants": list(pg267_report.get("capacity_variants") or []), "train_steps": int(pg267_report.get("train_steps", 0) or 0), "micro_batch_size": int((pg267_report.get("resource_profile") or {}).get("micro_batch_size", 0) or 0), "record_count": int((pg267_report.get("growth_counts") or {}).get("combined_records", 0) or (pg267_report.get("counts") or {}).get("records", 0) or 0), "pg267_record_count": int((pg267_report.get("growth_counts") or {}).get("pg267_records", 0) or 0), "fresh_holdout_rule_accuracy": float(pg267_fresh.get("rule_accuracy", 0.0) or 0.0), "fresh_holdout_family_accuracy": float(pg267_fresh.get("family_accuracy", 0.0) or 0.0), "route_seed_rule_accuracy": float(pg267_holdout.get("rule_accuracy", 0.0) or 0.0), "implementation_ood_family_accuracy": float(pg267_ood.get("family_accuracy", 0.0) or 0.0), "judge_pass": bool(pg267_judge.get("pass")), "structural_audit_pass": pg267_audit_pass, "canary_pass": bool((pg267_report.get("catastrophic_forgetting_canary") or {}).get("pass")), "promotion_blocked": True, "payload_strings_in_model_input": False, "oracle_target_in_model_input": False, "evidence_hash": str(pg267_audit.get("audit_sha256") or pg267_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg267", "label": "PG-267 grounded token adapter", "value": f"{float(pg267_fresh.get('rule_accuracy', 0.0) or 0.0) * 100:.0f}%" if pg267_done else "PENDING", "status": "pass" if bool(pg267_judge.get("pass")) else ("blocked" if pg267_done else "partial"), "note": "fresh Rule-IR 仅作候选；route-seed 门与结构审计分开显示"})
    if pg267_done:
        snapshot["capability"]["next"] = "PG-267 route-seed Rule-IR 未过 0.80；先补足未见表面/编码配对，再考虑第二靶场复放，权重与长期记忆均冻结。"
    pg268_counts = dict(pg268_report.get("counts") or pg268_catalog.get("counts") or {})
    pg268_done = str(pg268_report.get("status", "")) == "completed_local_parameterized_get_post_replay"
    pg268_audit_pass = bool(pg268_audit.get("all_required_fields_complete"))
    pg268_complete = int((pg268_audit.get("complete_replayed_surface_count") or 0) or 0)
    pg268_unsupported = int((pg268_audit.get("unsupported_surface_count") or 0) or 0)
    snapshot["source_reports"].append({"name": pg268_name, "updated_at": _report_time(pg268_name), "sha256": str(pg268_report.get("report_sha256", ""))})
    pg268_reasons = [
        f"{int(pg268_counts.get('confirmed_positive_count', 0) or 0)} local typed effects; {int(pg268_counts.get('abstain_count', 0) or 0)} abstain; false positives {int(pg268_counts.get('false_positive_count', 0) or 0)}",
        f"{pg268_complete} complete replay rows; {pg268_unsupported} multipart rows remain incomplete",
        "audit passed; training and memory promotion remain blocked",
    ] if pg268_done else ["waiting for PG-268B replay report"]
    pg268_task = {"id": "pg268-parameterized-replay", "role": "reviewer", "owner": "AI → 人工", "human_required": True, "status": "review_ready" if pg268_done and pg268_audit_pass else "ready_to_replay", "label": "PG-268B Pikachu 参数化 GET/POST 回放", "route": f"{int(pg268_counts.get('surface_count', 0) or 0)} surfaces · complete {pg268_complete}", "seed": 26802, "method": "GET/POST", "typed_effect": bool(pg268_counts.get("confirmed_positive_count", 0) or 0), "confirmed_positive": False, "reasons": pg268_reasons, "evidence_hash": str(pg268_audit.get("audit_sha256") or pg268_report.get("report_sha256", ""))[:16], "instruction": "先查看 AI/reference/negative wire、失败/abstain 原因、fresh reset、source/evidence hash；只有 typed oracle 同时满足才可形成抽象 Rule-IR，当前禁止训练晋级。", "raw_material_available": False, "human_catalog_available": bool(pg268_done)}
    snapshot["tasks"]["reviewer"].append(pg268_task)
    snapshot["tasks"]["all"].append(pg268_task)
    snapshot["capability"]["model"]["pg268"] = {"status": str(pg268_report.get("status", "not_run")), "surface_count": int(pg268_counts.get("surface_count", 0) or 0), "get_count": int(pg268_counts.get("get_count", 0) or 0), "post_count": int(pg268_counts.get("post_count", 0) or 0), "complete_replayed_surface_count": pg268_complete, "unsupported_multipart_count": pg268_unsupported, "ai_send_count": int(pg268_counts.get("ai_send_count", 0) or 0), "reference_send_count": int(pg268_counts.get("reference_send_count", 0) or 0), "negative_send_count": int(pg268_counts.get("negative_send_count", 0) or 0), "confirmed_local_effect_count": int(pg268_counts.get("confirmed_positive_count", 0) or 0), "abstain_count": int(pg268_counts.get("abstain_count", 0) or 0), "false_positive_count": int(pg268_counts.get("false_positive_count", 0) or 0), "fresh_reset_count": int(pg268_counts.get("fresh_reset_count", 0) or 0), "source_attested_count": int(pg268_counts.get("source_attested_count", 0) or 0), "audit_pass": pg268_audit_pass, "training_promotion_blocked": True, "memory_promotion_blocked": True, "raw_payloads_human_review_only": True, "oracle_target_in_model_input": False, "manifest_parameterized_surface_count": int((pg268_manifest.get("counts") or {}).get("with_parameter_context", 0) or 0), "evidence_hash": str(pg268_audit.get("audit_sha256") or pg268_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg268", "label": "PG-268B 参数化 GET/POST 回放", "value": f"{int(pg268_counts.get('confirmed_positive_count', 0) or 0)}/{int(pg268_counts.get('surface_count', 0) or 0)}" if pg268_done else "PENDING", "status": "pass" if pg268_done and pg268_audit_pass and int(pg268_counts.get("false_positive_count", 1) or 1) == 0 else ("blocked" if pg268_done else "partial"), "note": "40 条完整三通道复放；3 条 multipart incomplete；只显示 candidate，不宣称公网漏洞"})
    if pg268_done:
        snapshot["capability"]["limits"].append(f"PG-268B 参数化表面 {int(pg268_counts.get('surface_count', 0) or 0)} 条，完整复放 {pg268_complete} 条，typed effect {int(pg268_counts.get('confirmed_positive_count', 0) or 0)} 条，abstain {int(pg268_counts.get('abstain_count', 0) or 0)} 条；训练晋级冻结")
        snapshot["capability"]["next"] = "PG-269：用 PG-268B 完整抽象轨迹补多编码配对与失败修复，再训练独立 Rule-IR 解码器；先保持 payload decoder 与主模型分离。"
    pg269_counts = dict(pg269_report.get("counts") or pg269_catalog.get("counts") or {})
    pg269_done = str(pg269_report.get("status", "")) == "completed_local_failure_guided_replay"
    pg269_audit_pass = bool(pg269_audit.get("all_required_fields_complete"))
    pg269_task = {"id": "pg269-failure-guided-verification", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "review_ready" if pg269_done and pg269_audit_pass else "planned", "label": "PG-269 导师指导失败引导验证", "route": f"{int(pg269_counts.get('surface_count', 0) or 0)} multi-step surfaces · repair {int(pg269_counts.get('repair_attempt_count', 0) or 0)}", "seed": 26902, "method": "GET/POST", "typed_effect": bool(pg269_counts.get("final_confirmed_count", 0) or 0), "confirmed_positive": False, "reasons": ([f"初始确认 {int(pg269_counts.get('initial_confirmed_count', 0) or 0)} → 最终确认 {int(pg269_counts.get('final_confirmed_count', 0) or 0)}；repair {int(pg269_counts.get('repair_attempt_count', 0) or 0)}/{int(pg269_counts.get('repair_success_count', 0) or 0)}；abstain {int(pg269_counts.get('abstain_count', 0) or 0)}"] if pg269_done else ["PG-269 report unavailable"]), "evidence_hash": str(pg269_audit.get("audit_sha256") or pg269_report.get("report_sha256", ""))[:16], "instruction": "导师提供参考 action/Rule-IR/证据需求并给每步评分；模型按 observe → baseline/reference → candidate → failure diagnosis → repair/abstain → fresh replay 学习。context 与 target 分离，raw payload 只在人审 catalog。", "raw_material_available": False}
    snapshot["tasks"]["trainer"].append(pg269_task)
    snapshot["source_reports"].append({"name": pg269_name, "updated_at": _report_time(pg269_name), "sha256": str(pg269_report.get("report_sha256", ""))})
    snapshot["capability"]["model"]["pg269"] = {"status": str(pg269_report.get("status", "not_run")), "surface_count": int(pg269_counts.get("surface_count", 0) or 0), "get_count": int(pg269_counts.get("get_count", 0) or 0), "post_count": int(pg269_counts.get("post_count", 0) or 0), "complete_count": int(pg269_counts.get("complete_count", 0) or 0), "initial_confirmed_count": int(pg269_counts.get("initial_confirmed_count", 0) or 0), "final_confirmed_count": int(pg269_counts.get("final_confirmed_count", 0) or 0), "repair_attempt_count": int(pg269_counts.get("repair_attempt_count", 0) or 0), "repair_success_count": int(pg269_counts.get("repair_success_count", 0) or 0), "abstain_count": int(pg269_counts.get("abstain_count", 0) or 0), "false_positive_count": int(pg269_counts.get("false_positive_count", 0) or 0), "source_attested_count": int(pg269_counts.get("source_attested_count", 0) or 0), "audit_pass": pg269_audit_pass, "context_target_split": bool((pg269_report.get("safety") or {}).get("context_excludes_oracle")) and bool((pg269_catalog.get("contract") or {}).get("context_target_split")), "training_promotion_blocked": True, "memory_promotion_blocked": True, "evidence_hash": str(pg269_audit.get("audit_sha256") or pg269_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg269", "label": "PG-269 导师指导 repair", "value": f"{int(pg269_counts.get('final_confirmed_count', 0) or 0)}/{int(pg269_counts.get('surface_count', 0) or 0)}" if pg269_done else "PENDING", "status": "pass" if pg269_done and pg269_audit_pass and int(pg269_counts.get("false_positive_count", 1) or 1) == 0 else ("blocked" if pg269_done else "partial"), "note": "导师参考 + 失败诊断 + repair/abstain；先 SFT/preference，再 offline RL"})
    if pg269_done:
        snapshot["capability"]["limits"].append(f"PG-269 完整 {int(pg269_counts.get('complete_count', 0) or 0)} 条，repair {int(pg269_counts.get('repair_attempt_count', 0) or 0)}/{int(pg269_counts.get('repair_success_count', 0) or 0)}，最终确认 {int(pg269_counts.get('final_confirmed_count', 0) or 0)}，abstain {int(pg269_counts.get('abstain_count', 0) or 0)}；Teacher-SFT 尚未晋级")
        snapshot["capability"]["next"] = "PG-270：用 PG-269 context/target 轨迹生成导师参考答案、pairwise preference 和 process reward，先做 Teacher-SFT 小规模消融。"
    pg270_counts = dict(pg270_report.get("dataset", {}).get("counts") or pg270_dataset.get("counts") or {})
    pg270_done = str(pg270_report.get("status", "")) == "candidate_ablation_completed"
    pg270_audit_pass = bool(pg270_audit.get("all_required_fields_complete"))
    pg270_checks = dict(pg270_report.get("capability_gate", {}).get("checks") or {})
    pg270_family = pg270_report.get("evaluations", {}).get("guided_sft", {}).get("family_holdout", {})
    pg270_task = {"id": "pg270-teacher-sft-ablation", "role": "trainer", "owner": "AI → 人工", "human_required": True, "status": "review_ready" if pg270_done and pg270_audit_pass else "planned", "label": "PG-270 教师指导 SFT + preference/process reward", "route": f"train {int(pg270_counts.get('train', 0) or 0)} · route-dev {int(pg270_counts.get('route_dev', 0) or 0)} · family holdout {int(pg270_counts.get('family_holdout', 0) or 0)}", "seed": 27001, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": ([f"guided token {float(pg270_family.get('token_accuracy', 0.0) or 0.0) * 100:.1f}% vs plain {float(pg270_report.get('evaluations', {}).get('plain_sft', {}).get('family_holdout', {}).get('token_accuracy', 0.0) or 0.0) * 100:.1f}%；preference pair {int(pg270_counts.get('preference_pairs', 0) or 0)}；process reward {int(pg270_counts.get('process_reward_episodes', 0) or 0)}"] if pg270_done else ["PG-270 report unavailable"]), "evidence_hash": str(pg270_audit.get("audit_sha256") or pg270_report.get("report_sha256", ""))[:16], "instruction": "优先提出当前疑问，再组装抽象 action；用教师 target、pairwise preference 和逐步 reward 学习失败诊断、repair 或 calibrated abstain。A800 GPU0 训练；未见族 fresh replay 通过前不晋级。", "raw_material_available": False}
    snapshot["tasks"]["trainer"].append(pg270_task)
    snapshot["source_reports"].append({"name": pg270_name, "updated_at": _report_time(pg270_name), "sha256": str(pg270_report.get("report_sha256", ""))})
    snapshot["capability"]["model"]["pg270"] = {"status": str(pg270_report.get("status", "not_run")), "train_count": int(pg270_counts.get("train", 0) or 0), "route_dev_count": int(pg270_counts.get("route_dev", 0) or 0), "family_holdout_count": int(pg270_counts.get("family_holdout", 0) or 0), "preference_pair_count": int(pg270_counts.get("preference_pairs", 0) or 0), "process_reward_count": int(pg270_counts.get("process_reward_episodes", 0) or 0), "guided_token_accuracy": float(pg270_family.get("token_accuracy", 0.0) or 0.0), "guided_next_action_accuracy": float(pg270_family.get("next_action_accuracy", 0.0) or 0.0), "guided_preference_win_rate": float(pg270_report.get("evaluations", {}).get("guided_sft", {}).get("family_holdout_preference", {}).get("preference_win_rate", 0.0) or 0.0), "audit_pass": pg270_audit_pass, "cuda_assignment": dict(pg270_report.get("source", {}).get("cuda_assignment") or {}), "training_promotion_blocked": True, "memory_promotion_blocked": True, "evidence_hash": str(pg270_audit.get("audit_sha256") or pg270_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg270", "label": "PG-270 教师指导消融", "value": f"{float(pg270_family.get('token_accuracy', 0.0) or 0.0) * 100:.1f}%" if pg270_done else "PENDING", "status": "pass" if pg270_done and pg270_audit_pass and all(bool(value) for value in pg270_checks.values()) else ("blocked" if pg270_done else "partial"), "note": "guided SFT + preference/process reward；A800 GPU0；未见族仅作候选能力证据"})
    if pg270_done:
        snapshot["capability"]["limits"].append(f"PG-270 40 条抽象轨迹，family holdout token={float(pg270_family.get('token_accuracy', 0.0) or 0.0) * 100:.1f}%、next-action={float(pg270_family.get('next_action_accuracy', 0.0) or 0.0) * 100:.1f}%；exact trajectory 仍为 {float(pg270_family.get('exact_trajectory_rate', 0.0) or 0.0) * 100:.1f}%，不代表漏洞检测")
        snapshot["capability"]["next"] = "PG-271：独立 seed + fresh Pikachu 回放，验证疑问→组装→失败更新→repair/abstain 的稳定性；通过后才考虑 offline RL。"
    pg271_evaluations = dict(pg271_report.get("evaluations") or {})
    pg271_all = dict(pg271_evaluations.get("fresh_seed_all") or {})
    pg271_family = dict(pg271_evaluations.get("fresh_seed_family_holdout") or {})
    pg271_done = str(pg271_report.get("status", "")) == "candidate_replay_completed"
    pg271_audit_pass = bool(pg271_audit.get("all_required_fields_complete"))
    pg271_task = {"id": "pg271-fresh-seed-candidate-replay", "role": "reviewer", "owner": "AI → 人工", "human_required": True, "status": "review_ready" if pg271_done and pg271_audit_pass else "planned", "label": "PG-271 独立 seed + fresh candidate 回放", "route": f"seed {pg271_report.get('source', {}).get('fresh_seed', '—')} · {int(pg271_all.get('count', 0) or 0)} surfaces", "seed": int(pg271_report.get("source", {}).get("fresh_seed", 27102) or 27102), "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": ([f"family holdout next-action {float(pg271_family.get('next_action_accuracy', 0.0) or 0.0) * 100:.0f}% · belief {float(pg271_family.get('final_belief_accuracy', 0.0) or 0.0) * 100:.0f}% · unsupported positive {int(pg271_family.get('unsupported_positive_count', 0) or 0)}"] if pg271_done else ["PG-271 report unavailable"]), "evidence_hash": str(pg271_audit.get("audit_sha256") or pg271_report.get("report_sha256", ""))[:16], "instruction": "把 PG-270 guided checkpoint 放到新 seed 的抽象 context 上；只记录 next-action/belief/abstain 候选，fresh typed oracle 仍是最终判官。", "raw_material_available": False}
    snapshot["tasks"]["reviewer"].append(pg271_task)
    snapshot["source_reports"].append({"name": pg271_name, "updated_at": _report_time(pg271_name), "sha256": str(pg271_report.get("report_sha256", ""))})
    snapshot["capability"]["model"]["pg271"] = {"status": str(pg271_report.get("status", "not_run")), "fresh_seed": int(pg271_report.get("source", {}).get("fresh_seed", 0) or 0), "surface_count": int(pg271_all.get("count", 0) or 0), "next_action_accuracy": float(pg271_all.get("next_action_accuracy", 0.0) or 0.0), "final_belief_accuracy": float(pg271_all.get("final_belief_accuracy", 0.0) or 0.0), "abstain_calibration_accuracy": float(pg271_all.get("abstain_calibration_accuracy", 0.0) or 0.0), "family_holdout_count": int(pg271_family.get("count", 0) or 0), "family_next_action_accuracy": float(pg271_family.get("next_action_accuracy", 0.0) or 0.0), "family_final_belief_accuracy": float(pg271_family.get("final_belief_accuracy", 0.0) or 0.0), "family_abstain_calibration_accuracy": float(pg271_family.get("abstain_calibration_accuracy", 0.0) or 0.0), "unsupported_positive_count": int(pg271_all.get("unsupported_positive_count", 0) or 0), "audit_pass": pg271_audit_pass, "training_promotion_blocked": True, "memory_promotion_blocked": True, "vulnerability_claim_blocked": True, "evidence_hash": str(pg271_audit.get("audit_sha256") or pg271_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg271", "label": "PG-271 fresh seed 候选", "value": f"{float(pg271_family.get('next_action_accuracy', 0.0) or 0.0) * 100:.0f}%" if pg271_done else "PENDING", "status": "pass" if pg271_done and pg271_audit_pass and int(pg271_all.get("unsupported_positive_count", 0)) == 0 else ("blocked" if pg271_done else "partial"), "note": "新 seed 抽象 action/belief/abstain；不是漏洞确认"})
    if pg271_done:
        snapshot["capability"]["next"] = "PG-272：增加独立实现/新页面表面而非同一路由换 seed，验证疑问→组装→失败更新→repair/abstain 的泛化；通过后才考虑受约束 offline RL。"
    # PG-272→277 are the falsifiable question-composition evidence chain.
    pg272_metrics = dict(pg272_report.get("metrics") or {})
    pg272_done = str(pg272_report.get("status", "")) == "completed_independent_implementation_evaluation"
    snapshot["source_reports"].append({"name": pg272_name, "updated_at": _report_time(pg272_name), "sha256": str(pg272_report.get("report_sha256", ""))})
    snapshot["capability"]["model"]["pg272"] = {"status": str(pg272_report.get("status", "not_run")), "positive_recall": float(pg272_metrics.get("positive_recall_candidate", 0.0) or 0.0), "negative_reject": float(pg272_metrics.get("negative_reject_candidate", 0.0) or 0.0), "false_negative_count": int(pg272_metrics.get("false_negative_candidate_count", 0) or 0), "audit_pass": bool(pg272_audit.get("all_required_fields_complete")), "promotion_blocked": True, "evidence_hash": str(pg272_audit.get("audit_sha256") or pg272_report.get("report_sha256", ""))[:16]}
    snapshot["tasks"]["reviewer"].append({"id": "pg272-independent-surface-diagnosis", "role": "reviewer", "owner": "AI → 人工", "human_required": True, "status": "promotion_blocked" if pg272_done else "planned", "label": "PG-272 独立实现基线诊断", "route": "9 unseen surfaces", "seed": 27201, "method": "GET", "typed_effect": False, "confirmed_positive": False, "reasons": [f"positive recall {float(pg272_metrics.get('positive_recall_candidate', 0.0) or 0.0) * 100:.0f}%；negative reject {float(pg272_metrics.get('negative_reject_candidate', 0.0) or 0.0) * 100:.0f}%；正例漏检 {int(pg272_metrics.get('false_negative_candidate_count', 0) or 0)}"], "evidence_hash": str(pg272_audit.get("audit_sha256") or pg272_report.get("report_sha256", ""))[:16], "instruction": "把 PG-272 当作表示/疑问状态失败证据，不把全 abstain 当成功；不得晋级训练记忆。", "raw_material_available": False})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg272", "label": "PG-272 独立实现诊断", "value": f"{float(pg272_metrics.get('positive_recall_candidate', 0.0) or 0.0) * 100:.0f}%" if pg272_done else "PENDING", "status": "blocked" if pg272_done else "partial", "note": "正例召回 0%；证明需要改表示/轨迹，不是继续堆 RL"})
    for name, report_obj, audit_obj, label, seed, role in ((pg274_name, pg274_report, pg274_audit, "PG-274 分数 + 随机 REINFORCE", 27401, "trainer"), (pg275_name, pg275_report, pg275_audit, "PG-275 表示/保守策略/DPO 消融", 27501, "trainer"), (pg276_name, pg276_report, pg276_audit, "PG-276 第三实现 + 旧 canary", 27601, "reviewer")):
        snapshot["source_reports"].append({"name": name, "updated_at": _report_time(name), "sha256": str(report_obj.get("report_sha256", ""))})
        status = str(report_obj.get("status", "not_run"))
        gate = dict(report_obj.get("capability_gate") or report_obj.get("gates") or {})
        snapshot["tasks"][role].append({"id": name.removesuffix("_report_v1.json"), "role": role, "owner": "AI → 人工", "human_required": True, "status": "review_ready" if status.startswith("completed") else "planned", "label": label, "route": "abstract token ablation / implementation holdout", "seed": seed, "method": "offline", "typed_effect": False, "confirmed_positive": False, "reasons": [str(report_obj.get("interpretation") or report_obj.get("formal_conclusion") or "capability claim blocked")], "evidence_hash": str(audit_obj.get("audit_sha256") or report_obj.get("report_sha256", ""))[:16], "instruction": "只按 v2/v3 留出正例召回、负例拒绝、误报、漏报和 canary 判断；结果不写长期记忆。", "raw_material_available": False})
        snapshot["capability"]["model"][name.removesuffix("_report_v1.json")] = {"status": status, "gate": gate, "audit_pass": audit_obj.get("status") == "passed", "promotion_blocked": True, "evidence_hash": str(audit_obj.get("audit_sha256") or report_obj.get("report_sha256", ""))[:16]}
    if pg276_report.get("status") == "completed_third_implementation_replay":
        snapshot["capability"]["next"] = "PG-277：同一 v3 generator 多 seed + failure/repair canary，并加入一个未见漏洞族；检查是否超越 HTML 属性模板记忆。"
    pg277_agg = dict(pg277_report.get("aggregated") or {})
    pg277_done = pg277_report.get("status") == "completed_question_composition_ablation"
    pg277_process = dict(pg277_agg.get("enriched_process_sft") or {})
    pg277_conservative = dict(pg277_agg.get("conservative_offline_update") or {})
    pg277_dpo = dict(pg277_agg.get("dpo_preference_update") or {})
    pg277_coarse = dict(pg277_agg.get("coarse_process_sft") or {})
    pg277_final_only = dict(pg277_agg.get("enriched_final_only_sft") or {})
    pg277_collision = dict((pg277_dataset.get("projection_collision_audit") or {}).get("coarse") or {})
    pg277_audit_pass = pg277_audit.get("status") == "passed"
    snapshot["source_reports"].append({"name": pg277_name, "updated_at": _report_time(pg277_name), "sha256": str(pg277_report.get("report_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg277-question-composition-audit",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "review_ready" if pg277_done and pg277_audit_pass else "planned",
        "label": "PG-277 信息碰撞 / 过程监督 / 奖励消融",
        "route": "alpha+beta train → gamma implementation holdout",
        "seed": 27711,
        "method": "offline / loopback fixture",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"coarse collision {int(pg277_collision.get('conflicting_record_count', 0) or 0)} rows / {int(pg277_collision.get('conflict_group_count', 0) or 0)} groups",
            f"process exact-question worst seed {_metric_bound(pg277_report, 'enriched_process_sft', 'missing_observation', 'question_recovery_rate', 'min') * 100:.0f}%",
            f"DPO exact-question worst seed {_metric_bound(pg277_report, 'dpo_preference_update', 'missing_observation', 'question_recovery_rate', 'min') * 100:.0f}%",
            "one controlled family; promotion remains blocked",
        ],
        "evidence_hash": str(pg277_audit.get("audit_sha256") or pg277_report.get("report_sha256", ""))[:16],
        "instruction": "按采集任务书补 missing-question 反事实、失败→repair 和真实多族实现留出；不得用更多 RL 步数掩盖缺观测。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg277_question_composition"] = {
        "status": str(pg277_report.get("status", "not_run")),
        "controlled_row_count": int((pg277_dataset.get("counts") or {}).get("train", 0) or 0) + int((pg277_dataset.get("counts") or {}).get("holdout", 0) or 0),
        "coarse_conflict_group_count": int(pg277_collision.get("conflict_group_count", 0) or 0),
        "coarse_conflicting_record_count": int(pg277_collision.get("conflicting_record_count", 0) or 0),
        "coarse_positive_recall": _metric_bound(pg277_report, "coarse_process_sft", "holdout", "positive_recall", "mean"),
        "final_only_pre_question_accuracy": _metric_bound(pg277_report, "enriched_final_only_sft", "holdout", "pre_question_accuracy", "mean"),
        "process_positive_recall": _metric_bound(pg277_report, "enriched_process_sft", "holdout", "positive_recall", "mean"),
        "process_ask_recovery": _metric_bound(pg277_report, "enriched_process_sft", "missing_observation", "ask_recovery_rate", "mean"),
        "process_question_recovery_min": _metric_bound(pg277_report, "enriched_process_sft", "missing_observation", "question_recovery_rate", "min"),
        "conservative_question_recovery_min": _metric_bound(pg277_report, "conservative_offline_update", "missing_observation", "question_recovery_rate", "min"),
        "dpo_question_recovery_min": _metric_bound(pg277_report, "dpo_preference_update", "missing_observation", "question_recovery_rate", "min"),
        "audit_pass": pg277_audit_pass,
        "promotion_blocked": True,
        "evidence_hash": str(pg277_audit.get("audit_sha256") or pg277_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg277",
        "label": "PG-277 疑问恢复最坏 seed",
        "value": f"{_metric_bound(pg277_report, 'enriched_process_sft', 'missing_observation', 'question_recovery_rate', 'min') * 100:.0f}%" if pg277_done else "PENDING",
        "status": "blocked" if pg277_done else "partial",
        "note": "完整输入 recall 100%，但精确下一问跨 seed 不稳；真实多族 gold=0",
    })
    if pg277_done:
        snapshot["capability"]["limits"].append("PG-277 证明 coarse context 不可学、final-only 不会提问；process/DPO 精确问题恢复最坏 seed=0%，只有保守更新在本受控族稳定为 100%。")
        snapshot["capability"]["next"] = "PG-278：按采集任务书补 missing-question 反事实与真实多族 failure→repair 轨迹；用跨实现/族外/最坏 seed 门验证。"
    pg278_agg = dict(pg278_report.get("aggregated") or {})
    pg278_gate = dict(pg278_report.get("hypothesis_gate") or {})
    pg278_done = pg278_report.get("status") == "completed_controlled_multifamily_question_policy_study"
    pg278_dataset_audit_pass = pg278_audit.get("status") == "passed"
    pg278_model_audit_pass = pg278_model_audit.get("status") == "passed"
    pg278_audit_pass = pg278_dataset_audit_pass and pg278_model_audit_pass
    pg278_counts = dict(pg278_dataset.get("counts") or {})
    pg278_collisions = dict(pg278_dataset.get("projection_collision_audit") or {})
    pg278_family_question_values = [float(dict(value).get("pre_question_accuracy", {}).get("min", 0.0) or 0.0) for value in dict(pg278_report.get("family_holdout_abstract_question") or {}).values()]
    pg278_family_question = min(pg278_family_question_values, default=0.0)
    snapshot["source_reports"].append({"name": pg278_name, "updated_at": _report_time(pg278_name), "sha256": str(pg278_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg278_multifamily_question_policy_audit_v1.json", "updated_at": _report_time("pg278_multifamily_question_policy_audit_v1.json"), "sha256": str(pg278_model_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg278-multifamily-question-policy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "review_ready" if pg278_done and pg278_audit_pass and str(pg278_gate.get("status")) == "passed" else "promotion_blocked",
        "label": "PG-278 四族缺失观测 / 失败修复训练",
        "route": f"4 families · {int(pg278_counts.get('total', 0) or 0)} controlled rows · implementation holdout",
        "seed": 27811,
        "method": "GET/POST / loopback fixture",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"gate={str(pg278_gate.get('status', 'blocked'))}",
            f"independent audit={str(pg278_model_audit.get('status', 'missing'))}",
            f"pre/post transition worst seed {_metric_bound(pg278_report, 'enriched_process_sft', 'implementation_holdout', 'pre_transition_accuracy', 'min') * 100:.0f}%/{_metric_bound(pg278_report, 'enriched_process_sft', 'implementation_holdout', 'post_transition_accuracy', 'min') * 100:.0f}%",
            f"pair flip {_metric_bound(pg278_report, 'enriched_process_sft', 'paired_counterfactual', 'paired_counterfactual_transition_accuracy', 'min') * 100:.0f}%；real gold={int((pg278_dataset.get('source') or {}).get('real_multifamily_gold_rows', 0) or 0)}",
        ],
        "evidence_hash": str(pg278_model_audit.get("audit_sha256") or pg278_report.get("report_sha256", ""))[:16],
        "instruction": "先检查 request condition、post-observation collision 和 paired failure→repair；controlled gate 通过也不晋级真实漏洞能力或长期记忆。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg278"] = {
        "status": str(pg278_report.get("status", "not_run")),
        "controlled_row_count": int(pg278_counts.get("total", 0) or 0),
        "family_count": len(pg278_counts.get("families") or {}),
        "implementation_holdout_pre_transition": _metric_bound(pg278_report, "enriched_process_sft", "implementation_holdout", "pre_transition_accuracy", "min"),
        "implementation_holdout_post_transition": _metric_bound(pg278_report, "enriched_process_sft", "implementation_holdout", "post_transition_accuracy", "min"),
        "slot_binding_worst_seed": _metric_bound(pg278_report, "enriched_process_sft", "implementation_holdout", "pre_slot_accuracy", "min"),
        "paired_counterfactual_worst_seed": _metric_bound(pg278_report, "enriched_process_sft", "paired_counterfactual", "paired_counterfactual_transition_accuracy", "min"),
        "missing_safe_worst_seed": _metric_bound(pg278_report, "enriched_process_sft", "missing_observation", "safe_non_supported_rate", "min"),
        "family_question_worst_seed": pg278_family_question,
        "coarse_conflict_groups": int(dict(pg278_collisions.get("coarse") or {}).get("conflict_group_count", 0) or 0),
        "post_conflict_groups": int(dict(pg278_collisions.get("post") or {}).get("conflict_group_count", 0) or 0),
        "dataset_audit_pass": pg278_dataset_audit_pass,
        "model_audit_pass": pg278_model_audit_pass,
        "audit_pass": pg278_audit_pass,
        "gate_pass": str(pg278_gate.get("status")) == "passed",
        "real_multifamily_gold_rows": int((pg278_dataset.get("source") or {}).get("real_multifamily_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg278_model_audit.get("audit_sha256") or pg278_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg278", "label": "PG-278 受控四族 slot binding", "value": f"{_metric_bound(pg278_report, 'enriched_process_sft', 'paired_counterfactual', 'paired_counterfactual_transition_accuracy', 'min') * 100:.0f}%" if pg278_done else "PENDING", "status": "pass" if pg278_done and pg278_audit_pass and str(pg278_gate.get("status")) == "passed" else ("blocked" if pg278_done else "partial"), "note": "pre/post/paired 最坏 seed；controlled only，真实 gold=0"})
    if pg278_done:
        snapshot["capability"]["limits"].append(f"PG-278 288 条四族受控记录，pre/post transition={_metric_bound(pg278_report, 'enriched_process_sft', 'implementation_holdout', 'pre_transition_accuracy', 'min') * 100:.0f}%/{_metric_bound(pg278_report, 'enriched_process_sft', 'implementation_holdout', 'post_transition_accuracy', 'min') * 100:.0f}%、pair flip={_metric_bound(pg278_report, 'enriched_process_sft', 'paired_counterfactual', 'paired_counterfactual_transition_accuracy', 'min') * 100:.0f}%；真实 gold=0，不能声称公网能力")
        snapshot["capability"]["next"] = "PG-279：把同一请求条件/观测/失败修复契约接到授权本地真实回放，再做族外与遗忘矩阵。"
    pg279_replay = dict(pg279_dataset.get("replay_contract") or {})
    pg279_counts = dict(pg279_dataset.get("counts") or {})
    pg279_source = dict(pg279_dataset.get("source") or {})
    pg279_gate = dict(pg279_report.get("hypothesis_gate") or {})
    pg279_retention = dict(pg279_report.get("retention_matrix") or {})
    pg279_family_values = [float(dict(value).get("pre_question_accuracy", {}).get("min", 0.0) or 0.0) for value in dict(pg279_report.get("family_holdout_abstract_question") or {}).values()]
    pg279_family_question = min(pg279_family_values, default=0.0)
    pg279_policy_audit_pass = pg279_model_audit.get("status") == "passed"
    snapshot["source_reports"].append({"name": pg279_name, "updated_at": _report_time(pg279_name), "sha256": str(pg279_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg279_remote_replay_policy_audit_v1.json", "updated_at": _report_time("pg279_remote_replay_policy_audit_v1.json"), "sha256": str(pg279_model_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg279-remote-replay-policy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-279 远程 GET/POST replay + 遗忘保持",
        "route": f"4 families · {int(pg279_counts.get('total', 0) or 0)} rows · remote A800 GPU0",
        "seed": 27911,
        "method": "GET/POST / remote loopback",
        "typed_effect": bool(pg279_replay.get("typed_effect_rows", 0)),
        "confirmed_positive": False,
        "reasons": [
            f"operational audit={str(pg279_model_audit.get('status', 'missing'))}",
            f"family-heldout scientific gate={str(pg279_gate.get('status', 'blocked'))}",
            f"retention={str(pg279_retention.get('status', 'blocked'))}",
            f"real application gold={int(pg279_source.get('real_application_gold_rows', 0) or 0)}; promotion blocked",
        ],
        "evidence_hash": str(pg279_model_audit.get("audit_sha256") or pg279_report.get("report_sha256", ""))[:16],
        "instruction": "查看 GET/POST wire 投影、failure→repair、typed/abstain、两次 fresh hash 与 PG-278 retention；族外 gate blocked 不得当作成功。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg279"] = {
        "status": str(pg279_report.get("status", "not_run")),
        "controlled_row_count": int(pg279_counts.get("total", 0) or 0),
        "family_count": len(pg279_counts.get("families") or {}),
        "get_rows": int(pg279_replay.get("get_rows", 0) or 0),
        "post_rows": int(pg279_replay.get("post_rows", 0) or 0),
        "failure_repair_rows": int(pg279_replay.get("failure_repair_rows", 0) or 0),
        "typed_effect_rows": int(pg279_replay.get("typed_effect_rows", 0) or 0),
        "abstain_rows": int(pg279_replay.get("abstain_rows", 0) or 0),
        "coarse_conflict_groups": int(dict((pg279_dataset.get("projection_collision_audit") or {}).get("coarse") or {}).get("conflict_group_count", 0) or 0),
        "enriched_conflict_groups": int(dict((pg279_dataset.get("projection_collision_audit") or {}).get("enriched") or {}).get("conflict_group_count", 0) or 0),
        "post_conflict_groups": int(dict((pg279_dataset.get("projection_collision_audit") or {}).get("post") or {}).get("conflict_group_count", 0) or 0),
        "family_question_worst_seed": pg279_family_question,
        "retention_status": str(pg279_retention.get("status", "blocked")),
        "retention_pre_min": float(dict(pg279_retention.get("after_min") or {}).get("pre_transition_accuracy", 0.0) or 0.0),
        "retention_post_min": float(dict(pg279_retention.get("after_min") or {}).get("post_transition_accuracy", 0.0) or 0.0),
        "retention_missing_safe_min": float(dict(pg279_retention.get("after_min") or {}).get("missing_safe_rate", 0.0) or 0.0),
        "operational_audit_pass": pg279_policy_audit_pass,
        "scientific_gate_status": str(pg279_gate.get("status", "blocked")),
        "real_application_gold_rows": int(pg279_source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg279_model_audit.get("audit_sha256") or pg279_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg279", "label": "PG-279 remote replay / retention", "value": f"{int(pg279_replay.get('get_rows', 0) or 0)}/{int(pg279_replay.get('post_rows', 0) or 0)}", "status": "pass" if pg279_policy_audit_pass and str(pg279_retention.get("status")) == "passed" else "blocked", "note": f"GET/POST · failure→repair {int(pg279_replay.get('failure_repair_rows', 0) or 0)} · family gate {str(pg279_gate.get('status', 'blocked'))}"})
    if pg279_policy_audit_pass:
        snapshot["capability"]["limits"].append(f"PG-279 远程回放审计通过，GET/POST={int(pg279_replay.get('get_rows', 0) or 0)}/{int(pg279_replay.get('post_rows', 0) or 0)}、failure→repair={int(pg279_replay.get('failure_repair_rows', 0) or 0)}、遗忘保持={str(pg279_retention.get('status', 'blocked'))}；族外 question 最坏={pg279_family_question * 100:.0f}%，real gold=0")
        snapshot["capability"]["next"] = "PG-280：授权远程 Docker/真实应用可用后，补 shared slot ontology 与族外 hard-negative，再做真实 gold 验收。"
    pg280_counts = dict(pg280_dataset.get("counts") or {})
    pg280_ident = dict(pg280_report.get("identifiability") or pg280_dataset.get("identifiability") or {})
    pg280_comparison = dict(pg280_report.get("comparison") or {})
    pg280_final_only = dict(pg280_comparison.get("final_only") or {})
    pg280_process = dict(pg280_comparison.get("process") or {})
    pg280_hard_negative = dict(pg280_report.get("family_ood_hard_negative") or {})
    pg280_source = dict(pg280_report.get("source") or pg280_dataset.get("source") or {})
    pg280_gate = dict(pg280_report.get("hypothesis_gate") or {})
    pg280_docker = dict(pg280_report.get("docker_probe") or pg280_docker_probe or {})
    pg280_policy_audit_pass = pg280_model_audit.get("status") == "passed"
    pg280_report_ready = pg280_report.get("status") == "completed_remote_pg280_ontology_policy_study"
    pg280_families = len({str(item.get("family")) for item in list(pg280_dataset.get("records") or []) if isinstance(item, dict) and item.get("family")})
    pg280_impls = len({str(item.get("implementation")) for item in list(pg280_dataset.get("records") or []) if isinstance(item, dict) and item.get("implementation")})
    pg280_seeds = len({item.get("collection_seed") for item in list(pg280_dataset.get("records") or []) if isinstance(item, dict) and item.get("collection_seed") is not None})
    pg280_encodings = len({str(item.get("encoding")) for item in list(pg280_dataset.get("records") or []) if isinstance(item, dict) and item.get("encoding")})
    snapshot["source_reports"].append({"name": pg280_name, "updated_at": _report_time(pg280_name), "sha256": str(pg280_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg280_ontology_policy_audit_v1.json", "updated_at": _report_time("pg280_ontology_policy_audit_v1.json"), "sha256": str(pg280_model_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg280-ontology-policy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-280 shared slot ontology + 族外 hard-negative",
        "route": f"{pg280_families} families · {int(pg280_counts.get('total', 0) or 0)} rows · remote A800 GPU0",
        "seed": 28011,
        "method": "abstract process / remote-only",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"operational/math audit={str(pg280_model_audit.get('status', 'missing'))}",
            f"scientific family gate={str(pg280_gate.get('status', 'blocked'))}",
            f"final-only post={float(pg280_final_only.get('post_transition_accuracy_min', 0.0) or 0.0) * 100:.0f}% but pre supervision={int(pg280_final_only.get('pre_supervision_rows', 0) or 0)}",
            f"process ASK={float(pg280_process.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}% / safe={float(pg280_process.get('missing_safe_non_supported_min', 0.0) or 0.0) * 100:.0f}%",
            f"family-OOD hard-negative={int(pg280_hard_negative.get('rows', 0) or 0)} evaluation-only; Docker={str(pg280_docker.get('status', 'unavailable'))}; real gold=0",
        ],
        "evidence_hash": str(pg280_model_audit.get("audit_sha256") or pg280_report.get("report_sha256", ""))[:16],
        "instruction": "查看 shared slot ontology、coarse collision/熵、final-only 与 process 对照、族外 hard-negative 和 Docker probe；只有真实应用 gold 与科学 gate 同时通过才可晋级。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg280"] = {
        "status": str(pg280_report.get("status", "not_run")),
        "controlled_row_count": int(pg280_counts.get("total", 0) or 0),
        "family_count": pg280_families,
        "shared_slot_token_count": len(list((pg280_dataset.get("shared_slot_ontology") or {}).get("tokens") or [])),
        "conditional_entropy_bits": float(pg280_ident.get("conditional_entropy_bits", 0.0) or 0.0),
        "bayes_error_lower_bound": float(pg280_ident.get("bayes_error_lower_bound", 0.0) or 0.0),
        "final_only_pre_supervision_rows": int(pg280_final_only.get("pre_supervision_rows", 0) or 0),
        "final_only_post_accuracy": float(pg280_final_only.get("post_transition_accuracy_min", 0.0) or 0.0),
        "final_only_ask_rate": float(pg280_final_only.get("missing_ask_rate_min", 0.0) or 0.0),
        "process_pre_supervision_rows": int(pg280_process.get("pre_supervision_rows", 0) or 0),
        "process_post_accuracy": float(pg280_process.get("post_transition_accuracy_min", 0.0) or 0.0),
        "process_ask_rate": float(pg280_process.get("missing_ask_rate_min", 0.0) or 0.0),
        "process_safe_rate": float(pg280_process.get("missing_safe_non_supported_min", 0.0) or 0.0),
        "hard_negative_rows": int(pg280_hard_negative.get("rows", 0) or 0),
        "hard_negative_training_eligible": bool(pg280_hard_negative.get("training_eligible", False)),
        "docker_status": str(pg280_docker.get("status", "unavailable")),
        "remote_adapter_status": str(pg280_remote_adapter_probe.get("status", "not_run")),
        "remote_adapter_audit_pass": pg280_remote_adapter_audit.get("status") == "passed",
        "remote_adapter_probe_hash": str(pg280_remote_adapter_probe.get("evidence_sha256", ""))[:16],
        "remote_adapter_mutations_allowed": bool((pg280_remote_adapter_probe.get("scope") or {}).get("mutating_docker_commands_allowed", True)),
        "remote_adapter_real_application_gold_rows": int(pg280_remote_adapter_probe.get("real_application_gold_rows", 0) or 0),
        "scientific_gate_status": str(pg280_gate.get("status", "blocked")),
        "operational_audit_pass": pg280_policy_audit_pass,
        "promotion_blocked": True,
        "real_application_gold_rows": int(pg280_source.get("real_application_gold_rows", 0) or 0),
        "evidence_hash": str(pg280_model_audit.get("audit_sha256") or pg280_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg280", "label": "PG-280 shared ontology / ASK", "value": f"{float(pg280_process.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%" if pg280_report_ready else "PENDING", "status": "blocked" if pg280_report_ready else "partial", "note": f"final-only post {float(pg280_final_only.get('post_transition_accuracy_min', 0.0) or 0.0) * 100:.0f}% / ASK {float(pg280_final_only.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%；process ASK {float(pg280_process.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%；Docker {str(pg280_docker.get('status', 'unavailable'))}"})
    if pg280_report_ready:
        snapshot["capability"]["limits"].append(f"PG-280 shared ontology：coarse H={float(pg280_ident.get('conditional_entropy_bits', 0.0) or 0.0):.1f} bit、Bayes 下界={float(pg280_ident.get('bayes_error_lower_bound', 0.0) or 0.0):.2f}；final-only post={float(pg280_final_only.get('post_transition_accuracy_min', 0.0) or 0.0) * 100:.0f}% 但 pre/ASK=0，process ASK/safe={float(pg280_process.get('missing_ask_rate_min', 0.0) or 0.0) * 100:.0f}%/{float(pg280_process.get('missing_safe_non_supported_min', 0.0) or 0.0) * 100:.0f}%；hard-negative evaluation-only，Docker={str(pg280_docker.get('status', 'unavailable'))}，promotion blocked")
        snapshot["capability"]["next"] = "PG-281：授权远程 Docker 可用后接真实应用 evaluator；保留 shared ontology、族外 hard-negative、ASK/未决和独立审计门。"
    pg281_counts = dict(pg281_dataset.get("counts") or {})
    pg281_agg = dict(pg281_report.get("aggregated") or {})
    pg281_guarded = dict(pg281_agg.get("guarded_sft") or {})
    pg281_sweep = dict(pg281_report.get("risk_weight_sweep") or {})
    pg281_route = dict(pg281_guarded.get("route_dev") or {})
    pg281_family = dict(pg281_guarded.get("family_holdout") or {})
    pg281_hard = dict(pg281_guarded.get("hard_negative") or {})
    pg281_source = dict(pg281_report.get("source") or {})
    pg281_gate = dict(pg281_report.get("hypothesis_gate") or {})
    pg281_policy_audit_pass = pg281_model_audit.get("status") == "passed"
    pg281_report_ready = pg281_report.get("status") == "completed_remote_pg281_payload_policy_study"
    pg281_families = len({str(row.get("family")) for row in list(pg281_dataset.get("records") or []) if isinstance(row, dict) and row.get("family")})
    snapshot["source_reports"].append({"name": pg281_name, "updated_at": _report_time(pg281_name), "sha256": str(pg281_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg281_payload_policy_audit_v1.json", "updated_at": _report_time("pg281_payload_policy_audit_v1.json"), "sha256": str(pg281_model_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg281-abstract-payload-policy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-281 抽象 probe plan + safe gate",
        "route": f"{int(pg281_counts.get('total', 0) or 0)} rows · remote A800 GPU0 · risk-weight sweep",
        "seed": 28111,
        "method": "abstract GET/POST plan",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"route positive replay recall={float(pg281_route.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"family holdout positive replay recall={float(pg281_family.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"hard-negative reject={float(pg281_hard.get('safe_reject_rate', {}).get('min', 0.0) or 0.0) * 100:.0f}% / false-allow={int(pg281_hard.get('false_allow_count', {}).get('max', 0) or 0)}",
            f"selected variant={str(pg281_sweep.get('selected_variant', 'not_recorded'))}",
            "literal payload generation=false; live send=false; typed evaluator required",
        ],
        "evidence_hash": str(pg281_model_audit.get("audit_sha256") or pg281_report.get("report_sha256", ""))[:16],
        "instruction": "查看 abstract probe class/channel/encoding/action、正例 replay recall、hard-negative zero false-allow；PG-282 真实 evaluator 接通前不得生成 payload gold。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg281"] = {
        "status": str(pg281_report.get("status", "not_run")),
        "record_count": int(pg281_counts.get("total", 0) or 0),
        "train_count": int(pg281_counts.get("train", 0) or 0),
        "route_dev_count": int(pg281_counts.get("route_dev", 0) or 0),
        "family_holdout_count": int(pg281_counts.get("family_holdout", 0) or 0),
        "hard_negative_count": int(pg281_counts.get("hard_negative", 0) or 0),
        "route_positive_recall": float(pg281_route.get("positive_replay_recall", {}).get("min", 0.0) or 0.0),
        "family_positive_recall": float(pg281_family.get("positive_replay_recall", {}).get("min", 0.0) or 0.0),
        "route_plan_exact_accuracy": float(pg281_route.get("plan_exact_accuracy", {}).get("min", 0.0) or 0.0),
        "family_plan_exact_accuracy": float(pg281_family.get("plan_exact_accuracy", {}).get("min", 0.0) or 0.0),
        "hard_negative_safe_reject": float(pg281_hard.get("safe_reject_rate", {}).get("min", 0.0) or 0.0),
        "hard_negative_false_allow": int(pg281_hard.get("false_allow_count", {}).get("max", 0) or 0),
        "selected_variant": str(pg281_sweep.get("selected_variant", "not_recorded")),
        "risk_weight_variant_count": len(dict(pg281_sweep.get("variants") or {})),
        "literal_payload_generation": False,
        "live_send": False,
        "docker_status": "unavailable" if pg281_source.get("remote_docker_available") is False else "unknown",
        "scientific_gate_status": str(pg281_gate.get("status", "blocked")),
        "operational_audit_pass": pg281_policy_audit_pass,
        "promotion_blocked": True,
        "real_application_gold_rows": int(pg281_source.get("real_application_gold_rows", 0) or 0),
        "evidence_hash": str(pg281_model_audit.get("audit_sha256") or pg281_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg281", "label": "PG-281 abstract payload plan", "value": f"{float(pg281_route.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg281_report_ready else "PENDING", "status": "blocked" if pg281_report_ready else "partial", "note": f"family {float(pg281_family.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% · hard reject {float(pg281_hard.get('safe_reject_rate', {}).get('min', 0.0) or 0.0) * 100:.0f}% · false-allow {int(pg281_hard.get('false_allow_count', {}).get('max', 0) or 0)} · live send false"})
    if pg281_report_ready:
        snapshot["capability"]["limits"].append(f"PG-281 抽象 plan：route/family 正例 replay recall={float(pg281_route.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%/{float(pg281_family.get('positive_replay_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%，hard-negative reject={float(pg281_hard.get('safe_reject_rate', {}).get('min', 0.0) or 0.0) * 100:.0f}%、false-allow={int(pg281_hard.get('false_allow_count', {}).get('max', 0) or 0)}；literal payload/live send=false，Docker unavailable，promotion blocked")
        snapshot["capability"]["next"] = "PG-282：授权远程 Docker 可用后，把抽象 plan 绑定到一个非破坏性 GET/POST evaluator，比较 AI plan、reference wire、negative 与 typed oracle。"
    pg282_counts = dict(pg282_report.get("counts") or {})
    pg282_checks = dict(pg282_report.get("checks") or {})
    pg282_by_status = dict(pg282_counts.get("by_status") or {})
    pg282_ready = pg282_report.get("status") == "completed_offline_pg282_binding_contract"
    pg282_audit_pass = pg282_audit.get("status") == "passed"
    snapshot["tasks"]["trainer"].append({
        "id": "pg282-evaluator-binding",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-282 abstract plan → evaluator binding",
        "route": f"{int(pg282_counts.get('total', 0) or 0)} rows · offline contract",
        "seed": 28201,
        "method": "abstract GET/POST binding",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"remote Docker={str(pg282_report.get('source', {}).get('remote_docker_status', 'unknown'))}",
            f"await evaluator={int(pg282_by_status.get('await_evaluator', 0) or 0)} / abstain={int(pg282_by_status.get('abstain', 0) or 0)}",
            f"hard-negative abstain={'pass' if pg282_checks.get('hard_negative_all_abstain') else 'blocked'}",
            "literal payload/live replay=false; typed evidence required",
        ],
        "evidence_hash": str(pg282_audit.get("audit_sha256") or pg282_report.get("report_sha256", ""))[:16],
        "instruction": "只检查 abstract plan 到已授权 surface 的方法/通道/编码绑定；远程 Docker 与 typed evaluator 未就绪前不发送、不生成训练 gold。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg282"] = {
        "status": str(pg282_report.get("status", "not_run")),
        "record_count": int(pg282_counts.get("total", 0) or 0),
        "await_evaluator_count": int(pg282_by_status.get("await_evaluator", 0) or 0),
        "abstain_count": int(pg282_by_status.get("abstain", 0) or 0),
        "confirmed_positive_count": int(pg282_by_status.get("confirmed_positive", 0) or 0),
        "hard_negative_abstain": bool(pg282_checks.get("hard_negative_all_abstain", False)),
        "remote_docker_status": str(pg282_report.get("source", {}).get("remote_docker_status", "unknown")),
        "operational_audit_pass": pg282_audit_pass,
        "literal_payload_generation": False,
        "live_replay": False,
        "real_application_gold_rows": 0,
        "promotion_blocked": True,
        "evidence_hash": str(pg282_audit.get("audit_sha256") or pg282_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg282", "label": "PG-282 evaluator binding", "value": f"{int(pg282_by_status.get('await_evaluator', 0) or 0)} pending" if pg282_ready else "PENDING", "status": "blocked" if pg282_ready else "partial", "note": f"hard-negative abstain={'pass' if pg282_checks.get('hard_negative_all_abstain') else 'blocked'} · Docker {str(pg282_report.get('source', {}).get('remote_docker_status', 'unknown'))}"})
    if pg282_ready:
        snapshot["capability"]["limits"].append(f"PG-282 abstract binding={int(pg282_counts.get('total', 0) or 0)} rows、await evaluator={int(pg282_by_status.get('await_evaluator', 0) or 0)}、hard-negative abstain={'pass' if pg282_checks.get('hard_negative_all_abstain') else 'blocked'}；Docker={str(pg282_report.get('source', {}).get('remote_docker_status', 'unknown'))}，live replay/promotion blocked")
        snapshot["capability"]["next"] = "PG-282 live step：远程 Docker 与目标 typed evaluator 可用后，执行 fresh GET/POST + reference/negative + replay evidence。"
    pg283_counts = dict(pg283_report.get("split") or {})
    pg283_agg = dict(pg283_report.get("aggregated") or {})
    pg283_sweep = dict(pg283_report.get("risk_weight_sweep") or {})
    pg283_selected = str(pg283_sweep.get("selected_variant", "not_recorded"))
    pg283_selected_summary = dict((pg283_sweep.get("variants") or {}).get(pg283_selected) or {})
    pg283_engineering_gate = dict(pg283_report.get("hypothesis_gate") or {})
    pg283_scientific_gate = dict(pg283_report.get("scientific_gate") or {})
    pg283_source = dict(pg283_report.get("source") or {})
    pg283_ready = pg283_report.get("status") == "completed_remote_pg283_feedback_policy"
    pg283_audit_pass = pg283_audit.get("status") == "passed"
    snapshot["tasks"]["trainer"].append({
        "id": "pg283-feedback-policy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-283 failure→repair→replay policy",
        "route": f"{int(pg283_counts.get('train', 0) or 0)} train · {int(pg283_counts.get('family_holdout', 0) or 0)} family holdout · A800 GPU0",
        "seed": 28311,
        "method": "multi-step abstract feedback",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"engineering gate={str(pg283_engineering_gate.get('status', 'missing'))}",
            f"scientific gate={str(pg283_scientific_gate.get('status', 'missing'))}",
            f"selected={pg283_selected} · route exact={float(pg283_selected_summary.get('route_action_safe_exact_min', 0.0) or 0.0) * 100:.0f}% · family exact={float(pg283_selected_summary.get('family_action_safe_exact_min', 0.0) or 0.0) * 100:.0f}%",
            f"hard-negative false-allow={int(pg283_selected_summary.get('hard_negative_false_allow_max', 0) or 0)}",
            "template-derived transitions; live evaluator/payload success absent",
        ],
        "evidence_hash": str(pg283_audit.get("audit_sha256") or pg283_report.get("report_sha256", ""))[:16],
        "instruction": "查看每步 failure signature、next-action、safe gate、族外 hard-negative 和科学阻断原因；不要把模板化过程分数当作真实 payload 能力。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg283"] = {
        "status": str(pg283_report.get("status", "not_run")),
        "train_count": int(pg283_counts.get("train", 0) or 0),
        "route_dev_count": int(pg283_counts.get("route_dev", 0) or 0),
        "family_holdout_count": int(pg283_counts.get("family_holdout", 0) or 0),
        "hard_negative_count": int(pg283_counts.get("hard_negative", 0) or 0),
        "selected_variant": pg283_selected,
        "route_action_safe_exact": float(pg283_selected_summary.get("route_action_safe_exact_min", 0.0) or 0.0),
        "family_action_safe_exact": float(pg283_selected_summary.get("family_action_safe_exact_min", 0.0) or 0.0),
        "hard_negative_safe_reject": float(pg283_selected_summary.get("hard_negative_safe_reject_min", 0.0) or 0.0),
        "hard_negative_false_allow": int(pg283_selected_summary.get("hard_negative_false_allow_max", 0) or 0),
        "engineering_gate_status": str(pg283_engineering_gate.get("status", "blocked")),
        "scientific_gate_status": str(pg283_scientific_gate.get("status", "blocked")),
        "remote_docker_status": str(pg283_source.get("remote_docker_status", "unknown")),
        "live_send": bool(pg283_source.get("live_send", False)),
        "literal_payload_generation": False,
        "operational_audit_pass": pg283_audit_pass,
        "real_application_gold_rows": int(pg283_source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg283_audit.get("audit_sha256") or pg283_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg283", "label": "PG-283 feedback policy", "value": f"{float(pg283_selected_summary.get('route_action_safe_exact_min', 0.0) or 0.0) * 100:.0f}%" if pg283_ready else "PENDING", "status": "blocked" if pg283_ready else "partial", "note": f"scientific gate {str(pg283_scientific_gate.get('status', 'blocked'))} · hard false-allow {int(pg283_selected_summary.get('hard_negative_false_allow_max', 0) or 0)} · Docker {str(pg283_source.get('remote_docker_status', 'unknown'))}"})
    if pg283_ready:
        snapshot["capability"]["limits"].append(f"PG-283 工程过程分数 route/family={float(pg283_selected_summary.get('route_action_safe_exact_min', 0.0) or 0.0) * 100:.0f}%/{float(pg283_selected_summary.get('family_action_safe_exact_min', 0.0) or 0.0) * 100:.0f}%，但科学 gate={str(pg283_scientific_gate.get('status', 'blocked'))}：模板化轨迹、Docker/evaluator unavailable、real gold=0；promotion blocked")
        snapshot["capability"]["next"] = "PG-284：将 failure-policy 接到真实 evaluator，测量成功 replay 与安全 abstain 的分离指标。"
    pg284_counts = dict(pg284_report.get("counts") or {})
    pg284_checks = dict(pg284_report.get("checks") or {})
    pg284_engineering_gate = dict(pg284_report.get("engineering_gate") or {})
    pg284_scientific_gate = dict(pg284_report.get("scientific_gate") or {})
    pg284_source = dict(pg284_report.get("source") or {})
    pg284_ready = pg284_report.get("status") == "completed_offline_pg284_evaluator_contract"
    pg284_audit_pass = pg284_audit.get("status") == "passed"
    snapshot["source_reports"].append({"name": pg284_name, "updated_at": _report_time(pg284_name), "sha256": str(pg284_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg284_evaluator_contract_audit_v1.json", "updated_at": _report_time("pg284_evaluator_contract_audit_v1.json"), "sha256": str(pg284_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg284-evaluator-contract",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-284 typed evaluator / fresh reset contract",
        "route": f"{int(pg284_counts.get('total', 0) or 0)} offline contract rows",
        "seed": 28401,
        "method": "GET/POST evaluator projection",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"engineering gate={str(pg284_engineering_gate.get('status', 'missing'))}",
            f"scientific gate={str(pg284_scientific_gate.get('status', 'missing'))}",
            f"remote Docker={str(pg284_source.get('remote_docker_status', 'unknown'))} · confirmed_effect={int(pg284_counts.get('by_status', {}).get('confirmed_effect', 0) or 0)}",
            "fresh reset/reference/negative/replay projection required; no live request",
        ],
        "evidence_hash": str(pg284_audit.get("audit_sha256") or pg284_report.get("report_sha256", ""))[:16],
        "instruction": "远端 evaluator 可用后提交 bounded projection；当前只验证 fail-closed，不发送 payload、不升级漏洞结论。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg284"] = {
        "status": str(pg284_report.get("status", "not_run")),
        "contract_rows": int(pg284_counts.get("total", 0) or 0),
        "blocked_rows": int(pg284_counts.get("by_status", {}).get("blocked", 0) or 0),
        "confirmed_effect_rows": int(pg284_counts.get("by_status", {}).get("confirmed_effect", 0) or 0),
        "hard_negative_blocked": bool(pg284_checks.get("hard_negative_blocked", False)),
        "engineering_gate_status": str(pg284_engineering_gate.get("status", "blocked")),
        "scientific_gate_status": str(pg284_scientific_gate.get("status", "blocked")),
        "remote_docker_status": str(pg284_source.get("remote_docker_status", "unknown")),
        "live_replay": bool(pg284_source.get("live_replay", False)),
        "operational_audit_pass": pg284_audit_pass,
        "real_application_gold_rows": int(pg284_source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg284_audit.get("audit_sha256") or pg284_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg284", "label": "PG-284 typed evaluator", "value": f"{int(pg284_counts.get('by_status', {}).get('blocked', 0) or 0)} blocked" if pg284_ready else "PENDING", "status": "blocked" if pg284_ready else "partial", "note": f"confirmed effect={int(pg284_counts.get('by_status', {}).get('confirmed_effect', 0) or 0)} · Docker {str(pg284_source.get('remote_docker_status', 'unknown'))}"})
    if pg284_ready:
        snapshot["capability"]["limits"].append(f"PG-284 evaluator contract={int(pg284_counts.get('total', 0) or 0)} rows，blocked={int(pg284_counts.get('by_status', {}).get('blocked', 0) or 0)}，confirmed_effect=0；Docker={str(pg284_source.get('remote_docker_status', 'unknown'))}，live replay/promotion blocked")
        snapshot["capability"]["next"] = "PG-284-live：远程 Docker/evaluator 可用后提交真实 fresh reset + GET/POST reference/negative/candidate/replay projections。"
    pg285_counts = dict(pg285_report.get("split") or {})
    pg285_sweep = dict(pg285_report.get("risk_weight_sweep") or {})
    pg285_variants = dict(pg285_sweep.get("variants") or {})
    pg285_selected = str(pg285_sweep.get("selected_variant", "not_recorded"))
    pg285_selected_summary = dict(pg285_variants.get(pg285_selected) or {})
    pg285_engineering_gate = dict(pg285_report.get("engineering_gate") or {})
    pg285_scientific_gate = dict(pg285_report.get("scientific_gate") or {})
    pg285_source = dict(pg285_report.get("source") or {})
    pg285_ready = pg285_report.get("status") == "completed_remote_pg285_payload_grounding"
    pg285_audit_pass = pg285_audit.get("status") == "passed"
    snapshot["source_reports"].append({"name": pg285_name, "updated_at": _report_time(pg285_name), "sha256": str(pg285_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg285_payload_grounding_audit_v1.json", "updated_at": _report_time("pg285_payload_grounding_audit_v1.json"), "sha256": str(pg285_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg285-payload-grounding",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-285 failure-driven payload grounding",
        "route": f"{int(pg285_counts.get('train', 0) or 0)} train · {int(pg285_counts.get('family_holdout', 0) or 0)} family holdout · A800 GPU0",
        "seed": 28511,
        "method": "autoregressive structured wire-plan decoder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"selected={pg285_selected} · route sequence={float(pg285_selected_summary.get('route_sequence_exact_min', 0.0) or 0.0) * 100:.0f}% · family sequence={float(pg285_selected_summary.get('family_sequence_exact_min', 0.0) or 0.0) * 100:.0f}%",
            f"hard-negative false-allow={int(pg285_selected_summary.get('hard_negative_false_allow_max', 0) or 0)}",
            f"engineering gate={str(pg285_engineering_gate.get('status', 'missing'))}",
            f"scientific gate={str(pg285_scientific_gate.get('status', 'missing'))}",
            "abstract wire plan + runtime canary placeholder; no literal payload/live replay",
        ],
        "evidence_hash": str(pg285_audit.get("audit_sha256") or pg285_report.get("report_sha256", ""))[:16],
        "instruction": "人工查看 route/family sequence、repair 状态和 hard-negative 风险；Docker/evaluator 可用前不绑定真实 payload，不提升长期记忆。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg285"] = {
        "status": str(pg285_report.get("status", "not_run")),
        "train_count": int(pg285_counts.get("train", 0) or 0),
        "route_dev_count": int(pg285_counts.get("route_dev", 0) or 0),
        "family_holdout_count": int(pg285_counts.get("family_holdout", 0) or 0),
        "hard_negative_count": int(pg285_counts.get("hard_negative", 0) or 0),
        "selected_variant": pg285_selected,
        "route_sequence_exact": float(pg285_selected_summary.get("route_sequence_exact_min", 0.0) or 0.0),
        "family_sequence_exact": float(pg285_selected_summary.get("family_sequence_exact_min", 0.0) or 0.0),
        "route_action_accuracy": float(pg285_selected_summary.get("route_action_min", 0.0) or 0.0),
        "family_action_accuracy": float(pg285_selected_summary.get("family_action_min", 0.0) or 0.0),
        "hard_negative_safe_reject": float(pg285_selected_summary.get("hard_negative_safe_reject_min", 0.0) or 0.0),
        "hard_negative_false_allow": int(pg285_selected_summary.get("hard_negative_false_allow_max", 0) or 0),
        "engineering_gate_status": str(pg285_engineering_gate.get("status", "blocked")),
        "scientific_gate_status": str(pg285_scientific_gate.get("status", "blocked")),
        "remote_docker_status": str(pg285_source.get("remote_docker_status", "unknown")),
        "literal_payload_generation": bool(pg285_report.get("policy_scope", {}).get("literal_payload_generation", False)),
        "runtime_canary_placeholder": bool(pg285_report.get("policy_scope", {}).get("runtime_canary_placeholder", False)),
        "live_send": bool(pg285_source.get("live_send", False)),
        "operational_audit_pass": pg285_audit_pass,
        "real_application_gold_rows": int(pg285_source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg285_audit.get("audit_sha256") or pg285_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg285", "label": "PG-285 payload grounding", "value": f"{float(pg285_selected_summary.get('route_sequence_exact_min', 0.0) or 0.0) * 100:.0f}%" if pg285_ready else "PENDING", "status": "blocked" if pg285_ready else "partial", "note": f"selected {pg285_selected} · hard false-allow {int(pg285_selected_summary.get('hard_negative_false_allow_max', 0) or 0)} · Docker {str(pg285_source.get('remote_docker_status', 'unknown'))}"})
    if pg285_ready:
        snapshot["capability"]["limits"].append(f"PG-285 结构化 wire-plan route/family sequence={float(pg285_selected_summary.get('route_sequence_exact_min', 0.0) or 0.0) * 100:.0f}%/{float(pg285_selected_summary.get('family_sequence_exact_min', 0.0) or 0.0) * 100:.0f}%，guarded hard-negative false-allow=0；但模板轨迹、Docker/evaluator unavailable、无真实 replay，promotion blocked")
        snapshot["capability"]["next"] = "PG-285-live：远程 Docker/evaluator 可用后，把 guarded wire plan 绑定真实 GET/POST、fresh reset、negative/reference/replay 和 typed effect。"
    pg286_counts = dict(pg286_catalog.get("counts") or {})
    pg286_contract = dict(pg286_catalog.get("training_contract") or {})
    pg286_independent_pass = pg286_independent_audit.get("status") == "passed"
    pg286_docker_status = str(pg280_remote_adapter_probe.get("status", pg280_docker_probe.get("status", "unknown")))
    pg286_catalog_pass = pg286_builder_audit.get("status") == "passed" and pg286_catalog.get("catalog_sha256") == pg286_independent_audit.get("catalog_sha256")
    snapshot["source_reports"].append({"name": pg286_name, "updated_at": _report_time(pg286_name), "sha256": str(pg286_catalog.get("catalog_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg286_observation_token_catalog_independent_audit_v1.json", "updated_at": _report_time("pg286_observation_token_catalog_independent_audit_v1.json"), "sha256": str(pg286_independent_audit.get("audit_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg286_live_protocol_v1.json", "updated_at": _report_time("pg286_live_protocol_v1.json"), "sha256": str(pg286_protocol.get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg286_live_batch_audit_v1.json", "updated_at": _report_time("pg286_live_batch_audit_v1.json"), "sha256": str(pg286_batch_audit.get("audit_sha256", ""))})
    snapshot["tasks"]["collector"].append({
        "id": "pg286-observation-token-gate",
        "role": "collector",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "needs_authorized_remote_evaluator" if pg286_docker_status != "available" else "ready_for_collection",
        "label": "PG-286 shared observation slots / family-OOD hard-negative",
        "route": f"{int(pg286_counts.get('total', 0) or 0)} catalog · {int(pg286_counts.get('hard_negative', 0) or 0)} hard-negative",
        "seed": 28601,
        "method": "GET/POST response + DOM/SQL AST/redirect/logic projection",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"complete={int(pg286_counts.get('complete', 0) or 0)} · incomplete={int(pg286_counts.get('incomplete', 0) or 0)}",
            f"SQL AST available={int(pg286_independent_audit.get('sql_ast_available_rows', 0) or 0)}",
            f"Docker={pg286_docker_status} · batch={str(pg286_batch_audit.get('status', 'blocked'))} · training gold={int(pg286_batch_audit.get('training_eligible_rows', 0) or 0)}",
            "fresh GET/POST reference/negative/candidate/replay and typed evaluator required",
        ],
        "evidence_hash": str(pg286_independent_audit.get("audit_sha256") or pg286_catalog.get("catalog_sha256", ""))[:16],
        "instruction": "目标侧完成授权 GET/POST 后，将 bounded projection 送入 /api/maze/remote-docker/observation；先补 SQL AST、DOM/redirect/logic typed modality 与 fresh reset，再由独立审计决定是否能进入训练，不把 response shape 当漏洞结论。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg286"] = {
        "status": "catalog_audited_collection_only" if pg286_independent_pass else "audit_blocked",
        "total_rows": int(pg286_counts.get("total", 0) or 0),
        "complete_rows": int(pg286_counts.get("complete", 0) or 0),
        "incomplete_rows": int(pg286_counts.get("incomplete", 0) or 0),
        "sql_rows": int(pg286_counts.get("sql", 0) or 0),
        "xss_rows": int(pg286_counts.get("xss", 0) or 0),
        "redirect_rows": int(pg286_counts.get("redirect", 0) or 0),
        "hard_negative_rows": int(pg286_counts.get("hard_negative", 0) or 0),
        "sql_ast_available_rows": int(pg286_independent_audit.get("sql_ast_available_rows", 0) or 0),
        "training_eligible_rows": int(pg286_independent_audit.get("training_eligible_rows", 0) or 0),
        "memory_promotion_allowed_rows": int(pg286_independent_audit.get("memory_promotion_allowed_rows", 0) or 0),
        "shared_slot_count": 15,
        "family_hidden_in_context": bool(pg286_contract.get("family_hidden_in_context", False)),
        "oracle_label_in_context": bool(pg286_contract.get("oracle_label_in_context", True)),
        "remote_docker_status": pg286_docker_status,
        "operational_audit_pass": bool(pg286_catalog_pass and pg286_independent_pass),
        "scientific_gate_status": "blocked",
        "real_application_gold_rows": 0,
        "batch_status": str(pg286_batch_audit.get("status", "blocked")),
        "batch_record_count": int(pg286_batch_audit.get("record_count", 0) or 0),
        "batch_training_eligible_rows": int(pg286_batch_audit.get("training_eligible_rows", 0) or 0),
        "batch_audit_sha256": str(pg286_batch_audit.get("audit_sha256", "")),
        "promotion_blocked": True,
        "evidence_hash": str(pg286_independent_audit.get("audit_sha256") or pg286_catalog.get("catalog_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg286", "label": "PG-286 observation evidence", "value": f"{int(pg286_counts.get('complete', 0) or 0)}/{int(pg286_counts.get('total', 0) or 0)}", "status": "blocked", "note": f"SQL AST {int(pg286_independent_audit.get('sql_ast_available_rows', 0) or 0)} · hard-negative {int(pg286_counts.get('hard_negative', 0) or 0)} · Docker {pg286_docker_status}"})
    snapshot["capability"]["limits"].append(f"PG-286 catalog={int(pg286_counts.get('total', 0) or 0)} 行，完整={int(pg286_counts.get('complete', 0) or 0)}、不完整={int(pg286_counts.get('incomplete', 0) or 0)}；SQL AST=0、live batch={str(pg286_batch_audit.get('status', 'blocked'))}、training gold={int(pg286_batch_audit.get('training_eligible_rows', 0) or 0)}，Docker={pg286_docker_status}，promotion blocked")
    snapshot["capability"]["next"] = "PG-286-live：授权远程 Docker/evaluator 可用后补齐真实 GET/POST + SQL AST/DOM/redirect/logic typed projection，再做族外 hard-negative 训练与评估。"
    # PG-287 is a remote A800 identifiability diagnostic.  Keep the explicit
    # family-heldout failure visible in both the task queue and capability
    # card; a perfect ambiguous ASK score alone is only a safety result.
    pg287_split = dict(pg287_report.get("split") or {})
    pg287_variants = dict(pg287_report.get("variants") or {})
    pg287_selected = str(pg287_report.get("selected_variant", "not_recorded"))
    pg287_metrics = dict(pg287_variants.get(pg287_selected) or pg287_variants.get("plain_sft") or {})
    pg287_source = dict(pg287_report.get("source") or {})
    pg287_engineering = dict(pg287_report.get("engineering_gate") or {})
    pg287_scientific = dict(pg287_report.get("scientific_gate") or {})
    pg287_promotion = dict(pg287_report.get("promotion") or {})
    pg287_report_ready = pg287_report.get("status") == "completed_remote_pg287_identifiability_training"
    pg287_audit_pass = pg287_dataset_audit.get("status") == "passed"
    pg287_route_ask = float((pg287_metrics.get("route_ambiguous_ask_recall") or {}).get("min", 0.0) or 0.0)
    pg287_route_encoding = float((pg287_metrics.get("route_resolved_encoding_accuracy") or {}).get("min", 0.0) or 0.0)
    pg287_family_ask = float((pg287_metrics.get("family_ambiguous_ask_recall") or {}).get("min", 0.0) or 0.0)
    pg287_family_encoding = (pg287_metrics.get("family_resolved_encoding_accuracy") or {}).get("min")
    pg287_family_encoding = None if pg287_family_encoding is None else float(pg287_family_encoding)
    pg287_family_encoding_count = int((pg287_metrics.get("family_resolved_encoding_accuracy") or {}).get("available_count", 0) or 0)
    pg287_family_encoding_label = "N/A" if pg287_family_encoding is None else f"{pg287_family_encoding * 100:.0f}%"
    pg287_hard_ask = float((pg287_metrics.get("hard_negative_ask_recall") or {}).get("min", 0.0) or 0.0)
    pg287_hard_false = int(pg287_metrics.get("hard_negative_false_allow_max", 0) or 0)
    snapshot["source_reports"].append({"name": pg287_name, "updated_at": _report_time(pg287_name), "sha256": str(pg287_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg287_identifiability_dataset_audit_v1.json", "updated_at": _report_time("pg287_identifiability_dataset_audit_v1.json"), "sha256": str(pg287_dataset_audit.get("audit_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg287_identifiability_training_trace_v1.json", "updated_at": _report_time("pg287_identifiability_training_trace_v1.json"), "sha256": str(pg287_trace.get("trace_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg287_identifiability_training_protocol_v1.json", "updated_at": _report_time("pg287_identifiability_training_protocol_v1.json"), "sha256": str(pg287_protocol.get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg287_live_protocol_v1.json", "updated_at": _report_time("pg287_live_protocol_v1.json"), "sha256": str(pg287_live_protocol.get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg287_live_batch_audit_v1.json", "updated_at": _report_time("pg287_live_batch_audit_v1.json"), "sha256": str(pg287_live_batch_audit.get("audit_sha256", ""))})
    for report_name, report_obj in ((pg288_name, pg288_report), (pg289_name, pg289_report), (pg290_name, pg290_report), (pg291_name, pg291_report), (pg292_name, pg292_report), (pg293_name, pg293_report)):
        snapshot["source_reports"].append({"name": report_name, "updated_at": _report_time(report_name), "sha256": str(report_obj.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg292_feature_gate_trace_v1.json", "updated_at": _report_time("pg292_feature_gate_trace_v1.json"), "sha256": str(_read_json("pg292_feature_gate_trace_v1.json", {}).get("trace_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg292_feature_gate_protocol_v1.json", "updated_at": _report_time("pg292_feature_gate_protocol_v1.json"), "sha256": str(_read_json("pg292_feature_gate_protocol_v1.json", {}).get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg293_failure_next_action_training_trace_v1.json", "updated_at": _report_time("pg293_failure_next_action_training_trace_v1.json"), "sha256": str(_read_json("pg293_failure_next_action_training_trace_v1.json", {}).get("trace_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg293_failure_next_action_training_protocol_v1.json", "updated_at": _report_time("pg293_failure_next_action_training_protocol_v1.json"), "sha256": str(_read_json("pg293_failure_next_action_training_protocol_v1.json", {}).get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": pg293_local_name, "updated_at": _report_time(pg293_local_name), "sha256": str(pg293_local_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg293_failure_next_action_training_trace_v1_local_morning.json", "updated_at": _report_time("pg293_failure_next_action_training_trace_v1_local_morning.json"), "sha256": str(_read_json("pg293_failure_next_action_training_trace_v1_local_morning.json", {}).get("trace_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg293_failure_next_action_training_protocol_v1_local_morning.json", "updated_at": _report_time("pg293_failure_next_action_training_protocol_v1_local_morning.json"), "sha256": str(_read_json("pg293_failure_next_action_training_protocol_v1_local_morning.json", {}).get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": pg295_name, "updated_at": _report_time(pg295_name), "sha256": str(pg295_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg295_causal_moe_training_trace_v1_local_morning.json", "updated_at": _report_time("pg295_causal_moe_training_trace_v1_local_morning.json"), "sha256": str(_read_json("pg295_causal_moe_training_trace_v1_local_morning.json", {}).get("trace_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg295_causal_moe_training_protocol_v1_local_morning.json", "updated_at": _report_time("pg295_causal_moe_training_protocol_v1_local_morning.json"), "sha256": str(_read_json("pg295_causal_moe_training_protocol_v1_local_morning.json", {}).get("protocol_sha256", ""))})
    snapshot["source_reports"].append({"name": pg300_name, "updated_at": _report_time(pg300_name), "sha256": str(pg300_report.get("report_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg300_question_policy_dataset_v1.json", "updated_at": _report_time("pg300_question_policy_dataset_v1.json"), "sha256": str(pg300_dataset.get("dataset_sha256", ""))})
    snapshot["source_reports"].append({"name": "pg300_question_policy_audit_v1.json", "updated_at": _report_time("pg300_question_policy_audit_v1.json"), "sha256": str(pg300_audit.get("audit_sha256", ""))})
    for report_name, report_obj in ((pg301_name, pg301_report), (pg302_name, pg302_report), (pg302b_name, pg302b_report), (pg303_name, pg303_report), (pg304_name, pg304_report), (pg305_name, pg305_report), (pg306_name, pg306_report), (pg306b_name, pg306b_report), (pg306c_name, pg306c_report), (pg307_name, pg307_report), (pg308_name, pg308_report), (pg309_name, pg309_report), (pg310_name, pg310_report), (pg311_name, pg311_report), (pg312_name, pg312_report), (pg313_name, pg313_report), (pg314_name, pg314_report), (pg315_name, pg315_report), (pg316_name, pg316_report), (pg316_live_name, pg316_live_report), (pg317_name, pg317_report), (pg317_live_name, pg317_live_report), (pg318_name, pg318_report), (pg319_name, pg319_report), (pg320_name, pg320_report), (pg320_live_name, pg320_live_report), (pg321_name, pg321_report), (pg321_live_name, pg321_live_report), (pg322_name, pg322_report), (pg323_name, pg323_report), (pg323_live_name, pg323_live_report)):
        snapshot["source_reports"].append({"name": report_name, "updated_at": _report_time(report_name), "sha256": str(report_obj.get("report_sha256", ""))})
    for artifact_name, artifact_key in ((pg318_catalog_name, "catalog_sha256"), (pg318_trace_name, "trace_sha256"), (pg318_protocol_name, "protocol_sha256"), (pg321_catalog_name, "catalog_sha256"), (pg321_trace_name, "trace_sha256"), (pg321_protocol_name, "protocol_sha256"), (pg323_catalog_name, "catalog_sha256"), (pg323_trace_name, "trace_sha256"), (pg323_protocol_name, "protocol_sha256")):
        artifact = _read_json(artifact_name, {})
        snapshot["source_reports"].append({"name": artifact_name, "updated_at": _report_time(artifact_name), "sha256": str(artifact.get(artifact_key, ""))})
    for dataset_name, dataset_obj in ((pg307_dataset_name, pg307_dataset), (pg307_audit_name, pg307_audit), (pg308_dataset_name, pg308_dataset), (pg308_audit_name, pg308_audit), (pg309_dataset_name, pg309_dataset), (pg309_audit_name, pg309_audit), (pg313_dataset_name, pg313_dataset), (pg313_audit_name, pg313_audit), (pg314_dataset_name, pg314_dataset), (pg315_dataset_name, pg315_dataset), (pg316_dataset_name, pg316_dataset), (pg317_dataset_name, pg317_dataset), (pg317_audit_name, pg317_audit), (pg319_dataset_name, pg319_dataset), (pg319_audit_name, pg319_audit), (pg320_dataset_name, pg320_dataset), (pg320_audit_name, pg320_audit), (pg321_dataset_name, pg321_dataset), (pg321_audit_name, pg321_audit), (pg322_dataset_name, pg322_dataset), (pg322_audit_name, pg322_audit), (pg323_dataset_name, pg323_dataset), (pg323_audit_name, pg323_audit)):
        snapshot["source_reports"].append({"name": dataset_name, "updated_at": _report_time(dataset_name), "sha256": str(dataset_obj.get("dataset_sha256") or dataset_obj.get("audit_sha256", ""))})
    snapshot["tasks"]["trainer"].append({
        "id": "pg287-identifiability",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-287 encoding identifiability / ASK gate",
        "route": f"{int(pg287_split.get('train', 0) or 0)} train · {int(pg287_split.get('route_dev', 0) or 0)} route · {int(pg287_split.get('family_holdout', 0) or 0)} family · A800 GPU0",
        "seed": 28711,
        "method": "structured next-token Rule-IR decoder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"selected={pg287_selected} · route ASK={pg287_route_ask * 100:.0f}% / resolved encoding={pg287_route_encoding * 100:.0f}%",
            f"family ASK={pg287_family_ask * 100:.0f}% / resolved encoding={pg287_family_encoding_label} (coverage={pg287_family_encoding_count})",
            f"hard-negative ASK={pg287_hard_ask * 100:.0f}% / false-allow={pg287_hard_false}",
            f"engineering={str(pg287_engineering.get('status', 'blocked'))} · scientific={str(pg287_scientific.get('status', 'blocked'))}",
            f"Docker={str(pg287_source.get('remote_docker_status', 'unavailable'))} · real gold={int(pg287_source.get('real_application_gold_rows', 0) or 0)}",
            f"PG-287-live batch={str(pg287_live_batch_audit.get('status', 'blocked'))} · records={int(pg287_live_batch_audit.get('record_count', 0) or 0)}",
        ],
        "evidence_hash": str(pg287_trace.get("trace_sha256") or pg287_report.get("report_sha256", ""))[:16],
        "instruction": "人工重点查看 family-heldout resolved encoding coverage=0（准确率 N/A）；下一轮必须接真实 GET/POST evaluator 的 observed encoding/field-role projection，并先通过 PG-287-live batch gate。ambiguous 时 ask_typed，不能用模板标签或最终分数补齐。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg287"] = {
        "status": str(pg287_report.get("status", "not_run")),
        "train_count": int(pg287_split.get("train", 0) or 0),
        "route_dev_count": int(pg287_split.get("route_dev", 0) or 0),
        "family_holdout_count": int(pg287_split.get("family_holdout", 0) or 0),
        "hard_negative_count": int(pg287_split.get("hard_negative", 0) or 0),
        "ambiguous_count": int((pg287_dataset_audit.get("counts") or {}).get("ambiguous", 0) or 0),
        "resolved_count": int((pg287_dataset_audit.get("counts") or {}).get("resolved", 0) or 0),
        "selected_variant": pg287_selected,
        "route_ambiguous_ask_recall": pg287_route_ask,
        "route_resolved_encoding_accuracy": pg287_route_encoding,
        "family_ambiguous_ask_recall": pg287_family_ask,
        "family_resolved_encoding_accuracy": pg287_family_encoding,
        "family_resolved_encoding_count": pg287_family_encoding_count,
        "hard_negative_ask_recall": pg287_hard_ask,
        "hard_negative_false_allow": pg287_hard_false,
        "route_sequence_exact": float((pg287_metrics.get("route_sequence_exact") or {}).get("min", 0.0) or 0.0),
        "family_sequence_exact": float((pg287_metrics.get("family_sequence_exact") or {}).get("min", 0.0) or 0.0),
        "engineering_gate_status": str(pg287_engineering.get("status", "blocked")),
        "scientific_gate_status": str(pg287_scientific.get("status", "blocked")),
        "remote_docker_status": str(pg287_source.get("remote_docker_status", "unavailable")),
        "live_send": bool(pg287_source.get("live_send", False)),
        "literal_payload_generation": False,
        "operational_audit_pass": bool(pg287_audit_pass and pg287_engineering.get("status") == "passed"),
        "real_application_gold_rows": int(pg287_source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "training_eligible_rows": int(pg287_dataset_audit.get("training_eligible_rows", 0) or 0),
        "live_protocol_status": str((pg287_live_protocol.get("current_status") or {}).get("remote_docker", "unavailable")),
        "live_protocol_sha256": str(pg287_live_protocol.get("protocol_sha256", "")),
        "live_batch_status": str(pg287_live_batch_audit.get("status", "blocked")),
        "live_batch_record_count": int(pg287_live_batch_audit.get("record_count", 0) or 0),
        "live_batch_family_resolved_count": int(pg287_live_batch_audit.get("family_resolved_count", 0) or 0),
        "live_batch_blocking_reasons": list(pg287_live_batch_audit.get("blocking_reasons") or []),
        "live_batch_audit_sha256": str(pg287_live_batch_audit.get("audit_sha256", "")),
        "checkpoint_sha256": "ce7ea8a9abb5af14a85d17bc876f1b326f6d360ffd73c51fe84bd23fa9eb86b0",
        "evidence_hash": str(pg287_trace.get("trace_sha256") or pg287_report.get("report_sha256", ""))[:16],
        "promotion_reason": str(pg287_promotion.get("reason", "live gold absent")),
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg287", "label": "PG-287 identifiability / ASK", "value": f"{pg287_family_encoding_label} resolved" if pg287_report_ready else "PENDING", "status": "blocked", "note": f"family ASK {pg287_family_ask * 100:.0f}% · resolved coverage {pg287_family_encoding_count} · hard false-allow {pg287_hard_false} · Docker {str(pg287_source.get('remote_docker_status', 'unavailable'))}"})
    if pg287_report_ready:
        snapshot["capability"]["limits"].append(f"PG-287 ambiguous ASK={pg287_family_ask * 100:.0f}%、hard-negative ASK={pg287_hard_ask * 100:.0f}%/false-allow={pg287_hard_false}，但 family resolved encoding coverage={pg287_family_encoding_count}（准确率 N/A）；route resolved={pg287_route_encoding * 100:.0f}%，当前不能判断族外具体编码能力。engineering={str(pg287_engineering.get('status', 'blocked'))}、scientific={str(pg287_scientific.get('status', 'blocked'))}、Docker={str(pg287_source.get('remote_docker_status', 'unavailable'))}，promotion blocked")
    pg292_selection = dict(pg292_report.get("selection") or {})
    pg292_candidates = list(pg292_selection.get("candidates") or [])
    pg292_selected_variant = str(pg292_selection.get("selected_variant", "not_recorded"))
    pg292_selected_threshold = float(pg292_selection.get("selected_threshold", 0.0) or 0.0)
    pg292_selected = next((dict(item) for item in pg292_candidates if str(item.get("variant")) == pg292_selected_variant and float(item.get("threshold", -1.0) or -1.0) == pg292_selected_threshold), {})
    pg292_source = dict(pg292_report.get("source") or {})
    pg292_engineering = dict(pg292_report.get("engineering_gate") or {})
    pg292_scientific = dict(pg292_report.get("scientific_gate") or {})
    pg292_split = dict(pg292_report.get("split") or {})
    pg292_report_ready = pg292_report.get("status") == "completed_remote_pg292_feature_gate"
    snapshot["tasks"]["trainer"].append({
        "id": "pg292-feature-gate",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-292 key/value feature safety gate",
        "route": f"{int(pg292_split.get('mixed_train', 0) or 0)} mixed train · {int(pg292_split.get('route_dev', 0) or 0)} route · {int(pg292_split.get('family_holdout', 0) or 0)} family · A800 GPU0",
        "seed": 29201,
        "method": "key/value feature gate + typed Rule-IR decoder boundary",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"selected={pg292_selected_variant}@{pg292_selected_threshold:g} · route recall={float(pg292_selected.get('route_positive_recall_min', 0.0) or 0.0) * 100:.0f}% / family recall={float(pg292_selected.get('family_positive_recall_min', 0.0) or 0.0) * 100:.0f}%",
            f"hard false-allow={int(pg292_selected.get('hard_negative_false_allow_max', 0) or 0)} · safe-reject={float(pg292_selected.get('hard_negative_safe_reject_min', 0.0) or 0.0) * 100:.0f}%",
            f"engineering={str(pg292_engineering.get('status', 'blocked'))} · scientific={str(pg292_scientific.get('status', 'blocked'))}",
            f"Docker={str(pg292_source.get('remote_docker_status', 'unavailable'))} · real gold={int(pg292_source.get('real_application_gold_rows', 0) or 0)}",
        ],
        "evidence_hash": str(pg292_report.get("report_sha256", ""))[:16],
        "instruction": "把 feature gate 只接在抽象 Rule-IR 的 safe_to_send 边界；先用 fresh typed evaluator 验证正/负对照与复放，再谈真实 payload。feature 结果不能替代 oracle。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg292"] = {
        "status": str(pg292_report.get("status", "not_run")),
        "mixed_train_count": int(pg292_split.get("mixed_train", 0) or 0),
        "counterfactual_train_count": int(pg292_split.get("counterfactual_train", 0) or 0),
        "route_holdout_count": int(pg292_split.get("route_dev", 0) or 0),
        "family_holdout_count": int(pg292_split.get("family_holdout", 0) or 0),
        "hard_negative_count": int(pg292_split.get("hard_negative", 0) or 0),
        "feature_count": int((pg292_report.get("vocabulary") or {}).get("feature_count", 0) or 0),
        "selected_variant": pg292_selected_variant,
        "selected_threshold": pg292_selected_threshold,
        "route_positive_recall": float(pg292_selected.get("route_positive_recall_min", 0.0) or 0.0),
        "family_positive_recall": float(pg292_selected.get("family_positive_recall_min", 0.0) or 0.0),
        "hard_negative_false_allow": int(pg292_selected.get("hard_negative_false_allow_max", 0) or 0),
        "hard_negative_safe_reject": float(pg292_selected.get("hard_negative_safe_reject_min", 0.0) or 0.0),
        "engineering_gate_status": str(pg292_engineering.get("status", "blocked")),
        "scientific_gate_status": str(pg292_scientific.get("status", "blocked")),
        "remote_docker_status": str(pg292_source.get("remote_docker_status", "unavailable")),
        "live_adapter": "app/pg292_live.py",
        "live_endpoint": "/api/maze/remote-docker/pg292-live",
        "wire_emission_allowed": False,
        "literal_payload_generation": False,
        "live_send": False,
        "operational_audit_pass": bool(pg292_report_ready and pg292_engineering.get("status") == "passed"),
        "real_application_gold_rows": int(pg292_source.get("real_application_gold_rows", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg292_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg292", "label": "PG-292 feature gate", "value": f"{float(pg292_selected.get('route_positive_recall_min', 0.0) or 0.0) * 100:.0f}% / {int(pg292_selected.get('hard_negative_false_allow_max', 0) or 0)}" if pg292_report_ready else "PENDING", "status": "blocked", "note": f"route/family recall {float(pg292_selected.get('route_positive_recall_min', 0.0) or 0.0) * 100:.0f}%/{float(pg292_selected.get('family_positive_recall_min', 0.0) or 0.0) * 100:.0f}% · hard reject {float(pg292_selected.get('hard_negative_safe_reject_min', 0.0) or 0.0) * 100:.0f}% · synthetic only"})
    if pg292_report_ready:
        snapshot["capability"]["limits"].append(f"PG-292 feature gate 在合成/反事实数据上 route/family recall={float(pg292_selected.get('route_positive_recall_min', 0.0) or 0.0) * 100:.0f}%/{float(pg292_selected.get('family_positive_recall_min', 0.0) or 0.0) * 100:.0f}%、hard false-allow={int(pg292_selected.get('hard_negative_false_allow_max', 0) or 0)}；没有 fresh evaluator/真实 gold，不能据此声称 payload 成功。")
    pg293_split = dict(pg293_report.get("split") or {})
    pg293_selection = dict(pg293_report.get("selection") or {})
    pg293_source = dict(pg293_report.get("source") or {})
    pg293_engineering = dict(pg293_report.get("engineering_gate") or {})
    pg293_scientific = dict(pg293_report.get("scientific_gate") or {})
    pg293_report_ready = pg293_report.get("status") == "completed_remote_pg293_failure_next_action"
    pg293_local_device = dict(pg293_local_report.get("device") or {})
    pg293_local_selection = dict(pg293_local_report.get("selection") or {})
    pg293_local_variants = list(pg293_local_report.get("variants") or [])
    pg293_local_selected_variant = next((dict(item) for item in pg293_local_variants if int(item.get("hidden_dim", -1)) == int(pg293_local_selection.get("hidden_dim", -2))), {})
    pg293_local_holdout = dict(pg293_local_selected_variant.get("holdout") or {})
    pg293_local_hard = dict(pg293_local_selected_variant.get("hard_negative") or {})
    pg293_variants = list(pg293_report.get("variants") or [])
    pg293_selected_variant = next((dict(item) for item in pg293_variants if int(item.get("hidden_dim", -1)) == int(pg293_selection.get("hidden_dim", -2))), {})
    pg293_holdout = dict(pg293_selected_variant.get("holdout") or {})
    pg293_hard = dict(pg293_selected_variant.get("hard_negative") or {})
    snapshot["tasks"]["trainer"].append({
        "id": "pg293-failure-next-action",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-293 failure-conditioned next-action decoder",
        "route": f"{int(pg293_split.get('train', 0) or 0)} train · {int(pg293_split.get('source_holdout', 0) or 0)} source holdout · {int(pg293_split.get('seed_holdout', 0) or 0)} seed holdout · A800 GPU0",
        "seed": 29301,
        "method": "autoregressive abstract action/repair decoder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"selected hidden={int(pg293_selection.get('hidden_dim', 0) or 0)} · holdout action={float(pg293_holdout.get('action_accuracy', {}).get('min', 0.0) or 0.0) * 100:.0f}% / positive recall={float(pg293_holdout.get('positive_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"same-context hard-negative false-allow={int(pg293_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)} · safe-reject={float(pg293_hard.get('safe_reject_rate', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"engineering={str(pg293_engineering.get('status', 'blocked'))} · scientific={str(pg293_scientific.get('status', 'blocked'))}",
            f"Docker={str(pg293_source.get('remote_docker_status', 'unavailable'))} · real gold={int(pg293_source.get('real_application_gold_rows', 0) or 0)}",
            f"local morning={str(pg293_local_report.get('status', 'not_run'))} · device={str(pg293_local_device.get('device_name', 'not_recorded'))} · hard false-allow={int(pg293_local_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)}",
        ],
        "evidence_hash": str(pg293_report.get("report_sha256", ""))[:16],
        "instruction": "先修正 hard-negative false-allow 与 evaluator/feedback 缺失的输入问题，再接 PG-292 gate；greedy decode 指标优先于 teacher-forcing loss，失败样本不得晋级。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg293"] = {
        "status": str(pg293_report.get("status", "not_run")),
        "train_count": int(pg293_split.get("train", 0) or 0),
        "source_holdout_count": int(pg293_split.get("source_holdout", 0) or 0),
        "seed_holdout_count": int(pg293_split.get("seed_holdout", 0) or 0),
        "hard_negative_eval_count": int(pg293_split.get("hard_negative_eval", 0) or 0),
        "selected_hidden_dim": int(pg293_selection.get("hidden_dim", 0) or 0),
        "holdout_action_accuracy": float(pg293_holdout.get("action_accuracy", {}).get("min", 0.0) or 0.0),
        "holdout_positive_recall": float(pg293_holdout.get("positive_recall", {}).get("min", 0.0) or 0.0),
        "hard_negative_false_allow": int(pg293_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "hard_negative_safe_reject": float(pg293_hard.get("safe_reject_rate", {}).get("min", 0.0) or 0.0),
        "engineering_gate_status": str(pg293_engineering.get("status", "blocked")),
        "scientific_gate_status": str(pg293_scientific.get("status", "blocked")),
        "remote_docker_status": str(pg293_source.get("remote_docker_status", "unavailable")),
        "real_application_gold_rows": int(pg293_source.get("real_application_gold_rows", 0) or 0),
        "local_morning_status": str(pg293_local_report.get("status", "not_run")),
        "local_morning_device": str(pg293_local_device.get("device_name", "not_recorded")),
        "local_morning_holdout_positive_recall": float(pg293_local_holdout.get("positive_recall", {}).get("min", 0.0) or 0.0),
        "local_morning_hard_negative_false_allow": int(pg293_local_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "local_morning_engineering_gate_status": str((pg293_local_report.get("engineering_gate") or {}).get("status", "blocked")),
        "local_morning_report_hash": str(pg293_local_report.get("report_sha256", "")),
        "autoregressive_eval": True,
        "wire_emission_allowed": False,
        "literal_payload_generation": False,
        "promotion_blocked": True,
        "evidence_hash": str(pg293_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg293", "label": "PG-293 failure next-action", "value": f"{float(pg293_holdout.get('positive_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% / {int(pg293_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)}" if pg293_report_ready else "PENDING", "status": "blocked", "note": f"greedy source/seed holdout · local morning={str(pg293_local_report.get('status', 'not_run'))} · local hard false-allow {int(pg293_local_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)} · Docker {str(pg293_source.get('remote_docker_status', 'unavailable'))}"})
    if pg293_report_ready:
        snapshot["capability"]["limits"].append(f"PG-293 greedy next-action 在 source/seed holdout 上 action={float(pg293_holdout.get('action_accuracy', {}).get('min', 0.0) or 0.0) * 100:.0f}%、positive recall={float(pg293_holdout.get('positive_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%，但同上下文 hard-negative false-allow={int(pg293_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)}；工程门阻塞，不能晋级。")
    pg295_selection = dict(pg295_report.get("selection") or {})
    pg295_variants = list(pg295_report.get("variants") or [])
    pg295_selected = next((dict(item) for item in pg295_variants if str(item.get("config_name")) == str(pg295_selection.get("config_name"))), {})
    pg295_seed_missing = dict(pg295_selected.get("seed_missing") or {})
    pg295_hard = dict(pg295_selected.get("hard_negative") or {})
    pg295_control = dict(pg295_report.get("answer_only_control") or {})
    pg295_control_missing = dict(pg295_control.get("missing_holdout") or {})
    pg295_engineering = dict(pg295_report.get("engineering_gate") or {})
    pg295_scientific = dict(pg295_report.get("scientific_gate") or {})
    pg295_device = dict(pg295_report.get("device") or {})
    pg295_ready = pg295_report.get("status") == "completed_local_morning_pg295_causal_moe"
    snapshot["tasks"]["trainer"].append({
        "id": "pg295-causal-moe-question-composition",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-295 causal Transformer-MoE question composition",
        "route": f"{int((pg295_report.get('split') or {}).get('train', 0) or 0)} train · {int((pg295_report.get('split') or {}).get('seed_missing', 0) or 0)} missing seed holdout · causal next-token · local morning",
        "seed": 29501,
        "method": "decoder-only causal LM + top-k MoE; answer-only matched control",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"selected={str(pg295_selection.get('config_name', 'not_recorded'))} · seed missing-question recall={float(pg295_seed_missing.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"hard-negative false-allow={int(pg295_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)} · answer-only missing-question recall={float(pg295_control_missing.get('missing_question_recall', {}).get('mean', 0.0) or 0.0) * 100:.0f}%",
            f"engineering={str(pg295_engineering.get('status', 'blocked'))} · scientific={str(pg295_scientific.get('status', 'blocked'))}",
            f"device={str(pg295_device.get('device_name', 'not_recorded'))} · Docker/evaluator not used",
        ],
        "evidence_hash": str(pg295_report.get("report_sha256", ""))[:16],
        "instruction": "只把缺失观测当成 question token 的训练目标；不能用 answer-only accuracy 证明会主动排错。hard-negative false-allow 未归零前不得晋级。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg295"] = {
        "status": str(pg295_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe",
        "selected_config": str(pg295_selection.get("config_name", "not_recorded")),
        "experts": int((pg295_selected.get("config") or {}).get("experts", 0) or 0),
        "layers": int((pg295_selected.get("config") or {}).get("n_layers", 0) or 0),
        "d_model": int((pg295_selected.get("config") or {}).get("d_model", 0) or 0),
        "causal_next_token_only": True,
        "seed_missing_question_recall": float(pg295_seed_missing.get("missing_question_recall", {}).get("min", 0.0) or 0.0),
        "hard_negative_false_allow": int(pg295_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "answer_only_missing_question_recall": float(pg295_control_missing.get("missing_question_recall", {}).get("mean", 0.0) or 0.0),
        "engineering_gate_status": str(pg295_engineering.get("status", "blocked")),
        "scientific_gate_status": str(pg295_scientific.get("status", "blocked")),
        "device": str(pg295_device.get("device_name", "not_recorded")),
        "wire_emission_allowed": False,
        "literal_payload_generation": False,
        "promotion_blocked": True,
        "evidence_hash": str(pg295_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg295", "label": "PG-295 causal MoE 提问", "value": f"{float(pg295_seed_missing.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% / {int(pg295_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)}" if pg295_ready else "PENDING", "status": "blocked", "note": f"answer-only={float(pg295_control_missing.get('missing_question_recall', {}).get('mean', 0.0) or 0.0) * 100:.0f}% · causal next-token + MoE · promotion blocked"})
    if pg295_ready:
        snapshot["capability"]["limits"].append(f"PG-295 causal MoE 在 seed missing holdout 上 question recall={float(pg295_seed_missing.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%，answer-only 对照={float(pg295_control_missing.get('missing_question_recall', {}).get('mean', 0.0) or 0.0) * 100:.0f}%，说明最终答案数据不能替代主动提问；但 hard-negative false-allow={int(pg295_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)} 且没有真实 evaluator，不能声称 payload 成功。")
    pg300_metrics = dict(pg300_report.get("metrics") or {})
    pg300_holdout = dict(pg300_metrics.get("implementation_holdout") or {})
    pg300_hard = dict(pg300_metrics.get("hard_negative") or {})
    pg300_engineering = dict(pg300_report.get("engineering_gate") or {})
    pg300_ready = pg300_report.get("status") == "completed_local_morning_pg300_question_policy"
    snapshot["tasks"]["trainer"].append({
        "id": "pg300-question-policy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-300 causal question-policy composition",
        "route": f"{int((pg300_report.get('split') or {}).get('train', 0) or 0)} train · {int((pg300_report.get('split') or {}).get('implementation_holdout', 0) or 0)} OOD · question-only causal next-token · local morning",
        "seed": 30001,
        "method": "canonical observation slots + surface holdout + token-level cost",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"OOD question recall={float(pg300_holdout.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"hard-negative false-allow={int(pg300_hard.get('hard_negative_false_allow', {}).get('max', 0) or 0)} · unnecessary-question={float(pg300_hard.get('unnecessary_question_rate', {}).get('max', 0.0) or 0.0) * 100:.0f}%",
            f"engineering={str(pg300_engineering.get('status', 'blocked'))} · no typed evaluator",
        ],
        "evidence_hash": str(pg300_report.get("report_sha256", ""))[:16],
        "instruction": "缺 slot 时问对应观测；已知 slot 不得无意义追问。两项都要跨 seed 通过，随后才能进入抽象 transport/field/encoding 组装。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg300"] = {
        "status": str(pg300_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_question_only",
        "dataset_count": int((pg300_report.get("split") or {}).get("total", 0) or 0),
        "implementation_holdout_count": int((pg300_report.get("split") or {}).get("implementation_holdout", 0) or 0),
        "question_recall_min": float(pg300_holdout.get("missing_question_recall", {}).get("min", 0.0) or 0.0),
        "hard_negative_false_allow_max": int(pg300_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "hard_negative_unnecessary_question_max": float(pg300_hard.get("unnecessary_question_rate", {}).get("max", 0.0) or 0.0),
        "engineering_gate_status": str(pg300_engineering.get("status", "blocked")),
        "scientific_gate_status": str((pg300_report.get("scientific_gate") or {}).get("status", "blocked")),
        "promotion_blocked": True,
        "evidence_hash": str(pg300_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg300", "label": "PG-300 主动提问组合", "value": f"{float(pg300_holdout.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% / {float(pg300_hard.get('unnecessary_question_rate', {}).get('max', 0.0) or 0.0) * 100:.0f}%" if pg300_ready else "PENDING", "status": "blocked", "note": "question recall 与 unnecessary-question 双门；未接 typed evaluator"})
    if pg300_ready:
        snapshot["capability"]["limits"].append(f"PG-300 OOD question recall={float(pg300_holdout.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%，但 hard-negative unnecessary-question={float(pg300_hard.get('unnecessary_question_rate', {}).get('max', 0.0) or 0.0) * 100:.0f}%；工程门阻塞，不能进入 payload 组装或漏洞声明。")
    def _assembly_report_view(report_obj: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        metrics_obj = dict(report_obj.get("metrics") or {})
        hold_obj = dict(metrics_obj.get("implementation_holdout") or {})
        hard_obj = dict(metrics_obj.get("hard_negative") or {})
        causal_obj = dict(hold_obj.get("causal") or hold_obj.get("causal_symbolic") or {})
        bound_obj = dict(hold_obj.get("assembly") or hold_obj.get("bound_abstract") or {})
        return causal_obj, bound_obj, hard_obj
    pg301_causal, pg301_bound, pg301_hard = _assembly_report_view(pg301_report)
    pg302_causal, pg302_bound, pg302_hard = _assembly_report_view(pg302_report)
    pg302b_causal, pg302b_bound, pg302b_hard = _assembly_report_view(pg302b_report)
    pg301_ready = pg301_report.get("status") == "completed_local_morning_pg301_payload_assembly"
    pg302_ready = pg302_report.get("status") == "completed_local_morning_pg302_symbolic_assembly"
    pg302b_ready = pg302b_report.get("status") == "completed_local_morning_pg302b_symbolic_curriculum"
    pg303_lanes = dict(pg303_report.get("lanes") or {})
    pg303_hold_guard = dict(dict(pg303_lanes.get("implementation_holdout") or {}).get("guarded") or {})
    pg303_hard_guard = dict(dict(pg303_lanes.get("hard_negative_eval") or {}).get("guarded") or {})
    pg303_ready = pg303_report.get("status") == "completed_local_morning_pg303_guarded_eval"
    snapshot["tasks"]["trainer"].append({
        "id": "pg301-302-assembly-composition",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-301/302 abstract payload assembly",
        "route": f"PG-301 {int((pg301_report.get('split') or {}).get('train', 0) or 0)} train · PG-302 symbolic · local morning",
        "seed": 30101,
        "method": "causal Transformer-MoE + symbolic slot references + deterministic binder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"PG-301 holdout question={float(pg301_causal.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% · slot exact={float(pg301_bound.get('assembly_slot_exact', {}).get('min', 0.0) or 0.0) * 100:.0f}%",
            f"PG-302 holdout question={float(pg302_causal.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% · unnecessary-question={float(pg302_causal.get('unnecessary_question_rate', {}).get('max', 0.0) or 0.0) * 100:.0f}%",
            f"PG-302B curriculum holdout question={float(pg302b_causal.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% · false-allow={int(pg302b_causal.get('hard_negative_false_allow', {}).get('max', 0) or 0)}",
            "no typed evaluator or literal payload",
        ],
        "evidence_hash": str(pg302b_report.get("report_sha256") or pg302_report.get("report_sha256") or pg301_report.get("report_sha256", ""))[:16],
        "instruction": "神经模型只生成抽象 slot/reference；缺观测先问，失败只 repair/abstain。raw、bound、guarded 指标必须分开，不得把 guard 结果当神经能力。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg301"] = {
        "status": str(pg301_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_abstract_assembly",
        "holdout_question_recall_min": float(pg301_causal.get("missing_question_recall", {}).get("min", 0.0) or 0.0),
        "holdout_assembly_slot_exact_min": float(pg301_bound.get("assembly_slot_exact", {}).get("min", 0.0) or 0.0),
        "hard_negative_false_allow_max": int(pg301_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "engineering_gate_status": str((pg301_report.get("engineering_gate") or {}).get("status", "blocked")),
        "promotion_blocked": True,
        "evidence_hash": str(pg301_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg302"] = {
        "status": str(pg302_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_symbolic_slot_reference",
        "holdout_question_recall_min": float(pg302_causal.get("missing_question_recall", {}).get("min", 0.0) or 0.0),
        "holdout_bound_assembly_slot_exact_min": float(pg302_bound.get("assembly_slot_exact", {}).get("min", 0.0) or 0.0),
        "holdout_unnecessary_question_max": float(pg302_causal.get("unnecessary_question_rate", {}).get("max", 0.0) or 0.0),
        "hard_negative_false_allow_max": int(pg302_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "engineering_gate_status": str((pg302_report.get("engineering_gate") or {}).get("status", "blocked")),
        "promotion_blocked": True,
        "evidence_hash": str(pg302_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg302b"] = {
        "status": str(pg302b_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_question_pretrain_plus_symbolic_sft",
        "holdout_question_recall_min": float(pg302b_causal.get("missing_question_recall", {}).get("min", 0.0) or 0.0),
        "holdout_bound_assembly_slot_exact_min": float(pg302b_bound.get("assembly_slot_exact", {}).get("min", 0.0) or 0.0),
        "holdout_false_allow_max": int(pg302b_causal.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "hard_negative_false_allow_max": int(pg302b_hard.get("hard_negative_false_allow", {}).get("max", 0) or 0),
        "engineering_gate_status": str((pg302b_report.get("engineering_gate") or {}).get("status", "blocked")),
        "promotion_blocked": True,
        "evidence_hash": str(pg302b_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg303"] = {
        "status": str(pg303_report.get("status", "not_run")),
        "architecture": "neural_symbolic_proposal_plus_visible_slot_guard",
        "raw_holdout_missing_question_recall": float(dict(dict(pg303_lanes.get("implementation_holdout") or {}).get("raw_bound") or {}).get("missing_question_recall", 0.0) or 0.0),
        "guarded_holdout_missing_question_recall": float(pg303_hold_guard.get("missing_question_recall", 0.0) or 0.0),
        "guarded_holdout_slot_exact": float(pg303_hold_guard.get("assembly_slot_exact", 0.0) or 0.0),
        "guarded_hard_negative_false_allow": int(pg303_hard_guard.get("hard_negative_false_allow", 0) or 0),
        "guarded_hard_negative_unnecessary_question": float(pg303_hard_guard.get("unnecessary_question_rate", 0.0) or 0.0),
        "engineering_gate_status": str((pg303_report.get("engineering_gate") or {}).get("status", "blocked")),
        "neural_claim_allowed": False,
        "promotion_blocked": True,
        "evidence_hash": str(pg303_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg301", "label": "PG-301 抽象组装", "value": f"{float(pg301_causal.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% / {float(pg301_bound.get('assembly_slot_exact', {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg301_ready else "PENDING", "status": "blocked", "note": "question/slot 双门；无 typed evaluator"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg302", "label": "PG-302 symbolic slot", "value": f"{float(pg302_causal.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}% / {float(pg302_bound.get('assembly_slot_exact', {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg302_ready else "PENDING", "status": "blocked", "note": "raw/绑定分开；无 typed evaluator"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg303", "label": "PG-303 guarded composer", "value": f"{float(pg303_hold_guard.get('missing_question_recall', 0.0) or 0.0) * 100:.0f}% / {int(pg303_hard_guard.get('hard_negative_false_allow', 0) or 0)}" if pg303_ready else "PENDING", "status": "blocked", "note": "guarded safety ≠ neural capability"})
    if pg301_ready:
        snapshot["capability"]["limits"].append(f"PG-301 族外 question recall 最差={float(pg301_causal.get('missing_question_recall', {}).get('min', 0.0) or 0.0) * 100:.0f}%、slot exact 最差={float(pg301_bound.get('assembly_slot_exact', {}).get('min', 0.0) or 0.0) * 100:.0f}%，神经组装未过门。")
    if pg303_ready:
        snapshot["capability"]["limits"].append(f"PG-303 guard 把缺观测提问召回提高到 {float(pg303_hold_guard.get('missing_question_recall', 0.0) or 0.0) * 100:.0f}%、hard false-allow={int(pg303_hard_guard.get('hard_negative_false_allow', 0) or 0)}，但 guarded slot exact={float(pg303_hold_guard.get('assembly_slot_exact', 0.0) or 0.0) * 100:.0f}%，且 guard 不能算神经能力。")
    pg304_metrics = dict(pg304_report.get("metrics") or {})
    pg304_checks = dict(pg304_report.get("checks") or {})
    pg304_ready = pg304_report.get("status") == "completed_loopback_evaluator_only"
    snapshot["tasks"]["reviewer"].append({
        "id": "pg304-loopback-replay-contract",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "ready_for_authorized_adapter" if pg304_ready else "promotion_blocked",
        "label": "PG-304 evaluator-only loopback replay",
        "route": f"{int(pg304_metrics.get('episode_count', 0) or 0)} fixture episodes · GET/POST pair · no wire",
        "seed": 30401,
        "method": "typed projection contract",
        # PG-304 only exercises the evaluator contract with fixture
        # projections.  Keep this false so the UI cannot turn fixture
        # positives into a real application-effect claim.
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"fixture typed positives={int(pg304_metrics.get('typed_positive_count', 0) or 0)} · blocked={int(pg304_metrics.get('blocked_count', 0) or 0)}",
            f"loopback={bool(pg304_checks.get('loopback_only'))} · external_network_disabled={bool(pg304_checks.get('external_network_disabled'))}",
            "fixture-only; no real Docker contact; promotion closed",
        ],
        "evidence_hash": str(pg304_report.get("batch_evidence_sha256") or pg304_report.get("report_sha256", ""))[:16],
        "instruction": "只有得到明确授权的本地 Docker evaluator 才能替换 fixture projection；验收 fresh reset、GET/POST 正负 replay、typed evidence hash 后仍需人工审核。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg304"] = {
        "status": str(pg304_report.get("status", "not_run")),
        "architecture": "evaluator_only_loopback_contract",
        "fixture_episode_count": int(pg304_metrics.get("episode_count", 0) or 0),
        "fixture_typed_positive_count": int(pg304_metrics.get("typed_positive_count", 0) or 0),
        "fixture_blocked_count": int(pg304_metrics.get("blocked_count", 0) or 0),
        "get_post_pair": bool((pg304_report.get("pair_contract") or {}).get("get_post_pair")),
        "loopback_only": bool(pg304_checks.get("loopback_only")),
        "external_network_disabled": bool(pg304_checks.get("external_network_disabled")),
        "training_eligible_count": int(pg304_metrics.get("training_eligible_count", 0) or 0),
        "memory_promotion_allowed_count": int(pg304_metrics.get("memory_promotion_allowed_count", 0) or 0),
        "engineering_gate_status": str((pg304_report.get("engineering_gate") or {}).get("status", "blocked")),
        "scientific_gate_status": str((pg304_report.get("scientific_gate") or {}).get("status", "blocked")),
        "promotion_blocked": True,
        "evidence_hash": str(pg304_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg304", "label": "PG-304 loopback typed contract", "value": f"{int(pg304_metrics.get('typed_positive_count', 0) or 0)}+ / {int(pg304_metrics.get('blocked_count', 0) or 0)} blocked" if pg304_ready else "PENDING", "status": "blocked", "note": "fixture-only; no training/memory promotion"})
    if pg304_ready:
        snapshot["capability"]["limits"].append(f"PG-304 fixture contract 覆盖 {int(pg304_metrics.get('typed_positive_count', 0) or 0)} 个 positive 和 {int(pg304_metrics.get('blocked_count', 0) or 0)} 个 blocked，但没有真实 Docker contact，不能作为漏洞或 payload 成功证据。")
    pg305_counts = dict(pg305_report.get("counts") or {})
    pg305_checks = dict(pg305_report.get("checks") or {})
    pg305_preflight = dict(pg305_report.get("preflight_identifiability") or {})
    pg305_ready = pg305_report.get("status") == "completed_real_local_docker_evaluator"
    snapshot["tasks"]["reviewer"].append({
        "id": "pg305-live-loopback-evaluator",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "ready_for_human_review" if pg305_ready else "promotion_blocked",
        "label": "PG-305 real local Docker GET/POST evaluator",
        "route": f"{int(pg305_counts.get('route_count', 0) or 0)} routes · GET={int(pg305_counts.get('get_count', 0) or 0)} / POST={int(pg305_counts.get('post_count', 0) or 0)}",
        "seed": 30501,
        "method": "fresh reset + negative/reference/candidate + typed local oracle",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"evaluator gold typed effects={int(pg305_counts.get('evaluator_gold_typed_effect_count', 0) or 0)} · model sends={int(pg305_counts.get('model_candidate_send_count', 0) or 0)} · model effects={int(pg305_counts.get('model_confirmed_effect_count', 0) or 0)}",
            f"model abstain={int(pg305_counts.get('model_abstain_count', 0) or 0)} · false positive={int(pg305_counts.get('false_positive_count', 0) or 0)}",
            f"fresh reset={int(pg305_counts.get('fresh_reset_count', 0) or 0)} · negative controls={int(pg305_counts.get('negative_control_count', 0) or 0)}",
            f"loopback={bool(pg305_checks.get('loopback_only'))} · external_network_disabled={bool(pg305_checks.get('external_network_disabled'))}",
            f"preflight question recall raw={float(dict(pg305_preflight.get('raw') or {}).get('missing_question_recall', 0.0) or 0.0) * 100:.0f}% · guarded={float(dict(pg305_preflight.get('guarded') or {}).get('missing_question_recall', 0.0) or 0.0) * 100:.0f}%",
            "evaluator gold is not model capability; payload/memory/vulnerability promotion remains closed",
        ],
        "evidence_hash": str(pg305_report.get("report_sha256", ""))[:16],
        "instruction": "人工只在授权 loopback catalog 中查看 wire 与回显；分别审查 evaluator gold 和模型是否真实发包，不能把 abstain 或参考答案算成模型成功。",
        # The catalog exists for a separately authorized human-review surface,
        # but this research snapshot must not expose raw wire material.
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg305"] = {
        "status": str(pg305_report.get("status", "not_run")),
        "architecture": str((pg305_report.get("model") or {}).get("architecture", "causal_transformer_moe_next_token")),
        "route_count": int(pg305_counts.get("route_count", 0) or 0),
        "get_count": int(pg305_counts.get("get_count", 0) or 0),
        "post_count": int(pg305_counts.get("post_count", 0) or 0),
        "model_candidate_send_count": int(pg305_counts.get("model_candidate_send_count", 0) or 0),
        "model_confirmed_effect_count": int(pg305_counts.get("model_confirmed_effect_count", 0) or 0),
        "evaluator_gold_typed_effect_count": int(pg305_counts.get("evaluator_gold_typed_effect_count", 0) or 0),
        "model_abstain_count": int(pg305_counts.get("model_abstain_count", 0) or 0),
        "false_positive_count": int(pg305_counts.get("false_positive_count", 0) or 0),
        "preflight_raw_missing_question_recall": float(dict(pg305_preflight.get("raw") or {}).get("missing_question_recall", 0.0) or 0.0),
        "preflight_guarded_missing_question_recall": float(dict(pg305_preflight.get("guarded") or {}).get("missing_question_recall", 0.0) or 0.0),
        "loopback_only": bool(pg305_checks.get("loopback_only")),
        "external_network_disabled": bool(pg305_checks.get("external_network_disabled")),
        "promotion_blocked": True,
        "evidence_hash": str(pg305_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg305", "label": "PG-305 真实 loopback evaluator", "value": f"gold {int(pg305_counts.get('evaluator_gold_typed_effect_count', 0) or 0)} / model send {int(pg305_counts.get('model_candidate_send_count', 0) or 0)}" if pg305_ready else "PENDING", "status": "blocked", "note": "真实 evaluator 已接通，但 gold 不等于模型能力；promotion 关闭"})
    if pg305_ready:
        snapshot["capability"]["limits"].append(f"PG-305 在 {int(pg305_counts.get('route_count', 0) or 0)} 个本地 Docker 路由完成 GET/POST 正负复放，evaluator gold={int(pg305_counts.get('evaluator_gold_typed_effect_count', 0) or 0)}；模型实际发包={int(pg305_counts.get('model_candidate_send_count', 0) or 0)}，因此当前只能证明 evaluator 工程闭环，不能声称 AI 会生成或确认 payload。")

    pg306_metrics = dict(pg306_report.get("metrics") or {})
    pg306b_metrics = dict(pg306b_report.get("metrics") or {})
    pg306c_metrics = dict(pg306c_report.get("metrics") or {})
    pg307_metrics = dict(pg307_report.get("metrics") or {})
    pg307_training = dict(pg307_report.get("training") or {})
    pg307_gate = dict(pg307_report.get("hypothesis_gate") or {})
    pg307_ready = pg307_report.get("status") == "completed_local_morning_pg307_symbolic_slot_copy"
    snapshot["tasks"]["trainer"].append({
        "id": "pg306-307-process-symbolic-ablation",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-306/307 过程轨迹与 symbolic slot-copy",
        "route": f"PG-307 train={int(pg307_training.get('train_count', 0) or 0)} · implementation/live holdout={int(pg307_training.get('holdout_count', 0) or 0)} · hard-negative={int(pg307_training.get('hard_negative_count', 0) or 0)} · local morning CPU",
        "seed": 30701,
        "method": "question pretrain → causal next-token symbolic slot refs → deterministic binder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"PG-306 baseline question min={float(dict(pg306_metrics.get('implementation_and_live_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · slot min={float(dict(pg306_metrics.get('implementation_and_live_holdout_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"PG-306B curriculum question min={float(dict(pg306b_metrics.get('implementation_and_live_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · slot min={float(dict(pg306b_metrics.get('implementation_and_live_holdout_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"PG-306C positive oversample hard false-allow max={float(dict(pg306c_metrics.get('hard_negative_false_allow') or {}).get('max', 0.0) or 0.0):.0f} · slot min={float(dict(pg306c_metrics.get('implementation_and_live_holdout_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"PG-307 question min={float(dict(pg307_metrics.get('implementation_and_live_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · bound slot min={float(dict(pg307_metrics.get('implementation_and_live_holdout_bound_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"PG-307 unnecessary-question max={float(dict(pg307_metrics.get('implementation_and_live_holdout_unnecessary_question_rate') or {}).get('max', 0.0) or 0.0) * 100:.1f}% · bound false-allow max={float(dict(pg307_metrics.get('hard_negative_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}",
            f"hypothesis gate={str(pg307_gate.get('status', 'blocked'))} · neural/payload promotion remains closed",
        ],
        "evidence_hash": str(pg307_report.get("report_sha256", ""))[:16],
        "instruction": "下一轮必须针对 unnecessary-question 与 slot-copy 失败做反事实/多源 holdout；不能用 guard 或 binder 的正确性冒充神经模型已经会发包。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg306"] = {
        "status": str(pg306_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_process_assembly",
        "question_recall_min": float(dict(pg306_metrics.get("implementation_and_live_holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "slot_exact_min": float(dict(pg306_metrics.get("implementation_and_live_holdout_assembly_slot_exact") or {}).get("min", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg306_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg306b"] = {
        "status": str(pg306b_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_question_curriculum",
        "question_recall_min": float(dict(pg306b_metrics.get("implementation_and_live_holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "slot_exact_min": float(dict(pg306b_metrics.get("implementation_and_live_holdout_assembly_slot_exact") or {}).get("min", 0.0) or 0.0),
        "unnecessary_question_max": float(dict(pg306b_metrics.get("implementation_and_live_holdout_unnecessary_question_rate") or {}).get("max", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg306b_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg306c"] = {
        "status": str(pg306c_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_balanced_positive_curriculum",
        "question_recall_min": float(dict(pg306c_metrics.get("implementation_and_live_holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "slot_exact_min": float(dict(pg306c_metrics.get("implementation_and_live_holdout_assembly_slot_exact") or {}).get("min", 0.0) or 0.0),
        "hard_negative_false_allow_max": float(dict(pg306c_metrics.get("hard_negative_false_allow") or {}).get("max", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg306c_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg307"] = {
        "status": str(pg307_report.get("status", "not_run")),
        "architecture": "causal_transformer_moe_symbolic_slot_copy",
        "question_recall_min": float(dict(pg307_metrics.get("implementation_and_live_holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "raw_sequence_exact_mean": float(dict(pg307_metrics.get("implementation_and_live_holdout_raw_sequence_exact") or {}).get("mean", 0.0) or 0.0),
        "bound_slot_exact_min": float(dict(pg307_metrics.get("implementation_and_live_holdout_bound_assembly_slot_exact") or {}).get("min", 0.0) or 0.0),
        "unnecessary_question_max": float(dict(pg307_metrics.get("implementation_and_live_holdout_unnecessary_question_rate") or {}).get("max", 0.0) or 0.0),
        "bound_false_allow_max": float(dict(pg307_metrics.get("hard_negative_bound_false_allow") or {}).get("max", 0.0) or 0.0),
        "symbolic_slot_copy": True,
        "deterministic_binder": "pg302",
        "dataset_count": int(pg307_dataset.get("counts", {}).get("total", 0) or 0),
        "dataset_audit_status": str(pg307_audit.get("status", "not_run")),
        "promotion_blocked": True,
        "evidence_hash": str(pg307_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg307", "label": "PG-307 symbolic slot-copy", "value": f"question {float(dict(pg307_metrics.get('implementation_and_live_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.0f}% / bound slot {float(dict(pg307_metrics.get('implementation_and_live_holdout_bound_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg307_ready else "PENDING", "status": "blocked", "note": f"unnecessary-question max={float(dict(pg307_metrics.get('implementation_and_live_holdout_unnecessary_question_rate') or {}).get('max', 0.0) or 0.0) * 100:.0f}%; raw model not a payload generator"})
    if pg307_ready:
        snapshot["capability"]["limits"].append(f"PG-307 symbolic slot-copy 将缺观测 question recall 最差提高到 {float(dict(pg307_metrics.get('implementation_and_live_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}%，但 bound slot exact 最差只有 {float(dict(pg307_metrics.get('implementation_and_live_holdout_bound_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%，unnecessary-question 最高 {float(dict(pg307_metrics.get('implementation_and_live_holdout_unnecessary_question_rate') or {}).get('max', 0.0) or 0.0) * 100:.1f}%；这证明‘会问一部分问题’不等于‘会可靠组装或会发包’。")
    pg308_metrics = dict(pg308_report.get("metrics") or {})
    pg309_metrics = dict(pg309_report.get("metrics") or {})
    pg310_variants = dict(pg310_report.get("variants") or {})
    pg310_wide = dict(pg310_variants.get("wide_zero_dropout") or {})
    pg310_wide_metrics = dict(pg310_wide.get("metrics") or {})
    pg311_metrics = dict(pg311_report.get("metrics") or {})
    pg312_counts = dict(pg312_report.get("counts") or {})
    pg312_checks = dict(pg312_report.get("checks") or {})
    pg312_model = dict(pg312_report.get("model") or {})
    pg312_ready = pg312_report.get("status") == "completed_real_local_docker_evaluator"
    pg313_metrics = dict(pg313_report.get("metrics") or {})
    pg313_gate = dict(pg313_report.get("hypothesis_gate") or {})
    pg313_gate_checks = dict(pg313_gate.get("checks") or {})
    pg313_training = dict(pg313_report.get("training") or {})
    pg313_ready = pg313_report.get("status") == "completed_local_morning_pg313_probe_variant"
    pg314_counts = dict(pg314_report.get("counts") or {})
    pg314_checks = dict(pg314_report.get("checks") or {})
    pg314_model = dict(pg314_report.get("model") or {})
    pg314_gate = dict(pg314_report.get("hypothesis_gate") or {})
    pg314_gate_checks = dict(pg314_gate.get("checks") or {})
    pg314_ready = pg314_report.get("status") == "completed_real_local_docker_independent_variant_replay"
    pg315_counts = dict(pg315_report.get("counts") or {})
    pg315_worst = dict(pg315_report.get("worst_seed_metrics") or {})
    pg315_gate = dict(pg315_report.get("hypothesis_gate") or {})
    pg315_gate_checks = dict(pg315_gate.get("checks") or {})
    pg315_ready = pg315_report.get("status") == "completed_real_local_docker_all_seed_replay"
    pg316_metrics = dict(pg316_report.get("metrics") or {})
    pg316_gate = dict(pg316_report.get("hypothesis_gate") or {})
    pg316_gate_checks = dict(pg316_gate.get("checks") or {})
    pg316_ready = pg316_report.get("status") == "completed_local_morning_pg316_failure_repair"
    pg316_live_counts = dict(pg316_live_report.get("counts") or {})
    pg316_live_gate = dict(pg316_live_report.get("hypothesis_gate") or {})
    pg316_live_ready = pg316_live_report.get("status") == "completed_real_local_docker_pg316_live_replay"
    pg317_metrics = dict(pg317_report.get("metrics") or {})
    pg317_gate = dict(pg317_report.get("hypothesis_gate") or {})
    pg317_gate_checks = dict(pg317_gate.get("checks") or {})
    pg317_ready = pg317_report.get("status") == "completed_local_morning_pg317_question_anchor"
    pg317_live_counts = dict(pg317_live_report.get("counts") or {})
    pg317_live_gate = dict(pg317_live_report.get("hypothesis_gate") or {})
    pg317_live_ready = pg317_live_report.get("status") == "completed_real_local_docker_pg317_live_replay"
    pg318_counts = dict(pg318_report.get("counts") or {})
    pg318_worst = dict(pg318_report.get("worst_seed_metrics") or {})
    pg318_checks = dict(pg318_report.get("checks") or {})
    pg318_gate = dict(pg318_report.get("hypothesis_gate") or {})
    pg318_gate_checks = dict(pg318_gate.get("checks") or {})
    pg318_ready = pg318_report.get("status") == "completed_real_local_docker_pg318_family_holdout"
    pg319_metrics = dict(pg319_report.get("metrics") or {})
    pg319_gate = dict(pg319_report.get("hypothesis_gate") or {})
    pg319_ready = pg319_report.get("status") == "completed_local_morning_pg319_cross_impl_moe"
    pg320_metrics = dict(pg320_report.get("metrics") or {})
    pg320_gate = dict(pg320_report.get("hypothesis_gate") or {})
    pg320_live_counts = dict(pg320_live_report.get("counts") or {})
    pg320_live_worst = dict(pg320_live_report.get("worst_seed_metrics") or {})
    pg320_live_ready = pg320_live_report.get("status") == "completed_real_local_docker_pg320_family_holdout"
    pg321_metrics = dict(pg321_report.get("metrics") or {})
    pg321_gate = dict(pg321_report.get("hypothesis_gate") or {})
    pg321_live_counts = dict(pg321_live_report.get("counts") or {})
    pg321_live_worst = dict(pg321_live_report.get("worst_seed_metrics") or {})
    pg321_live_checks = dict(pg321_live_report.get("checks") or {})
    pg321_live_gate = dict(pg321_live_report.get("hypothesis_gate") or {})
    pg321_ready = pg321_report.get("status") == "completed_local_morning_pg321_variant_role"
    pg321_live_ready = pg321_live_report.get("status") == "completed_real_local_docker_pg321_family_holdout"
    pg322_metrics = dict(pg322_report.get("metrics") or {})
    pg322_gate = dict(pg322_report.get("hypothesis_gate") or {})
    pg322_ready = pg322_report.get("status") == "completed_local_morning_pg322_cross_impl_decoy"
    pg323_metrics = dict(pg323_report.get("metrics") or {})
    pg323_gate = dict(pg323_report.get("hypothesis_gate") or {})
    pg323_live_counts = dict(pg323_live_report.get("counts") or {})
    pg323_live_worst = dict(pg323_live_report.get("worst_seed_metrics") or {})
    pg323_live_checks = dict(pg323_live_report.get("checks") or {})
    pg323_live_gate = dict(pg323_live_report.get("hypothesis_gate") or {})
    pg323_ready = pg323_report.get("status") == "completed_local_morning_pg323_decoy_ask_anchor"
    pg323_live_ready = pg323_live_report.get("status") == "completed_real_local_docker_pg323_vulnerableapp_role_replay"
    snapshot["tasks"]["trainer"].append({
        "id": "pg308-multisource-slot-copy",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-308 多源过程与排列 hard-negative",
        "route": f"{int(pg308_report.get('training', {}).get('train_count', 0) or 0)} train · {int(pg308_report.get('training', {}).get('holdout_count', 0) or 0)} source holdout · {int(pg308_report.get('training', {}).get('hard_negative_count', 0) or 0)} hard",
        "seed": 30801,
        "method": "causal Transformer-MoE symbolic slot-copy",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"question min={float(dict(pg308_metrics.get('source_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · bound slot min={float(dict(pg308_metrics.get('source_holdout_bound_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"holdout bound false-allow max={float(dict(pg308_metrics.get('source_holdout_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f} · hard max={float(dict(pg308_metrics.get('hard_negative_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}",
            f"slot permutation false-allow={float(pg308_metrics.get('slot_permutation_bound_false_allow', 0.0) or 0.0):.0f}; cross-source gate blocked",
        ],
        "evidence_hash": str(pg308_report.get("report_sha256", ""))[:16],
        "instruction": "保留 PG-308 的失败：一部分 seed 全拒答，另一部分 seed 误放；不能用均值掩盖 worst seed。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg309-counterfactual-balance",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-309 缺观测/完整/失败/错位成对反事实",
        "route": f"{int(pg309_report.get('training', {}).get('train_count', 0) or 0)} train · same PG-308 holdout",
        "seed": 30901,
        "method": "causal next-token + question anchor + symbolic binder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"question min={float(dict(pg309_metrics.get('source_holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · bound slot min={float(dict(pg309_metrics.get('source_holdout_bound_assembly_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"unnecessary max={float(dict(pg309_metrics.get('source_holdout_unnecessary_question_rate') or {}).get('max', 0.0) or 0.0) * 100:.1f}% · hard bound false-allow max={float(dict(pg309_metrics.get('hard_negative_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}",
            "安全性恢复但跨 seed/positive coverage 未过门",
        ],
        "evidence_hash": str(pg309_report.get("report_sha256", ""))[:16],
        "instruction": "反事实只能训练过程识别，不能被当成真实漏洞 gold；仍需 fresh target 复放。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg310-311-wide-moe-ablation",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-310/311 宽 MoE 优化与 question anchor",
        "route": "same PG-309 source holdout · 3 seeds each",
        "seed": 31101,
        "method": "wide zero-dropout causal Transformer-MoE",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"PG-310 wide question min={float(dict(pg310_wide_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · slot min={float(dict(pg310_wide_metrics.get('holdout_bound_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"PG-311 question min={float(dict(pg311_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · slot min={float(dict(pg311_metrics.get('holdout_bound_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"PG-311 raw/bound false-allow max={float(dict(pg311_metrics.get('holdout_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}/{float(dict(pg311_metrics.get('hard_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}",
        ],
        "evidence_hash": str(pg311_report.get("report_sha256", ""))[:16],
        "instruction": "宽度带来组合槽位稳定性，但 question worst seed 仍为 88%；不得按 best seed 晋级。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg312-live-wide-checkpoint",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "ready_for_human_review" if pg312_ready else "promotion_blocked",
        "label": "PG-312 宽 symbolic checkpoint 真实 loopback 复放",
        "route": f"{int(pg312_counts.get('route_count', 0) or 0)} routes · GET={int(pg312_counts.get('get_count', 0) or 0)} / POST={int(pg312_counts.get('post_count', 0) or 0)}",
        "seed": 31201,
        "method": "model abstract plan → source-grounded local candidate → typed oracle",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"model candidate sends={int(pg312_counts.get('model_candidate_send_count', 0) or 0)} · model typed effects={int(pg312_counts.get('model_confirmed_effect_count', 0) or 0)} · evaluator gold={int(pg312_counts.get('evaluator_gold_typed_effect_count', 0) or 0)}",
            f"fresh reset={int(pg312_counts.get('fresh_reset_count', 0) or 0)} · negatives={int(pg312_counts.get('negative_control_count', 0) or 0)} · false positive={int(pg312_counts.get('false_positive_count', 0) or 0)}",
            f"loopback={bool(pg312_checks.get('loopback_only'))} · evidence hashes={bool(pg312_checks.get('typed_evidence_hash_per_route'))} · symbolic={bool(pg312_model.get('symbolic_checkpoint'))}",
            "wire is source-grounded adapter output, not literal payload generated by the neural decoder",
        ],
        "evidence_hash": str(pg312_report.get("report_sha256", ""))[:16],
        "instruction": "人工查看 PG-312 catalog 的真实 wire/回显；把‘模型触发了受控候选发送’与‘模型自己生成了 payload’分开判定。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg313-probe-variant-moe",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-313 probe_variant / encoding_chain next-token",
        "route": f"{int(pg313_training.get('fit_count', 0) or 0)} train · {int(pg313_training.get('holdout_count', 0) or 0)} holdout · {int(pg313_training.get('hard_negative_count', 0) or 0)} hard",
        "seed": 31301,
        "method": "causal Transformer-MoE abstract probe-variant decoder",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"question min={float(dict(pg313_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · base slot min={float(dict(pg313_metrics.get('holdout_bound_base_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"variant exact min={float(dict(pg313_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · unnecessary max={float(dict(pg313_metrics.get('holdout_unnecessary_question') or {}).get('max', 0.0) or 0.0) * 100:.1f}%",
            f"holdout false-allow max={float(dict(pg313_metrics.get('holdout_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f} · hard max={float(dict(pg313_metrics.get('hard_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}",
            "cross-seed worst-case gate blocked; no literal payload or live wire was generated",
        ],
        "evidence_hash": str(pg313_report.get("report_sha256", ""))[:16],
        "instruction": "PG-313 证明模型开始看到抽象 probe variant/encoding chain，但 seed 31301/31303 的 hard-negative 和 base slot 不稳定；不得按 best seed 晋级，下一步在第二独立实现复放并保持 zero false-allow。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg314-independent-variant-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-314 第二独立实现 variant 复放",
        "route": f"{int(pg314_counts.get('route_count', 0) or 0)} routes · GET={int(pg314_counts.get('get_count', 0) or 0)} / POST={int(pg314_counts.get('post_count', 0) or 0)}",
        "seed": 31401,
        "method": "PG-313 next-token abstract variant → isolated Docker loopback",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"variant exact={int(pg314_counts.get('model_variant_exact_count', 0) or 0)}/{int(pg314_counts.get('model_variant_role_count', 0) or 0)} · model sends={int(pg314_counts.get('model_variant_send_count', 0) or 0)}",
            f"typed effect={int(pg314_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg314_counts.get('route_count', 0) or 0)} · negative violations={int(pg314_counts.get('negative_lane_violation_count', 0) or 0)}",
            f"preflight question={float(pg314_report.get('preflight_identifiability', {}).get('question_recall', 0.0) or 0.0) * 100:.1f}% · unsafe allow={int(pg314_report.get('preflight_identifiability', {}).get('unsafe_allow', 0) or 0)}",
            "独立实现复放通过，但 PG-313 worst-seed 离线门未通过，且 wire 仍由 source-grounded adapter 绑定",
        ],
        "evidence_hash": str(pg314_report.get("report_sha256", ""))[:16],
        "instruction": "人工查看 PG-314 两条 GET/POST SQL row-shape wire、负对照和 evidence hash；这证明模型选择抽象变体能进入第二实现，不证明已学会通用 literal payload。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg315-worst-seed-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-315 三 seed worst-case + failure repair",
        "route": f"{int(pg315_counts.get('seed_count', 0) or 0)} seeds · {int(pg315_counts.get('route_count', 0) or 0)} route episodes · GET={int(pg315_counts.get('get_count', 0) or 0)} / POST={int(pg315_counts.get('post_count', 0) or 0)}",
        "seed": 31501,
        "method": "all PG-313 seed checkpoints → independent variant replay",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"worst question={float(pg315_worst.get('question_recall_min', 0.0) or 0.0) * 100:.1f}% · variant={float(pg315_worst.get('variant_exact_min', 0.0) or 0.0) * 100:.1f}%",
            f"failure repair/abstain={float(pg315_worst.get('repair_abstain_min', 0.0) or 0.0) * 100:.1f}% · negative violations={int(pg315_worst.get('negative_lane_violation_max', 0) or 0)}",
            f"typed effect={int(pg315_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg315_counts.get('route_count', 0) or 0)}; worst-seed gate blocked",
        ],
        "evidence_hash": str(pg315_report.get("report_sha256", ""))[:16],
        "instruction": "保留 seed=31303 的 variant misselection/negative violation 和 failure 后错误的 assemble/no-repair；这是主动排错尚未学会的直接证据。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg316-failure-repair-anchor",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-316 failure→repair next-token anchor",
        "route": f"{int(pg316_report.get('training', {}).get('fit_count', 0) or 0)} train · {int(pg316_report.get('training', {}).get('holdout_count', 0) or 0)} holdout · repair train={int(pg316_report.get('training', {}).get('repair_train_rows', 0) or 0)}",
        "seed": 31601,
        "method": "causal Transformer-MoE weighted repair/variant next-token",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"question min={float(dict(pg316_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · variant min={float(dict(pg316_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"repair exact min={float(dict(pg316_metrics.get('holdout_repair_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · repair safe-allow max={float(dict(pg316_metrics.get('holdout_repair_safe_allow_max') or {}).get('max', 0.0) or 0.0):.0f}",
            f"hard false-allow max={float(dict(pg316_metrics.get('hard_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}; question worst seed remains below gate",
        ],
        "evidence_hash": str(pg316_report.get("report_sha256", ""))[:16],
        "instruction": "保留 PG-316 的 repair 提升，但不能用它掩盖 question=88%；必须在 live/三 seed 上同时复验。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg316-live-independent-variant",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-316 live repair/variant 复放",
        "route": f"{int(pg316_live_counts.get('route_count', 0) or 0)} routes · GET={int(pg316_live_counts.get('get_count', 0) or 0)} / POST={int(pg316_live_counts.get('post_count', 0) or 0)}",
        "seed": 31601,
        "method": "PG-316 checkpoint → independent network=none evaluator",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"variant={int(pg316_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg316_live_counts.get('variant_role_count', 0) or 0)} · typed={int(pg316_live_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg316_live_counts.get('route_count', 0) or 0)}",
            f"failure repair={int(pg316_live_report.get('failure_repair', {}).get('correct', 0) or 0)}/{int(pg316_live_report.get('failure_repair', {}).get('count', 0) or 0)} · negative violation={int(pg316_live_counts.get('negative_lane_violation_count', 0) or 0)}",
            "best PG-316 seed live lane passes, but offline worst question and all-seed stability remain unresolved",
        ],
        "evidence_hash": str(pg316_live_report.get("report_sha256", ""))[:16],
        "instruction": "人工查看 PG-316 live 的 failure→repair token、两条 GET/POST wire 和 evidence hash；只作为 evaluation evidence。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg317-question-anchor",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-317 多缺失槽位 ASK/complete question anchor",
        "route": f"{int(pg317_report.get('training', {}).get('fit_count', 0) or 0)} train · {int(pg317_report.get('training', {}).get('holdout_count', 0) or 0)} holdout · anchors={int((pg317_dataset.get('counts') or {}).get('anchor_rows', 0) or 0)}",
        "seed": 31701,
        "method": "causal Transformer-MoE next-token question anchor",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"anchor question min={float(dict(pg317_metrics.get('holdout_anchor_question_exact') or {}).get('min', 0.0) or 0.0) * 100:.2f}% · original missing question min={float(dict(pg317_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.2f}%",
            f"anchor safe-allow max={float(dict(pg317_metrics.get('holdout_anchor_safe_allow_max') or {}).get('max', 0.0) or 0.0):.0f} · unnecessary max={float(dict(pg317_metrics.get('holdout_anchor_unnecessary_question') or {}).get('max', 0.0) or 0.0) * 100:.2f}%",
            f"variant min={float(dict(pg317_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.2f}% · repair min={float(dict(pg317_metrics.get('holdout_repair_exact') or {}).get('min', 0.0) or 0.0) * 100:.2f}% · hard false-allow max={float(dict(pg317_metrics.get('hard_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}",
            "离线锚点门通过，但仍禁止 promotion；必须结合 fresh independent live replay 和更多族外实现",
        ],
        "evidence_hash": str(pg317_report.get("report_sha256", ""))[:16],
        "instruction": "人工复核多缺失槽位记录：模型必须先问优先级最高的缺口，不能把不可识别状态当成训练不足；完整配对不能多问。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg317-live-independent-variant",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-317 fresh GET/POST independent replay",
        "route": f"{int(pg317_live_counts.get('route_count', 0) or 0)} routes · GET={int(pg317_live_counts.get('get_count', 0) or 0)} / POST={int(pg317_live_counts.get('post_count', 0) or 0)}",
        "seed": 31701,
        "method": "PG-317 checkpoint → fresh network=none evaluator",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"variant={int(pg317_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg317_live_counts.get('variant_role_count', 0) or 0)} · typed={int(pg317_live_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg317_live_counts.get('route_count', 0) or 0)}",
            f"question={float(pg317_live_report.get('preflight_identifiability', {}).get('question_recall', 0.0) or 0.0) * 100:.1f}% · unsafe allow={int(pg317_live_report.get('preflight_identifiability', {}).get('unsafe_allow', 0) or 0)} · repair={int(pg317_live_report.get('failure_repair', {}).get('correct', 0) or 0)}/{int(pg317_live_report.get('failure_repair', {}).get('count', 0) or 0)}",
            f"negative violation={int(pg317_live_counts.get('negative_lane_violation_count', 0) or 0)} · fresh reset={int(pg317_live_counts.get('fresh_reset_count', 0) or 0)} · evidence hashes={int(pg317_live_counts.get('typed_evidence_hash_count', 0) or 0)}",
            "live SQL row-shape GET/POST 通过，但不是 DOM/XSS 或族外能力证明；promotion 继续关闭",
        ],
        "evidence_hash": str(pg317_live_report.get("report_sha256", ""))[:16],
        "instruction": "人工查看 fresh reset、GET/POST wire、negative/reference、repair 和 evidence hash；只将其作为评估证据，不写入长期记忆。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg318-family-heldout-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-318 族外 DOM/SQL 双 typed oracle 复放",
        "route": f"{int(pg318_counts.get('route_count', 0) or 0)} routes · GET={int(pg318_counts.get('get_count', 0) or 0)} / POST={int(pg318_counts.get('post_count', 0) or 0)} · seeds={int(pg318_counts.get('seed_count', 0) or 0)}",
        "seed": 31701,
        "method": "PG-317 checkpoint → Docker network=none internal curl + in-memory DOM oracle",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"variant={int(pg318_counts.get('variant_exact_count', 0) or 0)}/{int(pg318_counts.get('variant_role_count', 0) or 0)} · typed={int(pg318_counts.get('typed_effect_count', 0) or 0)}/{int(pg318_counts.get('route_count', 0) or 0)}",
            f"multi-missing ASK worst={float(pg318_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}% · unsafe allow={int(pg318_counts.get('multi_missing_unsafe_allow', 0) or 0)}",
            f"repair={int(pg318_counts.get('failure_repair_correct_count', 0) or 0)}/{int(pg318_counts.get('failure_repair_count', 0) or 0)} · negative violation={int(pg318_counts.get('negative_lane_violation_count', 0) or 0)}",
            "typed evidence is evaluator evidence only; family/implementation coverage is still too small for promotion",
        ],
        "evidence_hash": str(pg318_report.get("report_sha256", ""))[:16],
        "instruction": "人工查看 SQL row-shape、DOM execution、matched negative、fresh reset、evidence hash，以及首轮 href 失败到 void-0 修复的证据链；不得把 holdout 回放直接变成训练 gold。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg319-cross-impl-rule-ir",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-319 跨实现 Rule-IR / 遗忘 canary",
        "route": f"{int((pg319_dataset.get('counts') or {}).get('total', 0) or 0)} rows · train={int((pg319_dataset.get('counts') or {}).get('train', 0) or 0)} · implementation holdout={int((pg319_dataset.get('counts') or {}).get('implementation_holdout', 0) or 0)}",
        "seed": 31901,
        "method": "decoder-only Transformer-MoE cross-implementation fine-tune",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"route holdout question={float(dict(pg319_metrics.get('implementation_route_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · variant={float(dict(pg319_metrics.get('implementation_route_variant_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"family checkpoint question={float(dict(pg319_metrics.get('pg318_family_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · false allow={float(dict(pg319_metrics.get('pg318_family_false_allow_max') or {}).get('max', 0.0) or 0.0):.0f}",
            "新实现训练提高了部分离线 OOD，但 family 角色混淆和 new-only 遗忘保留为失败信号",
        ],
        "evidence_hash": str(pg319_report.get("report_sha256", ""))[:16],
        "instruction": "人工区分 cross-implementation 提升、family holdout 失败和 replay-mix 遗忘；不把离线 Rule-IR 分数当成漏洞 payload 能力。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg320-observation-lattice",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-320 缺观测 lattice 微调",
        "route": f"{int((pg320_dataset.get('counts') or {}).get('total', 0) or 0)} rows · lattice holdout",
        "seed": 31901,
        "method": "causal next-token observation lattice + replay mix",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"family ASK={float(dict(pg320_metrics.get('family_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · family variant={float(dict(pg320_metrics.get('family_variant_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"lattice ASK={float(dict(pg320_metrics.get('lattice_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · variant={float(dict(pg320_metrics.get('lattice_variant_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            f"new-only old drop={float(pg320_metrics.get('new_only_old_drop_max', 0.0) or 0.0):.3f} · replay-mix drop={float(pg320_metrics.get('replay_mix_old_drop_max', 0.0) or 0.0):.3f}",
            "离线 ASK 修复不等于 live 变体正确；PG-320 live 结果单独保留",
        ],
        "evidence_hash": str(pg320_report.get("report_sha256", ""))[:16],
        "instruction": "任何缺失关键观测只标记 incomplete/ASK；禁止用最终答案补标签或写入长期记忆。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg321-family-heldout-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-321 角色条件 variant 族外复放",
        "route": f"{int(pg321_live_counts.get('route_count', 0) or 0)} routes · GET={int(pg321_live_counts.get('get_count', 0) or 0)} / POST={int(pg321_live_counts.get('post_count', 0) or 0)} · seeds={int(pg321_live_counts.get('seed_count', 0) or 0)}",
        "seed": 31901,
        "method": "PG-321 checkpoint → fresh Docker network=none + DOM/SQL typed oracle",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"variant={int(pg321_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg321_live_counts.get('variant_role_count', 0) or 0)} · typed={int(pg321_live_counts.get('typed_effect_count', 0) or 0)}/{int(pg321_live_counts.get('route_count', 0) or 0)}",
            f"ASK worst={float(pg321_live_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}% · unsafe allow={int(pg321_live_counts.get('multi_missing_unsafe_allow', 0) or 0)} · repair={int(pg321_live_counts.get('failure_repair_correct_count', 0) or 0)}/{int(pg321_live_counts.get('failure_repair_count', 0) or 0)}",
            f"negative violation={int(pg321_live_counts.get('negative_lane_violation_count', 0) or 0)} · evidence all={bool(pg321_live_gate.get('checks', {}).get('typed_evidence_all'))}",
            "这次真实回放通过硬性行为检查，但仍是单一固定实现的 evaluation-only 证据，不自动晋级",
        ],
        "evidence_hash": str(pg321_live_report.get("report_sha256", ""))[:16],
        "instruction": "人工检查候选/参考/阴性 wire、DOM/SQL typed oracle、fresh reset 和证据哈希；不把回放响应体或 literal payload 放入模型训练。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg323-decoy-ask-anchor-training",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-323 decoy/ASK anchor next-token 微调",
        "route": f"{int((pg323_dataset.get('counts') or {}).get('total', 0) or 0)} rows · train={int((pg323_dataset.get('split_counts') or {}).get('train', 0) or 0)} · ask={int((pg323_dataset.get('counts') or {}).get('ask_train', 0) or 0)}",
        "seed": 31901,
        "method": "causal Transformer-MoE next-token + history-order anchors + replay mix",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"ASK worst={float(dict(pg323_metrics.get('ask_question_min') or {}).get('min', 0.0) or 0.0) * 100:.2f}% · unsafe allow={float(dict(pg323_metrics.get('ask_unsafe_allow_max') or {}).get('max', 0.0) or 0.0):.0f}",
            f"hard-negative false-allow={float(dict(pg323_metrics.get('hard_false_allow_max') or {}).get('max', 0.0) or 0.0):.0f} · third-surface variant={float(dict(pg323_metrics.get('third_surface_variant_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%",
            "相比 PG-322，新增历史顺序/decoy ASK anchor；仍需真实独立容器复放后才可讨论泛化",
        ],
        "evidence_hash": str(pg323_report.get("report_sha256", ""))[:16],
        "instruction": "缺关键观测只能 ASK/incomplete；原始 payload、响应体和 evaluator authority 不进入模型上下文或长期记忆。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg323-vulnerableapp-fresh-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-323 独立 VulnerableApp fresh GET/POST 复放",
        "route": f"{int(pg323_live_counts.get('route_count', 0) or 0)} routes · GET={int(pg323_live_counts.get('get_count', 0) or 0)} / POST={int(pg323_live_counts.get('post_count', 0) or 0)} · seeds={int(pg323_live_counts.get('seed_count', 0) or 0)}",
        "seed": 31901,
        "method": "PG-323 checkpoint → fresh Docker network=none + typed DOM oracle",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            f"variant={int(pg323_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg323_live_counts.get('variant_role_count', 0) or 0)} · typed={int(pg323_live_counts.get('positive_typed_effect_count', 0) or 0)}/{int(pg323_live_counts.get('positive_route_count', 0) or 0)}",
            f"ASK worst={float(pg323_live_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}% · unsafe allow={int(pg323_live_counts.get('multi_missing_unsafe_allow', 0) or 0)} · repair={int(pg323_live_counts.get('failure_repair_correct_count', 0) or 0)}/{int(pg323_live_counts.get('failure_repair_count', 0) or 0)}",
            f"negative violation={int(pg323_live_counts.get('negative_lane_violation_count', 0) or 0)} · evidence all={bool(pg323_live_gate.get('checks', {}).get('typed_evidence_all'))}",
            "行为硬门通过，但仍是单一独立实现的 evaluation-only 证据；promotion 保持关闭",
        ],
        "evidence_hash": str(pg323_live_report.get("report_sha256", ""))[:16],
        "instruction": "人工核对候选/reference/negative 角色、GET/POST 复放、fresh reset、typed oracle 与 SHA-256；禁止将此结果解释为任意 URL 的漏洞确认。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg324-juice-shop-source-heldout-fresh-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-324 Juice Shop source-heldout v2 fresh GET/POST 复放",
        "route": "3 seeds · 18 disposable routes · GET=9 / POST=9",
        "seed": 31901,
        "method": "PG-323 abstract Rule-IR → allow-listed local relay → typed challenge-state delta",
        "typed_effect": False,
        "confirmed_positive": False,
        "artifact_status": str(pg324_contract.get("artifact_status", "awaiting_fresh_replay")),
        "reasons": [
            f"artifact={str(pg324_contract.get('artifact_status', 'awaiting_fresh_replay'))} · schema={str(pg324_contract.get('schema_version', '')) or 'missing'}",
            f"typed={int(pg324_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('positive_route_count', 0) or 0)} · variant={int(pg324_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('variant_role_count', 0) or 0)} · audit=independent_passed",
            "v2 的 typed oracle 必须是 fresh baseline unsolved → evaluator challenge solved 状态差分；浏览器 dialog 只能作为诊断",
            "belief trace、GET/POST、candidate/reference/negative、证据哈希和所有安全检查齐全前，不生成训练样本",
        ],
        "evidence_hash": str(pg324_contract.get("report_evidence_hash", "")),
        "instruction": "人工只复核脱敏请求形状、typed state delta、negative、fresh reset、safety-mode attestation 和 SHA-256；本地 hard gate 通过仍不能当作模型能力或公网漏洞确认。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg325-sql-family-holdout-fresh-replay",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-325 SQL 族外 fresh GET/POST 复放",
        "route": "3 seeds · 9 disposable SQL routes · GET=6 / POST=3",
        "seed": 31901,
        "method": "PG-323 abstract Rule-IR → allow-listed local Pikachu SQL adapter → typed response-shape oracle",
        "typed_effect": False,
        "confirmed_positive": False,
        "artifact_status": str(pg325_contract.get("artifact_status", "awaiting_fresh_replay")),
        "audit_status": str(pg325_contract.get("audit_status", "not_embedded")),
        "reasons": [
            f"artifact={str(pg325_contract.get('artifact_status', 'awaiting_fresh_replay'))} · schema={str(pg325_contract.get('schema_version', '')) or 'missing'} · audit={str(pg325_contract.get('audit_status', 'not_embedded'))}",
            f"typed={int(pg325_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('positive_route_count', 0) or 0)} · variant={int(pg325_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('variant_role_count', 0) or 0)} · ASK={int(pg325_contract.get('counts', {}).get('multi_missing_question_rows', 0) or 0)}",
            f"repair={int(pg325_contract.get('counts', {}).get('failure_repair_correct_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('failure_repair_count', 0) or 0)} · action_changed={int(pg325_contract.get('counts', {}).get('failure_action_changed_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('failure_transition_required_count', 0) or 0)} · negative={int(pg325_contract.get('counts', {}).get('negative_lane_violation_count', 0) or 0)}",
            "跨实现 canary 只作为稳定性对照；PG-325 本身仍是同一 Pikachu 实现上的 SQL 族外 evaluation-only 证据",
            "typed response-shape、belief trace、failure action change、evidence hash 和安全门齐全前，不生成训练样本或长期记忆",
        ],
        "evidence_hash": str(pg325_contract.get("report_evidence_hash", "")),
        "instruction": "人工只复核脱敏 GET/POST 形状、candidate/reference/negative、fresh reset、数据库健康、typed oracle、role-bound belief evidence 和 SHA-256；不得把此结果解释为任意 URL 的漏洞确认。",
        "raw_material_available": False,
    })
    snapshot["tasks"]["reviewer"].append({
        "id": "pg326-cross-implementation-forgetting-matrix",
        "role": "reviewer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "promotion_blocked",
        "label": "PG-326 跨实现稳定性 / 灾难性遗忘矩阵",
        "route": "PG-323 + PG-324 + PG-325 · 45 routes · 9 seeds · GET=27 / POST=18",
        "seed": 31901,
        "method": "只读聚合 frozen canary report；不启动 Docker、不接触目标",
        "typed_effect": False,
        "confirmed_positive": False,
        "artifact_status": str(pg326_contract.get("artifact_status", "awaiting_matrix")),
        "audit_status": str(pg326_contract.get("audit_status", "not_embedded")),
        "reasons": [
            f"observed typed={int(pg326_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg326_contract.get('counts', {}).get('positive_route_count', 0) or 0)} · variant={int(pg326_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg326_contract.get('counts', {}).get('variant_role_count', 0) or 0)} · ASK={int(pg326_contract.get('counts', {}).get('multi_missing_question_rows', 0) or 0)}",
            f"repair={int(pg326_contract.get('counts', {}).get('failure_repair_correct_count', 0) or 0)}/{int(pg326_contract.get('counts', {}).get('failure_repair_count', 0) or 0)} · negative={int(pg326_contract.get('counts', {}).get('negative_lane_violation_count', 0) or 0)} · implementations={int(pg326_contract.get('implementation_count', 0) or 0)}",
            f"uniform_contract={bool((pg326_contract.get('matrix_gate_checks') or {}).get('uniform_observation_contract'))} · forgetting_pair={bool((pg326_contract.get('matrix_gate_checks') or {}).get('forgetting_pair'))}",
            "PG-323 缺少显式 failure-action-change/role-bound belief contract，且尚无训练前后同一 canary 的 checkpoint pair；缺字段保持 blocked，不把 perfect observed score 当作晋级",
        ],
        "evidence_hash": str(pg326_contract.get("report_evidence_hash", "")),
        "instruction": "人工复核三份 report 的实现 digest、族/方法覆盖、worst-seed、缺字段清单和 forgetting pair；下一步先补统一 strict schema 与 before/after replay，禁止扩容或写长期记忆。",
        "raw_material_available": False,
    })
    snapshot["capability"]["model"]["pg308"] = {"status": str(pg308_report.get("status", "not_run")), "architecture": "causal_transformer_moe_multisource_symbolic_slot", "question_recall_min": float(dict(pg308_metrics.get("source_holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0), "bound_slot_exact_min": float(dict(pg308_metrics.get("source_holdout_bound_assembly_slot_exact") or {}).get("min", 0.0) or 0.0), "hard_bound_false_allow_max": float(dict(pg308_metrics.get("hard_negative_bound_false_allow") or {}).get("max", 0.0) or 0.0), "slot_permutation_false_allow": float(pg308_metrics.get("slot_permutation_bound_false_allow", 0.0) or 0.0), "promotion_blocked": True, "evidence_hash": str(pg308_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["model"]["pg309"] = {"status": str(pg309_report.get("status", "not_run")), "architecture": "causal_transformer_moe_balanced_counterfactual", "question_recall_min": float(dict(pg309_metrics.get("source_holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0), "bound_slot_exact_min": float(dict(pg309_metrics.get("source_holdout_bound_assembly_slot_exact") or {}).get("min", 0.0) or 0.0), "unnecessary_question_max": float(dict(pg309_metrics.get("source_holdout_unnecessary_question_rate") or {}).get("max", 0.0) or 0.0), "hard_bound_false_allow_max": float(dict(pg309_metrics.get("hard_negative_bound_false_allow") or {}).get("max", 0.0) or 0.0), "promotion_blocked": True, "evidence_hash": str(pg309_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["model"]["pg310"] = {"status": str(pg310_report.get("status", "not_run")), "architecture": "causal_transformer_moe_optimization_ablation", "wide_question_min": float(dict(pg310_wide_metrics.get("holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0), "wide_slot_min": float(dict(pg310_wide_metrics.get("holdout_bound_slot_exact") or {}).get("min", 0.0) or 0.0), "wide_hard_false_allow_max": float(dict(pg310_wide_metrics.get("hard_bound_false_allow") or {}).get("max", 0.0) or 0.0), "promotion_blocked": True, "evidence_hash": str(pg310_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["model"]["pg311"] = {"status": str(pg311_report.get("status", "not_run")), "architecture": "causal_transformer_moe_wide_question_anchor", "question_recall_min": float(dict(pg311_metrics.get("holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0), "bound_slot_exact_min": float(dict(pg311_metrics.get("holdout_bound_slot_exact") or {}).get("min", 0.0) or 0.0), "unnecessary_question_max": float(dict(pg311_metrics.get("holdout_unnecessary_question") or {}).get("max", 0.0) or 0.0), "bound_false_allow_max": float(dict(pg311_metrics.get("holdout_bound_false_allow") or {}).get("max", 0.0) or 0.0), "promotion_blocked": True, "evidence_hash": str(pg311_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["model"]["pg312"] = {"status": str(pg312_report.get("status", "not_run")), "architecture": str(pg312_model.get("architecture", "causal_transformer_moe_next_token")), "symbolic_checkpoint": bool(pg312_model.get("symbolic_checkpoint")), "route_count": int(pg312_counts.get("route_count", 0) or 0), "model_candidate_send_count": int(pg312_counts.get("model_candidate_send_count", 0) or 0), "model_confirmed_effect_count": int(pg312_counts.get("model_confirmed_effect_count", 0) or 0), "evaluator_gold_typed_effect_count": int(pg312_counts.get("evaluator_gold_typed_effect_count", 0) or 0), "false_positive_count": int(pg312_counts.get("false_positive_count", 0) or 0), "loopback_only": bool(pg312_checks.get("loopback_only")), "typed_evidence_hash_per_route": bool(pg312_checks.get("typed_evidence_hash_per_route")), "source_grounded_wire": str(pg312_model.get("wire_generation", "")) == "source_grounded_binding_after_guard", "promotion_blocked": True, "evidence_hash": str(pg312_report.get("report_sha256", ""))[:16]}
    snapshot["capability"]["model"]["pg313"] = {
        "status": str(pg313_report.get("status", "not_run")),
        "architecture": str(pg313_training.get("architecture", "causal_transformer_moe_next_token")),
        "target_representation": str(pg313_training.get("target_representation", "symbolic_slot_copy_plus_probe_variant_ref")),
        "question_recall_min": float(dict(pg313_metrics.get("holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "base_slot_exact_min": float(dict(pg313_metrics.get("holdout_bound_base_slot_exact") or {}).get("min", 0.0) or 0.0),
        "variant_exact_min": float(dict(pg313_metrics.get("holdout_variant_exact") or {}).get("min", 0.0) or 0.0),
        "unnecessary_question_max": float(dict(pg313_metrics.get("holdout_unnecessary_question") or {}).get("max", 0.0) or 0.0),
        "holdout_false_allow_max": float(dict(pg313_metrics.get("holdout_bound_false_allow") or {}).get("max", 0.0) or 0.0),
        "hard_false_allow_max": float(dict(pg313_metrics.get("hard_bound_false_allow") or {}).get("max", 0.0) or 0.0),
        "dataset_count": int((pg313_dataset.get("counts") or {}).get("total", 0) or 0),
        "dataset_audit_status": str(pg313_audit.get("status", "not_run")),
        "promotion_blocked": True,
        "evidence_hash": str(pg313_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg314"] = {
        "status": str(pg314_report.get("status", "not_run")),
        "architecture": str(pg314_model.get("architecture", "causal_transformer_moe_next_token")),
        "target_representation": str(pg314_model.get("target_representation", "symbolic_slot_copy_plus_probe_variant_ref")),
        "route_count": int(pg314_counts.get("route_count", 0) or 0),
        "get_count": int(pg314_counts.get("get_count", 0) or 0),
        "post_count": int(pg314_counts.get("post_count", 0) or 0),
        "variant_role_count": int(pg314_counts.get("model_variant_role_count", 0) or 0),
        "variant_exact_count": int(pg314_counts.get("model_variant_exact_count", 0) or 0),
        "model_variant_send_count": int(pg314_counts.get("model_variant_send_count", 0) or 0),
        "model_typed_effect_count": int(pg314_counts.get("model_typed_effect_count", 0) or 0),
        "negative_lane_violation_count": int(pg314_counts.get("negative_lane_violation_count", 0) or 0),
        "preflight_question_recall": float(pg314_report.get("preflight_identifiability", {}).get("question_recall", 0.0) or 0.0),
        "preflight_unsafe_allow": int(pg314_report.get("preflight_identifiability", {}).get("unsafe_allow", 0) or 0),
        "independent_image": str(pg314_report.get("runtime", {}).get("image", "")),
        "docker_network_none": bool(pg314_checks.get("docker_network_none")),
        "fresh_reset_per_route": bool(pg314_checks.get("fresh_reset_per_route")),
        "typed_evidence_hash_per_route": bool(pg314_checks.get("typed_evidence_hash_per_route")),
        "source_grounded_wire": str(pg314_model.get("wire_generation", "")) == "source_grounded_binding_after_model_variant_guard",
        "promotion_blocked": True,
        "evidence_hash": str(pg314_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg315"] = {
        "status": str(pg315_report.get("status", "not_run")),
        "architecture": str(pg315_report.get("model", {}).get("architecture", "causal_transformer_moe_next_token")),
        "seed_count": int(pg315_counts.get("seed_count", 0) or 0),
        "route_count": int(pg315_counts.get("route_count", 0) or 0),
        "get_count": int(pg315_counts.get("get_count", 0) or 0),
        "post_count": int(pg315_counts.get("post_count", 0) or 0),
        "variant_role_count": int(pg315_counts.get("model_variant_role_count", 0) or 0),
        "variant_exact_count": int(pg315_counts.get("model_variant_exact_count", 0) or 0),
        "model_typed_effect_count": int(pg315_counts.get("model_typed_effect_count", 0) or 0),
        "negative_lane_violation_count": int(pg315_counts.get("negative_lane_violation_count", 0) or 0),
        "repair_row_count": int(pg315_counts.get("repair_row_count", 0) or 0),
        "repair_abstain_correct_count": int(pg315_counts.get("repair_abstain_correct_count", 0) or 0),
        "worst_question_recall": float(pg315_worst.get("question_recall_min", 0.0) or 0.0),
        "worst_variant_exact": float(pg315_worst.get("variant_exact_min", 0.0) or 0.0),
        "worst_repair_abstain": float(pg315_worst.get("repair_abstain_min", 0.0) or 0.0),
        "worst_negative_violation": int(pg315_worst.get("negative_lane_violation_max", 0) or 0),
        "docker_network_none": True,
        "promotion_blocked": True,
        "evidence_hash": str(pg315_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg316"] = {
        "status": str(pg316_report.get("status", "not_run")),
        "architecture": str(pg316_report.get("training", {}).get("architecture", "causal_transformer_moe_next_token")),
        "question_recall_min": float(dict(pg316_metrics.get("holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "variant_exact_min": float(dict(pg316_metrics.get("holdout_variant_exact") or {}).get("min", 0.0) or 0.0),
        "repair_exact_min": float(dict(pg316_metrics.get("holdout_repair_exact") or {}).get("min", 0.0) or 0.0),
        "repair_safe_allow_max": float(dict(pg316_metrics.get("holdout_repair_safe_allow_max") or {}).get("max", 0.0) or 0.0),
        "hard_false_allow_max": float(dict(pg316_metrics.get("hard_bound_false_allow") or {}).get("max", 0.0) or 0.0),
        "repair_train_rows": int(pg316_report.get("training", {}).get("repair_train_rows", 0) or 0),
        "dataset_count": int((pg316_dataset.get("counts") or {}).get("total", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg316_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg316_live"] = {
        "status": str(pg316_live_report.get("status", "not_run")),
        "route_count": int(pg316_live_counts.get("route_count", 0) or 0),
        "variant_role_count": int(pg316_live_counts.get("variant_role_count", 0) or 0),
        "variant_exact_count": int(pg316_live_counts.get("variant_exact_count", 0) or 0),
        "model_typed_effect_count": int(pg316_live_counts.get("model_typed_effect_count", 0) or 0),
        "negative_lane_violation_count": int(pg316_live_counts.get("negative_lane_violation_count", 0) or 0),
        "failure_repair_correct": int(pg316_live_report.get("failure_repair", {}).get("correct", 0) or 0),
        "failure_repair_count": int(pg316_live_report.get("failure_repair", {}).get("count", 0) or 0),
        "preflight_question_recall": float(pg316_live_report.get("preflight_identifiability", {}).get("question_recall", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg316_live_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg317"] = {
        "status": str(pg317_report.get("status", "not_run")),
        "architecture": str(pg317_report.get("training", {}).get("architecture", "causal_transformer_moe_next_token")),
        "anchor_question_min": float(dict(pg317_metrics.get("holdout_anchor_question_exact") or {}).get("min", 0.0) or 0.0),
        "anchor_safe_allow_max": float(dict(pg317_metrics.get("holdout_anchor_safe_allow_max") or {}).get("max", 0.0) or 0.0),
        "anchor_unnecessary_question_max": float(dict(pg317_metrics.get("holdout_anchor_unnecessary_question") or {}).get("max", 0.0) or 0.0),
        "question_recall_min": float(dict(pg317_metrics.get("holdout_missing_question_recall") or {}).get("min", 0.0) or 0.0),
        "variant_exact_min": float(dict(pg317_metrics.get("holdout_variant_exact") or {}).get("min", 0.0) or 0.0),
        "repair_exact_min": float(dict(pg317_metrics.get("holdout_repair_exact") or {}).get("min", 0.0) or 0.0),
        "hard_false_allow_max": float(dict(pg317_metrics.get("hard_bound_false_allow") or {}).get("max", 0.0) or 0.0),
        "dataset_count": int((pg317_dataset.get("counts") or {}).get("total", 0) or 0),
        "anchor_count": int((pg317_dataset.get("counts") or {}).get("anchor_rows", 0) or 0),
        "dataset_audit_status": str(pg317_audit.get("status", "not_run")),
        "promotion_blocked": True,
        "evidence_hash": str(pg317_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg317_live"] = {
        "status": str(pg317_live_report.get("status", "not_run")),
        "route_count": int(pg317_live_counts.get("route_count", 0) or 0),
        "variant_role_count": int(pg317_live_counts.get("variant_role_count", 0) or 0),
        "variant_exact_count": int(pg317_live_counts.get("variant_exact_count", 0) or 0),
        "model_typed_effect_count": int(pg317_live_counts.get("model_typed_effect_count", 0) or 0),
        "negative_lane_violation_count": int(pg317_live_counts.get("negative_lane_violation_count", 0) or 0),
        "failure_repair_correct": int(pg317_live_report.get("failure_repair", {}).get("correct", 0) or 0),
        "failure_repair_count": int(pg317_live_report.get("failure_repair", {}).get("count", 0) or 0),
        "preflight_question_recall": float(pg317_live_report.get("preflight_identifiability", {}).get("question_recall", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg317_live_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg318_family_holdout"] = {
        "status": str(pg318_report.get("status", "not_run")),
        "architecture": str(pg318_report.get("model", {}).get("architecture", "causal_transformer_moe_next_token")),
        "seed_count": int(pg318_counts.get("seed_count", 0) or 0),
        "route_count": int(pg318_counts.get("route_count", 0) or 0),
        "get_count": int(pg318_counts.get("get_count", 0) or 0),
        "post_count": int(pg318_counts.get("post_count", 0) or 0),
        "sql_route_count": int(pg318_counts.get("sql_route_count", 0) or 0),
        "xss_route_count": int(pg318_counts.get("xss_route_count", 0) or 0),
        "variant_exact_min": float(pg318_worst.get("variant_exact_min", 0.0) or 0.0),
        "typed_effect_route_rate_min": float(pg318_worst.get("typed_effect_route_rate_min", 0.0) or 0.0),
        "multi_missing_question_recall_min": float(pg318_worst.get("multi_missing_question_recall_min", 0.0) or 0.0),
        "multi_missing_unsafe_allow": int(pg318_counts.get("multi_missing_unsafe_allow", 0) or 0),
        "failure_repair_rate_min": float(pg318_worst.get("failure_repair_rate_min", 0.0) or 0.0),
        "negative_lane_violation_max": int(pg318_worst.get("negative_lane_violation_max", 0) or 0),
        "docker_network_none": bool(pg318_checks.get("docker_network_none")),
        "fresh_reset_all": bool(pg318_gate_checks.get("fresh_reset_all")),
        "typed_evidence_all": bool(pg318_gate_checks.get("typed_evidence_all")),
        "promotion_blocked": True,
        "evidence_hash": str(pg318_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg319_cross_impl"] = {
        "status": str(pg319_report.get("status", "not_run")),
        "architecture": str((pg319_report.get("training") or {}).get("architecture", "causal_transformer_moe_next_token")),
        "implementation_route_question_min": float(dict(pg319_metrics.get("implementation_route_question_min") or {}).get("min", 0.0) or 0.0),
        "implementation_route_variant_min": float(dict(pg319_metrics.get("implementation_route_variant_min") or {}).get("min", 0.0) or 0.0),
        "seed_holdout_question_min": float(dict(pg319_metrics.get("seed_holdout_question_min") or {}).get("min", 0.0) or 0.0),
        "family_question_min": float(dict(pg319_metrics.get("pg318_family_question_min") or {}).get("min", 0.0) or 0.0),
        "family_variant_min": float(dict(pg319_metrics.get("pg318_family_variant_min") or {}).get("min", 0.0) or 0.0),
        "new_only_forgetting_drop_max": float(pg319_metrics.get("new_only_forgetting_drop_max", 0.0) or 0.0),
        "replay_mix_forgetting_drop_max": float(pg319_metrics.get("replay_mix_forgetting_drop_max", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg319_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg320_observation_lattice"] = {
        "status": str(pg320_report.get("status", "not_run")),
        "architecture": str((pg320_report.get("training") or {}).get("architecture", "causal_transformer_moe_next_token")),
        "family_question_min": float(dict(pg320_metrics.get("family_question_min") or {}).get("min", 0.0) or 0.0),
        "family_variant_min": float(dict(pg320_metrics.get("family_variant_min") or {}).get("min", 0.0) or 0.0),
        "lattice_question_min": float(dict(pg320_metrics.get("lattice_question_min") or {}).get("min", 0.0) or 0.0),
        "lattice_variant_min": float(dict(pg320_metrics.get("lattice_variant_min") or {}).get("min", 0.0) or 0.0),
        "new_only_old_drop_max": float(pg320_metrics.get("new_only_old_drop_max", 0.0) or 0.0),
        "replay_mix_old_drop_max": float(pg320_metrics.get("replay_mix_old_drop_max", 0.0) or 0.0),
        "live_variant_exact_min": float(pg320_live_worst.get("variant_exact_min", 0.0) or 0.0),
        "live_typed_effect_route_rate_min": float(pg320_live_worst.get("typed_effect_route_rate_min", 0.0) or 0.0),
        "live_negative_lane_violation_max": int(pg320_live_worst.get("negative_lane_violation_max", 0) or 0),
        "promotion_blocked": True,
        "evidence_hash": str(pg320_live_report.get("report_sha256") or pg320_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg321_variant_role"] = {
        "status": str(pg321_report.get("status", "not_run")),
        "architecture": str((pg321_report.get("training") or {}).get("architecture", "causal_transformer_moe_next_token")),
        "family_question_min": float(dict(pg321_metrics.get("family_question_min") or {}).get("min", 0.0) or 0.0),
        "family_variant_min": float(dict(pg321_metrics.get("family_variant_min") or {}).get("min", 0.0) or 0.0),
        "family_false_allow_max": float(dict(pg321_metrics.get("family_false_allow_max") or {}).get("max", 0.0) or 0.0),
        "role_holdout_variant_min": float(dict(pg321_metrics.get("role_holdout_variant_min") or {}).get("min", 0.0) or 0.0),
        "live_variant_exact_min": float(pg321_live_worst.get("variant_exact_min", 0.0) or 0.0),
        "live_typed_effect_route_rate_min": float(pg321_live_worst.get("typed_effect_route_rate_min", 0.0) or 0.0),
        "live_multi_missing_question_recall_min": float(pg321_live_worst.get("multi_missing_question_recall_min", 0.0) or 0.0),
        "live_negative_lane_violation_max": int(pg321_live_worst.get("negative_lane_violation_max", 0) or 0),
        "live_fresh_reset_all": bool(pg321_live_gate.get("checks", {}).get("fresh_reset_all")),
        "live_typed_evidence_all": bool(pg321_live_gate.get("checks", {}).get("typed_evidence_all")),
        "promotion_blocked": True,
        "evidence_hash": str(pg321_live_report.get("report_sha256") or pg321_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg322_cross_impl_decoy"] = {
        "status": str(pg322_report.get("status", "not_run")),
        "architecture": str((pg322_report.get("training") or {}).get("architecture", "causal_transformer_moe_next_token")),
        "dataset_count": int((pg322_dataset.get("counts") or {}).get("total", 0) or 0),
        "ask_question_min": float(dict(pg322_metrics.get("ask_question_min") or {}).get("min", 0.0) or 0.0),
        "third_surface_variant_min": float(dict(pg322_metrics.get("third_surface_variant_min") or {}).get("min", 0.0) or 0.0),
        "hard_false_allow_max": float(dict(pg322_metrics.get("hard_false_allow_max") or {}).get("max", 0.0) or 0.0),
        "promotion_blocked": True,
        "evidence_hash": str(pg322_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg323_decoy_ask_anchor"] = {
        "status": str(pg323_report.get("status", "not_run")),
        "architecture": str((pg323_report.get("training") or {}).get("architecture", "causal_transformer_moe_next_token")),
        "dataset_count": int((pg323_dataset.get("counts") or {}).get("total", 0) or 0),
        "ask_question_min": float(dict(pg323_metrics.get("ask_question_min") or {}).get("min", 0.0) or 0.0),
        "ask_unsafe_allow_max": float(dict(pg323_metrics.get("ask_unsafe_allow_max") or {}).get("max", 0.0) or 0.0),
        "hard_false_allow_max": float(dict(pg323_metrics.get("hard_false_allow_max") or {}).get("max", 0.0) or 0.0),
        "third_surface_variant_min": float(dict(pg323_metrics.get("third_surface_variant_min") or {}).get("min", 0.0) or 0.0),
        "live_variant_exact_min": float(pg323_live_worst.get("variant_exact_min", 0.0) or 0.0),
        "live_typed_effect_route_rate_min": float(pg323_live_worst.get("positive_typed_effect_route_rate_min", 0.0) or 0.0),
        "live_multi_missing_question_recall_min": float(pg323_live_worst.get("multi_missing_question_recall_min", 0.0) or 0.0),
        "live_negative_lane_violation_max": int(pg323_live_worst.get("negative_lane_violation_max", 0) or 0),
        "live_fresh_reset_all": bool(pg323_live_gate.get("checks", {}).get("fresh_reset_all")),
        "live_typed_evidence_all": bool(pg323_live_gate.get("checks", {}).get("typed_evidence_all")),
        "promotion_blocked": True,
        "evidence_hash": str(pg323_live_report.get("report_sha256") or pg323_report.get("report_sha256", ""))[:16],
    }
    snapshot["capability"]["model"]["pg324_juice_shop_source_heldout"] = {
        **pg324_contract,
        "architecture": "frozen_pg323_decoder_only_rule_ir_plus_allowlisted_adapter",
        "target": "authorized local Juice Shop image digest only",
        "typed_oracle": "fresh baseline unsolved → evaluator challenge-state solved delta",
        "dom_script_execution_diagnostic_only": True,
        "evaluator_only": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    snapshot["capability"]["model"]["pg325_sql_family_holdout"] = {
        **pg325_contract,
        "architecture": "frozen_pg323_decoder_only_rule_ir_plus_allowlisted_sql_adapter",
        "target": "authorized local Pikachu SQL image digest only",
        "family_ood": True,
        "typed_oracle": "bounded SQL response-shape differential; no timing or write side effect",
        "role_bound_belief_evidence": True,
        "evaluator_only": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    snapshot["capability"]["model"]["pg326_cross_impl_forgetting_matrix"] = {
        **pg326_contract,
        "architecture": "read_only_matrix_over_frozen_pg323_rule_ir_checkpoint",
        "target": "no target contacted; aggregates PG-323/324/325 artifacts",
        "cross_implementation_count": int(pg326_contract.get("implementation_count", 0) or 0),
        "family_coverage": list(pg326_contract.get("families") or []),
        "forgetting_pair_required": True,
        "evaluator_only": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    snapshot["capability"]["model"]["pg327_a800_replay_training"] = {
        **pg327_contract,
        "architecture": "causal_transformer_moe_next_token_replay_mix",
        "target": "abstract Rule-IR/ASK/repair candidate training only",
        "remote_a800": True,
        "gpu0_only": True,
        "paired_forgetting_replay_required": True,
        "training_eligible": bool(pg327_contract.get("training_allowed")),
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    snapshot["capability"]["model"]["pg327b_paired_fresh_replay"] = {
        **pg327b_contract,
        "architecture": "before_after_checkpoint_paired_fresh_rule_ir_replay",
        "target": "authorized local Pikachu SQL canary only",
        "evaluator_only": True,
        "paired_forgetting_replay": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    snapshot["capability"]["model"]["pg370_multitask_moe_candidate"] = {
        **pg370_contract,
        "architecture": "shared_causal_moe_with_independent_lm_slot_ask_repair_negative_heads",
        "target": "abstract Rule-IR candidate training only",
        "evaluator_only": True,
        "training_eligible": False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    pg370_worst = dict(pg370_contract.get("worst_seed_metrics") or {})
    if pg370_contract.get("artifact_status") == "candidate_only":
        pg370_metric_value = (
            f"seq {float(pg370_worst.get('sequence_exact_min') or 0.0) * 100:.1f}% · "
            f"slot {float(pg370_worst.get('slot_accuracy_min') or 0.0) * 100:.1f}% · "
            f"ASK {float(pg370_worst.get('ask_recall_min') or 0.0) * 100:.1f}%"
        )
    elif pg370_contract.get("artifact_status") == "pending":
        pg370_metric_value = "PENDING"
    elif pg370_contract.get("artifact_status") == "stale_contract":
        pg370_metric_value = "STALE CONTRACT"
    else:
        pg370_metric_value = "INCOMPLETE"
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg370",
        "label": "PG-370 多任务 Causal-MoE 候选",
        "value": pg370_metric_value,
        "status": "blocked",
        "note": "仅显示抽象 vocab/最坏 seed/远程设备与 checkpoint 哈希；训练、记忆、payload 与漏洞声明全部关闭",
    })
    snapshot["capability"]["model"]["pg373_staged_pretrain_candidate"] = {
        **pg373_contract,
        "architecture": "staged_train_only_next_token_then_kl_anchored_rule_ir_heads",
        "target": "抽象 next-token/Rule-IR 组合候选训练；不是原始 payload 生成器",
        "evaluator_only": True,
        "training_eligible": False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    pg373_worst = dict(pg373_contract.get("worst_seed_metrics") or {})
    if pg373_contract.get("artifact_status") == "candidate_only":
        pg373_metric_value = (
            f"seq {float(pg373_worst.get('sequence_exact_min') or 0.0) * 100:.1f}% · "
            f"slot {float(pg373_worst.get('slot_accuracy_min') or 0.0) * 100:.1f}% · "
            f"entropy drop {float(pg373_worst.get('entropy_relative_drop_max') or 0.0) * 100:.1f}%"
        )
    elif pg373_contract.get("artifact_status") == "pending":
        pg373_metric_value = "PENDING"
    elif pg373_contract.get("artifact_status") == "stale_contract":
        pg373_metric_value = "STALE CONTRACT"
    else:
        pg373_metric_value = "INCOMPLETE"
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg373",
        "label": "PG-373 分阶段 next-token→Rule-IR 候选",
        "value": pg373_metric_value,
        "status": "blocked",
        "note": "熵基线改为 train-only 预训练后再比较；组合 exact/slot 仍不足，训练、记忆、payload 与漏洞声明全部关闭",
    })
    snapshot["capability"]["model"]["pg374_model_selected_replay_plan"] = {
        **pg374_contract,
        "architecture": "staged_candidate_rule_ir_to_allowlisted_binder_plan",
        "target": "WebGoat 第二实现 GET/POST fresh typed replay 计划",
        "evaluator_only": True,
        "training_eligible": False,
        "promotion_blocked": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    pg374_counts = dict(pg374_contract.get("counts") or {})
    if pg374_contract.get("artifact_status") == "planning_only_blocked":
        pg374_metric_value = f"rows {int(pg374_counts.get('roles') or 0)} · selected {int(pg374_counts.get('model_selected') or 0)} · target {int(pg374_counts.get('target_contacted') or 0)}"
    elif pg374_contract.get("artifact_status") == "pending":
        pg374_metric_value = "PENDING"
    elif pg374_contract.get("artifact_status") == "stale_contract":
        pg374_metric_value = "STALE CONTRACT"
    else:
        pg374_metric_value = "INCOMPLETE"
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg374",
        "label": "PG-374 模型选槽→第二实现计划",
        "value": pg374_metric_value,
        "status": "blocked",
        "note": "只读规划；缺 staged 13-slot 输出和 typed evidence 时保持 ASK，不创建 wire 或接触目标",
    })
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg312", "label": "PG-312 model-triggered local replay", "value": f"send {int(pg312_counts.get('model_candidate_send_count', 0) or 0)} / typed {int(pg312_counts.get('model_confirmed_effect_count', 0) or 0)}" if pg312_ready else "PENDING", "status": "blocked", "note": "真实 model-triggered replay；wire 由 source-grounded adapter 绑定，不是神经 literal payload"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg313", "label": "PG-313 probe variant / ASK", "value": f"question {float(dict(pg313_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.0f}% / variant {float(dict(pg313_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg313_ready else "PENDING", "status": "blocked", "note": "抽象 probe_variant/encoding_chain；hard-negative false-allow 和跨 seed worst-case 仍是硬门"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg314", "label": "PG-314 independent variant replay", "value": f"variant {int(pg314_counts.get('model_variant_exact_count', 0) or 0)}/{int(pg314_counts.get('model_variant_role_count', 0) or 0)} · typed {int(pg314_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg314_counts.get('route_count', 0) or 0)}" if pg314_ready else "PENDING", "status": "blocked", "note": "第二独立 digest、network=none、GET/POST SQL row-shape；科学晋级仍受 PG-313 worst-seed 门约束"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg315", "label": "PG-315 all-seed worst-case replay", "value": f"variant {int(pg315_counts.get('model_variant_exact_count', 0) or 0)}/{int(pg315_counts.get('model_variant_role_count', 0) or 0)} · repair {int(pg315_counts.get('repair_abstain_correct_count', 0) or 0)}/{int(pg315_counts.get('repair_row_count', 0) or 0)}" if pg315_ready else "PENDING", "status": "blocked", "note": "不再挑 best seed；seed=31303 暴露 variant misselection 与 failure 后 no-repair"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg316", "label": "PG-316 failure→repair anchor", "value": f"repair {float(dict(pg316_metrics.get('holdout_repair_exact') or {}).get('min', 0.0) or 0.0) * 100:.0f}% / variant {float(dict(pg316_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg316_ready else "PENDING", "status": "blocked", "note": "修复目标显著提升，但 question worst seed 仍低于 90%"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg316-live", "label": "PG-316 live repair replay", "value": f"variant {int(pg316_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg316_live_counts.get('variant_role_count', 0) or 0)} · repair {int(pg316_live_report.get('failure_repair', {}).get('correct', 0) or 0)}/{int(pg316_live_report.get('failure_repair', {}).get('count', 0) or 0)}" if pg316_live_ready else "PENDING", "status": "blocked", "note": "独立 network=none 容器复放；仍是 source-grounded adapter，不是 literal payload"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg317", "label": "PG-317 multi-missing question anchor", "value": f"ASK {float(dict(pg317_metrics.get('holdout_anchor_question_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · variant {float(dict(pg317_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%" if pg317_ready else "PENDING", "status": "blocked", "note": "多缺失槽位必须先问；离线门通过但不能代替族外 live"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg317-live", "label": "PG-317 fresh GET/POST replay", "value": f"variant {int(pg317_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg317_live_counts.get('variant_role_count', 0) or 0)} · repair {int(pg317_live_report.get('failure_repair', {}).get('correct', 0) or 0)}/{int(pg317_live_report.get('failure_repair', {}).get('count', 0) or 0)}" if pg317_live_ready else "PENDING", "status": "blocked", "note": "fresh network=none SQL row-shape；只证明受控复放闭环，不是通用漏洞能力"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg318", "label": "PG-318 family-heldout DOM/SQL replay", "value": f"typed {int(pg318_counts.get('typed_effect_count', 0) or 0)}/{int(pg318_counts.get('route_count', 0) or 0)} · ASK {float(pg318_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.0f}%" if pg318_ready else "PENDING", "status": "blocked", "note": "3 seed、6 routes、network=none；首轮真实失败已修复但 holdout 结果仍不晋级"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg319", "label": "PG-319 cross-implementation Rule-IR", "value": f"route ASK {float(dict(pg319_metrics.get('implementation_route_question_min') or {}).get('min', 0.0) or 0.0) * 100:.0f}% · family ASK {float(dict(pg319_metrics.get('pg318_family_question_min') or {}).get('min', 0.0) or 0.0) * 100:.0f}%" if pg319_ready else "PENDING", "status": "blocked", "note": "family holdout 暴露离线角色混淆；new-only 遗忘与 replay-mix 分开记录"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg320", "label": "PG-320 observation lattice", "value": f"lattice ASK {float(dict(pg320_metrics.get('lattice_question_min') or {}).get('min', 0.0) or 0.0) * 100:.0f}% · live variant {int(pg320_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg320_live_counts.get('variant_role_count', 0) or 0)}" if pg320_live_ready else "PENDING", "status": "blocked", "note": "离线 ASK 修复后仍需真实 candidate/reference/negative 回放；不以分类分数代替"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg321", "label": "PG-321 role-conditioned variant replay", "value": f"live typed {int(pg321_live_counts.get('typed_effect_count', 0) or 0)}/{int(pg321_live_counts.get('route_count', 0) or 0)} · ASK {float(pg321_live_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.0f}%" if pg321_live_ready else "PENDING", "status": "blocked", "note": "54/54 role variant、18/18 typed、0 negative violation；单一实现仍不足以晋级通用能力"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg322", "label": "PG-322 cross-implementation decoy", "value": f"ASK {float(dict(pg322_metrics.get('ask_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · hard allow {float(dict(pg322_metrics.get('hard_false_allow_max') or {}).get('max', 0.0) or 0.0):.0f}" if pg322_ready else "PENDING", "status": "blocked", "note": "离线 hard-negative/ASK 暴露缺口；因此启动 PG-323 anchor 修复"})
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg323", "label": "PG-323 decoy/ASK anchor + VulnerableApp", "value": f"live variant {int(pg323_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg323_live_counts.get('variant_role_count', 0) or 0)} · typed {int(pg323_live_counts.get('positive_typed_effect_count', 0) or 0)}/{int(pg323_live_counts.get('positive_route_count', 0) or 0)} · ASK {float(pg323_live_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.0f}%" if pg323_live_ready else "PENDING", "status": "blocked", "note": "3 seed、18 fresh routes、GET/POST、0 negative violation；单一实现 evaluation-only，不是任意网址漏洞能力"})
    pg324_metric_value = {
        "awaiting_fresh_replay": "PENDING",
        "stale_contract": "STALE CONTRACT",
        "incomplete": "INCOMPLETE",
        "completed_evaluation_only": f"evaluator typed {int(pg324_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('positive_route_count', 0) or 0)} · variant {int(pg324_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('variant_role_count', 0) or 0)}",
    }.get(str(pg324_contract.get("artifact_status")), "INCOMPLETE")
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg324", "label": "PG-324 Juice Shop source-heldout v2", "value": pg324_metric_value, "status": "blocked", "note": f"{str(pg324_contract.get('artifact_status', 'awaiting_fresh_replay'))}；hard gate 通过也只代表本地 evaluator evidence，模型能力、训练和长期记忆晋级均关闭"})
    pg325_metric_value = {
        "awaiting_fresh_replay": "PENDING",
        "stale_contract": "STALE CONTRACT",
        "incomplete": "INCOMPLETE",
        "completed_evaluation_only": f"typed {int(pg325_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('positive_route_count', 0) or 0)} · variant {int(pg325_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('variant_role_count', 0) or 0)} · ASK {int(pg325_contract.get('counts', {}).get('multi_missing_question_rows', 0) or 0)} · repair {int(pg325_contract.get('counts', {}).get('failure_repair_correct_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('failure_repair_count', 0) or 0)}",
    }.get(str(pg325_contract.get("artifact_status")), "INCOMPLETE")
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg325", "label": "PG-325 SQL family-heldout GET/POST", "value": pg325_metric_value, "status": "blocked", "note": f"{str(pg325_contract.get('artifact_status', 'awaiting_fresh_replay'))} · audit={str(pg325_contract.get('audit_status', 'not_embedded'))}；role-bound belief evidence 重复=0，但 promotion、训练和漏洞声明仍关闭"})
    pg326_metric_value = {
        "awaiting_matrix": "PENDING",
        "stale_contract": "STALE CONTRACT",
        "incomplete": "INCOMPLETE",
        "completed_evaluation_matrix_blocked": f"observed typed {int(pg326_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg326_contract.get('counts', {}).get('positive_route_count', 0) or 0)} · variant {int(pg326_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg326_contract.get('counts', {}).get('variant_role_count', 0) or 0)} · ASK {int(pg326_contract.get('counts', {}).get('multi_missing_question_rows', 0) or 0)} · forgetting pair={bool((pg326_contract.get('matrix_gate_checks') or {}).get('forgetting_pair'))}",
    }.get(str(pg326_contract.get("artifact_status")), "INCOMPLETE")
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg326", "label": "PG-326 cross-implementation / forgetting matrix", "value": pg326_metric_value, "status": "blocked", "note": f"{str(pg326_contract.get('artifact_status', 'awaiting_matrix'))} · audit={str(pg326_contract.get('audit_status', 'not_embedded'))}；观察分数与 strict contract/forgetting pair 分开，promotion 关闭"})
    pg327_metrics = dict(pg327_contract.get("metrics") or {})
    pg327_metric_value = {
        "awaiting_training": "PENDING",
        "stale_contract": "STALE CONTRACT",
        "incomplete": "INCOMPLETE",
        "completed_remote_a800_candidate": f"GPU0 ASK {float(dict(pg327_metrics.get('ask_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · variant {float(dict(pg327_metrics.get('implementation_variant_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}% · hard allow {float(dict(pg327_metrics.get('hard_false_allow_max') or {}).get('max', 0.0) or 0.0):.0f}",
    }.get(str(pg327_contract.get("artifact_status")), "INCOMPLETE")
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg327", "label": "PG-327 A800 replay-mix 候选训练", "value": pg327_metric_value, "status": "blocked", "note": "远程 GPU0 provenance 已记录；strict schema 与 paired forgetting replay 未通过，不能晋级模型、记忆或漏洞能力"})
    pg327b_counts = dict(pg327b_contract.get("counts") or {})
    pg327b_metric_value = {
        "awaiting_paired_replay": "PENDING",
        "stale_contract": "STALE CONTRACT",
        "incomplete": "INCOMPLETE",
        "audit_blocked": "AUDIT BLOCKED",
        "completed_paired_fresh_replay": f"before/after typed {int(pg327b_counts.get('before_typed_effect_count', 0) or 0)}/{int(pg327b_counts.get('after_typed_effect_count', 0) or 0)} · action {int(pg327b_counts.get('before_failure_action_changed_count', 0) or 0)}/{int(pg327b_counts.get('after_failure_action_changed_count', 0) or 0)}",
    }.get(str(pg327b_contract.get("artifact_status")), "INCOMPLETE")
    snapshot["capability"]["metrics"].insert(-2, {"id": "pg327b", "label": "PG-327B paired fresh forgetting replay", "value": pg327b_metric_value, "status": "blocked", "note": f"audit={str(pg327b_contract.get('audit_status', 'not_embedded'))} · paired={bool(pg327b_contract.get('paired_replay_present'))}；uniform source-row contract 仍需 PG-327C"})
    pg331_axis_coverage = dict(pg331_contract.get("axis_coverage") or {})
    pg331_missing_axes = list(pg331_contract.get("missing_axes") or [])
    pg331_counts = dict(pg331_vocab.get("counts") or {})
    pg331_capacity_variants = [dict(item) for item in list(pg331_capacity.get("variants") or []) if isinstance(item, dict)]
    pg331_legacy_variant = next((item for item in pg331_capacity_variants if str((item.get("config") or {}).get("id")) == "pg322_legacy"), {})
    pg331_capacity_candidates = [item for item in pg331_capacity_variants if str((item.get("config") or {}).get("id")) != "pg322_legacy"]
    pg331_capacity_pass = bool(pg331_capacity.get("status") == "passed" and any(bool(item.get("capacity_pass")) for item in pg331_capacity_candidates))
    pg331_model = {
        "status": str(pg331_contract.get("artifact_status", "awaiting_audit")),
        "audit_status": str(pg331_contract.get("audit_status", "not_run")),
        "record_count": int(pg331_contract.get("record_count", 0) or 0),
        "unique_sequence_ratio": float(pg331_contract.get("unique_sequence_ratio", 0.0) or 0.0),
        "axis_count": int(pg331_contract.get("axis_count", 0) or 0),
        "axis_coverage": pg331_axis_coverage,
        "missing_axes": pg331_missing_axes,
        "context_target_alignment": float(pg331_contract.get("context_target_alignment", 0.0) or 0.0),
        "split_isolation_status": str(pg331_contract.get("split_isolation_status", "unknown")),
        "context_forbidden_literal_count": int(pg331_contract.get("context_forbidden_literal_count", 0) or 0),
        "vocabulary_status": str(pg331_contract.get("vocabulary_status", "missing")),
        "vocabulary_training_allowed": bool(pg331_contract.get("vocabulary_training_allowed")),
        "vocabulary_context_total": int(pg331_counts.get("context_total", 0) or 0),
        "vocabulary_target_total": int(pg331_counts.get("target_total", 0) or 0),
        "vocabulary_shared_total": int(pg331_counts.get("shared_total", 0) or 0),
        "ontology_sha256": str(pg331_vocab.get("ontology_sha256", "")),
        "audit_evidence_hash": str(pg331_contract.get("audit_evidence_hash", "")),
        "vocabulary_evidence_hash": str(pg331_contract.get("vocabulary_evidence_hash", "")),
        "capacity_audit_status": str(pg331_capacity.get("status", "missing")),
        "capacity_audit_evidence_hash": str(pg331_capacity.get("audit_sha256", ""))[:16],
        "input_vocabulary_size": int(pg331_capacity.get("input_vocabulary_size", 0) or 0),
        "target_vocabulary_size": int(pg331_capacity.get("target_vocabulary_size", 0) or 0),
        "representative_model_context_tokens": int((pg331_capacity.get("representative_page") or {}).get("model_context_tokens", 0) or 0),
        "required_context_window": int(pg331_capacity.get("required_context_window", 0) or 0),
        "legacy_max_length": int((pg331_legacy_variant.get("config") or {}).get("max_length", 0) or 0),
        "legacy_capacity_pass": bool(pg331_legacy_variant.get("capacity_pass")),
        "capacity_candidates_pass": pg331_capacity_pass,
        "capacity_variants": pg331_capacity_variants,
        "promotion_blocked": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    pg331_model["legacy_web_manifest"] = dict(pg331_readonly_sources.get("legacy_web_manifest") or {})
    pg331_model["remote_a800_readonly_preflight"] = dict(pg331_readonly_sources.get("remote_a800_readonly_preflight") or {})
    pg331_model["readonly_evidence_training_allowed"] = False
    pg331_model["training_allowed"] = False
    pg331_model["live_source_collection"] = dict(pg331_source_collection_projection)
    pg331_model["typed_source_rows"] = dict(pg331_typed_source_rows_projection)
    pg331_model["typed_capacity"] = dict(pg331_typed_capacity_projection)
    pg331_model["train_holdout_diagnostic_v2"] = dict(pg331_holdout_diagnostic)
    pg331_model["pg332_extended"] = dict(pg332_extended)
    pg331_model["pg333_webgoat"] = dict(pg333_webgoat_projection)
    pg331_model["pg333_cross_impl"] = dict(pg333_cross_projection)
    pg331_model["pg334_process_tokens"] = dict(pg334_process_projection)
    pg331_model["pg335_real_process_tokens"] = dict(pg335_process_projection)
    pg331_model["pg337_cross_impl_process_tokens"] = dict(pg337_process_projection)
    pg331_model["pg338_information_preserving_process_tokens"] = dict(pg338_process_projection)
    pg331_model["pg339_multi_shape_information_preserving"] = dict(pg339_shape_projection)
    pg331_model["pg340_balanced_axis_representation"] = dict(pg340_axis_projection)
    pg331_model["pg341_target_conditioned_two_view"] = dict(pg341_target_projection)
    pg331_model["pg342_full_axis_failure_repair"] = dict(pg342_failure_projection)
    legacy_projection = pg331_model["legacy_web_manifest"]
    remote_projection = pg331_model["remote_a800_readonly_preflight"]
    pg331_model["legacy_manifest_status"] = str(legacy_projection.get("status", "not_run"))
    pg331_model["legacy_manifest_page_count"] = int(legacy_projection.get("page_count", 0) or 0)
    pg331_model["legacy_manifest_route_count"] = int(legacy_projection.get("route_count", 0) or 0)
    pg331_model["legacy_manifest_request_response_row_count"] = int(legacy_projection.get("request_response_row_count", 0) or 0)
    pg331_model["legacy_missing_observations"] = list(legacy_projection.get("missing_observations") or [])
    pg331_model["legacy_missing_observation_count"] = int(legacy_projection.get("missing_observation_count", 0) or 0)
    pg331_model["remote_a800_preflight_status"] = str(remote_projection.get("status", "not_run"))
    pg331_model["remote_a800_gpu0"] = dict(remote_projection.get("gpu0") or {})
    pg331_model["remote_a800_gpu0_resource_status"] = str(remote_projection.get("gpu0_resource_status", "not_observed"))
    pg331_model["source_collection_status"] = str(pg331_source_collection_projection.get("report_status", "pending"))
    pg331_model["source_collection_report_status"] = str(pg331_source_collection_projection.get("report_status", "pending"))
    pg331_model["source_collection_artifact_status"] = str(pg331_source_collection_projection.get("artifact_status", "pending"))
    pg331_model["source_row_collection_status"] = str(pg331_source_collection_projection.get("dataset_status", "pending"))
    pg331_model["source_collection_route_count"] = int(pg331_source_collection_projection.get("route_count", 0) or 0)
    pg331_model["source_collection_get_count"] = int(pg331_source_collection_projection.get("get_count", 0) or 0)
    pg331_model["source_collection_post_count"] = int(pg331_source_collection_projection.get("post_count", 0) or 0)
    pg331_model["source_collection_parameterized_get_count"] = int(pg331_source_collection_projection.get("parameterized_get_count", 0) or 0)
    pg331_model["source_collection_target_contacted_count"] = int(pg331_source_collection_projection.get("target_contacted_count", 0) or 0)
    pg331_model["source_collection_ask_count"] = int(pg331_source_collection_projection.get("ask_count", 0) or 0)
    pg331_model["source_collection_training_eligible"] = False
    snapshot["capability"]["model"]["pg331_information_preservation"] = pg331_model
    snapshot["capability"]["model"]["pg332_diagnostic"] = {
        **dict(pg332_extended),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg333_diagnostic"] = {
        **dict(pg333_webgoat_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg333_cross_impl_diagnostic"] = {
        **dict(pg333_cross_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg334_process_token_diagnostic"] = {
        **dict(pg334_process_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg335_real_process_token_diagnostic"] = {
        **dict(pg335_process_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg336_real_failure_process_token_diagnostic"] = {
        **dict(pg336_process_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg337_cross_impl_process_token_diagnostic"] = {
        **dict(pg337_process_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg338_information_preserving_process_token_diagnostic"] = {
        **dict(pg338_process_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg339_multi_shape_information_preserving_diagnostic"] = {
        **dict(pg339_shape_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg340_balanced_axis_representation_diagnostic"] = {
        **dict(pg340_axis_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg341_target_conditioned_two_view_diagnostic"] = {
        **dict(pg341_target_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg342_full_axis_failure_repair_diagnostic"] = {
        **dict(pg342_failure_projection),
        "evaluator_only": True,
        "training_eligible": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["capability"]["model"]["pg331_source_collection"] = {
        **pg331_source_collection_projection,
        "evaluator_only": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_material_available": False,
    }
    snapshot["source_reports"].append({"name": pg331_source_row_audit_name, "updated_at": _report_time(pg331_source_row_audit_name), "sha256": str(pg331_source_row_audit.get("audit_sha256", ""))})
    for _name, _doc, _key in (
        (pg333_cross_dataset_name, pg333_cross_dataset, "dataset_sha256"),
        (pg333_cross_source_audit_name, pg333_cross_source_audit, "audit_sha256"),
        (pg333_cross_information_name, pg333_cross_information, "dataset_information_sha256"),
        (pg333_cross_vocabulary_name, pg333_cross_vocabulary, "vocabulary_sha256"),
        (pg333_cross_capacity_name, pg333_cross_capacity, "audit_sha256"),
        (pg333_cross_a800_name, pg333_cross_a800, "report_sha256"),
    ):
        if _doc:
            snapshot["source_reports"].append({"name": _name, "updated_at": _report_time(_name), "sha256": str(_doc.get(_key, ""))})
    for _name, _doc, _key in (
        (pg339_shape_dataset_name, pg339_shape_dataset, "dataset_sha256"),
        (pg339_shape_audit_name, pg339_shape_audit, "audit_sha256"),
        (pg339_shape_vocab_name, pg339_shape_vocab, "vocabulary_sha256"),
        (pg339_shape_a800_name, pg339_shape_a800, "report_sha256"),
        (pg340_axis_dataset_name, pg340_axis_dataset, "dataset_sha256"),
        (pg340_axis_audit_name, pg340_axis_audit, "audit_sha256"),
        (pg340_axis_vocab_name, pg340_axis_vocab, "vocabulary_sha256"),
        (pg340_axis_a800_name, pg340_axis_a800, "report_sha256"),
        (pg341_target_dataset_name, pg341_target_dataset, "dataset_sha256"),
        (pg341_target_audit_name, pg341_target_audit, "audit_sha256"),
        (pg341_target_vocab_name, pg341_target_vocab, "vocabulary_sha256"),
        (pg342_failure_dataset_name, pg342_failure_dataset, "dataset_sha256"),
        (pg342_failure_audit_name, pg342_failure_audit, "audit_sha256"),
        (pg342_failure_vocab_name, pg342_failure_vocab, "vocabulary_sha256"),
        (pg342_failure_source_name, pg342_failure_source, "report_sha256"),
        (pg342_failure_a800_name, pg342_failure_a800, "report_sha256"),
    ):
        if _doc:
            snapshot["source_reports"].append({"name": _name, "updated_at": _report_time(_name), "sha256": str(_doc.get(_key, ""))})
    for _name, _doc in zip(pg341_target_a800_candidates, pg341_target_a800_reports):
        if _doc:
            snapshot["source_reports"].append({"name": _name, "updated_at": _report_time(_name), "sha256": str(_doc.get("report_sha256", ""))})
    pg331_model["source_row_audit_status"] = str(pg331_source_row_audit.get("status", "not_run"))
    pg331_model["source_row_record_count"] = int(pg331_source_row_audit.get("record_count", 0) or 0)
    pg331_model["source_row_training_eligible_count"] = int(pg331_source_row_audit.get("training_eligible_count", 0) or 0)
    pg331_model["source_row_audit_evidence_hash"] = str(pg331_source_row_audit.get("audit_sha256", ""))
    snapshot["source_reports"].append({"name": pg331_loopback_smoke_name, "updated_at": _report_time(pg331_loopback_smoke_name), "sha256": str(pg331_loopback_smoke.get("report_sha256", ""))})
    pg331_model["loopback_smoke_status"] = str(pg331_loopback_smoke.get("status", "unknown")) if pg331_loopback_smoke else "not_run"
    pg331_model["loopback_smoke_target_count"] = len(list(pg331_loopback_smoke.get("targets") or []))
    pg331_model["loopback_smoke_target_contacted_count"] = int(pg331_loopback_smoke.get("target_contacted_count", 0) or 0)
    pg331_model["loopback_smoke_training_eligible"] = bool(pg331_loopback_smoke.get("training_eligible"))
    legacy_projection = dict(pg331_readonly_sources.get("legacy_web_manifest") or {})
    remote_projection = dict(pg331_readonly_sources.get("remote_a800_readonly_preflight") or {})
    snapshot["source_reports"].append({"name": pg331_legacy_manifest_name, "updated_at": _report_time(pg331_legacy_manifest_name), "sha256": str(legacy_projection.get("audit_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_remote_preflight_name, "updated_at": _report_time(pg331_remote_preflight_name), "sha256": str(remote_projection.get("preflight_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_source_collection_name, "updated_at": _report_time(pg331_source_collection_name), "sha256": str(pg331_source_collection_projection.get("report_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_source_dataset_name, "updated_at": _report_time(pg331_source_dataset_name), "sha256": str(pg331_source_collection_projection.get("dataset_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_typed_source_rows_report_name, "updated_at": _report_time(pg331_typed_source_rows_report_name), "sha256": str(pg331_typed_source_rows_projection.get("report_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_typed_source_rows_audit_name, "updated_at": _report_time(pg331_typed_source_rows_audit_name), "sha256": str(pg331_typed_source_rows_projection.get("audit_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_typed_sidecars_name, "updated_at": _report_time(pg331_typed_sidecars_name), "sha256": str(pg331_typed_source_rows_projection.get("sidecar_evidence_hash", ""))})
    snapshot["source_reports"].append({"name": pg331_typed_capacity_name, "updated_at": _report_time(pg331_typed_capacity_name), "sha256": str(pg331_typed_capacity_projection.get("audit_evidence_hash", ""))})
    for name, document, key in (
        (pg332_dvwa_get_report_name, pg332_dvwa_get_report, "report_sha256"),
        (pg332_dvwa_get_audit_name, pg332_dvwa_get_audit, "audit_sha256"),
        (pg332_dvwa_get_sidecars_name, pg332_dvwa_get_sidecars, "sidecars_sha256"),
        (pg332_dvwa_get_dataset_name, pg332_dvwa_get_dataset, "dataset_sha256"),
        (pg332_dvwa_post_report_name, pg332_dvwa_post_report, "report_sha256"),
        (pg332_dvwa_post_audit_name, pg332_dvwa_post_audit, "audit_sha256"),
        (pg332_dvwa_post_sidecars_name, pg332_dvwa_post_sidecars, "sidecars_sha256"),
        (pg332_dvwa_post_dataset_name, pg332_dvwa_post_dataset, "dataset_sha256"),
        (pg332_cross_audit_name, pg332_cross_audit, "audit_sha256"),
        (pg332_information_name, pg332_information, "dataset_information_sha256"),
        (pg332_capacity_name, pg332_capacity, "audit_sha256"),
        (pg332_a800_name, pg332_a800, "report_sha256"),
        (pg333_webgoat_report_name, pg333_webgoat_report, "report_sha256"),
        (pg333_webgoat_audit_name, pg333_webgoat_audit, "audit_sha256"),
        (pg333_webgoat_sidecars_name, pg333_webgoat_sidecars, "sidecars_sha256"),
        (pg333_webgoat_dataset_name, pg333_webgoat_dataset, "dataset_sha256"),
        (pg334_process_dataset_name, pg334_process_dataset, "dataset_sha256"),
        (pg334_process_audit_name, pg334_process_audit, "audit_sha256"),
        (pg334_process_vocab_name, pg334_process_vocab, "vocabulary_sha256"),
        (pg334_process_a800_name, pg334_process_a800, "report_sha256"),
        (pg335_process_dataset_name, pg335_process_dataset, "dataset_sha256"),
        (pg335_process_audit_name, pg335_process_audit, "audit_sha256"),
        (pg335_process_vocab_name, pg335_process_vocab, "vocabulary_sha256"),
        (pg335_process_a800_name, pg335_process_a800, "report_sha256"),
        (pg336_process_dataset_name, pg336_process_dataset, "dataset_sha256"),
        (pg336_process_audit_name, pg336_process_audit, "audit_sha256"),
        (pg336_process_vocab_name, pg336_process_vocab, "vocabulary_sha256"),
        (pg336_process_a800_name, pg336_process_a800, "report_sha256"),
        (pg337_process_dataset_name, pg337_process_dataset, "dataset_sha256"),
        (pg337_process_audit_name, pg337_process_audit, "audit_sha256"),
        (pg337_process_vocab_name, pg337_process_vocab, "vocabulary_sha256"),
        (pg337_process_a800_name, pg337_process_a800, "report_sha256"),
        (pg338_process_dataset_name, pg338_process_dataset, "dataset_sha256"),
        (pg338_process_audit_name, pg338_process_audit, "audit_sha256"),
        (pg338_process_vocab_name, pg338_process_vocab, "vocabulary_sha256"),
        (pg338_process_a800_name, pg338_process_a800, "report_sha256"),
    ):
        snapshot["source_reports"].append({"name": name, "updated_at": _report_time(name), "sha256": str(document.get(key, ""))})
    for name, key in (
        (pg331_holdout_dataset_name, "report"),
        (pg331_holdout_source_audit_name, "source_audit"),
        (pg331_holdout_vocab_name, "vocabulary"),
        (pg331_holdout_information_name, "information"),
        (pg331_holdout_capacity_name, "capacity"),
        (pg331_holdout_plan_name, "plan"),
    ):
        snapshot["source_reports"].append({"name": name, "updated_at": _report_time(name), "sha256": str(dict(pg331_holdout_diagnostic.get("evidence_hashes") or {}).get(key, ""))})
    pg331_metric_value = (
        f"{len(pg331_missing_axes)}/{int(pg331_contract.get('axis_count', 0) or 0)} axes missing · "
        f"align {float(pg331_contract.get('context_target_alignment', 0.0) or 0.0) * 100:.1f}%"
        if pg331_contract.get("audit_status")
        else "PENDING"
    )
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331",
        "label": "PG-331 whole-web token preservation",
        "value": pg331_metric_value,
        "status": "blocked",
        "note": "词表由七轴 ontology 生成且不删低频 token；任何缺轴、熵/消融/分割失败都禁止训练、记忆和能力声明",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331-capacity",
        "label": "PG-331 token/model capacity",
        "value": f"legacy max{int((pg331_legacy_variant.get('config') or {}).get('max_length', 0) or 0)} FAIL · required {int(pg331_capacity.get('required_context_window', 0) or 0)} · candidate {'PASS' if pg331_capacity_pass else 'BLOCKED'}",
        "status": "blocked",
        "note": "上下文窗口不足时必须分 chunk；不能删词或截断来制造好看的分数。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331a-source-row",
        "label": "PG-331A strict source-row",
        "value": f"{int(pg331_source_row_audit.get('record_count', 0) or 0)} rows · {int(pg331_source_row_audit.get('training_eligible_count', 0) or 0)} trainable" if pg331_source_row_audit.get("status") == "passed" else "dataset missing · ASK only",
        "status": "pass" if pg331_source_row_audit.get("status") == "passed" else "blocked",
        "note": "sidecar 与 model context 分离；缺轴、原始旁路字段或 evaluator 不完整时不训练。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331a-loopback-smoke",
        "label": "PG-331A loopback observation",
        "value": f"{len(list(pg331_loopback_smoke.get('targets') or []))} targets · {str(pg331_loopback_smoke.get('status', 'unknown'))}" if pg331_loopback_smoke else "PENDING",
        "status": "blocked" if pg331_loopback_smoke else "partial",
        "note": "只验证本地页面结构采集；没有 fresh reset/typed evaluator，不能生成训练样本。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331-legacy-manifest",
        "label": "PG-331 historical web-manifest coverage",
        "value": f"{int(legacy_projection.get('page_count', 0) or 0)} pages · {int(legacy_projection.get('route_count', 0) or 0)} routes · {int(legacy_projection.get('missing_observation_count', 0) or 0)} missing axes",
        "status": "blocked",
        "note": "历史 baseline GET 清单只用于词表/字段覆盖诊断；不得当作训练集。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331-a800-resource",
        "label": "PG-331 weekend remote A800 GPU0 resource",
        "value": f"GPU0 {str(remote_projection.get('gpu0_resource_status', 'not_observed'))} · training {'allowed' if bool(remote_projection.get('training_allowed_now')) else 'blocked'}",
        "status": "blocked",
        "note": "资源预检只说明 GPU0 可用；source-row、信息保真和 fresh holdout 门仍优先。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331-source-collection",
        "label": "PG-331 Pikachu source collection",
        "value": (
            f"routes {int(pg331_source_collection_projection.get('route_count', 0) or 0)} · "
            f"GET/POST {int(pg331_source_collection_projection.get('get_count', 0) or 0)}/"
            f"{int(pg331_source_collection_projection.get('post_count', 0) or 0)} · "
            f"parameterized GET {int(pg331_source_collection_projection.get('parameterized_get_count', 0) or 0)} · "
            f"ASK {int(pg331_source_collection_projection.get('ask_count', 0) or 0)}"
        ) if pg331_source_collection_projection.get("artifact_status") not in {"pending", "blocked_incomplete"} else "PENDING",
        "status": "blocked",
        "note": "真实 GET/POST 结构仅为诊断/ASK；typed evaluator 缺失，training、memory、payload catalog 和 vulnerability claim 全部关闭。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331-typed-source-rows",
        "label": "PG-331 typed source-row evaluator evidence",
        "value": (
            f"routes {int(pg331_typed_source_rows_projection.get('route_count', 0) or 0)} · "
            f"rows {int(pg331_typed_source_rows_projection.get('row_count', 0) or 0)} · "
            f"typed positive {int(pg331_typed_source_rows_projection.get('typed_positive', 0) or 0)} · "
            f"GET/POST {int(pg331_typed_source_rows_projection.get('get_count', 0) or 0)}/"
            f"{int(pg331_typed_source_rows_projection.get('post_count', 0) or 0)}"
        ) if pg331_typed_source_rows_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "typed positive 仅为 evaluator-side response-shape evidence；audit/operator review 未通过前不得训练或晋级。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg331-typed-capacity",
        "label": "PG-331 typed source-row model capacity",
        "value": (
            f"context {int(pg331_typed_capacity_projection.get('context_min', 0) or 0)}-"
            f"{int(pg331_typed_capacity_projection.get('context_max', 0) or 0)} · "
            f"required {int(pg331_typed_capacity_projection.get('required_context_window', 0) or 0)} · "
            f"vocab {int(pg331_typed_capacity_projection.get('model_vocabulary_size', 0) or 0)} · "
            f"truncation {'risk' if pg331_typed_capacity_projection.get('truncation_risk') else 'clear'}"
        ) if pg331_typed_capacity_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "capacity audit is planning evidence only; information/promotion gates remain closed and no training is started.",
    })
    pg332_cross = dict(pg332_extended.get("cross_impl") or {})
    pg332_info = dict(pg332_extended.get("information") or {})
    pg332_a800_projection = dict(pg332_extended.get("a800_representation") or {})
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg332-cross-impl",
        "label": "PG-332 GET/POST cross-implementation diagnostic",
        "value": (
            f"rows {int(pg332_cross.get('record_count', 0) or 0)} · "
            f"typed {int(pg332_cross.get('typed_complete_count', 0) or 0)} · "
            f"fresh {int(pg332_cross.get('fresh_reset_complete_count', 0) or 0)} · "
            f"negative {int(pg332_cross.get('negative_control_complete_count', 0) or 0)} · "
            f"implementations {int(pg332_cross.get('implementation_count', 0) or 0)}"
        ) if pg332_extended.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "GET、持久状态 POST、候选/参考/阴性与复放均只作 evaluator 诊断；信息保真/第三实现/族外门未过，不能进入训练或宣称通用漏洞能力。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg332-information",
        "label": "PG-332 ontology information preservation",
        "value": (
            f"status {str(pg332_info.get('status', 'pending'))} · "
            f"axes {len(dict(pg332_info.get('axes') or {}))} · "
            f"accepted train rows {int(pg332_info.get('accepted_training_eligible_count', 0) or 0)} · "
            f"window {int(dict(pg332_extended.get('capacity') or {}).get('required_context_window', 0) or 0)}"
        ) if pg332_extended.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "A800 表示 smoke 已完成，但 information_promotion_gate=false；不可用低 loss 或 GPU 完成替代字段熵、消融和实现留出。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg332-a800-representation",
        "label": "PG-332 remote A800 context-only smoke",
        "value": (
            f"seeds {len(list(pg332_a800_projection.get('seeds') or []))} · "
            f"holdout rows {max([int(item.get('holdout_context_rows', 0) or 0) for item in list(pg332_a800_projection.get('loss') or [])] or [0])} · "
            f"gate {str(pg332_a800_projection.get('information_gate_status', 'pending'))}"
        ) if pg332_extended.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "只读 context token 的表示预训练候选；不读取 target token，不是 Rule-IR/主动探测/漏洞 payload 训练。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg333-webgoat",
        "label": "PG-333 WebGoat third-implementation method-shape canary",
        "value": (
            f"rows {int(pg333_webgoat_projection.get('source_row_count', 0) or 0)} · "
            f"GET/POST {int(dict(pg333_webgoat_projection.get('methods') or {}).get('GET', 0) or 0)}/"
            f"{int(dict(pg333_webgoat_projection.get('methods') or {}).get('POST', 0) or 0)} · "
            f"typed {int(pg333_webgoat_projection.get('typed_positive_route_seed_count', 0) or 0)} · "
            f"negative {int(pg333_webgoat_projection.get('negative_violation_count', 0) or 0)}"
        ) if pg333_webgoat_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "WebGoat 仅验证页面/重定向形状与 method 组合，不是 XSS/SQL/认证绕过结果；source audit、族外信息审计和 ASK/失败修复门未过前不训练。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg333-cross-implementation",
        "label": "PG-333 three-implementation GET/POST information diagnostic",
        "value": (
            f"rows {int(pg333_cross_projection.get('merged_record_count', 0) or 0)} · "
            f"impl {int(pg333_cross_projection.get('implementation_count', 0) or 0)} · "
            f"train {int(pg333_cross_projection.get('a800_train_rows', 0) or 0)} · "
            f"holdout {int(pg333_cross_projection.get('a800_holdout_rows', 0) or 0)} · "
            f"info {str(pg333_cross_projection.get('information_audit_status', 'pending'))}"
        ) if pg333_cross_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "只展示跨实现抽象信息和 context-only A800 smoke；不读取 target token，不代表 Rule-IR、漏洞或 payload 能力。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg334-process-tokens",
        "label": "PG-334 ASK/failure process-token diagnostic",
        "value": (
            f"rows {int(pg334_process_projection.get('record_count', 0) or 0)} · "
            f"ASK {int(pg334_process_projection.get('pre_question_count', 0) or 0)} · "
            f"negative {int(pg334_process_projection.get('negative_count', 0) or 0)} · "
            f"A800 {str(pg334_process_projection.get('a800_status', 'pending'))}"
        ) if pg334_process_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "去标识化过程 token 的 ASK/失败修复/负例表征 smoke；不读取 target token，不是漏洞或 payload 能力。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg335-real-process-tokens",
        "label": "PG-335 source-grounded ASK/failure diagnostic",
        "value": (
            f"source {int(pg335_process_projection.get('source_row_count', 0) or 0)} · "
            f"rows {int(pg335_process_projection.get('record_count', 0) or 0)} · "
            f"ASK {int(pg335_process_projection.get('ask_count', 0) or 0)} · "
            f"repair {int(pg335_process_projection.get('failure_count', 0) or 0)} · "
            f"negative {int(pg335_process_projection.get('negative_review_count', 0) or 0)}"
        ) if pg335_process_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "真实 source rows 锚定的逐轴 ASK/失败/阴性诊断；遮蔽行不是 real gold，信息审计/能力晋级仍关闭。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg336-real-failure-process-tokens",
        "label": "PG-336 real failure/ASK process tokens",
        "value": (
            f"rows {int(pg336_process_projection.get('record_count', 0) or 0)} · "
            f"ASK {int(pg336_process_projection.get('ask_preflight_count', 0) or 0)} · "
            f"repair {int(pg336_process_projection.get('failure_repair_count', 0) or 0)} · "
            f"negative {int(pg336_process_projection.get('negative_review_count', 0) or 0)} · "
            f"A800 {str(pg336_process_projection.get('a800_status', 'pending'))}"
        ) if pg336_process_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "真实 PG-325 失败动作变化已进入抽象过程轨道；单实现 seed holdout、信息/能力晋级仍关闭。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg337-cross-impl-process-tokens",
        "label": "PG-337 cross-implementation failure process",
        "value": (
            f"rows {int(pg337_process_projection.get('record_count', 0) or 0)} · "
            f"train/holdout {int(pg337_process_projection.get('train_count', 0) or 0)}/{int(pg337_process_projection.get('implementation_holdout_count', 0) or 0)} · "
            f"DVWA repair {int(pg337_process_projection.get('real_dvwa_failure_rows', 0) or 0)} · "
            f"A800 {str(pg337_process_projection.get('a800_status', 'pending'))}"
        ) if pg337_process_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "Pikachu 训练侧与 DVWA 独立实现留出的真实失败→修复/阴性过程 token；仍为表示候选，accepted rows、能力 SFT/RL 和晋级关闭。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg338-information-preserving-process-tokens",
        "label": "PG-338 full-axis information-preserving process",
        "value": (
            f"rows {int(pg338_process_projection.get('record_count', 0) or 0)} · "
            f"full-axis {int(pg338_process_projection.get('full_axis_rows', 0) or 0)} · "
            f"train/holdout {int(pg338_process_projection.get('train_count', 0) or 0)}/{int(pg338_process_projection.get('implementation_holdout_count', 0) or 0)} · "
            f"A800 {str(pg338_process_projection.get('a800_status', 'pending'))}"
        ) if pg338_process_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "保留整页七轴抽象序列；轴熵/字段消融仍是诊断门，不能把表征 loss 下降当作 Rule-IR capability 或 payload 能力。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg339-multi-shape-information-preserving",
        "label": "PG-339 multi-page-shape representation",
        "value": (
            f"rows {int(pg339_shape_projection.get('record_count', 0) or 0)} · "
            f"train/shape holdout {int(pg339_shape_projection.get('train_count', 0) or 0)}/"
            f"{int(pg339_shape_projection.get('shape_holdout_count', 0) or 0)} · "
            f"duplicates {int(pg339_shape_projection.get('duplicate_row_count', 0) or 0)} · "
            f"A800 {str(pg339_shape_projection.get('a800_status', 'pending'))}"
        ) if pg339_shape_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "PG-339 只做多页面形态的 full-axis 表征 smoke；shape holdout predictive entropy 未通过前，不能宣称 Rule-IR、漏洞判断或 payload 能力。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg340-balanced-axis-representation",
        "label": "PG-340 balanced multi-axis representation",
        "value": (
            f"rows {int(pg340_axis_projection.get('record_count', 0) or 0)} · "
            f"train/shape holdout {int(pg340_axis_projection.get('train_count', 0) or 0)}/"
            f"{int(pg340_axis_projection.get('shape_holdout_count', 0) or 0)} · "
            f"implementations {int(pg340_axis_projection.get('train_implementation_count', 0) or 0)}/"
            f"{int(pg340_axis_projection.get('holdout_implementation_count', 0) or 0)} · "
            f"A800 {str(pg340_axis_projection.get('a800_status', 'pending'))}"
        ) if pg340_axis_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "PG-340 用新实现留出补充轴变化；即使表征熵门通过，也不能替代 ASK/失败修复/typed capability gate。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg341-target-conditioned-two-view",
        "label": "PG-341 target-conditioned ASK/repair decoder",
        "value": (
            f"coarse train/holdout {int(pg341_target_projection.get('coarse_train_count', 0) or 0)}/"
            f"{int(pg341_target_projection.get('coarse_holdout_count', 0) or 0)} · "
            f"full-axis train/holdout {int(pg341_target_projection.get('full_axis_train_count', 0) or 0)}/"
            f"{int(pg341_target_projection.get('full_axis_holdout_count', 0) or 0)} · "
            f"ASK min {float(pg341_target_projection.get('a800_holdout_ask_recall_min', 0.0) or 0.0) * 100:.1f}% · "
            f"positive min {float(pg341_target_projection.get('a800_holdout_positive_recall_min', 0.0) or 0.0) * 100:.1f}% · "
            f"A800 {str(pg341_target_projection.get('a800_status', 'pending'))}"
        ) if pg341_target_projection.get('status') != "pending" else "PENDING",
        "status": "blocked",
        "note": "PG-341 只证明 coarse process 目标解码的诊断结果；full-axis 训练 split 尚无 ASK/repair/negative 目标，统一网页能力与 promotion 继续关闭。",
    })
    snapshot["capability"]["metrics"].insert(-2, {
        "id": "pg342-full-axis-failure-repair",
        "label": "PG-342 full-axis failure→repair representation",
        "value": (
            f"rows {int(pg342_failure_projection.get('record_count', 0) or 0)} · "
            f"train/implementation holdout {int(pg342_failure_projection.get('train_count', 0) or 0)}/"
            f"{int(pg342_failure_projection.get('implementation_holdout_count', 0) or 0)} · "
            f"GET/POST {int(pg342_failure_projection.get('get_count', 0) or 0)}/"
            f"{int(pg342_failure_projection.get('post_count', 0) or 0)} · "
            f"entropy drop {float(pg342_failure_projection.get('a800_entropy_drop_max', 0.0) or 0.0):.4f} · "
            f"A800 {str(pg342_failure_projection.get('a800_status', 'pending'))}"
        ) if pg342_failure_projection.get("status") != "pending" else "PENDING",
        "status": "blocked",
        "note": "PG-342 已补真实 failure→repair/negative 轨迹并完成 context-only A800 表征 smoke；非 document 轴依赖仍不足，不能进入 ASK/Rule-IR SFT/RL 或长期记忆。",
    })
    snapshot["capability"]["limits"].append(
        f"PG-331 当前 {int(pg331_contract.get('record_count', 0) or 0)} 条记录仅作审计："
        f"unique sequence ratio={float(pg331_contract.get('unique_sequence_ratio', 0.0) or 0.0):.6f}、"
        f"{len(pg331_missing_axes)}/{int(pg331_contract.get('axis_count', 0) or 0)} 轴缺失、"
        f"context-target alignment={float(pg331_contract.get('context_target_alignment', 0.0) or 0.0):.6f}；"
        "旧数据不能进入训练，下一步必须补齐整网页 token 观测。"
    )
    if pg312_ready:
        snapshot["capability"]["limits"].append(f"PG-312 宽 symbolic checkpoint 在单个 Pikachu 实现的 4 条 GET/POST 路由上触发候选发送 4/4、typed effect 4/4、false positive=0；这首次证明模型计划能进入真实受控回放，但候选 wire 仍由 source-grounded adapter 提供，且只有一个实现，不能等同于已学会通用 payload 生成。")
    if pg313_ready:
        snapshot["capability"]["limits"].append(f"PG-313 离线三 seed：question 最差 {float(dict(pg313_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}%、base slot 最差 {float(dict(pg313_metrics.get('holdout_bound_base_slot_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%、variant 最差 {float(dict(pg313_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%，hard-negative false-allow 最高 {float(dict(pg313_metrics.get('hard_bound_false_allow') or {}).get('max', 0.0) or 0.0):.0f}；因此只能算研究候选，不能算 payload 生成能力。")
    if pg314_ready:
        snapshot["capability"]["limits"].append(f"PG-314 在独立 image 上用 network=none 完成 {int(pg314_counts.get('model_variant_role_count', 0) or 0)} 次抽象变体选择，variant exact={int(pg314_counts.get('model_variant_exact_count', 0) or 0)}/{int(pg314_counts.get('model_variant_role_count', 0) or 0)}、typed row-shape={int(pg314_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg314_counts.get('route_count', 0) or 0)}、negative violation={int(pg314_counts.get('negative_lane_violation_count', 0) or 0)}；但只覆盖 SQL GET/POST 两条路由，不能等同于通用 payload 能力。")
    if pg315_ready:
        snapshot["capability"]["limits"].append(f"PG-315 三个 seed 全部真实复放：question worst={float(pg315_worst.get('question_recall_min', 0.0) or 0.0) * 100:.1f}%，variant worst={float(pg315_worst.get('variant_exact_min', 0.0) or 0.0) * 100:.1f}%，failure repair/abstain worst={float(pg315_worst.get('repair_abstain_min', 0.0) or 0.0) * 100:.1f}%，negative violations={int(pg315_worst.get('negative_lane_violation_max', 0) or 0)}；这明确证明 best-seed 的 PG-314 成功不能代表稳定能力。")
    if pg316_ready:
        snapshot["capability"]["limits"].append(f"PG-316 加权 repair/variant 训练把 holdout repair exact 最差提高到 {float(dict(pg316_metrics.get('holdout_repair_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%、variant exact 最差 {float(dict(pg316_metrics.get('holdout_variant_exact') or {}).get('min', 0.0) or 0.0) * 100:.1f}%、hard false-allow=0，但 question 最差仍 {float(dict(pg316_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.1f}%。")
    if pg316_live_ready:
        snapshot["capability"]["limits"].append(f"PG-316 最佳 seed 独立 live 复放：variant {int(pg316_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg316_live_counts.get('variant_role_count', 0) or 0)}、typed {int(pg316_live_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg316_live_counts.get('route_count', 0) or 0)}、failure repair {int(pg316_live_report.get('failure_repair', {}).get('correct', 0) or 0)}/{int(pg316_live_report.get('failure_repair', {}).get('count', 0) or 0)}；仍不能代表三 seed 泛化。")
    if pg317_ready:
        snapshot["capability"]["limits"].append(f"PG-317 多缺失 ASK/complete 反事实的 question anchor 最坏 {float(dict(pg317_metrics.get('holdout_anchor_question_exact') or {}).get('min', 0.0) or 0.0) * 100:.2f}%、原始 missing question 最坏 {float(dict(pg317_metrics.get('holdout_missing_question_recall') or {}).get('min', 0.0) or 0.0) * 100:.2f}%，safe-allow=0、complete unnecessary=0；这是缺观测可识别性的离线证据，不是 payload 能力证明。")
    if pg317_live_ready:
        snapshot["capability"]["limits"].append(f"PG-317 fresh independent live：GET/POST {int(pg317_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg317_live_counts.get('variant_role_count', 0) or 0)} variant、typed {int(pg317_live_counts.get('model_typed_effect_count', 0) or 0)}/{int(pg317_live_counts.get('route_count', 0) or 0)}、repair {int(pg317_live_report.get('failure_repair', {}).get('correct', 0) or 0)}/{int(pg317_live_report.get('failure_repair', {}).get('count', 0) or 0)}；仍只有 SQL row-shape 两路，DOM/XSS/更多实现未验证。")
    if pg318_ready:
        snapshot["capability"]["limits"].append(f"PG-318 族外三 seed fresh replay：typed {int(pg318_counts.get('typed_effect_count', 0) or 0)}/{int(pg318_counts.get('route_count', 0) or 0)}、variant {int(pg318_counts.get('variant_exact_count', 0) or 0)}/{int(pg318_counts.get('variant_role_count', 0) or 0)}、多缺失 ASK 最坏 {float(pg318_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}%、unsafe allow={int(pg318_counts.get('multi_missing_unsafe_allow', 0) or 0)}、repair {int(pg318_counts.get('failure_repair_correct_count', 0) or 0)}/{int(pg318_counts.get('failure_repair_count', 0) or 0)}；首次证明同一抽象流程可过 SQL 与 DOM typed oracle，但实现/路由数量仍不足，不能宣称通用漏洞能力。")
    if pg319_ready:
        snapshot["capability"]["limits"].append(f"PG-319 跨实现训练：route holdout ASK {float(dict(pg319_metrics.get('implementation_route_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%、variant {float(dict(pg319_metrics.get('implementation_route_variant_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%，但 family ASK 最坏仅 {float(dict(pg319_metrics.get('pg318_family_question_min') or {}).get('min', 0.0) or 0.0) * 100:.1f}%；new-only 遗忘 drop={float(pg319_metrics.get('new_only_forgetting_drop_max', 0.0) or 0.0):.3f}，因此不能把离线 OOD 当成主动排错已学会。")
    if pg320_live_ready:
        snapshot["capability"]["limits"].append(f"PG-320 live：variant {int(pg320_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg320_live_counts.get('variant_role_count', 0) or 0)}、typed {int(pg320_live_counts.get('typed_effect_count', 0) or 0)}/{int(pg320_live_counts.get('route_count', 0) or 0)}、negative violation={int(pg320_live_counts.get('negative_lane_violation_count', 0) or 0)}；离线 lattice 提升没有直接等于真实回放提升。")
    if pg321_live_ready:
        snapshot["capability"]["limits"].append(f"PG-321 角色条件 fresh replay：三 seed、18 routes、variant {int(pg321_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg321_live_counts.get('variant_role_count', 0) or 0)}、typed {int(pg321_live_counts.get('typed_effect_count', 0) or 0)}/{int(pg321_live_counts.get('route_count', 0) or 0)}、ASK 最坏 {float(pg321_live_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}%、repair {int(pg321_live_counts.get('failure_repair_correct_count', 0) or 0)}/{int(pg321_live_counts.get('failure_repair_count', 0) or 0)}、negative violation={int(pg321_live_counts.get('negative_lane_violation_count', 0) or 0)}；这是受控 evaluator 证据，不是通用 payload 生成或漏洞声明。")
    if pg323_live_ready:
        snapshot["capability"]["limits"].append(f"PG-323 三 seed 独立 VulnerableApp fresh replay：GET/POST {int(pg323_live_counts.get('route_count', 0) or 0)} routes、variant {int(pg323_live_counts.get('variant_exact_count', 0) or 0)}/{int(pg323_live_counts.get('variant_role_count', 0) or 0)}、typed {int(pg323_live_counts.get('positive_typed_effect_count', 0) or 0)}/{int(pg323_live_counts.get('positive_route_count', 0) or 0)}、ASK 最坏 {float(pg323_live_worst.get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}%、repair {int(pg323_live_counts.get('failure_repair_correct_count', 0) or 0)}/{int(pg323_live_counts.get('failure_repair_count', 0) or 0)}、negative violation={int(pg323_live_counts.get('negative_lane_violation_count', 0) or 0)}；硬行为门全通过，但仍只证明受控实现闭环，不能宣称通用 payload 生成。")
    if str(pg324_contract.get("artifact_status")) == "completed_evaluation_only":
        snapshot["capability"]["limits"].append(f"PG-324 Juice Shop 三 seed、18 fresh routes：typed {int(pg324_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('positive_route_count', 0) or 0)}、variant {int(pg324_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('variant_role_count', 0) or 0)}、ASK {float(pg324_contract.get('worst_seed_metrics', {}).get('multi_missing_question_recall_min', 0.0) or 0.0) * 100:.1f}%、repair {int(pg324_contract.get('counts', {}).get('failure_repair_correct_count', 0) or 0)}/{int(pg324_contract.get('counts', {}).get('failure_repair_count', 0) or 0)}、negative violation={int(pg324_contract.get('counts', {}).get('negative_lane_violation_count', 0) or 0)}；artifact audit 通过但仍只有一个实现，下一步必须 PG-325 族外复放。")
    if str(pg325_contract.get("artifact_status")) == "completed_evaluation_only":
        snapshot["capability"]["limits"].append(f"PG-325 SQL 族外三 seed、9 fresh routes：GET/POST={int(pg325_contract.get('counts', {}).get('get_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('post_count', 0) or 0)}、typed {int(pg325_contract.get('counts', {}).get('positive_typed_effect_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('positive_route_count', 0) or 0)}、variant {int(pg325_contract.get('counts', {}).get('variant_exact_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('variant_role_count', 0) or 0)}、ASK rows={int(pg325_contract.get('counts', {}).get('multi_missing_question_rows', 0) or 0)}、repair {int(pg325_contract.get('counts', {}).get('failure_repair_correct_count', 0) or 0)}/{int(pg325_contract.get('counts', {}).get('failure_repair_count', 0) or 0)}、role-bound belief duplicate={int(pg325_contract.get('counts', {}).get('belief_duplicate_evidence_count', 0) or 0)}；read-only audit={str(pg325_contract.get('audit_status', 'not_embedded'))}，但仍是同一 Pikachu 实现的 evaluator-only 证据，不能宣称通用 payload 能力。")
    snapshot["capability"]["next"] = "PG-331A：先补 PG-327C strict source-row schema，再用 ontology tokenizer 补齐整网页七轴 token、not_observed、字段消融与 source/implementation holdout；信息保真门通过前不再盲目扩大 A800 容量。"
    snapshot["research_goal"]["next_experiment"] = snapshot["capability"]["next"]
    snapshot["learning_requirements"]["promotion_gate"]["next_experiment"] = snapshot["capability"]["next"]
    snapshot["tasks"]["collector"].append({
        "id": "pg331a-whole-web-token-collection",
        "role": "collector",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "blocked_on_information_audit",
        "label": "PG-331A 整网页七轴 token 采集与词表审计",
        "route": "document / navigation / request / response / JavaScript / failure / belief-replay",
        "seed": 33101,
        "method": "GET/POST + browser observation",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            *pg331_missing_axes,
            "context-target alignment gate failed",
            "source/implementation isolation is not yet clean",
        ],
        "evidence_hash": str(pg331_contract.get("audit_evidence_hash", "")),
        "instruction": "使用 app/pg331_source_row.py 的 strict source-row collector 和 research/pg331_source_row_schema_v1.json；逐条保存整网页结构 token。每轴必须有 observed 或 not_observed，保留 GET/POST、参数角色、编码链、响应/302、JS AST/source/sink、失败和 belief/replay。source_meta/reset/evaluator 只能作为 sidecar，不得把原始 payload、响应正文、路由字面量或最终答案写进 context；缺字段就标 incomplete/ASK。",
        "raw_material_available": False,
        "training_eligible": False,
    })
    snapshot["tasks"]["trainer"].append({
        "id": "pg331b-model-capacity-audit",
        "role": "trainer",
        "owner": "AI → 人工",
        "human_required": True,
        "status": "blocked_on_capacity_contract",
        "label": "PG-331B 词表—上下文窗口—模型容量验收",
        "route": "PG-331 ontology vocabulary + decoder capacity variants",
        "seed": 33102,
        "method": "offline read-only audit",
        "typed_effect": False,
        "confirmed_positive": False,
        "reasons": [
            "legacy PG-322 max_length=72 truncates the representative whole-web sequence",
            f"required context window={int(pg331_capacity.get('required_context_window', 0) or 0)}",
            "information audit remains blocked",
        ],
        "evidence_hash": str(pg331_capacity.get("audit_sha256", ""))[:16],
        "instruction": "先核对 input/target vocabulary、代表页面 token 长度、chunk 边界、max_length、embedding/lm-head 参数和显存估算；容量不够时扩窗口/分块，不准删低频词或改变 ontology。",
        "raw_material_available": False,
        "training_eligible": False,
    })
    snapshot["tasks"]["collector"].append({"id": "pg179-parameterized-surface-completion", "role": "collector", "owner": "AI → 人工", "human_required": True, "status": "needs_parameterized_replay" if surface_catalog["counts"]["parameterized_response_observed"] < surface_catalog["counts"]["with_parameter_context"] else "ready_for_review", "label": "PG-179 参数化 GET/POST 表面补采", "route": f"{surface_catalog['counts']['routes']} route catalog", "seed": 17901, "method": "GET/POST", "typed_effect": False, "confirmed_positive": False, "reasons": [f"{surface_catalog['counts']['missing_parameter_context']} routes lack parameter context", f"{surface_catalog['counts']['with_parameter_context'] - surface_catalog['counts']['parameterized_response_observed']} parameterized responses not replayed"], "evidence_hash": str(crawl_manifest.get("manifest_id") or "")[:16], "instruction": "按 route catalog 逐条补齐真实 GET query / POST form、302 status chain 和 parameterized response；缺字段或只看 baseline 的记录不得训练。", "raw_material_available": False})
    snapshot["research_goal"]["question_composition_loop"] = dict(research_goal.get("question_composition_loop") or {})
    # Keep the paired replay report and its audit as separate provenance rows;
    # the report row must carry the replay report hash, not the audit hash.
    for source in snapshot["source_reports"]:
        if source.get("name") == pg327b_name:
            source["sha256"] = str(pg327b_report.get("report_sha256", ""))
    snapshot["source_reports"].append({
        "name": pg370_name,
        "updated_at": _report_time(pg370_name),
        "sha256": str(pg370_report.get("report_sha256", "")),
    })
    # Keep the unfiltered work queue in sync with every role-specific task
    # appended after the initial snapshot was created.
    snapshot["tasks"]["all"] = [
        *snapshot["tasks"]["collector"],
        *snapshot["tasks"]["reviewer"],
        *snapshot["tasks"]["trainer"],
    ]
    return snapshot


def _wire_request(method: str, path: str, values: dict[str, str]) -> dict[str, Any]:
    """Build a human-readable request projection without a live target.

    This is intentionally a presentation helper.  It never accepts a URL from
    the caller and it only emits loopback-origin placeholders, so the UI cannot
    accidentally become an arbitrary-target request builder.
    """

    encoded = urlencode(list(values.items()), quote_via=quote)
    if method.upper() == "GET":
        wire = f"GET <LOOPBACK_ORIGIN>{path}?{encoded}"
    else:
        wire = (
            f"POST <LOOPBACK_ORIGIN>{path}\n"
            "Content-Type: application/x-www-form-urlencoded\n\n"
            f"{encoded}"
        )
    return {
        "logical_values": dict(values),
        "encoded_values": {key: quote(str(value), safe="") for key, value in values.items()},
        "wire": wire,
    }


def _payload_entry(
    *,
    entry_id: str,
    family: str,
    route: str,
    method: str,
    fields: list[str],
    status: str,
    oracle: str,
    effect: str,
    ai: dict[str, Any],
    reference: dict[str, Any],
    negative: dict[str, Any],
    notes: list[str],
    oracle_evidence: dict[str, Any] | None = None,
    source: str = "pg255/pg250 local replay projection",
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "family": family,
        "route": route,
        "method": method,
        "fields": fields,
        "validation_status": status,
        "expected_oracle": oracle,
        "effect_claim": effect,
        "source": source,
        "ai": ai,
        "reference": reference,
        "negative": negative,
        "oracle_evidence": oracle_evidence or {"available": False, "status": "not_attached"},
        "notes": notes,
        "review_only": True,
        "persisted": False,
        "training_eligible": False,
    }


def _safe_evidence_projection(value: Any) -> dict[str, Any]:
    """Keep only bounded response facts useful for a human replay review.

    The runner never persists bodies or executable payloads.  This allow-list
    is intentionally narrower than the report projection: hashes, status and
    typed effect fields are enough to check the oracle without turning the API
    into a response dump.
    """

    if not isinstance(value, dict):
        return {}
    nested = value.get("response_projection") or value.get("projection") or value
    if not isinstance(nested, dict):
        return {}
    allowed = {
        "status", "status_code", "status_class", "body_length", "body_length_bucket", "content_type", "content_type_class",
        "shape", "header_names", "status_chain", "redirect_chain",
        "backend_state", "transport_error", "status_changed", "state_changed",
        "row_marker_count", "row_count_capped", "result_shape", "sql_error_shape",
        "marker_observed", "dom_effect", "script_execution", "oracle_available",
        "oracle_mode", "external_network", "external_request_blocked",
        "external_blocked_count", "candidate_effect", "reference_effect",
        "candidate_class", "reference_class", "baseline_row_count_capped",
        "candidate_row_count_capped", "reference_row_count_capped", "negative_row_count_capped",
        # PG-266 keeps only a bounded marker-context excerpt in the human
        # review lane.  It is never copied into the training dataset.
        "body_length", "body_sha256", "marker_reflected", "echo_excerpt",
        "location", "executed", "observed_marker", "dom_excerpt", "reason",
        "typed_effect", "wire_sha256",
    }
    result: dict[str, Any] = {}
    for key in allowed:
        if key in nested:
            value = nested[key]
            # Shape/header lists are bounded metadata, not body material.
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            elif isinstance(value, list):
                result[key] = [item for item in value[:32] if isinstance(item, (str, int, float, bool))]
            elif isinstance(value, dict):
                result[key] = {str(k): v for k, v in list(value.items())[:16] if isinstance(v, (str, int, float, bool)) or v is None}
            if key in {"echo_excerpt", "dom_excerpt"} and isinstance(result.get(key), str):
                result[key] = str(result[key])[:480]
    return result


def _oracle_evidence(
    *,
    report_name: str,
    row: dict[str, Any] | None,
    typed: dict[str, Any] | None = None,
    candidate: Any = None,
    reference: Any = None,
    negative: Any = None,
) -> dict[str, Any]:
    """Create the BP-like evidence card without retaining raw bodies."""

    typed = dict(typed or (row or {}).get("typed_oracle") or (row or {}).get("oracle") or {})
    evidence = dict(typed.get("evidence") or {})
    confirmed = bool(typed.get("confirmed_positive") or typed.get("typed_effect_confirmed") or typed.get("boolean_effect_confirmed") or typed.get("widebyte_effect_confirmed"))
    reasons = [str(item) for item in list(typed.get("reasons") or [])]
    pattern_id = str(evidence.get("pattern_id") or evidence.get("contract") or evidence.get("oracle_id") or typed.get("oracle_id") or typed.get("schema_version") or "typed_projection")
    match_count = evidence.get("match_count")
    if match_count is None:
        for key in ("row_count_capped", "candidate_row_count_capped", "true_row_marker_count"):
            if key in evidence:
                match_count = evidence[key]
                break
    candidate_projection = _safe_evidence_projection(candidate)
    candidate_true_projection = _safe_evidence_projection((candidate or {}).get("true")) if isinstance(candidate, dict) else {}
    candidate_false_projection = _safe_evidence_projection((candidate or {}).get("false")) if isinstance(candidate, dict) else {}
    fact_keys = {
        "candidate_sql_error_shape", "negative_sql_error_shape", "candidate_negative_differential",
        "candidate_boolean_differential", "reference_boolean_differential", "true_candidate_shape",
        "false_candidate_shape", "true_reference_shape", "false_reference_shape", "negative_shape",
        "candidate_effect", "reference_effect", "candidate_class", "reference_class",
        "baseline_row_count_capped", "candidate_row_count_capped", "reference_row_count_capped",
        "negative_row_count_capped", "candidate_reference_agreement", "fresh_reset",
    }
    oracle_facts = {key: evidence[key] for key in fact_keys if key in evidence and isinstance(evidence[key], (str, int, float, bool))}
    return {
        "available": bool(row or typed),
        "status": "confirmed_positive" if confirmed else "oracle_gap_or_negative",
        "source_report": report_name,
        "seed": int((row or {}).get("seed", 0) or 0),
        "pattern_id": pattern_id,
        "matched": confirmed,
        "match_count": match_count,
        "span_buckets": list(evidence.get("span_buckets") or []),
        "reasons": reasons,
        "candidate_reference_agreement": evidence.get("candidate_reference_agreement", (row or {}).get("candidate_reference_agreement")),
        "negative_clean": evidence.get("candidate_negative_differential", (row or {}).get("negative_clean")),
        "evidence_sha256": str(typed.get("evidence_hash") or evidence.get("evidence_hash") or ""),
        "oracle_facts": oracle_facts,
        "candidate_projection": candidate_projection,
        "candidate_true_projection": candidate_true_projection,
        "candidate_false_projection": candidate_false_projection,
        "reference_projection": _safe_evidence_projection(reference),
        "negative_projection": _safe_evidence_projection(negative),
        "raw_response_body_stored": False,
        "raw_payload_stored": False,
    }


def _review_evidence_index() -> dict[tuple[str, str], dict[str, Any]]:
    """Index the latest bounded local reports for the human payload panel."""

    index: dict[tuple[str, str], dict[str, Any]] = {}

    def add(report_name: str, rows: list[dict[str, Any]], *, family: str, route_key: str | None = None) -> None:
        for row in rows:
            route = str(row.get("path") or row.get("route") or "")
            method = str(row.get("method") or (row.get("request_anatomy") or {}).get("method") or "GET").upper()
            key = route_key or route
            typed = dict(row.get("typed_oracle") or row.get("oracle") or {})
            candidate = dict(row.get("candidate") or row.get("ai") or {}).get("response") or row.get("candidate") or (row.get("ai") or {}).get("response")
            reference = dict(row.get("reference") or {}).get("response") or row.get("reference")
            negative = row.get("negative")
            projection = _oracle_evidence(report_name=report_name, row=row, typed=typed, candidate=candidate, reference=reference, negative=negative)
            index[(family, key)] = projection
            index[(family, f"{method} {route}")] = projection

    add("pg255_pikachu_fixed_sql_pg254_replay_report_v1.json", list((_read_json("pg255_pikachu_fixed_sql_pg254_replay_report_v1.json", {}) or {}).get("episodes") or []), family="sql")
    add("pg221_pikachu_boolean_blind_oracle_report_v1.json", list((_read_json("pg221_pikachu_boolean_blind_oracle_report_v1.json", {}) or {}).get("results") or []), family="boolean")
    add("pg256_pikachu_widebyte_oracle_report_v1.json", list((_read_json("pg256_pikachu_widebyte_oracle_report_v1.json", {}) or {}).get("episodes") or []), family="widebyte")
    add("pg242_pikachu_xss_dom_acceptance_report_v1.json", list((_read_json("pg242_pikachu_xss_dom_acceptance_report_v1.json", {}) or {}).get("results") or []), family="xss")
    add("pg224_pikachu_parameter_surface_collection_report_v1.json", list((_read_json("pg224_pikachu_parameter_surface_collection_report_v1.json", {}) or {}).get("results") or []), family="surface")
    return index


def _pg266_payload_entries() -> list[dict[str, Any]]:
    """Project the latest real local PG-266 replay into the review schema.

    This is the only UI lane allowed to show executable lab values.  The
    catalog is pinned to loopback/fresh-container metadata and the projection
    carries only bounded response excerpts; the abstract training dataset
    remains payload-free.
    """

    catalog = _read_json("pg266_pikachu_payload_grounding_catalog_v1.json", {})
    if str(catalog.get("status", "")) != "completed_human_review_catalog":
        return []

    def channel(row: dict[str, Any], *, label: str) -> dict[str, Any]:
        payload = dict(row.get("payload") or {})
        wire = dict(row.get("wire") or {})
        request = {
            "logical_values": payload,
            "encoded_values": {str(key): quote(str(value), safe="") for key, value in payload.items()},
            "wire": str(wire.get("request_line", "")) + ("\nContent-Type: application/x-www-form-urlencoded\n\n" + str(wire.get("body")) if wire.get("body") is not None else ""),
            "wire_sha256": str(wire.get("wire_sha256", "")),
        }
        return {"status": label, "request": request}

    result: list[dict[str, Any]] = []
    for row in list(catalog.get("entries") or []):
        if not isinstance(row, dict):
            continue
        route = dict(row.get("route") or {})
        oracle = dict(row.get("oracle") or {})
        evidence = dict(row.get("evidence") or {})
        ai = dict(row.get("ai") or {})
        reference = dict(row.get("reference") or {})
        negative = dict(row.get("negative") or {})
        matched = bool(oracle.get("confirmed_positive"))
        result.append(_payload_entry(
            entry_id=f"pg266-review-{row.get('record_id', '')}",
            family=str(route.get("family", "surface")),
            route=str(route.get("path", "")),
            method=str(route.get("method", "GET")),
            fields=[str(item) for item in list(route.get("fields") or [])],
            status="validated_local_effect" if matched else "oracle_gap",
            oracle=str(route.get("oracle", "typed_local_effect")),
            effect=str(oracle.get("reason", "PG-266 local replay did not confirm an effect.")),
            ai=channel(ai, label=f"AI selected / {ai.get('variant', 'candidate')}"),
            reference=channel(reference, label="independent reference"),
            negative=channel(negative, label="matched negative control"),
            notes=[
                "PG-266 fresh-container replay；wire 仅供授权本地人工复核。",
                "回显只保留 marker 附近的有限 excerpt，完整响应不会进入训练集。",
                "confirmed local effect 不是公网漏洞声明；长期记忆与自动晋级关闭。",
            ],
            oracle_evidence={
                "available": True,
                "status": "confirmed_positive" if matched else "oracle_gap_or_negative",
                "source_report": "pg266_pikachu_payload_grounding_replay_report_v1.json",
                "seed": 26601,
                "pattern_id": str(route.get("oracle", "typed_local_effect")),
                "matched": matched,
                "match_count": None,
                "span_buckets": [],
                "reasons": [str(oracle.get("reason", ""))],
                "candidate_reference_agreement": bool(evidence.get("browser_ai", {}).get("executed") == evidence.get("browser_reference", {}).get("executed")) if route.get("family") == "xss" else matched,
                "negative_clean": not bool(evidence.get("browser_negative", {}).get("executed")),
                "evidence_sha256": str(oracle.get("evidence_hash", "")),
                "oracle_facts": {"typed_effect": matched, "vulnerability_claim_allowed": False, "raw_payload_stored": False, "raw_response_body_stored": False},
                "candidate_projection": _safe_evidence_projection(ai.get("response") or {}),
                "candidate_true_projection": _safe_evidence_projection(ai.get("browser_oracle") or {}),
                "candidate_false_projection": {},
                "reference_projection": _safe_evidence_projection(reference.get("response") or {}),
                "negative_projection": _safe_evidence_projection(negative.get("response") or {}) | _safe_evidence_projection(negative.get("browser_oracle") or {}),
                "raw_response_body_stored": False,
                "raw_payload_stored": False,
            },
            source="PG-266 fresh local Pikachu payload-grounding catalog",
        ))
    return result


def _pg268_payload_entries() -> list[dict[str, Any]]:
    """Project PG-268B exact local candidate/reference/negative wires.

    This is a human-review lane.  It may show bounded request values and
    response projections for an authorized loopback replay, while the
    abstract PG-268 dataset remains payload-free.
    """

    catalog = _read_json("pg268_pikachu_parameterized_replay_catalog_v1.json", {})
    if str(catalog.get("status", "")) != "completed_human_review_catalog":
        return []

    def channel(row: dict[str, Any], *, label: str) -> dict[str, Any]:
        payload = dict(row.get("payload") or {})
        wire = dict(row.get("wire") or {})
        if not str(wire.get("request_line", "")):
            return {"status": label, "request": None}
        body = wire.get("body")
        request = {
            "logical_values": payload,
            "encoded_values": {str(key): quote(str(value), safe="") for key, value in payload.items()},
            "wire": str(wire.get("request_line", "")) + ("\nContent-Type: application/x-www-form-urlencoded\n\n" + str(body) if body is not None else ""),
            "wire_sha256": str(wire.get("wire_sha256", "")),
            "response": _safe_evidence_projection(wire),
        }
        return {"status": label, "request": request}

    result: list[dict[str, Any]] = []
    for row in list(catalog.get("entries") or []):
        if not isinstance(row, dict):
            continue
        route = dict(row.get("route") or {})
        oracle = dict(row.get("oracle") or {})
        ai = dict(row.get("ai") or {})
        reference = dict(row.get("reference") or {})
        negative = dict(row.get("negative") or {})
        matched = bool(oracle.get("confirmed_positive"))
        fields = list(dict.fromkeys([str(item) for item in list(route.get("query_params") or []) + list(route.get("form_params") or [])]))
        outcome = str(oracle.get("outcome_class", ""))
        result.append(_payload_entry(
            entry_id=f"pg268-review-{row.get('record_id', '')}",
            family=str(route.get("family", "surface")),
            route=str(route.get("path", "")),
            method=str(route.get("method", "GET")),
            fields=fields,
            status="validated_local_effect" if matched else ("incomplete" if outcome.startswith("unsupported_") else "oracle_gap"),
            oracle=str(oracle.get("oracle_type", "typed_local_effect")),
            effect=str(oracle.get("reason", "PG-268 local replay did not confirm an effect.")),
            ai=channel(ai, label=f"AI selected / {ai.get('selection_reason', 'candidate')}"),
            reference=channel(reference, label="independent reference"),
            negative=channel(negative, label="matched negative control"),
            notes=[
                "PG-268B 每条路由使用全新 Pikachu 容器；只允许 loopback。",
                "这里展示 wire、状态/长度/哈希和有限 echo；训练集不包含原始 payload。",
                "typed local effect 不是公网漏洞声明；oracle gap 与 multipart incomplete 必须 abstain。",
            ],
            oracle_evidence={
                "available": bool(oracle),
                "status": "confirmed_positive" if matched else "oracle_gap_or_negative",
                "source_report": "pg268_pikachu_parameterized_replay_report_v1.json",
                "seed": int(route.get("seed", 0) or 0),
                "pattern_id": str(oracle.get("oracle_type", "typed_local_effect")),
                "matched": matched,
                "reasons": [str(oracle.get("reason", ""))],
                "negative_clean": bool(oracle.get("negative_clean")),
                "evidence_sha256": str(oracle.get("evidence_hash", "")),
                "oracle_facts": {"typed_effect": matched, "outcome_class": outcome, "fresh_complete": bool(oracle.get("fresh_complete")), "vulnerability_claim_allowed": False},
                "candidate_projection": _safe_evidence_projection(ai.get("wire") or {}),
                "reference_projection": _safe_evidence_projection(reference.get("wire") or {}),
                "negative_projection": _safe_evidence_projection(negative.get("wire") or {}),
                "raw_response_body_stored": False,
                "raw_payload_stored": False,
            },
            source="PG-268B fresh local Pikachu parameterized replay catalog",
        ))
    return result


def _review_payload_entries() -> list[dict[str, Any]]:
    """Return allow-listed, non-destructive local Pikachu request examples.

    These are deliberately separate from the report/catalog path.  Reports keep
    hashes and projections; this bounded endpoint exposes the exact request
    shape only so a human can inspect what the AI/reference lane would send.
    No timing, write, exfiltration, credential, or external-origin payload is
    included.
    """

    canary = "sift-review-canary"
    inert = f'<span data-sift-marker="{canary}">{canary}</span>'
    pg256_report = _read_json("pg256_pikachu_widebyte_oracle_report_v1.json", {})
    pg256_counts = dict(pg256_report.get("counts") or {})
    pg256_episode_count = int(pg256_counts.get("episode_count", 0) or 0)
    pg256_confirmed_count = int(pg256_counts.get("confirmed_positive_count", 0) or 0)
    pg256_has_ai_widebyte = "widebyte_escape_boundary" in {str(item) for item in list(pg256_counts.get("ai_candidate_classes") or [])}
    evidence_index = _review_evidence_index()
    entries: list[dict[str, Any]] = []

    def catalog_value(field: str, *, negative: bool = False) -> str:
        key = field.lower()
        if key == "submit":
            return "submit"
        if key in {"id", "uid", "user_id", "userid", "number"}:
            return "0" if negative else "1"
        if key in {"url", "redirect", "next", "path", "target"}:
            return "/" if negative else f"/?{canary}"
        if key in {"xml", "data"}:
            return "<root/>" if negative else "<sift-probe/>"
        if "file" in key or "upload" in key:
            return "[multipart-file-omitted]"
        return "baseline" if negative else canary

    def add_catalog_wire_entries() -> None:
        """Expose discovered request shapes without pretending they are findings.

        The browser crawl is a source of route/field metadata, not a typed
        evaluator.  These entries therefore remain oracle_gap and are never
        training-eligible; they exist so a collector can see exactly which
        GET/POST shape still needs a fresh replay.
        """
        manifest = _read_json("pg179_pikachu_browser_crawl_manifest_v1.json", {})
        catalog = _surface_catalog_projection(manifest)
        existing = {(str(entry.get("method", "GET")), str(entry.get("route", ""))) for entry in entries}
        for route in catalog["routes"]:
            path = str(route.get("path") or "")
            query_fields = list(route.get("query_params") or []) + list(route.get("form_params") or [])
            post_fields = list(route.get("post_form_params") or [])
            methods = {str(method).upper() for method in list(route.get("methods") or [])}
            candidates: list[tuple[str, list[str]]] = []
            if "GET" in methods and query_fields:
                candidates.append(("GET", query_fields))
            if "POST" in methods and post_fields:
                candidates.append(("POST", post_fields))
            for method, fields in candidates:
                if (method, path) in existing:
                    continue
                if any("file" in field.lower() or "upload" in field.lower() for field in fields):
                    # A file upload needs a multipart fixture; do not render a
                    # fake urlencoded wire that a reviewer could mistake for a
                    # valid replay.
                    continue
                ai_values = {field: catalog_value(field) for field in fields}
                ref_values = {field: catalog_value(field, negative=True) for field in fields}
                negative_values = {field: "" if field.lower() != "submit" else "submit" for field in fields}
                family = "sql" if "/sqli/" in path else "xss" if "/xss/" in path else "xxe" if "/xxe/" in path else "surface"
                slug = path.strip("/").replace("/", "-").replace(".", "-") or "root"
                entries.append(_payload_entry(
                    entry_id=f"pg-catalog-{method.lower()}-{slug}",
                    family=family,
                    route=path,
                    method=method,
                    fields=[str(field) for field in fields],
                    status="oracle_gap",
                    oracle="parameterized_response_pending",
                    effect="只显示爬虫发现的请求形状；尚未进行 fresh 参数化回放，不能称为漏洞。",
                    ai={"status": "model_surface_canary_pending_oracle", "request": _wire_request(method, path, ai_values)},
                    reference={"status": "baseline_reference_pending", "request": _wire_request(method, path, ref_values)},
                    negative={"status": "matched_shape_control_pending_replay", "request": _wire_request(method, path, negative_values)},
                    notes=["来源：PG-179 浏览器 manifest。", "这是采集任务的可读 wire，不是成功 payload；完成 typed oracle、negative、fresh reset 和证据哈希前不得训练。"],
                    source="PG-179 browser crawl manifest (parameterized replay pending)",
                ))
                existing.add((method, path))

    def sql_get(
        route: str,
        field: str = "name",
        *,
        status: str = "validated_local_effect",
        oracle: str = "sql_error_shape",
        effect: str = "本地只读响应形状发生类型化变化；不等于漏洞确认。",
        notes: list[str] | None = None,
    ) -> None:
        ai_values = {field: f"{canary}'", "submit": "submit"}
        ref_values = {field: "kobe'", "submit": "submit"}
        neg_values = {field: canary, "submit": "submit"}
        entries.append(
            _payload_entry(
                entry_id=f"pg-review-{route.rsplit('/', 1)[-1].replace('.php', '')}",
                family="sql",
                route=route,
                method="GET",
                fields=[field, "submit"],
                status=status,
                oracle=oracle,
                effect=effect,
                ai={"status": "model_bound_syntax_probe", "request": _wire_request("GET", route, ai_values)},
                reference={"status": "audited_reference_probe", "request": _wire_request("GET", route, ref_values)},
                negative={"status": "matched_negative_control", "request": _wire_request("GET", route, neg_values)},
                notes=notes or ["只读 syntax-boundary probe；不会导出数据、写入数据库或执行时间延迟。"],
            )
        )

    def sql_post(
        route: str,
        field: str,
        ai_value: str,
        ref_value: str,
        negative_value: str,
        *,
        status: str,
        oracle: str,
        effect: str,
        notes: list[str],
        ai_status: str = "model_bound_syntax_probe",
        ref_status: str = "audited_reference_probe",
    ) -> None:
        entries.append(
            _payload_entry(
                entry_id=f"pg-review-{route.rsplit('/', 1)[-1].replace('.php', '')}",
                family="sql",
                route=route,
                method="POST",
                fields=[field, "submit"],
                status=status,
                oracle=oracle,
                effect=effect,
                ai={"status": ai_status, "request": _wire_request("POST", route, {field: ai_value, "submit": "submit"})},
                reference={"status": ref_status, "request": _wire_request("POST", route, {field: ref_value, "submit": "submit"})},
                negative={"status": "matched_negative_control", "request": _wire_request("POST", route, {field: negative_value, "submit": "submit"})},
                notes=notes,
            )
        )

    # Boolean blind: this is the one SQL branch whose local effect oracle is
    # a true/false row differential rather than an SQL error string.
    bool_true = {"name": "kobe' AND '1'='1", "submit": "submit"}
    bool_false = {"name": "kobe' AND '1'='2", "submit": "submit"}
    bool_negative = {"name": "kobe", "submit": "submit"}
    entries.append(
        _payload_entry(
            entry_id="pg-review-sqli-blind-b",
            family="sql",
            route="/vul/sqli/sqli_blind_b.php",
            method="GET",
            fields=["name", "submit"],
            status="validated_local_effect",
            oracle="boolean_row_differential",
            effect="true/false 分支在 fresh 本地容器出现可复现的行存在性差分。",
            ai={"status": "model_bound_boolean_pair", "true": _wire_request("GET", "/vul/sqli/sqli_blind_b.php", bool_true), "false": _wire_request("GET", "/vul/sqli/sqli_blind_b.php", bool_false)},
            reference={"status": "audited_reference_pair", "true": _wire_request("GET", "/vul/sqli/sqli_blind_b.php", bool_true), "false": _wire_request("GET", "/vul/sqli/sqli_blind_b.php", bool_false)},
            negative={"status": "matched_baseline_control", "request": _wire_request("GET", "/vul/sqli/sqli_blind_b.php", bool_negative)},
            notes=["PG-221 两个 fresh seed 均复放通过；这里只展示布尔对照，不做数据导出。", "真正的确认来自 response projection，不来自 payload 文本本身。"],
        )
    )

    sql_get(
        "/vul/sqli/sqli_blind_t.php",
        status="oracle_gap",
        oracle="timing_oracle_disabled",
        effect="仅显示语法边界探针；时间通道被策略明确禁止，不能声称已确认。",
        notes=["不展示 SLEEP/benchmark 等时间延迟 payload；当前只能 abstain。"],
    )
    sql_post(
        "/vul/sqli/sqli_id.php",
        "id",
        "1'",
        "1'",
        "1",
        status="validated_local_effect",
        oracle="sql_error_shape",
        effect="本地响应出现可类型化 SQL 语法错误形状。",
        notes=["数值上下文 quote probe；不把错误字符串当作数据读取。"],
    )
    sql_get("/vul/sqli/sqli_search.php")
    sql_get("/vul/sqli/sqli_str.php")
    sql_get("/vul/sqli/sqli_x.php")

    # Wide-byte is shown because a human must be able to inspect what is still
    # pending.  The lower-case percent bytes are intentional: they model the
    # raw form body accepted by the local lab, not a stored training sample.
    wide_route = "/vul/sqli/sqli_widebyte.php"
    wide_body = "name=kobe%df%27%20OR%201%3D1%23&submit=submit"
    wide_baseline = "name=kobe&submit=submit"
    wide_reference = {
        "status": "audited_local_reference",
        "request": {
            "logical_values": {"name": "kobe<0xDF>' OR 1=1#", "submit": "submit"},
            "encoded_values": {"name": "kobe%df%27%20OR%201%3D1%23", "submit": "submit"},
            "wire": f"POST <LOOPBACK_ORIGIN>{wide_route}\nContent-Type: application/x-www-form-urlencoded\n\n{wide_body}",
        },
    }
    wide_ai = {
        "status": f"model_replayed_on_{pg256_confirmed_count}_of_{pg256_episode_count}_fresh_seeds" if pg256_has_ai_widebyte else "model_not_yet_selected",
        "request": wide_reference["request"] if pg256_has_ai_widebyte else None,
    }
    wide_status = "validated_local_effect" if pg256_confirmed_count else "candidate_pending_pg256"
    wide_effect = f"PG-256 在 {pg256_confirmed_count}/{pg256_episode_count} 个 fresh seed 由模型选择 widebyte_escape_boundary 后，通过行数差分、reference、negative、source hash 和 fresh reset。" if pg256_confirmed_count else "候选请求已在本地直接回显出更大的只读结果形状，但需 PG-256 evaluator 以 capped row projection + negative + fresh reset 正式确认。"
    entries.append(
        _payload_entry(
            entry_id="pg-review-sqli-widebyte",
            family="sql",
            route=wide_route,
            method="POST",
            fields=["name", "submit"],
            status=wide_status,
            oracle="row_count_differential_and_escape_boundary",
            effect=wide_effect,
            ai=wide_ai,
            reference=wide_reference,
            negative={"status": "matched_baseline_control", "request": {"logical_values": {"name": "kobe", "submit": "submit"}, "encoded_values": {"name": "kobe", "submit": "submit"}, "wire": f"POST <LOOPBACK_ORIGIN>{wide_route}\nContent-Type: application/x-www-form-urlencoded\n\n{wide_baseline}"}},
            notes=["这是本地 Pikachu 的 escape/GBK 边界候选，不是已晋级训练样本。", "AI 选择的是抽象 Rule-IR class，raw wire 由 allow-listed binder 临时绑定；报告只存哈希和投影。"],
            source="PG-256 local replay report + allow-listed binder",
        )
    )

    def xss_entry(route: str, field: str, confirmed: bool) -> None:
        ai_values = {field: inert, **({"submit": "submit"} if field != "text" else {})}
        ref_values = {field: inert, **({"submit": "submit"} if field != "text" else {})}
        neg_values = {field: canary, **({"submit": "submit"} if field != "text" else {})}
        entries.append(
            _payload_entry(
                entry_id=f"pg-review-{route.rsplit('/', 1)[-1].replace('.php', '')}",
                family="xss",
                route=route,
                method="GET",
                fields=[field] + ([] if field == "text" else ["submit"]),
                status="validated_local_effect" if confirmed else "oracle_gap",
                oracle="dom_nojs_dual",
                effect="无脚本 inert DOM 节点的解析/反射效果" if confirmed else "尚无通过的 DOM 双重 oracle",
                ai={"status": "model_bound_inert_dom_probe", "request": _wire_request("GET", route, ai_values)},
                reference={"status": "audited_reference_probe", "request": _wire_request("GET", route, ref_values)},
                negative={"status": "matched_negative_control", "request": _wire_request("GET", route, neg_values)},
                notes=["只使用 inert markup；不包含 script、事件处理器、外带请求或 cookie 读取。", "DOM effect 不是自动执行 XSS 的证明。"],
            )
        )

    for route, confirmed in (("/vul/xss/xss_01.php", True), ("/vul/xss/xss_02.php", False), ("/vul/xss/xss_03.php", False), ("/vul/xss/xss_04.php", True), ("/vul/xss/xss_reflected_get.php", True)):
        xss_entry(route, "message", confirmed)
    xss_entry("/vul/xss/xss_dom_x.php", "text", True)

    entries.append(
        _payload_entry(
            entry_id="pg-review-dir-list",
            family="logic",
            route="/vul/dir/dir_list.php",
            method="GET",
            fields=["title"],
            status="oracle_gap",
            oracle="typed_logic_surface_missing",
            effect="尚未定义可独立验证的逻辑效果 oracle。",
            ai={"status": "model_not_in_active_lane", "request": _wire_request("GET", "/vul/dir/dir_list.php", {"title": canary})},
            reference={"status": "reference_canary_only", "request": _wire_request("GET", "/vul/dir/dir_list.php", {"title": canary})},
            negative={"status": "matched_negative_control", "request": _wire_request("GET", "/vul/dir/dir_list.php", {"title": ""})},
            notes=["页面路由已知，但没有把‘成功’定义成可复核的授权/资源差分；保持 abstain。"],
            source="PG-253 route catalog (no typed replay)",
        )
    )
    entries.append(
        _payload_entry(
            entry_id="pg-review-urlredirect",
            family="url_redirect",
            route="/vul/urlredirect/urlredirect.php",
            method="GET",
            fields=["url"],
            status="oracle_gap",
            oracle="same_origin_redirect",
            effect="只允许 same-origin canary；尚未建立独立重定向 evaluator。",
            ai={"status": "model_not_in_active_lane", "request": _wire_request("GET", "/vul/urlredirect/urlredirect.php", {"url": "/?pg=" + canary})},
            reference={"status": "reference_same_origin_canary", "request": _wire_request("GET", "/vul/urlredirect/urlredirect.php", {"url": "/?pg=" + canary})},
            negative={"status": "matched_negative_control", "request": _wire_request("GET", "/vul/urlredirect/urlredirect.php", {"url": "/"})},
            notes=["不使用外部 origin；没有 redirect oracle 就不标记 confirmed_positive。"],
            source="PG-253 route catalog (no typed replay)",
        )
    )
    # Add the manifest-derived request shapes after the audited examples have
    # been assembled, so a catalog entry cannot duplicate a validated route.
    add_catalog_wire_entries()

    # Attach the latest bounded runtime evidence after the allow-listed wire
    # examples have been assembled.  The boolean and widebyte entries use
    # their own typed reports instead of the generic SQL response-shape lane.
    for entry in entries:
        family = str(entry.get("family", ""))
        route = str(entry.get("route", ""))
        key_family = "xss" if family == "xss" else "boolean" if entry.get("expected_oracle") == "boolean_row_differential" else "widebyte" if entry.get("expected_oracle") == "row_count_differential_and_escape_boundary" else "surface" if family == "surface" else "sql"
        entry_key = f"{str(entry.get('method', 'GET')).upper()} {route}"
        evidence = evidence_index.get((key_family, entry_key), evidence_index.get((key_family, route)))
        # PG-224 has useful baseline/candidate projections even when the
        # route-specific typed oracle report is absent.  Keep its family as
        # `surface` and never upgrade the entry from oracle_gap.
        if evidence is None and key_family != "surface":
            evidence = evidence_index.get(("surface", entry_key), evidence_index.get(("surface", route)))
        entry["oracle_evidence"] = evidence or {"available": False, "status": "not_run", "source_report": "", "pattern_id": "typed_projection", "matched": False, "match_count": None, "span_buckets": [], "reasons": ["没有找到对应的 bounded runtime report"], "candidate_reference_agreement": None, "negative_clean": None, "evidence_sha256": "", "candidate_projection": {}, "reference_projection": {}, "negative_projection": {}, "raw_response_body_stored": False, "raw_payload_stored": False}
    return entries


def build_payload_review() -> dict[str, Any]:
    """Build the human-auditable payload panel projection.

    This endpoint is intentionally read-only and bounded to known local lab
    routes.  It is not a scanner and cannot be pointed at an arbitrary URL.
    """

    pg266_entries = _pg266_payload_entries()
    pg268_entries = _pg268_payload_entries()
    entries = pg268_entries + pg266_entries + _review_payload_entries()
    pg266_catalog = _read_json("pg266_pikachu_payload_grounding_catalog_v1.json", {})
    pg268_catalog = _read_json("pg268_pikachu_parameterized_replay_catalog_v1.json", {})
    return {
        "schema_version": "sift-review-payloads-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_only": True,
        "persisted": False,
        "training_eligible": False,
        "target_scope": {"kind": "loopback_only", "allowed_origins": ["127.0.0.1", "localhost"], "arbitrary_target_input": False},
        "safety": {"non_destructive": True, "read_only": True, "no_timing_channel": True, "no_external_network": True, "no_script_execution": True, "no_credential_access": True, "no_data_exfiltration": True},
        "disclaimer": "普通 surface lane 只展示非执行 probe；PG-266/PG-268B lane 的实际候选/回显仅供授权本地人工复核，payload 文本本身不是漏洞证明。",
        "pg266": {"available": bool(pg266_entries), "status": str(pg266_catalog.get("status", "not_run")), "counts": dict(pg266_catalog.get("counts") or {}), "raw_payloads_human_review_only": True, "training_dataset_excludes_payloads": True},
        "pg268": {"available": bool(pg268_entries), "status": str(pg268_catalog.get("status", "not_run")), "counts": dict(pg268_catalog.get("counts") or {}), "raw_payloads_human_review_only": True, "training_dataset_excludes_payloads": True, "audit_file": "pg268_pikachu_parameterized_replay_audit_v1.json"},
        "entries": entries,
    }


__all__ = ["build_research_ops_snapshot", "build_payload_review"]
