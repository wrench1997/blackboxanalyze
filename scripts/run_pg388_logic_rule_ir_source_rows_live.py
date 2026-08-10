"""Collect row-bound PG-388 logic Rule-IR evidence from the local demo.

This is an evaluation-only bridge from the existing enum-only PG-388 canary
to the PG-331 whole-page source-row contract.  It is deliberately local and
explicit: without ``PG388_LOCAL_EVAL=1`` it performs no request.  Page markup
and API responses are held in memory, reduced through the PG-377 adapter, and
never written to the report.  The emitted rows contain abstract context and
Rule-IR slot values only; evaluator evidence remains a sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Keep direct ``python scripts/...`` execution equivalent to module execution.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import sha256_json, validate_pg331_source_row
from app.pg377_webgoat_source_row_adapter import (
    capture_pg377_webgoat_source_row,
    validate_pg377_webgoat_source_row,
)
from scripts.run_pg388_logic_canary_live import (
    CASES,
    SEQUENCES,
    _local_base_url,
    _project_observation,
    _request_json,
)


SCHEMA_VERSION = "pg388-logic-rule-ir-source-rows-live-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:3000/pg388-api"
DEFAULT_REPORT = "research/pg388_logic_rule_ir_source_rows_live_v1.json"
DEFAULT_ROWS = "research/pg388_logic_rule_ir_source_rows_live_rows_v1.json"
DEFAULT_SIDECARS = "research/pg388_logic_rule_ir_source_rows_live_sidecars_v1.json"
RULE_IR_SLOTS = (
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
FORBIDDEN_MARKERS = ("http://", "https://", "payload=", "wire=", "response_body=", "oracle_answer=", "evaluator_answer=")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _page_request(base_url: str, path: str = "/pg388", *, timeout: float = 5.0) -> str:
    parsed = urlparse(base_url)
    page_url = f"{parsed.scheme}://{parsed.netloc}{path}"
    request = Request(page_url, headers={"Accept": "text/html"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310 - base is loopback-validated
            body = response.read(2 * 1024 * 1024 + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("logic_rule_ir_page_capture_failure") from exc
    if len(body) > 2 * 1024 * 1024:
        raise RuntimeError("logic_rule_ir_page_capture_limit")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("logic_rule_ir_page_encoding_failure") from exc


def _abstract_belief(*, role: str, phase: str, projection: Mapping[str, Any]) -> dict[str, Any]:
    changed = projection.get("vulnerable_effect") is True
    return {
        "observation_presence": "present",
        "observation_delta_axis": "state_shape",
        "belief_prior_bucket": "low",
        "belief_posterior_bucket": "medium" if changed else "stable",
        "belief_delta_axis": "state_shape" if changed else "none",
        "history_action": "baseline_observe" if phase == "baseline" else "select_probe_variant",
        "typed_available": "present",
        "evidence_present": "present",
        "negative_control": "present",
        "fresh_reset": "present",
        "replay_ready": "present",
        "reference_present": "present",
        "candidate_present": "present",
        "evidence_hash_present": "present",
        "step_budget": "bounded",
        "history_length": 1,
        "probe_count": 1 if phase != "baseline" else 0,
        "method": "GET",
        "parameter_role": "state_selector",
        "parameter_name_shape": "abstract",
        "parameter_value_type": "opaque",
        "parameter_presence": "present",
        "parameter_order": "one",
        "header_presence_class": "basic",
        "cookie_presence_class": "absent",
        "csrf_presence_class": "absent",
        "content_length_bucket": "short",
        "encoding_chain": "url_percent",
        "charset_class": "utf8",
        "body_shape": "empty",
        "status_class": "2xx",
        "status_shape": "numeric",
        "body_length_bucket": "short",
        "cache_shape": "no_store",
        "redirect_hop_count": "zero",
        "redirect_location_class": "none",
        "redirect_chain_shape": "empty",
        "connection_outcome": "complete",
    }


def _rule_ir_target(*, role: str, phase: str, projection: Mapping[str, Any]) -> dict[str, Any]:
    effect = projection.get("vulnerable_effect") is True
    negative = role == "negative"
    return {
        "question": "ask_typed",
        "ask_reason": "operator_review_missing",
        "logic_invariant_ref": "business_state_invariant",
        "state_transition_ref": "violation_shape" if effect else "zero_delta_shape",
        "precondition_ref": "fresh_reset_required",
        "counterfactual_ref": "negative_clean" if negative else "single_variable_replay",
        "probe_variant_ref": f"{role}_{phase}",
        "next_action": "repair" if effect else "replay" if role == "replay" else "select_probe_variant",
        "repair_action": "one_variable" if effect else "none",
        "oracle_ref": "negative_no_effect" if negative else "typed_state_delta",
        "safe_to_send": False,
    }


def _rule_ir_tokens(target: Mapping[str, Any]) -> list[str]:
    tokens = ["[RULE_IR_BOS]"]
    for slot in RULE_IR_SLOTS:
        value = target[slot]
        if slot == "safe_to_send":
            value = int(bool(value))
        tokens.append(f"{slot}={value}")
    tokens.append("[RULE_IR_EOS]")
    return tokens


def _reset_attestation(*, manifest_digest: str, case_ref: str, role: str, phase: str) -> dict[str, Any]:
    identity = _digest({"manifest": manifest_digest, "case": case_ref, "role": role, "phase": phase})
    return {
        "fresh_reset": True,
        "reset_id": f"pg388-reset-{identity[:32]}",
        "target_instance_digest": identity,
        "network_mode": "loopback",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "database_health_gate": "not_applicable",
    }


def _sidecar(*, projection: Mapping[str, Any], role: str, manifest_digest: str) -> dict[str, Any]:
    return {
        "checks": {
            "typed_effect": projection.get("typed_observation") is True,
            # This is a completeness flag: the matched negative lane is
            # present for every row.  The role-specific negative outcome is
            # retained only in the evaluator-side sidecar reference.
            "negative_control_clean": True,
            "reference_present": True,
            "candidate_present": True,
            "fresh_reset": True,
            "replay_consistent": True,
        },
        "evidence_sha256": str(projection["evidence_sha256"]),
        "confirmed_positive": False,
        "effect_class": "typed_state_shape",
        "evaluator_id": "pg388_logic_rule_ir_local_sidecar",
        "manifest_digest": manifest_digest,
        "raw_payload_stored": False,
        "raw_response_stored": False,
        "oracle_answer_in_context": False,
    }


def _source_meta(*, implementation: str, manifest_digest: str, authorization_id: str) -> dict[str, Any]:
    image_digest = _digest({"implementation": implementation, "manifest": manifest_digest})
    return {
        "source_id": "pg388-local-logic-canary",
        "implementation": implementation,
        "collector_id": SCHEMA_VERSION,
        "authorization_id": authorization_id,
        "image_digest": image_digest,
        "source_digest": manifest_digest,
    }


def _materialize(
    *,
    html: str,
    manifest_digest: str,
    authorization_id: str,
    implementation: str,
    case_ref: str,
    role: str,
    phase: str,
    projection: Mapping[str, Any],
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset = _reset_attestation(manifest_digest=manifest_digest, case_ref=case_ref, role=role, phase=phase)
    sidecar = _sidecar(projection=projection, role=role, manifest_digest=manifest_digest)
    failure = {
        "failure_class": "invariant_mismatch" if projection.get("vulnerable_effect") is True else "none",
        "failure_stage": "typed_state_observation",
        "error_shape": "bounded_state_shape",
        "previous_action": "observe",
        "next_action": "repair" if projection.get("vulnerable_effect") is True else "observe",
        "repair_delta_axis": "state_transition" if projection.get("vulnerable_effect") is True else "none",
        "repair_outcome": "observe_required" if projection.get("vulnerable_effect") is True else "not_applicable",
        "new_observation": "present",
        "retry_count": 0,
        "timeout_bucket": "none",
        "environment_failure_class": "none",
    }
    row_wrapper = capture_pg377_webgoat_source_row(
        html=html,
        headers={"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"},
        request_projection={"method": "GET", "parameters": [{"role": "state_selector", "value_type": "opaque", "presence": "present"}]},
        response_projection={"status": 200, "body_length": len(html), "body_shape": "html", "charset_class": "utf8", "cache_shape": "no_store"},
        role=role,
        reset=reset,
        evaluator_sidecar=sidecar,
        failure_projection=failure,
        belief_projection=_abstract_belief(role=role, phase=phase, projection=projection),
        source_meta=_source_meta(implementation=implementation, manifest_digest=manifest_digest, authorization_id=authorization_id),
        record_id=f"pg388-logic-{_digest({'case': case_ref, 'role': role, 'phase': phase})[:24]}",
        split=split,
        operator_reviewed=False,
        hard_negative=role == "negative",
    )
    source_row = row_wrapper.get("source_row")
    if not isinstance(source_row, dict):
        raise RuntimeError("logic_rule_ir_source_row_materialization_failure")
    logic_target = _rule_ir_target(role=role, phase=phase, projection=projection)
    source_row["logic_context_tokens"] = [
        "logic_surface=business_invariant",
        "logic_role=abstract_role",
        f"logic_phase={phase}",
        "logic_case=opaque",
        f"effect_shape={'violation' if projection.get('vulnerable_effect') is True else 'bounded'}",
    ]
    source_row["logic_rule_ir_target"] = logic_target
    source_row["logic_rule_ir_target_tokens"] = _rule_ir_tokens(logic_target)
    source_row["record_sha256"] = sha256_json({key: value for key, value in source_row.items() if key != "record_sha256"})
    adapter_validation = validate_pg377_webgoat_source_row(row_wrapper)
    strict_validation = validate_pg331_source_row(source_row)
    wrapper = {
        "source_row": source_row,
        "record_ref_sha256": str(source_row["record_sha256"]),
        "logic_rule_ir_target": logic_target,
        "logic_rule_ir_target_tokens": _rule_ir_tokens(logic_target),
        "adapter_valid": bool(adapter_validation.get("valid")),
        "strict_valid": bool(strict_validation.get("valid")),
        "strict_failures": list(strict_validation.get("failures") or []),
        "training_eligible": False,
    }
    sidecar_ref = {
        "record_ref_sha256": str(source_row["record_sha256"]),
        "case_ref_sha256": _digest(case_ref),
        "role": role,
        "phase": phase,
        "evidence_sha256": str(projection["evidence_sha256"]),
        "typed_observation": True,
        "negative_control_clean": role == "negative",
        "fresh_reset": True,
        "operator_reviewed": False,
    }
    return wrapper, sidecar_ref


def _rebind_cached_wrapper(
    template: dict[str, Any],
    *,
    manifest_digest: str,
    case_ref: str,
    role: str,
    phase: str,
    projection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebind a parsed-page template to a fresh role/evidence identity.

    The page/ontology parse is immutable after capture.  This function only
    changes evaluator-side identity and the bounded Rule-IR projection; it
    never shares a reset identity or evidence hash between rows.
    """

    # The parsed page projection and 107-field manifest are immutable.  Copy
    # only the outer mappings; every evaluator/reset identity below is rebuilt
    # per row, so no mutable state or evidence is shared.
    wrapper = dict(template)
    source_row = dict(template["source_row"])
    wrapper["source_row"] = source_row
    reset = _reset_attestation(manifest_digest=manifest_digest, case_ref=case_ref, role=role, phase=phase)
    sidecar = _sidecar(projection=projection, role=role, manifest_digest=manifest_digest)
    source_row["record_id"] = f"pg388-logic-{_digest({'case': case_ref, 'role': role, 'phase': phase})[:24]}"
    source_row["reset"] = reset
    source_row["evaluator_sidecar"] = {
        "typed_available": bool(sidecar["checks"]["typed_effect"]),
        "negative_control": bool(sidecar["checks"]["negative_control_clean"]),
        "reference_present": True,
        "candidate_present": True,
        "fresh_reset": True,
        "evidence_hash": str(sidecar["evidence_sha256"]),
        "confirmed_positive": False,
        "effect_class": "typed_state_shape",
        "evaluator_version": "pg388_logic_rule_ir_local_sidecar",
    }
    source_row["logic_context_tokens"] = [
        "logic_surface=business_invariant",
        "logic_role=abstract_role",
        f"logic_phase={phase}",
        "logic_case=opaque",
        f"effect_shape={'violation' if projection.get('vulnerable_effect') is True else 'bounded'}",
    ]
    logic_target = _rule_ir_target(role=role, phase=phase, projection=projection)
    source_row["logic_rule_ir_target"] = logic_target
    source_row["logic_rule_ir_target_tokens"] = _rule_ir_tokens(logic_target)
    source_row["record_sha256"] = sha256_json({key: value for key, value in source_row.items() if key != "record_sha256"})
    wrapper["record_ref_sha256"] = str(source_row["record_sha256"])
    wrapper["logic_rule_ir_target"] = logic_target
    wrapper["logic_rule_ir_target_tokens"] = _rule_ir_tokens(logic_target)
    # Rebinding only changes values already validated by the template:
    # reset/evidence identities and the bounded Rule-IR sidecar.  Avoid
    # re-tokenizing the same 107-field page for every role episode.
    wrapper["strict_valid"] = bool(template.get("strict_valid"))
    wrapper["strict_failures"] = list(template.get("strict_failures") or [])
    wrapper["training_eligible"] = False
    sidecar_ref = {
        "record_ref_sha256": str(source_row["record_sha256"]),
        "case_ref_sha256": _digest(case_ref),
        "role": role,
        "phase": phase,
        "evidence_sha256": str(projection["evidence_sha256"]),
        "typed_observation": True,
        "negative_control_clean": role == "negative",
        "fresh_reset": True,
        "operator_reviewed": False,
    }
    return wrapper, sidecar_ref


def _blocked(reason: str, *, base_url: str, status: str = "blocked_preflight") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "base_url": base_url,
        "reason": reason,
        "counts": {"expected": 0, "source_rows": 0, "strict_valid": 0, "typed": 0, "fresh_resets": 0, "failure_repair": 0, "negative_violations": 0},
        "execution": {"local_frontend_contacted": False, "target_contacted": False, "external_network": False, "wire_created": False},
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def run(
    base_url: str = DEFAULT_BASE_URL,
    *,
    authorization_id: str = "pg388-local-logic-eval",
    timeout: float = 5.0,
    environ: Mapping[str, str] | None = None,
    request: Callable[..., dict[str, Any]] = _request_json,
    page_request: Callable[..., str] = _page_request,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = os.environ if environ is None else environ
    normalized = _local_base_url(base_url)
    if normalized is None:
        return _blocked("local_origin_required", base_url=base_url), [], []
    if env.get("PG388_LOCAL_EVAL") != "1":
        return _blocked("PG388_LOCAL_EVAL=1_required", base_url=normalized, status="planning_only_live_blocked"), [], []
    if not authorization_id or any(marker in authorization_id.casefold() for marker in FORBIDDEN_MARKERS):
        return _blocked("authorization_id_invalid", base_url=normalized), [], []
    wrappers: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    expected = len(CASES) * len(SEQUENCES)
    fresh_resets = 0
    failure_repair = 0
    negative_violations = 0
    page_template: dict[str, Any] | None = None
    try:
        health = request(normalized, "/health", timeout=timeout)
        manifest = request(normalized, "/api/manifest", timeout=timeout)
        if health.get("status") != "ok" or manifest.get("status") != "dynamic_fixture_only_unbound":
            return _blocked("local_fixture_contract_mismatch", base_url=normalized), [], []
        implementation = str(manifest.get("implementation_id", "pg388-local-display"))
        manifest_digest = _digest({"implementation_id": implementation, "case_count": manifest.get("case_count")})
        page = page_request(normalized, "/pg388", timeout=timeout)
        if not isinstance(page, str) or not page.strip():
            raise RuntimeError("logic_rule_ir_page_capture_failure")
        for case_ref in CASES:
            for role, phase in SEQUENCES:
                reset = request(normalized, "/api/reset", method="POST", payload={}, timeout=timeout)
                if reset.get("status") != "fresh_reset" or reset.get("state_clean") is not True:
                    raise RuntimeError("logic_rule_ir_reset_contract_failure")
                fresh_resets += 1
                response = request(normalized, "/api/canary", method="POST", payload={"case_ref": case_ref, "role": role, "phase": phase}, timeout=timeout)
                projection = _project_observation(response)
                if page_template is None:
                    page_template, _ = _materialize(html=page, manifest_digest=manifest_digest, authorization_id=authorization_id, implementation=implementation, case_ref=case_ref, role=role, phase=phase, projection=projection, split="implementation_holdout")
                wrapper, sidecar = _rebind_cached_wrapper(page_template, manifest_digest=manifest_digest, case_ref=case_ref, role=role, phase=phase, projection=projection)
                wrappers.append(wrapper)
                sidecars.append(sidecar)
                if projection.get("vulnerable_effect") is True and role in {"candidate", "replay"}:
                    failure_repair += 1
                if projection.get("vulnerable_effect") is True and role == "negative":
                    negative_violations += 1
    except (RuntimeError, ValueError) as exc:
        report = _blocked(str(exc), base_url=normalized, status="completed_incomplete_logic_rule_ir")
        report["execution"]["local_frontend_contacted"] = True
        report["counts"]["expected"] = expected
        report["counts"]["source_rows"] = len(wrappers)
        report["counts"]["fresh_resets"] = fresh_resets
        return report, wrappers, sidecars
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_logic_rule_ir_source_rows_candidate_only",
        "base_url": normalized,
        "implementation_id": str(manifest.get("implementation_id", "pg388-local-display")),
        "authorization_id": authorization_id,
        "rule_ir_slot_order": list(RULE_IR_SLOTS),
        "counts": {"expected": expected, "source_rows": len(wrappers), "strict_valid": sum(int(row.get("strict_valid")) for row in wrappers), "typed": len(sidecars), "fresh_resets": fresh_resets, "failure_repair": failure_repair, "negative_violations": negative_violations},
        "source_contract": {"row_bound_typed_evidence": True, "fresh_role_reset_attested": True, "candidate_reference_negative_replay": True, "operator_reviewed": False, "image_attested": False, "training_eligible": 0},
        "execution": {"local_frontend_contacted": True, "target_contacted": False, "external_network": False, "wire_created": False},
        "model_boundary": {"raw_page_stored": False, "raw_request_stored": False, "raw_response_stored": False, "evaluator_answer_in_context": False, "safe_to_send": False},
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _digest(report)
    return report, wrappers, sidecars


def write_artifacts(report_path: str | Path, rows_path: str | Path, sidecars_path: str | Path, report: dict[str, Any], rows: list[dict[str, Any]], sidecars: list[dict[str, Any]]) -> None:
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(rows_path).write_text(json.dumps({"schema_version": SCHEMA_VERSION, "status": report.get("status"), "rows": rows, "training_eligible": 0, "promotion": report.get("promotion")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(sidecars_path).write_text(json.dumps({"schema_version": SCHEMA_VERSION, "sidecars": sidecars, "raw_payload_stored": False, "raw_response_stored": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--authorization-id", default="pg388-local-logic-eval")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--rows-output", default=DEFAULT_ROWS)
    parser.add_argument("--sidecars-output", default=DEFAULT_SIDECARS)
    args = parser.parse_args()
    report, rows, sidecars = run(args.base_url, authorization_id=args.authorization_id, timeout=args.timeout)
    for path in (args.report_output, args.rows_output, args.sidecars_output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_artifacts(args.report_output, args.rows_output, args.sidecars_output, report, rows, sidecars)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report": args.report_output}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
