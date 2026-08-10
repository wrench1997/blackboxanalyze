"""Build a bounded, in-process PG-388 logic page source-row diagnostic.

This is a local fixture generator, not a crawler or a target launcher.  It
feeds two deliberately different page/transport surfaces through the existing
PG-377 whole-page adapter and writes only abstract PG-331 source rows.  The
ephemeral HTML/JS is discarded before serialization; evaluator/reset facts are
sidecars.  Operator review remains false, so the output is diagnostic-only and
cannot silently become training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import sha256_json, validate_pg331_source_row
from app.pg377_webgoat_source_row_adapter import capture_pg377_webgoat_source_row


DEFAULT_OUTPUT = ROOT / "research" / "pg388_logic_surface_source_rows_v1.json"
SCHEMA_VERSION = "pg388-logic-surface-source-rows-v1"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}
SEEDS = (38801, 38802, 38803)
ROLES = ("candidate", "reference", "negative", "replay")
IMPLEMENTATIONS = ("pg388_logic_surface_c", "pg388_logic_surface_d")
CASES = (
    ("purchase_total", "quantity", "integer", "transaction_price"),
    ("coupon_state", "coupon", "opaque_state", "business_risk"),
    ("reset_subject", "account", "opaque_state", "password_reset"),
    ("factor_order", "factor", "enum", "two_factor"),
    ("resource_scope", "record", "identifier", "authorization"),
    ("execution_order", "step", "enum", "workflow_order"),
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _html(implementation: str, case_ref: str, method: str, role: str) -> str:
    if implementation.endswith("c"):
        script = "const state = readState(); function normalize(){ return state; } function guard(){ return true; }"
        surface = "card"
        chain = "trim_then_casefold"
        event = "submit"
    else:
        script = "const state = parseState(); function bound(){ return state; } function transition(){ return true; }"
        surface = "panel"
        chain = "parse_then_bounds"
        event = "change"
    form_method = method.lower()
    return (
        "<!doctype html><html lang='en'><head><title>logic surface</title>"
        "<meta name='surface' content='abstract'><style>.panel{display:block}</style>"
        f"<script>{script}</script></head><body><nav><a href='#step'>step</a>"
        f"<a href='#{surface}'>state</a></nav><main id='{surface}' class='logic-card' "
        f"data-parameter-role='{case_ref}' data-encoding-chain='{'url_percent' if method == 'GET' else 'form_urlencoded'}' "
        f"data-transport-method='{method}' data-role-shape='{role}'><h1>workflow</h1>"
        f"<form method='{form_method}' action='/local/logic'><input name='{case_ref}' "
        f"data-parameter-role='{case_ref}' type='text'><button type='submit'>Continue</button></form>"
        f"<section id='step' data-event-shape='{event}' data-state-shape='bounded'>pending</section>"
        "</main></body></html>"
    )


def _javascript_overlay(implementation: str, case_ref: str, source_digest: str) -> dict[str, Any]:
    if implementation.endswith("c"):
        context = {
            "source_kind": "inline",
            "parser_kind": "bounded_ast",
            "normalization_chain": "trim_then_casefold",
            "filter_shape": "allowlisted_symbol",
            "guard_shape": "role_gate",
            "control_flow_shape": "branch_then_state",
            "event_shape": "submit",
            "ast_shape": "call_chain",
            "source_to_sink_shape": "state_projection",
            "sink_context": "text_only",
            "external_or_dynamic_loader": False,
            "persistent_state": False,
            "dynamic_code": False,
            "tokens": ["state_read", "normalize", "role_gate", "state_projection", case_ref],
        }
        semantic = ["state_read", "normalize", "role_gate", "state_projection", "submit_event"]
    else:
        context = {
            "source_kind": "inline",
            "parser_kind": "bounded_ast",
            "normalization_chain": "parse_then_bounds",
            "filter_shape": "range_and_enum",
            "guard_shape": "owner_gate",
            "control_flow_shape": "branch_then_transition",
            "event_shape": "change",
            "ast_shape": "conditional_call",
            "source_to_sink_shape": "state_projection",
            "sink_context": "form_text",
            "external_or_dynamic_loader": False,
            "persistent_state": False,
            "dynamic_code": False,
            "tokens": ["state_parse", "bounds_check", "owner_gate", "state_transition", case_ref],
        }
        semantic = ["state_parse", "bounds_check", "owner_gate", "state_transition", "change_event"]
    return {
        "schema_version": "pg388-logic-surface-js-overlay-v1",
        "source_sha256": source_digest,
        "source_text_stored": False,
        "script_count": 1,
        "local_fixture": True,
        "javascript_context": context,
        "js_semantic_tokens": semantic,
    }


def _belief(implementation: str, method: str, role: str) -> dict[str, Any]:
    replay = role == "replay"
    return {
        "observation_presence": "present",
        "observation_delta_axis": "response_shape",
        "belief_prior_bucket": "low" if implementation.endswith("c") else "medium",
        "belief_posterior_bucket": "medium" if implementation.endswith("c") else "high",
        "belief_delta_axis": "response_shape",
        "history_action": "replay_observe" if replay else "baseline_observe",
        "history_length": 2 if replay else 1,
        "history_length_bucket": "short",
        "typed_available": "present",
        "evidence_present": "present",
        "negative_control": "present",
        "fresh_reset": "present",
        "replay_ready": "present",
        "reference_present": "present",
        "candidate_present": "present",
        "step_budget": "bounded",
        "probe_count": 2 if replay else 1,
        "evidence_hash_present": "present",
        "failure_class": "none",
        "failure_stage": "none",
        "error_shape": "empty",
        "parse_error_class": "none",
        "encoding_error_class": "none",
        "redirect_error_class": "none",
        "timeout_bucket": "none",
        "blocked_reason_class": "none",
        "previous_action": "observe",
        "next_action": "assemble_rule_ir",
        "repair_delta_axis": "none",
        "repair_outcome": "not_applicable",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "content_type_class": "none" if method == "GET" else "form_urlencoded",
        "query_count": 1 if method == "GET" else 0,
        "form_count": 0 if method == "GET" else 1,
        "json_field_count": 0,
        "multipart_part_count": 0,
        "parameter_role": "abstract_field",
        "parameter_name_shape": "abstract",
        "parameter_value_type": "enum",
        "parameter_presence": "present",
        "parameter_order": 1,
        "header_presence_class": "basic",
        "cookie_presence_class": "absent",
        "csrf_presence_class": "absent",
        "content_length_bucket": "short",
        "encoding_chain": "url_percent" if method == "GET" else "form_urlencoded",
        "charset_class": "utf8",
        "body_shape": "empty" if method == "GET" else "form",
        "status_class": "success",
        "status_shape": "numeric",
        "body_length_bucket": "medium",
        "cache_shape": "absent",
        "redirect_hop_count": 0,
        "redirect_location_class": "none",
        "redirect_chain_shape": "empty",
        "connection_outcome": "complete",
    }


def _sidecar(implementation: str, seed: int, case_ref: str, role: str) -> dict[str, Any]:
    evidence = _digest(f"pg388-surface-evidence|{implementation}|{seed}|{case_ref}|{role}")
    return {
        "evaluator_id": "pg388_local_logic_surface_evaluator",
        "evaluator_version": "v1",
        "evidence_sha256": evidence,
        "checks": {
            "typed_effect": True,
            "negative_control_clean": True,
            "reference_present": True,
            "candidate_present": True,
            "fresh_reset": True,
            "replay_consistent": True,
        },
        "raw_payload_stored": False,
        "raw_response_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def _row(implementation: str, seed: int, case: tuple[str, str, str, str], role: str, split: str, source_digest: str) -> dict[str, Any]:
    case_ref, parameter_role, value_type, _family = case
    method = "GET" if implementation.endswith("c") else "POST"
    target_digest = _digest(f"pg388-surface-instance|{implementation}|{seed}|{case_ref}|{role}")
    reset = {
        "fresh_reset": True,
        "reset_id": f"reset_{implementation[-1]}_{seed}_{case_ref}_{role}",
        "target_instance_digest": target_digest,
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "volume_mount_count": 0,
        "container_restart_used": False,
    }
    image_digest = _digest(f"pg388-surface-image|{implementation}")
    meta = {
        "source_id": "pg388_logic_surface_local_v1",
        "implementation": implementation,
        "family_id": "logic_business_surface",
        "surface_id": case_ref,
        "collector_id": "pg388_local_memory_adapter_v1",
        "authorization_id": "pg388_local_fixture_authorization",
        "image_digest": image_digest,
        "source_digest": source_digest,
    }
    request = {
        "method": method,
        "parameters": [{"role": parameter_role, "value_type": value_type, "presence": "present"}],
        "csrf_presence_class": "absent",
        "cookie_presence_class": "absent",
        "content_length": 24 if method == "POST" else 0,
    }
    response = {
        "status": 200 if method == "GET" else 303,
        "body_length": 512 + (seed % 3) * 64,
        "body_shape": "html",
        "connection_outcome": "complete",
        "failure_class": "none",
        "failure_stage": "none",
        "error_shape": "empty",
        "charset_class": "utf8",
        "cache_shape": "absent",
    }
    # Keep headers abstract.  A concrete Location/path is evaluator-side and
    # is intentionally represented by redirect shape below, not serialized.
    headers = {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}
    sidecar = _sidecar(implementation, seed, case_ref, role)
    record_id = f"pg388_surface_{implementation[-1]}_{seed}_{case_ref}_{role}"
    projection = capture_pg377_webgoat_source_row(
        html=_html(implementation, case_ref, method, role),
        headers=headers,
        request_projection=request,
        response_projection=response,
        role=role,
        reset=reset,
        evaluator_sidecar=sidecar,
        belief_projection=_belief(implementation, method, role),
        javascript_context_projection=_javascript_overlay(implementation, case_ref, source_digest),
        post_supported=True,
        source_meta=meta,
        record_id=record_id,
        split=split,
        operator_reviewed=False,
        hard_negative=role == "negative",
    )
    source_row = projection.get("source_row")
    if not isinstance(source_row, dict):
        raise RuntimeError("PG-388 surface adapter did not materialize source_row")
    validation = validate_pg331_source_row(source_row)
    wrapper = {
        "source_row": source_row,
        "record_ref_sha256": _digest(record_id),
        "logic_rule_ir_target": dict(source_row.get("target_projection") or {}),
        "logic_rule_ir_target_tokens": list(source_row.get("target_tokens") or []),
        "adapter_valid": True,
        "strict_valid": bool(validation.get("valid")),
        "strict_failures": list(validation.get("failures") or []),
        "training_eligible": False,
    }
    return wrapper


def build(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    wrappers: list[dict[str, Any]] = []
    for implementation in IMPLEMENTATIONS:
        split = "train" if implementation.endswith("c") else "implementation_holdout"
        for seed in SEEDS:
            for case in CASES:
                for role in ROLES:
                    wrappers.append(_row(implementation, seed, case, role, split, source_digest))
    strict_valid = sum(1 for item in wrappers if item["strict_valid"])
    typed = sum(1 for item in wrappers if item["source_row"].get("evaluator_sidecar", {}).get("typed_available") is True)
    fresh = sum(1 for item in wrappers if item["source_row"].get("reset", {}).get("fresh_reset") is True)
    rows = [item["source_row"] for item in wrappers]
    raw_serialized = json.dumps(wrappers, ensure_ascii=False, separators=(",", ":")).casefold()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_inprocess_diagnostic_only",
        "purpose": "bounded local logic page surfaces for PG-331 information audit",
        "source_contract": {
            "in_process_fixture_only": True,
            "docker_started": False,
            "target_contacted": False,
            "external_network": False,
            "row_bound_typed_evidence": True,
            "fresh_role_reset_attested": True,
            "operator_reviewed": False,
            "training_eligible": 0,
        },
        "counts": {
            "source_rows": len(wrappers),
            "strict_valid": strict_valid,
            "typed": typed,
            "fresh_resets": fresh,
            "train": sum(1 for row in rows if row.get("split") == "train"),
            "implementation_holdout": sum(1 for row in rows if row.get("split") == "implementation_holdout"),
            "negative_violations": 0,
            "training_eligible": 0,
        },
        "implementation_count": len(IMPLEMENTATIONS),
        "case_count": len(CASES),
        "seed_count": len(SEEDS),
        "role_count": len(ROLES),
        "raw_marker_hits": sum(raw_serialized.count(marker) for marker in ("http://", "https://", "payload=", "wire=", "response_body=")),
        "rows_emitted": True,
        "promotion": dict(PROMOTION),
    }
    report["report_sha256"] = sha256_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Canonical ordering keeps the artifact byte-stable even though the
    # adapter's allowlisted overlay keys are held in a set.
    output_path.write_text(json.dumps({**report, "rows": wrappers}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The report hash is over the bounded metadata, not the serialized rows;
    # the rows remain local abstract source data and are never promoted.
    return {**report, "rows": wrappers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"status": result["status"], "source_rows": result["counts"]["source_rows"], "strict_valid": result["counts"]["strict_valid"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build"]
