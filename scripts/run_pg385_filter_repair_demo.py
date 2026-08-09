"""PG-385 bounded filter-feedback -> abstract repair -> local canary demo.

The first request is an inert delimiter canary which the local fixture rejects
in its raw form.  Only the fixture's abstract filter projection is passed to
the repair reasoner.  The reasoner chooses a one-variable encoding change;
the evaluator then binds that Rule-IR to a reviewed local template and sends
the resulting concrete canary.  Concrete values/wires are ephemeral and are
never written to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg350_runtime_payload_binder import bind_runtime_probe  # noqa: E402
from app.pg380_abstract_probe_reasoner import derive_abstract_probe_plan  # noqa: E402
from app.pg385_filter_canary_fixture import FIELD_NAME, ROUTE_PATH, start_filter_canary_server  # noqa: E402


SCHEMA_VERSION = "pg385-filter-repair-demo-v1"
TEMPLATE_ID = "pg385_query_delimiter_canary_v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _catalog(rule_ir: Mapping[str, Any]) -> dict[str, Any]:
    template = "{{MARKER}}:"
    return {
        "templates": [
            {
                "template_id": TEMPLATE_ID,
                "shape": str(rule_ir["payload_shape_ref"]),
                "syntax_category_ref": str(rule_ir["syntax_category_ref"]),
                "template": template,
                "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "local_only": True,
                "non_destructive": True,
                "stateful_allowed": False,
            }
        ]
    }


def _runtime(origin: str) -> dict[str, Any]:
    return {
        "target_origin": origin,
        "route": {"method": "GET", "path": ROUTE_PATH, "field_name": FIELD_NAME},
        "loopback_only": True,
        "external_network": False,
        "source_attested": True,
        "route_attested": True,
        "field_attested": True,
        "fresh_reset": True,
        "candidate_reference_negative": True,
        "replay_consistency": True,
        "authorization_id": "pg385_local_filter_fixture",
        "allowed_template_ids": [TEMPLATE_ID],
        "stateful_evaluator": False,
    }


def _baseline_rule() -> dict[str, str]:
    return {
        "transport_ref": "get_query",
        "field_role_ref": "query_term",
        "encoding_ref": "identity",
        "payload_shape_ref": "query_marker",
        "syntax_category_ref": "delimiter_boundary",
        "probe_variant_ref": "source_attested_candidate",
        "oracle_ref": "response_shape",
        "safe_to_send": "1",
    }


def _authorize_reasoner_rule(rule_ir: Mapping[str, Any]) -> dict[str, str]:
    """Apply the evaluator gate without changing the model-facing slots."""

    if not isinstance(rule_ir, Mapping) or rule_ir.get("wire_binding_requested") is False:
        raise ValueError("reasoner did not request a bounded binding")
    result = {str(key): str(value) for key, value in rule_ir.items() if key not in {"safe_to_send"}}
    if result.get("probe_variant_ref") == "one_variable_repair":
        result["probe_variant_ref"] = "runtime_canary"
    elif result.get("probe_variant_ref") not in {"source_attested_candidate", "fresh_replay", "runtime_canary"}:
        raise ValueError("reasoner variant is not evaluator-allowlisted")
    result["safe_to_send"] = "1"
    return result


def _abstract_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "transport_ref",
        "field_role_ref",
        "encoding_ref",
        "payload_shape_ref",
        "syntax_category_ref",
        "probe_variant_ref",
        "oracle_ref",
        "safe_to_send",
        "next_action",
        "repair_action",
        "question",
        "ask_reason",
    }
    return {str(key): str(value) for key, value in rule.items() if str(key) in allowed}


def _send(bound: Any) -> dict[str, Any]:
    request = bound.ephemeral_request()
    data = request["body"].encode("utf-8") if request.get("body") is not None else None
    http_request = Request(
        str(request["url"]),
        data=data,
        headers={str(key): str(value) for key, value in dict(request.get("headers") or {}).items()},
        method=str(request["method"]),
    )
    try:
        with urlopen(http_request, timeout=3.0) as response:  # noqa: S310 - loopback origin is attested below
            body = response.read(8192)
    except HTTPError as error:
        body = error.read(8192)
    except OSError:
        return {
            "status_class": "5xx",
            "response_shape": "environment_failure",
            "filter_state": "environment_failure",
            "filter_class": "transport_failure",
            "failure_shape": "loopback_unavailable",
            "effect_class": "none",
            "typed_effect_confirmed": False,
            "external_network": False,
            "raw_response_stored": False,
        }
    try:
        projection = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "status_class": "5xx",
            "response_shape": "bounded_projection_parse_error",
            "filter_state": "parser_error",
            "filter_class": "projection_parse_error",
            "failure_shape": "bounded_projection_parse_error",
            "effect_class": "none",
            "typed_effect_confirmed": False,
            "external_network": False,
            "raw_response_stored": False,
        }
    if not isinstance(projection, Mapping):
        raise ValueError("fixture returned a non-object projection")
    # Copy only the evaluator's abstract allowlist; do not retain the body.
    return {
        "status_class": str(projection.get("status_class", "unknown")),
        "response_shape": str(projection.get("response_shape", "unknown")),
        "filter_state": str(projection.get("filter_state", "unknown")),
        "filter_class": str(projection.get("filter_class", "unknown")),
        "failure_shape": str(projection.get("failure_shape", "unknown")),
        "effect_class": str(projection.get("effect_class", "none")),
        "typed_effect_confirmed": bool(projection.get("typed_effect_confirmed", False)),
        "encoding_acceptance": str(projection.get("encoding_acceptance", "unknown")),
        "external_network": False,
        "raw_response_stored": False,
        "response_evidence_sha256": str(projection.get("evidence_sha256", "")),
    }


def _send_and_project(bound: Any) -> dict[str, Any]:
    projection = _send(bound)
    projection["wire_sha256"] = str(bound.persisted_projection()["wire_sha256"])
    projection["raw_wire_stored"] = False
    projection["raw_value_stored"] = False
    projection["evaluator_only"] = True
    return projection


def _scrub(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    for marker in ("http://", "https://", "wire=", "payload=", "response_body=", "raw_value="):
        if marker in text:
            raise ValueError("raw_or_wire_leak")
    for key in ('"url":', '"body":', '"response_body":', '"raw_value":', '"wire":'):
        if key in text:
            raise ValueError("raw_key_leak")


def run_demo(*, show_wire: bool = False) -> tuple[dict[str, Any], list[str]]:
    server, thread = start_filter_canary_server()
    origin = f"http://127.0.0.1:{server.server_port}"
    wires: list[str] = []
    steps: list[dict[str, Any]] = []
    try:
        # Baseline: an evaluator-authorized identity canary is intentionally
        # rejected by the fixture's raw-delimiter filter.
        server.fresh_reset()
        baseline = _baseline_rule()
        baseline_bound = bind_runtime_probe(baseline, _runtime(origin), _catalog(baseline), marker="PG385_BASE_0001")
        if show_wire:
            wires.append(baseline_bound.human_review_wire())
        baseline_projection = _send_and_project(baseline_bound)
        steps.append(
            {
                "step": "baseline_send",
                "rule_ir": _abstract_rule(baseline),
                "wire_generated": True,
                "wire_sha256": baseline_bound.persisted_projection()["wire_sha256"],
                "response": baseline_projection,
            }
        )

        observation = {
            "method": "GET",
            "surface_context": "query",
            "parameter_role": "query_term",
            "filter_feedback": {
                "state": baseline_projection["filter_state"],
                "filter_class": baseline_projection["filter_class"],
                "encoding_observed": "identity",
            },
            "response_shape": baseline_projection["response_shape"],
            "negative_control": True,
        }
        reasoner = derive_abstract_probe_plan(observation)
        if reasoner.get("status") != "abstract_variant_selected":
            raise RuntimeError("abstract reasoner did not select a bounded repair")
        model_rule = reasoner.get("rule_ir")
        if not isinstance(model_rule, Mapping):
            raise RuntimeError("reasoner returned no Rule-IR")
        steps.append(
            {
                "step": "model_reads_sanitized_filter_feedback",
                "observation": {
                    "method": observation["method"],
                    "surface_context": observation["surface_context"],
                    "parameter_role": observation["parameter_role"],
                    "filter_feedback": observation["filter_feedback"],
                    "response_shape": observation["response_shape"],
                },
                "rule_ir": _abstract_rule(model_rule),
                "model_safe_to_send": False,
                "action_changed": str(model_rule.get("encoding_ref")) != str(baseline["encoding_ref"]),
            }
        )

        evaluator_rule = _authorize_reasoner_rule(model_rule)
        candidate_roles = ("candidate", "reference", "negative", "replay")
        role_projections: dict[str, Any] = {}
        for role in candidate_roles:
            server.fresh_reset()
            marker = {
                "candidate": "PG385_CAND_0002",
                "reference": "PG385_REF_0002",
                "negative": "PG385_NEG_0002",
                "replay": "PG385_REPLAY_0002",
            }[role]
            bound = bind_runtime_probe(evaluator_rule, _runtime(origin), _catalog(evaluator_rule), marker=marker)
            if show_wire:
                wires.append(bound.human_review_wire())
            role_projections[role] = _send_and_project(bound)
        steps.append(
            {
                "step": "evaluator_binds_and_replays_one_variable_encoding_repair",
                "rule_ir": _abstract_rule(evaluator_rule),
                "wire_generated": True,
                "roles": role_projections,
                "negative_control_clean": not role_projections["negative"]["typed_effect_confirmed"],
                "replay_consistent": role_projections["replay"]["typed_effect_confirmed"] == role_projections["candidate"]["typed_effect_confirmed"],
            }
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_filter_repair_loopback_only",
            "fixture": {
                "loopback_only": True,
                "external_network": False,
                "non_destructive": True,
                "raw_response_stored": False,
                "stateful_business_write": False,
            },
            "steps": steps,
            "counts": {
                "baseline_filtered": int(baseline_projection["filter_state"] == "filtered"),
                "model_repair_selected": 1,
                "action_changed": int(steps[1]["action_changed"]),
                "candidate_typed": int(role_projections["candidate"]["typed_effect_confirmed"]),
                "reference_typed": int(role_projections["reference"]["typed_effect_confirmed"]),
                "negative_violation": int(role_projections["negative"]["typed_effect_confirmed"]),
                "replay_typed": int(role_projections["replay"]["typed_effect_confirmed"]),
            },
            "model_boundary": {
                "model_outputs_abstract_rule_ir": True,
                "model_raw_value": False,
                "model_wire": False,
                "evaluator_last_hop_canary_binding": True,
                "raw_response_in_context": False,
            },
            "promotion": {
                "training_allowed": False,
                "memory_promotion_allowed": False,
                "payload_catalog_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
            },
        }
        _scrub(report)
        report["report_sha256"] = _sha(report)
        return report, wires
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-385 local filter feedback repair demo")
    parser.add_argument("--show-wire", action="store_true", help="print ephemeral loopback canary wires; never persist them")
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg385_filter_repair_demo_v1.json")
    args = parser.parse_args()
    report, wires = run_demo(show_wire=args.show_wire)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    if args.show_wire:
        print("EPHEMERAL_LOCAL_CANARY_WIRE_PREVIEW (not persisted):")
        for wire in wires:
            print(wire)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
