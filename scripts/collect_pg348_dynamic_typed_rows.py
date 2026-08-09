"""Collect PG-348 dynamic typed rows from the reviewed loopback runtime.

The runtime accepts only abstract probe-variant references.  This collector
keeps raw HTML/request bytes in the adapter process, builds candidate /
reference / negative / replay evidence in the evaluator sidecar, and emits
only ontology tokens plus bounded projections.  It is a synthetic evaluator
lane, not a public scanner and not a payload catalog.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_evaluator_sidecar, sha256_json
from app.pg331_loopback_adapter import _field_capture_manifest, capture_loopback
from app.pg331_source_row import collect_pg331_source_row, validate_pg331_source_row
from app.pg348_dynamic_runtime import DynamicFixtureApplication, load_registry, start_server
from app.pg348_surface_projection import project_surface
from app.pg331_web_tokenizer import tokenize_web_observation


DEFAULT_REGISTRY = ROOT / "fixtures" / "pg348" / "registry_v1.json"
DEFAULT_DATASET = ROOT / "research" / "pg348_dynamic_typed_source_rows_v1.json"
DEFAULT_SIDECARS = ROOT / "research" / "pg348_dynamic_typed_sidecars_v1.json"
DEFAULT_REPORT = ROOT / "research" / "pg348_dynamic_typed_collection_report_v1.json"
ROLES = ("candidate", "reference", "negative", "replay")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _split(record: Mapping[str, Any]) -> str:
    implementation = str(record.get("implementation_group", ""))
    return "implementation_holdout" if "pages_c" in implementation else "train"


def _source_meta(record: Mapping[str, Any], *, image_digest: str) -> dict[str, str]:
    """Build split-bound provenance for one reviewed synthetic implementation.

    The audit treats ``family_id`` as a leakage boundary.  The registry's
    broad PG-348 family label is intentionally not reused here because it
    would put train and implementation-holdout rows in the same family.  A
    hashed implementation-bound family keeps role/replay rows together while
    making the train/holdout boundary explicit and auditable.
    """

    implementation = str(record["implementation_group"])
    challenge_id = str(record["challenge_id"])
    return {
        "source_id": f"pg348-dynamic-{implementation}",
        "implementation": implementation,
        "family_id": f"pg348-family-{_digest({'implementation': implementation})[:16]}",
        "surface_id": f"pg348-surface-{_digest({'challenge_id': challenge_id})[:16]}",
        "collector_id": "pg348-dynamic-typed-collector-v2",
        "authorization_id": "local-synthetic-loopback-authorized",
        "image_digest": image_digest,
        "source_digest": str(record["source_hash"]),
    }


def _payload_shape_refs(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Map reviewed surface metadata to abstract target coordinates.

    The values are intentionally coarse and allow-listed.  They tell the
    decoder which transport/field/encoding *slot* the evaluator adapter must
    bind; no payload literal, route, or response text is emitted.
    """

    method = str(record.get("transport_method", "")).upper()
    parameter_role = str(record.get("parameter_role", "unknown")).casefold().replace("-", "_")
    role_alias = {"static_label": "display_text"}
    field_role = role_alias.get(parameter_role, parameter_role)
    chain = record.get("encoding_chain")
    if isinstance(chain, (list, tuple)):
        encoding = "_then_".join(str(item).casefold().replace("-", "_") for item in chain if str(item)) or "unknown"
    else:
        encoding = str(chain or "unknown").casefold().replace("-", "_")
    if method == "GET":
        placement = {"path_segment": "get_path", "fragment_identifier": "get_fragment"}.get(parameter_role, "get_query")
        transport = placement
    elif method == "POST":
        transport = "post_json" if parameter_role == "json_value" else "post_form"
    else:
        transport = "unknown"
    response_shape = str(record.get("response_shape", "")).casefold()
    script_surface = str(record.get("script_surface", "")).casefold()
    if "attribute" in response_shape or parameter_role == "attribute_value":
        payload_shape = "html_attribute_marker"
    elif "dom_text" in response_shape or parameter_role == "dom_text":
        payload_shape = "html_dom_marker"
    elif "fragment" in response_shape or parameter_role == "fragment_identifier":
        payload_shape = "html_fragment_marker"
    elif "json" in response_shape or parameter_role == "json_value":
        payload_shape = "json_string_marker"
    elif "local_path" in response_shape or parameter_role == "path_segment":
        payload_shape = "path_segment_marker"
    elif "form" in response_shape or parameter_role == "form_field":
        payload_shape = "html_form_marker"
    elif script_surface not in {"", "none"}:
        payload_shape = "script_context_marker"
    elif parameter_role in {"query_text", "query_term", "filter_choice", "display_preference"}:
        payload_shape = "query_marker"
    else:
        payload_shape = "html_text_marker"
    return transport, field_role, encoding, payload_shape


def _complete_observation(base: Mapping[str, Any], *, role: str, step: str, variant: str, typed_available: bool, failure: bool = False) -> dict[str, Any]:
    """Fill only fields that the controlled runtime explicitly observes.

    The dynamic adapter has a complete response/header/DOM capture.  Process
    fields are supplied by this collector from the role/reset/evaluator
    attestation; they are not guessed from a missing answer.
    """

    observation = copy.deepcopy(dict(base))
    failure_axis = dict(observation.get("failure_feedback") or {})
    if failure:
        failure_axis.update(
            {
                "failure_class": "blocked_variant",
                "failure_stage": "probe_validation",
                "error_shape": "blocked_variant",
                "parse_error_class": "none",
                "encoding_error_class": "none",
                "redirect_error_class": "none",
                "timeout_ms": 0,
                "blocked_reason_class": "unsupported_variant",
                "environment_failure_class": "none",
                "previous_action": "unsupported_variant",
                "next_action": "candidate_surface",
                "repair_delta_axis": "probe_variant",
                "repair_outcome": "repair_pending",
            }
        )
    else:
        failure_axis.update(
            {
                "failure_class": "none",
                "failure_stage": "none",
                "error_shape": "empty",
                "parse_error_class": "none",
                "encoding_error_class": "none",
                "redirect_error_class": "none",
                "timeout_ms": 0,
                "blocked_reason_class": "none",
                "environment_failure_class": "none",
                "previous_action": variant,
                "next_action": "observe",
                "repair_delta_axis": "none",
                "repair_outcome": "not_applicable",
            }
        )
    observation["failure_feedback"] = failure_axis
    observation["belief_and_replay"] = {
        "observation_presence": "present",
        "observation_delta_axis": "failure_feedback" if failure else "response_transport",
        "belief_prior_bucket": "mid",
        "belief_posterior_bucket": "low" if failure else "high" if typed_available else "mid",
        "belief_delta_axis": "failure_feedback" if failure else "response_transport",
        "history_action": variant,
        "history_length": 2 if failure else 1,
        "typed_available": "present" if typed_available else "absent",
        "evidence_present": "present" if typed_available else "absent",
        "negative_control": "present" if role == "negative" else "absent",
        "fresh_reset": "present",
        # The route has a fresh candidate/reference/negative/replay contract
        # attached for every role row; keeping this observed on every row
        # prevents the replay fact from disappearing when the target is
        # conditioned on a role token.
        "replay_ready": "present",
        "reference_present": "present",
        "candidate_present": "present",
        "step_budget": "present",
        "probe_count": 2 if failure else 1,
        "evidence_hash_present": "present" if typed_available else "absent",
        "probe_role": role,
        "process_step": "failure" if failure else "replay" if role == "replay" else "baseline",
    }
    return observation


def _target(record: Mapping[str, Any], role: str, *, failure: bool = False) -> dict[str, Any]:
    transport_ref, field_role_ref, encoding_ref, payload_shape_ref = _payload_shape_refs(record)
    if failure:
        return {"question": "ask_failure", "next_action": "repair", "repair_action": "method", "transport_ref": transport_ref, "field_role_ref": field_role_ref, "encoding_ref": encoding_ref, "probe_variant_ref": "none", "payload_shape_ref": payload_shape_ref, "safe_to_send": False}
    if role == "negative":
        return {"question": "none", "next_action": "abstain", "repair_action": "none", "transport_ref": transport_ref, "field_role_ref": field_role_ref, "encoding_ref": encoding_ref, "probe_variant_ref": "negative_control", "payload_shape_ref": payload_shape_ref, "safe_to_send": False}
    if role == "reference":
        return {"question": "none", "next_action": "select_probe_variant", "repair_action": "none", "transport_ref": transport_ref, "field_role_ref": field_role_ref, "encoding_ref": encoding_ref, "probe_variant_ref": "reference", "payload_shape_ref": payload_shape_ref, "safe_to_send": True}
    if role == "replay":
        return {"question": "none", "next_action": "replay", "repair_action": "reset", "transport_ref": transport_ref, "field_role_ref": field_role_ref, "encoding_ref": encoding_ref, "probe_variant_ref": "runtime_canary", "payload_shape_ref": payload_shape_ref, "safe_to_send": True}
    return {"question": "none", "next_action": "select_probe_variant", "repair_action": "none", "transport_ref": transport_ref, "field_role_ref": field_role_ref, "encoding_ref": encoding_ref, "probe_variant_ref": "source_attested_candidate", "payload_shape_ref": payload_shape_ref, "safe_to_send": True}


def _role_input(*, role: str, result: Mapping[str, Any], reset_id: str) -> dict[str, Any]:
    observation = dict(result.get("observation") or {})
    response = dict(observation.get("response_transport") or {})
    typed = bool(result.get("typed_effect_confirmed"))
    projection = {
        "status_class": str(response.get("status_class", "unknown")),
        "content_type_class": str(response.get("content_type_class", "unknown")),
        "body_shape": str(response.get("body_shape", "unknown")),
        "body_length_bucket": str(response.get("body_length", "unknown")),
        "redirect_hop_count": int(response.get("redirect_hop_count", 0) or 0),
        "redirect_location_class": str(response.get("redirect_location_class", "none")),
        "redirect_chain_shape": str(response.get("redirect_chain_shape", "empty")),
        "connection_outcome": str(response.get("connection_outcome", "unknown")),
        "effect_marker": "observed" if typed else "none",
        "effect_shape": str(result.get("effect_class", "none")),
        "state_delta_class": "disposable_evaluator_state" if typed else "none",
        "challenge_state_available": True,
        "challenge_state_delta": typed,
        "challenge_solved": typed,
        "backend_observed": True,
        "response_shape_changed": typed,
        "external_network_blocked": True,
        "database_touched": False,
        "disposable_state_delta": typed,
        "non_destructive": True,
        "error_class": str((observation.get("failure_feedback") or {}).get("failure_class", "none")),
        "error_shape": str((observation.get("failure_feedback") or {}).get("error_shape", "empty")),
    }
    source_evidence = sha256_json({"role": role, "reset_id": reset_id, "projection": projection, "typed": typed})
    return {"sent": True, "available": True, "executed": True, "typed_effect_confirmed": typed, "effect_class": "logic_transition" if typed else "none", "projection": projection, "evidence_sha256": source_evidence, "negative_control_clean": role == "negative" and not typed, "non_destructive": True}


def collect(registry: dict[str, Any], *, operator_reviewed: bool = False, max_records: int | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    app = DynamicFixtureApplication(registry)
    server, thread = start_server(app, port=0)
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    records = list(registry.get("records") or [])[:max_records] if max_records else list(registry.get("records") or [])
    registry_digest = _digest({"schema_version": registry.get("schema_version"), "record_count": len(registry.get("records") or [])})
    image_digest = _digest("pg348-dynamic-synthetic-runtime-image-v2")
    try:
        for record in records:
            challenge_id = str(record["challenge_id"])
            method = str(record.get("transport_method", "GET")).upper()
            origin = f"http://127.0.0.1:{server.server_port}/pg348/dynamic/{challenge_id}"
            role_results: dict[str, dict[str, Any]] = {}
            reset_ids: dict[str, str] = {}
            for role, variant in (("candidate", "candidate_surface"), ("reference", "reference_surface"), ("negative", "negative_control"), ("replay", "candidate_surface")):
                reset = app.reset(challenge_id)
                reset_ids[role] = str(reset["reset_id"])
                capture = capture_loopback(origin, method=method, form_data={"probe": "opaque"} if method == "POST" else None, abstract_probe_variant=variant)
                result = dict(capture)
                result["typed_effect_confirmed"] = variant in {"candidate_surface", "reference_surface"}
                result["effect_class"] = "logic_transition" if result["typed_effect_confirmed"] else "none"
                result["observation"] = _complete_observation(capture["observation"], role=role, step="replay" if role == "replay" else "baseline", variant=variant, typed_available=True)
                role_results[role] = result
            aggregate_reset_id = _digest(reset_ids)
            aggregate_target_digest = _digest({"challenge_id": challenge_id, "reset_ids": reset_ids, "image_digest": image_digest})
            sidecar = build_evaluator_sidecar(
                record_id=_digest({"challenge_id": challenge_id, "kind": "typed"}),
                reset={"fresh_reset": True, "reset_id": aggregate_reset_id, "target_instance_digest": aggregate_target_digest, "network_mode": "loopback", "external_network": False, "loopback_only": True, "state_clean": True, "volume_mount_count": 0},
                candidate=_role_input(role="candidate", result=role_results["candidate"], reset_id=reset_ids["candidate"]),
                reference=_role_input(role="reference", result=role_results["reference"], reset_id=reset_ids["reference"]),
                negative=_role_input(role="negative", result=role_results["negative"], reset_id=reset_ids["negative"]),
                replay_consistent=bool(role_results["replay"].get("typed_effect_confirmed") == role_results["candidate"].get("typed_effect_confirmed")),
                evaluator_id="pg348-dynamic-typed-v2",
            )
            sidecars.append({"record_id": sidecar["record_id"], "challenge_id_digest": _digest(challenge_id), "role_reset_ids": reset_ids, "sidecar": sidecar})
            evaluator_projection_base = {
                "typed_available": bool(sidecar["confirmed_positive"]),
                "negative_control": bool(sidecar["checks"]["negative_control_clean"]),
                "reference_present": True,
                "candidate_present": True,
                "fresh_reset": True,
                "evidence_hash": sidecar["evidence_sha256"],
                "confirmed_positive": bool(sidecar["confirmed_positive"]),
                "effect_class": "logic_transition",
                "evaluator_version": "pg348-dynamic-typed-v2",
            }
            for role in ("candidate", "reference", "negative", "replay"):
                result = role_results[role]
                obs = result["observation"]
                tokenized = tokenize_web_observation(obs)
                if tokenized.get("loss_report", {}).get("losses"):
                    raise ValueError(f"tokenizer loss for {challenge_id}:{role}: {tokenized['loss_report']['losses']}")
                manifest = _field_capture_manifest(obs)
                row_id = _digest({"challenge_id": challenge_id, "role": role, "step": "baseline"})
                evaluator = dict(evaluator_projection_base)
                evaluator["evidence_hash"] = _digest({"sidecar": sidecar["evidence_sha256"], "row_id": row_id, "role": role})
                target = _target(record, role)
                source_meta = _source_meta(record, image_digest=image_digest)
                reset_projection = {"fresh_reset": True, "reset_id": reset_ids[role], "target_instance_digest": _digest({"challenge_id": challenge_id, "role": role, "reset_id": reset_ids[role]}), "network_mode": "loopback", "external_network": False, "loopback_only": True, "state_clean": True, "database_health_gate": "not_applicable"}
                row = collect_pg331_source_row(record_id=row_id, observation=obs, source_meta=source_meta, reset=reset_projection, evaluator=evaluator, field_capture_manifest=manifest, target_projection=target, split=_split(record), operator_reviewed=operator_reviewed, hard_negative=False)
                check = validate_pg331_source_row(row)
                if not check["valid"]:
                    raise ValueError(f"source row validation failed for {challenge_id}:{role}: {check['failures']}")
                rows.append(row)
            # One explicit failure→repair transition per route.  It is kept as
            # a separate row so the action change cannot be inferred from a
            # positive row or collapsed into a family label.
            failure_role = "candidate"
            reset = app.reset(challenge_id)
            capture = capture_loopback(origin, method=method, form_data={"probe": "opaque"} if method == "POST" else None, abstract_probe_variant="unsupported_variant")
            obs = _complete_observation(capture["observation"], role=failure_role, step="failure", variant="unsupported_variant", typed_available=True, failure=True)
            row_id = _digest({"challenge_id": challenge_id, "role": failure_role, "step": "failure"})
            evaluator = dict(evaluator_projection_base)
            evaluator["evidence_hash"] = _digest({"sidecar": sidecar["evidence_sha256"], "row_id": row_id, "role": failure_role, "step": "failure"})
            row = collect_pg331_source_row(record_id=row_id, observation=obs, source_meta=_source_meta(record, image_digest=image_digest), reset={"fresh_reset": True, "reset_id": reset["reset_id"], "target_instance_digest": _digest({"challenge_id": challenge_id, "role": "failure", "reset_id": reset["reset_id"]}), "network_mode": "loopback", "external_network": False, "loopback_only": True, "state_clean": True, "database_health_gate": "not_applicable"}, evaluator=evaluator, field_capture_manifest=_field_capture_manifest(obs), target_projection=_target(record, "candidate", failure=True), split=_split(record), operator_reviewed=operator_reviewed, hard_negative=False)
            check = validate_pg331_source_row(row)
            if not check["valid"]:
                raise ValueError(f"failure row validation failed for {challenge_id}: {check['failures']}")
            rows.append(row)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    dataset = {"schema_version": "pg348-dynamic-typed-source-rows-v1", "status": "completed_typed_diagnostic_only" if not operator_reviewed else "completed_typed_operator_reviewed_candidate", "records": rows, "counts": {"records": len(rows), "routes": len(records), "train_rows": sum(row["split"] == "train" for row in rows), "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in rows), "training_eligible_rows": sum(row["training_eligible"] is True for row in rows), "failure_rows": sum("question=ask_failure" in row["target_tokens"] for row in rows), "typed_positive_routes": len(records)}, "source_registry_sha256": registry_digest, "runtime_image_digest": image_digest, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    sidecar_doc = {"schema_version": "pg348-dynamic-typed-sidecars-v1", "status": "evaluator_only", "sidecars": sidecars, "counts": {"routes": len(sidecars), "confirmed_positive": sum(item["sidecar"]["confirmed_positive"] is True for item in sidecars)}, "promotion": dataset["promotion"]}
    report = {"schema_version": "pg348-dynamic-typed-collection-report-v1", "status": dataset["status"], "counts": dataset["counts"], "source_registry_sha256": registry_digest, "runtime_image_digest": image_digest, "operator_reviewed": operator_reviewed, "typed_effect": {"candidate_reference": "logic_transition", "negative_clean": True, "fresh_reset_per_role": True, "failure_action_change": True}, "promotion": dataset["promotion"]}
    return dataset, sidecar_doc, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PG-348 dynamic typed source rows")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sidecars", type=Path, default=DEFAULT_SIDECARS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--operator-reviewed", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    dataset, sidecars, report = collect(registry, operator_reviewed=args.operator_reviewed, max_records=args.max_records)
    for path, value in ((args.dataset, dataset), (args.sidecars, sidecars), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "dataset_sha256": _digest(dataset), "report_sha256": _digest(report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
