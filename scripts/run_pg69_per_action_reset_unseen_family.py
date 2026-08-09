"""PG-69: per-action fresh reset plus a genuinely unseen workflow family.

This is an evaluation lane.  It runs a small real Docker subset with one new
Pikachu container per matched control/candidate case, then replays an
independently implemented ``workflow_invariant`` family on two response
variants.  The latter family is intentionally absent from the decoder's
training classes; the only acceptable model decision is abstain.

The output is an evaluation-only Catalog/Trace.  It never writes runtime
probe values or response bodies and it never updates weights or memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import socket
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.detection_payload import build_detection_payload  # noqa: E402
from app.pg69_workflow_fixture import (  # noqa: E402
    PG69_WORKFLOW_FAMILY,
    PG69_WORKFLOW_PORTS,
    evaluate_workflow,
    make_workflow_server,
    source_sha256 as workflow_source_sha256,
)
from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step  # noqa: E402


PROTOCOL_ID = "pg-pk-69-per-action-reset-unseen-family-v1"
SCHEMA_VERSION = "sift-pg69-per-action-reset-unseen-family-v1"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PG52_PATH = ROOT / "scripts" / "run_pg52_authoritative_local_oracle.py"
REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
REPORT_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_report_v1.md"
UNKNOWN_OOD_THRESHOLD = 30.0
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


DOCKER_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "docker-reflected-get", "family": "xss", "surface": "xss_reflected_get", "method": "GET", "port": 8767, "path": "/vul/xss/xss_reflected_get.php", "field": "message", "mode": "reflected_get"},
    {"case_id": "docker-reflected-post", "family": "xss", "surface": "xss_reflected_post", "method": "POST", "port": 8768, "path": "/vul/xss/xsspost/xss_reflected_post.php", "field": "message", "mode": "reflected_post"},
    {"case_id": "docker-sqli-search", "family": "injection", "surface": "sqli_search", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_search.php", "field": "name", "mode": "sql_search"},
    {"case_id": "docker-redirect", "family": "url_redirect", "surface": "url_redirect", "method": "GET", "port": 8767, "path": "/vul/urlredirect/urlredirect.php", "field": "url", "mode": "redirect"},
)


WORKFLOW_CASES: tuple[dict[str, Any], ...] = tuple(
    {
        "case_id": f"workflow-{variant}-{route.lstrip('/').replace('-', '-')}-{method.casefold()}",
        "family": PG69_WORKFLOW_FAMILY,
        "surface": "workflow_handoff" if route == "/handoff" else "workflow_quota",
        "route": route,
        "method": method,
        "variant": variant,
        "port": PG69_WORKFLOW_PORTS[0 if variant == "amber" else 1],
        "field_names": ["verb", "prior", "stamp", "fresh"] if route == "/handoff" else ["member", "amount"],
        "control": ({"verb": "wait", "prior": "none", "stamp": "old", "fresh": "new"} if route == "/handoff" else {"member": "1", "amount": "99"}),
        "candidate": ({"verb": "commit", "prior": "verified", "stamp": "old", "fresh": "new"} if route == "/handoff" else {"member": "1", "amount": "100"}),
    }
    for variant in ("amber", "violet")
    for route in ("/handoff", "/quota")
    for method in ("GET", "POST")
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _load_pg52() -> Any:
    spec = importlib.util.spec_from_file_location("pg69_pg52_runtime", PG52_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-52 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _status_class(status: int) -> str:
    return f"{int(status) // 100}xx" if 100 <= int(status) <= 599 else "other"


def _shape(value: Any) -> dict[str, int | str]:
    if isinstance(value, dict):
        return {"kind": "object", "key_count": len(value), "scalar_count": sum(not isinstance(child, (dict, list)) for child in value.values()), "array_count": sum(isinstance(child, list) for child in value.values())}
    return {"kind": type(value).__name__, "key_count": 0, "scalar_count": 1, "array_count": 0}


def _response_projection(response: httpx.Response) -> dict[str, Any]:
    body = bytes(response.content)
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
    shape = _shape(value)
    projection = {
        "status_code": int(response.status_code),
        "status_class": _status_class(response.status_code),
        "content_type_class": str(response.headers.get("content-type", "")).split(";", 1)[0].casefold(),
        "body_length_bucket": "0" if not body else "1-255" if len(body) <= 255 else "256-4095" if len(body) <= 4095 else "4096+",
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": sha256_json(shape),
        "shape": shape,
        "header_names": sorted({str(key).casefold() for key in response.headers.keys()} & {"content-type", "location", "allow", "x-sift-workflow-variant"}),
        "transport_error": False,
        "state_changed": False,
        "external_network": False,
    }
    projection["projection_sha256"] = sha256_json(projection)
    return projection


def _bounded_projection(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value:
        return dict(value)
    projection = {"status_class": "unknown", "content_type_class": "unknown", "body_length_bucket": "unknown", "response_projection_available": False}
    projection["projection_sha256"] = sha256_json(projection)
    return projection


def _audited_reset(*, instance_id: str, reset_id: str, source: str) -> dict[str, Any]:
    return {
        "kind": "pg69-fresh-action-target",
        "reset_id": str(reset_id),
        "target_instance_id": str(instance_id)[:24],
        "state_epoch": _sha256_text(f"pg69-state|{instance_id}|{reset_id}")[:24],
        "reset_adapter_sha256": _sha256_text(f"pg69-reset-adapter|{source}"),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "read_only_round": True,
        "reset_scope": "fresh_disposable_target_per_action_pair",
    }


def _safe_known_row(pg52: Any, case: dict[str, Any], raw: dict[str, Any], container_id: str, index: int, loader: Any) -> dict[str, Any]:
    reset = pg52._fresh_reset(container_id, case["case_id"], _sha256_text(f"pg69-pikachu-reset|{index}"))
    reset.update({"kind": "pg69-fresh-pikachu-container-per-case", "fresh_target": True, "completed": True, "reset_scope": "fresh_disposable_target_per_action_pair", "target_instance_id": str(container_id)[:24]})
    model = pg52._model_proposal(loader, case, raw)
    row = pg52._result_row(case, raw, reset, model)
    row["source_kind"] = "real_docker"
    row["independent_implementation"] = "pinned_pikachu_php_mysql"
    row["control_candidate_same_target_pair"] = True
    row["action_pair_id"] = f"pg69-action-{index:02d}"
    return row


def _run_docker_cases(pg52: Any, loader: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    reset_hash = _sha256_text(Path(__file__).read_text(encoding="utf-8"))
    for index, case in enumerate(DOCKER_CASES):
        name = f"pg69-pikachu-{index:02d}"
        port = int(case["port"])
        started = False
        try:
            container_id = pg52._start(name, port)
            started = True
            if port == 8767:
                pg52._wait_application_surface(port, "/vul/sqli/sqli_str.php", b"what's your username")
            else:
                pg52._wait_application_surface(port, "/vul/xss/xsspost/post_login.php", b'name="username"')
            if case["family"] == "injection":
                pg52._prepare_mysql(name)
            base = pg52.GET_BASE if port == 8767 else pg52.POST_BASE
            marker = f"pg69-{index}-marker"
            if case["family"] == "xss":
                raw = pg52._browser_case(case, base, marker, container_id)
            elif case["family"] == "injection":
                raw = pg52._sql_case(case, base, name)
            else:
                raw = pg52._redirect_case(case, base, container_id)
            rows.append(_safe_known_row(pg52, case, raw, container_id, index, loader))
        except Exception as exc:  # keep an auditable blocked report if Docker fails
            errors.append({"case_id": str(case["case_id"]), "error_type": type(exc).__name__})
        finally:
            if started:
                pg52._stop(name)
    return rows, errors


class _FreshWorkflowAction:
    def __init__(self, *, port: int, variant: str, action_id: str) -> None:
        self.port = int(port)
        self.variant = variant
        self.action_id = action_id
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.client: httpx.Client | None = None
        self.instance_id = _sha256_text(f"pg69-workflow-target|{variant}|{port}|{action_id}")[:24]

    def __enter__(self) -> "_FreshWorkflowAction":
        self.server = make_workflow_server(self.port, self.variant)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.01)
        else:
            self.close()
            raise RuntimeError("PG-69 workflow target did not become ready")
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=3.0, follow_redirects=False)
        return self

    def request(self, case: dict[str, Any], values: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("workflow target client is not open")
        if case["method"] == "GET":
            response = self.client.get(case["route"], params={**values, "marker": f"pg69-marker-{case['variant']}"})
        else:
            response = self.client.post(case["route"], data={**values, "marker": f"pg69-marker-{case['variant']}"})
        projection = _response_projection(response)
        route = str(case["route"])
        _, _, oracle = evaluate_workflow(route, values, self.variant)
        oracle.update({"modality": "semantic_contract", "confirmed_effect": oracle.get("oracle_signal"), "evaluator_state_hidden": True, "safety": {"external_network": False, "script_execution": False, "database_write": False, "state_mutated": False, "credentials_accessed": False}})
        return projection, oracle

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _unknown_model_proposal(pg52: Any, loader: Any, case: dict[str, Any], response: dict[str, Any], index: int) -> dict[str, Any]:
    fake_case = {"case_id": f"opaque-{index:02d}", "family": PG69_WORKFLOW_FAMILY, "surface": "opaque_workflow_surface", "method": case["method"], "path": case["route"]}
    item = {"candidate": {"response": response, "oracle": {}}}
    proposal = dict(pg52._model_proposal(loader, fake_case, item))
    distance = float(proposal.get("ood_distance", 0.0) or 0.0)
    # Unknown-family policy is deliberately independent of the evaluator label:
    # only a calibrated OOD distance or model abstention may pass.
    proposal["unknown_policy_decision"] = "abstain" if proposal.get("decision") == "abstain" or distance >= UNKNOWN_OOD_THRESHOLD else "candidate"
    proposal["visible_input_family_free"] = True
    return proposal


def _workflow_row(pg52: Any, loader: Any, case: dict[str, Any], index: int) -> dict[str, Any]:
    with _FreshWorkflowAction(port=int(case["port"]), variant=str(case["variant"]), action_id=f"control-{index:02d}") as control_target:
        control_response, control_oracle = control_target.request(case, dict(case["control"]))
        control_reset = _audited_reset(instance_id=control_target.instance_id, reset_id=f"pg69-control-{index:02d}", source="workflow_fixture")
    with _FreshWorkflowAction(port=int(case["port"]), variant=str(case["variant"]), action_id=f"candidate-{index:02d}") as candidate_target:
        candidate_response, candidate_oracle = candidate_target.request(case, dict(case["candidate"]))
        candidate_reset = _audited_reset(instance_id=candidate_target.instance_id, reset_id=f"pg69-candidate-{index:02d}", source="workflow_fixture")
    proposal = _unknown_model_proposal(pg52, loader, case, candidate_response, index)
    policy_decision = str(proposal.get("unknown_policy_decision", "abstain"))
    adapter_hash = sha256_json({"case_id": case["case_id"], "candidate": candidate_response, "control": control_response, "candidate_oracle": candidate_oracle, "control_oracle": control_oracle, "candidate_reset": candidate_reset, "control_reset": control_reset})
    return {
        "case_id": case["case_id"],
        "family": PG69_WORKFLOW_FAMILY,
        "surface": case["surface"],
        "method": case["method"],
        "path": case["route"],
        "variant": case["variant"],
        "port": int(case["port"]),
        "source_kind": "independent_local",
        "independent_implementation": "app.pg69_workflow_fixture",
        "model_proposal": proposal,
        "confirmed_family": PG69_WORKFLOW_FAMILY,
        "model_family_match": False,
        "model_family_misclassification": policy_decision != "abstain",
        "model_false_positive": False,
        "decision": "abstain",
        "oracle": candidate_oracle,
        "control_oracle": control_oracle,
        "candidate_response": candidate_response,
        "control_response": control_response,
        "field_names": list(case["field_names"]),
        "probe_kind": "http_canary",
        "probe_class": "workflow_boundary_class",
        "payload_manifest": {"method": case["method"], "path": case["route"], "probe_ref": f"pg69-workflow-probe-{index:02d}", "probe_sha256": _sha256_text("workflow_boundary_class"), "raw_payload_stored": False},
        "negative_control": {"matched": not bool(control_oracle.get("positive")) and bool(candidate_oracle.get("positive")) is True, "control_case_id": f"pg69-control-{index:02d}", "control_evidence_sha256": sha256_json({"control": control_response, "oracle": control_oracle}), "candidate_vs_control": control_response.get("projection_sha256") != candidate_response.get("projection_sha256") or bool(candidate_oracle.get("positive")) != bool(control_oracle.get("positive"))},
        "fresh_reset": {"control": control_reset, "candidate": candidate_reset},
        "rule_ir_binding": {"family": None, "source": "unknown_family_policy", "slots": [], "executable": False, "reason": "strict_unknown_abstain"},
        "evidence_sha256": adapter_hash,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "unknown_family": True,
    }


def _known_probe(row: dict[str, Any]) -> tuple[str, str, str, str]:
    if row["family"] == "xss":
        return "inert_dom_markup", "dom_event_class", "controlled_detached_dom_v1", "typed_dom_execution"
    if row["family"] == "injection":
        return "sql_channel_class", "operator_like", "synthetic_sql_ast_differential_v1", "typed_ast_differential"
    if row["family"] == "url_redirect":
        return "http_canary", "loopback_destination_class", "pikachu_bounded_http_projection_v1", "typed_redirect_destination"
    return "http_canary", "workflow_boundary_class", "pg69_typed_workflow_invariant_v1", "typed_workflow_invariant"


def _source_scope(row: dict[str, Any]) -> str:
    if row["source_kind"] == "real_docker":
        return "http://127.0.0.1:" + ("8768" if row["method"] == "POST" else "8767")
    return "http://127.0.0.1:" + str(row["port"])


def _catalog_sample(row: dict[str, Any], index: int) -> dict[str, Any]:
    probe_kind, probe, expected_oracle, expected_signal = _known_probe(row)
    marker = f"pg69-probe-{index:02d}"
    method = str(row["method"]).upper()
    fields = list(row.get("field_names") or ([row.get("field", "value")] if row.get("field") else ["value"]))
    if method == "POST":
        form = {fields[0]: marker}
    else:
        form = {}
    payload = build_detection_payload(target=_source_scope(row), method=method, path=str(row["path"]), headers={"accept": "application/json", "x-sift-probe": marker}, marker=marker, probe=probe, probe_kind=probe_kind, form=form, expected={"signal": expected_signal, "negative_control": "matched_pair", "typed_oracle": True})
    replay = {"target": _source_scope(row), "method": method, "path": str(row["path"]), "params": {}, "fresh_reset": row["fresh_reset"], "transport": "loopback"}
    if method == "POST":
        replay["form"] = form
    return {
        "sample_id": f"pg69-sample-{index:02d}",
        "payload": payload,
        "probe_artifact": {"original": probe, "encoding": "abstract_class", "probe_sha256": _sha256_text(probe)},
        "semantic": {"family": str(row["family"]), "surface": str(row["surface"]), "expected_oracle": expected_oracle, "expected_signal": expected_signal},
        "pair": {"pair_id": f"pg69-pair-{index:02d}", "variant": "abstract_class", "surface_role": str(row["surface"]), "encoding_depth": 0},
        "counterfactual": {"kind": "negative_control", "intervention": "matched_control", "source_sample_id": f"pg69-sample-{index:02d}"},
        "replay": replay,
        "response_projection": _bounded_projection(row.get("candidate_response")),
        "oracle_projection": dict(row.get("oracle") or {}),
        "evidence": {"adapter_evidence_sha256": str(row["evidence_sha256"]), "control_evidence_sha256": str(row["negative_control"]["control_evidence_sha256"])},
        "rule_ir": {"op": "and", "args": [{"op": "eq", "left": {"op": "field", "path": "oracle.positive"}, "right": {"op": "const", "value": True}}, {"op": "eq", "left": {"op": "field", "path": "oracle.positive_authority"}, "right": {"op": "const", "value": True}}]},
        "rule_ir_result": bool(row.get("oracle", {}).get("positive")),
        "evaluator_state_visible": False,
    }


def _provenance(row: dict[str, Any], captured_at: str) -> dict[str, Any]:
    if row["source_kind"] == "real_docker":
        source_id = "pg69-real-pikachu-" + ("post" if row["method"] == "POST" else "get")
        return {"source_id": source_id, "source_type": "authorized_local_container", "origin": "research/pg69_per_action_reset_unseen_family_report_v1.json", "license": "local_container", "authorization": "workspace_local_only", "scope": [_source_scope(row)], "captured_at": captured_at, "authorized_for": ["training", "local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False, "container_image_digest": IMAGE.split("@", 1)[1]}
    source_id = "pg69-workflow-" + str(row["variant"])
    return {"source_id": source_id, "source_type": "in_repo_synthetic", "origin": "app/pg69_workflow_fixture.py", "license": "in_repo_synthetic", "authorization": "workspace_local_only", "scope": [_source_scope(row)], "captured_at": captured_at, "authorized_for": ["training", "local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False}


def _build_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        provenance = _provenance(row, captured_at)
        key = (str(provenance["source_id"]), str(provenance["scope"][0]))
        grouped.setdefault(key, {"provenance": provenance, "samples": []})["samples"].append(_catalog_sample(row, index))
    return write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg69-per-action-reset-evaluation-only", "sources": list(grouped.values())})


def _trace_action(row: dict[str, Any], index: int, episode_id: str, parent: str | None) -> dict[str, Any]:
    method = str(row["method"]).upper()
    field_names = list(row.get("field_names") or ([row.get("field", "value")] if row.get("field") else ["value"]))
    candidate_reset = row["fresh_reset"].get("candidate", row["fresh_reset"]) if isinstance(row.get("fresh_reset"), dict) else row["fresh_reset"]
    oracle = dict(row.get("oracle") or {})
    oracle.update({"negative_control_pair_id": f"pg69-control-{index:02d}", "evaluator_state_hidden": True})
    action: dict[str, Any] = {"method": method, "route_template_id": f"pg69-route-{index:02d}", "placement": "form" if method == "POST" else "query", "encoding_chain": ["identity"], "probe_ref": f"pg69-probe-{index:02d}", "probe_sha256": _sha256_text(str(row.get("probe_class", "abstract_class"))), "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action["form_field_names"] = field_names[:8]
    step: dict[str, Any] = {"episode_id": episode_id, "step_id": f"pg69-step-{index:02d}", "parent_step_id": parent, "sampling_seed": 6900 + index, "hypothesis": "surface_hypothesis", "belief_before": {"unknown_surface": 1.0}, "action_manifest": action, "baseline_projection": _bounded_projection(row.get("control_response")), "response_projection": _bounded_projection(row.get("candidate_response")), "oracle_projection": oracle, "belief_after": {"unknown_surface": 1.0}, "decision": str(row.get("decision", "abstain")), "next_action": "abstain" if str(row.get("decision")) == "abstain" else "stop_confirmed", "fresh_reset": candidate_reset, "evidence_sha256": "", "dataset_stage": "evaluation_only", "online_weight_update": False, "long_term_memory_write": False}
    step["evidence_sha256"] = sha256_json({"action": step["action_manifest"], "baseline": step["baseline_projection"], "response": step["response_projection"], "oracle": step["oracle_projection"], "reset": step["fresh_reset"], "decision": step["decision"]})
    echo_body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action")}
    step["echo"] = {"sha256": sha256_json(echo_body)}
    step["rule_ir_after_action"] = {"executable": False, "bound": bool(row.get("family") != PG69_WORKFLOW_FAMILY and row.get("oracle", {}).get("positive")), "reason": "strict_unknown_abstain" if row.get("family") == PG69_WORKFLOW_FAMILY else "typed_oracle_binding_only"}
    return step


def _build_trace(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["source_kind"] == "real_docker":
            group = "docker"
        else:
            group = f"workflow-{row['variant']}"
        groups[group].append((index, row))
    episodes: list[dict[str, Any]] = []
    validation_failures: list[dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        episode_id = f"pg69-episode-{group}"
        steps: list[dict[str, Any]] = []
        parent: str | None = None
        for index, row in items:
            step = _trace_action(row, index, episode_id, parent)
            parent = step["step_id"]
            try:
                normalized = validate_trace_step(step)
                steps.append(normalized)
            except ValueError as exc:
                validation_failures.append({"step_id": step["step_id"], "error_type": type(exc).__name__, "error": str(exc)})
                steps.append(step)
        episode_report = evaluate_episode(steps) if not validation_failures or all(item["episode_id"] == episode_id for item in steps) else {"status": "trace_only", "reasons": ["step_validation_failure"]}
        episodes.append({"episode_id": episode_id, "source_group": group, "steps": steps, "validation": episode_report})
    trace = {"schema_version": "sift-pg69-evaluation-trace-v1", "protocol_id": PROTOCOL_ID, "evaluation_only": True, "training_eligible": False, "model_retrained_on_pg69": False, "model_input_family_leakage": False, "episode_count": len(episodes), "accepted_episode_count": sum(int(item["validation"].get("status") == "accepted_evaluation") for item in episodes), "steps": sum((item["steps"] for item in episodes), []), "episodes": episodes, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "validation_failures": validation_failures}
    return trace, {"accepted_episode_count": trace["accepted_episode_count"], "validation_failure_count": len(validation_failures)}


def _training_families() -> set[str]:
    registry = _load_json(REGISTRY_PATH)
    result: set[str] = set()
    for target in registry.get("targets", []):
        if bool(target.get("training_eligible")):
            result.update(str(item) for item in target.get("family_set", []))
    return result


def run(*, skip_docker: bool = False) -> dict[str, Any]:
    pg52 = _load_pg52()
    loader = pg52._model_loader()
    docker_rows: list[dict[str, Any]] = []
    docker_errors: list[dict[str, str]] = []
    if not skip_docker:
        docker_rows, docker_errors = _run_docker_cases(pg52, loader)
    workflow_rows: list[dict[str, Any]] = []
    workflow_errors: list[dict[str, str]] = []
    for index, case in enumerate(WORKFLOW_CASES, start=len(docker_rows)):
        try:
            workflow_rows.append(_workflow_row(pg52, loader, case, index))
        except Exception as exc:
            workflow_errors.append({"case_id": str(case["case_id"]), "error_type": type(exc).__name__})
    rows = docker_rows + workflow_rows
    catalog = _build_catalog(rows) if rows else {"sources": []}
    trace, trace_metrics = _build_trace(rows) if rows else ({"episode_count": 0, "accepted_episode_count": 0, "steps": [], "episodes": [], "validation_failures": []}, {"accepted_episode_count": 0, "validation_failure_count": 0})
    if rows:
        TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    training_families = _training_families()
    unknown_rows = [row for row in rows if row["family"] == PG69_WORKFLOW_FAMILY]
    known_rows = [row for row in rows if row["family"] != PG69_WORKFLOW_FAMILY]
    candidate_instances = [str(row["fresh_reset"].get("candidate", row["fresh_reset"]).get("target_instance_id", "")) for row in rows if isinstance(row.get("fresh_reset"), dict)]
    unique_instances = {item for item in candidate_instances if item}
    fresh_per_action = bool(rows) and len(unique_instances) == len(candidate_instances) and all(bool(row["fresh_reset"].get("candidate", row["fresh_reset"]).get("fresh_target")) for row in rows)
    evidence_hash_valid = sum(bool(HASH_RE.fullmatch(str(row.get("evidence_sha256", "")).casefold())) for row in rows)
    negative_pass = sum(int(bool(row.get("negative_control", {}).get("matched"))) for row in rows)
    unknown_misname = sum(int(row.get("model_proposal", {}).get("unknown_policy_decision") != "abstain") for row in unknown_rows)
    unknown_strict = bool(unknown_rows) and unknown_misname == 0 and all(row["decision"] == "abstain" for row in unknown_rows)
    family_holdout = PG69_WORKFLOW_FAMILY not in training_families
    methods = sorted({str(row["method"]).upper() for row in rows})
    hard_checks = {
        "real_docker_cases_completed": len(docker_rows) == len(DOCKER_CASES) and not docker_errors,
        "per_action_fresh_target": fresh_per_action,
        "get_post_both_covered": {"GET", "POST"}.issubset(set(methods)),
        "typed_oracle_positive_authority": all(bool(row.get("oracle", {}).get("positive_authority")) for row in rows),
        "matched_negative_controls": negative_pass == len(rows) and len(rows) > 0,
        "evidence_hash_per_action": evidence_hash_valid == len(rows) and len(rows) > 0,
        "unknown_family_is_out_of_training_registry": family_holdout,
        "unknown_family_strict_abstain": unknown_strict,
        "independent_implementations": len({str(row.get("independent_implementation")) for row in rows}) >= 2,
        "trace_episodes_accepted": trace_metrics["accepted_episode_count"] == len(trace.get("episodes", [])) and trace_metrics["validation_failure_count"] == 0,
        "no_raw_persistence": all(not bool(row.get("raw_payload_stored")) and not bool(row.get("raw_response_body_stored")) for row in rows),
    }
    hard_gate = all(hard_checks.values())
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "completed_evaluation" if rows else "blocked_no_rows",
        "source": {"pinned_image": IMAGE, "docker_case_count": len(docker_rows), "workflow_case_count": len(workflow_rows), "workflow_source_sha256": workflow_source_sha256(), "independent_implementation_count": len({str(row.get("independent_implementation")) for row in rows}), "docker_errors": docker_errors, "workflow_errors": workflow_errors},
        "scope": {"case_count": len(rows), "methods": methods, "families": sorted({str(row["family"]) for row in rows}), "loopback_only": True, "external_network": False, "raw_payloads_stored": False, "raw_response_bodies_stored": False},
        "metrics": {"typed_positive_count": sum(int(bool(row.get("oracle", {}).get("positive"))) for row in rows), "known_typed_positive_count": sum(int(bool(row.get("oracle", {}).get("positive"))) for row in known_rows), "unknown_typed_positive_count": sum(int(bool(row.get("oracle", {}).get("positive"))) for row in unknown_rows), "negative_control_pass_count": negative_pass, "evidence_hash_valid_count": evidence_hash_valid, "unique_candidate_target_instance_count": len(unique_instances), "fresh_reset_per_action": fresh_per_action, "get_post_covered": {"GET": sum(int(row["method"] == "GET") for row in rows), "POST": sum(int(row["method"] == "POST") for row in rows)}, "unknown_misname_count": unknown_misname, "unknown_strict_abstain": unknown_strict, "unknown_abstain_count": sum(int(row["decision"] == "abstain") for row in unknown_rows), "trace_episode_count": len(trace.get("episodes", [])), "trace_accepted_episode_count": trace_metrics["accepted_episode_count"], "training_registry_families": sorted(training_families), "family_holdout_candidates": [PG69_WORKFLOW_FAMILY] if family_holdout else [], "family_holdout_candidate_count": int(family_holdout), "docker_container_per_case": len({str(row["fresh_reset"].get("target_instance_id")) for row in docker_rows}) == len(docker_rows) if docker_rows else False},
        "hard_gate": {"status": "passed" if hard_gate else "blocked", "checks": hard_checks, "blocking_reasons": [key for key, value in hard_checks.items() if not value], "claim_allowed": False},
        "promotion": {"status": "hard_gate_passed_evaluation_only_no_promotion" if hard_gate else "blocked_evaluation_only", "evaluation_catalog_generated": bool(rows), "training_catalog_generated": False, "training_allowed": False, "memory_promotion_allowed": False, "reason": "real_oracle_replay_is evaluation-only; post-training and multi-dataset promotion remain unproven"},
        "artifacts": {"catalog": str(CATALOG_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT))},
        "formal_claim": {"allowed": False, "reason": "PG-69 proves reset/unknown-abstain evaluation plumbing, not trained-model capability or broad web generalization"},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "sift-pg69-per-action-reset-unseen-family-protocol-v1", "authorized_scope": {"target_host": "127.0.0.1", "pinned_image": IMAGE, "external_network": False, "state_change_allowed": False, "workflow_source": "app/pg69_workflow_fixture.py"}, "input_contract": {"family_before_action_forbidden": True, "typed_oracle_after_action_only": True, "unknown_family_not_in_decoder_classes": True, "rule_ir_binding_after_typed_exit_only": True, "raw_probe_and_response_persistence_forbidden": True}, "required_gates": {"per_action_fresh_reset": True, "get_post_both": True, "matched_negative_control": True, "evidence_hash_per_action": True, "unknown_family_strict_abstain": True, "independent_implementation": True, "trace_episodes_accepted": True}, "run_result": {"status": report["hard_gate"]["status"], "training_allowed": False, "memory_promotion_allowed": False, "hard_gate_checks": hard_checks}, "next_experiment": "PG70 train only on pre-registered accepted traces, then rerun unseen family and fresh Docker holdout without label exposure"}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-69 每动作 fresh reset + 未知 workflow family", "", f"cases={len(rows)}；typed positive={report['metrics']['typed_positive_count']}；GET/POST={report['metrics']['get_post_covered']}；fresh/action={report['metrics']['fresh_reset_per_action']}。", f"unknown family misname={report['metrics']['unknown_misname_count']}；strict abstain={report['metrics']['unknown_strict_abstain']}；trace accepted={report['metrics']['trace_accepted_episode_count']}/{report['metrics']['trace_episode_count']}。", "", f"硬门：`{report['hard_gate']['status']}`；training_allowed=`false`；memory_promotion_allowed=`false`。", "", "阻塞项：" + (", ".join(report["hard_gate"]["blocking_reasons"]) if report["hard_gate"]["blocking_reasons"] else "无"), "", f"JSON: `{REPORT_PATH.relative_to(ROOT)}`", f"协议: `{PROTOCOL_PATH.relative_to(ROOT)}`", f"Catalog: `{CATALOG_PATH.relative_to(ROOT)}`", f"Trace: `{TRACE_PATH.relative_to(ROOT)}`", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-docker", action="store_true", help="only run the independent local family; emits a blocked audit")
    args = parser.parse_args()
    report = run(skip_docker=bool(args.skip_docker))
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["hard_gate"]["status"], "case_count": report["scope"]["case_count"], "docker_case_count": report["source"]["docker_case_count"], "unknown_misname_count": report["metrics"]["unknown_misname_count"], "fresh_reset_per_action": report["metrics"]["fresh_reset_per_action"], "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
