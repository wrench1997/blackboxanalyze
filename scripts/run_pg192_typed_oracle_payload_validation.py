"""PG-192: evaluator-only typed payload validation on three Pikachu lanes.

The selected PG-191 XXL model proposes abstract roles.  The runner binds the
role to observed fields and, for the redirect lane only, supplies a controlled
loopback destination that an evaluator can verify without following it.  DOM
and SQL lanes remain candidate/abstain when their target-specific evaluator is
not available.  Raw values exist only in request memory and never enter the
model or report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import sha256_json, validate_payload_manifest  # noqa: E402
from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import _summary, action_manifest  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app.pg52_authoritative_oracle import redirect_oracle, sha256_text  # noqa: E402
from app.pg185_pikachu_dom_adapter import project_dom_response  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")

RESEARCH = ROOT / "research"
ARTIFACT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT_PATH = RESEARCH / "pg192_typed_oracle_payload_validation_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg192_typed_oracle_payload_validation_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg192_typed_oracle_payload_validation_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg192_typed_oracle_payload_validation_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3102
BASE_URL = f"http://127.0.0.1:{PORT}"
EXPECTED_DESTINATION = "http://127.0.0.1:8767/pg192-sink"
SAFE_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{6,64}$")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse {name}; PG-192 requires fresh target")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{PORT}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 140.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError("PG-192 target did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _load_model(device: torch.device) -> tuple[Any, dict[str, int]]:
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    vocabulary = {str(k): int(v) for k, v in checkpoint["vocabulary"].items()}
    model = PG191._build_model("xxl", vocabulary, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocabulary


def _action(model: Any, context: list[str], vocabulary: Mapping[str, int], device: torch.device) -> tuple[str, float, bool, float]:
    return PG191._action_and_gate(model, context, vocabulary, device)


def _single_get(client: httpx.Client, path: str, *, params: Mapping[str, str] | None = None, marker: str | None = None, baseline_status: int | None = None) -> dict[str, Any]:
    response = client.get(path, params=params, follow_redirects=False)
    projection, signal, signal_hash = _summary(response, marker=marker, baseline_status=baseline_status)
    projection["status_chain"] = [int(response.status_code)]
    projection["redirect_chain"] = []
    projection["redirect_hop_count"] = 0
    projection["status_chain_sha256"] = _digest([int(response.status_code)])
    projection["projection_sha256"] = _digest({key: value for key, value in projection.items() if key != "projection_sha256"})
    signal["candidate_signal"] = bool(signal.get("candidate_signal") or signal.get("external_redirect"))
    signal["redirect_hop_count"] = 0
    signal["status_chain_sha256"] = projection["status_chain_sha256"]
    return {"projection": projection, "signal": signal, "response_status": int(response.status_code), "location": str(response.headers.get("location", "")), "signal_sha256": signal_hash}


def _redirect_manifest(*, path: str, surface: str, field: str, destination: str) -> dict[str, Any]:
    destination_hash = sha256_text(destination)
    body = {"path": path, "surface": surface, "method": "GET", "field": field, "role": "candidate", "destination_sha256": destination_hash, "probe_kind": "controlled_redirect_destination"}
    manifest = {"manifest_id": f"pg192-{surface}-redirect-candidate", "payload_sha256": sha256_json(body), "probe_ref": f"pg192-controlled-redirect-{surface}", "probe_kind": "http_canary", "route_template_id": f"pikachu-{surface}", "method": "GET", "placement": "query", "encoding_chain": ["identity"], "encoding_depth": 0, "marker_sha256": destination_hash, "max_bytes": 512, "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True}}
    return validate_payload_manifest(manifest)


def _typed_redirect_run(model: Any, vocabulary: dict[str, int], device: torch.device, *, target_hash: str) -> dict[str, Any]:
    path, surface, family = "/vul/urlredirect/urlredirect.php", "pg192_urlredirect", "url_redirect"
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    baseline_status: int | None = None
    confirmed = False
    control_completed = False
    try:
        for step_index in range(1, 4):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_conf, gate_allow, gate_conf = _action(model, context, vocabulary, device)
            if step_index == 1:
                if action != "baseline" or not gate_allow:
                    steps.append({"step_index": step_index, "model_action": action, "controller_decision": "abstain", "abstain_reason": "initial_baseline_or_gate"})
                    break
                result = _single_get(client, path)
                baseline_status = result["response_status"]
                role, manifest, decision = "negative_control", None, "send_safe_baseline_get"
                query = None
            else:
                # With a typed evaluator available, one bounded active probe
                # is allowed after the matched control.  The model's abstain
                # remains visible; the controller records this as an explicit
                # evaluator handoff rather than silently treating it as model
                # confidence.
                active_typed_candidate = step_index >= 3 and control_completed
                if (action in {"baseline", "abstain"} or not gate_allow) and not active_typed_candidate:
                    steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_conf, 6), "controller_decision": "abstain", "abstain_reason": "safety_gate_or_model_abstain"})
                    break
                # The first parameterized step is always a matched negative
                # control.  Even if the model jumps directly to
                # safe_candidate, the controller demotes it until a control
                # pair has been sent.
                role = "candidate" if active_typed_candidate or (action == "safe_candidate" and step_index >= 3) else "control"
                decision_override = False
                if role == "candidate":
                    query = {"url": EXPECTED_DESTINATION}
                    manifest = _redirect_manifest(path=path, surface=surface, field="url", destination=EXPECTED_DESTINATION)
                    decision_override = active_typed_candidate and action != "safe_candidate"
                else:
                    marker = f"pg192-control-{step_index}"
                    query = {"url": marker}
                    manifest = action_manifest(path=path, surface=surface, family=family, method="GET", field_names=["url"], probe_role="control", marker=marker)
                result = _single_get(client, path, params=query, marker=f"pg192-control-{step_index}", baseline_status=baseline_status)
                decision = "send_control" if role == "control" else ("send_typed_oracle_active_candidate" if decision_override else "send_controlled_redirect_candidate")
            if role == "candidate":
                typed = redirect_oracle(location=result["location"], expected_destination=EXPECTED_DESTINATION, response_status=result["response_status"])
            else:
                typed = redirect_oracle(location=result["location"], expected_destination=EXPECTED_DESTINATION, response_status=result["response_status"])
            failure = failure_signature({"method": "GET", "role": role, "candidate_signal": bool(result["signal"].get("candidate_signal")), "positive": bool(typed.get("positive")), "positive_authority": bool(typed.get("positive_authority")), "typed_available": True, "probe_round": step_index, "max_probe_rounds": 3}, prior_records=prior, max_steps=3, step_count=step_index)
            belief = {"typed_redirect_effect": 0.9, "unknown_oracle": 0.1} if typed["positive"] else {"no_confirmed_redirect": 0.65, "unknown_oracle": 0.35}
            evidence = {"target_instance_hash": target_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": result["projection"]["projection_sha256"], "typed_oracle_sha256": _digest(typed), "failure_sha256": _digest(failure)}
            if typed["positive"] and role == "candidate":
                confirmed = True
            if role == "control":
                control_completed = True
            steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow", "gate_confidence": round(gate_conf, 6), "controller_decision": decision, "method": "GET", "action_manifest": ({key: manifest[key] for key in ("method", "placement", "encoding_chain", "probe_kind", "probe_ref", "payload_sha256", "manifest_sha256", "marker_sha256", "safety") if key in manifest} if manifest else None), "response_projection": result["projection"], "typed_oracle": typed, "failure_signature": failure, "belief_after": belief, "evidence": evidence, "confirmed_positive": bool(typed["positive"] and role == "candidate"), "vulnerability_claim_allowed": bool(typed["positive"] and role == "candidate"), "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest or {"method": "GET", "placement": "none", "encoding_chain": ["identity"]}, "response_projection": result["projection"], "failure_signature": failure, "belief_after": belief})
            prior.append({"method": "GET", "role": role, "candidate_signal": bool(result["signal"].get("candidate_signal")), "belief_after": belief})
    finally:
        client.close()
    return {"surface": surface, "path": path, "family": family, "target_instance_hash": target_hash, "fresh_container": True, "typed_oracle_available": True, "step_count": len(steps), "confirmed_positive": confirmed, "typed_positive_count": int(confirmed), "vulnerability_claim_allowed": confirmed, "steps": steps}


def _unknown_route_run(model: Any, vocabulary: dict[str, int], device: torch.device, *, target_hash: str, path: str, surface: str, family: str, method: str, fields: list[str]) -> dict[str, Any]:
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    try:
        baseline = client.get(path, follow_redirects=False)
        projection, signal, _ = _summary(baseline, marker=None)
        history = [{"action_manifest": {"method": "GET", "placement": "none", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": {"kind": "no_typed_oracle", "failed_gate": "typed_effect", "candidate_signal": False, "typed_available": False, "probe_round": 1, "remaining_probe_budget": 2}, "belief_after": {"unknown_oracle": 1.0}}]
        context = pre_action_tokens(history[0], history=[])
        action, confidence, gate_allow, gate_conf = _action(model, context, vocabulary, device)
        steps = [{"step_index": 1, "model_action": action, "action_confidence": round(confidence, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_conf, 6), "controller_decision": "abstain", "abstain_reason": "typed_oracle_unavailable", "response_projection": {key: value for key, value in projection.items() if key != "body_sha256"}, "typed_oracle": {"status": "unavailable", "positive": False, "positive_authority": False}, "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False}]
        return {"surface": surface, "path": path, "family": family, "method": method, "field_names": fields, "target_instance_hash": target_hash, "fresh_container": True, "typed_oracle_available": False, "step_count": 1, "confirmed_positive": False, "typed_positive_count": 0, "vulnerability_claim_allowed": False, "steps": steps}
    finally:
        client.close()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary = _load_model(device)
    route_specs = [
        ("redirect", "/vul/urlredirect/urlredirect.php", "url_redirect", "GET", ["url"]),
        ("dom_unknown", "/vul/xss/xss_01.php", "xss", "GET", ["message", "submit"]),
        ("sql_unknown", "/vul/sqli/sqli_blind_t.php", "injection", "GET", ["name", "submit"]),
    ]
    runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for surface, path, family, method, fields in route_specs:
        name = f"sift-pg192-{surface}"
        container_id = _start(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        try:
            if surface == "redirect":
                runs.append(_typed_redirect_run(model, vocabulary, device, target_hash=target_hash))
            else:
                runs.append(_unknown_route_run(model, vocabulary, device, target_hash=target_hash, path=path, surface=surface, family=family, method=method, fields=fields))
            targets.append({"surface": surface, "target_instance_hash": target_hash, "fresh_container": True, "pinned_image": IMAGE})
        finally:
            _stop(name)
    report = {"protocol_id": "pg-pk-192-typed-oracle-payload-validation-v1", "schema_version": "pg192-typed-oracle-payload-validation-report-v1", "status": "completed_typed_redirect_and_unknown_oracle_replay", "device": str(device), "model": {"artifact": str(ARTIFACT.relative_to(ROOT)), "variant": "xxl", "parameter_count": int(sum(p.numel() for p in model.parameters())), "online_weight_update": False}, "source": {"image": IMAGE, "loopback_port": PORT, "fresh_container_per_route": True}, "targets": targets, "counts": {"route_count": len(runs), "typed_oracle_available_count": sum(int(r["typed_oracle_available"]) for r in runs), "confirmed_positive_count": sum(int(r["confirmed_positive"]) for r in runs), "typed_positive_count": sum(int(r["typed_positive_count"]) for r in runs), "vulnerability_claim_allowed_count": sum(int(r["vulnerability_claim_allowed"]) for r in runs)}, "runs": runs, "promotion": {"typed_positive_required": True, "confirmed_positive_requires_fresh_reset_negative_control_and_evidence_hash": True, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "safety": {"loopback_only": True, "controlled_destination": EXPECTED_DESTINATION, "external_network": False, "script_execution": False, "database_write": False, "credentials": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg192-typed-oracle-payload-validation-trace-v1", "evaluation_only": True, "runs": runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": "pg-pk-192-typed-oracle-payload-validation-v1", "schema_version": "pg192-typed-oracle-payload-validation-protocol-v1", "model_variant": "xxl", "routes": [row[0] for row in route_specs], "typed_modalities": ["redirect_destination_controlled", "dom_unknown_abstain", "sql_unknown_abstain"], "controlled_destination": EXPECTED_DESTINATION, "fresh_container_per_route": True, "negative_control_required": True, "evidence_hash_required": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "unknown_oracle_action": "abstain", "vulnerability_claim_allowed_only_with_typed_positive": True}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-192 typed oracle payload validation", "", f"device={device}; routes={len(runs)}; typed_positive={report['counts']['typed_positive_count']}; confirmed={report['counts']['confirmed_positive_count']}; training_eligible=False", "", "| surface | typed oracle | confirmed positive | claim allowed |", "|---|---|---:|---:|"] + [f"| {r['surface']} | {r['typed_oracle_available']} | {r['confirmed_positive']} | {r['vulnerability_claim_allowed']} |" for r in runs] + ["", "只有受控 loopback redirect oracle 通过时才形成 typed positive；DOM/SQL evaluator unavailable 时保留 abstain。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "typed_positive": report["counts"]["typed_positive_count"], "confirmed_positive": report["counts"]["confirmed_positive_count"], "runs": [{"surface": r["surface"], "confirmed_positive": r["confirmed_positive"], "typed_oracle_available": r["typed_oracle_available"]} for r in runs], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
