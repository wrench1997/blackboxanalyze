"""PG-193: browser DOM surface oracle plus independent SQL AST adapter.

Pikachu's reflected XSS route is replayed in a real headless browser with
JavaScript disabled and all network requests aborted.  A marker appearing as
an element is a typed DOM-surface effect only.  The SQL lane uses the
independent in-repo v3 fixture and its read-only AST oracle; it is explicitly
not counted as a Pikachu backend result.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import sha256_json  # noqa: E402
from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg179b_iterative_probe import _summary  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app.pg185_pikachu_dom_adapter import build_dom_action_manifest, build_query  # noqa: E402
from app.pg193_browser_dom_oracle import run_browser_dom_oracle  # noqa: E402
from app.sql_differential_fixture_v3 import (  # noqa: E402
    SqlV3Collector,
    default_sql_v3_specs,
    make_sql_v3_fixture_server,
    sql_v3_source_sha256,
)


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
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3103
BASE_URL = f"http://127.0.0.1:{PORT}"
REPORT_PATH = RESEARCH / "pg193_dom_sql_typed_adapters_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg193_dom_sql_typed_adapters_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg193_dom_sql_typed_adapters_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg193_dom_sql_typed_adapters_report_v1.md"


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
        raise RuntimeError(f"refusing to reuse {name}; PG-193 requires fresh target")
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
    raise RuntimeError("PG-193 Pikachu container did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _load_model(device: torch.device) -> tuple[Any, dict[str, int]]:
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    model = PG191._build_model("xxl", vocabulary, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocabulary


def _action(model: Any, context: list[str], vocabulary: dict[str, int], device: torch.device) -> tuple[str, float, bool, float]:
    return PG191._action_and_gate(model, context, vocabulary, device)


def _request_projection(client: httpx.Client, path: str, *, params: dict[str, str] | None = None, marker: str | None = None, baseline_status: int | None = None) -> dict[str, Any]:
    response = client.get(path, params=params, follow_redirects=False)
    projection, signal, _ = _summary(response, marker=marker, baseline_status=baseline_status)
    projection["status_chain"] = [int(response.status_code)]
    projection["redirect_chain"] = []
    projection["redirect_hop_count"] = 0
    projection["status_chain_sha256"] = _digest([int(response.status_code)])
    projection["projection_sha256"] = _digest({key: value for key, value in projection.items() if key != "projection_sha256"})
    signal["candidate_signal"] = bool(signal.get("candidate_signal") or signal.get("external_redirect"))
    return {"projection": projection, "signal": signal, "body_text": response.text, "status": int(response.status_code)}


def _dom_oracle(*, body_text: str, marker: str) -> dict[str, Any]:
    browser = run_browser_dom_oracle(body_text, marker=marker)
    # Browser parsing is authoritative for DOM structure, but script execution
    # is deliberately impossible.  Therefore positive remains false.
    return {"oracle_id": "pg193-browser-dom-nojs-v1", "modality": browser["modality"], "typed_surface_effect": bool(browser["dom_change"] and browser["marker_hits"] > 0), "positive": False, "positive_authority": False, "confirmed_effect": "dom_structure" if browser["dom_change"] else "none", "signals": {key: browser[key] for key in ("marker_hits", "body_text_hits", "element_count", "script_tag_count", "browser_dom_observed", "dom_change", "script_execution", "network_request_count", "evidence_hash")}, "safety": {"external_network": False, "script_execution": False, "database_write": False, "navigation": False, "raw_body_stored": False}}


def _dom_run(model: Any, vocabulary: dict[str, int], device: torch.device, *, target_hash: str) -> dict[str, Any]:
    path, surface, family, fields = "/vul/xss/xss_reflected_get.php", "pg193_xss_reflected_get", "xss", ["message", "submit"]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    prior: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    baseline_status: int | None = None
    control_completed = False
    typed_effect = False
    try:
        for step_index in range(1, 4):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_conf, gate_allow, gate_conf = _action(model, context, vocabulary, device)
            active_candidate = step_index >= 3 and control_completed
            if step_index == 1:
                if action != "baseline" or not gate_allow:
                    steps.append({"step_index": step_index, "model_action": action, "controller_decision": "abstain", "abstain_reason": "initial_baseline_or_gate"})
                    break
                result = _request_projection(client, path)
                baseline_status = result["status"]
                role, manifest, decision = "negative_control", None, "send_safe_baseline_get"
            else:
                if (action in {"baseline", "abstain"} or not gate_allow) and not active_candidate:
                    steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_conf, 6), "controller_decision": "abstain", "abstain_reason": "safety_gate_or_model_abstain"})
                    break
                role = "candidate" if active_candidate or (action == "safe_candidate" and step_index >= 3) else "control"
                marker = f"pg193-{surface[-8:]}-{step_index}"
                manifest = build_dom_action_manifest(path=path, surface=surface, field_names=fields, probe_role=role, marker=marker)
                query, oracle_marker = build_query(field_names=fields, role=role, marker=marker)
                result = _request_projection(client, path, params=query, marker=oracle_marker, baseline_status=baseline_status)
                decision = "send_typed_dom_active_candidate" if role == "candidate" and active_candidate and action != "safe_candidate" else ("send_safe_candidate" if role == "candidate" else "send_matched_control")
            oracle = _dom_oracle(body_text=result["body_text"], marker=(f"pg193-{surface[-8:]}-{step_index}" if role == "candidate" else f"pg193-{surface[-8:]}-{step_index}"))
            failure = failure_signature({"method": "GET", "role": role, "candidate_signal": bool(result["signal"].get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": True, "probe_round": step_index, "max_probe_rounds": 3}, prior_records=prior, max_steps=3, step_count=step_index)
            belief = {"typed_dom_surface_effect": 0.7, "unknown_oracle": 0.3} if oracle["typed_surface_effect"] else {"no_dom_effect": 0.65, "unknown_oracle": 0.35}
            evidence = {"target_instance_hash": target_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": result["projection"]["projection_sha256"], "oracle_sha256": _digest(oracle), "failure_sha256": _digest(failure)}
            typed_effect = typed_effect or bool(oracle["typed_surface_effect"] and role == "candidate")
            steps.append({"step_index": step_index, "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow", "gate_confidence": round(gate_conf, 6), "controller_decision": decision, "method": "GET", "action_manifest": ({key: manifest[key] for key in ("method", "placement", "encoding_chain", "probe_kind", "probe_ref", "payload_sha256", "manifest_sha256", "marker_sha256", "safety") if key in manifest} if manifest else None), "response_projection": result["projection"], "typed_oracle": oracle, "typed_surface_effect": bool(oracle["typed_surface_effect"]), "failure_signature": failure, "belief_after": belief, "evidence": evidence, "confirmed_positive": False, "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": manifest or {"method": "GET", "placement": "none", "encoding_chain": ["identity"]}, "response_projection": result["projection"], "failure_signature": failure, "belief_after": belief})
            prior.append({"method": "GET", "role": role, "candidate_signal": bool(result["signal"].get("candidate_signal")), "belief_after": belief})
            if role == "control":
                control_completed = True
    finally:
        client.close()
    return {"surface": surface, "path": path, "family": family, "target_instance_hash": target_hash, "fresh_container": True, "typed_oracle_available": True, "typed_surface_effect": typed_effect, "confirmed_positive": False, "vulnerability_claim_allowed": False, "steps": steps}


def _sql_collect(collector: SqlV3Collector, spec: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(collector.collect(spec))


def _sql_run(model: Any, vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    port, target, variant = 8810, "http://127.0.0.1:8810", "beta"
    server = make_sql_v3_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source_hash = sql_v3_source_sha256()
    collector = SqlV3Collector(base_url=target, target_instance_id="pg193-sql-beta", source_hash=source_hash)
    specs = default_sql_v3_specs(dataset_id="pg193-sql", target=target, marker="pg193-sql-marker")
    by_lab = {str(spec["lab_id"]): spec for spec in specs}
    chosen = [by_lab["baseline-control"], by_lab["literal-only-plain"], by_lab["branch_check-plain"]]
    steps: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(chosen, start=1):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            action, action_conf, gate_allow, gate_conf = _action(model, context, vocabulary, device)
            record = _sql_collect(collector, spec)
            oracle = dict(record["oracle_projection"])
            role = "negative_control" if index == 1 else ("control" if index == 2 else "candidate")
            failure = failure_signature({"method": "GET", "role": role, "candidate_signal": bool(oracle.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": True, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
            projection = dict(record["response_projection"])
            belief = {"typed_sql_differential": 0.9, "unknown_oracle": 0.1} if oracle.get("interpreter_boundary") else {"no_typed_boundary": 0.65, "unknown_oracle": 0.35}
            evidence = dict(record["evidence"])
            steps.append({"step_index": index, "lab_id": spec["lab_id"], "mode": spec.get("decoded_mode"), "model_action": action, "action_confidence": round(action_conf, 6), "safety_gate": "allow" if gate_allow else "abstain", "gate_confidence": round(gate_conf, 6), "controller_decision": "send_sql_typed_fixture_probe", "typed_oracle": oracle, "response_projection": projection, "failure_signature": failure, "belief_after": belief, "evidence_hash": evidence.get("evidence_hash"), "raw_probe_stored": False, "raw_response_stored": False, "confirmed_positive": bool(oracle.get("interpreter_boundary") and index == 3), "vulnerability_claim_allowed": False})
            history.append({"action_manifest": {"method": "GET", "placement": "query", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": belief})
        return {"surface": "pg193_sql_v3_fixture", "target": target, "variant": variant, "fresh_target": True, "typed_oracle_available": True, "source_hash": source_hash, "steps": steps, "typed_positive_count": sum(int(step["confirmed_positive"]) for step in steps), "vulnerability_claim_allowed": False, "training_eligible": False}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary = _load_model(device)
    dom_name = "sift-pg193-dom"
    dom_container = _start(dom_name)
    dom_target_hash = hashlib.sha256(dom_container.encode("utf-8")).hexdigest()
    try:
        dom_run = _dom_run(model, vocabulary, device, target_hash=dom_target_hash)
    finally:
        _stop(dom_name)
    sql_run = _sql_run(model, vocabulary, device)
    report = {"protocol_id": "pg-pk-193-dom-sql-typed-adapters-v1", "schema_version": "pg193-dom-sql-typed-adapters-report-v1", "status": "completed_browser_dom_and_sql_ast_adapter_replay", "device": str(device), "model": {"artifact": str(ARTIFACT.relative_to(ROOT)), "variant": "xxl", "parameter_count": int(sum(p.numel() for p in model.parameters())), "online_weight_update": False}, "runs": [dom_run, sql_run], "counts": {"dom_typed_surface_effect": int(dom_run["typed_surface_effect"]), "dom_confirmed_positive": 0, "sql_typed_positive": sql_run["typed_positive_count"], "vulnerability_claim_allowed": 0}, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "dom_structure_is_not_xss_positive": True, "sql_fixture_is_not_pikachu_backend": True}, "safety": {"loopback_only": True, "fresh_pikachu_container": True, "browser_javascript_enabled": False, "browser_network_aborted": True, "script_execution": False, "external_network": False, "database_write": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg193-dom-sql-typed-adapters-trace-v1", "evaluation_only": True, "runs": [dom_run, sql_run], "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": "pg-pk-193-dom-sql-typed-adapters-v1", "schema_version": "pg193-dom-sql-typed-adapters-protocol-v1", "model_variant": "xxl", "pikachu_dom_route": "/vul/xss/xss_reflected_get.php", "sql_adapter_target": "independent loopback sql_v3 fixture; not Pikachu backend", "dom_oracle": "browser_dom_nojs_v1", "sql_oracle": "synthetic_sql_ast_differential_v1", "fresh_reset_required": True, "matched_negative_required": True, "evidence_hash_required": True, "dom_script_execution_forbidden": True, "sql_database_execution_forbidden": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-193 DOM/SQL typed adapters", "", f"device={device}; dom_effect={dom_run['typed_surface_effect']}; sql_typed_positive={sql_run['typed_positive_count']}; vulnerability_claim=0", "", "| lane | evaluator | typed effect | confirmed vulnerability |", "|---|---|---:|---:|"] + [f"| Pikachu DOM | browser no-JS | {dom_run['typed_surface_effect']} | false |", f"| SQL v3 fixture | AST differential | {sql_run['typed_positive_count']} | false |", "", "DOM effect is not XSS positive; SQL fixture result is not a Pikachu backend claim.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "dom_typed_surface_effect": dom_run["typed_surface_effect"], "sql_typed_positive": sql_run["typed_positive_count"], "vulnerability_claim_allowed": 0, "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
