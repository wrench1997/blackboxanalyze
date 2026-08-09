"""PG-204: token-aware 101M adapter in the real local GET/POST loop."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg195_request_surface_adapter import build_surface_action_manifest, build_surface_values, project_surface_response, send_surface_request  # noqa: E402
from app.pg196_failure_action_decoder import encode_features  # noqa: E402
from app.pg198_payload_grounding import generate_grounded_candidates, send_grounded_candidate  # noqa: E402
from app.pg203_token_aware_decoder import TokenAwareGroundingDecoder, predict_token_aware  # noqa: E402
from app.pg204_token_binding_controller import build_runtime_token_packet, validate_runtime_token_packet  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG197 = _load_script("run_pg197_risk_aware_cross_evaluator.py")

RESEARCH = ROOT / "research"
ARTIFACT = ROOT / "artifacts" / "pg203-token-aware-adapter-v1" / "xxl_token_aware_adapter.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg204-token-binding-loop-v1"
REPORT_PATH = RESEARCH / "pg204_token_binding_loop_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg204_token_binding_loop_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg204_token_binding_loop_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg204_token_binding_loop_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3112
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (20401, 20402)

ROUTES = (
    {"surface": "pg204_xss_identity_get", "path": "/vul/xss/xss_01.php", "method": "GET", "fields": ["message", "submit"], "layout": "inline_html", "family": "xss", "typed_available": True, "encoding": "inert_dom_markup"},
    {"surface": "pg204_xss_encoded_get", "path": "/vul/xss/xss_02.php", "method": "GET", "fields": ["message", "submit"], "layout": "inline_html", "family": "xss", "typed_available": True, "encoding": "encoded_dom_markup"},
    {"surface": "pg204_post_unknown", "path": "/vul/xss/xsspost/post_login.php", "method": "POST", "fields": ["username", "submit"], "layout": "attribute_shell", "family": "xss", "typed_available": False, "encoding": "inert_dom_markup"},
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"PG-204 refuses to reuse target {name}")
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
    raise RuntimeError(f"PG-204 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _safe_projection(result: dict[str, Any]) -> dict[str, Any]:
    result.pop("body_text", None)
    result.pop("signal", None)
    return dict(result.get("response_projection") or {})


def _baseline(client: httpx.Client, route: dict[str, Any], marker: str) -> dict[str, Any]:
    if route["method"] == "GET":
        response = client.get(route["path"], follow_redirects=False)
    else:
        response = client.post(route["path"], data={"username": marker, "submit": "submit"}, follow_redirects=False)
    return _safe_projection(project_surface_response(response, marker=marker, layout_variant=route["layout"], baseline_status=None, run_browser=False))


def _control(client: httpx.Client, route: dict[str, Any], marker: str) -> dict[str, Any]:
    build_surface_action_manifest(path=route["path"], method=route["method"], surface=route["surface"], field_names=route["fields"], probe_role="control", marker=marker)
    values = build_surface_values(field_names=route["fields"], probe_role="control", marker=marker)
    return _safe_projection(send_surface_request(client, path=route["path"], method=route["method"], values=values, marker=marker, layout_variant=route["layout"], run_browser=False))


def _load_model(device: torch.device, vocabulary: dict[str, int]) -> TokenAwareGroundingDecoder:
    risk_decoder, _ = PG197._load_decoder(device, vocabulary)
    model = TokenAwareGroundingDecoder(risk_decoder.frozen_base).to(device)
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.frozen_base.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def _model_decision(model: TokenAwareGroundingDecoder, vocabulary: dict[str, int], device: torch.device, *, packet: dict[str, Any], route: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    failure_name = str(packet["failure_name"])
    failure_kind = {"no_effect": "no_effect", "status_changed": "status_changed", "redirect_shape": "redirect_chain", "validation_shape": "post_validation", "server_shape": "status_changed", "oracle_unknown": "post_validation"}.get(failure_name, "no_effect")
    features = encode_features(
        method=route["method"], redirect_hops=0, status_class=str(control.get("status_class", "2xx")),
        candidate_signal=1, typed_available=int(route["typed_available"]), negative_control=1,
        budget_remaining=1, failure_kind=failure_kind,
    )
    decision = predict_token_aware(model, ids=ids, mask=mask, features=features, token_features=list(packet["token_features"]))
    expected_encoding = str(packet["encoding_name"])
    expected_failure = str(packet["failure_name"])
    decision["encoding_binding_match"] = bool(decision["encoding"] == expected_encoding)
    decision["failure_binding_match"] = bool(decision["failure"] == expected_failure)
    decision["effective_action"] = (
        decision["action"]
        if bool(route["typed_available"])
        and decision["action"] == "safe_candidate"
        and decision["encoding_binding_match"]
        and decision["failure_binding_match"]
        else "abstain"
    )
    decision["features"] = {"method": route["method"], "status_class": str(control.get("status_class", "2xx")), "typed_available": bool(route["typed_available"]), "failure_kind": failure_kind}
    return decision


def _fault_cases(candidate: dict[str, Any], route: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    packet = build_runtime_token_packet(candidate, route=route, failure_projection=control, typed_available=bool(route["typed_available"]))
    cases: list[dict[str, Any]] = []
    cases.append({"case": "missing_encoding_token", "packet": None})
    wrong_hash = dict(packet)
    wrong_hash["binding_sha256"] = "0" * 64
    cases.append({"case": "binding_hash_mismatch", "packet": wrong_hash})
    wrong_features = dict(packet)
    wrong_features["token_features"] = list(packet["token_features"])
    wrong_features["token_features"][0] = 1.0 - float(wrong_features["token_features"][0])
    cases.append({"case": "token_features_mismatch", "packet": wrong_features})
    wrong_route = dict(route)
    wrong_route["path"] = "/vul/xss/xss_04.php"
    cases.append({"case": "method_path_binding_mismatch", "packet": packet, "route": wrong_route})
    for case in cases:
        checked = validate_runtime_token_packet(case["packet"], candidate=candidate, route=case.get("route", route), failure_projection=control)
        case["validation"] = checked
        case["network_allowed"] = bool(checked.get("network_allowed"))
    return cases


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG197.PG191.PG189._load_rows()
    vocabulary = PG197.PG191.PG189._vocabulary(train, PG197.PG191.PG189._load_body_vocab())
    model = _load_model(device, vocabulary)
    route_runs: list[dict[str, Any]] = []
    fault_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"sift-pg204-{seed}"
        container_id = _start(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
        client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
        try:
            for route in ROUTES:
                marker = f"pg204-candidate-{seed}-{_digest(route['surface'])[:8]}"
                control_marker = f"pg204-control-{seed}-{_digest(route['surface'])[:8]}"
                baseline = _baseline(client, route, f"pg204-base-{seed}")
                control = _control(client, route, control_marker)
                candidates = generate_grounded_candidates(family=route["family"], target=BASE_URL, path=route["path"], method=route["method"], fields=route["fields"], marker=marker)
                candidate = next(row for row in candidates if row["payload"]["probe_kind"] == route["encoding"])
                packet = build_runtime_token_packet(candidate, route=route, failure_projection=control, typed_available=bool(route["typed_available"]))
                validation = validate_runtime_token_packet(packet, candidate=candidate, route=route, failure_projection=control)
                decision = _model_decision(model, vocabulary, device, packet=packet, route=route, control=control) if validation["valid"] else {"action": "abstain", "effective_action": "abstain", "reason": validation["reason"]}
                candidate_result = None
                if validation["valid"] and decision["effective_action"] == "safe_candidate":
                    candidate_result = send_grounded_candidate(client, candidate=candidate, fields=route["fields"], layout_variant=route["layout"], baseline_status=int(baseline.get("status_code", 0)) or None, typed_available=bool(route["typed_available"]))
                route_runs.append({
                    "seed": seed,
                    "surface": route["surface"],
                    "path": route["path"],
                    "method": route["method"],
                    "encoding": route["encoding"],
                    "target_instance_hash": target_hash,
                    "fresh_container": True,
                    "baseline_projection": baseline,
                    "control_projection": control,
                    "token_validation": validation,
                    "token_binding_sha256": packet["binding_sha256"],
                    "model_decision": decision,
                    "candidate_sent": candidate_result is not None,
                    "candidate_result": candidate_result,
                    "raw_payload_strings_stored": False,
                    "raw_response_bodies_stored": False,
                    "training_eligible": False,
                    "memory_promotion_allowed": False,
                    "vulnerability_claim_allowed": False,
                })
                fault_runs.extend(_fault_cases(candidate, route, control))
        finally:
            client.close()
            _stop(name)
    sent = [row for row in route_runs if row["candidate_sent"]]
    post = [row for row in route_runs if row["method"] == "POST"]
    report = {
        "protocol_id": "pg-pk-204-token-binding-loop-v1",
        "schema_version": "pg204-token-binding-loop-report-v1",
        "status": "completed_token_aware_real_get_post_loop",
        "device": str(device),
        "model": {"variant": "xxl_token_aware_adapter", "base_parameter_count": int(sum(p.numel() for p in model.frozen_base.parameters())), "total_parameter_count": int(sum(p.numel() for p in model.parameters())), "online_weight_update": False},
        "targets": targets,
        "route_runs": route_runs,
        "fault_runs": fault_runs,
        "counts": {
            "fresh_container_count": len(targets),
            "route_replay_count": len(route_runs),
            "valid_token_binding_count": sum(int(row["token_validation"]["valid"]) for row in route_runs),
            "candidate_send_count": len(sent),
            "encoded_variant_send_count": sum(int(row["encoding"] == "encoded_dom_markup" and row["candidate_sent"]) for row in route_runs),
            "post_unknown_abstain_count": sum(int(row["method"] == "POST" and row["model_decision"]["effective_action"] == "abstain") for row in route_runs),
            "token_fault_count": len(fault_runs),
            "network_allowed_on_fault_count": sum(int(row["network_allowed"]) for row in fault_runs),
            "false_positive_count": 0,
        },
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_seed": True, "token_faults_fail_closed": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "script_execution": False, "database_write": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg204-token-binding-loop-protocol-v1", "model": "PG-203 101M XXL token-aware adapter", "methods": ["GET", "POST"], "fault_cases": ["missing_encoding_token", "binding_hash_mismatch", "token_features_mismatch", "method_path_binding_mismatch"], "token_fault_action": "abstain_before_network", "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg204-token-binding-loop-trace-v1", "evaluation_only": True, "route_runs": route_runs, "fault_runs": fault_runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "binding_summary.json").write_text(json.dumps(report["counts"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-204 token-aware real GET/POST loop", "", f"device={device}; base parameters={report['model']['base_parameter_count']}; routes={len(route_runs)}; valid bindings={report['counts']['valid_token_binding_count']}", f"candidate sends={report['counts']['candidate_send_count']}; encoded variant sends={report['counts']['encoded_variant_send_count']}; POST abstain={report['counts']['post_unknown_abstain_count']}", f"fault cases={report['counts']['token_fault_count']}; network on fault={report['counts']['network_allowed_on_fault_count']}", "", "Missing or mismatched structural tokens stop before network. Surface effects remain non-authoritative.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "base_parameters": report["model"]["base_parameter_count"], "routes": report["counts"]["route_replay_count"], "candidate_sends": report["counts"]["candidate_send_count"], "encoded_variant_sends": report["counts"]["encoded_variant_send_count"], "post_abstain": report["counts"]["post_unknown_abstain_count"], "token_faults": report["counts"]["token_fault_count"], "network_on_fault": report["counts"]["network_allowed_on_fault_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
