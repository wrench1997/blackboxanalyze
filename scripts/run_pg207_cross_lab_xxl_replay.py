"""PG-207: cross-lab, three-seed replay with the selected XXL adapter.

WebGoat and DVWA are used only as local, pinned transport surfaces.  The
executor sends a bounded empty/submit-shaped request and keeps response shape
and hashes only; no credentials or vulnerability payloads are used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG205 = _load_script("run_pg205_field_token_training_and_replay.py")

from app.pg195_request_surface_adapter import build_surface_values, project_surface_response  # noqa: E402
from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg196_failure_action_decoder import encode_features  # noqa: E402
from app.pg201_multitask_decoder import FAILURE_NAMES  # noqa: E402
from app.pg203_token_aware_decoder import token_features_for_row  # noqa: E402
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet  # noqa: E402
from app.pg205_field_token_decoder import FieldTokenGroundingDecoder, predict_field_aware  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT = ROOT / "artifacts" / "pg206-body-capacity-v1" / "xxl_field_token_adapter.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg207-cross-lab-xxl-v1"
REPORT_PATH = RESEARCH / "pg207_cross_lab_xxl_replay_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg207_cross_lab_xxl_replay_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg207_cross_lab_xxl_replay_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg207_cross_lab_xxl_replay_report_v1.md"
SEEDS = (20701, 20702, 20703)

LABS: tuple[dict[str, Any], ...] = (
    {"lab": "webgoat", "image": "webgoat/webgoat@sha256:3101bd9e7bcfe122d7ef91e690ef3720de36cc4e86b3d06763a1ddf2e2751a4b", "container_port": 8080, "path": "/WebGoat/login", "ready_timeout": 180, "base_port": 31500},
    {"lab": "dvwa", "image": "vulnerables/web-dvwa@sha256:dae203fe11646a86937bf04db0079adef295f426da68a92b40e3b181f337daa7", "container_port": 80, "path": "/login.php", "ready_timeout": 100, "base_port": 31510},
)

FAILURE_KIND = {"no_effect": "no_effect", "status_changed": "status_changed", "redirect_shape": "redirect_chain", "validation_shape": "post_validation", "server_shape": "status_changed", "oracle_unknown": "post_validation"}
ENCODING_KIND = {"identity": "http_canary", "dom_markup": "inert_dom_markup", "encoded_dom": "encoded_dom_markup", "abstract_sql": "sql_channel_class"}


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


def _start(lab: Mapping[str, Any], seed: int) -> tuple[str, int, str]:
    name = f"sift-pg207-{lab['lab']}-{seed}"
    if _exists(name):
        raise RuntimeError(f"PG-207 refuses to reuse target {name}")
    port = int(lab["base_port"]) + (int(seed) - SEEDS[0])
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{port}:{int(lab['container_port'])}", str(lab["image"]))
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + float(lab["ready_timeout"])
    while time.monotonic() < deadline:
        try:
            response = httpx.get(base_url + str(lab["path"]), timeout=3.0, follow_redirects=False)
            if response.status_code < 500:
                return name, port, _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-207 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _projection(client: httpx.Client, *, path: str, method: str, values: Mapping[str, str] | None, marker: str, layout: str = "inline_html") -> dict[str, Any]:
    if str(method).upper() == "GET":
        response = client.get(path, params=dict(values or {}), follow_redirects=False)
    else:
        response = client.post(path, data=dict(values or {}), follow_redirects=False)
    result = project_surface_response(response, marker=marker, layout_variant=layout, run_browser=False)
    result.pop("body_text", None)
    result.pop("signal", None)
    return dict(result["response_projection"])


def _load_model(device: torch.device, vocabulary: dict[str, int]) -> FieldTokenGroundingDecoder:
    risk_decoder, _ = PG205.PG197._load_decoder(device, vocabulary)
    model = FieldTokenGroundingDecoder(risk_decoder.frozen_base, hidden_dim=96).to(device)
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.frozen_base.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def _model_decision(model: FieldTokenGroundingDecoder, vocabulary: Mapping[str, int], device: torch.device, *, packet: Mapping[str, Any], route: Mapping[str, Any], projection: Mapping[str, Any]) -> dict[str, Any]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    failure_name = str(packet["failure_name"])
    failure_kind = FAILURE_KIND.get(failure_name, "no_effect")
    features = encode_features(method=str(route["method"]), redirect_hops=0, status_class=str(projection.get("status_class", "2xx")), candidate_signal=0, typed_available=0, negative_control=1, budget_remaining=1, failure_kind=failure_kind)
    legacy = token_features_for_row({"encoding_label": 0, "failure_label": FAILURE_NAMES.index(failure_name) if failure_name in FAILURE_NAMES else 0})
    decision = predict_field_aware(model, ids=ids, mask=mask, features=features, token_features=legacy, field_tokens=list(packet["field_tokens"]))
    # No lab-specific typed oracle is registered for these login surfaces.
    decision["effective_action"] = "abstain"
    decision["abstain_reason"] = "cross_lab_oracle_unknown"
    decision["features"] = {"method": route["method"], "status_class": projection.get("status_class", "2xx"), "typed_available": False, "failure_kind": failure_kind, "field_token_dim": len(packet["field_tokens"])}
    return decision


def _fault_cases(candidate: Mapping[str, Any], route: Mapping[str, Any], projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    packet = build_field_token_packet(candidate, route=route, response_projection=projection, typed_available=False, redirect_hops=0)
    cases: list[dict[str, Any]] = [
        {"case": "missing_field_tokens", "packet": None},
        {"case": "field_hash_mismatch", "packet": {**packet, "binding_sha256": "0" * 64}},
    ]
    wrong = dict(packet)
    wrong["field_tokens"] = list(packet["field_tokens"])
    wrong["field_tokens"][0] = 1.0 - float(wrong["field_tokens"][0])
    cases.append({"case": "field_token_mismatch", "packet": wrong})
    # Flip to a *different* transport bucket instead of hard-coding 3xx.
    # Login pages commonly redirect, so setting 3xx was a no-op and let a
    # stale packet through the structural token gate in the first run.
    stale = dict(projection)
    current_status = str(stale.get("status_class", ""))
    stale["status_class"] = "2xx" if current_status != "2xx" else "4xx_5xx"
    cases.append({"case": "stale_projection", "packet": packet, "projection": stale})
    out: list[dict[str, Any]] = []
    for case in cases:
        checked = validate_field_token_packet(case["packet"], candidate=candidate, route=route, response_projection=case.get("projection", projection), typed_available=False, redirect_hops=0)
        out.append({"case": case["case"], "validation": checked, "network_allowed": bool(checked.get("network_allowed"))})
    return out


def main() -> int:
    _train, _replay, _holdout, vocabulary, device = PG205._load_vocabulary()
    model = _load_model(device, vocabulary)
    route_runs: list[dict[str, Any]] = []
    fault_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        for lab in LABS:
            name = ""
            try:
                name, port, container_id = _start(lab, seed)
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                targets.append({"seed": seed, "lab": lab["lab"], "target_instance_hash": target_hash, "image": lab["image"], "fresh_container": True, "loopback_port": port})
                route = {"surface": f"pg207-{lab['lab']}-login", "path": lab["path"], "method": "GET", "fields": ["submit"], "family": "logic", "typed_available": False, "redirect_chain": False}
                post_route = {**route, "method": "POST"}
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    for current in (route, post_route):
                        marker = f"pg207-{lab['lab']}-{seed}-{current['method'].lower()}"
                        control_marker = f"pg207-control-{lab['lab']}-{seed}-{current['method'].lower()}"
                        baseline = _projection(client, path=current["path"], method=current["method"], values=None, marker=f"pg207-base-{seed}")
                        control_values = build_surface_values(field_names=current["fields"], probe_role="control", marker=control_marker)
                        control = _projection(client, path=current["path"], method=current["method"], values=control_values, marker=control_marker)
                        candidates = generate_grounded_candidates(family="logic", target=f"http://127.0.0.1:{port}", path=current["path"], method=current["method"], fields=current["fields"], marker=marker)
                        candidate = next(row for row in candidates if row["payload"]["probe_kind"] == "http_canary")
                        packet = build_field_token_packet(candidate, route=current, response_projection=control, typed_available=False, redirect_hops=0)
                        validation = validate_field_token_packet(packet, candidate=candidate, route=current, response_projection=control, typed_available=False, redirect_hops=0)
                        decision = _model_decision(model, vocabulary, device, packet=packet, route=current, projection=control) if validation["valid"] else {"action": "abstain", "effective_action": "abstain", "reason": validation["reason"]}
                        route_runs.append({"seed": seed, "lab": lab["lab"], "surface": current["surface"], "path": current["path"], "method": current["method"], "fields": list(current["fields"]), "target_instance_hash": target_hash, "baseline_projection": baseline, "control_projection": control, "model_decision": decision, "token_validation": validation, "candidate_generated": True, "candidate_sent": False, "candidate_summary": candidate_summary(candidate), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False})
                        fault_runs.extend(_fault_cases(candidate, current, control))
                finally:
                    client.close()
            finally:
                if name:
                    _stop(name)
    report = {
        "protocol_id": "pg-pk-207-cross-lab-xxl-replay-v1",
        "schema_version": "pg207-cross-lab-xxl-replay-report-v1",
        "status": "completed_cross_lab_three_seed_xxl_replay",
        "device": str(device),
        "model": {"variant": "xxl_field_token_adapter", "base_parameter_count": 101487169, "artifact": str(ARTIFACT.relative_to(ROOT)), "online_weight_update": False},
        "sources": [{"lab": lab["lab"], "image": lab["image"], "repository": "public_vulnerability_lab_local_image", "typed_oracle": False} for lab in LABS],
        "targets": targets,
        "route_runs": route_runs,
        "fault_runs": fault_runs,
        "counts": {"source_count": len(LABS), "seed_count": len(SEEDS), "fresh_container_count": len(targets), "route_replay_count": len(route_runs), "get_count": sum(int(row["method"] == "GET") for row in route_runs), "post_count": sum(int(row["method"] == "POST") for row in route_runs), "unknown_oracle_abstain_count": sum(int(row["model_decision"]["effective_action"] == "abstain") for row in route_runs), "candidate_send_count": sum(int(row["candidate_sent"]) for row in route_runs), "field_fault_count": len(fault_runs), "network_allowed_on_fault_count": sum(int(row["network_allowed"]) for row in fault_runs), "false_positive_count": 0},
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "pinned_images": True, "fresh_container_per_seed_and_lab": True, "get_post_balanced": True, "credentials_used": False, "typed_oracle_available": False, "unknown_oracle_abstain_required": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "script_execution": False, "database_write": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg207-cross-lab-xxl-replay-protocol-v1", "sources": [lab["lab"] for lab in LABS], "seeds": list(SEEDS), "surface": "GET/POST login routes with submit-only bounded canary", "unknown_oracle_action": "abstain", "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg207-cross-lab-xxl-replay-trace-v1", "evaluation_only": True, "targets": targets, "route_runs": route_runs, "fault_runs": fault_runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(report["counts"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-207 cross-lab XXL replay", "", f"device={device}; sources={len(LABS)}; seeds={len(SEEDS)}; fresh containers={len(targets)}", f"GET={report['counts']['get_count']}; POST={report['counts']['post_count']}; unknown abstain={report['counts']['unknown_oracle_abstain_count']}; candidate sends={report['counts']['candidate_send_count']}", f"field faults={report['counts']['field_fault_count']}; network on fault={report['counts']['network_allowed_on_fault_count']}", "", "WebGoat/DVWA surfaces have no typed vulnerability oracle in this track; all candidate decisions remain abstain and evaluation-only.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "sources": len(LABS), "seeds": len(SEEDS), "containers": len(targets), "routes": len(route_runs), "get": report["counts"]["get_count"], "post": report["counts"]["post_count"], "unknown_abstain": report["counts"]["unknown_oracle_abstain_count"], "candidate_sends": report["counts"]["candidate_send_count"], "field_faults": report["counts"]["field_fault_count"], "network_on_fault": report["counts"]["network_allowed_on_fault_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
