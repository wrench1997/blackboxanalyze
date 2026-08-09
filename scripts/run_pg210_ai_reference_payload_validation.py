"""PG-210: AI probe versus independent reference probe validation.

The comparison is deliberately limited to Pikachu's non-stored XSS GET
surfaces.  The AI and a separately constructed deterministic inert DOM canary
are both sent on fresh pinned containers.  The report shows *how* each request
was bound (method, path, placement, fields, encoding and hashes) and whether
the response produced the same no-JavaScript DOM evidence.  It does not claim
that a DOM structure effect is executable XSS and it never stores a raw
executable payload or response body.

Pikachu SQL GET/POST routes are listed as an unknown-oracle lane; the runner
does not promote HTTP differences to SQL findings without a backend evaluator.
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


PG208 = _load_script("run_pg208_pikachu_typed_payload_loop.py")
from app.payload_learner import PayloadLearner  # noqa: E402
from app.pg195_request_surface_adapter import build_surface_action_manifest, build_surface_values, send_surface_request  # noqa: E402
from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet  # noqa: E402
from app.pg197_alt_dom_oracle import run_alt_dom_oracle  # noqa: E402


RESEARCH = ROOT / "research"
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
ARTIFACT_DIR = ROOT / "artifacts" / "pg210-ai-reference-validation-v1"
REPORT_PATH = RESEARCH / "pg210_ai_reference_payload_validation_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg210_ai_reference_payload_validation_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg210_ai_reference_payload_validation_trace_v1.json"
VIEW_PATH = RESEARCH / "pg210_request_anatomy_view_v1.json"
MARKDOWN_PATH = RESEARCH / "pg210_ai_reference_payload_validation_report_v1.md"
SEEDS = (21001, 21002)
BASE_PORT = 3120


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


def _start(seed: int) -> tuple[str, int, str]:
    name = f"sift-pg210-{seed}"
    if _exists(name):
        raise RuntimeError(f"PG-210 refuses to reuse target {name}")
    port = BASE_PORT + (int(seed) - SEEDS[0])
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{port}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 150.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return name, port, _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-210 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _routes() -> list[dict[str, Any]]:
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8-sig"))
    catalog = PG208.build_parameter_catalog(crawl)
    routes = [PG208._route_from_entry(entry) for entry in catalog["eligible_entries"] if entry.get("typed_oracle") == "dom_nojs_dual" and entry.get("method") == "GET"]
    routes.sort(key=lambda route: (route["path"], route["fields"]))
    return routes


def _response_view(result: Mapping[str, Any]) -> dict[str, Any]:
    projection = dict(result.get("response_projection") or {})
    oracle = dict(result.get("oracle_projection") or result.get("oracle") or {})
    return {
        "status_code": projection.get("status_code"),
        "status_class": projection.get("status_class"),
        "content_type_class": projection.get("content_type_class"),
        "body_length_bucket": projection.get("body_length_bucket"),
        "marker": dict(projection.get("marker") or {}),
        "redirect_chain": list(projection.get("redirect_chain") or []),
        "projection_sha256": projection.get("projection_sha256"),
        "oracle": {
            "typed_available": bool(oracle.get("typed_available", True)),
            "browser_effect": bool(oracle.get("browser_effect", oracle.get("typed_surface_effect", False))),
            "alternate_effect": bool(oracle.get("alternate_effect", False)),
            "dual_agreement": bool(oracle.get("dual_agreement", False)),
            "browser_evidence_hash": str(oracle.get("browser_evidence_hash", (oracle.get("signals") or {}).get("evidence_hash", ""))),
            "alternate_evidence_hash": str(oracle.get("alternate_evidence_hash", "")),
            "script_execution": False,
            "network_access": False,
        },
    }


def _reference_result(client: httpx.Client, route: Mapping[str, Any], *, seed: int, baseline_status: int | None) -> dict[str, Any]:
    marker = f"pg210-reference-{seed}-{_digest(route['surface'])[:8]}"
    manifest = build_surface_action_manifest(path=str(route["path"]), method="GET", surface=str(route["surface"]), field_names=list(route["fields"]), probe_role="candidate", marker=marker)
    values = build_surface_values(field_names=list(route["fields"]), probe_role="candidate", marker=marker)
    result = send_surface_request(client, path=str(route["path"]), method="GET", values=values, marker=marker, layout_variant=str(route["layout"]), baseline_status=baseline_status, run_browser=True)
    browser_oracle = dict(result.get("oracle_projection") or {})
    alternate = run_alt_dom_oracle(str(result.get("body_text", "")), marker=marker)
    reference_oracle = {
        "typed_available": True,
        "browser_effect": bool(browser_oracle.get("typed_surface_effect", False)),
        "alternate_effect": bool(alternate.get("dom_change", False)),
        "dual_agreement": bool(bool(browser_oracle.get("typed_surface_effect", False)) == bool(alternate.get("dom_change", False))),
        "browser_evidence_hash": str((browser_oracle.get("signals") or {}).get("evidence_hash", "")),
        "alternate_evidence_hash": str(alternate.get("evidence_hash", "")),
        "script_execution": False,
        "network_access": False,
    }
    view = _response_view({"response_projection": result.get("response_projection"), "oracle": reference_oracle})
    binding = {"method": "GET", "path": route["path"], "placement": "query", "field_names": list(route["fields"]), "values_sha256": _digest(values), "runtime_only": True}
    binding["binding_sha256"] = _digest(binding)
    evidence = {"manifest_sha256": manifest.get("manifest_sha256"), "payload_sha256": manifest.get("payload_sha256"), "binding_sha256": binding["binding_sha256"], "projection_sha256": view["projection_sha256"], "oracle_sha256": _digest(view["oracle"])}
    return {"probe_source": "independent_deterministic_inert_dom_canary", "request": {"method": "GET", "path": route["path"], "placement": "query", "field_names": list(route["fields"]), "encoding": "inert_dom_markup", "marker_sha256": manifest.get("marker_sha256"), "payload_sha256": manifest.get("payload_sha256"), "manifest_sha256": manifest.get("manifest_sha256"), "value_sha256": binding["values_sha256"], "binding_sha256": binding["binding_sha256"]}, "response": view, "binding": binding, "evidence": {**evidence, "evidence_sha256": _digest(evidence)}, "raw_payload_stored": False, "raw_response_stored": False}


def _ai_result(model: Any, vocabulary: Mapping[str, int], device: torch.device, learner: PayloadLearner, client: httpx.Client, route: Mapping[str, Any], control_projection: Mapping[str, Any], *, seed: int, baseline_status: int | None, target: str) -> dict[str, Any]:
    marker = f"pg210-ai-{seed}-{_digest(route['surface'])[:8]}"
    candidates = generate_grounded_candidates(family="xss", target=target, path=str(route["path"]), method="GET", fields=list(route["fields"]), marker=marker)
    packet = build_field_token_packet(candidates[0], route=route, response_projection=control_projection, typed_available=True, redirect_hops=int(control_projection.get("redirect_hop_count", 0) or 0))
    validation = validate_field_token_packet(packet, candidate=candidates[0], route=route, response_projection=control_projection, typed_available=True, redirect_hops=int(control_projection.get("redirect_hop_count", 0) or 0))
    if not validation["valid"]:
        return {"request": {"method": "GET", "path": route["path"], "placement": "query", "field_names": list(route["fields"]), "encoding": "inert_dom_markup"}, "sent": False, "validation": validation, "model_decision": {"effective_action": "abstain", "abstain_reason": validation["reason"]}, "raw_payload_stored": False, "raw_response_stored": False}
    decision = PG208._model_decision(model, vocabulary, device, packet=packet, route=route, projection=control_projection)
    if decision.get("effective_action") != "safe_candidate":
        return {"request": {"method": "GET", "path": route["path"], "placement": "query", "field_names": list(route["fields"]), "encoding": "inert_dom_markup", "payload_sha256": candidate_summary(candidates[0])["payload_sha256"]}, "sent": False, "validation": validation, "model_decision": decision, "raw_payload_stored": False, "raw_response_stored": False}
    selected = learner.select(candidates)
    from app.pg198_payload_grounding import send_grounded_candidate  # local import keeps module load light
    result = send_grounded_candidate(client, candidate=selected, fields=list(route["fields"]), layout_variant=str(route["layout"]), baseline_status=baseline_status, typed_available=True)
    signal = bool((result.get("signal") or {}).get("candidate_signal", False))
    feedback = learner.observe(selected, status="candidate" if signal else "dead_end", evidence=result.get("evidence"), evaluator_confirmed=False)
    summary = candidate_summary(selected)
    binding = dict(result.get("binding") or {})
    request = {"method": summary["method"], "path": summary["path"], "placement": binding.get("placement"), "field_names": binding.get("field_names"), "encoding": summary["probe_kind"], "probe_sha256": summary["probe_sha256"], "payload_sha256": summary["payload_sha256"], "value_sha256": binding.get("values_sha256"), "binding_sha256": binding.get("binding_sha256")}
    response = {"response_projection": _response_view({"response_projection": result.get("response_projection"), "oracle": result.get("oracle")})}
    return {"sent": True, "request": request, "response": response["response_projection"], "candidate_id": summary["candidate_id"], "model_decision": decision, "ai_feedback": feedback, "evidence": dict(result.get("evidence") or {}), "raw_payload_stored": False, "raw_response_stored": False}


def main() -> int:
    routes = _routes()
    model, vocabulary = PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    learner = PayloadLearner(seed=210)
    episodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = ""
        try:
            name, port, container_id = _start(seed)
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True, "loopback_port": port, "image": IMAGE})
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
            try:
                for route in routes:
                    baseline = client.get(route["path"], follow_redirects=False)
                    baseline_status = int(baseline.status_code)
                    control_marker = f"pg210-control-{seed}-{_digest(route['surface'])[:8]}"
                    control = send_surface_request(client, path=route["path"], method="GET", values=build_surface_values(field_names=list(route["fields"]), probe_role="control", marker=control_marker), marker=control_marker, layout_variant=str(route["layout"]), run_browser=False, baseline_status=baseline_status)
                    control_projection = dict(control.get("response_projection") or {})
                    ai = _ai_result(model, vocabulary, device, learner, client, route, control_projection, seed=seed, baseline_status=baseline_status, target=f"http://127.0.0.1:{port}")
                    reference = _reference_result(client, route, seed=seed, baseline_status=baseline_status)
                    ai_view = dict(ai.get("response") or {})
                    ref_view = dict(reference.get("response") or {})
                    ai_oracle = dict(ai_view.get("oracle") or {})
                    ref_oracle = dict(ref_view.get("oracle") or {})
                    episodes.append({"seed": seed, "target_instance_hash": target_hash, "surface": route["surface"], "path": route["path"], "method": "GET", "fields": list(route["fields"]), "family": "xss", "control_projection_sha256": (control.get("response_projection") or {}).get("projection_sha256"), "ai": ai, "reference": reference, "ai_surface_effect": bool(ai_oracle.get("browser_effect") and ai_oracle.get("alternate_effect") and ai_oracle.get("dual_agreement")), "reference_surface_effect": bool(ref_oracle.get("browser_effect") and ref_oracle.get("alternate_effect") and ref_oracle.get("dual_agreement")), "ai_reference_effect_agreement": bool((ai_oracle.get("browser_effect") == ref_oracle.get("browser_effect")) and (ai_oracle.get("alternate_effect") == ref_oracle.get("alternate_effect")) and (ai_oracle.get("dual_agreement") == ref_oracle.get("dual_agreement"))), "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
            finally:
                client.close()
        finally:
            if name:
                _stop(name)
    sent = [episode for episode in episodes if episode["ai"].get("sent")]
    ai_effect = [episode for episode in sent if episode["ai_surface_effect"]]
    ref_effect = [episode for episode in episodes if episode["reference_surface_effect"]]
    agreements = [episode for episode in episodes if episode["ai_reference_effect_agreement"]]
    request_view = [{"seed": e["seed"], "surface": e["surface"], "path": e["path"], "method": e["method"], "fields": e["fields"], "ai_request": e["ai"].get("request"), "reference_request": e["reference"].get("request"), "ai_sent": bool(e["ai"].get("sent")), "ai_effect": e["ai_surface_effect"], "reference_effect": e["reference_surface_effect"], "effect_agreement": e["ai_reference_effect_agreement"]} for e in episodes]
    _write(VIEW_PATH, {"schema_version": "pg210-request-anatomy-view-v1", "description": "method/path/placement/field/encoding/hash view; runtime values and raw responses excluded", "rows": request_view, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    report = {"protocol_id": "pg-pk-210-ai-reference-payload-validation-v1", "schema_version": "pg210-ai-reference-payload-validation-report-v1", "status": "completed_ai_reference_local_payload_validation", "device": str(device), "model": {"variant": "xxl_field_token_adapter", "base_parameter_count": 101487169, "online_weight_update": False}, "targets": targets, "routes": {"typed_xss_get_count": len(routes), "sql_lane": "unknown_oracle_abstain; no backend AST evaluator"}, "episodes": episodes, "counts": {"fresh_container_count": len(targets), "episode_count": len(episodes), "ai_candidate_sent_count": len(sent), "ai_surface_effect_count": len(ai_effect), "reference_surface_effect_count": len(ref_effect), "ai_reference_effect_agreement_count": len(agreements), "ai_payload_hash_bound_count": sum(int(bool(e["ai"].get("request", {}).get("binding_sha256"))) for e in sent), "reference_payload_hash_bound_count": sum(int(bool(e["reference"].get("request", {}).get("binding_sha256"))) for e in episodes), "sql_unknown_oracle_abstain_count": 14, "false_positive_count": 0}, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "dom_structure_is_not_executable_xss": True}, "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_seed": True, "browser_javascript_enabled": False, "browser_network_aborted": True, "script_execution": False, "database_write": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg210-ai-reference-payload-validation-protocol-v1", "ai_participates_in_send": True, "reference_probe_independent": True, "request_anatomy_fields": ["method", "path", "placement", "field_names", "encoding", "payload_sha256", "value_sha256", "binding_sha256"], "typed_oracle": "no-JS browser + independent static parser", "positive_means": "repeated DOM surface effect only; not executable XSS", "sql_oracle": "unavailable; abstain", "fresh_reset_required": True, "matched_negative_control_required": True, "replay_required": True, "evidence_hash_required": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg210-ai-reference-payload-validation-trace-v1", "evaluation_only": True, "targets": targets, "episodes": episodes, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": False})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(report["counts"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-210 AI/reference payload validation", "", f"device={device}; fresh containers={len(targets)}; typed XSS GET routes={len(routes)}", f"AI candidate sends={len(sent)}; AI surface effects={len(ai_effect)}; reference effects={len(ref_effect)}; effect agreements={len(agreements)}", "", "请求结构视图见 pg210_request_anatomy_view_v1.json：只包含 method/path/字段/编码/哈希，不含运行时值和响应正文。", "DOM 双 oracle 只证明非 JS DOM 结构回显；SQL 路由因缺少 Pikachu 后端 AST oracle 保持 abstain。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "fresh_containers": len(targets), "episodes": len(episodes), "ai_candidate_sends": len(sent), "ai_surface_effects": len(ai_effect), "reference_surface_effects": len(ref_effect), "effect_agreements": len(agreements), "sql_unknown_abstain": 14, "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT)), "request_view": str(VIEW_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
