"""PG-197: risk-aware XXL action loop with independent DOM/SQL evaluators."""

from __future__ import annotations

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

from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app.pg193_browser_dom_oracle import run_browser_dom_oracle  # noqa: E402
from app.pg195_request_surface_adapter import (  # noqa: E402
    build_surface_action_manifest,
    build_surface_values,
    project_surface_response,
    send_surface_request,
)
from app.pg197_alt_dom_oracle import run_alt_dom_oracle  # noqa: E402
from app.pg197_risk_aware_decoder import RiskAwareActionDecoder, predict, train_risk_decoder  # noqa: E402
from app.pg196_failure_action_decoder import encode_features, enumerate_rows  # noqa: E402
from app.sql_differential_fixture_v4 import collect_sql_v4, make_sql_v4_fixture_server, sql_v4_source_sha256  # noqa: E402
from app.sql_differential_fixture_v5 import collect_sql_v5, make_sql_v5_fixture_server, sql_v5_source_sha256  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")
PG194 = _load_script("run_pg194_evaluator_aware_gate_cross_replay.py")

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg197-risk-aware-cross-evaluator-v1"
REPORT_PATH = RESEARCH / "pg197_risk_aware_cross_evaluator_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg197_risk_aware_cross_evaluator_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg197_risk_aware_cross_evaluator_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg197_risk_aware_cross_evaluator_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3106
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (19701, 19702, 19703)


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
        raise RuntimeError(f"PG-197 refuses to reuse target {name}")
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
    raise RuntimeError(f"PG-197 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _load_decoder(device: torch.device, vocabulary: dict[str, int]) -> tuple[RiskAwareActionDecoder, dict[str, Any]]:
    base = PG194._load_model(vocabulary, device)
    prior = ROOT / "artifacts" / "pg195-get-post-layout-sql-v1" / "xxl_evaluator_aware.pt"
    if prior.exists():
        checkpoint = torch.load(prior, map_location="cpu", weights_only=False)
        base.load_state_dict(checkpoint["model_state"])
    decoder = RiskAwareActionDecoder(base).to(device)
    train_rows, holdout_rows = enumerate_rows()
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    metrics = train_risk_decoder(decoder, train_rows, holdout_rows, ids, mask, epochs=30, gate_epochs=200)
    return decoder, metrics


def _manifest_view(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    keys = ("manifest_id", "payload_sha256", "probe_ref", "probe_kind", "route_template_id", "method", "placement", "encoding_chain", "encoding_depth", "marker_sha256", "manifest_sha256", "form_field_names", "form_content_type", "safety")
    return {key: manifest[key] for key in keys if key in manifest}


def _model_decision(decoder: RiskAwareActionDecoder, *, vocabulary: dict[str, int], device: torch.device, context: list[str], state: dict[str, Any]) -> dict[str, Any]:
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    output = predict(decoder, ids=ids, mask=mask, features=encode_features(**state))
    output["state"] = {key: state[key] for key in ("method", "redirect_hops", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")}
    return output


def _surface_episode(decoder: RiskAwareActionDecoder, vocabulary: dict[str, int], device: torch.device, *, target_hash: str, seed: int, path: str, method: str, surface: str, family: str, fields: list[str], observed_fields: list[str], layout: str) -> dict[str, Any]:
    typed_available = family == "xss"
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    control_done = False
    effect = False
    agreement = False
    baseline_status: int | None = None
    try:
        for index in range(1, 4):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            if index == 1:
                state = {"method": method, "redirect_hops": 0, "status_class": "2xx", "candidate_signal": 0, "typed_available": int(typed_available), "negative_control": 0, "budget_remaining": 3, "failure_kind": "no_effect"}
                decision = _model_decision(decoder, vocabulary=vocabulary, device=device, context=context, state=state)
                if method == "GET":
                    response = client.get(path, follow_redirects=False)
                else:
                    response = client.post(path, data={}, follow_redirects=False)
                projected = project_surface_response(response, marker=f"pg197-base-{seed}", layout_variant=layout, baseline_status=None, run_browser=False)
                baseline_status = int(response.status_code)
                role, controller, manifest = "negative_control", f"send_safe_baseline_{method.casefold()}", None
            elif index == 2:
                marker = f"pg197-control-{seed}"
                manifest = build_surface_action_manifest(path=path, method=method, surface=surface, field_names=fields, probe_role="control", marker=marker)
                values = build_surface_values(field_names=fields, probe_role="control", marker=marker)
                projected = send_surface_request(client, path=path, method=method, values=values, marker=marker, layout_variant=layout, baseline_status=baseline_status, run_browser=False)
                state = {"method": method, "redirect_hops": 0, "status_class": str(projected["response_projection"].get("status_class", "2xx")), "candidate_signal": int(projected["response_projection"].get("marker", {}).get("reflected", False)), "typed_available": int(typed_available), "negative_control": 0, "budget_remaining": 2, "failure_kind": "status_changed" if projected["response_projection"].get("status_changed") else "no_effect"}
                decision = _model_decision(decoder, vocabulary=vocabulary, device=device, context=context, state=state)
                role, controller = "control", f"send_matched_control_{method.casefold()}"
                control_done = True
            else:
                marker = f"pg197-candidate-{seed}"
                state = {"method": method, "redirect_hops": 0, "status_class": "2xx", "candidate_signal": 1, "typed_available": int(typed_available), "negative_control": int(control_done), "budget_remaining": 1, "failure_kind": "no_effect"}
                decision = _model_decision(decoder, vocabulary=vocabulary, device=device, context=context, state=state)
                if not typed_available:
                    failure = failure_signature({"method": method, "role": "candidate", "candidate_signal": False, "positive": False, "positive_authority": False, "typed_available": False, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
                    steps.append({"step_index": index, "model": decision, "controller_decision": "abstain_unknown_oracle", "abstain_reason": "pikachu_surface_oracle_unknown", "method": method, "role": "candidate", "response_projection": None, "typed_oracle": {"positive": False, "positive_authority": False}, "failure_signature": failure, "evidence": {"target_instance_hash": target_hash, "failure_sha256": _digest(failure)}, "confirmed_positive": False, "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
                    break
                manifest = build_surface_action_manifest(path=path, method=method, surface=surface, field_names=fields, probe_role="candidate", marker=marker)
                values = build_surface_values(field_names=fields, probe_role="candidate", marker=marker)
                projected = send_surface_request(client, path=path, method=method, values=values, marker=marker, layout_variant=layout, baseline_status=baseline_status, run_browser=True)
                browser = run_browser_dom_oracle(f"<main>{projected['body_text']}</main>", marker=marker)
                alternate = run_alt_dom_oracle(f"<main>{projected['body_text']}</main>", marker=marker)
                effect = bool(projected["typed_surface_effect"])
                agreement = bool(browser["dom_change"] == alternate["dom_change"] == effect)
                role, controller = "candidate", f"send_dual_dom_evaluator_candidate_{method.casefold()}"
                decision["dom_evaluators"] = {"browser_effect": bool(browser["dom_change"]), "alternate_effect": bool(alternate["dom_change"]), "agreement": agreement, "browser_evidence_hash": browser["evidence_hash"], "alternate_evidence_hash": alternate["evidence_hash"]}
            projection = dict(projected["response_projection"])
            oracle = {"typed_surface_effect": bool(projected.get("typed_surface_effect", False)), "positive": False, "positive_authority": False, "confirmed_effect": "dom_structure" if projected.get("typed_surface_effect") else "none"} if typed_available else {"positive": False, "positive_authority": False, "confirmed_effect": "none"}
            failure = failure_signature({"method": method, "role": role, "candidate_signal": bool(projection.get("marker", {}).get("reflected")), "positive": False, "positive_authority": False, "typed_available": typed_available, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
            evidence = {"target_instance_hash": target_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection.get("projection_sha256"), "oracle_sha256": _digest(oracle), "failure_sha256": _digest(failure)}
            steps.append({"step_index": index, "model": decision, "controller_decision": controller, "method": method, "role": role, "action_manifest": _manifest_view(manifest), "response_projection": projection, "typed_oracle": oracle, "failure_signature": failure, "evidence": evidence, "confirmed_positive": False, "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": _manifest_view(manifest) or {"method": method, "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"typed_dom_surface_effect": 0.8 if effect else 0.0, "unknown_oracle": 0.2 if not effect else 0.0}})
        return {"surface": surface, "path": path, "method": method, "family": family, "observed_field_names": observed_fields, "replay_field_names": fields, "layout_variant": layout, "target_instance_hash": target_hash, "seed": seed, "fresh_container": True, "typed_oracle_available": typed_available, "dual_dom_effect_agreement": agreement if typed_available else False, "typed_surface_effect": effect, "confirmed_positive": False, "vulnerability_claim_allowed": False, "steps": steps}
    finally:
        client.close()


def _sql_pair(variant4: str, port4: int, variant5: str, port5: int) -> dict[str, Any]:
    server4 = make_sql_v4_fixture_server(port=port4, variant=variant4)
    server5 = make_sql_v5_fixture_server(port=port5, variant=variant5)
    thread4, thread5 = threading.Thread(target=server4.serve_forever, daemon=True), threading.Thread(target=server5.serve_forever, daemon=True)
    thread4.start(); thread5.start()
    runs: list[dict[str, Any]] = []
    try:
        for method in ("GET", "POST"):
            v4 = collect_sql_v4(target=f"http://127.0.0.1:{port4}", port=port4, variant=variant4, method=method, mode="branch", sample_id=f"{variant4}-{method.casefold()}")
            v5 = collect_sql_v5(target=f"http://127.0.0.1:{port5}", port=port5, variant=variant5, method=method, mode="branch", sample_id=f"{variant5}-{method.casefold()}")
            a, b = v4["oracle_projection"], v5["oracle_projection"]
            runs.append({"method": method, "v4_typed_positive": bool(a["interpreter_boundary"]), "v5_typed_positive": bool(b["interpreter_boundary"]), "agreement": bool(a["candidate_signal"] == b["candidate_signal"] and a["interpreter_boundary"] == b["interpreter_boundary"]), "v4_evidence_hash": v4["evidence_hash"], "v5_evidence_hash": v5["evidence_hash"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    finally:
        server4.shutdown(); server5.shutdown(); server4.server_close(); server5.server_close(); thread4.join(timeout=2.0); thread5.join(timeout=2.0)
    return {"v4_variant": variant4, "v5_variant": variant5, "v4_source_hash": sql_v4_source_sha256(), "v5_source_hash": sql_v5_source_sha256(), "runs": runs, "agreement_count": sum(int(row["agreement"]) for row in runs), "typed_positive_count": sum(int(row["v5_typed_positive"]) for row in runs), "database_execution": False, "vulnerability_claim_allowed": False}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG191.PG189._load_rows()
    vocabulary = PG191.PG189._vocabulary(train, PG191.PG189._load_body_vocab())
    decoder, decoder_training = _load_decoder(device, vocabulary)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg197-risk-aware-cross-evaluator-v1", "vocabulary": vocabulary, "model_state": decoder.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_risk_aware_decoder.pt")
    route_specs = [
        {"path": "/vul/xss/xss_01.php", "method": "GET", "surface": "pg197_xss_get", "family": "xss", "fields": ["message", "submit"], "observed_fields": ["message", "submit"], "layout": "inline_html"},
        {"path": "/vul/sqli/sqli_search.php", "method": "GET", "surface": "pg197_sql_get", "family": "sql_unknown", "fields": ["name", "submit"], "observed_fields": ["name", "submit"], "layout": "table_cell"},
        {"path": "/vul/xss/xsspost/post_login.php", "method": "POST", "surface": "pg197_post_unknown", "family": "unknown_post", "fields": ["submit", "username"], "observed_fields": ["password", "submit", "username"], "layout": "attribute_shell"},
    ]
    route_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"sift-pg197-{seed}"
        container_id = _start(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
        try:
            for spec in route_specs:
                route_runs.append(_surface_episode(decoder, vocabulary, device, target_hash=target_hash, seed=seed, **spec))
        finally:
            _stop(name)
    sql_pairs = [_sql_pair(v4, 8820 + index, v5, 8830 + index) for index, (v4, v5) in enumerate(zip(("delta", "epsilon", "zeta"), ("indigo", "jade", "krypton")))]
    dom_runs = [row for row in route_runs if row["family"] == "xss"]
    unknown_runs = [row for row in route_runs if row["family"] != "xss"]
    report = {"protocol_id": "pg-pk-197-risk-aware-cross-evaluator-v1", "schema_version": "pg197-risk-aware-cross-evaluator-report-v1", "status": "completed_risk_aware_decoder_and_dual_evaluator_replay", "device": str(device), "model": {"variant": "xxl", "base_parameter_count": int(sum(p.numel() for p in decoder.frozen_base.parameters())), "total_parameter_count": int(sum(p.numel() for p in decoder.parameters())), "online_weight_update": False}, "decoder_training": decoder_training, "targets": targets, "route_runs": route_runs, "sql_pairs": sql_pairs, "counts": {"fresh_container_count": len(targets), "route_replay_count": len(route_runs), "dom_run_count": len(dom_runs), "dom_dual_agreement_count": sum(int(row["dual_dom_effect_agreement"]) for row in dom_runs), "dom_typed_surface_effect_count": sum(int(row["typed_surface_effect"]) for row in dom_runs), "unknown_oracle_run_count": len(unknown_runs), "unknown_oracle_abstain_count": sum(int(any(step.get("abstain_reason") == "pikachu_surface_oracle_unknown" for step in row["steps"])) for row in unknown_runs), "sql_pair_count": len(sql_pairs), "sql_get_post_agreement_count": sum(int(row["agreement_count"]) for row in sql_pairs), "sql_typed_positive_count": sum(int(row["typed_positive_count"]) for row in sql_pairs), "false_positive_count": 0}, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "learned_gate_required": True, "cross_source_agreement_required": True}, "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_seed": True, "browser_javascript_enabled": False, "browser_network_aborted": True, "sql_database_execution": False, "external_network": False, "script_execution": False, "database_write": False, "credentials_accessed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg197-risk-aware-cross-evaluator-trace-v1", "evaluation_only": True, "route_runs": route_runs, "sql_pairs": sql_pairs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg197-risk-aware-cross-evaluator-protocol-v1", "model_variant": "xxl", "action_decoder": "risk_aware_action_plus_candidate_gate", "dom_evaluators": ["pg193-browser-dom-nojs-v1", "pg197-static-dom-parser-v1"], "sql_evaluators": ["synthetic_sql_shape_differential_v4", "synthetic_sql_shape_differential_v5"], "pikachu_routes": ["/vul/xss/xss_01.php", "/vul/sqli/sqli_search.php", "/vul/xss/xsspost/post_login.php"], "methods": ["GET", "POST"], "fresh_container_per_seed": True, "dual_evaluator_agreement_required": True, "unknown_oracle_action": "abstain", "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-197 risk-aware XXL decoder and dual evaluator", "", f"device={device}; raw_holdout_unsafe={decoder_training['holdout']['raw_unsafe_allow_count']}; gated_holdout_unsafe={decoder_training['holdout']['gated_unsafe_allow_count']}; DOM agreement={report['counts']['dom_dual_agreement_count']}/{len(dom_runs)}; SQL agreement={report['counts']['sql_get_post_agreement_count']}/{len(sql_pairs)*2}", "", "| lane | runs | agreement/effect | claim allowed |", "|---|---:|---:|---:|", f"| Pikachu DOM dual oracle | {len(dom_runs)} | {report['counts']['dom_dual_agreement_count']} | false |", f"| Pikachu unknown SQL/POST | {len(unknown_runs)} | {report['counts']['unknown_oracle_abstain_count']} abstain | false |", f"| SQL v4/v5 source pair | {len(sql_pairs)*2} | {report['counts']['sql_get_post_agreement_count']} | false |", "", "The raw decoder remains diagnostic; learned candidate gate and cross-source evaluator agreement are required before any candidate send.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "raw_holdout": decoder_training["holdout"], "dom_agreement": report["counts"]["dom_dual_agreement_count"], "unknown_abstain": report["counts"]["unknown_oracle_abstain_count"], "sql_agreement": report["counts"]["sql_get_post_agreement_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
