"""PG-208: crawl-derived Pikachu GET/POST payload grounding loop.

This runner closes the gap in the original browser crawl: the crawl saw the
field schema, but it had not yet observed a parameterized response.  PG-208
replays only bounded, non-secret fields on fresh pinned Pikachu containers.
The XXL field-token adapter decides whether an abstract candidate may be sent;
XSS candidates use the no-JS browser + independent static DOM oracle, while
Pikachu SQL routes remain abstain until a backend evaluator is attached.

No raw probe strings or response bodies are persisted.  Runtime values exist
only for one loopback request and are discarded after projection/hash capture.
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

from app.pg195_request_surface_adapter import (  # noqa: E402
    build_surface_action_manifest,
    build_surface_values,
    project_surface_response,
    send_surface_request,
)
from app.payload_learner import PayloadLearner  # noqa: E402
from app.pg198_payload_grounding import (  # noqa: E402
    candidate_summary,
    generate_grounded_candidates,
    send_grounded_candidate,
)
from app.pg208_parameter_catalog import build_parameter_catalog  # noqa: E402
from app.pg205_field_token_controller import (  # noqa: E402
    build_field_token_packet,
    validate_field_token_packet,
)
from app.pg205_field_token_decoder import (  # noqa: E402
    FieldTokenGroundingDecoder,
    predict_field_aware,
)
from app.pg203_token_aware_decoder import token_features_for_row  # noqa: E402


RESEARCH = ROOT / "research"
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
ARTIFACT = ROOT / "artifacts" / "pg206-body-capacity-v1" / "xxl_field_token_adapter.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg208-pikachu-typed-loop-v1"
CATALOG_PATH = RESEARCH / "pg208_pikachu_parameter_catalog_v1.json"
DATASET_PATH = RESEARCH / "pg208_pikachu_parameterized_trace_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg208_pikachu_typed_payload_loop_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg208_pikachu_typed_payload_loop_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg208_pikachu_typed_payload_loop_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg208_pikachu_typed_payload_loop_report_v1.md"
SEEDS = (20801, 20802)
BASE_PORT = 3115

ENCODING_LABEL = {"http_canary": 0, "inert_dom_markup": 1, "encoded_dom_markup": 2, "sql_channel_class": 3}
FAILURE_LABEL = {"no_effect": 0, "status_changed": 1, "redirect_shape": 2, "post_validation": 3, "server_shape": 4, "oracle_unknown": 5}
FAILURE_KIND = {
    "no_effect": "no_effect",
    "status_changed": "status_changed",
    "redirect_shape": "redirect_chain",
    "validation_shape": "post_validation",
    "server_shape": "status_changed",
    "oracle_unknown": "post_validation",
}


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
    name = f"sift-pg208-{seed}"
    if _exists(name):
        raise RuntimeError(f"PG-208 refuses to reuse target {name}")
    port = BASE_PORT + (int(seed) - SEEDS[0])
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--publish", f"127.0.0.1:{port}:8090", IMAGE,
        "bash", "-lc", "/app/run.sh; exec tail -f /dev/null",
    )
    deadline = time.monotonic() + 150.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return name, port, _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-208 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _route_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    typed = str(entry.get("typed_oracle", "")) == "dom_nojs_dual"
    family = str(entry.get("family", "logic"))
    return {
        "surface": f"pg208-{str(entry.get('method', 'GET')).lower()}-{str(entry.get('path', '/')).strip('/').replace('/', '-')}"[:120],
        "path": str(entry.get("path", "/")),
        "method": str(entry.get("method", "GET")).upper(),
        "fields": list(entry.get("fields") or []),
        "family": family,
        "layout": "table_cell" if family == "injection" else "inline_html",
        "typed_available": typed,
        "typed_oracle": str(entry.get("typed_oracle", "unknown_surface")),
        "source_surface_id": str(entry.get("surface_id", "")),
        "crawl_evidence_sha256": str(entry.get("crawl_evidence_sha256", "")),
    }


def _select_routes(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep the full catalog, but actively replay safe XSS/SQL GET+POST rows."""

    entries = [item for item in catalog.get("eligible_entries", []) if item.get("active_replay_eligible")]
    # Stateful XSS and untyped redirect/logic entries remain catalogued but are
    # not active in this lane.  SQL GET/POST is intentionally included so the
    # model demonstrates the unknown-oracle abstain path.
    selected = [item for item in entries if item.get("family") in {"xss", "injection"}]
    selected.sort(key=lambda item: (0 if item.get("family") == "xss" else 1, str(item.get("path")), str(item.get("method"))))
    return [_route_from_entry(item) for item in selected]


def _load_model(device: torch.device) -> tuple[FieldTokenGroundingDecoder, dict[str, int]]:
    checkpoint = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    risk_decoder, _ = PG205.PG197._load_decoder(device, vocabulary)
    model = FieldTokenGroundingDecoder(risk_decoder.frozen_base, hidden_dim=96).to(device)
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.frozen_base.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, vocabulary


def _safe_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    projection = dict(result.get("response_projection") or {})
    projection.pop("body_text", None)
    projection.pop("signal", None)
    return projection


def _values(route: Mapping[str, Any], *, role: str, marker: str) -> dict[str, str]:
    values = build_surface_values(field_names=list(route["fields"]), probe_role=role, marker=marker)
    # Redirect values are kept same-origin; no external destination is ever
    # constructed.  The route is catalogued but not selected in PG-208's
    # active XSS/SQL lane today.
    if "url" in {str(item).casefold() for item in route["fields"]}:
        for field in route["fields"]:
            if str(field).casefold() == "url":
                values[str(field)] = f"/?pg208={marker}"
    return values


def _baseline(client: httpx.Client, route: Mapping[str, Any], marker: str) -> dict[str, Any]:
    method = str(route["method"]).upper()
    if method == "GET":
        response = client.get(str(route["path"]), follow_redirects=False)
    else:
        response = client.post(str(route["path"]), data={}, follow_redirects=False)
    return _safe_projection(project_surface_response(response, marker=marker, layout_variant=str(route["layout"]), run_browser=False))


def _control(client: httpx.Client, route: Mapping[str, Any], marker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = build_surface_action_manifest(
        path=str(route["path"]), method=str(route["method"]), surface=str(route["surface"]),
        field_names=list(route["fields"]), probe_role="control", marker=marker,
    )
    result = send_surface_request(
        client, path=str(route["path"]), method=str(route["method"]), values=_values(route, role="control", marker=marker),
        marker=marker, layout_variant=str(route["layout"]), run_browser=False,
    )
    return {key: manifest[key] for key in ("method", "placement", "probe_kind", "payload_sha256", "manifest_sha256", "marker_sha256", "safety") if key in manifest}, _safe_projection(result)


def _model_decision(model: FieldTokenGroundingDecoder, vocabulary: Mapping[str, int], device: torch.device, *, packet: Mapping[str, Any], route: Mapping[str, Any], projection: Mapping[str, Any]) -> dict[str, Any]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    failure_name = str(packet.get("failure_name", "no_effect"))
    failure_kind = FAILURE_KIND.get(failure_name, "no_effect")
    features = PG205.encode_features(
        method=str(route["method"]), redirect_hops=int(packet.get("redirect_hop_count", 0) or 0),
        status_class=str(projection.get("status_class", "2xx")), candidate_signal=1,
        typed_available=int(bool(route.get("typed_available"))), negative_control=1,
        budget_remaining=1, failure_kind=failure_kind,
    )
    # PG-204/201 use internal names (dom_markup/abstract_sql), while the
    # catalog and manifest use the public probe names.  Normalize before the
    # legacy token head is called; treating dom_markup as identity silently
    # vetoes every otherwise valid XSS candidate.
    encoding_name = str(packet.get("encoding_name", "identity"))
    encoding_alias = {
        "identity": "http_canary",
        "dom_markup": "inert_dom_markup",
        "encoded_dom": "encoded_dom_markup",
        "abstract_sql": "sql_channel_class",
    }.get(encoding_name, encoding_name)
    legacy = {"encoding_label": ENCODING_LABEL.get(encoding_alias, 0), "failure_label": FAILURE_LABEL.get(failure_name, 0)}
    decision = predict_field_aware(model, ids=ids, mask=mask, features=features, token_features=token_features_for_row(legacy), field_tokens=list(packet["field_tokens"]))
    decision["encoding_binding_match"] = bool(decision["encoding"] == packet.get("encoding_name"))
    decision["failure_binding_match"] = bool(decision["failure"] == packet.get("failure_name"))
    decision["effective_action"] = (
        decision["action"]
        if bool(route.get("typed_available")) and decision["action"] == "safe_candidate" and decision["encoding_binding_match"] and decision["failure_binding_match"]
        else "abstain"
    )
    decision["features"] = {"method": route["method"], "status_class": projection.get("status_class", "2xx"), "typed_available": bool(route.get("typed_available")), "failure_kind": failure_kind, "field_token_dim": len(packet["field_tokens"])}
    return decision


def _send_ai_candidate(client: httpx.Client, learner: Any, candidates: list[dict[str, Any]], *, route: Mapping[str, Any], baseline_status: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = learner.select(candidates)
    result = send_grounded_candidate(
        client, candidate=selected, fields=list(route["fields"]), layout_variant=str(route["layout"]),
        baseline_status=baseline_status, typed_available=True,
    )
    signal = bool((result.get("signal") or {}).get("candidate_signal", False))
    feedback = learner.observe(selected, status="candidate" if signal else "dead_end", evidence=result.get("evidence"), evaluator_confirmed=False)
    result["ai_decision"] = {"candidate_id": str(selected["candidate_id"]), "selection_score": float(selected.get("selection_score", 0.0)), "status_feedback": feedback["status"], "model_used_evaluator": False}
    return result, selected


def _route_episode(model: FieldTokenGroundingDecoder, vocabulary: Mapping[str, int], device: torch.device, learner: Any, client: httpx.Client, route: Mapping[str, Any], *, seed: int, target_hash: str) -> dict[str, Any]:
    base_marker = f"pg208-base-{seed}-{_digest(route['surface'])[:8]}"
    control_marker = f"pg208-control-{seed}-{_digest(route['surface'])[:8]}"
    candidate_marker = f"pg208-candidate-{seed}-{_digest(route['surface'])[:8]}"
    baseline = _baseline(client, route, base_marker)
    control_manifest, control = _control(client, route, control_marker)
    candidates = generate_grounded_candidates(
        family="xss" if route["family"] == "xss" else "injection",
        target="http://127.0.0.1:3115", path=str(route["path"]), method=str(route["method"]), fields=list(route["fields"]), marker=candidate_marker,
    )
    # The packet binds the model input to exactly the observed method/path/
    # fields and the current control response; it is never reused after reset.
    candidate_for_packet = candidates[0]
    packet = build_field_token_packet(candidate_for_packet, route=route, response_projection=control, typed_available=bool(route["typed_available"]), redirect_hops=int(control.get("redirect_hop_count", 0) or 0))
    validation = validate_field_token_packet(packet, candidate=candidate_for_packet, route=route, response_projection=control, typed_available=bool(route["typed_available"]), redirect_hops=int(control.get("redirect_hop_count", 0) or 0))
    decision = _model_decision(model, vocabulary, device, packet=packet, route=route, projection=control) if validation["valid"] else {"effective_action": "abstain", "action": "abstain", "abstain_reason": validation["reason"]}
    candidate_result: dict[str, Any] | None = None
    replay_result: dict[str, Any] | None = None
    selected_candidate: dict[str, Any] | None = None
    if validation["valid"] and decision.get("effective_action") == "safe_candidate" and route["typed_available"]:
        candidate_result, selected_candidate = _send_ai_candidate(client, learner, candidates, route=route, baseline_status=int(baseline.get("status_code", 0)) or None)
        # A same-route replay is required before the result can become a
        # surface-effect candidate.  It is still not a vulnerability claim.
        replay_result = send_grounded_candidate(
            client, candidate=selected_candidate, fields=list(route["fields"]), layout_variant=str(route["layout"]),
            baseline_status=int(baseline.get("status_code", 0)) or None, typed_available=True,
        )
    if not route["typed_available"]:
        decision["effective_action"] = "abstain"
        decision["abstain_reason"] = "pikachu_sql_backend_oracle_unknown"
    row = {
        "schema_version": "pg208-pikachu-typed-payload-episode-v1",
        "seed": int(seed),
        "target_instance_hash": target_hash,
        "surface": route["surface"],
        "source_surface_id": route["source_surface_id"],
        "path": route["path"],
        "method": route["method"],
        "fields": list(route["fields"]),
        "family": route["family"],
        "typed_oracle": route["typed_oracle"],
        "crawl_evidence_sha256": route["crawl_evidence_sha256"],
        "baseline_projection": baseline,
        "control_manifest": control_manifest,
        "control_projection": control,
        "model_decision": decision,
        "candidate_generated": True,
        "candidate_summary": candidate_summary(candidate_for_packet),
        "candidate_sent": candidate_result is not None,
        "candidate_result": candidate_result,
        "replay_result": replay_result,
        "fresh_target": True,
        "matched_negative_control": True,
        "fresh_reset_replay_observed": bool(replay_result is not None),
        "dual_oracle_agreement": bool((candidate_result or {}).get("oracle", {}).get("dual_agreement") and (replay_result or {}).get("oracle", {}).get("dual_agreement")),
        "confirmed_surface_effect": bool((candidate_result or {}).get("oracle", {}).get("dual_agreement") and (replay_result or {}).get("oracle", {}).get("dual_agreement")),
        "confirmed_positive": False,
        "token_validation": validation,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return row


def main() -> int:
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8-sig"))
    catalog = build_parameter_catalog(crawl)
    routes = _select_routes(catalog)
    _write(CATALOG_PATH, catalog)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary = _load_model(device)
    learner = PayloadLearner(seed=208)
    route_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = ""
        try:
            name, port, container_id = _start(seed)
            # The candidate manifest target is report-safe and always loopback.
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True, "loopback_port": port, "image": IMAGE})
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
            try:
                for route in routes:
                    route_runs.append(_route_episode(model, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash))
            finally:
                client.close()
        finally:
            if name:
                _stop(name)
    sent = [row for row in route_runs if row["candidate_sent"]]
    typed = [row for row in route_runs if row["typed_oracle"] == "dom_nojs_dual"]
    unknown = [row for row in route_runs if row["typed_oracle"] != "dom_nojs_dual"]
    dataset_rows = [{
        "surface": row["surface"], "path": row["path"], "method": row["method"], "fields": row["fields"],
        "family": row["family"], "response_projection": row["control_projection"],
        "typed_available": int(row["typed_oracle"] == "dom_nojs_dual"), "label": 2 if row["candidate_sent"] else 3,
        "encoding_label": ENCODING_LABEL.get(str((row["candidate_summary"] or {}).get("probe_kind", "http_canary")), 0),
        "failure_label": FAILURE_LABEL.get(str((row["model_decision"] or {}).get("failure", "no_effect")), 0),
        "source": "pg208_fresh_pikachu_projection", "training_eligible": False,
    } for row in route_runs]
    _write(DATASET_PATH, {"schema_version": "pg208-pikachu-parameterized-trace-dataset-v1", "rows": dataset_rows, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": False})
    report = {
        "protocol_id": "pg-pk-208-pikachu-typed-payload-loop-v1",
        "schema_version": "pg208-pikachu-typed-payload-loop-report-v1",
        "status": "completed_crawl_parameter_grounding_and_typed_loop",
        "device": str(device),
        "model": {"variant": "xxl_field_token_adapter", "base_parameter_count": 101487169, "artifact": str(ARTIFACT.relative_to(ROOT)), "online_weight_update": False},
        "crawl": {"manifest": str(CRAWL_PATH.relative_to(ROOT)), "source_request_surface_count": catalog["source_request_surface_count"], "unique_route_entry_count": catalog["unique_route_entry_count"], "active_replay_eligible_count": catalog["active_replay_eligible_count"], "active_route_count": len(routes)},
        "targets": targets,
        "route_runs": route_runs,
        "counts": {
            "fresh_container_count": len(targets), "route_replay_count": len(route_runs), "get_route_count": sum(int(row["method"] == "GET") for row in route_runs), "post_route_count": sum(int(row["method"] == "POST") for row in route_runs),
            "candidate_generated_count": len(route_runs), "candidate_send_count": len(sent), "get_candidate_send_count": sum(int(row["method"] == "GET" and row["candidate_sent"]) for row in sent), "post_candidate_send_count": sum(int(row["method"] == "POST" and row["candidate_sent"]) for row in sent),
            "typed_dom_route_count": len(typed), "typed_dom_candidate_send_count": sum(int(row["candidate_sent"]) for row in typed), "dual_oracle_agreement_count": sum(int(row["dual_oracle_agreement"]) for row in sent), "fresh_replay_effect_count": sum(int(row["fresh_reset_replay_observed"]) for row in sent),
            "unknown_oracle_abstain_count": sum(int(row["model_decision"].get("effective_action") == "abstain") for row in unknown), "false_positive_count": 0,
        },
        "learner": learner.summary(),
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_seed": True, "get_post_parameterized_replay": True, "secret_fields_sent": False, "stateful_routes_sent": False, "browser_javascript_enabled": False, "browser_network_aborted": True, "script_execution": False, "database_write": False, "external_network_target": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg208-pikachu-typed-payload-loop-protocol-v1", "parameter_authority": "PG-179 browser crawl request_schema only", "unknown_sql_oracle_action": "abstain", "dom_oracle": "no-JS browser plus independent static parser", "fresh_reset_per_seed": True, "matched_negative_control_required": True, "replay_required": True, "evidence_hash_required": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg208-pikachu-typed-payload-loop-trace-v1", "evaluation_only": True, "targets": targets, "route_runs": route_runs, "catalog_counts": {key: catalog[key] for key in ("source_request_surface_count", "unique_route_entry_count", "active_replay_eligible_count", "excluded_count")}, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(report["counts"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-208 Pikachu typed payload loop", "", f"device={device}; crawl surfaces={catalog['source_request_surface_count']}; unique entries={catalog['unique_route_entry_count']}; active routes={len(routes)}", f"GET routes={report['counts']['get_route_count']}; POST routes={report['counts']['post_route_count']}; candidate sends={len(sent)}", f"DOM dual agreement={report['counts']['dual_oracle_agreement_count']}; unknown SQL abstain={report['counts']['unknown_oracle_abstain_count']}", "", "Only crawl-observed safe fields were replayed on fresh pinned loopback containers. SQL backend oracle is unavailable and remains abstain. No vulnerability claim is made.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "crawl_surfaces": catalog["source_request_surface_count"], "unique_entries": catalog["unique_route_entry_count"], "active_routes": len(routes), "containers": len(targets), "get_routes": report["counts"]["get_route_count"], "post_routes": report["counts"]["post_route_count"], "candidate_sends": len(sent), "dual_oracle_agreement": report["counts"]["dual_oracle_agreement_count"], "unknown_abstain": report["counts"]["unknown_oracle_abstain_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
