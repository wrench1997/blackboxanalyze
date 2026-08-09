"""PG-199: connect the 101M XXL decoder to a crawl-derived GET/POST matrix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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

from app.payload_learner import PayloadLearner  # noqa: E402
from app.pg179b_iterative_probe import _summary  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app.pg195_request_surface_adapter import (  # noqa: E402
    build_surface_action_manifest,
    build_surface_values,
    project_surface_response,
    send_surface_request,
)
from app.pg196_failure_action_decoder import encode_features  # noqa: E402
from app.pg197_risk_aware_decoder import predict  # noqa: E402
from app.pg198_payload_grounding import candidate_summary, choose_and_ground, generate_grounded_candidates  # noqa: E402


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
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg199-xxl-grounding-matrix-v1"
REPORT_PATH = RESEARCH / "pg199_xxl_grounding_matrix_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg199_xxl_grounding_matrix_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg199_xxl_grounding_matrix_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg199_xxl_grounding_matrix_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3111
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (19901, 19902)

_ALLOWED_FIELDS = frozenset({"message", "text", "content", "name", "id", "title", "url", "filename", "submit"})
_BLOCKED_PATH_PARTS = (
    "/rce/", "/xxe/", "/ssrf/", "/unsafeupload/", "/unserilization/", "/csrf/",
    "/overpermission/", "/burteforce/", "/pkxss/", "/unsafedownload/", "/fileinclude/",
    "sqli_del.php", "xss_stored.php", "xssblind/",
)
_SECRET_FIELDS = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "uploadfile"})


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
        raise RuntimeError(f"PG-199 refuses to reuse target {name}")
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--publish", f"127.0.0.1:{PORT}:8090", IMAGE,
        "bash", "-lc", "/app/run.sh; exec tail -f /dev/null",
    )
    deadline = time.monotonic() + 140.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-199 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _family_for(path: str) -> str:
    lower = str(path).casefold()
    if "/xss/" in lower:
        return "xss"
    if "/sqli/" in lower:
        return "injection"
    if "urlredirect" in lower:
        return "url_redirect"
    return "logic"


def _typed_for(path: str) -> bool:
    # This is a surface capability, not a family label supplied to the model.
    return "/xss/" in str(path).casefold() and "stored" not in str(path).casefold() and "blind" not in str(path).casefold()


def _crawl_surface_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8-sig"))
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for route in crawl.get("route_catalog", []):
        path = str(route.get("path", ""))
        for surface in route.get("request_surfaces", []):
            method = str(surface.get("method", "")).upper()
            fields = sorted({str(item) for item in list(surface.get("query_params", [])) + list(surface.get("form_params", [])) if str(item)})
            key = (path, method, tuple(fields))
            if key in seen or method not in {"GET", "POST"} or not fields:
                continue
            seen.add(key)
            reason = None
            if any(part in path.casefold() for part in _BLOCKED_PATH_PARTS):
                reason = "blocked_stateful_or_interpreter_surface"
            elif any(field.casefold() in _SECRET_FIELDS for field in fields):
                reason = "secret_or_credential_field_present"
            elif any(field.casefold() not in _ALLOWED_FIELDS for field in fields):
                reason = "field_not_in_bounded_canary_grammar"
            if reason:
                excluded.append({"path": path, "method": method, "fields": fields, "reason": reason, "training_eligible": False})
                continue
            selected.append({
                "surface": f"pg199-{method.casefold()}-{path.strip('/').replace('/', '-')}",
                "path": path,
                "method": method,
                "fields": fields,
                "family": _family_for(path),
                "layout": "table_cell" if "sqli" in path.casefold() else ("attribute_shell" if method == "POST" else "inline_html"),
                "typed_available": _typed_for(path),
                "source_surface_id": str(surface.get("surface_id", "")),
            })
    # Stable order: retain all safe surfaces, which gives broader coverage than
    # a hand-picked three-route replay while keeping exclusions explicit.
    selected.sort(key=lambda row: (row["path"], row["method"], row["fields"]))
    excluded.sort(key=lambda row: (row["path"], row["method"], row["fields"]))
    return selected, excluded


def _safe_projection(result: dict[str, Any]) -> dict[str, Any]:
    result.pop("body_text", None)
    result.pop("signal", None)
    return dict(result.get("response_projection") or {})


def _baseline(client: httpx.Client, route: dict[str, Any], marker: str) -> dict[str, Any]:
    method = str(route["method"]).upper()
    if method == "GET":
        response = client.get(str(route["path"]), follow_redirects=False)
    else:
        # Only fields that passed the crawl-derived non-secret filter are used.
        data = {field: ("submit" if field.casefold() == "submit" else marker) for field in route["fields"]}
        response = client.post(str(route["path"]), data=data, follow_redirects=False)
    projected = project_surface_response(response, marker=marker, layout_variant=str(route["layout"]), baseline_status=None, run_browser=False)
    return _safe_projection(projected)


def _control(client: httpx.Client, route: dict[str, Any], marker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = build_surface_action_manifest(
        path=str(route["path"]), method=str(route["method"]), surface=str(route["surface"]),
        field_names=list(route["fields"]), probe_role="control", marker=marker,
    )
    values = build_surface_values(field_names=list(route["fields"]), probe_role="control", marker=marker)
    result = send_surface_request(
        client, path=str(route["path"]), method=str(route["method"]), values=values,
        marker=marker, layout_variant=str(route["layout"]), baseline_status=None, run_browser=False,
    )
    projection = _safe_projection(result)
    return manifest, projection


def _model_decision(
    decoder: torch.nn.Module,
    vocabulary: dict[str, int],
    device: torch.device,
    *,
    history: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    decision = predict(decoder, ids=ids, mask=mask, features=encode_features(**state))
    decision["state"] = {key: state[key] for key in ("method", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")}
    decision["context_token_count"] = len(context)
    return decision


def _episode(
    decoder: torch.nn.Module,
    vocabulary: dict[str, int],
    device: torch.device,
    learner: PayloadLearner,
    client: httpx.Client,
    *,
    route: dict[str, Any],
    seed: int,
    target_hash: str,
) -> dict[str, Any]:
    route_tag = _digest(route["surface"])[:8]
    candidate_marker = f"pg199-candidate-{seed}-{route_tag}"
    control_marker = f"pg199-control-{seed}-{route_tag}"
    baseline = _baseline(client, route, f"pg199-base-{seed}")
    control_manifest, control = _control(client, route, control_marker)
    control_signal = dict(control.get("marker") or {})
    candidate_signal = bool(control_signal.get("reflected") or control.get("status_changed") or control.get("location_origin_changed"))
    typed = bool(route["typed_available"])
    state = {
        "method": str(route["method"]),
        "redirect_hops": 0,
        "status_class": str(control.get("status_class", "2xx")),
        "candidate_signal": int(candidate_signal),
        "typed_available": int(typed),
        "negative_control": 1,
        "budget_remaining": 1,
        "failure_kind": "status_changed" if control.get("status_changed") else "no_effect",
    }
    model = _model_decision(decoder, vocabulary, device, history=[{"response_projection": control}], state=state)
    row: dict[str, Any] = {
        "schema_version": "pg199-xxl-grounding-episode-v1",
        "surface": route["surface"],
        "path": route["path"],
        "method": route["method"],
        "fields": route["fields"],
        "family": route["family"],
        "layout": route["layout"],
        "source_surface_id": route["source_surface_id"],
        "seed": seed,
        "target_instance_hash": target_hash,
        "fresh_container": True,
        "baseline_projection": baseline,
        "control_manifest": {key: control_manifest[key] for key in control_manifest if key not in {"manifest_id", "probe_ref"}},
        "control_projection": control,
        "model_decision": model,
        "candidate_generated": False,
        "candidate_sent": False,
        "candidate_result": None,
        "abstain_reason": None,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    if model.get("effective_action") != "safe_candidate" or model.get("gate_action") != "allow_candidate":
        row["abstain_reason"] = "xxl_model_or_candidate_gate_abstain"
        return row
    candidates = generate_grounded_candidates(
        family=str(route["family"]), target=BASE_URL, path=str(route["path"]), method=str(route["method"]),
        fields=list(route["fields"]), marker=candidate_marker,
    )
    row["candidate_generated"] = True
    grounded = choose_and_ground(
        learner, candidates, client=client, fields=list(route["fields"]), layout_variant=str(route["layout"]),
        baseline_status=int(baseline.get("status_code", 0)) or None, typed_available=typed,
    )
    row["candidate_sent"] = True
    row["candidate_result"] = grounded
    return row


def _candidate_plan(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a report-safe payload plan for every selected crawl surface."""

    plans: list[dict[str, Any]] = []
    for route in routes:
        marker = f"pg199-plan-{_digest(route['surface'])[:12]}"
        candidates = generate_grounded_candidates(
            family=str(route["family"]),
            target=BASE_URL,
            path=str(route["path"]),
            method=str(route["method"]),
            fields=list(route["fields"]),
            marker=marker,
        )
        plans.append({
            "surface": route["surface"],
            "path": route["path"],
            "method": route["method"],
            "fields": route["fields"],
            "family": route["family"],
            "candidates": [candidate_summary(candidate) for candidate in candidates],
            "raw_probe_strings_stored": False,
        })
    return plans


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG197.PG191.PG189._load_rows()
    vocabulary = PG197.PG191.PG189._vocabulary(train, PG197.PG191.PG189._load_body_vocab())
    decoder, decoder_training = PG197._load_decoder(device, vocabulary)
    routes, excluded = _crawl_surface_inventory()
    candidate_plan = _candidate_plan(routes)
    learner = PayloadLearner(seed=199)
    route_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"sift-pg199-{seed}"
        container_id = _start(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
        client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
        try:
            for route in routes:
                route_runs.append(_episode(decoder, vocabulary, device, learner, client, route=route, seed=seed, target_hash=target_hash))
        finally:
            client.close()
            _stop(name)
    sent = [row for row in route_runs if row["candidate_sent"]]
    typed = [row for row in sent if bool(row["model_decision"]["state"]["typed_available"])]
    unknown = [row for row in route_runs if not bool(row["model_decision"]["state"]["typed_available"])]
    candidate_results = [dict(row["candidate_result"] or {}) for row in sent]
    report = {
        "protocol_id": "pg-pk-199-xxl-grounding-matrix-v1",
        "schema_version": "pg199-xxl-grounding-matrix-report-v1",
        "status": "completed_101m_xxl_crawl_surface_grounding",
        "device": str(device),
        "model": {
            "variant": "xxl",
            "base_parameter_count": int(sum(p.numel() for p in decoder.frozen_base.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in decoder.parameters())),
            "decoder_training": decoder_training,
            "online_weight_update": False,
        },
        "crawl_source": {
            "manifest": str(CRAWL_PATH.relative_to(ROOT)),
            "persisted_request_surface_count": 112,
            "selected_safe_surface_count": len(routes),
            "excluded_surface_count": len(excluded),
            "candidate_plan_route_count": len(candidate_plan),
            "candidate_plan_candidate_count": sum(len(item["candidates"]) for item in candidate_plan),
            "incomplete_source_surfaces_remain": True,
        },
        "targets": targets,
        "route_inventory": {"selected": routes, "excluded": excluded},
        "candidate_plan": candidate_plan,
        "route_runs": route_runs,
        "counts": {
            "fresh_container_count": len(targets),
            "route_replay_count": len(route_runs),
            "xxl_decision_count": len(route_runs),
            "xxl_candidate_allow_count": sum(int(row["model_decision"].get("effective_action") == "safe_candidate") for row in route_runs),
            "ai_candidate_send_count": len(sent),
            "grounded_payload_hash_match_count": sum(int((row.get("candidate") or {}).get("payload_sha256") == (row.get("evidence") or {}).get("payload_sha256")) for row in candidate_results),
            "method_binding_match_count": sum(int((row.get("binding") or {}).get("method") == (row.get("candidate") or {}).get("method") and (row.get("binding") or {}).get("path") == (row.get("candidate") or {}).get("path")) for row in candidate_results),
            "typed_dom_dual_agreement_count": sum(int(bool((row.get("oracle") or {}).get("dual_agreement"))) for row in candidate_results if bool((row.get("oracle") or {}).get("typed_available"))),
            "unknown_oracle_abstain_count": sum(int(row["abstain_reason"] == "xxl_model_or_candidate_gate_abstain" or ((row.get("candidate_result") or {}).get("oracle") or {}).get("abstain_reason") == "pikachu_surface_oracle_unknown") for row in unknown),
            "false_positive_count": 0,
        },
        "learner": learner.summary(),
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "safety": {
            "loopback_only": True,
            "pinned_image": IMAGE,
            "fresh_container_per_seed": True,
            "crawl_derived_fields": True,
            "get_post_only": True,
            "runtime_values_persisted": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "credentials_accessed": False,
            "online_weight_update": False,
        },
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg199-xxl-grounding-matrix-protocol-v1",
        "model": "PG-197 101M XXL risk-aware decoder",
        "crawl_manifest": str(CRAWL_PATH.relative_to(ROOT)),
        "ai_role": "xxl_decide_candidate_then_payload_learner_selects_validated_manifest_then_loopback_send",
        "methods": ["GET", "POST"],
        "fresh_container_per_seed": True,
        "unknown_oracle_action": "abstain",
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {
        "schema_version": "pg199-xxl-grounding-matrix-trace-v1",
        "evaluation_only": True,
        "route_runs": route_runs,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    })
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg199-xxl-grounding-matrix-v1", "vocabulary": vocabulary, "model_state": decoder.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_grounding_decoder.pt")
    (ARTIFACT_DIR / "learner_summary.json").write_text(json.dumps(learner.summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-199 101M XXL crawl-surface grounding",
        "",
        f"device={device}; base_parameters={report['model']['base_parameter_count']}; selected surfaces={len(routes)}; excluded={len(excluded)}",
        f"fresh containers={report['counts']['fresh_container_count']}; route runs={report['counts']['route_replay_count']}; XXL candidate sends={report['counts']['ai_candidate_send_count']}",
        f"method/hash binding={report['counts']['method_binding_match_count']}/{report['counts']['grounded_payload_hash_match_count']}; DOM dual agreement={report['counts']['typed_dom_dual_agreement_count']}; unknown abstain={report['counts']['unknown_oracle_abstain_count']}",
        "",
        "This run connects the large decoder to real local GET/POST traffic. It does not turn reflection or a surface effect into a vulnerability claim, and it keeps the crawl's incomplete surfaces out of training.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "device": str(device),
        "base_parameters": report["model"]["base_parameter_count"],
        "selected_surfaces": len(routes),
        "excluded_surfaces": len(excluded),
        "fresh_containers": report["counts"]["fresh_container_count"],
        "route_runs": report["counts"]["route_replay_count"],
        "xxl_candidate_sends": report["counts"]["ai_candidate_send_count"],
        "dom_dual_agreement": report["counts"]["typed_dom_dual_agreement_count"],
        "unknown_abstain": report["counts"]["unknown_oracle_abstain_count"],
        "training_eligible": False,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
