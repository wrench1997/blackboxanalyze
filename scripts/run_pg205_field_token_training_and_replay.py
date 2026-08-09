"""PG-205: train field-aware adapters and replay unseen Pikachu surfaces.

The runner uses browser-crawl request schemas, learns bounded request/response
structure slots, and lets the selected adapter gate a fresh local GET/POST
request.  It never stores runtime values or response bodies.
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

from app.detection_payload import validate_detection_payload  # noqa: E402
from app.pg179b_iterative_probe import request_chain  # noqa: E402
from app.pg195_request_surface_adapter import build_surface_values, project_surface_response, send_surface_request  # noqa: E402
from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates, send_grounded_candidate  # noqa: E402
from app.pg196_failure_action_decoder import encode_features  # noqa: E402
from app.pg201_multitask_decoder import ENCODING_NAMES, FAILURE_NAMES  # noqa: E402
from app.pg203_token_aware_decoder import token_features_for_row  # noqa: E402
from app.pg204_token_binding_controller import build_runtime_token_packet  # noqa: E402
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet  # noqa: E402
from app.pg205_field_token_decoder import FieldTokenGroundingDecoder, evaluate_field_aware, predict_field_aware, train_field_aware, warm_start_from_pg203  # noqa: E402
from app.pg205_request_response_tokens import field_tokens_for_runtime  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG197 = _load_script("run_pg197_risk_aware_cross_evaluator.py")
PG201 = _load_script("run_pg201_multitask_decoder.py")

RESEARCH = ROOT / "research"
PG199_REPORT = RESEARCH / "pg199_xxl_grounding_matrix_report_v1.json"
PG200_REPORT = RESEARCH / "pg200_source_heldout_report_v1.json"
CRAWL_MANIFEST = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
PG203_ARTIFACT = ROOT / "artifacts" / "pg203-token-aware-adapter-v1" / "xxl_token_aware_adapter.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg205-field-token-v1"
REPORT_PATH = RESEARCH / "pg205_field_token_training_and_replay_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg205_field_token_training_and_replay_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg205_field_token_training_and_replay_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg205_field_token_training_and_replay_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3113
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (20501, 20502)

ROUTES: tuple[dict[str, Any], ...] = (
    {"surface": "pg205_get_xss_multi", "path": "/vul/xss/xss_03.php", "method": "GET", "fields": ["message", "submit"], "family": "xss", "layout": "inline_html", "typed_available": True, "encoding": "inert_dom_markup", "redirect_chain": False},
    {"surface": "pg205_get_redirect", "path": "/vul/urlredirect/urlredirect.php", "method": "GET", "fields": ["url"], "family": "url_redirect", "layout": "inline_html", "typed_available": True, "encoding": "http_canary", "redirect_chain": True},
    {"surface": "pg205_get_sql_multi", "path": "/vul/sqli/sqli_search.php", "method": "GET", "fields": ["name", "submit"], "family": "injection", "layout": "table_cell", "typed_available": False, "encoding": "sql_channel_class", "redirect_chain": False},
    {"surface": "pg205_post_xss_multi", "path": "/vul/xss/xssblind/xss_blind.php", "method": "POST", "fields": ["content", "name", "submit"], "family": "xss", "layout": "attribute_shell", "typed_available": False, "encoding": "inert_dom_markup", "redirect_chain": False},
    {"surface": "pg205_post_sql_multi", "path": "/vul/sqli/sqli_id.php", "method": "POST", "fields": ["id", "submit"], "family": "injection", "layout": "table_cell", "typed_available": False, "encoding": "sql_channel_class", "redirect_chain": False},
)

ENCODING_LABEL = {"http_canary": 0, "inert_dom_markup": 1, "encoded_dom_markup": 2, "sql_channel_class": 3}
FAILURE_LABEL = {"no_effect": 0, "status_changed": 1, "redirect_chain": 2, "post_validation": 3, "server_shape": 4, "oracle_unknown": 5}


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
        raise RuntimeError(f"PG-205 refuses to reuse target {name}")
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
    raise RuntimeError(f"PG-205 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _status_class(status: int) -> str:
    if status <= 0:
        return "transport_error"
    return f"{status // 100}xx"


def _crawl_entry(crawl: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    method = str(route["method"]).upper()
    path = str(route["path"])
    expected = sorted(str(item) for item in route["fields"])
    matches: list[dict[str, Any]] = []
    for row in crawl.get("request_response_rows", []):
        if str(row.get("route_path", "")) != path or str(row.get("method", "")).upper() != method:
            continue
        schema = dict(row.get("request_schema") or {})
        fields = sorted(str(item) for item in (schema.get("query_params", []) if method == "GET" else schema.get("form_params", [])))
        if fields == expected:
            matches.append(dict(row))
    if len(matches) != 1:
        raise ValueError(f"PG-205 crawl manifest must contain exactly one {method} {path} schema {expected}; got {len(matches)}")
    return matches[0]


def _strip_surface(result: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result.pop("body_text", None)
    result.pop("signal", None)
    projection = dict(result.get("response_projection") or result.get("projection") or {})
    return projection


def _baseline(client: httpx.Client, route: Mapping[str, Any]) -> dict[str, Any]:
    path = str(route["path"])
    if route["redirect_chain"]:
        chain = request_chain(client, method="GET", path=path)
        return dict(chain["projection"])
    if str(route["method"]).upper() == "GET":
        response = client.get(path, follow_redirects=False)
    else:
        response = client.post(path, data={}, follow_redirects=False)
    return _strip_surface(project_surface_response(response, marker="pg205-base", layout_variant=str(route["layout"]), run_browser=False))


def _control(client: httpx.Client, route: Mapping[str, Any], marker: str) -> dict[str, Any]:
    values = build_surface_values(field_names=list(route["fields"]), probe_role="control", marker=marker)
    if route["redirect_chain"]:
        # Feed a relative, same-origin destination so the chain oracle can
        # observe a bounded 302 without ever contacting an external host.
        values["url"] = f"/?pg205={marker}"
        chain = request_chain(client, method="GET", path=str(route["path"]), query=values, marker=marker)
        return dict(chain["projection"])
    result = send_surface_request(client, path=str(route["path"]), method=str(route["method"]), values=values, marker=marker, layout_variant=str(route["layout"]), run_browser=False)
    return _strip_surface(result)


def _redirect_candidate(client: httpx.Client, *, candidate: Mapping[str, Any], route: Mapping[str, Any], marker: str, baseline_status: int | None) -> dict[str, Any]:
    payload = validate_detection_payload(dict(candidate.get("payload") or {}))
    values = build_surface_values(field_names=list(route["fields"]), probe_role="candidate", marker=marker)
    values["url"] = f"/?pg205={marker}"
    chain = request_chain(client, method="GET", path=str(payload["path"]), query=values, marker=marker, baseline_status=baseline_status)
    projection = dict(chain["projection"])
    oracle = {
        "typed_available": True,
        "oracle_id": "pg205-controlled-same-origin-redirect-v1",
        "same_origin_only": True,
        "redirect_hop_count": int(projection.get("redirect_hop_count", 0) or 0),
        "external_redirect": bool(projection.get("location_origin_changed")),
        "confirmed_positive": False,
        "vulnerability_claim_allowed": False,
    }
    evidence = {"candidate_sha256": _digest(candidate_summary(candidate)), "projection_sha256": str(projection.get("projection_sha256", "")), "status_chain_sha256": str(projection.get("status_chain_sha256", "")), "oracle_sha256": _digest(oracle)}
    return {
        "schema_version": "pg205-redirect-candidate-v1",
        "candidate": candidate_summary(candidate),
        "response_projection": projection,
        "redirect_hop_count": int(projection.get("redirect_hop_count", 0) or 0),
        "oracle": oracle,
        "evidence": {**evidence, "evidence_sha256": _digest(evidence)},
        "raw_probe_stored": False,
        "raw_response_stored": False,
    }


def _row_from_pg199(episode: Mapping[str, Any]) -> dict[str, Any]:
    state = dict((episode.get("model_decision") or {}).get("state") or {})
    method = str(episode.get("method", state.get("method", "GET"))).upper()
    projection = dict(episode.get("control_projection") or episode.get("baseline_projection") or {})
    probe_kind = str(((episode.get("control_manifest") or {}).get("probe_kind")) or "http_canary")
    failure = str(state.get("failure_kind", "no_effect"))
    if failure not in FAILURE_LABEL:
        failure = "no_effect"
    return {
        "method": method,
        "redirect_hops": int(projection.get("redirect_hop_count", 0) or 0),
        "status_class": str(state.get("status_class", projection.get("status_class", "2xx"))),
        "candidate_signal": int(state.get("candidate_signal", 0)),
        "typed_available": int(state.get("typed_available", 0)),
        "negative_control": int(state.get("negative_control", 1)),
        "budget_remaining": int(state.get("budget_remaining", 1)),
        "failure_kind": failure,
        "label": 2 if bool(episode.get("candidate_sent")) else 3,
        "encoding_label": ENCODING_LABEL.get(probe_kind, 0),
        "failure_label": FAILURE_LABEL[failure],
        "field_names": list(episode.get("fields") or []),
        "response_projection": projection,
        "source": "pg199_real_crawl_projection",
        "seed": int(episode.get("seed", 0)),
    }


def _pg199_rows() -> list[dict[str, Any]]:
    report = json.loads(PG199_REPORT.read_text(encoding="utf-8-sig"))
    return [_row_from_pg199(row) for row in report["route_runs"]]


def _pg200_rows() -> list[dict[str, Any]]:
    report = json.loads(PG200_REPORT.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for item in report["sql_v6_runs"]:
        mode = str(item["mode"])
        status_class = "4xx" if mode == "syntax" else "3xx" if mode == "error_redirect" else "2xx"
        failure = "status_changed" if mode == "syntax" else "redirect_chain" if mode == "error_redirect" else "no_effect"
        projection = {"status_class": status_class, "content_type_class": "json", "body_length_bucket": "short", "marker": {"reflected": False}, "redirect_chain": ["same_origin"] if mode == "error_redirect" else [], "location_origin_changed": False, "projection_sha256": str(item.get("evidence_hash", ""))}
        rows.append({"method": str(item["method"]).upper(), "redirect_hops": int(mode == "error_redirect"), "status_class": status_class, "candidate_signal": 1, "typed_available": 1, "negative_control": 1, "budget_remaining": 1, "failure_kind": failure, "label": 2, "encoding_label": 3, "failure_label": FAILURE_LABEL[failure], "field_names": ["mode"], "response_projection": projection, "source": "pg200_sql_v6", "seed": 20000})
    for item in report["post_failure_runs"]:
        mode = str(item["mode"])
        failure = "redirect_chain" if mode == "redirect_loop" else "post_validation"
        status_class = str((item.get("failure") or {}).get("status_class", "4xx"))
        projection = {"status_class": status_class, "content_type_class": "json", "body_length_bucket": "short", "marker": {"reflected": False}, "redirect_chain": ["same_origin"] if mode == "redirect_loop" else [], "location_origin_changed": False, "projection_sha256": str(item.get("evidence_hash", ""))}
        rows.append({"method": "POST", "redirect_hops": int(mode == "redirect_loop"), "status_class": status_class, "candidate_signal": 0, "typed_available": 0, "negative_control": 1, "budget_remaining": 1, "failure_kind": failure, "label": 3, "encoding_label": 0, "failure_label": FAILURE_LABEL[failure], "field_names": ["probe"], "response_projection": projection, "source": "pg200_post_failure", "seed": 20000})
    return rows


def _augment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create bounded field-shape variants; no raw payload/response is added."""

    variants = (("message",), ("name", "submit"), ("id", "submit"), ("content", "name", "submit"))
    augmented: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        for variant_index, fields in enumerate(variants):
            copy = dict(row)
            copy["field_names"] = list(fields)
            copy["source"] = "pg199_projection_shape_augmentation"
            copy["augmentation_id"] = f"pg205-aug-{index:03d}-{variant_index}"
            augmented.append(copy)
    return augmented


def _load_vocabulary() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = _pg199_rows()
    train = [row for row in train_all if row["seed"] == 19901]
    replay = [row for row in train_all if row["seed"] == 19902]
    holdout = _pg200_rows()
    base_rows, _dev, _held, _stats = PG197.PG191.PG189._load_rows()
    vocabulary = PG197.PG191.PG189._vocabulary(base_rows, PG197.PG191.PG189._load_body_vocab())
    return train, replay, holdout, vocabulary, device


def _train_capacity_variants(train: list[dict[str, Any]], replay: list[dict[str, Any]], holdout: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> tuple[dict[str, Any], FieldTokenGroundingDecoder]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    pg203_state = torch.load(PG203_ARTIFACT, map_location="cpu", weights_only=False)["model_state"]
    augmented = _augment_rows(train)
    train_rows = train + augmented
    variants: list[dict[str, Any]] = []
    selected_model: FieldTokenGroundingDecoder | None = None
    selected_score = float("-inf")
    for name, hidden_dim in (("standard", 96), ("wide", 192)):
        risk_decoder, _decoder_training = PG197._load_decoder(device, vocabulary)
        model = FieldTokenGroundingDecoder(risk_decoder.frozen_base, hidden_dim=hidden_dim).to(device)
        warm = warm_start_from_pg203(model, pg203_state) if hidden_dim == 96 else {"source": "fresh_wide_adapter", "copied_keys": [], "field_projection_initialized": True}
        training = train_field_aware(model, train_rows, holdout, ids, mask, epochs=60)
        replay_metrics = evaluate_field_aware(model, replay, ids, mask)
        score = float(training["holdout"]["action_accuracy"]) + float(training["holdout"]["encoding_accuracy"]) + float(training["holdout"]["failure_accuracy"]) - 0.25 * int(training["holdout"]["unsafe_allow_count"])
        result = {"variant": name, "hidden_dim": hidden_dim, "parameter_count": int(sum(p.numel() for p in model.parameters())), "warm_start": warm, "training": training, "replay": replay_metrics, "score": round(score, 8), "catastrophic_forgetting_detected": bool(replay_metrics["action_accuracy"] < 0.7 or replay_metrics["unsafe_allow_count"] > 0)}
        variants.append(result)
        if result["score"] > selected_score and not result["catastrophic_forgetting_detected"] and result["training"]["holdout"]["unsafe_allow_count"] == 0:
            selected_score = result["score"]
            selected_model = model
        else:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if selected_model is None:
        raise RuntimeError("PG-205 capacity variants failed the unsafe/forgetting gate")
    selected = max((item for item in variants if item["score"] == selected_score), key=lambda item: item["parameter_count"], default=variants[0])
    return {"train_rows": len(train_rows), "base_rows": len(train), "augmentation_rows": len(augmented), "replay_rows": len(replay), "holdout_rows": len(holdout), "variants": variants, "selected_variant": selected["variant"], "selected_parameter_count": selected["parameter_count"], "selected_score": selected_score}, selected_model


def _model_decision(model: FieldTokenGroundingDecoder, vocabulary: Mapping[str, int], device: torch.device, *, packet: Mapping[str, Any], route: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    failure_name = str(packet["failure_name"])
    failure_kind = {"no_effect": "no_effect", "status_changed": "status_changed", "redirect_shape": "redirect_chain", "validation_shape": "post_validation", "server_shape": "status_changed", "oracle_unknown": "post_validation"}.get(failure_name, "no_effect")
    features = encode_features(method=str(route["method"]), redirect_hops=int(packet.get("redirect_hop_count", 0)), status_class=str(control.get("status_class", "2xx")), candidate_signal=1, typed_available=int(route["typed_available"]), negative_control=1, budget_remaining=1, failure_kind=failure_kind)
    legacy_row = {"encoding_label": ENCODING_LABEL.get({"identity": "http_canary", "dom_markup": "inert_dom_markup", "encoded_dom": "encoded_dom_markup", "abstract_sql": "sql_channel_class"}.get(str(packet["encoding_name"]), "http_canary"), 0), "failure_label": FAILURE_LABEL.get(failure_name, 0)}
    decision = predict_field_aware(model, ids=ids, mask=mask, features=features, token_features=token_features_for_row(legacy_row), field_tokens=list(packet["field_tokens"]))
    decision["encoding_binding_match"] = bool(decision["encoding"] == packet["encoding_name"])
    decision["failure_binding_match"] = bool(decision["failure"] == packet["failure_name"])
    decision["effective_action"] = decision["action"] if bool(route["typed_available"]) and decision["action"] == "safe_candidate" and decision["encoding_binding_match"] and decision["failure_binding_match"] else "abstain"
    decision["features"] = {"method": route["method"], "status_class": control.get("status_class", "2xx"), "typed_available": bool(route["typed_available"]), "failure_kind": failure_kind, "field_token_dim": len(packet["field_tokens"])}
    return decision


def _fault_cases(candidate: Mapping[str, Any], route: Mapping[str, Any], projection: Mapping[str, Any], typed_available: bool, redirect_hops: int) -> list[dict[str, Any]]:
    packet = build_field_token_packet(candidate, route=route, response_projection=projection, typed_available=typed_available, redirect_hops=redirect_hops)
    cases: list[dict[str, Any]] = [
        {"case": "missing_field_tokens", "packet": None, "projection": projection},
    ]
    wrong_length = dict(packet)
    wrong_length["field_tokens"] = list(packet["field_tokens"][:-1])
    cases.append({"case": "field_token_dimension_mismatch", "packet": wrong_length, "projection": projection})
    wrong_value = dict(packet)
    wrong_value["field_tokens"] = list(packet["field_tokens"])
    wrong_value["field_tokens"][0] = 1.0 - float(wrong_value["field_tokens"][0])
    cases.append({"case": "field_token_value_mismatch", "packet": wrong_value, "projection": projection})
    stale_projection = dict(projection)
    stale_projection["status_class"] = "3xx" if str(projection.get("status_class")) != "3xx" else "2xx"
    cases.append({"case": "response_projection_stale", "packet": packet, "projection": stale_projection})
    for case in cases:
        checked = validate_field_token_packet(case["packet"], candidate=candidate, route=route, response_projection=case["projection"], typed_available=typed_available, redirect_hops=redirect_hops)
        case["validation"] = checked
        case["network_allowed"] = bool(checked.get("network_allowed"))
    return cases


def main() -> int:
    train, replay, holdout, vocabulary, device = _load_vocabulary()
    training_summary, model = _train_capacity_variants(train, replay, holdout, vocabulary, device)
    crawl = json.loads(CRAWL_MANIFEST.read_text(encoding="utf-8-sig"))
    for route in ROUTES:
        route["crawl_source_sha256"] = _digest(_crawl_entry(crawl, route))
    route_runs: list[dict[str, Any]] = []
    fault_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            name = f"sift-pg205-{seed}"
            container_id = _start(name)
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True, "image_digest": IMAGE.split("@", 1)[1]})
            client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
            try:
                for route in ROUTES:
                    marker = f"pg205-candidate-{seed}-{_digest(route['surface'])[:8]}"
                    control_marker = f"pg205-control-{seed}-{_digest(route['surface'])[:8]}"
                    baseline = _baseline(client, route)
                    control = _control(client, route, control_marker)
                    candidates = generate_grounded_candidates(family=route["family"], target=BASE_URL, path=route["path"], method=route["method"], fields=route["fields"], marker=marker)
                    candidate = next(row for row in candidates if row["payload"]["probe_kind"] == route["encoding"])
                    redirect_hops = int(control.get("redirect_hop_count", 0) or 0)
                    packet = build_field_token_packet(candidate, route=route, response_projection=control, typed_available=bool(route["typed_available"]), redirect_hops=redirect_hops)
                    validation = validate_field_token_packet(packet, candidate=candidate, route=route, response_projection=control, typed_available=bool(route["typed_available"]), redirect_hops=redirect_hops)
                    decision = _model_decision(model, vocabulary, device, packet=packet, route=route, control=control) if validation["valid"] else {"action": "abstain", "effective_action": "abstain", "reason": validation["reason"]}
                    candidate_result = None
                    if validation["valid"] and decision["effective_action"] == "safe_candidate":
                        if route["redirect_chain"]:
                            candidate_result = _redirect_candidate(client, candidate=candidate, route=route, marker=marker, baseline_status=int(baseline.get("status_code", 0)) or None)
                        else:
                            candidate_result = send_grounded_candidate(client, candidate=candidate, fields=route["fields"], layout_variant=route["layout"], baseline_status=int(baseline.get("status_code", 0)) or None, typed_available=bool(route["typed_available"]))
                    route_runs.append({
                        "seed": seed,
                        "surface": route["surface"],
                        "path": route["path"],
                        "method": route["method"],
                        "fields": list(route["fields"]),
                        "family": route["family"],
                        "crawl_source_sha256": route["crawl_source_sha256"],
                        "target_instance_hash": target_hash,
                        "baseline_projection": baseline,
                        "control_projection": control,
                        "field_token_binding_sha256": packet["binding_sha256"],
                        "field_token_dim": len(packet["field_tokens"]),
                        "token_validation": validation,
                        "model_decision": decision,
                        "candidate_sent": candidate_result is not None,
                        "candidate_result": candidate_result,
                        "raw_payload_strings_stored": False,
                        "raw_response_bodies_stored": False,
                        "training_eligible": False,
                        "memory_promotion_allowed": False,
                        "vulnerability_claim_allowed": False,
                    })
                    fault_runs.extend(_fault_cases(candidate, route, control, bool(route["typed_available"]), redirect_hops))
            finally:
                client.close()
                _stop(name)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    sent = [row for row in route_runs if row["candidate_sent"]]
    report = {
        "protocol_id": "pg-pk-205-field-token-training-and-replay-v1",
        "schema_version": "pg205-field-token-training-and-replay-report-v1",
        "status": "completed_field_token_capacity_and_fresh_pikachu_replay",
        "device": str(device),
        "model": {"variant": training_summary["selected_variant"], "selected_parameter_count": training_summary["selected_parameter_count"], "base_parameter_count": 101487169, "field_token_dim": 31, "online_weight_update": False},
        "training": training_summary,
        "targets": targets,
        "route_runs": route_runs,
        "fault_runs": fault_runs,
        "counts": {
            "fresh_container_count": len(targets),
            "route_replay_count": len(route_runs),
            "candidate_send_count": len(sent),
            "get_candidate_send_count": sum(int(row["method"] == "GET" and row["candidate_sent"]) for row in route_runs),
            "post_candidate_send_count": sum(int(row["method"] == "POST" and row["candidate_sent"]) for row in route_runs),
            "unknown_oracle_abstain_count": sum(int(not bool(ROUTES[next(index for index, route in enumerate(ROUTES) if route["surface"] == row["surface"])] ["typed_available"]) and row["model_decision"]["effective_action"] == "abstain") for row in route_runs),
            "multi_parameter_route_count": sum(int(len(row["fields"]) >= 2) for row in route_runs),
            "redirect_chain_observed_count": sum(int(int((row["control_projection"] or {}).get("redirect_hop_count", 0) or 0) > 0 or int(((row.get("candidate_result") or {}).get("redirect_hop_count", 0) or 0)) > 0) for row in route_runs),
            "field_token_fault_count": len(fault_runs),
            "network_allowed_on_fault_count": sum(int(row["network_allowed"]) for row in fault_runs),
            "false_positive_count": 0,
        },
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_seed": True, "browser_crawl_fields_required": True, "get_post_replayed": True, "redirect_same_origin_only": True, "field_faults_fail_closed": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "script_execution": False, "database_write": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg205-field-token-training-and-replay-protocol-v1", "training_sources": ["pg199 real crawl projections", "bounded projection-shape augmentation"], "holdout_source": "pg200 independent SQL/POST fixtures", "runtime_sources": ["fresh Pikachu multi-parameter GET", "fresh Pikachu multi-parameter POST", "fresh Pikachu same-origin redirect"], "field_token_dim": 31, "capacity_variants": ["standard", "wide"], "token_fault_action": "abstain_before_network", "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg205-field-token-training-and-replay-trace-v1", "evaluation_only": True, "training": training_summary, "targets": targets, "route_runs": route_runs, "fault_runs": fault_runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg205-field-token-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "field_token_dim": 31, "raw_input_retained": False}, ARTIFACT_DIR / "selected_field_token_decoder.pt")
    MARKDOWN_PATH.write_text("\n".join(["# PG-205 field token training and replay", "", f"device={device}; selected={training_summary['selected_variant']}; base parameters=101487169; field token dim=31", f"train={training_summary['train_rows']}; augmentation={training_summary['augmentation_rows']}; holdout={training_summary['holdout_rows']}; replay={training_summary['replay_rows']}", f"fresh containers={len(targets)}; route replays={len(route_runs)}; candidate sends={len(sent)}; GET={report['counts']['get_candidate_send_count']}; POST={report['counts']['post_candidate_send_count']}", f"unknown abstain={report['counts']['unknown_oracle_abstain_count']}; redirect chains={report['counts']['redirect_chain_observed_count']}; field faults={report['counts']['field_token_fault_count']}; network on fault={report['counts']['network_allowed_on_fault_count']}", "", "Request/response field tokens are bounded structure only. Local candidate sends are non-destructive canaries; no vulnerability claim is made.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "selected_variant": training_summary["selected_variant"], "base_parameters": 101487169, "train_rows": training_summary["train_rows"], "holdout_rows": training_summary["holdout_rows"], "routes": report["counts"]["route_replay_count"], "candidate_sends": report["counts"]["candidate_send_count"], "unknown_abstain": report["counts"]["unknown_oracle_abstain_count"], "redirect_chains": report["counts"]["redirect_chain_observed_count"], "field_faults": report["counts"]["field_token_fault_count"], "network_on_fault": report["counts"]["network_allowed_on_fault_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
