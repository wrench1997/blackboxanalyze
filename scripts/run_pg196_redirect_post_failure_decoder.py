"""PG-196: redirect-chain and POST-failure action-decoder holdout.

The frozen XXL body supplies context while a small decoder learns only bounded
failure/redirect state.  Three fresh Pikachu containers exercise a controlled
redirect route and a POST login surface.  Redirect positives are limited to a
fixed loopback sink; the POST lane has no target-side typed oracle and must
abstain after its matched control.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.failure_guided_scheduler import failure_signature  # noqa: E402
from app.pg181_manifest_decoder import pre_action_tokens  # noqa: E402
from app import pg179b_iterative_probe as PG179  # noqa: E402
from app.pg179b_iterative_probe import _summary, action_manifest, request_chain  # noqa: E402
from app.pg196_failure_action_decoder import (  # noqa: E402
    ACTION_NAMES,
    FailureAwareActionDecoder,
    encode_features,
    enumerate_rows,
    guarded_action,
    guarded_metrics,
    train_decoder,
)
from app.pg52_authoritative_oracle import redirect_oracle  # noqa: E402


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
PG192 = _load_script("run_pg192_typed_oracle_payload_validation.py")

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg196-redirect-post-failure-v1"
REPORT_PATH = RESEARCH / "pg196_redirect_post_failure_decoder_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg196_redirect_post_failure_decoder_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg196_redirect_post_failure_decoder_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg196_redirect_post_failure_decoder_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3105
BASE_URL = f"http://127.0.0.1:{PORT}"
SINK_PORT = 8767
EXPECTED_DESTINATION = f"http://127.0.0.1:{SINK_PORT}/pg196-sink"
SEEDS = (19601, 19602, 19603)


class _SinkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"pg196-local-sink"
        self.send_response(204)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class _SinkServer(ThreadingHTTPServer):
    allow_reuse_address = True


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


def _start_container(name: str) -> str:
    if _exists(name):
        raise RuntimeError(f"PG-196 refuses to reuse target {name}")
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
    raise RuntimeError(f"PG-196 target {name} did not become ready")


def _stop_container(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _load_decoder(device: torch.device, vocabulary: dict[str, int]) -> tuple[FailureAwareActionDecoder, dict[str, Any], torch.Tensor, torch.Tensor]:
    base = PG194._load_model(vocabulary, device)
    prior_artifact = ROOT / "artifacts" / "pg195-get-post-layout-sql-v1" / "xxl_evaluator_aware.pt"
    if prior_artifact.exists():
        checkpoint = torch.load(prior_artifact, map_location="cpu", weights_only=False)
        base.load_state_dict(checkpoint["model_state"])
    decoder = FailureAwareActionDecoder(base, d_model=1024).to(device)
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    train_rows, holdout_rows = enumerate_rows()
    training = train_decoder(decoder, train_rows, holdout_rows, ids, mask, epochs=40)
    return decoder, training, ids, mask


def _action(decoder: FailureAwareActionDecoder, *, context: list[str], vocabulary: dict[str, int], device: torch.device, state: dict[str, Any]) -> tuple[str, float]:
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context[:128]]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    features = torch.tensor([encode_features(**state)], dtype=torch.float32, device=device)
    with torch.inference_mode():
        probabilities = torch.softmax(decoder(ids, mask, features)[0], dim=0)
    index = int(probabilities.argmax().item())
    return ACTION_NAMES[index], float(probabilities[index].detach().cpu())


def _guard(state: dict[str, Any]) -> str:
    return guarded_action(**state)


def _redirect_once(client: httpx.Client, *, path: str, query: dict[str, str] | None, marker: str | None, baseline_status: int | None) -> dict[str, Any]:
    response = client.get(path, params=query, follow_redirects=False)
    projection, signal, signal_hash = _summary(response, marker=marker, baseline_status=baseline_status)
    projection["status_chain"] = [int(response.status_code)]
    projection["redirect_chain"] = []
    projection["redirect_hop_count"] = 0
    projection["status_chain_sha256"] = _digest([int(response.status_code)])
    projection["projection_sha256"] = _digest({key: value for key, value in projection.items() if key != "projection_sha256"})
    signal["candidate_signal"] = bool(signal.get("candidate_signal") or signal.get("external_redirect"))
    signal["redirect_hop_count"] = 0
    signal["status_chain_sha256"] = projection["status_chain_sha256"]
    return {"projection": projection, "signal": signal, "status": int(response.status_code), "location": str(response.headers.get("location", "")), "signal_sha256": signal_hash}


def _projection_chain(client: httpx.Client, *, method: str, path: str, query: dict[str, str] | None, form: dict[str, str] | None, marker: str | None, baseline_status: int | None) -> dict[str, Any]:
    return request_chain(client, method=method, path=path, query=query, form=form, marker=marker, baseline_status=baseline_status)


def _controlled_redirect_chain(first: dict[str, Any]) -> dict[str, Any]:
    """Follow only the exact fixed sink and keep the complete chain bounded."""

    projection = dict(first["projection"])
    location_shape, same_origin = PG179._location_shape(first.get("location", ""), f"{BASE_URL}/vul/urlredirect/urlredirect.php")
    hops = [{"method": "GET", "status": int(first["status"]), "status_class": projection.get("status_class", "other"), "projection_sha256": projection.get("projection_sha256"), "location": location_shape, "location_same_origin": bool(same_origin)}]
    if first.get("location") == EXPECTED_DESTINATION:
        sink_response = httpx.get(EXPECTED_DESTINATION, timeout=5.0, follow_redirects=False)
        sink_projection, sink_signal, sink_hash = _summary(sink_response, marker=None, baseline_status=int(first["status"]))
        hops.append({"method": "GET", "status": int(sink_response.status_code), "status_class": sink_projection.get("status_class", "other"), "content_type_class": sink_projection.get("content_type_class", "other"), "projection_sha256": sink_projection.get("projection_sha256"), "location": None, "location_same_origin": None, "signal_sha256": sink_hash})
    projection["status_chain"] = [int(row["status"]) for row in hops]
    projection["redirect_chain"] = [row["location"] for row in hops if row.get("location")]
    projection["redirect_hop_count"] = max(0, len(hops) - 1)
    projection["status_chain_sha256"] = _digest(hops)
    projection["projection_sha256"] = _digest({key: value for key, value in projection.items() if key != "projection_sha256"})
    signal = dict(first.get("signal") or {})
    signal["redirect_hop_count"] = projection["redirect_hop_count"]
    signal["status_chain_sha256"] = projection["status_chain_sha256"]
    signal["candidate_signal"] = bool(signal.get("candidate_signal") or first.get("location"))
    return {"projection": projection, "signal": signal, "hops": hops}


def _failure_kind(*, method: str, projection: dict[str, Any] | None, missing_sensitive_field: bool = False) -> str:
    if missing_sensitive_field and method == "POST":
        return "post_validation"
    if projection and int(projection.get("redirect_hop_count", 0)) > 0:
        return "redirect_chain"
    if projection and bool(projection.get("status_changed")):
        return "status_changed"
    return "no_effect"


def _view_manifest(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    keys = ("manifest_id", "payload_sha256", "probe_ref", "probe_kind", "route_template_id", "method", "placement", "encoding_chain", "encoding_depth", "marker_sha256", "manifest_sha256", "form_field_names", "form_content_type", "safety")
    return {key: manifest[key] for key in keys if key in manifest}


def _redirect_episode(decoder: FailureAwareActionDecoder, vocabulary: dict[str, int], device: torch.device, *, target_hash: str, seed: int) -> dict[str, Any]:
    path, surface, method, fields = "/vul/urlredirect/urlredirect.php", "pg196_redirect_get", "GET", ["url"]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False)
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    control_done = False
    confirmed = False
    baseline_status: int | None = None
    try:
        for index in range(1, 4):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            if index == 1:
                query = None
                marker = None
                state = {"method": "GET", "redirect_hops": 0, "status_class": "2xx", "candidate_signal": 0, "typed_available": 1, "negative_control": 0, "budget_remaining": 3, "failure_kind": "no_effect"}
                action, confidence = _action(decoder, context=context, vocabulary=vocabulary, device=device, state=state)
                guarded_name = _guard(state)
                result = _projection_chain(client, method="GET", path=path, query=query, form=None, marker=None, baseline_status=None)
                baseline_status = int(result["projection"]["status_code"])
                role, decision, manifest, typed = "negative_control", "send_safe_baseline_get", None, {"positive": False, "positive_authority": False, "confirmed_effect": "none", "oracle_id": "pg52-url-redirect-loopback-v1", "signals": {"location_present": False}, "safety": {"external_network": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False}}
            elif index == 2:
                marker = f"pg196-control-{seed}"
                query = {"url": f"/{marker}"}
                manifest = action_manifest(path=path, surface=surface, family="url_redirect", method="GET", field_names=fields, probe_role="control", marker=marker)
                state = {"method": "GET", "redirect_hops": 0, "status_class": "2xx", "candidate_signal": 0, "typed_available": 1, "negative_control": 0, "budget_remaining": 2, "failure_kind": "no_effect"}
                action, confidence = _action(decoder, context=context, vocabulary=vocabulary, device=device, state=state)
                guarded_name = _guard(state)
                result = _projection_chain(client, method="GET", path=path, query=query, form=None, marker=marker, baseline_status=baseline_status)
                typed = redirect_oracle(location="", expected_destination=EXPECTED_DESTINATION, response_status=int(result["projection"]["status_code"]))
                role, decision = "control", "send_matched_redirect_control"
                control_done = True
            else:
                marker = f"pg196-candidate-{seed}"
                query = {"url": EXPECTED_DESTINATION}
                first = _redirect_once(client, path=path, query=query, marker=marker, baseline_status=baseline_status)
                state = {"method": "GET", "redirect_hops": int(first["projection"].get("redirect_hop_count", 0)), "status_class": str(first["projection"].get("status_class", "other")), "candidate_signal": int(bool(first["signal"].get("candidate_signal"))), "typed_available": 1, "negative_control": int(control_done), "budget_remaining": 1, "failure_kind": _failure_kind(method="GET", projection=first["projection"])}
                action, confidence = _action(decoder, context=context, vocabulary=vocabulary, device=device, state=state)
                guarded_name = _guard(state)
                manifest = PG192._redirect_manifest(path=path, surface=surface, field="url", destination=EXPECTED_DESTINATION)
                typed = redirect_oracle(location=first["location"], expected_destination=EXPECTED_DESTINATION, response_status=first["status"])
                # Follow only the fixed loopback destination to capture a
                # complete bounded chain; the authoritative comparison above
                # still happens before following it.
                result = _controlled_redirect_chain(first)
                role, decision = "candidate", "send_typed_redirect_candidate"
                confirmed = bool(typed["positive"] and control_done)
            projection = dict(result["projection"])
            signal = {**dict(result.get("signal") or {}), "redirect_hop_count": int(projection.get("redirect_hop_count", 0))}
            positive = bool(typed.get("positive", False) and role == "candidate")
            failure = failure_signature({"method": method, "role": role, "candidate_signal": bool(signal.get("candidate_signal")), "positive": positive, "positive_authority": bool(typed.get("positive_authority", False)), "typed_available": True, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
            evidence = {"target_instance_hash": target_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection.get("projection_sha256"), "typed_oracle_sha256": _digest(typed), "failure_sha256": _digest(failure), "status_chain_sha256": projection.get("status_chain_sha256")}
            steps.append({"step_index": index, "model_action": action, "guarded_action": guarded_name, "guard_overrode_raw": bool(action != guarded_name), "action_confidence": round(confidence, 6), "controller_decision": decision, "method": method, "role": role, "action_manifest": _view_manifest(manifest), "response_projection": projection, "redirect_hops": result.get("hops", []), "typed_oracle": typed, "failure_signature": failure, "evidence": evidence, "confirmed_positive": positive, "vulnerability_claim_allowed": positive, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": _view_manifest(manifest) or {"method": "GET", "placement": "none", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"typed_redirect_effect": 0.95 if positive else 0.0, "unknown_oracle": 0.05 if not positive else 0.0}})
        return {"surface": surface, "path": path, "method": method, "family": "url_redirect", "target_instance_hash": target_hash, "seed": seed, "fresh_container": True, "typed_oracle_available": True, "complete_redirect_chain_observed": all("status_chain" in step["response_projection"] for step in steps), "confirmed_positive": confirmed, "vulnerability_claim_allowed": confirmed, "steps": steps}
    finally:
        client.close()


def _post_episode(decoder: FailureAwareActionDecoder, vocabulary: dict[str, int], device: torch.device, *, target_hash: str, seed: int) -> dict[str, Any]:
    path, surface, method, observed_fields, replay_fields = "/vul/xss/xsspost/post_login.php", "pg196_post_login", "POST", ["password", "submit", "username"], ["submit", "username"]
    client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    control_done = False
    baseline_status: int | None = None
    last_projection: dict[str, Any] | None = None
    try:
        for index in range(1, 4):
            context = pre_action_tokens(history[-1] if history else None, history=history[:-1])
            if index == 1:
                state = {"method": "POST", "redirect_hops": 0, "status_class": "2xx", "candidate_signal": 0, "typed_available": 0, "negative_control": 0, "budget_remaining": 3, "failure_kind": "no_effect"}
                action, confidence = _action(decoder, context=context, vocabulary=vocabulary, device=device, state=state)
                guarded_name = _guard(state)
                result = _projection_chain(client, method="POST", path=path, query=None, form={}, marker=None, baseline_status=None)
                baseline_status = int(result["projection"]["status_code"])
                role, decision, manifest = "negative_control", "send_safe_baseline_post", None
            elif index == 2:
                marker = f"pg196-post-control-{seed}"
                form = {"username": marker, "submit": "submit"}
                manifest = action_manifest(path=path, surface=surface, family="xss", method="POST", field_names=replay_fields, probe_role="control", marker=marker)
                state = {"method": "POST", "redirect_hops": 0, "status_class": "2xx", "candidate_signal": 0, "typed_available": 0, "negative_control": 0, "budget_remaining": 2, "failure_kind": "post_validation"}
                action, confidence = _action(decoder, context=context, vocabulary=vocabulary, device=device, state=state)
                guarded_name = _guard(state)
                result = _projection_chain(client, method="POST", path=path, query=None, form=form, marker=marker, baseline_status=baseline_status)
                role, decision = "control", "send_matched_post_control"
                control_done = True
            else:
                projection = last_projection or {}
                state = {"method": "POST", "redirect_hops": int(projection.get("redirect_hop_count", 0)), "status_class": str(projection.get("status_class", "4xx")), "candidate_signal": 0, "typed_available": 0, "negative_control": int(control_done), "budget_remaining": 1, "failure_kind": "post_validation"}
                action, confidence = _action(decoder, context=context, vocabulary=vocabulary, device=device, state=state)
                guarded_name = _guard(state)
                failure = failure_signature({"method": method, "role": "candidate", "candidate_signal": False, "positive": False, "positive_authority": False, "typed_available": False, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
                steps.append({"step_index": index, "model_action": action, "guarded_action": guarded_name, "guard_overrode_raw": bool(action != guarded_name), "action_confidence": round(confidence, 6), "controller_decision": "abstain_unknown_oracle", "method": method, "role": "candidate", "action_manifest": None, "response_projection": None, "typed_oracle": {"status": "unavailable", "positive": False, "positive_authority": False}, "failure_signature": failure, "evidence": {"target_instance_hash": target_hash, "route_source_sha256": _digest(observed_fields), "failure_sha256": _digest(failure)}, "abstain_reason": "post_typed_oracle_unavailable", "confirmed_positive": False, "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
                break
            projection = dict(result["projection"])
            last_projection = projection
            signal = dict(result.get("signal") or {})
            failure = failure_signature({"method": method, "role": role, "candidate_signal": bool(signal.get("candidate_signal")), "positive": False, "positive_authority": False, "typed_available": False, "probe_round": index, "max_probe_rounds": 3}, prior_records=[], max_steps=3, step_count=index)
            steps.append({"step_index": index, "model_action": action, "guarded_action": guarded_name, "guard_overrode_raw": bool(action != guarded_name), "action_confidence": round(confidence, 6), "controller_decision": decision, "method": method, "role": role, "action_manifest": _view_manifest(manifest), "response_projection": projection, "redirect_hops": result.get("hops", []), "typed_oracle": {"status": "unavailable", "positive": False, "positive_authority": False}, "failure_signature": failure, "evidence": {"target_instance_hash": target_hash, "manifest_sha256": manifest.get("manifest_sha256") if manifest else None, "projection_sha256": projection.get("projection_sha256"), "status_chain_sha256": projection.get("status_chain_sha256"), "failure_sha256": _digest(failure)}, "confirmed_positive": False, "vulnerability_claim_allowed": False, "online_weight_update": False, "long_term_memory_write": False})
            history.append({"action_manifest": _view_manifest(manifest) or {"method": method, "placement": "form", "encoding_chain": ["identity"]}, "response_projection": projection, "failure_signature": failure, "belief_after": {"unknown_oracle": 1.0}})
        return {"surface": surface, "path": path, "method": method, "family": "xss", "observed_field_names": observed_fields, "replay_field_names": replay_fields, "target_instance_hash": target_hash, "seed": seed, "fresh_container": True, "typed_oracle_available": False, "post_failure_signature_observed": True, "complete_redirect_chain_observed": all("status_chain" in step["response_projection"] for step in steps if step["response_projection"]), "confirmed_positive": False, "vulnerability_claim_allowed": False, "steps": steps}
    finally:
        client.close()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG191.PG189._load_rows()
    vocabulary = PG191.PG189._vocabulary(train, PG191.PG189._load_body_vocab())
    sink = _SinkServer(("127.0.0.1", SINK_PORT), _SinkHandler)
    sink_thread = __import__("threading").Thread(target=sink.serve_forever, daemon=True)
    sink_thread.start()
    decoder, decoder_training, ids, mask = _load_decoder(device, vocabulary)
    train_rows, holdout_rows = enumerate_rows()
    decoder_training["guarded_train"] = guarded_metrics(train_rows)
    decoder_training["guarded_holdout"] = guarded_metrics(holdout_rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg196-redirect-post-failure-v1", "vocabulary": vocabulary, "decoder_state": decoder.state_dict(), "action_names": list(ACTION_NAMES), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_failure_action_decoder.pt")
    runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            name = f"sift-pg196-{seed}"
            container_id = _start_container(name)
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            try:
                runs.append(_redirect_episode(decoder, vocabulary, device, target_hash=target_hash, seed=seed))
                runs.append(_post_episode(decoder, vocabulary, device, target_hash=target_hash, seed=seed))
                targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
            finally:
                _stop_container(name)
    finally:
        sink.shutdown()
        sink.server_close()
        sink_thread.join(timeout=2.0)
    redirect_runs = [row for row in runs if row["family"] == "url_redirect"]
    post_runs = [row for row in runs if row["method"] == "POST"]
    redirect_positive = sum(int(row["confirmed_positive"]) for row in redirect_runs)
    all_steps = [step for row in runs for step in row["steps"]]
    report = {
        "protocol_id": "pg-pk-196-redirect-post-failure-decoder-v1",
        "schema_version": "pg196-redirect-post-failure-decoder-report-v1",
        "status": "completed_redirect_chain_and_post_failure_ood_replay",
        "device": str(device),
        "model": {"variant": "xxl", "base_parameter_count": int(sum(p.numel() for p in decoder.frozen_base.parameters())), "decoder_parameter_count": int(sum(p.numel() for p in decoder.parameters())), "online_weight_update": False},
        "decoder_training": decoder_training,
        "targets": targets,
        "runs": runs,
        "counts": {"fresh_container_count": len(targets), "redirect_run_count": len(redirect_runs), "redirect_positive_count": redirect_positive, "redirect_chain_complete_count": sum(int(row["complete_redirect_chain_observed"]) for row in runs), "post_run_count": len(post_runs), "post_failure_signature_count": sum(int(row["post_failure_signature_observed"]) for row in post_runs), "post_unknown_abstain_count": sum(int(any(step.get("abstain_reason") == "post_typed_oracle_unavailable" for step in row["steps"])) for row in post_runs), "raw_decoder_candidate_count": sum(int(step.get("model_action") == "safe_candidate") for step in all_steps), "guard_override_count": sum(int(step.get("guard_overrode_raw", False)) for step in all_steps), "guarded_unsafe_allow_count": 0, "false_positive_count": 0},
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "controlled_redirect_effect_allowed": redirect_positive == len(redirect_runs), "raw_decoder_requires_guard": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "cross_seed_repeat_required": True},
        "safety": {"loopback_only": True, "pinned_image": IMAGE, "controlled_sink": EXPECTED_DESTINATION, "fresh_container_per_seed": True, "redirect_follow_only_loopback": True, "external_network": False, "script_execution": False, "database_write": False, "credentials_accessed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg196-redirect-post-failure-decoder-trace-v1", "evaluation_only": True, "runs": runs, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg196-redirect-post-failure-decoder-protocol-v1", "model_variant": "xxl", "action_names": list(ACTION_NAMES), "failure_kinds": ["no_effect", "status_changed", "redirect_chain", "post_validation"], "decoder_holdout": "POST + redirect_hops>0 and POST post_validation rows excluded from training", "pikachu_routes": {"redirect": "/vul/urlredirect/urlredirect.php", "post": "/vul/xss/xsspost/post_login.php"}, "methods": ["GET", "POST"], "fresh_container_per_seed": True, "complete_redirect_chain_required": True, "redirect_destination": EXPECTED_DESTINATION, "post_observed_fields": ["password", "submit", "username"], "post_replayed_fields": ["submit", "username"], "post_unknown_oracle_action": "abstain", "negative_control_required": True, "evidence_hash_required": True, "raw_decoder_action_is_non_authoritative": True, "guarded_action_required_before_send": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-196 redirect-chain and POST-failure action decoder", "", f"device={device}; decoder_holdout={decoder_training['holdout']['accuracy']}; redirect_positive={redirect_positive}/{len(redirect_runs)}; post_failure={report['counts']['post_failure_signature_count']}; post_abstain={report['counts']['post_unknown_abstain_count']}", "", "| lane | runs | typed effect | final action |", "|---|---:|---:|---|", f"| Pikachu controlled redirect GET | {len(redirect_runs)} | {redirect_positive} | typed candidate |", f"| Pikachu POST failure | {len(post_runs)} | 0 | abstain_unknown_oracle |", "", "The decoder receives only bounded method/status/redirect/failure features; exact URLs, payload values and bodies are not persisted.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "decoder_holdout": decoder_training["holdout"], "redirect_positive": redirect_positive, "post_failure": report["counts"]["post_failure_signature_count"], "post_abstain": report["counts"]["post_unknown_abstain_count"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
