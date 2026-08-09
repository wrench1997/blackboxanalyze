"""PG-212: AI/reference SQL response-shape loop on fresh Pikachu containers.

The route catalog contains real GET and POST SQL surfaces, but the pinned
Pikachu image currently serves a database-configuration failure page.  This
runner records that fact instead of converting it into a SQL finding.  The AI
is still put in the send path: it receives the route/control packet and may
authorize an abstract candidate only when the evaluator says a backend is
available.  A deterministic syntax-shape reference is sent independently for
comparison.  No time-delay, write, comment, or external-network probe is
allowed, and raw request/response strings are not persisted.
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
from app.pg195_request_surface_adapter import build_surface_values  # noqa: E402
from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet  # noqa: E402
from app.pg212_sql_response_oracle import build_sql_probe_values, compare_sql_shapes, project_sql_response  # noqa: E402
from app.maze_engine import sha256_json  # noqa: E402


RESEARCH = ROOT / "research"
CRAWL_PATH = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
REPORT_PATH = RESEARCH / "pg212_pikachu_sql_response_shape_loop_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg212_pikachu_sql_response_shape_loop_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg212_pikachu_sql_response_shape_loop_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg212_pikachu_sql_response_shape_loop_report_v1.md"
SEEDS = (21201, 21202)
BASE_PORT = 3515


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(seed: int, run_index: int) -> tuple[str, int, str, dict[str, Any]]:
    """Create one disposable target for one route episode.

    No volume or bind mount is supplied.  That is stronger than a plain
    ``docker restart``: the writable layer and any in-container database are
    discarded before the next route.  Restart remains an allowed health
    recovery action, but a restarted container is never accepted as a clean
    reset unless its fresh-container attestation is re-established.
    """

    name = f"sift-pg212-{seed}-{run_index}"
    if _exists(name):
        raise RuntimeError(f"PG-212 refuses to reuse target {name}")
    port = BASE_PORT + int(run_index)
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--label", "sift.pg212=true", "--label", f"sift.pg212.reset_epoch={seed}-{run_index}", "--publish", f"127.0.0.1:{port}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 150.0
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                container_id = _docker("inspect", "--format", "{{.Id}}", name)
                mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
                image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
                if mounts:
                    raise RuntimeError("PG-212 clean reset requires zero mounts/volumes")
                if image_ref != IMAGE:
                    raise RuntimeError("PG-212 image digest attestation mismatch")
                reset = {
                    "reset_id": f"pg212-reset-{seed}-{run_index}",
                    "reset_epoch": f"{seed}-{run_index}",
                    "fresh_target": True,
                    "completed": True,
                    "container_recreated": True,
                    "container_restart_used": False,
                    "container_id_sha256": hashlib.sha256(container_id.encode("utf-8")).hexdigest(),
                    "image": image_ref,
                    "volume_mount_count": len(mounts),
                    "database_clean_contract": "fresh_writable_layer_no_volume_no_stateful_probe",
                    "state_change_allowed": False,
                    "evaluator_state_hidden": True,
                    "external_network": False,
                }
                return name, port, container_id, reset
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"PG-212 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _routes() -> list[dict[str, Any]]:
    crawl = json.loads(CRAWL_PATH.read_text(encoding="utf-8-sig"))
    catalog = PG208.build_parameter_catalog(crawl)
    entries = [entry for entry in catalog["eligible_entries"] if entry.get("active_replay_eligible") and entry.get("family") == "injection"]
    entries.sort(key=lambda entry: (str(entry.get("path")), str(entry.get("method"))))
    return [PG208._route_from_entry(entry) for entry in entries]


def _send_sql(client: httpx.Client, route: Mapping[str, Any], *, values: Mapping[str, str], marker: str, baseline_status: int | None) -> dict[str, Any]:
    method = str(route["method"]).upper()
    if method == "GET":
        response = client.get(str(route["path"]), params=dict(values), follow_redirects=False)
    else:
        response = client.post(str(route["path"]), data=dict(values), follow_redirects=False)
    return project_sql_response(response, marker=marker, baseline_status=baseline_status)


def _state_fingerprint(result: Mapping[str, Any]) -> str:
    projection = dict(result.get("response_projection") or {})
    return _digest({key: projection.get(key) for key in ("backend_state", "status_class", "body_length_bucket", "shape", "header_names")})


def _route_episode(model: Any, vocabulary: Mapping[str, int], device: torch.device, learner: Any, client: httpx.Client, route: Mapping[str, Any], *, seed: int, target_hash: str, reset: Mapping[str, Any], target_url: str = "http://127.0.0.1:3515") -> dict[str, Any]:
    surface_hash = _digest(route["surface"])[:8]
    base_marker = f"pg212-base-{seed}-{surface_hash}"
    control_marker = f"pg212-control-{seed}-{surface_hash}"
    reference_marker = f"pg212-reference-{seed}-{surface_hash}"
    base_values = build_sql_probe_values(field_names=list(route["fields"]), marker=base_marker, probe_class="control")
    baseline = _send_sql(client, route, values=base_values, marker=base_marker, baseline_status=None)
    reset_view = dict(reset)
    reset_view["baseline_state_fingerprint"] = _state_fingerprint(baseline)
    reset_view["database_connection_state"] = (baseline.get("oracle") or {}).get("backend_state", "unknown")
    baseline_status = int((baseline.get("response_projection") or {}).get("status_code", 0) or 0) or None
    control = _send_sql(client, route, values=build_sql_probe_values(field_names=list(route["fields"]), marker=control_marker, probe_class="control"), marker=control_marker, baseline_status=baseline_status)
    evaluator_available = bool((control.get("oracle") or {}).get("typed_available"))
    candidates = generate_grounded_candidates(family="injection", target=target_url, path=str(route["path"]), method=str(route["method"]), fields=list(route["fields"]), marker=f"pg212-ai-{seed}-{surface_hash}")
    candidate_for_packet = candidates[0]
    packet = build_field_token_packet(candidate_for_packet, route={**route, "typed_available": evaluator_available}, response_projection=dict(control.get("response_projection") or {}), typed_available=evaluator_available, redirect_hops=0)
    validation = validate_field_token_packet(packet, candidate=candidate_for_packet, route={**route, "typed_available": evaluator_available}, response_projection=dict(control.get("response_projection") or {}), typed_available=evaluator_available, redirect_hops=0)
    decision = PG208._model_decision(model, vocabulary, device, packet=packet, route={**route, "typed_available": evaluator_available}, projection=dict(control.get("response_projection") or {})) if validation["valid"] else {"effective_action": "abstain", "abstain_reason": validation["reason"]}
    ai: dict[str, Any] = {"sent": False, "model_decision": decision, "validation": validation, "raw_payload_stored": False, "raw_response_stored": False}
    if evaluator_available and validation["valid"] and decision.get("effective_action") == "safe_candidate":
        selected = learner.select(candidates)
        abstract_class = str(((selected.get("payload") or {}).get("expected") or {}).get("channel", "syntax_shape"))
        if abstract_class not in {"operator_like", "subquery_like", "blind_boolean", "row_shape", "syntax_error"}:
            abstract_class = "syntax_error"
        runtime_class = "syntax_shape" if abstract_class == "syntax_error" else "encoded_syntax_shape"
        marker = f"pg212-ai-{seed}-{surface_hash}"
        result = _send_sql(client, route, values=build_sql_probe_values(field_names=list(route["fields"]), marker=marker, probe_class=runtime_class), marker=marker, baseline_status=baseline_status)
        feedback_status = "candidate" if bool((result.get("signal") or {}).get("sql_error_shape")) else "dead_end"
        bounded_evidence = {
            "feedback_status": feedback_status,
            "backend_state": str((result.get("oracle") or {}).get("backend_state", "unknown")),
            "sql_error_shape": bool((result.get("signal") or {}).get("sql_error_shape")),
            "marker_reflected": bool((result.get("signal") or {}).get("marker_reflected")),
            "body_length_bucket": str((result.get("response_projection") or {}).get("body_length_bucket", "unknown")),
            "safety_flags": {"script_execution": False, "network_access": False, "navigation": False, "database_touched": False, "real_sleep_performed": False},
        }
        bounded_evidence["evidence_hash"] = sha256_json(bounded_evidence)
        feedback = learner.observe(selected, status=feedback_status, evidence=bounded_evidence, evaluator_confirmed=False)
        ai = {"sent": True, "abstract_probe_class": abstract_class, "runtime_probe_class": runtime_class, "candidate": candidate_summary(selected), "response": result, "feedback": feedback, "model_decision": decision, "validation": validation, "raw_payload_stored": False, "raw_response_stored": False}
    reference = _send_sql(client, route, values=build_sql_probe_values(field_names=list(route["fields"]), marker=reference_marker, probe_class="syntax_shape"), marker=reference_marker, baseline_status=baseline_status)
    comparison = compare_sql_shapes(control, reference)
    if ai.get("sent"):
        comparison["ai_reference_shape_agreement"] = bool(compare_sql_shapes(control, ai["response"])["response_shape_differential"] == comparison["response_shape_differential"])
    else:
        comparison["ai_reference_shape_agreement"] = False
    return {
        "seed": int(seed),
        "target_instance_hash": target_hash,
        "surface": route["surface"],
        "path": route["path"],
        "method": route["method"],
        "fields": list(route["fields"]),
        "family": "injection",
        "fresh_target": True,
        "reset": reset_view,
        "database_clean_reset_verified": bool(reset_view.get("container_recreated") and reset_view.get("volume_mount_count") == 0 and reset_view.get("state_change_allowed") is False),
        "baseline": baseline,
        "control": control,
        "ai": ai,
        "reference": {"probe_source": "independent_runtime_syntax_shape", "sent": True, "response": reference, "raw_payload_stored": False, "raw_response_stored": False},
        "comparison": comparison,
        "database_unavailable": (control.get("oracle") or {}).get("backend_state") == "database_unavailable",
        "confirmed_positive": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


def main() -> int:
    routes = _routes()
    model, vocabulary = PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    from app.payload_learner import PayloadLearner
    learner = PayloadLearner(seed=212)
    episodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = _start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                targets.append({"seed": seed, "route": route["path"], "method": route["method"], "target_instance_hash": target_hash, "fresh_container": True, "loopback_port": port, "image": IMAGE, "reset_id": reset["reset_id"]})
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    episodes.append(_route_episode(model, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash, reset=reset))
                finally:
                    client.close()
            finally:
                if name:
                    _stop(name)
            run_index += 1
    counts = {
        "fresh_container_count": len(targets),
        "episode_count": len(episodes),
        "get_episode_count": sum(int(row["method"] == "GET") for row in episodes),
        "post_episode_count": sum(int(row["method"] == "POST") for row in episodes),
        "reference_send_count": sum(int(row["reference"]["sent"]) for row in episodes),
        "ai_candidate_send_count": sum(int(row["ai"].get("sent")) for row in episodes),
        "database_unavailable_count": sum(int(row["database_unavailable"]) for row in episodes),
        "sql_evaluator_typed_available_count": sum(int((row["control"].get("oracle") or {}).get("typed_available")) for row in episodes),
        "ai_reference_shape_agreement_count": sum(int(row["comparison"].get("ai_reference_shape_agreement")) for row in episodes),
        "abstain_count": sum(int(row["ai"].get("model_decision", {}).get("effective_action") == "abstain") for row in episodes),
        "false_positive_count": 0,
    }
    report = {
        "protocol_id": "pg-pk-212-pikachu-sql-response-shape-loop-v1",
        "schema_version": "pg212-pikachu-sql-response-shape-loop-report-v1",
        "status": "completed_sql_backend_unavailable_abstain" if counts["database_unavailable_count"] == len(episodes) else "completed_sql_response_shape_evaluator_only",
        "device": str(device),
        "model": {"variant": "xxl_field_token_adapter", "base_parameter_count": 101487169, "online_weight_update": False},
        "routes": {"count": len(routes), "get_count": sum(int(route["method"] == "GET") for route in routes), "post_count": sum(int(route["method"] == "POST") for route in routes)},
        "targets": targets,
        "episodes": episodes,
        "counts": counts,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "pinned_image": IMAGE, "fresh_container_per_episode": True, "fresh_reset_per_route": True, "no_volume_or_bind_mount": True, "docker_restart_allowed_for_health_recovery_only": True, "get_post_replay": True, "time_delay_used": False, "database_write": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg212-pikachu-sql-response-shape-loop-protocol-v1", "ai_participates_in_send": True, "independent_reference_sent": True, "evaluator": "response_shape_only; no backend AST", "backend_failure_is_not_vulnerability": True, "allowed_probe_classes": ["syntax_shape", "encoded_syntax_shape"], "forbidden_probe_classes": ["time_delay", "local_side_channel", "write", "destructive"], "fresh_reset_required": True, "fresh_reset_unit": "one disposable no-volume container per route episode", "database_clean_required": True, "database_clean_attestation": ["pinned_image_digest", "fresh_container_recreated", "zero_volume_or_bind_mount", "state_change_allowed_false", "baseline_state_fingerprint"], "docker_restart_allowed": "health recovery only; restarted writable layer is not accepted as a clean reset", "matched_negative_control_required": True, "evidence_hash_required": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg212-pikachu-sql-response-shape-loop-trace-v1", "evaluation_only": True, "targets": targets, "episodes": episodes, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    lines = ["# PG-212 Pikachu SQL response-shape loop", "", f"device={device}; fresh containers={len(targets)}; episodes={len(episodes)}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}", f"AI sends={counts['ai_candidate_send_count']}; independent reference sends={counts['reference_send_count']}; database unavailable={counts['database_unavailable_count']}; abstain={counts['abstain_count']}", "", "每个 route episode 都使用 pinned digest 的全新无 volume 容器；docker restart 仅作为健康恢复手段，不作为干净数据库 reset。Pikachu 当前返回 database configuration failure，这是环境阻塞，不是 SQL 漏洞。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "counts": counts, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
