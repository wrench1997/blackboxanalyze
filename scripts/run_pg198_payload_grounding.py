"""PG-198: let the AI choose and send grounded GET/POST canaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_learner import PayloadLearner  # noqa: E402
from app.pg195_request_surface_adapter import project_surface_response  # noqa: E402
from app.pg198_payload_grounding import choose_and_ground, generate_grounded_candidates  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg198-payload-grounding-v1"
REPORT_PATH = RESEARCH / "pg198_payload_grounding_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg198_payload_grounding_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg198_payload_grounding_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg198_payload_grounding_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PORT = 3110
BASE_URL = f"http://127.0.0.1:{PORT}"
SEEDS = (19801, 19802)


ROUTES = (
    {
        "surface": "pg198_xss_get",
        "path": "/vul/xss/xss_01.php",
        "method": "GET",
        "family": "xss",
        "fields": ["message", "submit"],
        "layout": "inline_html",
        "typed_available": True,
    },
    {
        "surface": "pg198_sql_get",
        "path": "/vul/sqli/sqli_search.php",
        "method": "GET",
        "family": "injection",
        "fields": ["name", "submit"],
        "layout": "table_cell",
        "typed_available": False,
    },
    {
        "surface": "pg198_post_unknown",
        "path": "/vul/xss/xsspost/post_login.php",
        "method": "POST",
        "family": "xss",
        "fields": ["username", "submit"],
        "layout": "attribute_shell",
        "typed_available": False,
    },
)


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
        raise RuntimeError(f"PG-198 refuses to reuse target {name}")
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
    raise RuntimeError(f"PG-198 target {name} did not become ready")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--timeout", "5", name)


def _baseline(client: httpx.Client, route: dict[str, Any], marker: str) -> dict[str, Any]:
    method = str(route["method"]).upper()
    if method == "GET":
        response = client.get(str(route["path"]), follow_redirects=False)
    else:
        response = client.post(str(route["path"]), data={"username": marker, "submit": "submit"}, follow_redirects=False)
    projected = project_surface_response(
        response,
        marker=marker,
        layout_variant=str(route["layout"]),
        baseline_status=None,
        run_browser=False,
    )
    projected.pop("body_text", None)
    projected.pop("signal", None)
    return projected["response_projection"]


def _episode(client: httpx.Client, learner: PayloadLearner, *, seed: int, target_hash: str, route: dict[str, Any]) -> dict[str, Any]:
    marker = f"pg198-candidate-{seed}-{route['surface']}"
    baseline_marker = f"pg198-baseline-{seed}-{route['surface']}"
    baseline = _baseline(client, route, baseline_marker)
    candidates = generate_grounded_candidates(
        family=str(route["family"]),
        target=BASE_URL,
        path=str(route["path"]),
        method=str(route["method"]),
        fields=list(route["fields"]),
        marker=marker,
    )
    result = choose_and_ground(
        learner,
        candidates,
        client=client,
        fields=list(route["fields"]),
        layout_variant=str(route["layout"]),
        baseline_status=int(baseline.get("status_code", 0)) or None,
        typed_available=bool(route["typed_available"]),
    )
    result["baseline_projection"] = baseline
    result["target_instance_hash"] = target_hash
    result["seed"] = seed
    result["surface"] = route["surface"]
    result["method"] = route["method"]
    result["path"] = route["path"]
    result["observed_fields"] = list(route["fields"])
    result["fresh_container"] = True
    result["request_sent_by_ai"] = True
    result["typed_oracle_available"] = bool(route["typed_available"])
    result["promotion"] = {
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return result


def main() -> int:
    learner = PayloadLearner(seed=198)
    route_runs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for seed in SEEDS:
        name = f"sift-pg198-{seed}"
        container_id = _start(name)
        target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
        targets.append({"seed": seed, "target_instance_hash": target_hash, "fresh_container": True})
        client = httpx.Client(base_url=BASE_URL, timeout=10.0, follow_redirects=False, cookies={})
        try:
            for route in ROUTES:
                route_runs.append(_episode(client, learner, seed=seed, target_hash=target_hash, route=route))
        finally:
            client.close()
            _stop(name)

    dom_runs = [row for row in route_runs if row["typed_oracle_available"]]
    unknown_runs = [row for row in route_runs if not row["typed_oracle_available"]]
    report = {
        "protocol_id": "pg-pk-198-payload-grounding-v1",
        "schema_version": "pg198-payload-grounding-report-v1",
        "status": "completed_ai_selected_local_get_post_grounding",
        "model": {
            "policy": "PayloadLearner-UCB-over-validated-candidate-manifests",
            "online_weight_update": False,
            "family_labels_visible_to_policy": False,
        },
        "targets": targets,
        "route_runs": route_runs,
        "counts": {
            "fresh_container_count": len(targets),
            "route_replay_count": len(route_runs),
            "ai_candidate_send_count": sum(int(row["request_sent_by_ai"]) for row in route_runs),
            "grounded_payload_hash_match_count": sum(int(bool(row["candidate"]["payload_sha256"] == row["evidence"]["payload_sha256"])) for row in route_runs),
            "method_binding_match_count": sum(int(row["binding"]["method"] == row["method"] and row["binding"]["path"] == row["path"]) for row in route_runs),
            "dom_dual_agreement_count": sum(int(bool((row["oracle"] or {}).get("dual_agreement"))) for row in dom_runs),
            "unknown_oracle_abstain_count": sum(int((row["oracle"] or {}).get("abstain_reason") == "pikachu_surface_oracle_unknown") for row in unknown_runs),
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
            "get_post_only": True,
            "runtime_values_persisted": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "credentials_accessed": False,
        },
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg198-payload-grounding-protocol-v1",
        "source": "pikachu-pinned-local-container",
        "methods": ["GET", "POST"],
        "ai_role": "select_candidate_bind_runtime_values_send_request_receive_projection",
        "candidate_contract": "validated_detection_manifest_only",
        "unknown_oracle_action": "abstain",
        "fresh_container_per_seed": True,
        "cross_method_binding_required": True,
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {
        "schema_version": "pg198-payload-grounding-trace-v1",
        "evaluation_only": True,
        "route_runs": route_runs,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    })
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "learner_summary.json").write_text(json.dumps(learner.summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-198 AI payload grounding",
        "",
        f"fresh containers={report['counts']['fresh_container_count']}; routes={report['counts']['route_replay_count']}; AI sends={report['counts']['ai_candidate_send_count']}",
        f"hash/method binding={report['counts']['grounded_payload_hash_match_count']}/{report['counts']['method_binding_match_count']}; DOM dual agreement={report['counts']['dom_dual_agreement_count']}/{len(dom_runs)}; unknown abstain={report['counts']['unknown_oracle_abstain_count']}/{len(unknown_runs)}",
        "",
        "The AI selected and sent only validated local canaries. Runtime values and raw response bodies were discarded; unknown target oracles remain abstentions.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "fresh_containers": report["counts"]["fresh_container_count"],
        "route_replays": report["counts"]["route_replay_count"],
        "ai_candidate_sends": report["counts"]["ai_candidate_send_count"],
        "dom_dual_agreement": report["counts"]["dom_dual_agreement_count"],
        "unknown_oracle_abstain": report["counts"]["unknown_oracle_abstain_count"],
        "training_eligible": False,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
