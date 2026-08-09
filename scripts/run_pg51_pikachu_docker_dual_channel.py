"""Run a fresh, loopback-only Pikachu Docker GET/POST surface replay.

The output is deliberately an evaluation-only Catalog.  A bounded reflection
signal is not treated as proof of XSS/SQL/redirect exploitation; when the
Docker-side authoritative oracle is absent, the trace must abstain.
"""

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

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.pg51_docker_replay import PIKACHU_IMAGE_DIGEST, SAFE_PATHS, collect_pair  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json, validate_trace_step  # noqa: E402

REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
CATALOG_PATH = ROOT / "research" / "pg51_pikachu_docker_dual_channel_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg51_pikachu_docker_dual_channel_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg51_pikachu_docker_dual_channel_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg51_pikachu_docker_dual_channel_report_v1.md"
IMAGE = f"tavenli/pikachu-labs@{PIKACHU_IMAGE_DIGEST}"
ORACLE_CONTRACT_SHA256 = hashlib.sha256(b"pg51-pikachu-docker-surface-signal-oracle-v1").hexdigest()
ROUNDS = (("pg51-pikachu-get", 8767, "GET"), ("pg51-pikachu-post", 8768, "POST"))
MARKER = "pg51-inert-canary-a1"


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(name: str, port: int) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse pre-existing container {name}")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{port}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 120.0
    last = "not-ready"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name)
            last = f"http-{response.status_code}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(1.0)
    raise RuntimeError(f"fresh Pikachu container did not become ready: {last}")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--time", "5", name)


def _source(registry: dict[str, Any], collector_hash: str, reset_hash: str) -> dict[str, Any]:
    return {"target_id": "pikachu_docker_dual_channel", "app_family": "pikachu", "source_id": "pg51-pikachu-docker-image", "source_type": "authorized_local_container", "origin_ref": "pg51-pikachu-docker-dual-channel", "license": "local-container", "authorization": "workspace_local_only", "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": 8767}, "container_image_digest": PIKACHU_IMAGE_DIGEST, "collector_sha256": collector_hash, "reset_adapter_sha256": reset_hash, "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "read_only": True, "external_network": False}


def _annotate(row: dict[str, Any], method: str, surface: str, family: str, channel: str, dataset_role: str, pair_role: str) -> dict[str, Any]:
    row.update({"dataset_role": dataset_role, "implementation": "pikachu-docker", "surface_id": surface, "surface_variant": "plain", "semantic_reference": {"xss": "markup-context", "injection": "operator-context", "url_redirect": "destination-context"}[family], "channel_reference": channel, "required_channel": channel, "family": family, "method": method, "phase": "confirm", "pair_role": pair_role})
    return row


def _trace_step(row: dict[str, Any], control: dict[str, Any], episode_id: str, parent: str | None, next_action: str, belief: dict[str, float]) -> dict[str, Any]:
    manifest = row["payload_manifest"]
    action = {"method": manifest["method"], "route_template_id": manifest["route_template_id"], "placement": manifest["placement"], "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if manifest["method"] == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    positive = bool(row["oracle_projection"].get("positive", False))
    after = {"unknown": 1.0}
    # The real Docker target has no authorized execution/AST evaluator in this
    # track, so every candidate must abstain even when a bounded signal exists.
    decision = "abstain"
    oracle = dict(row["oracle_projection"])
    oracle["negative_control_pair_id"] = control["sample_id"]
    echo = {"action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_before": belief, "belief_after": after, "decision": decision, "next_action": next_action}
    return validate_trace_step({"episode_id": episode_id, "step_id": f"{episode_id}-{manifest['method'].casefold()}", "parent_step_id": parent, "sampling_seed": 51, "target_instance_id": row["target_instance_id"], "hypothesis": "unknown_surface", "belief_before": belief, "action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_after": after, "decision": decision, "next_action": next_action, "fresh_reset": row["reset"], "evidence_sha256": row["evidence"]["evidence_hash"], "dataset_stage": "trace_only", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": trace_sha256_json(echo)}})


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256((ROOT / "app" / "pg51_docker_replay.py").read_bytes()).hexdigest()
    reset_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source = _source(registry, collector_hash, reset_hash)
    bound_collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    rows: list[dict[str, Any]] = []
    steps: dict[str, list[dict[str, Any]]] = {path: [] for path in SAFE_PATHS}
    started: list[str] = []
    container_ids: dict[str, str] = {}
    try:
        round_clients: dict[str, tuple[httpx.Client, str, str]] = {}
        for name, port, method in ROUNDS:
            container_id = _start(name, port)
            started.append(name)
            container_ids[method] = container_id
            round_clients[method] = (httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=8.0, follow_redirects=False), container_id, name)
        for path, spec in sorted(SAFE_PATHS.items()):
            episode_id = f"pg51-pikachu-{spec['surface']}"
            parent: str | None = None
            belief = {"unknown": 1.0}
            for method in ("GET", "POST"):
                client, container_id, container_name = round_clients[method]
                raw_spec = {"target": f"http://127.0.0.1:{8767 if method == 'GET' else 8768}", "path": path, "method": method, "marker": MARKER}
                control, candidate, info = collect_pair(source=source, registry=registry, spec=raw_spec, client=client, target_instance_id=container_id[:24], reset_id=f"pg51-reset-{container_name}-{spec['surface']}")
                _annotate(control, method, spec["surface"], spec["family"], "query-channel" if method == "GET" else "form-channel", "negative_control", "control")
                _annotate(candidate, method, spec["surface"], spec["family"], "query-channel" if method == "GET" else "form-channel", "docker_evaluation", "candidate")
                rows.extend([control, candidate])
                step = _trace_step(candidate, control, episode_id, parent, "probe_post" if method == "GET" else "stop_episode", belief)
                steps[path].append(step)
                parent = step["step_id"]
        for client, _, _ in round_clients.values():
            client.close()
    finally:
        for name in reversed(started):
            _stop(name)
    source_catalog = build_catalog("pg51-pikachu-docker-source-catalog-v1", bound_collector.source, rows)
    episodes = [evaluate_episode(step_rows) for step_rows in steps.values()]
    flat_steps = [step for step_rows in steps.values() for step in step_rows]
    catalog = {"schema_version": "pg-pk-51-pikachu-docker-dual-channel-catalog-v1", "catalog_id": "pg51-pikachu-docker-dual-channel-v1", "purpose": "evaluation-only real Docker GET/POST bounded surface signals", "runtime_replay": True, "independent_target_implementation": True, "evaluation_only": True, "training_eligible": False, "training_artifact_generated": False, "model_evaluation_completed": False, "methods": ["GET", "POST"], "phases": ["confirm"], "surface_variants": ["plain"], "implementations": ["pikachu-docker"], "families": sorted({row["family"] for row in rows}), "sources": [bound_collector.source], "source_catalogs": [{"source_id": source_catalog["source"]["source_id"], "source_sha256": source_catalog["source"]["source_sha256"], "catalog_sha256": source_catalog["catalog_sha256"], "sample_count": len(rows), "training_eligible": False}], "samples": rows, "trace_dataset": str(TRACE_PATH.relative_to(ROOT)), "trace_episode_count": len(episodes), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes), "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows), "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows), "fresh_reset_count": len(rows), "source_count": 1, "target_instance_count": len(container_ids), "target_instance_ids": sorted(container_ids.values()), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "authorization": "workspace_local_only", "vulnerability_claims": [], "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in rows], "episodes": episodes})}
    trace = {"schema_version": "pg-pk-51-pikachu-docker-dual-channel-trace-v1", "purpose": "real Docker GET/POST bounded surface trace with mandatory abstain when execution oracle is absent", "evaluation_only": True, "training_eligible": False, "methods": ["GET", "POST"], "episodes": episodes, "episode_count": len(episodes), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes), "steps": flat_steps, "target_instance_ids": sorted(container_ids.values()), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "trace_manifest_sha256": trace_sha256_json([step["trace_sha256"] for step in flat_steps])}
    report = {"protocol_id": "sift-pg51-pikachu-docker-dual-channel-v1", "schema_version": "pg-pk-51-pikachu-docker-dual-channel-report-v1", "status": "diagnostic_only", "target": {"image": IMAGE, "container_ids": container_ids, "loopback_only": True, "external_network": False, "fresh_target_rounds": 2}, "catalog": {"row_count": len(rows), "typed_positive_count": catalog["typed_positive_count"], "negative_count": catalog["negative_control_count"], "fresh_reset_count": catalog["fresh_reset_count"], "source_count": catalog["source_count"], "target_instance_count": catalog["target_instance_count"], "get_post_covered": True}, "oracle_contract": {"surface_reflection_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows), "execution_oracle_available": False, "sql_ast_oracle_available": False, "external_redirect_oracle_available": False, "vulnerability_claim_allowed": False}, "trace": {"episode_count": len(episodes), "accepted_episode_count": trace["accepted_evaluation_episode_count"], "abstain_count": sum(int(step["decision"] == "abstain") for step in flat_steps), "all_steps_fresh": all(step["fresh_reset"]["fresh_target"] for step in flat_steps)}, "promotion": {"status": "quarantined_real_docker_shadow", "training_allowed": False, "memory_promotion_allowed": False}, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "formal_capability_claim_allowed": False, "formal_claim_blockers": ["docker_surface_signal_is_not_execution_oracle", "sql_ast_and_logic_oracles_are_not_attested", "single_image_source"]}
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-51 Pikachu Docker GET/POST replay", "", f"rows: {len(rows)}; episodes: {len(episodes)}; typed surface signals: {catalog['typed_positive_count']}; vulnerability claims: 0", "", "没有脚本执行、SQL 语法/延时、外部重定向或写状态；缺少权威执行 oracle 的候选统一 abstain。", "", f"safety gate: `passed_no_promotion`", ""]) , encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "row_count": len(rows), "typed_positive_count": catalog["typed_positive_count"], "negative_count": catalog["negative_control_count"], "episode_count": len(episodes), "accepted_episode_count": trace["accepted_evaluation_episode_count"], "abstain_count": report["trace"]["abstain_count"], "vulnerability_claim_allowed": False, "catalog": str(CATALOG_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
