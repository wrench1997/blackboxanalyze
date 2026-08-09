"""PG-239: cross-image Pikachu GET/POST replay with explicit environment gate.

The alternate image is evaluated only for bounded request binding and
fail-closed behavior.  Its missing PHP/SQL oracle is an environment result,
never a vulnerability label or training sample.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")
PG217 = _load("run_pg217_pikachu_typed_sql_oracle.py")
from app.pg230_next_token_quality_funnel import digest  # noqa: E402


RESEARCH = ROOT / "research"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
SEEDS = (23901, 23902)
BASE_PORT = 9600
REPORT = RESEARCH / "pg239_alt_pikachu_get_post_replay_report_v1.json"
DATASET = RESEARCH / "pg239_alt_pikachu_get_post_replay_dataset_v1.json"
TRACE = RESEARCH / "pg239_alt_pikachu_get_post_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg239_alt_pikachu_get_post_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg239_alt_pikachu_get_post_replay_report_v1.md"


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(seed: int, index: int) -> tuple[str, int, str, dict[str, Any]]:
    name = f"sift-pg239-alt-{seed}-{index}"
    if _exists(name):
        raise RuntimeError(f"PG-239 refuses to reuse {name}")
    port = BASE_PORT + index
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{port}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 120.0
    healthy = False
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                healthy = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    if not healthy:
        logs = _docker("logs", name, "--tail", "30") if _exists(name) else ""
        if _exists(name):
            _docker("stop", "--timeout", "5", name)
        raise RuntimeError(f"alternate image health failure: {logs[-1000:]}")
    container_id = _docker("inspect", "--format", "{{.Id}}", name)
    mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
    image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
    if mounts:
        raise RuntimeError("PG-239 alternate replay requires zero mounts")
    reset = {"reset_id": f"pg239-alt-reset-{seed}-{index}", "reset_epoch": f"{seed}-{index}", "fresh_target": True, "completed": True, "container_recreated": True, "container_restart_used": False, "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest(), "image": image_ref, "image_digest_attested": image_ref == IMAGE, "volume_mount_count": 0, "database_health_gate": "unavailable_php_oracle", "database_clean_contract": "not_attestable", "state_change_allowed": False, "evaluator_state_hidden": True, "external_network": False, "environment_failure": True}
    return name, port, container_id, reset


def _stop(name: str) -> None:
    if name and _exists(name):
        _docker("stop", "--timeout", "5", name)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    routes = PG214.PG212._routes()
    model, vocabulary = PG214.PG212.PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=239)
    results: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = _start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode()).hexdigest()
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    episode = PG214.PG212._route_episode(model, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash, reset=reset, target_url=f"http://127.0.0.1:{port}")
                    baseline = episode.get("baseline") or {}
                    baseline_status = int((baseline.get("response_projection") or {}).get("status_code", 0) or 0) or None
                    negative_values = PG217._negative_values(route, f"pg239-negative-{seed}-{run_index}")
                    negative = PG217._send(client, route, negative_values, marker=f"pg239-negative-{seed}-{run_index}", baseline_status=baseline_status)
                    ai = dict(episode.get("ai") or {})
                    reference = dict(episode.get("reference") or {})
                    route_hash = digest({"image": IMAGE, "route": route.get("path"), "method": route.get("method"), "fields": route.get("fields")})
                    result = {"seed": seed, "target_instance_hash": target_hash, "route": route["path"], "method": route["method"], "fields": list(route["fields"]), "reset": reset, "route_source_sha256": route_hash, "ai": {"sent": bool(ai.get("sent")), "candidate": {key: (ai.get("candidate") or {}).get(key) for key in ("method", "path", "probe_kind", "probe_sha256", "payload_sha256")}, "response_projection": dict((ai.get("response") or {}).get("response_projection") or {}), "raw_payload_stored": False, "raw_response_stored": False}, "reference": {"sent": bool(reference.get("sent")), "response_projection": dict((reference.get("response") or {}).get("response_projection") or {}), "raw_payload_stored": False, "raw_response_stored": False}, "negative": {"sent": True, "response_projection": negative, "raw_payload_stored": False, "raw_response_stored": False}, "oracle": {"typed_available": False, "confirmed_positive": False, "abstain": True, "reason": "alternate_image_php_or_sql_oracle_unavailable", "vulnerability_claim_allowed": False}, "environment_failure": True, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
                    result["evidence_sha256"] = digest({"target": target_hash, "route_source": route_hash, "ai": result["ai"], "reference": result["reference"], "negative": result["negative"], "oracle": result["oracle"], "reset": reset})
                    results.append(result)
                finally:
                    client.close()
            finally:
                if name:
                    _stop(name)
            run_index += 1
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    counts = {"fresh_container_count": len(results), "get_count": sum(int(row["method"] == "GET") for row in results), "post_count": sum(int(row["method"] == "POST") for row in results), "ai_send_count": sum(int(row["ai"]["sent"]) for row in results), "reference_send_count": sum(int(row["reference"]["sent"]) for row in results), "negative_send_count": len(results), "typed_oracle_available_count": 0, "confirmed_positive_count": 0, "abstain_count": len(results), "environment_failure_count": len(results), "training_eligible_count": 0, "false_positive_count": 0}
    report = {"protocol_id": "pg-pk-239-alt-pikachu-get-post-replay-v1", "schema_version": "pg239-alt-pikachu-get-post-replay-v1", "status": "completed_cross_image_environment_gated_replay", "device": str(device), "runtime_image": IMAGE, "seeds": list(SEEDS), "routes": routes, "counts": counts, "results": results, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "environment_failure_rows_trainable": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "honesty": {"alternate_php_oracle_unavailable": True, "all_rows_abstain_only": True, "cross_implementation_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    dataset = {"schema_version": "pg239-alt-pikachu-get-post-replay-dataset-v1", "records": [{"source": "pg239_alt_pikachu_get_post_replay", "seed": row["seed"], "method": row["method"], "family": "sql", "surface_class": "sql_surface", "candidate_sent": row["ai"]["sent"], "reference_sent": row["reference"]["sent"], "negative_sent": True, "oracle_available": False, "abstain": True, "environment_failure": True, "route_source_sha256": row["route_source_sha256"], "evidence_sha256": row["evidence_sha256"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False} for row in results], "counts": counts, "contract": {"cross_image_only": True, "environment_failure_is_not_model_label": True, "typed_oracle_required_for_positive": True, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    dataset["dataset_sha256"] = digest(dataset)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg239-alt-pikachu-get-post-replay-protocol-v1", "alternate_image": IMAGE, "methods": ["GET", "POST"], "fresh_container_per_route": True, "candidate_reference_negative_required": True, "typed_oracle_required_for_positive": True, "environment_failure_quarantined": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg239-alt-pikachu-get-post-replay-trace-v1", "results": results, "training_eligible": False, "environment_failure_count": len(results), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-239 alternate Pikachu GET/POST replay", "", f"image={IMAGE}; fresh={counts['fresh_container_count']}; GET={counts['get_count']}; POST={counts['post_count']}", f"AI={counts['ai_send_count']}; reference={counts['reference_send_count']}; negative={counts['negative_send_count']}; oracle_available={counts['typed_oracle_available_count']}; abstain={counts['abstain_count']}", "", "tavenli 镜像可返回 HTTP 页面，但容器内 PHP/SQL oracle 不可用；这是 environment_failure，不是漏洞阴性，也不进入训练或长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

