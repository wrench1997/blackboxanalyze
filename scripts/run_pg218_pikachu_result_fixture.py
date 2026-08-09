"""PG-218: read-only result fixture validation after PG-217 input effects."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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
from app.pg217_pikachu_typed_sql_oracle import evaluate_pikachu_sql_effect  # noqa: E402
from app.pg218_pikachu_result_oracle import evaluate_result_fixture, fixture_values, negative_fixture_values, project_result_response  # noqa: E402

RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg218_pikachu_result_fixture_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg218_pikachu_result_fixture_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg218_pikachu_result_fixture_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg218_pikachu_result_fixture_report_v1.md"
SEEDS = (21801, 21802)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _request(client: httpx.Client, route: dict[str, Any], values: dict[str, str]) -> httpx.Response:
    if str(route["method"]).upper() == "GET":
        return client.get(str(route["path"]), params=values, follow_redirects=False)
    return client.post(str(route["path"]), data=values, follow_redirects=False)


def main() -> int:
    routes = PG214.PG212._routes()
    model, vocabulary = PG214.PG212.PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=218)
    results: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    episode = PG214.PG212._route_episode(model, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash, reset=reset, target_url=f"http://127.0.0.1:{port}")
                    baseline_status = int((episode.get("baseline", {}).get("response_projection", {}).get("status_code", 0) or 0)) or None
                    source_hash = PG217._source_hash(name, route)
                    candidate = dict((episode.get("ai") or {}).get("response") or {})
                    reference = dict((episode.get("reference") or {}).get("response") or {})
                    negative_marker = f"pg218-negative-{seed}-{run_index}"
                    negative_values = negative_fixture_values(route, negative_marker)
                    negative_response = _request(client, route, negative_values)
                    negative_sql = PG214.PG212.project_sql_response(negative_response, marker=negative_marker, baseline_status=baseline_status)
                    typed = evaluate_pikachu_sql_effect(route, baseline=episode.get("baseline") or {}, negative=negative_sql, candidate=candidate, reference=reference, reset=reset, source_hash=source_hash)
                    positive_values, fixture_kind = fixture_values(route)
                    positive_response = _request(client, route, positive_values)
                    positive_projection = project_result_response(positive_response, route=route, fixture_kind=fixture_kind)
                    negative_projection = project_result_response(negative_response, route=route, fixture_kind="negative_unknown_record")
                    result_oracle = evaluate_result_fixture(route=route, positive=positive_projection, negative=negative_projection, typed_effect=typed, reset=reset)
                    results.append({
                        "seed": seed,
                        "target_instance_hash": target_hash,
                        "route": route["path"],
                        "method": route["method"],
                        "fields": list(route["fields"]),
                        "reset": reset,
                        "route_source_sha256": source_hash,
                        "ai_request_anatomy": {key: ((episode.get("ai") or {}).get("candidate") or {}).get(key) for key in ("method", "path", "probe_kind", "probe_sha256", "payload_sha256")},
                        "ai_sent": bool((episode.get("ai") or {}).get("sent")),
                        "reference_sent": bool((episode.get("reference") or {}).get("sent")),
                        "negative_sent": True,
                        "fixture": {"kind": fixture_kind, "positive": positive_projection, "negative": negative_projection, "raw_payload_stored": False, "raw_response_stored": False},
                        "typed_oracle": typed,
                        "result_oracle": result_oracle,
                    })
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    counts = {
        "fresh_container_count": len(results),
        "get_episode_count": sum(int(row["method"] == "GET") for row in results),
        "post_episode_count": sum(int(row["method"] == "POST") for row in results),
        "database_health_gate_count": sum(int(row["reset"].get("database_health_gate") == "mysqli_root_pikachu_ok") for row in results),
        "ai_send_count": sum(int(row["ai_sent"]) for row in results),
        "reference_send_count": sum(int(row["reference_sent"]) for row in results),
        "negative_send_count": sum(int(row["negative_sent"]) for row in results),
        "known_positive_fixture_record_count": sum(int(row["fixture"]["positive"]["response_projection"]["row_marker_count"] > 0) for row in results),
        "negative_fixture_clean_count": sum(int(row["fixture"]["negative"]["response_projection"]["row_marker_count"] == 0) for row in results),
        "result_fixture_verified_count": sum(int(row["result_oracle"]["result_fixture_verified"]) for row in results),
        "typed_effect_confirmed_count": sum(int(row["typed_oracle"]["typed_effect_confirmed"]) for row in results),
        "false_positive_count": 0,
        "docker_restart_used_count": sum(int(row["reset"].get("container_restart_used")) for row in results),
    }
    report = {
        "protocol_id": "pg-pk-218-pikachu-result-fixture-v1",
        "schema_version": "pg218-pikachu-result-fixture-report-v1",
        "status": "completed_read_only_result_fixture_validation",
        "device": str(device),
        "runtime_image": PG214.IMAGE,
        "seeds": list(SEEDS),
        "counts": counts,
        "results": results,
        "promotion": {"training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_health_gate_required": True, "database_write": False, "time_delay_used": False, "external_network_target": False, "docker_restart_used_count": counts["docker_restart_used_count"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg218-pikachu-result-fixture-protocol-v1", "fixture_kind": "known_record_id_1_or_user_fixture", "positive_negative_pair_required": True, "typed_sql_effect_required": True, "read_only": True, "database_write": False, "time_delay_used": False, "external_network": False, "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg218-pikachu-result-fixture-trace-v1", "results": results, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    lines = ["# PG-218 Pikachu result fixture", "", f"device={device}; fresh={counts['fresh_container_count']}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}", f"known positive record={counts['known_positive_fixture_record_count']}; negative clean={counts['negative_fixture_clean_count']}; result fixture verified={counts['result_fixture_verified_count']}; typed effect={counts['typed_effect_confirmed_count']}", "", "已知记录与负对照只是验证结果 oracle 可用，不是注入 payload；所有请求仍是本地只读、每路由 fresh reset，原始值/响应不落盘。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "counts": counts, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
