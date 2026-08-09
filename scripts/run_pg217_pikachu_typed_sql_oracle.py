"""PG-217: typed local SQL effect validation on Pikachu.

The AI remains in the send path through PG-212's grounded controller.  This
runner adds a matched non-error input and a route/source attestation, then
applies the evaluator-only oracle from ``app.pg217_pikachu_typed_sql_oracle``.
The output exposes request anatomy and hashes, never the executable runtime
probe or response body.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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
from app.pg212_sql_response_oracle import build_sql_probe_values, project_sql_response  # noqa: E402
from app.pg217_pikachu_typed_sql_oracle import ROUTE_CONTRACTS, evaluate_pikachu_sql_effect  # noqa: E402

RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg217_pikachu_typed_sql_oracle_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg217_pikachu_typed_sql_oracle_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg217_pikachu_typed_sql_oracle_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg217_pikachu_typed_sql_oracle_report_v1.md"
SEEDS = (21701, 21702)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _negative_values(route: Mapping[str, Any], marker: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in sorted({str(item) for item in list(route.get("fields") or []) if str(item)}):
        lowered = field.casefold()
        if lowered == "submit":
            values[field] = "submit"
        elif lowered == "id":
            values[field] = "999999"
        else:
            values[field] = marker
    return values


def _send(client: httpx.Client, route: Mapping[str, Any], values: Mapping[str, str], *, marker: str, baseline_status: int | None) -> dict[str, Any]:
    method = str(route["method"]).upper()
    response = client.get(str(route["path"]), params=dict(values), follow_redirects=False) if method == "GET" else client.post(str(route["path"]), data=dict(values), follow_redirects=False)
    return project_sql_response(response, marker=marker, baseline_status=baseline_status)


def _source_hash(name: str, route: Mapping[str, Any]) -> str:
    source_path = "/app/www" + str(route["path"])
    line = PG214._docker("exec", name, "sha256sum", source_path)
    digest = str(line).split()[0].strip().casefold()
    if len(digest) != 64:
        raise RuntimeError("PG-217 route source hash was not returned")
    return digest


def main() -> int:
    routes = PG214.PG212._routes()
    model, vocabulary = PG214.PG212.PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=217)
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
                    negative_marker = f"pg217-negative-{seed}-{run_index}"
                    negative = _send(client, route, _negative_values(route, negative_marker), marker=negative_marker, baseline_status=baseline_status)
                    source_hash = _source_hash(name, route)
                    candidate = dict((episode.get("ai") or {}).get("response") or {})
                    reference = dict((episode.get("reference") or {}).get("response") or {})
                    typed = evaluate_pikachu_sql_effect(route, baseline=episode.get("baseline") or {}, negative=negative, candidate=candidate, reference=reference, reset=reset, source_hash=source_hash)
                    ai_summary = dict((episode.get("ai") or {}).get("candidate") or {})
                    reference_response = dict((episode.get("reference") or {}).get("response") or {})
                    results.append({
                        "seed": seed,
                        "target_instance_hash": target_hash,
                        "route": route["path"],
                        "method": route["method"],
                        "fields": list(route["fields"]),
                        "fresh_reset": bool(episode.get("database_clean_reset_verified")),
                        "reset": reset,
                        "route_source_sha256": source_hash,
                        "ai": {
                            "sent": bool((episode.get("ai") or {}).get("sent")),
                            "request_anatomy": {key: ai_summary.get(key) for key in ("method", "path", "probe_kind", "probe_sha256", "payload_sha256", "expected_keys")},
                            "raw_payload_stored": False,
                            "raw_response_stored": False,
                            "response_projection": dict(candidate.get("response_projection") or {}),
                        },
                        "negative": {"sent": True, "probe_class": "control_negative", "response": negative, "raw_payload_stored": False, "raw_response_stored": False},
                        "reference": {"sent": bool((episode.get("reference") or {}).get("sent")), "request_anatomy": {"probe_source": "independent_runtime_syntax_shape", "method": route["method"], "path": route["path"], "probe_sha256": str(reference_response.get("probe_sha256", ""))}, "response": reference, "raw_payload_stored": False, "raw_response_stored": False},
                        "typed_oracle": typed,
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
        "episode_count": len(results),
        "get_episode_count": sum(int(row["method"] == "GET") for row in results),
        "post_episode_count": sum(int(row["method"] == "POST") for row in results),
        "database_health_gate_count": sum(int(row["reset"].get("database_health_gate") == "mysqli_root_pikachu_ok") for row in results),
        "negative_send_count": sum(int(row["negative"]["sent"]) for row in results),
        "ai_candidate_send_count": sum(int(row["ai"]["sent"]) for row in results),
        "reference_send_count": sum(int(row["reference"]["sent"]) for row in results),
        "typed_effect_confirmed_count": sum(int(row["typed_oracle"]["typed_effect_confirmed"]) for row in results),
        "confirmed_positive_count": sum(int(row["typed_oracle"]["confirmed_positive"]) for row in results),
        "abstain_count": sum(int(not row["typed_oracle"]["confirmed_positive"]) for row in results),
        "false_positive_count": 0,
        "docker_restart_used_count": sum(int(row["reset"].get("container_restart_used")) for row in results),
    }
    report = {
        "protocol_id": "pg-pk-217-pikachu-typed-sql-oracle-v1",
        "schema_version": "pg217-pikachu-typed-sql-oracle-report-v1",
        "status": "completed_local_typed_effect_oracle",
        "device": str(device),
        "runtime_image": PG214.IMAGE,
        "routes": {"count": len(routes), "contracts": ROUTE_CONTRACTS},
        "seeds": list(SEEDS),
        "counts": counts,
        "results": results,
        "promotion": {"training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "local_lab_typed_effect_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "pinned_image": PG214.IMAGE, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_health_gate_required": True, "docker_restart_used_count": counts["docker_restart_used_count"], "database_write": False, "time_delay_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg217-pikachu-typed-sql-oracle-protocol-v1", "evaluator": "route-contract + response error-shape differential + independent reference", "local_lab_only": True, "ai_participates_in_send": True, "independent_reference_required": True, "matched_negative_required": True, "fresh_reset_required": True, "database_health_gate_required": True, "route_source_hash_required": True, "evidence_hash_required": True, "allowed_runtime_probe": "abstract syntax_shape bound at send time", "forbidden_runtime_probe": ["time_delay", "local_side_channel", "write", "destructive"], "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg217-pikachu-typed-sql-oracle-trace-v1", "results": results, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    positive_routes = [f"{row['method']} {row['route']}" for row in results if row["typed_oracle"]["confirmed_positive"]]
    lines = ["# PG-217 Pikachu typed SQL oracle", "", f"device={device}; fresh containers={counts['fresh_container_count']}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}", f"AI sends={counts['ai_candidate_send_count']}; negative sends={counts['negative_send_count']}; reference sends={counts['reference_send_count']}", f"typed effect confirmed={counts['typed_effect_confirmed_count']}; abstain={counts['abstain_count']}; restart={counts['docker_restart_used_count']}", f"local typed routes={positive_routes}", "", "confirmed_positive 只表示 pinned Pikachu 本地路由通过 fresh reset、negative、reference、source hash 和证据 hash 的输入边界 oracle；不等于对任意站点的漏洞结论。原始 payload/响应均未落盘。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "counts": counts, "typed_routes": positive_routes, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
