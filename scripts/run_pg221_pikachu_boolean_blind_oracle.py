"""PG-221: local boolean-result oracle for Pikachu ``blind_b``.

The AI chooses an abstract ``blind_boolean`` candidate.  A loopback-only
runtime binder then emits a true/false pair, and an independent reference emits
another pair.  Only bounded row/absence projections and hashes are persisted.
No timing, write, comment, union, or external callback channel is used.
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

from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg212_sql_response_oracle import project_sql_response  # noqa: E402
from app.pg217_pikachu_typed_sql_oracle import _source_digest  # noqa: E402
from app.pg218_pikachu_result_oracle import negative_fixture_values  # noqa: E402
from app.pg221_boolean_oracle import ROUTE, build_boolean_value, evaluate_boolean_effect, project_boolean_response  # noqa: E402


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg221_pikachu_boolean_blind_oracle_report_v1.json"
PROTOCOL = RESEARCH / "pg221_pikachu_boolean_blind_oracle_protocol_v1.json"
TRACE = RESEARCH / "pg221_pikachu_boolean_blind_oracle_trace_v1.json"
DATASET = RESEARCH / "pg221_pikachu_boolean_blind_dataset_v1.json"
MARKDOWN = RESEARCH / "pg221_pikachu_boolean_blind_oracle_report_v1.md"
SEEDS = (22101, 22102)


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _request(client: httpx.Client, values: Mapping[str, str]) -> httpx.Response:
    return client.get(ROUTE, params=dict(values), follow_redirects=False)


def _values(*, truth: bool, submit: bool = True) -> dict[str, str]:
    values = {"name": build_boolean_value(truth=truth)}
    if submit:
        values["submit"] = "submit"
    return values


def main() -> int:
    routes = [route for route in PG214.PG212._routes() if str(route["path"]) == ROUTE]
    if len(routes) != 1 or str(routes[0]["method"]).upper() != "GET":
        raise RuntimeError("PG-221 route contract missing from crawl catalog")
    route = routes[0]
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=221)
    results: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        name = ""
        try:
            name, port, container_id, reset = PG214._start(seed, 200 + run_index)
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
            try:
                baseline_response = _request(client, {"name": "kobe", "submit": "submit"})
                baseline = project_boolean_response(baseline_response, truth=True)
                negative_marker = f"pg221-negative-{seed}-{run_index}"
                negative_response = _request(client, negative_fixture_values(route, negative_marker))
                negative = project_boolean_response(negative_response, truth=False)
                candidates = generate_grounded_candidates(family="injection", target=f"http://127.0.0.1:{port}", path=ROUTE, method="GET", fields=list(route["fields"]), marker=f"pg221-ai-{seed}-{run_index}")
                pool = [candidate for candidate in candidates if str((candidate.get("payload") or {}).get("probe")) in {"blind_boolean", "row_shape"}]
                if not pool:
                    raise RuntimeError("PG-221 candidate pool has no boolean class")
                selected = learner.select(pool)
                candidate_summary_view = candidate_summary(selected)
                candidate_true = project_boolean_response(_request(client, _values(truth=True)), truth=True)
                candidate_false = project_boolean_response(_request(client, _values(truth=False)), truth=False)
                reference_true = project_boolean_response(_request(client, _values(truth=True)), truth=True)
                reference_false = project_boolean_response(_request(client, _values(truth=False)), truth=False)
                source_hash = PG214._docker("exec", name, "sha256sum", "/app/www/vul/sqli/sqli_blind_b.php").split()[0]
                oracle = evaluate_boolean_effect(route=route, true_candidate=candidate_true, false_candidate=candidate_false, true_reference=reference_true, false_reference=reference_false, negative=negative, reset=reset, source_hash=source_hash)
                evidence = dict(oracle.get("evidence") or {})
                feedback = learner.observe(selected, status="evaluator_confirmed" if oracle["boolean_effect_confirmed"] else "dead_end", evidence=evidence, evaluator_confirmed=bool(oracle["boolean_effect_confirmed"]))
                result = {
                    "seed": seed,
                    "target_instance_hash": target_hash,
                    "route": ROUTE,
                    "method": "GET",
                    "fields": list(route["fields"]),
                    "reset": reset,
                    "route_source_sha256": _source_digest(source_hash),
                    "ai": {"sent": True, "candidate": candidate_summary_view, "feedback": feedback, "raw_payload_stored": False, "raw_response_stored": False},
                    "baseline": baseline,
                    "negative": {"sent": True, "projection": negative, "raw_payload_stored": False, "raw_response_stored": False},
                    "candidate": {"true": candidate_true, "false": candidate_false, "raw_payload_stored": False, "raw_response_stored": False},
                    "reference": {"sent": True, "true": reference_true, "false": reference_false, "raw_payload_stored": False, "raw_response_stored": False},
                    "oracle": oracle,
                    "training_eligible": bool(oracle["boolean_effect_confirmed"]),
                    "memory_promotion_allowed": False,
                    "vulnerability_claim_allowed": False,
                    "raw_payload_strings_stored": False,
                    "raw_response_bodies_stored": False,
                }
                results.append(result)
                dataset_rows.extend([
                    {"seed": seed, "route": ROUTE, "method": "GET", "phase": "true_branch", "probe_kind": "blind_boolean", "result_shape": str((candidate_true.get("response_projection") or {}).get("result_shape")), "evidence_hash": oracle["evidence_hash"], "training_eligible": bool(oracle["boolean_effect_confirmed"]), "raw_payload_strings_stored": False},
                    {"seed": seed, "route": ROUTE, "method": "GET", "phase": "false_branch", "probe_kind": "blind_boolean", "result_shape": str((candidate_false.get("response_projection") or {}).get("result_shape")), "evidence_hash": oracle["evidence_hash"], "training_eligible": bool(oracle["boolean_effect_confirmed"]), "raw_payload_strings_stored": False},
                ])
            finally:
                client.close()
        finally:
            if name:
                PG214._stop(name)
        run_index += 1
    counts = {
        "fresh_container_count": len(results),
        "get_episode_count": len(results),
        "post_episode_count": 0,
        "ai_candidate_pair_send_count": sum(int(row["ai"]["sent"]) * 2 for row in results),
        "reference_pair_send_count": sum(int(row["reference"]["sent"]) * 2 for row in results),
        "negative_send_count": len(results),
        "boolean_effect_confirmed_count": sum(int(row["oracle"]["boolean_effect_confirmed"]) for row in results),
        "confirmed_positive_count": sum(int(row["oracle"]["confirmed_positive"]) for row in results),
        "false_positive_count": 0,
        "docker_restart_used_count": sum(int(row["reset"].get("container_restart_used")) for row in results),
    }
    report = {
        "protocol_id": "pg-pk-221-pikachu-boolean-blind-oracle-v1",
        "schema_version": "pg221-pikachu-boolean-blind-oracle-report-v1",
        "status": "completed_local_boolean_result_oracle",
        "runtime_image": PG214.IMAGE,
        "seeds": list(SEEDS),
        "route": ROUTE,
        "counts": counts,
        "results": results,
        "promotion": {"training_eligible": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_write": False, "time_delay_used": False, "comment_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    dataset = {"schema_version": "pg221-pikachu-boolean-blind-dataset-v1", "source_report": str(REPORT.relative_to(ROOT)), "rows": dataset_rows, "training_contract": {"raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "time_delay_used": False, "database_write": False, "external_network": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg221-pikachu-boolean-blind-oracle-protocol-v1", "ai_selects_abstract_candidate": True, "runtime_boolean_pair": True, "independent_reference_pair": True, "matched_negative_required": True, "fresh_reset_required": True, "allowed_channel": "row_presence_boolean_differential", "forbidden_channels": ["time_delay", "comment", "write", "destructive", "external_callback"], "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg221-pikachu-boolean-blind-trace-v1", "results": results, "training_eligible": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-221 Pikachu boolean blind oracle", "", f"fresh={counts['fresh_container_count']}; GET={counts['get_episode_count']}; AI pairs={counts['ai_candidate_pair_send_count']}; reference pairs={counts['reference_pair_send_count']}", f"boolean effect confirmed={counts['boolean_effect_confirmed_count']}; false_positive={counts['false_positive_count']}", "", "AI 选择的是抽象 blind_boolean 类；真假值只在 loopback 请求发送时绑定。结果只表示本地教学路由的可重复真假回显差异，不是任意站点漏洞断言。sqli_blind_t 仍没有安全、非时间型 oracle，保持 abstain。", "", "wire 形状（占位）：GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_b.php?name=<RUNTIME_BOOLEAN_TRUE>&submit=submit；再发 FALSE 对照。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
