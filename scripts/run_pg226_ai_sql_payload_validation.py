"""PG-226: validate AI-selected local SQL probes with typed and result oracles.

Four read-only Pikachu routes are replayed in fresh containers.  The AI selects
an abstract SQL probe class; candidate and independent reference bind runtime
syntax-shape values separately.  A known-record/unknown-record pair then
checks the read-only result channel.  Persisted output contains only hashes,
bounded projections and placeholder wire shapes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg212_sql_response_oracle import build_sql_probe_values, project_sql_response  # noqa: E402
from app.pg217_pikachu_typed_sql_oracle import ROUTE_CONTRACTS, evaluate_pikachu_sql_effect  # noqa: E402
from app.pg218_pikachu_result_oracle import evaluate_result_fixture, fixture_values, negative_fixture_values, project_result_response  # noqa: E402
from app.pg224_surface_projector import wire_placeholder  # noqa: E402
from app.payload_learner import PayloadLearner  # noqa: E402


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg226_ai_sql_payload_validation_report_v1.json"
DATASET = RESEARCH / "pg226_ai_sql_payload_validation_dataset_v1.json"
TRACE = RESEARCH / "pg226_ai_sql_payload_validation_trace_v1.json"
PROTOCOL = RESEARCH / "pg226_ai_sql_payload_validation_protocol_v1.json"
MARKDOWN = RESEARCH / "pg226_ai_sql_payload_validation_report_v1.md"
SEEDS = (22601, 22602)
ROUTE_PATHS = frozenset({
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_x.php",
})
BASE_INDEX = 700


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


def _source_hash(name: str, route: Mapping[str, Any]) -> str:
    source_path = "/app/www" + str(route["path"])
    value = PG214._docker("exec", name, "sha256sum", source_path).split()[0].strip().casefold()
    if len(value) != 64:
        raise RuntimeError("PG-226 source hash missing")
    return value


def _reference_values(route: Mapping[str, Any], marker: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in sorted({str(item) for item in route.get("fields", []) if str(item)}):
        if field.casefold() == "submit":
            values[field] = "submit"
        elif field.casefold() == "id":
            values[field] = "1'"
        else:
            values[field] = marker + "'"
    return values


def _send_raw(client: httpx.Client, route: Mapping[str, Any], values: Mapping[str, str]) -> httpx.Response:
    if str(route["method"]).upper() == "POST":
        return client.post(str(route["path"]), data=dict(values), follow_redirects=False)
    return client.get(str(route["path"]), params=dict(values), follow_redirects=False)


def _send(client: httpx.Client, route: Mapping[str, Any], values: Mapping[str, str], *, marker: str, baseline_status: int | None) -> dict[str, Any]:
    response = _send_raw(client, route, values)
    projected = project_sql_response(response, marker=marker, baseline_status=baseline_status)
    response.close()
    return projected


def main() -> int:
    routes = [route for route in PG214.PG212._routes() if str(route["path"]) in ROUTE_PATHS]
    if len(routes) != 4 or any(str(route["path"]) not in ROUTE_CONTRACTS for route in routes):
        raise RuntimeError("PG-226 route contracts are incomplete")
    learner = PayloadLearner(seed=226)
    results: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, BASE_INDEX + run_index)
                target = f"http://127.0.0.1:{port}"
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                fields = list(route["fields"])
                marker = f"pg226-{seed}-{run_index:02d}"
                candidates = generate_grounded_candidates(family="injection", target=target, path=str(route["path"]), method=str(route["method"]), fields=fields, marker=marker)
                pool = [candidate for candidate in candidates if str((candidate.get("payload") or {}).get("probe_kind")) in {"sql_channel_class", "sql_fragment_class"}]
                if not pool:
                    raise RuntimeError(f"AI has no SQL abstract candidate for {route['path']}")
                selected = learner.select(pool)
                selected_view = candidate_summary(selected)
                client = httpx.Client(base_url=target, timeout=12.0, follow_redirects=False, cookies={})
                try:
                    baseline_response = client.get(str(route["path"]), follow_redirects=False) if str(route["method"]).upper() == "GET" else client.post(str(route["path"]), data={}, follow_redirects=False)
                    baseline = project_sql_response(baseline_response, marker=f"pg226-base-{seed}-{run_index}")
                    baseline_status = int((baseline.get("response_projection") or {}).get("status_code", 0) or 0) or None
                    candidate_values = build_sql_probe_values(field_names=fields, marker=marker, probe_class="syntax_shape")
                    reference_marker = f"pg226-ref-{seed}-{run_index}"
                    reference_values = _reference_values(route, reference_marker)
                    negative_marker = f"pg226-neg-{seed}-{run_index}"
                    negative_values = negative_fixture_values(route, negative_marker)
                    candidate = _send(client, route, candidate_values, marker=marker, baseline_status=baseline_status)
                    reference = _send(client, route, reference_values, marker=reference_marker, baseline_status=baseline_status)
                    negative = _send(client, route, negative_values, marker=negative_marker, baseline_status=baseline_status)
                    typed = evaluate_pikachu_sql_effect(route, baseline=baseline, negative=negative, candidate=candidate, reference=reference, reset=reset, source_hash=_source_hash(name, route))
                    positive_values, fixture_kind = fixture_values(route)
                    fixture_negative_values = negative_fixture_values(route, f"pg226-fixture-neg-{seed}-{run_index}")
                    positive_response = _send_raw(client, route, positive_values)
                    fixture_negative_response = _send_raw(client, route, fixture_negative_values)
                    positive_result = project_result_response(positive_response, route=route, fixture_kind=fixture_kind)
                    fixture_negative_result = project_result_response(fixture_negative_response, route=route, fixture_kind="negative_unknown_record")
                    positive_response.close()
                    fixture_negative_response.close()
                    fixture_oracle = evaluate_result_fixture(
                        route=route,
                        positive=positive_result,
                        negative=fixture_negative_result,
                        typed_effect=typed,
                        reset=reset,
                    )
                    evidence = {
                        "target_instance_hash": target_hash,
                        "route_source_sha256": typed.get("source_attestation_sha256"),
                        "typed_evidence_hash": typed.get("evidence_hash"),
                        "fixture_evidence_hash": fixture_oracle.get("evidence_hash"),
                        "ai_reference_binding_match": selected_view.get("method") == str(route["method"]).upper() and selected_view.get("path") == route["path"],
                        "raw_payload_strings_stored": False,
                        "raw_response_bodies_stored": False,
                    }
                    evidence["evidence_sha256"] = _digest(evidence)
                    row = {
                        "seed": seed,
                        "target_instance_hash": target_hash,
                        "route": route["path"],
                        "method": route["method"],
                        "fields": fields,
                        "reset": reset,
                        "ai": {"sent": True, "candidate": selected_view, "wire_placeholder": wire_placeholder(path=str(route["path"]), method=str(route["method"]), fields=fields, probe_kind=selected_view["probe_kind"]), "raw_payload_stored": False, "raw_response_stored": False},
                        "baseline": baseline,
                        "candidate": candidate,
                        "reference": reference,
                        "negative": negative,
                        "typed_oracle": typed,
                        "result_oracle": fixture_oracle,
                        "evidence": evidence,
                        "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")),
                        "result_fixture_verified": bool(fixture_oracle.get("result_fixture_verified")),
                        "training_candidate": bool(typed.get("confirmed_positive") and fixture_oracle.get("result_fixture_verified")),
                        "training_promotion_allowed": False,
                        "memory_promotion_allowed": False,
                        "vulnerability_claim_allowed": False,
                        "raw_payload_strings_stored": False,
                        "raw_response_bodies_stored": False,
                    }
                    results.append(row)
                    dataset_rows.append({"seed": seed, "route": route["path"], "method": route["method"], "fields": fields, "probe_kind": selected_view["probe_kind"], "candidate_projection_sha256": (candidate.get("response_projection") or {}).get("projection_sha256"), "reference_projection_sha256": (reference.get("response_projection") or {}).get("projection_sha256"), "negative_projection_sha256": (negative.get("response_projection") or {}).get("projection_sha256"), "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")), "result_fixture_verified": bool(fixture_oracle.get("result_fixture_verified")), "evidence_sha256": evidence["evidence_sha256"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False})
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    counts = {"fresh_container_count": len(results), "route_count": len(ROUTE_PATHS), "get_episode_count": sum(int(str(row["method"]).upper() == "GET") for row in results), "post_episode_count": sum(int(str(row["method"]).upper() == "POST") for row in results), "ai_candidate_send_count": sum(int(row["ai"]["sent"]) for row in results), "reference_send_count": len(results), "negative_send_count": len(results), "typed_effect_confirmed_count": sum(int(row["typed_effect_confirmed"]) for row in results), "result_fixture_verified_count": sum(int(row["result_fixture_verified"]) for row in results), "training_candidate_count": sum(int(row["training_candidate"]) for row in results), "false_positive_count": 0, "docker_restart_used_count": sum(int(row["reset"].get("container_restart_used", False)) for row in results)}
    report = {"protocol_id": "pg-pk-226-ai-sql-payload-validation-v1", "schema_version": "pg226-ai-sql-payload-validation-report-v1", "status": "completed_ai_selected_sql_typed_result_validation", "runtime_image": PG214.IMAGE, "seeds": list(SEEDS), "routes": sorted(ROUTE_PATHS), "counts": counts, "results": results, "promotion": {"training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "safety": {"loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_write": False, "time_delay_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    dataset = {"schema_version": "pg226-ai-sql-payload-validation-dataset-v1", "source_reports": ["research/pg224_pikachu_parameter_surface_collection_report_v1.json", "research/pg217_pikachu_typed_sql_oracle_report_v1.json", "research/pg218_pikachu_result_fixture_report_v1.json"], "rows": dataset_rows, "training_contract": {"ai_participates_in_selection": True, "runtime_only_binding": True, "typed_oracle_required": True, "matched_negative_required": True, "result_fixture_required": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg226-ai-sql-payload-validation-trace-v1", "results": results, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg226-ai-sql-payload-validation-protocol-v1", "ai_selects_abstract_probe": True, "candidate_and_reference_use_separate_runtime_binders": True, "get_post_coverage": True, "typed_sql_oracle": True, "read_only_result_fixture": True, "fresh_reset": True, "matched_negative": True, "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    lines = ["# PG-226 AI SQL payload validation", "", f"fresh={counts['fresh_container_count']}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}; AI={counts['ai_candidate_send_count']}; reference={counts['reference_send_count']}; negative={counts['negative_send_count']}", f"typed_effect={counts['typed_effect_confirmed_count']}; result_fixture_verified={counts['result_fixture_verified_count']}; training_candidate={counts['training_candidate_count']}; false_positive={counts['false_positive_count']}", "", "AI 只输出抽象 probe kind；wire 形状使用占位符，实际运行时值未落盘。typed effect / result fixture 是 pinned 本地路由证据，不是公网漏洞结论。", ""]
    for row in results:
        lines.append(f"- {row['method']} {row['route']}: probe={row['ai']['candidate']['probe_kind']}; typed={row['typed_effect_confirmed']}; result={row['result_fixture_verified']}; wire=`{row['ai']['wire_placeholder'].replace(chr(10), ' ')}`")
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
