"""PG-224: ground the browser-discovered GET/POST surfaces in fresh local runs.

The crawl manifest is authoritative for paths and field names.  For each
allow-listed read-only surface the AI selects an abstract probe class, the
runtime binder creates a bounded canary, and the runner records baseline,
candidate, negative and redirect projections.  Stateful/unsafe routes remain
visible in the dataset as explicit preflight-only rows instead of being sent.
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
from app.pg224_surface_projector import (  # noqa: E402
    PG224_SCHEMA,
    build_runtime_values,
    project_response,
    route_policy,
    sha256_json,
    wire_placeholder,
)
from app.payload_learner import PayloadLearner  # noqa: E402


RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
REPORT = RESEARCH / "pg224_pikachu_parameter_surface_collection_report_v1.json"
DATASET = RESEARCH / "pg224_pikachu_parameter_surface_dataset_v1.json"
TRACE = RESEARCH / "pg224_pikachu_parameter_surface_trace_v1.json"
PROTOCOL = RESEARCH / "pg224_pikachu_parameter_surface_protocol_v1.json"
MARKDOWN = RESEARCH / "pg224_pikachu_parameter_surface_report_v1.md"
SEEDS = (22401, 22402)
BASE_RUN_INDEX = 600


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load_script("run_pg214_pikachu_fixed_sql_loop.py")
PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _family(family_hint: str) -> str:
    value = str(family_hint)
    return {"ordinary_response": "logic", "input_validation": "logic", "authentication": "access_control"}.get(value, value)


def _matrix() -> list[dict[str, Any]]:
    matrix = [dict(row) for row in PG191._load_matrix()]
    # PG-191's assertion is the crawl/data-lineage check: 44 unique
    # method/path/field surfaces, not an invented route list.
    if len(matrix) != 44:
        raise RuntimeError(f"PG-224 expected 44 parameterized surfaces, got {len(matrix)}")
    return matrix


def _candidate(*, row: Mapping[str, Any], target: str, marker: str, learner: PayloadLearner) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    family = _family(str(row.get("family_hint", "logic")))
    try:
        candidates = generate_grounded_candidates(
            family=family,
            target=target,
            path=str(row["path"]),
            method=str(row["method"]),
            fields=list(row["field_names"]),
            marker=marker,
        )
    except (ValueError, KeyError):
        return None, None
    # Keep only abstract classes whose runtime binder is non-destructive for
    # this route.  The learner still decides among the remaining candidates.
    allowed_kinds = {"sql_channel_class", "sql_fragment_class"} if family == "injection" else {"http_canary", "inert_dom_markup", "encoded_dom_markup"}
    pool = [candidate for candidate in candidates if str((candidate.get("payload") or {}).get("probe_kind")) in allowed_kinds]
    if not pool:
        return None, None
    selected = learner.select(pool)
    return selected, candidate_summary(selected)


def _request(client: httpx.Client, *, method: str, path: str, values: Mapping[str, str]) -> httpx.Response:
    if str(method).upper() == "POST":
        return client.post(path, data=dict(values), follow_redirects=False)
    return client.get(path, params=dict(values), follow_redirects=False)


def _safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    safe = dict(row)
    for key in ("raw_payload", "payload", "raw_response", "response_body"):
        safe.pop(key, None)
    return safe


def main() -> int:
    matrix = _matrix()
    learner = PayloadLearner(seed=224)
    results: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route_index, route in enumerate(matrix):
            path = str(route["path"])
            method = str(route["method"]).upper()
            fields = [str(item) for item in route["field_names"]]
            policy = route_policy(path, method, fields)
            marker = f"pg224-{seed}-{route_index:02d}"
            base_record: dict[str, Any] = {
                "seed": seed,
                "route_id": str(route["route_id"]),
                "route": path,
                "method": method,
                "fields": fields,
                "family_hint": str(route.get("family_hint", "ordinary_response")),
                "source_row_sha256": str(route.get("source_row_sha256", "")),
                "policy": policy,
                "fresh_reset": False,
                "ai": {"sent": False, "abstained": False, "raw_payload_stored": False},
                "request_anatomy": {"method": method, "path": path, "fields": fields, "wire_placeholder": None},
                "baseline": None,
                "candidate": None,
                "negative": None,
                "oracle": {"typed_available": False, "projection_only": True, "vulnerability_claim_allowed": False},
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
            if not policy["send_allowed"]:
                base_record["status"] = "preflight_only"
                base_record["request_anatomy"]["wire_placeholder"] = wire_placeholder(path=path, method=method, fields=fields, probe_kind="http_canary")
                results.append(base_record)
                dataset_rows.append(_safe_row(base_record))
                continue
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, BASE_RUN_INDEX + run_index)
                target = f"http://127.0.0.1:{port}"
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                selected, selected_view = _candidate(row=route, target=target, marker=marker, learner=learner)
                if selected is None or selected_view is None:
                    base_record["status"] = "ai_abstain_no_grounded_candidate"
                    base_record["fresh_reset"] = bool(reset.get("fresh_target"))
                    results.append(base_record)
                    dataset_rows.append(_safe_row(base_record))
                    continue
                probe_kind = str(selected_view["probe_kind"])
                client = httpx.Client(base_url=target, timeout=12.0, follow_redirects=False, cookies={})
                try:
                    # A GET baseline is harmless even for a POST-backed form;
                    # the candidate is sent only through the observed method.
                    baseline_response = client.get(path, follow_redirects=False)
                    baseline = project_response(baseline_response, marker=marker)
                    baseline_projection = baseline["response_projection"]
                    candidate_values = build_runtime_values(path=path, method=method, fields=fields, marker=marker, probe_kind=probe_kind, control=False)
                    negative_values = build_runtime_values(path=path, method=method, fields=fields, marker=f"{marker}-n", probe_kind=probe_kind, control=True)
                    candidate_response = _request(client, method=method, path=path, values=candidate_values)
                    negative_response = _request(client, method=method, path=path, values=negative_values)
                    candidate = project_response(candidate_response, marker=marker, baseline=baseline_projection)
                    negative = project_response(negative_response, marker=f"{marker}-n", baseline=baseline_projection)
                    evidence = {
                        "source_row_sha256": route.get("source_row_sha256"),
                        "target_instance_hash": target_hash,
                        "baseline_projection_sha256": baseline_projection.get("projection_sha256"),
                        "candidate_projection_sha256": candidate["response_projection"].get("projection_sha256"),
                        "negative_projection_sha256": negative["response_projection"].get("projection_sha256"),
                        "candidate_negative_status_shape_differential": candidate["response_projection"].get("status_class") != negative["response_projection"].get("status_class"),
                        "candidate_negative_body_shape_differential": candidate["response_projection"].get("body_length_bucket") != negative["response_projection"].get("body_length_bucket"),
                        "typed_available": False,
                        "vulnerability_claim_allowed": False,
                    }
                    evidence["evidence_sha256"] = sha256_json(evidence)
                    base_record.update({
                        "status": "completed_projection_only",
                        "target_instance_hash": target_hash,
                        "fresh_reset": bool(reset.get("fresh_target")),
                        "reset": reset,
                        "ai": {"sent": True, "abstained": False, "candidate": selected_view, "raw_payload_stored": False},
                        "request_anatomy": {"method": method, "path": path, "fields": fields, "wire_placeholder": wire_placeholder(path=path, method=method, fields=fields, probe_kind=probe_kind)},
                        "baseline": baseline,
                        "candidate": candidate,
                        "negative": negative,
                        "oracle": {"typed_available": False, "projection_only": True, "evidence": evidence, "vulnerability_claim_allowed": False},
                    })
                    results.append(base_record)
                    dataset_rows.append(_safe_row(base_record))
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    counts = {
        "route_inventory_count": len(matrix),
        "seed_count": len(SEEDS),
        "surface_observation_count": len(results),
        "safe_send_count": sum(int(row["ai"]["sent"]) for row in results),
        "get_candidate_send_count": sum(int(row["ai"]["sent"]) and row["method"] == "GET" for row in results),
        "post_candidate_send_count": sum(int(row["ai"]["sent"]) and row["method"] == "POST" for row in results),
        "preflight_only_count": sum(int(row["status"] == "preflight_only") for row in results),
        "ai_abstain_count": sum(int(row["status"] == "ai_abstain_no_grounded_candidate") for row in results),
        "baseline_projection_count": sum(int(row["baseline"] is not None) for row in results),
        "candidate_projection_count": sum(int(row["candidate"] is not None) for row in results),
        "negative_projection_count": sum(int(row["negative"] is not None) for row in results),
        "typed_effect_count": 0,
        "false_positive_count": 0,
        "docker_restart_used_count": sum(int((row.get("reset") or {}).get("container_restart_used", False)) for row in results),
    }
    report = {
        "protocol_id": "pg-pk-224-pikachu-parameter-surface-collection-v1",
        "schema_version": PG224_SCHEMA,
        "status": "completed_crawl_grounded_parameter_surface_projection",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "runtime_image": PG214.IMAGE,
        "seeds": list(SEEDS),
        "counts": counts,
        "results": results,
        "promotion": {"training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_safe_route_episode": True, "no_volume_or_bind_mount": True, "database_write": False, "time_delay_used": False, "external_network_target": False, "unsafe_routes_preflight_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    dataset = {"schema_version": "pg224-pikachu-parameter-surface-dataset-v1", "source_manifest": str(MANIFEST.relative_to(ROOT)), "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), "rows": dataset_rows, "training_contract": {"request_context_required": True, "get_post_context": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "typed_oracle_required_before_training": True, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg224-pikachu-parameter-surface-trace-v1", "results": results, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg224-pikachu-parameter-surface-protocol-v1", "crawl_manifest_authoritative": True, "expected_parameterized_surface_count": 44, "ai_selects_abstract_probe": True, "runtime_canary_binding": True, "baseline_required": True, "negative_required": True, "redirect_not_followed_cross_origin": True, "unsafe_route_preflight_only": True, "fresh_container_per_safe_route": True, "typed_oracle_required_for_training": True, "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    lines = ["# PG-224 Pikachu parameter-surface collection", "", f"routes={counts['route_inventory_count']} seeds={counts['seed_count']} safe_send={counts['safe_send_count']} GET={counts['get_candidate_send_count']} POST={counts['post_candidate_send_count']} preflight_only={counts['preflight_only_count']}", "", "AI 只选择抽象 probe_kind；实际值在 loopback 请求瞬间绑定。下面的 wire 形状使用占位符，原始值和响应正文没有保存。", "", "| method | route | fields | policy/status | wire shape |", "|---|---|---|---|---|"]
    for row in results:
        wire = str((row.get("request_anatomy") or {}).get("wire_placeholder") or "")
        lines.append(f"| {row['method']} | {row['route']} | {','.join(row['fields'])} | {row['status']} | `{wire.replace(chr(10), ' ')}` |")
    lines.extend(["", "projection-only rows do not establish a vulnerability. Typed family oracle, fresh replays and matched negative controls are required before any training promotion.", ""])
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
