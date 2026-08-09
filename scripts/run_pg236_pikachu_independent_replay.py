"""PG-236: collect fresh safe GET/POST Pikachu trajectories from the second image.

The runner is deliberately limited to PG-51's inert marker probes.  It sends a
candidate and a separately bound reference on fresh, mount-free containers and
keeps only projections/hashes for the training catalog.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg51_docker_replay import SAFE_PATHS, collect_pair  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402
from app.pg233_cross_family_capacity import add_family_context, family_class  # noqa: E402


RESEARCH = ROOT / "research"
REGISTRY = RESEARCH / "pg_pk_24_cross_lab_registry_v1.json"
RUN_SUFFIX = str(os.environ.get("PG236_RUN_SUFFIX", "seed" + str(os.environ.get("PG236_SEED", "23621"))))
DATASET = RESEARCH / f"pg236_pikachu_independent_replay_dataset_{RUN_SUFFIX}.json"
REPORT = RESEARCH / f"pg236_pikachu_independent_replay_report_{RUN_SUFFIX}.json"
TRACE = RESEARCH / f"pg236_pikachu_independent_replay_trace_{RUN_SUFFIX}.json"
PROTOCOL = RESEARCH / f"pg236_pikachu_independent_replay_protocol_{RUN_SUFFIX}.json"
MARKDOWN = RESEARCH / f"pg236_pikachu_independent_replay_report_{RUN_SUFFIX}.md"
SEED = int(os.environ.get("PG236_SEED", "23621"))
SEEDS = (SEED,)


def _load_pg214() -> Any:
    path = ROOT / "scripts" / "run_pg214_pikachu_fixed_sql_loop.py"
    spec = importlib.util.spec_from_file_location("pg214_for_pg236", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-214 container manager")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load_pg214()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _source() -> dict[str, Any]:
    return {
        "target_id": "pg236_pikachu_fixed_independent",
        "app_family": "pikachu",
        "source_id": "pikachu-local-container-pg-pk-01",
        "source_type": "authorized_local_container",
        "origin_ref": "pg236-independent-sift-pikachu-fixed",
        "license": "local-container",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": "127.0.0.1", "port": 8767},
        "container_image_digest": PG214.IMAGE.split("@", 1)[1] if "@" in PG214.IMAGE else PG214.IMAGE,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reset_adapter_sha256": hashlib.sha256((ROOT / "scripts" / "run_pg214_pikachu_fixed_sql_loop.py").read_bytes()).hexdigest(),
        "oracle_contract_sha256": _digest({"protocol": "pg236-projection-only-reference-negative-v1"}),
        "read_only": True,
        "external_network": False,
    }


def _send_reference(client: httpx.Client, spec: dict[str, Any], marker: str) -> dict[str, Any]:
    normalized = dict(spec)
    query_name = str(normalized["query_name"])
    params = {query_name: marker, "submit": "submit"}
    baseline = client.get(str(normalized["path"]), follow_redirects=False)
    if str(normalized["method"]).upper() == "POST":
        response = client.post(str(normalized["path"]), data=params, headers={"content-type": "application/x-www-form-urlencoded"}, follow_redirects=False)
    else:
        response = client.get(str(normalized["path"]), params=params, follow_redirects=False)
    # Reuse the bounded adapter projection and close response bodies immediately.
    from app.pg51_docker_replay import projection
    signal = projection(response, baseline, marker)
    result = {"signal": signal, "signal_sha256": _digest(signal), "status_class": f"{response.status_code // 100}xx", "raw_response_retained": False}
    response.close()
    baseline.close()
    return result


def _normalize(candidate: dict[str, Any], reference: dict[str, Any], control: dict[str, Any], *, seed: int, route: str, family: str, method: str, surface_variant: str) -> dict[str, Any]:
    oracle = candidate.get("oracle_projection") or {}
    candidate_signals = oracle.get("signals") or {}
    ref_signals = reference.get("signal") or {}
    control_oracle = control.get("oracle_projection") or {}
    control_signals = control_oracle.get("signals") or {}
    agreement_keys = ("marker_reflected", "marker_in_script_source", "marker_in_attribute", "sql_error_shape", "external_redirect")
    agreement = all(bool(candidate_signals.get(key, False)) == bool(ref_signals.get(key, False)) for key in agreement_keys)
    negative_clean = not any(bool(control_signals.get(key, False)) for key in agreement_keys)
    response = candidate.get("response_projection") or {}
    raw = {
        "source": "pg236_pikachu_fixed_independent",
        "seed": int(seed),
        "surface_role": {"sql": "sql_surface", "dom": "xss_surface", "redirect": "redirect_surface"}.get(family, "generic_surface"),
        "method": str(method).upper(),
        "status_class": str(response.get("status_class", "unknown")),
        "field_count": int((response.get("shape") or {}).get("field_count", 0) or 0),
        "history_len": 1,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "reset_not_attempted": False,
        "candidate_sent": True,
        "oracle_available": False,
        "typed_effect_confirmed": False,
        "typed_effect_observed": False,
        "result_fixture_verified": False,
        "candidate_reference_agreement": agreement,
        "negative_clean": negative_clean,
        "binding_valid": True,
        "transport_error": bool(response.get("transport_error", False)),
        "result_mismatch_observed": False,
        "next_step": "recheck_oracle",
        "previous_feedback": "none",
        "candidate_result_present": any(bool(candidate_signals.get(key, False)) for key in agreement_keys),
        "candidate_sql_error_shape": bool(candidate_signals.get("sql_error_shape", False)),
        "boolean_differential": False,
        "negative_result_absent": negative_clean,
        "hard_gate_observed": False,
        "backend_observed": True,
        "database_health_ok": True,
        "reference_sent": True,
        "negative_sent": True,
        "model_abstained": True,
        "model_claimed_positive": False,
        "evidence_hash": str((candidate.get("evidence") or {}).get("evidence_hash", "")),
        "payload_grounded_eligible": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    if len(raw["evidence_hash"]) != 64:
        raw["evidence_hash"] = _digest({"source": raw["source"], "seed": seed, "route": route, "method": method})
    result = add_family_context(prepare_feedback_record(raw), family=family, channel=method, pair_role="candidate", source_role="observed")
    tokens = list(result["tokens"])
    failure_index = next((index for index, token in enumerate(tokens) if str(token).startswith("failure=")), len(tokens) - 1)
    tokens.insert(failure_index, f"surface_variant={surface_variant}")
    result.update({"tokens": tokens, "trajectory_hash": _digest(tokens), "classification_position": int(result.get("classification_position", failure_index)) + 1, "surface_variant": surface_variant})
    result.update({"candidate_reference_agreement": agreement, "negative_clean": negative_clean, "source_family": family})
    return result


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8-sig"))
    source = _source()
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    containers = 0
    run_index = 0
    for seed in SEEDS:
        for path, spec in sorted(SAFE_PATHS.items()):
            for method in ("GET", "POST"):
                name = ""
                try:
                    # PG-51's adapter intentionally allow-lists only its two
                    # loopback ports.  Reuse one port per method sequentially;
                    # each episode still gets a new container and --rm.
                    episode_index = 1000 + run_index
                    PG214.BASE_PORT = (8767 if method == "GET" else 8768) - episode_index
                    name, port, container_id, reset = PG214._start(seed, episode_index)
                    containers += 1
                    target = f"http://127.0.0.1:{port}"
                    client = httpx.Client(base_url=target, timeout=12.0, follow_redirects=False, cookies={})
                    try:
                        marker = f"pg236-{seed}-{run_index:03d}"
                        raw_spec = {"target": target, "path": path, "method": method, "marker": marker}
                        control, candidate, info = collect_pair(source=source, registry=registry, spec=raw_spec, client=client, target_instance_id=container_id[:24], reset_id=f"pg236-reset-{seed}-{run_index:03d}")
                        ref_marker = f"pg236-ref-{seed}-{run_index:03d}"
                        normalized_spec = {"path": path, "method": method, "query_name": spec["query_name"]}
                        reference = _send_reference(client, normalized_spec, ref_marker)
                        record = _normalize(candidate, reference, control, seed=seed, route=path, family={"injection": "sql", "xss": "dom", "url_redirect": "redirect"}.get(spec["family"], "generic"), method=method, surface_variant=str(spec["surface"]))
                        records.append(record)
                        trace_rows.append({"seed": seed, "route_family": {"injection": "sql", "xss": "dom", "url_redirect": "redirect"}.get(spec["family"], "generic"), "method": method, "candidate_evidence_hash": record["source_evidence_hash"], "reference_signal_sha256": reference["signal_sha256"], "candidate_reference_agreement": bool(record["candidate_reference_agreement"]), "negative_clean": bool(record["negative_clean"]), "fresh_reset": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
                    finally:
                        client.close()
                finally:
                    if name:
                        PG214._stop(name)
                run_index += 1
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for record in records:
        key = str(record["trajectory_hash"])
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(record)
    counts = {"seed_count": len(SEEDS), "route_count": len(SAFE_PATHS), "fresh_container_count": containers, "raw_record_count": len(records), "unique_record_count": len(unique), "duplicate_record_count": duplicate_count, "get_record_count": sum(int(row["method"] == "GET") for row in records), "post_record_count": sum(int(row["method"] == "POST") for row in records), "family_counts": {family: sum(int(row["family_class"] == family) for row in records) for family in ("sql", "dom", "redirect")}, "reference_sent_count": len(records), "negative_control_count": len(records), "oracle_available_count": 0, "model_self_error_count": 0}
    dataset = {"schema_version": "pg236-pikachu-independent-replay-dataset-v1", "source": {"implementation": "sift-pikachu-fixed", "image": PG214.IMAGE, "loopback_only": True, "external_network": False}, "records": unique, "counts": counts, "contract": {"fresh_reset_per_route_episode": True, "candidate_reference_negative": True, "projection_only": True, "typed_oracle_unavailable_is_silver": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    report = {"protocol_id": "pg-pk-236-pikachu-independent-replay-v1", "schema_version": "pg236-pikachu-independent-replay-v1", "status": "completed_fresh_independent_pikachu_get_post_projection_replay", "image": PG214.IMAGE, "counts": counts, "dataset_file": str(DATASET.relative_to(ROOT)), "promotion": {"training_promotion_allowed": True, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "safety": {"loopback_only": True, "external_network": False, "fresh_container_per_route_episode": True, "database_write": False, "script_execution": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg236-pikachu-independent-replay-protocol-v1", "methods": ["GET", "POST"], "families": ["sql", "dom", "redirect"], "fresh_reset_per_route_episode": True, "ai_candidate": True, "independent_reference": True, "matched_negative": True, "typed_oracle": False, "silver_abstention_only": True, "raw_payload_and_response_excluded": True, "memory_promotion_blocked": True}
    protocol["protocol_sha256"] = _digest(protocol)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE.write_text(json.dumps({"schema_version": "pg236-pikachu-independent-replay-trace-v1", "rows": trace_rows, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-236 independent Pikachu replay", "", f"containers={containers}; raw={len(records)}; unique={len(unique)}; duplicates={duplicate_count}; GET={counts['get_record_count']}; POST={counts['post_record_count']}", f"families={counts['family_counts']}", "每个 route×method 都使用 fresh container；candidate/reference/negative 只保留 projection 和哈希，未见 typed oracle 的记录只能训练 abstain。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
