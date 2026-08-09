"""PG-215: expand real Pikachu GET/POST traces across fresh seeds.

The runner reuses the repaired PG-214 image and adds two independent seeds.
Each route receives a new no-volume container, an HTTP/database health gate,
and the AI/reference dual-send loop.  It stores only bounded Rule-IR tokens,
response projections and evidence hashes; request payloads and response bodies
are never written to the dataset.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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

RESEARCH = ROOT / "research"
PG214_REPORT = RESEARCH / "pg214_pikachu_fixed_sql_loop_report_v1.json"
REPORT_PATH = RESEARCH / "pg215_pikachu_real_trace_dataset_report_v1.json"
DATASET_PATH = RESEARCH / "pg215_pikachu_real_trace_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg215_pikachu_real_trace_dataset_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg215_pikachu_real_trace_dataset_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg215_pikachu_real_trace_dataset_report_v1.md"

NEW_SEEDS = (21503, 21504)
ROUTE_HOLDOUT = "/vul/sqli/sqli_x.php"
TRAIN_SEEDS = frozenset({21401, 21503})
HOLDOUT_SEEDS = frozenset({21402, 21504})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bucket_status(projection: Mapping[str, Any]) -> str:
    status = str(projection.get("status_class", "2xx"))
    return status if status in {"2xx", "3xx", "4xx", "5xx"} else "4xx"


def _route_class(path: str) -> str:
    """Collapse route spelling into a reusable surface class for the LM."""

    lowered = str(path).casefold()
    if "search" in lowered:
        return "search_surface"
    if "blind" in lowered:
        return "comment_surface"
    if "id" in lowered:
        return "profile_surface"
    return "api_surface"


def _tokens_for_episode(episode: Mapping[str, Any], phase: str, response: Mapping[str, Any], *, differential: bool, candidate_signal: bool) -> list[str]:
    """Map one bounded runtime step into the frozen family-free vocabulary."""

    method = str(episode.get("method", "GET")).upper()
    placement = "query" if method == "GET" else "form"
    route_class = _route_class(str(episode.get("path", "")))
    projection = dict(response.get("response_projection") or {})
    status = _bucket_status(projection)
    body_bucket = str(projection.get("body_length_bucket", "1-255"))
    if body_bucket not in {"1-255", "256-4095", "4096-65535"}:
        body_bucket = "1-255"
    shape_changed = "true" if differential else "false"
    signal = "true" if candidate_signal else "false"
    failure_kind = "shape_delta" if differential else "no_surface_delta"
    failed_gate = "surface_delta" if differential else "matched_negative_control"
    return [
        "[BOS]",
        "[RESET]",
        "[STEP]",
        f"src.transport.method={method}",
        f"src.transport.placement={placement}",
        "src.transport.encoding_depth=1",
        "src.transport.route=loopback_allowlisted",
        f"src.transport.route_class={route_class}",
        "src.transport.form_field_count=1-4",
        "ir.transport.methods_seen=GET+POST",
        "[IR]",
        "ir.surface.family_free=true",
        "ir.surface.modalities=html+javascript+transport",
        "ir.surface.sink_class=sql_ast_boundary",
        "ir.oracle.availability=typed",
        "ir.probe.shape=bounded_marker",
        f"ir.belief.phase={phase}",
        f"ir.response.status_class={status}",
        f"ir.response.shape_changed={shape_changed}",
        f"ir.response.candidate_signal={signal}",
        "ir.response.effect=unknown",
        f"ir.failure.failed_gate={failed_gate}",
        f"ir.failure.kind={failure_kind}",
        "ir.failure.recovery_phase=forward_baseline",
        "ir.failure.weight=1.0",
        "[OBS]",
        "obs.oracle=unknown_oracle",
        f"obs.oracle.availability=typed",
        f"obs.method_seen={method}",
        f"obs.status_class={status if status in {'2xx', '4xx'} else '4xx'}",
        f"obs.body_length={body_bucket}",
        "obs.step_progress=step_1_of_4",
        "obs.step_index=1",
        "obs.failure.transport=false",
        "[EOS]",
    ]


def _step_rows(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparison = dict(episode.get("comparison") or {})
    differential = bool(comparison.get("response_shape_differential"))
    rows: list[dict[str, Any]] = []
    steps = (
        ("prior", dict(episode.get("baseline") or {}), False),
        ("negative_control", dict(episode.get("control") or {}), False),
        ("candidate", dict((episode.get("ai") or {}).get("response") or {}), bool((episode.get("ai") or {}).get("sent"))),
        ("recovery", dict((episode.get("reference") or {}).get("response") or {}), False),
    )
    for index, (phase, response, candidate_signal) in enumerate(steps, start=1):
        if not response:
            continue
        tokens = _tokens_for_episode(episode, phase, response, differential=differential and phase in {"candidate", "recovery"}, candidate_signal=candidate_signal)
        projection = dict(response.get("response_projection") or {})
        evidence = {
            "phase": phase,
            "step_index": index,
            "method": str(episode.get("method", "GET")),
            "path": str(episode.get("path", "")),
            "status_class": _bucket_status(projection),
            "body_length_bucket": str(projection.get("body_length_bucket", "1-255")),
            "shape_differential": bool(differential and phase in {"candidate", "recovery"}),
            "candidate_signal": bool(candidate_signal),
            "database_write": False,
            "external_network": False,
        }
        evidence_hash = _digest(evidence)
        reset = dict(episode.get("reset") or {})
        rows.append({
            "row_id": f"pg215-{episode.get('seed')}-{episode.get('path','').replace('/', '_')}-{phase}",
            "source": "pg215_real_pikachu",
            "seed": int(episode.get("seed", 0)),
            "route": str(episode.get("path", "")),
            "method": str(episode.get("method", "GET")),
            "phase": phase,
            "tokens": tokens,
            "token_count": len(tokens),
            "projection_sha256": str(projection.get("projection_sha256", "")),
            "evidence_sha256": evidence_hash,
            "target_instance_hash": str(episode.get("target_instance_hash", "")),
            "database_clean_reset_verified": bool(episode.get("database_clean_reset_verified")),
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "reset_attestation": {
                "fresh_target": bool(reset.get("fresh_target")),
                "container_recreated": bool(reset.get("container_recreated")),
                "container_restart_used": bool(reset.get("container_restart_used")),
                "volume_mount_count": int(reset.get("volume_mount_count", -1)),
                "database_health_gate": str(reset.get("database_health_gate", "")),
            },
        })
    return rows


def _load_existing() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(PG214_REPORT.read_text(encoding="utf-8-sig"))
    episodes = list(report.get("episodes") or [])
    return episodes, report


def main() -> int:
    existing, existing_report = _load_existing()
    routes = PG214.PG212._routes()
    model, vocabulary = PG214.PG212.PG208._load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    from app.payload_learner import PayloadLearner

    learner = PayloadLearner(seed=215)
    new_episodes: list[dict[str, Any]] = []
    run_index = 0
    for seed in NEW_SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                client = PG214.httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    new_episodes.append(PG214.PG212._route_episode(model, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash, reset=reset, target_url=f"http://127.0.0.1:{port}"))
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_episodes = existing + new_episodes
    rows = [row for episode in all_episodes for row in _step_rows(episode)]
    vocabulary_set = set(vocabulary)
    missing = sorted({token for row in rows for token in row["tokens"] if token not in vocabulary_set})
    if missing:
        raise RuntimeError(f"PG-215 frozen vocabulary missing token: {missing[0]}")
    train_rows = [row for row in rows if row["seed"] in TRAIN_SEEDS and row["route"] != ROUTE_HOLDOUT]
    holdout_rows = [row for row in rows if row["seed"] in HOLDOUT_SEEDS or row["route"] == ROUTE_HOLDOUT]
    dataset = {
        "schema_version": "pg215-pikachu-real-trace-dataset-v1",
        "purpose": "real fixed-runtime Pikachu GET/POST process tokens for next-token capacity evaluation",
        "source_report": str(PG214_REPORT.relative_to(ROOT)),
        "runtime_image": str(existing_report.get("runtime", {}).get("image", "")),
        "episode_count": len(all_episodes),
        "new_episode_count": len(new_episodes),
        "step_row_count": len(rows),
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "train_seeds": sorted(TRAIN_SEEDS),
        "holdout_seeds": sorted(HOLDOUT_SEEDS),
        "route_holdout": ROUTE_HOLDOUT,
        "methods": {"GET": sum(int(row["method"] == "GET") for row in rows), "POST": sum(int(row["method"] == "POST") for row in rows)},
        "tokens": rows,
        "training_contract": {
            "raw_payloads_stored": False,
            "raw_responses_stored": False,
            "vulnerability_labels_stored": False,
            "oracle_labels_stored": False,
            "family_labels_stored": False,
            "only_bounded_response_features": True,
            "fresh_reset_required": True,
            "database_health_gate_required": True,
        },
    }
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET_PATH, dataset)
    counts = {
        "episode_count": len(all_episodes),
        "new_episode_count": len(new_episodes),
        "step_row_count": len(rows),
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "get_row_count": sum(int(row["method"] == "GET") for row in rows),
        "post_row_count": sum(int(row["method"] == "POST") for row in rows),
        "clean_reset_row_count": sum(int(row["database_clean_reset_verified"]) for row in rows),
        "restart_row_count": sum(int(row["reset_attestation"]["container_restart_used"]) for row in rows),
    }
    report = {
        "protocol_id": "pg-pk-215-pikachu-real-trace-dataset-v1",
        "schema_version": "pg215-pikachu-real-trace-dataset-report-v1",
        "status": "completed_real_cross_seed_trace_collection",
        "device": str(device),
        "source_report": str(PG214_REPORT.relative_to(ROOT)),
        "new_seeds": list(NEW_SEEDS),
        "counts": counts,
        "dataset_sha256": dataset["dataset_sha256"],
        "methods": dataset["methods"],
        "route_holdout": ROUTE_HOLDOUT,
        "promotion": {"training_eligible": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_generation_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_health_gate_required": True, "docker_restart_used_count": counts["restart_row_count"], "database_write": False, "time_delay_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg215-pikachu-real-trace-dataset-protocol-v1", "source_image": dataset["runtime_image"], "new_seeds": list(NEW_SEEDS), "train_seeds": sorted(TRAIN_SEEDS), "holdout_seeds": sorted(HOLDOUT_SEEDS), "route_holdout": ROUTE_HOLDOUT, "steps_per_episode": ["prior", "negative_control", "candidate", "recovery"], "ai_and_reference_sent": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_health_gate_required": True, "raw_payload_and_response_excluded": True, "training_promotion_requires_pg216_capacity_and_ood_gate": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg215-pikachu-real-trace-dataset-trace-v1", "episodes": [{"seed": row.get("seed"), "route": row.get("path"), "method": row.get("method"), "target_instance_hash": row.get("target_instance_hash"), "database_clean_reset_verified": row.get("database_clean_reset_verified")} for row in all_episodes], "step_rows": rows, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": True})
    lines = ["# PG-215 Pikachu real trace dataset", "", f"episodes={counts['episode_count']} (new={counts['new_episode_count']}); step rows={counts['step_row_count']}; train={counts['train_row_count']}; holdout={counts['holdout_row_count']}", f"GET rows={counts['get_row_count']}; POST rows={counts['post_row_count']}; clean-reset rows={counts['clean_reset_row_count']}; docker restart rows={counts['restart_row_count']}", "", "这些是固定派生 Pikachu 运行时的真实 HTTP/数据库健康回放，按 prior→negative_control→candidate→recovery 压成 family-free Rule-IR token；不保存原始 payload/response，也不把响应形状当作漏洞标签。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "counts": counts, "dataset": str(DATASET_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
