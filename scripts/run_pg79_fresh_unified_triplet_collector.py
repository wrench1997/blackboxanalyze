"""PG-79: fresh unified neutral/negative/positive triplets across PG-53 fixtures.

The PG-53 implementations are already authorized, loopback-only and
read-only.  This collector intentionally bypasses the old active-policy screen
omission: every fresh target receives control, screen and candidate probes.
Only bounded projections, typed oracle metadata and hashes are persisted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step  # noqa: E402


PG53_SCRIPT = ROOT / "scripts" / "run_pg53_cross_source_typed_replay.py"
PROTOCOL_ID = "pg-pk-79-fresh-unified-triplet-collector-v1"
REPORT_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_report_v1.md"
SEEDS = (7901, 7907, 7911)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe(value: Any) -> Any:
    forbidden = {"body", "raw_body", "body_preview", "request_body", "raw_probe", "password", "token", "cookie", "authorization"}
    if isinstance(value, dict):
        return {str(key): _safe(child) for key, child in value.items() if str(key).casefold() not in forbidden}
    if isinstance(value, list):
        return [_safe(child) for child in value]
    return value


def _probe_oracle(probe: dict[str, Any], *, role: str) -> dict[str, Any]:
    oracle = _safe(dict(probe.get("oracle") or {}))
    oracle["triplet_role"] = role
    oracle["evaluator_state_hidden"] = True
    oracle.setdefault("positive_authority", False)
    oracle.setdefault("positive", False)
    oracle.setdefault("confirmed_effect", "none" if not oracle["positive"] else "typed_effect")
    return oracle


def _triplet_step(pg53: Any, target: dict[str, Any], surface: str, family: str, method: str, seed: int, ordinal: int, control: dict[str, Any], screen: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    target_id = str(candidate["fresh_reset"]["target_instance_id"])
    neutral_projection = _safe(control["response"])
    negative_projection = _safe(screen["response"])
    positive_projection = _safe(candidate["response"])
    neutral_oracle = _probe_oracle(control, role="neutral")
    negative_oracle = _probe_oracle(screen, role="negative_probe")
    positive_oracle = _probe_oracle(candidate, role="candidate")
    positive_oracle["negative_control_pair_id"] = f"pg79-control-{ordinal:04d}"
    action = {"method": str(method).upper(), "route_template_id": f"pg79-route-{ordinal:04d}", "placement": "query" if method == "GET" else "form", "encoding_chain": ["identity"], "probe_ref": f"pg79-probe-{ordinal:04d}", "probe_sha256": hashlib.sha256(b"pg79-abstract-triplet-probe").hexdigest(), "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action["form_field_names"] = ["abstract_field"]
    reset = {"fresh_target": True, "completed": True, "evaluator_state_hidden": True, "target_instance_id": target_id, "control": _safe(control["fresh_reset"]), "screen": _safe(screen["fresh_reset"]), "candidate": _safe(candidate["fresh_reset"]), "external_network": False, "state_change_allowed": False}
    evidence = sha256_json({"target_instance_id": target_id, "surface_index": ordinal, "method": method, "neutral": neutral_projection, "negative": negative_projection, "positive": positive_projection, "neutral_oracle": neutral_oracle, "negative_oracle": negative_oracle, "positive_oracle": positive_oracle, "reset": reset})
    positive = bool(positive_oracle.get("positive"))
    step = {"episode_id": f"pg79-episode-s{seed}", "step_id": f"pg79-step-{ordinal:04d}", "parent_step_id": None, "sampling_seed": int(seed), "target_instance_id": target_id, "hypothesis": "anonymous_surface_triplet", "belief_before": {"unknown_surface": 1.0}, "action_manifest": action, "baseline_projection": neutral_projection, "neutral_projection": neutral_projection, "negative_probe_projection": negative_projection, "response_projection": positive_projection, "neutral_oracle_projection": neutral_oracle, "negative_oracle_projection": negative_oracle, "oracle_projection": positive_oracle, "belief_after": {"unknown_surface": 1.0}, "decision": "confirmed_positive" if positive else "confirmed_negative", "next_action": "stop_confirmed" if positive else "continue_probe", "fresh_reset": reset, "evidence_sha256": evidence, "dataset_stage": "evaluation_only", "online_weight_update": False, "long_term_memory_write": False}
    echo_body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")}
    step["echo"] = {"sha256": sha256_json(echo_body)}
    row = {"source_id": target["source_id"], "implementation": target["implementation"], "variant": target["variant"], "surface": surface, "family": family, "method": method, "sampling_seed": int(seed), "target_instance_id": target_id, "neutral_response": neutral_projection, "negative_response": negative_projection, "positive_response": positive_projection, "neutral_oracle": neutral_oracle, "negative_oracle": negative_oracle, "positive_oracle": positive_oracle, "fresh_reset": reset, "raw_probe_stored": False, "raw_response_body_stored": False}
    return step, row


def _catalog(rows: list[dict[str, Any]]) -> None:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        payload = {"target": "http://127.0.0.1:3100", "method": row["method"], "path": "/abstract/surface", "probe": "abstract_triplet_probe", "encoding": "identity"}
        replay = {"target": "http://127.0.0.1:3100", "method": row["method"], "path": "/abstract/surface", "transport": "loopback", "fresh_reset": row["fresh_reset"]}
        if row["method"] == "POST":
            payload["form"] = {"abstract_field": "abstract_triplet_probe"}
            replay["form"] = {"abstract_field": "abstract_triplet_probe"}
        samples.append({"sample_id": f"pg79-sample-{index:04d}", "payload": payload, "probe_artifact": {"original": "abstract_triplet_probe", "encoding": "identity", "probe_sha256": hashlib.sha256(b"abstract_triplet_probe").hexdigest()}, "semantic": {"family": "logic", "surface": "anonymous_surface", "expected_oracle": "synthetic_rule_surface_v1", "expected_signal": "typed_effect_after_probe"}, "pair": {"pair_id": f"pg79-pair-{index:04d}", "variant": "identity", "surface_role": "anonymous_surface", "encoding_depth": 0}, "counterfactual": {"kind": "negative_control", "intervention": "matched_triplet", "source_sample_id": f"pg79-sample-{index:04d}"}, "replay": replay, "response_projection": row["positive_response"], "oracle_projection": row["positive_oracle"], "evidence": {"triplet_evidence_sha256": sha256_json({"neutral": row["neutral_response"], "negative": row["negative_response"], "positive": row["positive_response"], "reset": row["fresh_reset"]})}, "rule_ir": {"op": "and", "args": [{"op": "eq", "left": {"op": "field", "path": "oracle.positive"}, "right": {"op": "const", "value": True}}]}, "rule_ir_result": bool(row["positive_oracle"].get("positive")), "evaluator_state_visible": False})
    write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg79-fresh-unified-triplet-evaluation-only", "sources": [{"provenance": {"source_id": "pg53-independent-fixtures-pg79", "source_type": "in_repo_synthetic", "origin": "app/pg53_cross_source_oracle.py", "license": "in_repo_synthetic", "authorization": "workspace_local_only", "scope": ["http://127.0.0.1:3100"], "captured_at": captured_at, "authorized_for": ["training", "local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False}, "samples": samples}]})


def run() -> dict[str, Any]:
    pg53 = _load(PG53_SCRIPT, "pg79_pg53_runtime")
    steps: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ordinal = 0
    for seed in SEEDS:
        for target in pg53.TARGETS:
            for surface in pg53.SURFACES:
                for method in pg53.METHODS:
                    family = str(pg53._spec(target, surface).get("family", "unknown"))
                    try:
                        fresh_target = pg53._FreshTarget(target)
                        with fresh_target as client:
                            control = pg53._run_probe(target, surface=surface, family=family, method=method, seed=seed, stage="control", positive=False, client=client, target_instance_id=fresh_target.instance_id)
                            screen = pg53._run_probe(target, surface=surface, family=family, method=method, seed=seed, stage="screen", positive=True, client=client, target_instance_id=fresh_target.instance_id)
                            candidate = pg53._run_probe(target, surface=surface, family=family, method=method, seed=seed, stage="candidate", positive=True, client=client, target_instance_id=fresh_target.instance_id)
                        step, row = _triplet_step(pg53, target, surface, family, method, seed, ordinal, control, screen, candidate)
                        steps.append(step)
                        rows.append(row)
                        ordinal += 1
                    except Exception as exc:  # keep the run's failure surface explicit, never silently fill a row
                        errors.append({"seed": seed, "source_id": target["source_id"], "surface": surface, "method": method, "error_type": type(exc).__name__})
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[int(step["sampling_seed"])].append(step)
    normalized_steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    validation_failures: list[dict[str, Any]] = []
    for seed, episode_steps in sorted(grouped.items()):
        parent = None
        normalized_episode: list[dict[str, Any]] = []
        for raw in episode_steps:
            raw["parent_step_id"] = parent
            try:
                normalized = validate_trace_step(raw)
            except ValueError as exc:
                validation_failures.append({"step_id": raw["step_id"], "error_type": type(exc).__name__})
                normalized = raw
            normalized_episode.append(normalized)
            normalized_steps.append(normalized)
            parent = normalized["step_id"]
        validation = evaluate_episode(normalized_episode) if not validation_failures else {"status": "trace_only", "reasons": ["step_validation_failure"]}
        episodes.append({"episode_id": f"pg79-episode-s{seed}", "seed": seed, "steps": normalized_episode, "validation": validation})
    _catalog(rows)
    metrics = {"triplet_case_count": len(rows), "typed_positive_count": sum(int(row["positive_oracle"].get("positive")) for row in rows), "typed_negative_oracle_count": sum(int(not row["negative_oracle"].get("positive")) + int(not row["neutral_oracle"].get("positive")) for row in rows), "unique_target_instance_count": len({row["target_instance_id"] for row in rows}), "fresh_reset_per_case": all(bool(row["fresh_reset"].get("fresh_target")) and bool(row["fresh_reset"].get("completed")) for row in rows), "get_post_counts": dict(Counter(row["method"] for row in rows)), "source_count": len({row["source_id"] for row in rows}), "implementation_count": len({(row["implementation"], row["variant"]) for row in rows}), "family_count": len({row["family"] for row in rows}), "trace_episode_count": len(episodes), "trace_accepted_episode_count": sum(int(episode["validation"].get("status") == "accepted_evaluation") for episode in episodes), "negative_probe_positive_count": sum(int(row["negative_oracle"].get("positive")) for row in rows), "collection_error_count": len(errors), "validation_failure_count": len(validation_failures)}
    checks = {"triplet_complete_per_case": len(rows) == 270 and not errors, "typed_positive_per_case": metrics["typed_positive_count"] >= 200, "typed_negative_oracle_per_case": metrics["typed_negative_oracle_count"] == len(rows) * 2, "negative_probe_strictly_negative": metrics["negative_probe_positive_count"] == 0, "fresh_target_per_case": metrics["unique_target_instance_count"] == len(rows), "get_post_covered": metrics["get_post_counts"] == {"GET": 135, "POST": 135}, "multi_source": metrics["source_count"] >= 5, "multi_family": metrics["family_count"] >= 8, "trace_episodes_accepted": metrics["trace_episode_count"] == metrics["trace_accepted_episode_count"] and not validation_failures, "no_raw_persistence": all(not row["raw_probe_stored"] and not row["raw_response_body_stored"] for row in rows)}
    trace = {"schema_version": "pg79-fresh-unified-triplet-trace-v1", "protocol_id": PROTOCOL_ID, "evaluation_only": True, "training_eligible": False, "steps": normalized_steps, "episodes": episodes, "episode_count": len(episodes), "accepted_episode_count": metrics["trace_accepted_episode_count"], "errors": errors, "validation_failures": validation_failures, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"protocol_id": PROTOCOL_ID, "schema_version": "pg79-fresh-unified-triplet-collector-report-v1", "status": "completed_evaluation", "source": {"collector": str(PG53_SCRIPT.relative_to(ROOT)), "independent_source_count": metrics["source_count"], "independent_implementation_count": metrics["implementation_count"], "family_count": metrics["family_count"], "model_retrained": False}, "scope": {"loopback_only": True, "external_network": False, "database_write": False, "script_execution": False, "state_mutation": False, "raw_probe_persistence": False, "raw_response_persistence": False}, "metrics": metrics, "hard_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "catalog_training_eligible": False, "status": "triplet_collection_only", "reason": "collection must be paired with PG-80 model replay and independent source audit before any promotion"}, "artifacts": {"catalog": str(CATALOG_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT))}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg79-fresh-unified-triplet-collector-protocol-v1", "source_contract": {"collector": str(PG53_SCRIPT.relative_to(ROOT)), "loopback_only": True, "fresh_target_per_case": True, "independent_source_count_min": 5, "independent_family_count_min": 8, "get_post_required": True}, "triplet_contract": {"neutral_projection": True, "negative_probe_projection": True, "positive_probe_projection": True, "neutral_oracle": True, "negative_oracle": True, "positive_oracle": True, "fresh_reset": True, "evidence_hash": True, "raw_persistence_forbidden": True}, "required_gates": {"triplet_complete_per_case": True, "typed_positive_per_case": True, "typed_negative_oracle_per_case": True, "negative_probe_strictly_negative": True, "fresh_target_per_case": True, "get_post_covered": True, "multi_source": True, "multi_family": True, "trace_episodes_accepted": True, "no_raw_persistence": True}, "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG80 frozen PG77 replay on PG79 unified triplets"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-79 Fresh unified triplet collector\n\n" + f"triplets={metrics['triplet_case_count']}；positive={metrics['typed_positive_count']}；typed negatives={metrics['typed_negative_oracle_count']}；negative probe positives={metrics['negative_probe_positive_count']}；sources={metrics['source_count']}；families={metrics['family_count']}；GET/POST={metrics['get_post_counts']}。\n\n硬门：`{report['hard_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["hard_gate"]["status"], "triplet_case_count": result["metrics"]["triplet_case_count"], "typed_negative_oracle_count": result["metrics"]["typed_negative_oracle_count"], "negative_probe_positive_count": result["metrics"]["negative_probe_positive_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
