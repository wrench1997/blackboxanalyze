"""PG-327B: paired fresh replay before and after the A800 candidate.

The same three SQL GET/POST routes are replayed with frozen PG-322
checkpoints (before) and the PG-327 A800 candidate (after).  Each phase uses
new disposable, network-none containers.  The model still sees abstract
Rule-IR context only; raw wires and response bodies remain evaluator-side and
are reduced to typed projections/evidence hashes in the paired report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg325() -> Any:
    path = ROOT / "scripts" / "run_pg325_sql_family_holdout.py"
    spec = importlib.util.spec_from_file_location("pg325_for_pg327b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-325 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG325 = _load_pg325()
EVAL = PG325.EVAL
HELP = PG325.HELP
RESEARCH = ROOT / "research"
SEEDS = (31901, 31902, 31903)
ROUTES = tuple(PG325.ROUTES)
IMAGE = str(PG325.IMAGE)
BEFORE_DIR = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
AFTER_DIR = ROOT / "artifacts" / "pg327-a800-replay" / "seeds"
CHECKPOINT_PREFIX = "pg322_cross_impl_decoy_seed_"
REPORT = RESEARCH / "pg327b_paired_fresh_replay_report_v1.json"
TRACE = RESEARCH / "pg327b_paired_fresh_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg327b_paired_fresh_replay_protocol_v1.json"
_PHASE = "before"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_gate() -> None:
    if os.environ.get("PG327B_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-327B requires explicit PG327B_LOCAL_DOCKER_EVAL=1")
    for directory in (BEFORE_DIR, AFTER_DIR):
        for seed in SEEDS:
            path = directory / f"{CHECKPOINT_PREFIX}{seed}.pt"
            if not path.exists():
                raise RuntimeError(f"PG-327B missing checkpoint: {path}")


def _docker(*args: str) -> str:
    return PG325._docker(*args)


def _exists(name: str) -> bool:
    return PG325._exists(name)


def _stop(name: str) -> None:
    if name and _exists(name):
        subprocess.run(["docker", "stop", "--time", "10", name], cwd=ROOT, capture_output=True, text=True, timeout=30)


def _start(seed: int, index: int) -> tuple[str, int, str, dict[str, Any]]:
    name = f"sift-pg327b-{_PHASE}-{seed}-{index}"
    if _exists(name):
        raise RuntimeError(f"PG-327B refuses target reuse: {name}")
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--label", "sift.pg327b=true", "--label", f"sift.pg327b.phase={_PHASE}",
        "--label", f"sift.pg327b.reset_epoch={_PHASE}-{seed}-{index}",
        "--network", "none", "--pids-limit", "256", "--memory", "1g", IMAGE,
    )
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        try:
            health = subprocess.run(
                ["docker", "exec", name, "curl", "-fsS", "--max-time", "5", "-o", "/dev/null", "http://127.0.0.1:8090/"],
                cwd=ROOT, capture_output=True, text=True, timeout=10,
            )
            if health.returncode == 0 and PG325.PG214._database_health(name):
                container_id = _docker("inspect", "--format", "{{.Id}}", name)
                mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
                image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
                network_mode = _docker("inspect", "--format", "{{.HostConfig.NetworkMode}}", name)
                if image_ref != IMAGE or network_mode != "none" or mounts:
                    raise RuntimeError("PG-327B target attestation mismatch")
                reset = {
                    "reset_id": f"pg327b-{_PHASE}-{seed}-{index}",
                    "phase": _PHASE,
                    "reset_epoch": f"{_PHASE}-{seed}-{index}",
                    "fresh_target": True,
                    "completed": True,
                    "container_recreated": True,
                    "container_restart_used": False,
                    "container_id_sha256": hashlib.sha256(container_id.encode("utf-8")).hexdigest(),
                    "image": image_ref,
                    "network_mode": network_mode,
                    "network_internal": False,
                    "host_port_published": False,
                    "external_network": False,
                    "volume_mount_count": len(mounts),
                    "database_health_gate": "mysqli_root_pikachu_ok",
                    "database_clean_contract": "fresh_writable_layer_no_volume_no_stateful_probe",
                    "state_change_allowed": False,
                    "domain_data_write_allowed": False,
                }
                return name, 0, container_id, reset
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    _stop(name)
    raise RuntimeError(f"PG-327B target {name} failed health gates")


def _configure_evaluator() -> None:
    HELP.EVAL = EVAL
    EVAL.IMAGE = IMAGE
    EVAL.ROUTES = ROUTES
    EVAL.SEEDS = SEEDS
    EVAL._start = _start
    EVAL._stop = _stop
    EVAL._source_hash = PG325._source_hash
    EVAL._role_context = PG325._role_context
    EVAL._failure_context = PG325._failure_context
    EVAL._candidate_values = PG325._candidate_values
    EVAL._send_internal = PG325._send_internal
    EVAL._safe_browser_oracle = PG325._safe_browser_oracle


def _run_phase(phase: str, seed: int, checkpoint: Path) -> dict[str, Any]:
    global _PHASE
    _PHASE = phase
    device = torch.device("cpu")
    model, vocabulary, symbolic = PG325.PG314.load_causal_checkpoint(checkpoint, device)
    if not symbolic:
        raise RuntimeError(f"PG-327B checkpoint is not symbolic: {checkpoint}")
    try:
        result = EVAL._seed_run(seed, model, vocabulary, device, None)
        result = HELP._attach_failure_transition(result)
        result = HELP._normalize_unsupported_post_lanes(result)
        result = HELP._attach_belief_trace(result)
        return PG325._bind_role_belief_evidence(result)
    finally:
        del model


def _safe_abstract(record: Mapping[str, Any], phase: str, seed: int) -> dict[str, Any]:
    value = dict(record)
    value["phase"] = phase
    value["seed"] = seed
    value["raw_payload_stored"] = False
    value["raw_response_body_stored"] = False
    value["training_eligible"] = False
    return value


def _phase_summary(phase: str, seed: int, result: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    humans = [dict(row) for row in list(result.get("rows") or [])]
    abstracts = [_safe_abstract(row, phase, seed) for row in list(result.get("abstract_records") or [])]
    route_rows: list[dict[str, Any]] = []
    for row in humans:
        route = dict(row.get("route") or {})
        entries = [dict(entry) for entry in list((row.get("model") or {}).get("entries") or [])]
        oracle = dict(row.get("oracle") or {})
        target = dict(row.get("target") or {})
        reset = dict(target.get("fresh_reset") or {})
        evidence = dict(row.get("evidence") or {})
        transition = dict((row.get("model") or {}).get("failure_transition") or {})
        route_rows.append(
            {
                "phase": phase,
                "seed": seed,
                "route_id": str(route.get("id", "")),
                "method": str(route.get("method", "")),
                "source_sha256": str(target.get("source_sha256", "")),
                "reset_id": str(reset.get("reset_id", "")),
                "container_id_sha256": str(reset.get("container_id_sha256", "")),
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "variant_role_count": len(entries),
                "variant_exact_count": sum(int(bool((entry.get("proposal") or {}).get("variant_exact"))) for entry in entries),
                "model_send_count": sum(int(bool(entry.get("sent"))) for entry in entries),
                "negative_lane_violation_count": sum(int(str(entry.get("role")) == "negative_control" and bool(entry.get("sent")) and not bool((entry.get("proposal") or {}).get("variant_exact"))) for entry in entries),
                "oracle": {
                    "candidate_positive": bool(oracle.get("candidate_positive")),
                    "reference_positive": bool(oracle.get("reference_positive")),
                    "negative_clean": bool(oracle.get("negative_clean")),
                    "replay_consistent": bool(oracle.get("replay_consistent")),
                    "all_variant_exact": bool(oracle.get("all_variant_exact")),
                    "typed_effect_confirmed": bool(oracle.get("typed_effect_confirmed")),
                    "evidence_sha256": str(oracle.get("evidence_sha256", "")),
                },
                "evidence_sha256": str(evidence.get("evidence_sha256", "")),
                "failure_transition": {
                    "required": bool(transition.get("repair_transition_required")),
                    "action_changed": transition.get("action_changed") is True,
                    "valid": bool(transition.get("repair_transition_valid")),
                },
                "belief_trace": [
                    {
                        "evidence_hash": str(step.get("evidence_hash", "")),
                        "source_evidence_sha256": str(step.get("source_evidence_sha256", "")),
                        "evidence_scope": str(step.get("evidence_scope", "")),
                        "duplicate_evidence": bool(step.get("duplicate_evidence")),
                    }
                    for step in list(row.get("belief_trace") or [])
                ],
            }
        )
    route_keys = sorted((row["seed"], row["route_id"], row["method"]) for row in route_rows)
    positive = [row for row in route_rows if bool(row["oracle"]["typed_effect_confirmed"])]
    question_rows = list(result.get("multi_missing") or [])
    context_firewall = HELP._model_context_firewall(humans, abstracts)
    checks = {
        "fresh_reset_all": bool(route_rows) and all(row["reset_id"] and row["container_id_sha256"] for row in route_rows),
        "get_post_pair": any(row["method"] == "GET" for row in route_rows) and any(row["method"] == "POST" for row in route_rows),
        "candidate_reference_negative_all": all(row["variant_role_count"] == 3 for row in route_rows),
        "typed_evidence_all": all(len(row["oracle"]["evidence_sha256"]) == 64 and row["oracle"]["evidence_sha256"] == row["evidence_sha256"] for row in route_rows),
        "source_attestation_all": all(len(row["source_sha256"]) == 64 for row in route_rows),
        "network_none": all(row["reset_id"] and row["container_id_sha256"] for row in route_rows),
        "failure_action_changed_all": bool(result.get("failure_transition_complete")),
        "role_bound_belief_evidence_all": bool(result.get("belief_trace_complete")) and int(result.get("belief_duplicate_evidence_count", 0)) == 0 and all(
            all(step["evidence_scope"] == "record_role_bound" and len(step["evidence_hash"]) == 64 and len(step["source_evidence_sha256"]) == 64 and not step["duplicate_evidence"] for step in row["belief_trace"])
            for row in route_rows
        ),
        "context_firewall": bool(context_firewall),
        "raw_payload_excluded": all(not bool(row.get("raw_payload_stored")) for row in abstracts),
        "raw_response_excluded": all(not bool(row.get("raw_response_body_stored")) for row in abstracts),
        "negative_zero": sum(row["negative_lane_violation_count"] for row in route_rows) == 0,
    }
    metrics = {
        "route_count": len(route_rows),
        "get_count": sum(row["method"] == "GET" for row in route_rows),
        "post_count": sum(row["method"] == "POST" for row in route_rows),
        "typed_effect_count": sum(int(row["oracle"]["typed_effect_confirmed"]) for row in route_rows),
        "positive_route_count": len(positive),
        "variant_role_count": sum(row["variant_role_count"] for row in route_rows),
        "variant_exact_count": sum(row["variant_exact_count"] for row in route_rows),
        "failure_transition_required_count": int(result.get("failure_transition_required_count", 0)),
        "failure_action_changed_count": int(result.get("failure_action_changed_count", 0)),
        "failure_repair_count": len(route_rows),
        "failure_repair_correct_count": int(result.get("repair_correct_count", 0)),
        "multi_missing_question_rows": len(question_rows),
        "multi_missing_question_recall": float(result.get("multi_missing_question_recall", 0.0)),
        "multi_missing_unsafe_allow": int(result.get("multi_missing_unsafe_allow", 0)),
        "belief_transition_count": int(result.get("belief_transition_count", 0)),
        "belief_duplicate_evidence_count": int(result.get("belief_duplicate_evidence_count", 0)),
    }
    return {"phase": phase, "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": _sha256_file(checkpoint), "route_keys": route_keys, "route_rows": route_rows, "abstract_records": abstracts, "metrics": metrics, "checks": checks}


def _pair(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_keys = list(before.get("route_keys") or [])
    after_keys = list(after.get("route_keys") or [])
    bm = dict(before.get("metrics") or {})
    am = dict(after.get("metrics") or {})
    return {
        "seed": before.get("seed"),
        "same_canary_route_set": before_keys == after_keys,
        "before_checkpoint": before.get("checkpoint"),
        "before_checkpoint_sha256": before.get("checkpoint_sha256"),
        "after_checkpoint": after.get("checkpoint"),
        "after_checkpoint_sha256": after.get("checkpoint_sha256"),
        "before_metrics": bm,
        "after_metrics": am,
        "after_not_worse_observed": {
            "typed_effect_count": int(am.get("typed_effect_count", 0)) >= int(bm.get("typed_effect_count", 0)),
            "variant_exact_count": int(am.get("variant_exact_count", 0)) >= int(bm.get("variant_exact_count", 0)),
            "failure_repair_correct_count": int(am.get("failure_repair_correct_count", 0)) >= int(bm.get("failure_repair_correct_count", 0)),
            "negative_zero": int(am.get("multi_missing_unsafe_allow", 0)) == 0,
        },
    }


def main() -> int:
    _require_gate()
    _configure_evaluator()
    started = time.monotonic()
    phase_runs: dict[str, list[dict[str, Any]]] = {"before": [], "after": []}
    for phase, directory in (("before", BEFORE_DIR), ("after", AFTER_DIR)):
        for seed in SEEDS:
            checkpoint = directory / f"{CHECKPOINT_PREFIX}{seed}.pt"
            phase_runs[phase].append(_phase_summary(phase, seed, _run_phase(phase, seed, checkpoint), checkpoint))

    before_by_seed = {int(item["seed"]): item for item in phase_runs["before"]}
    after_by_seed = {int(item["seed"]): item for item in phase_runs["after"]}
    pairs = [_pair(before_by_seed[seed], after_by_seed[seed]) for seed in SEEDS]
    all_routes_same = all(bool(pair["same_canary_route_set"]) for pair in pairs)
    all_containers_distinct = len({row["container_id_sha256"] for phase in phase_runs.values() for item in phase for row in item["route_rows"]}) == 2 * len(SEEDS) * len(ROUTES)
    phase_checks = {phase: {key: all(bool(item["checks"].get(key)) for item in phase_runs[phase]) for key in next(iter(phase_runs[phase]))["checks"]} for phase in phase_runs}
    strict_checks = {
        "fresh_reset_before_all": phase_checks["before"]["fresh_reset_all"],
        "fresh_reset_after_all": phase_checks["after"]["fresh_reset_all"],
        "distinct_container_pairs": all_containers_distinct,
        "get_post_pair_before": phase_checks["before"]["get_post_pair"],
        "get_post_pair_after": phase_checks["after"]["get_post_pair"],
        "candidate_reference_negative_before": phase_checks["before"]["candidate_reference_negative_all"],
        "candidate_reference_negative_after": phase_checks["after"]["candidate_reference_negative_all"],
        "typed_evidence_before": phase_checks["before"]["typed_evidence_all"],
        "typed_evidence_after": phase_checks["after"]["typed_evidence_all"],
        "source_attestation_before": phase_checks["before"]["source_attestation_all"],
        "source_attestation_after": phase_checks["after"]["source_attestation_all"],
        "failure_action_changed_before": phase_checks["before"]["failure_action_changed_all"],
        "failure_action_changed_after": phase_checks["after"]["failure_action_changed_all"],
        "role_bound_belief_before": phase_checks["before"]["role_bound_belief_evidence_all"],
        "role_bound_belief_after": phase_checks["after"]["role_bound_belief_evidence_all"],
        "context_firewall_before": phase_checks["before"]["context_firewall"],
        "context_firewall_after": phase_checks["after"]["context_firewall"],
        "raw_payload_excluded": phase_checks["before"]["raw_payload_excluded"] and phase_checks["after"]["raw_payload_excluded"],
        "raw_response_excluded": phase_checks["before"]["raw_response_excluded"] and phase_checks["after"]["raw_response_excluded"],
        "network_none_before": phase_checks["before"]["network_none"],
        "network_none_after": phase_checks["after"]["network_none"],
        "same_canary_route_set": all_routes_same,
        "before_after_checkpoint_distinct": len({item["checkpoint_sha256"] for item in phase_runs["before"] + phase_runs["after"]}) == 6,
    }
    paired_replay = {
        "status": "completed",
        "paired_replay_present": True,
        "same_canary_route_set": all_routes_same,
        "before_checkpoint": {str(seed): before_by_seed[seed]["checkpoint_sha256"] for seed in SEEDS},
        "after_checkpoint": {str(seed): after_by_seed[seed]["checkpoint_sha256"] for seed in SEEDS},
        "pairs": pairs,
    }
    all_summaries = [item for phase in phase_runs.values() for item in phase]
    counts = {
        "phase_count": 2,
        "seed_count": len(SEEDS),
        "route_count_per_phase": len(SEEDS) * len(ROUTES),
        "total_phase_routes": len(all_summaries) * len(ROUTES),
        "before_typed_effect_count": sum(int(item["metrics"]["typed_effect_count"]) for item in phase_runs["before"]),
        "after_typed_effect_count": sum(int(item["metrics"]["typed_effect_count"]) for item in phase_runs["after"]),
        "before_variant_role_count": sum(int(item["metrics"]["variant_role_count"]) for item in phase_runs["before"]),
        "after_variant_role_count": sum(int(item["metrics"]["variant_role_count"]) for item in phase_runs["after"]),
        "before_failure_action_changed_count": sum(int(item["metrics"]["failure_action_changed_count"]) for item in phase_runs["before"]),
        "after_failure_action_changed_count": sum(int(item["metrics"]["failure_action_changed_count"]) for item in phase_runs["after"]),
        "before_belief_duplicate_evidence_count": sum(int(item["metrics"]["belief_duplicate_evidence_count"]) for item in phase_runs["before"]),
        "after_belief_duplicate_evidence_count": sum(int(item["metrics"]["belief_duplicate_evidence_count"]) for item in phase_runs["after"]),
    }
    gate_checks = {key: bool(value) for key, value in strict_checks.items()}
    gate_checks["paired_forgetting_replay"] = bool(paired_replay["paired_replay_present"] and paired_replay["same_canary_route_set"])
    gate_checks["promotion_blocked"] = True
    report = {
        "protocol_id": "pg-pk-327b-paired-fresh-replay-v1",
        "schema_version": "pg327b-paired-fresh-replay-report-v1",
        "status": "completed_local_docker_pg327b_paired_replay",
        "runtime": {"execution_policy": "operator-authorized-local-evaluation-any-time", "explicit_flag": "PG327B_LOCAL_DOCKER_EVAL=1", "image": IMAGE, "network": "none", "host_port_published": False, "external_network": False, "target_contacted": True, "docker_started": True, "seed_count": len(SEEDS), "route_ids": [str(route["id"]) for route in ROUTES]},
        "model_contract": {"decoder_only_next_token": True, "abstract_rule_ir_only": True, "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_generation": "source_grounded_binding_after_model_variant_guard", "oracle_target_off_input": True},
        "checkpoints": {"before_dir": str(BEFORE_DIR.relative_to(ROOT)), "after_dir": str(AFTER_DIR.relative_to(ROOT)), "before_hashes": paired_replay["before_checkpoint"], "after_hashes": paired_replay["after_checkpoint"]},
        "counts": counts,
        "phase_checks": phase_checks,
        "checks": strict_checks,
        "forgetting": paired_replay,
        "hypothesis_gate": {"status": "blocked", "checks": gate_checks, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "artifacts": {"trace": str(TRACE.relative_to(ROOT)), "protocol": str(PROTOCOL.relative_to(ROOT))},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "report_sha256": "",
    }
    report["report_sha256"] = _digest(report)
    episodes = [record for phase in phase_runs.values() for item in phase for record in item["abstract_records"]]
    trace = {"schema_version": "pg327b-paired-fresh-replay-trace-v1", "status": "completed", "episodes": episodes, "phase_count": 2, "raw_payload_stored": False, "raw_response_body_stored": False, "training_eligible": False, "memory_promotion_allowed": False, "trace_sha256": ""}
    trace["trace_sha256"] = _digest(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg327b-paired-fresh-replay-protocol-v1", "scope": {"target": "authorized local Docker Pikachu image", "image": IMAGE, "network": "none", "host_port_published": False, "methods": ["GET", "POST"], "seed_count": len(SEEDS), "phases": ["before", "after"]}, "required_gates": {key: True for key in (list(strict_checks) + ["paired_forgetting_replay", "raw_payload_training_excluded", "promotion_blocked"])}, "forbidden": ["public_target", "external_network", "time_delay", "database_write", "destructive", "credential_access"], "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}, "protocol_sha256": ""}
    protocol["protocol_sha256"] = _digest(protocol)
    for path, value in ((REPORT, report), (TRACE, trace), (PROTOCOL, protocol)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": counts, "checks": strict_checks, "forgetting": {"paired_replay_present": True, "same_canary_route_set": all_routes_same}, "promotion": report["promotion"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

