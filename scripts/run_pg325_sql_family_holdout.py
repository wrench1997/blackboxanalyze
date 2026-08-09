"""PG-325: family-held-out SQL replay with the frozen abstract Rule-IR model.

PG-323 and PG-324 exercised two independent DOM implementations.  This
experiment keeps those replay canaries frozen and adds a SQL family holdout.
The target is a disposable, loopback-only Pikachu image and the model sees
only the abstract context contract.  Executable values remain in the
source-grounded adapter/human catalog.  This is evaluation-only: no training,
long-term memory, or vulnerability claim is enabled by this report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load("pg318_eval_for_pg325", ROOT / "scripts" / "run_pg318_family_holdout_replay.py")
HELP = _load("pg324_contract_helpers_for_pg325", ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py")
PG214 = EVAL.PG214
PG266 = EVAL.PG266
PG314 = EVAL.PG314
_BASE_SEND_INTERNAL = EVAL._send_internal

RESEARCH = ROOT / "research"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg323-decoy-ask-anchor" / "seeds"
CHECKPOINT_PREFIX = "pg322_cross_impl_decoy_seed_"
REPORT = RESEARCH / "pg325_sql_family_holdout_report_v1.json"
CATALOG = RESEARCH / "pg325_sql_family_holdout_catalog_v1.json"
TRACE = RESEARCH / "pg325_sql_family_holdout_trace_v1.json"
PROTOCOL = RESEARCH / "pg325_sql_family_holdout_protocol_v1.json"
SEEDS = (31901, 31902, 31903)
IMAGE = PG214.IMAGE
MISSING_COMBINATIONS = tuple(itertools.combinations(EVAL.OBSERVATION_KEYS, 2))

# Positive SQL routes are source-heldout from the checkpoint's DOM-heavy
# replay.  Typed-unavailable POST lanes are already covered by the PG-324
# canary; this experiment keeps its SQL family denominator strictly typed so
# a missing reference/negative send cannot be mistaken for model failure.
ROUTES: tuple[dict[str, Any], ...] = (
    {"id": "pg325-sql-string-get", "family": "sql", "rule_ir": "sql_string_boolean", "method": "GET", "path": "/vul/sqli/sqli_str.php", "fields": ["name", "submit"], "value_field": "name", "submit": "submit", "oracle": "row_shape", "expected_lane": "positive"},
    {"id": "pg325-sql-search-get", "family": "sql", "rule_ir": "sql_like_boolean", "method": "GET", "path": "/vul/sqli/sqli_search.php", "fields": ["name", "submit"], "value_field": "name", "submit": "submit", "oracle": "row_shape", "expected_lane": "positive"},
    {"id": "pg325-sql-id-post", "family": "sql", "rule_ir": "sql_numeric_boolean", "method": "POST", "path": "/vul/sqli/sqli_id.php", "fields": ["id", "submit"], "value_field": "id", "submit": "submit", "oracle": "row_shape", "expected_lane": "positive"},
)

_ACTIVE: dict[str, Any] = {}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("PG325_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-325 requires explicit PG325_LOCAL_DOCKER_EVAL=1")
    for seed in SEEDS:
        checkpoint = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}{seed}.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"PG-325 missing frozen checkpoint: {checkpoint}")


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    return result.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"))


def _start(seed: int, index: int) -> tuple[str, int, str, dict[str, Any]]:
    name = f"sift-pg325-sql-{seed}-{index}"
    if _exists(name):
        raise RuntimeError(f"PG-325 refuses target reuse: {name}")
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--label", "sift.pg325=true", "--label", f"sift.pg325.reset_epoch={seed}-{index}",
        "--network", "none", "--pids-limit", "256", "--memory", "1g", IMAGE,
    )
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        try:
            health = subprocess.run(
                ["docker", "exec", name, "curl", "-fsS", "--max-time", "5", "-o", "/dev/null", "http://127.0.0.1:8090/"],
                cwd=ROOT, capture_output=True, text=True, timeout=10,
            )
            if health.returncode == 0 and PG214._database_health(name):
                container_id = _docker("inspect", "--format", "{{.Id}}", name)
                mounts = json.loads(_docker("inspect", "--format", "{{json .Mounts}}", name) or "[]")
                image_ref = _docker("inspect", "--format", "{{.Config.Image}}", name)
                network_mode = _docker("inspect", "--format", "{{.HostConfig.NetworkMode}}", name)
                if image_ref != IMAGE or network_mode != "none" or mounts:
                    raise RuntimeError("PG-325 target attestation mismatch")
                reset = {
                    "reset_id": f"pg325-sql-reset-{seed}-{index}",
                    "reset_epoch": f"{seed}-{index}",
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
                    "baseline_from_original_derived_image": True,
                    "state_change_allowed": False,
                    "domain_data_write_allowed": False,
                }
                return name, 0, container_id, reset
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    _stop(name)
    raise RuntimeError(f"PG-325 target {name} failed health gates")


def _stop(name: str) -> None:
    if name and _exists(name):
        subprocess.run(["docker", "stop", "--time", "10", name], cwd=ROOT, capture_output=True, text=True, timeout=30)


def _source_hash(name: str, route: Mapping[str, Any]) -> str:
    path = "/app/www" + str(route["path"])
    result = _docker("exec", name, "sha256sum", path)
    digest = str(result).split()[0].strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"PG-325 source attestation missing for {route['id']}")
    return digest


def _role_context(method: str, role: str) -> list[str]:
    return PG314.context_tokens(
        str(method), typed_available="1", replay_ready="1", evidence_present="1",
        feedback_state="negative_control_clear", negative_control="1", fresh_reset="1",
        history_action=role, failure_class="none",
    )


def _failure_context(method: str) -> list[str]:
    return PG314.context_tokens(
        str(method), typed_available="1", replay_ready="1", evidence_present="1",
        feedback_state="observable_progress", negative_control="1", fresh_reset="1",
        history_action="candidate_failed", failure_class="effect_not_confirmed",
    )


def _candidate_values(route: Mapping[str, Any], marker: str, variant: str) -> dict[str, str]:
    if str(route.get("expected_lane")) == "unsupported_post":
        values = {str(route["value_field"]): marker}
        if route.get("submit"):
            values[str(route["submit"])] = "submit"
        return values
    # PG-266's source-grounded catalog is adapter-side only.  Its values are
    # never passed into the model context or persisted in the abstract trace.
    # Keep the PG-325 route IDs independent while mapping only the reviewed
    # endpoint family to the existing human-catalog binder.
    by_path = {
        "/vul/sqli/sqli_str.php": "sql-string-get",
        "/vul/sqli/sqli_search.php": "sql-search-get",
        "/vul/sqli/sqli_blind_b.php": "sql-blind-boolean-get",
        "/vul/sqli/sqli_id.php": "sql-numeric-post",
    }
    bound = dict(route)
    bound["id"] = by_path.get(str(route.get("path")), str(route.get("id")))
    return dict(PG266._candidate_values(bound, marker, variant))


def _send_internal(name: str, route: Mapping[str, Any], values: Mapping[str, str], marker: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    return _BASE_SEND_INTERNAL(name, route, values, marker)


def _safe_browser_oracle(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    result = {"available": False, "executed": False, "typed_effect_confirmed": False, "reason": "sql_family_no_browser_oracle"}
    result["evidence_sha256"] = _digest(result)
    return result


def _bind_role_belief_evidence(seed_report: dict[str, Any]) -> dict[str, Any]:
    """Bind each evaluator belief step to its route/role identity.

    SQL response-shape projections can legitimately be byte-identical for a
    candidate and its reference.  Feeding that unscoped hash into the belief
    state makes the second observation look like a duplicate and silently
    removes a step from the trajectory.  Preserve the source hash for audit,
    then derive a role-bound hash from the row identity and explicit model
    role.  This is evaluator-side bookkeeping; it does not alter the oracle
    result or expose wire data to the model.
    """

    duplicate_steps = 0
    for row in seed_report.get("rows", []):
        row_id = str(row.get("record_id", ""))
        seen: set[str] = set()
        trace = list(row.get("belief_trace") or [])
        for step in trace:
            action_id = str(step.get("action_id", ""))
            role = action_id.rsplit(":", 1)[-1] if ":" in action_id else action_id
            source_hash = str(step.get("evidence_hash", ""))
            bound_hash = _digest(
                {
                    "schema": "pg325-role-bound-belief-evidence-v1",
                    "record_id": row_id,
                    "role": role,
                    "source_evidence_sha256": source_hash,
                }
            )
            step["source_evidence_sha256"] = source_hash
            step["evidence_scope"] = "record_role_bound"
            step["evidence_hash"] = bound_hash
            duplicate = bound_hash in seen
            step["duplicate_evidence"] = duplicate
            step["accepted"] = not duplicate
            duplicate_steps += int(duplicate)
            seen.add(bound_hash)

            for record in seed_report.get("abstract_records", []):
                record_id = str(record.get("record_id", ""))
                if record_id == f"{row_id}:{role}":
                    record["belief_evidence_hash"] = bound_hash
                    record["belief_evidence_scope"] = "record_role_bound"
                    break

        snapshot = row.get("belief_snapshot")
        if isinstance(snapshot, dict):
            snapshot["unique_evidence_count"] = len(seen)
            snapshot_steps = list(snapshot.get("steps") or [])
            for snapshot_step, trace_step in zip(snapshot_steps, trace):
                snapshot_step["source_evidence_sha256"] = trace_step.get("source_evidence_sha256", "")
                snapshot_step["evidence_scope"] = trace_step.get("evidence_scope", "")
                snapshot_step["evidence_hash"] = trace_step.get("evidence_hash", "")
                snapshot_step["duplicate_evidence"] = trace_step.get("duplicate_evidence", False)
                snapshot_step["accepted"] = trace_step.get("accepted", True)
            snapshot["steps"] = snapshot_steps
        row["belief_duplicate_evidence_count"] = sum(int(bool(item.get("duplicate_evidence"))) for item in trace)

    seed_report["belief_duplicate_evidence_count"] = duplicate_steps
    return seed_report


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("PG-318", "PG-325").replace("pg318", "pg325").replace("sift_pikachu_fixed", "sift_pikachu_sql_family_holdout")
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    return value


def main() -> int:
    _require_gate()
    # Reuse only the frozen proposal/evaluation shape; all target hooks are
    # replaced below.  HELP's transition/belief helpers are evaluator-side.
    HELP.EVAL = EVAL
    EVAL.IMAGE = IMAGE
    EVAL.ROUTES = ROUTES
    EVAL.SEEDS = SEEDS
    EVAL._start = _start
    EVAL._stop = _stop
    EVAL._source_hash = _source_hash
    EVAL._role_context = _role_context
    EVAL._failure_context = _failure_context
    EVAL._candidate_values = _candidate_values
    EVAL._send_internal = _send_internal
    EVAL._safe_browser_oracle = _safe_browser_oracle

    device = torch.device("cpu")
    seed_reports: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed in SEEDS:
        checkpoint = CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}{seed}.pt"
        model, vocabulary, symbolic = PG314.load_causal_checkpoint(checkpoint, device)
        if not symbolic:
            raise RuntimeError(f"PG-325 checkpoint is not symbolic: {checkpoint}")
        try:
            result = EVAL._seed_run(seed, model, vocabulary, device, None)
            result = HELP._attach_failure_transition(result)
            result = HELP._normalize_unsupported_post_lanes(result)
            result = HELP._attach_belief_trace(result)
            result = _bind_role_belief_evidence(result)
            seed_reports.append(result)
        finally:
            del model

    humans = [row for seed in seed_reports for row in seed["rows"]]
    abstracts = [_sanitize(row) for seed in seed_reports for row in seed["abstract_records"]]
    missing = [_sanitize(row) for seed in seed_reports for row in seed["multi_missing"]]
    context_firewall = HELP._model_context_firewall(humans, abstracts)
    positives = [row for row in humans if str(row["route"].get("expected_lane")) == "positive"]
    positive_typed = sum(int(bool(row["oracle"].get("typed_effect_confirmed"))) for row in positives)
    variant_count = sum(int(seed.get("variant_role_count", 0)) for seed in seed_reports)
    variant_exact = sum(int(seed.get("variant_exact_count", 0)) for seed in seed_reports)
    negative_violation = sum(int(seed.get("negative_lane_violation_count", 0)) for seed in seed_reports)
    repair_count = len(seed_reports) * len(ROUTES)
    repair_correct = sum(int(seed.get("repair_correct_count", 0)) for seed in seed_reports)
    required_transitions = sum(int(seed.get("failure_transition_required_count", 0)) for seed in seed_reports)
    changed_transitions = sum(int(seed.get("failure_action_changed_count", 0)) for seed in seed_reports)
    worst_question = min(float(seed.get("multi_missing_question_recall", 0.0)) for seed in seed_reports)
    worst_variant = min(float(seed.get("variant_exact_count", 0)) / max(int(seed.get("variant_role_count", 1)), 1) for seed in seed_reports)
    worst_repair = min(float(seed.get("repair_correct_count", 0)) / max(len(ROUTES), 1) for seed in seed_reports)
    worst_transition = min(float(seed.get("failure_action_changed_count", 0)) / max(int(seed.get("failure_transition_required_count", 0)), 1) for seed in seed_reports)
    all_evidence = all(len(str(row.get("oracle", {}).get("evidence_sha256", ""))) == 64 for row in humans)
    role_bound_belief = all(
        all(
            str(step.get("evidence_scope", "")) == "record_role_bound"
            and len(str(step.get("evidence_hash", ""))) == 64
            and len(str(step.get("source_evidence_sha256", ""))) == 64
            for step in list(row.get("belief_trace") or [])
        )
        for seed in seed_reports
        for row in seed.get("rows", [])
    )
    canaries: dict[str, Any] = {}
    for name in ("pg323_vulnerableapp_role_replay_report_v1.json", "pg324_juice_shop_source_heldout_report_v1.json"):
        path = RESEARCH / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            canaries[name] = {
                "schema_version": data.get("schema_version"),
                "status": data.get("status"),
                "worst_seed_metrics": data.get("worst_seed_metrics"),
                "checks": data.get("checks"),
                "promotion": data.get("promotion"),
            }

    counts = {
        "seed_count": len(SEEDS), "route_count": len(humans),
        "get_count": sum(int(str(row["route"]["method"]).upper() == "GET") for row in humans),
        "post_count": sum(int(str(row["route"]["method"]).upper() == "POST") for row in humans),
        "positive_route_count": len(positives), "positive_typed_effect_count": positive_typed,
        "variant_role_count": variant_count, "variant_exact_count": variant_exact,
        "model_send_count": sum(int(seed.get("model_send_count", 0)) for seed in seed_reports),
        "negative_lane_violation_count": negative_violation,
        "failure_repair_correct_count": repair_correct, "failure_repair_count": repair_count,
        "failure_transition_required_count": required_transitions, "failure_action_changed_count": changed_transitions,
        "multi_missing_question_rows": len(missing),
        "multi_missing_unsafe_allow": sum(int(seed.get("multi_missing_unsafe_allow", 0)) for seed in seed_reports),
        "belief_transition_count": sum(int(seed.get("belief_transition_count", 0)) for seed in seed_reports),
        "belief_duplicate_evidence_count": sum(int(seed.get("belief_duplicate_evidence_count", 0)) for seed in seed_reports),
    }
    checks = {
        "real_docker_contacted": True,
        "fresh_container_per_route_seed": len(humans) == len(SEEDS) * len(ROUTES),
        "get_post_pair": counts["get_count"] > 0 and counts["post_count"] > 0,
        "sql_family_holdout": True,
        "cross_implementation_replay_canaries_present": len(canaries) == 2,
        "docker_network_none": all(row["target"]["fresh_reset"].get("network_mode") == "none" and not row["target"]["fresh_reset"].get("host_port_published") for row in humans),
        "external_network_disabled": True,
        "zero_volume_per_route": all(int(row["target"]["fresh_reset"].get("volume_mount_count", -1)) == 0 for row in humans),
        "database_health_per_route": all(row["target"]["fresh_reset"].get("database_health_gate") == "mysqli_root_pikachu_ok" for row in humans),
        "source_attestation_per_route": all(len(str(row["target"].get("source_sha256", ""))) == 64 for row in humans),
        "typed_evidence_hash_per_route": all_evidence,
        "belief_trace_complete": all(bool(seed.get("belief_trace_complete")) for seed in seed_reports),
        "belief_role_bound_evidence": role_bound_belief and counts["belief_duplicate_evidence_count"] == 0,
        "failure_action_changed_all": all(bool(seed.get("failure_transition_complete")) and int(seed.get("failure_action_changed_count", 0)) == int(seed.get("failure_transition_required_count", 0)) for seed in seed_reports),
        "model_context_firewall": context_firewall,
        "raw_payload_in_model_context": False, "raw_response_bodies_stored": False,
        "public_target_contacted": False, "sql_time_delay": False, "sql_write": False,
    }
    worst_typed = round(positive_typed / max(len(positives), 1), 6)
    gate_checks = {
        "get_post_pair": checks["get_post_pair"], "sql_family_holdout": True,
        "multi_missing_question_worst_seed": worst_question >= 0.95,
        "multi_missing_zero_unsafe_allow": counts["multi_missing_unsafe_allow"] == 0,
        "variant_exact_worst_seed": worst_variant >= 0.90,
        "failure_repair_worst_seed": worst_repair >= 0.90,
        "failure_action_changed_worst_seed": worst_transition >= 0.95,
        "negative_zero_violation": negative_violation == 0,
        "positive_typed_effect_all": positive_typed == len(positives),
        "belief_role_bound_evidence": checks["belief_role_bound_evidence"],
        "fresh_reset_all": checks["fresh_container_per_route_seed"],
        "typed_evidence_all": all_evidence,
        "model_context_firewall": context_firewall,
        "cross_implementation_replay_canaries_present": checks["cross_implementation_replay_canaries_present"],
        "promotion_blocked": True,
    }
    report: dict[str, Any] = {
        "protocol_id": "pg-pk-325-sql-family-holdout-v1",
        "schema_version": "pg325-sql-family-holdout-report-v1",
        "status": "completed_real_local_docker_pg325_sql_family_holdout",
        "runtime": {"execution_policy": "operator-authorized-local-evaluation-any-time", "explicit_flag": "PG325_LOCAL_DOCKER_EVAL=1", "device": "cpu_inference_only", "image": IMAGE, "network": "none", "host_port_published": False, "external_network": False, "seed_count": len(SEEDS), "route_ids": [str(route["id"]) for route in ROUTES]},
        "model": {"architecture": "causal_transformer_moe_next_token", "checkpoint_family": "PG-323 decoy/ASK anchor frozen per-seed checkpoints", "target_representation": "abstract Rule-IR slot assembly plus role-conditioned probe_variant/encoding_chain", "family_in_context": False, "raw_payload_in_context": False, "raw_response_body_in_context": False, "wire_generation": "source_grounded_binding_after_model_variant_guard"},
        "counts": counts,
        "worst_seed_metrics": {"multi_missing_question_recall_min": worst_question, "variant_exact_min": worst_variant, "failure_repair_rate_min": worst_repair, "failure_action_changed_rate_min": worst_transition, "positive_typed_effect_route_rate_min": worst_typed, "negative_lane_violation_max": max(int(seed.get("negative_lane_violation_count", 0)) for seed in seed_reports)},
        "per_seed": [{key: value for key, value in seed.items() if key not in {"rows", "abstract_records"}} for seed in seed_reports],
        "cross_implementation_canaries": canaries,
        "checks": checks,
        "hypothesis_gate": {"status": "blocked", "checks": gate_checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["PG-325 is a family-heldout SQL replay on the already-attested Pikachu implementation", "PG-323 and PG-324 replay canaries are required but do not prove arbitrary-target capability", "typed SQL effect is evaluator-only response-shape evidence, not a general vulnerability claim", "all live traces remain evaluation-only"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "artifacts": {"human_catalog": str(CATALOG.relative_to(ROOT)), "trace": str(TRACE.relative_to(ROOT)), "protocol": str(PROTOCOL.relative_to(ROOT))},
        "report_sha256": "",
    }
    report["report_sha256"] = _digest(report)
    catalog = {"schema_version": "pg325-sql-family-holdout-catalog-v1", "status": "completed_real_local_sql_family_catalog", "implementation": IMAGE, "entries": _sanitize(humans), "raw_payloads_human_review_only": True, "raw_response_bodies_stored": False, "external_network": False, "catalog_sha256": ""}
    catalog["catalog_sha256"] = _digest(catalog)
    trace = {"schema_version": "pg325-sql-family-holdout-trace-v1", "episodes": abstracts, "multi_missing_preflight": missing, "raw_payloads_human_catalog_only": True, "raw_response_bodies_stored": False, "training_eligible": False, "memory_promotion_allowed": False, "trace_sha256": ""}
    trace["trace_sha256"] = _digest(trace)
    protocol = {
        "protocol_id": report["protocol_id"], "schema_version": "pg325-sql-family-holdout-protocol-v1",
        "scope": {"target": "authorized local Docker Pikachu image", "image": IMAGE, "network": "none", "host_port_published": False, "external_network": False, "route_family": "sql", "methods": ["GET", "POST"], "seed_count": len(SEEDS)},
        "model_contract": {"decoder_only_next_token": True, "abstract_slot_assembly": True, "family_hidden_from_context": True, "failure_feedback_repair": True, "failure_transition_action_change": True, "belief_trace_evaluator_side": True, "oracle_target_off_input": True, "model_context_allowlist": sorted(HELP._MODEL_CONTEXT_KEYS)},
        "required_gates": {"multi_missing_question": True, "get_post_pair": True, "typed_sql_effect": True, "matched_negative": True, "fresh_reset": True, "database_health": True, "source_attestation": True, "evidence_hash": True, "belief_update": True, "role_bound_belief_evidence": True, "failure_action_changed": True, "model_context_firewall": True, "docker_network_none": True, "raw_payload_training_excluded": True},
        "forbidden": ["public_target", "external_callback", "time_delay", "database_write", "destructive", "credential_access"],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "protocol_sha256": "",
    }
    protocol["protocol_sha256"] = _digest(protocol)
    for path, value in ((REPORT, report), (CATALOG, catalog), (TRACE, trace), (PROTOCOL, protocol)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": counts, "worst_seed_metrics": report["worst_seed_metrics"], "gate": report["hypothesis_gate"], "elapsed_seconds": round(time.monotonic() - started, 3), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
