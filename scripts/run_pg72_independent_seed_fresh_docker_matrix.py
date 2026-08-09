"""PG-72: frozen Trace head replay on an independent fresh-Docker matrix.

This runner is deliberately an evaluation lane.  It does not retrain the
head, does not add rows to a training catalog, and does not write long-term
memory.  Each seed/case pair gets a disposable Pikachu container; the
authoritative PG-52 browser/SQL/redirect oracles run in memory and only
bounded response projections, typed-oracle fields and hashes are persisted.

The purpose is to distinguish a genuine representation/capability failure
from a one-seed artifact before any data or weights are promoted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step  # noqa: E402


PROTOCOL_ID = "pg-pk-72-independent-seed-fresh-docker-matrix-v1"
SCHEMA_VERSION = "sift-pg72-independent-seed-fresh-docker-matrix-v1"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
SEEDS = (72101, 72102, 72103)
PG52_PATH = ROOT / "scripts" / "run_pg52_authoritative_local_oracle.py"
PG69_PATH = ROOT / "scripts" / "run_pg69_per_action_reset_unseen_family.py"
PG71_PATH = ROOT / "scripts" / "train_pg71_trace_abstention_head_v2.py"
PG69_TRACE_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg71-trace-abstention-v2" / "trace_decision_head_v2.pt"
REPORT_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_report_v1.md"
HASH_RE = __import__("re").compile(r"^[0-9a-f]{64}$")

# Seven read-only, typed-oracle cases cover both transport methods and four
# known families.  The case id is salted per seed before reaching PG-52 so
# SQL canaries and evidence hashes differ without persisting their values.
MATRIX_CASES: tuple[dict[str, Any], ...] = (
    {"base_case_id": "reflected-get", "family": "xss", "surface": "xss_reflected_get", "method": "GET", "port": 8767, "path": "/vul/xss/xss_reflected_get.php", "field": "message", "mode": "reflected_get"},
    {"base_case_id": "dom-get", "family": "xss", "surface": "xss_dom_source", "method": "GET", "port": 8767, "path": "/vul/xss/xss_dom.php", "field": "text", "mode": "dom_get"},
    {"base_case_id": "reflected-post", "family": "xss", "surface": "xss_reflected_post", "method": "POST", "port": 8768, "path": "/vul/xss/xsspost/xss_reflected_post.php", "field": "message", "mode": "reflected_post"},
    {"base_case_id": "sql-string-get", "family": "injection", "surface": "sqli_str", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_str.php", "field": "name", "mode": "sql_string"},
    {"base_case_id": "sql-search-get", "family": "injection", "surface": "sqli_search", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_search.php", "field": "name", "mode": "sql_search"},
    {"base_case_id": "sql-boolean-get", "family": "injection", "surface": "sqli_blind_b", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_blind_b.php", "field": "name", "mode": "sql_boolean"},
    {"base_case_id": "redirect-get", "family": "url_redirect", "surface": "url_redirect", "method": "GET", "port": 8767, "path": "/vul/urlredirect/urlredirect.php", "field": "url", "mode": "redirect"},
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _case_for_seed(case: dict[str, Any], seed: int, ordinal: int) -> dict[str, Any]:
    result = dict(case)
    result["case_id"] = f"pg72-s{seed}-c{ordinal:02d}-{case['base_case_id']}"
    return result


def _run_case(pg52: Any, pg69: Any, loader: Any, case: dict[str, Any], seed: int, ordinal: int) -> dict[str, Any]:
    name = f"pg72-pikachu-s{seed}-c{ordinal:02d}"
    port = int(case["port"])
    started = False
    try:
        container_id = pg52._start(name, port)
        started = True
        if port == 8767:
            pg52._wait_application_surface(port, "/vul/sqli/sqli_str.php", b"what's your username")
        else:
            pg52._wait_application_surface(port, "/vul/xss/xsspost/post_login.php", b'name="username"')
        if case["family"] == "injection":
            pg52._prepare_mysql(name)
        base = pg52.GET_BASE if port == 8767 else pg52.POST_BASE
        marker = f"pg72-s{seed}-c{ordinal}-marker"
        if case["family"] == "xss":
            raw = pg52._browser_case(case, base, marker, container_id)
        elif case["family"] == "injection":
            raw = pg52._sql_case(case, base, name)
        else:
            raw = pg52._redirect_case(case, base, container_id)
        reset = pg52._fresh_reset(container_id, case["case_id"], _sha256_text(f"pg72-reset|{seed}|{ordinal}|{name}"))
        model = pg52._model_proposal(loader, case, raw)
        row = pg52._result_row(case, raw, reset, model)
        row.update({"seed": int(seed), "ordinal": int(ordinal), "source_kind": "real_docker", "independent_implementation": "pinned_pikachu_php_mysql_pg72", "target_instance_id": str(container_id)[:24], "fresh_reset_per_case": True, "port": port, "field": str(case.get("field", "value"))})
        return row
    finally:
        if started:
            pg52._stop(name)


def _safe_projection(value: Any) -> dict[str, Any]:
    # PG-52 projections are already bounded.  A second projection guard keeps
    # this runner from persisting an accidental raw field if the adapter grows.
    if not isinstance(value, dict):
        return {"projection_available": False}
    forbidden = {"body", "raw_body", "body_preview", "raw_probe", "password", "token", "cookie", "authorization"}
    cleaned = {str(key): child for key, child in value.items() if str(key).casefold() not in forbidden}
    return json.loads(json.dumps(cleaned, ensure_ascii=False, sort_keys=True))


def _trace_step(row: dict[str, Any], index: int, episode_id: str, parent: str | None) -> dict[str, Any]:
    method = str(row["method"]).upper()
    baseline = _safe_projection(row.get("control_response"))
    response = _safe_projection(row.get("candidate_response"))
    oracle = _safe_projection(row.get("oracle"))
    oracle.update({"negative_control_pair_id": f"pg72-control-{index:03d}", "evaluator_state_hidden": True})
    action: dict[str, Any] = {
        "method": method,
        "route_template_id": f"pg72-route-{index:03d}",
        "placement": "form" if method == "POST" else "query",
        "encoding_chain": ["identity"],
        "probe_ref": f"pg72-probe-{index:03d}",
        "probe_sha256": _sha256_text("pg72-abstract-safe-probe"),
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
    }
    if method == "POST":
        action["form_field_names"] = [str(row.get("field", "value"))]
    decision = "confirmed_positive" if bool(row.get("oracle", {}).get("positive")) else "confirmed_negative"
    reset = dict(row.get("fresh_reset") or {})
    step: dict[str, Any] = {
        "episode_id": episode_id,
        "step_id": f"pg72-step-{index:03d}",
        "parent_step_id": parent,
        "sampling_seed": int(row["seed"]),
        "target_instance_id": str(row.get("target_instance_id") or reset.get("target_instance_id") or f"pg72-target-{index:03d}"),
        "hypothesis": "bounded_surface_hypothesis",
        "belief_before": {"unknown_surface": 1.0},
        "action_manifest": action,
        "baseline_projection": baseline,
        "response_projection": response,
        "oracle_projection": oracle,
        "belief_after": {"unknown_surface": 1.0},
        "decision": decision,
        "next_action": "stop_confirmed" if decision == "confirmed_positive" else "continue_probe",
        "fresh_reset": reset,
        "evidence_sha256": "",
        "dataset_stage": "evaluation_only",
        "online_weight_update": False,
        "long_term_memory_write": False,
    }
    step["evidence_sha256"] = sha256_json({"action": action, "baseline": baseline, "response": response, "oracle": oracle, "reset": reset, "decision": decision})
    echo_body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action")}
    step["echo"] = {"sha256": sha256_json(echo_body)}
    return step


def _build_trace(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["seed"])].append((index, row))
    episodes: list[dict[str, Any]] = []
    normalized_steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed, items in sorted(grouped.items()):
        episode_id = f"pg72-episode-s{seed}"
        steps: list[dict[str, Any]] = []
        parent: str | None = None
        for index, row in items:
            raw_step = _trace_step(row, index, episode_id, parent)
            try:
                normalized = validate_trace_step(raw_step)
            except ValueError as exc:
                failures.append({"step_id": raw_step["step_id"], "error_type": type(exc).__name__})
                normalized = raw_step
            steps.append(normalized)
            normalized_steps.append(normalized)
            parent = normalized["step_id"]
        failed_ids = {str(item["step_id"]) for item in failures}
        validation = evaluate_episode(steps) if not any(str(step["step_id"]) in failed_ids for step in steps) else {"status": "trace_only", "reasons": ["step_validation_failure"]}
        episodes.append({"episode_id": episode_id, "seed": seed, "steps": steps, "validation": validation})
    return {"schema_version": "sift-pg72-independent-seed-trace-v1", "protocol_id": PROTOCOL_ID, "evaluation_only": True, "training_eligible": False, "steps": normalized_steps, "episodes": episodes, "episode_count": len(episodes), "accepted_episode_count": sum(int(item["validation"].get("status") == "accepted_evaluation") for item in episodes), "validation_failures": failures, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}, failures


def _load_frozen_head(v2: Any) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("PG-71 v2 checkpoint is missing")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    base = v2._load_pg70()
    model = base.TraceDecisionHead().cpu()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    steps = [dict(step) for step in json.loads(PG69_TRACE_PATH.read_text(encoding="utf-8")).get("steps", [])]
    train_rows, _, _ = v2._build_examples(steps)
    reference, mean, std = v2._normalise(train_rows, train_rows)
    return model, mean, std, reference, torch.device("cpu")


def _score_head(v2: Any, model: Any, mean: torch.Tensor, std: torch.Tensor, reference: torch.Tensor, step: dict[str, Any], role: str) -> dict[str, Any]:
    pair = v2._pair_features(step) if role == "candidate" else [0.0] * int(v2.FEATURE_DIM)
    values = ((torch.tensor([pair], dtype=torch.float32) - mean) / std).clamp(-float(v2.CLIP), float(v2.CLIP))
    with torch.inference_mode():
        probability = torch.softmax(model(values), dim=-1)[0]
    confidence, predicted = torch.max(probability, dim=0)
    distance = float(torch.cdist(values, reference).min().item()) if len(reference) else float("inf")
    raw = v2.CLASSES[int(predicted)]
    decision = "abstain" if distance >= float(v2.OOD_DISTANCE_THRESHOLD) or float(confidence) < float(v2.CONFIDENCE_THRESHOLD) else raw
    expected = "confirm" if role == "candidate" else "reject"
    return {"step_id": step["step_id"], "seed": step["sampling_seed"], "role": role, "expected": expected, "raw_prediction": raw, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6), "feature_l2": round(float(torch.linalg.vector_norm(torch.tensor(pair)).item()), 6)}


def _evaluate_frozen(v2: Any, trace: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, mean, std, reference, _ = _load_frozen_head(v2)
    details: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        details.append(_score_head(v2, model, mean, std, reference, step, "candidate"))
        details.append(_score_head(v2, model, mean, std, reference, step, "matched_control"))
    candidates = [item for item in details if item["role"] == "candidate"]
    controls = [item for item in details if item["role"] == "matched_control"]
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted({int(item["seed"]) for item in candidates}):
        seed_candidates = [item for item in candidates if int(item["seed"]) == seed]
        seed_controls = [item for item in controls if int(item["seed"]) == seed]
        per_seed[str(seed)] = {"candidate_count": len(seed_candidates), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in seed_candidates) / max(len(seed_candidates), 1), 6), "false_accept_count": sum(int(item["decision"] == "confirm") for item in seed_controls), "abstain_count": sum(int(item["decision"] == "abstain") for item in seed_candidates + seed_controls)}
    metrics = {"candidate_count": len(candidates), "control_count": len(controls), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in candidates) / max(len(candidates), 1), 6), "false_accept_count": sum(int(item["decision"] == "confirm") for item in controls), "candidate_abstain_count": sum(int(item["decision"] == "abstain") for item in candidates), "control_abstain_count": sum(int(item["decision"] == "abstain") for item in controls), "per_seed": per_seed}
    return metrics, details


def _build_catalog(pg69: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sources: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        source_id = f"pg72-real-pikachu-s{row['seed']}-p{row['port']}"
        source = sources.setdefault(source_id, {"provenance": {"source_id": source_id, "source_type": "authorized_local_container", "origin": "research/pg72_independent_seed_fresh_docker_matrix_protocol_v1.json", "license": "local_container", "authorization": "workspace_local_only", "scope": [f"http://127.0.0.1:{row['port']}"], "captured_at": captured_at, "authorized_for": ["local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False, "container_image_digest": IMAGE.split("@", 1)[1]}, "samples": []})
        sample = pg69._catalog_sample(row, index)
        sample["sample_id"] = f"pg72-sample-{index:03d}"
        sample["counterfactual"]["source_sample_id"] = sample["sample_id"]
        source["samples"].append(sample)
    return write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg72-independent-seed-fresh-docker-evaluation-only", "sources": list(sources.values())})


def run(*, skip_docker: bool = False, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    pg52 = _load_module(PG52_PATH, "pg72_pg52_runtime")
    pg69 = _load_module(PG69_PATH, "pg72_pg69_runtime")
    v2 = _load_module(PG71_PATH, "pg72_pg71_v2")
    loader = pg52._model_loader()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    rng = random.Random(7200)
    for seed in seeds:
        cases = list(MATRIX_CASES)
        random.Random(int(seed)).shuffle(cases)
        for ordinal, base_case in enumerate(cases):
            case = _case_for_seed(base_case, int(seed), ordinal)
            try:
                if skip_docker:
                    continue
                rows.append(_run_case(pg52, pg69, loader, case, int(seed), ordinal))
            except Exception as exc:  # retain a bounded failure ledger; never stop cleanup
                errors.append({"seed": int(seed), "case_id": str(case["case_id"]), "error_type": type(exc).__name__})
    trace, validation_failures = _build_trace(rows)
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        _build_catalog(pg69, rows)
    frozen_metrics, frozen_details = _evaluate_frozen(v2, trace) if rows else ({"candidate_count": 0, "control_count": 0, "confirm_recall": 0.0, "false_accept_count": 0, "per_seed": {}}, [])
    complete_seeds = {int(seed) for seed in seeds if sum(int(row["seed"] == seed) for row in rows) == len(MATRIX_CASES)}
    candidate_instances = [str(row.get("target_instance_id", "")) for row in rows]
    fresh_reset = bool(rows) and len(candidate_instances) == len(set(candidate_instances)) and all(bool(row.get("fresh_reset_per_case")) for row in rows)
    typed_positive = sum(int(bool(row.get("oracle", {}).get("positive"))) for row in rows)
    negative_controls = sum(int(bool(row.get("negative_control", {}).get("matched"))) for row in rows)
    evidence_hashes = sum(int(bool(HASH_RE.fullmatch(str(row.get("evidence_sha256", "")).casefold()))) for row in rows)
    trace_accepted = trace["accepted_episode_count"] == trace["episode_count"] and not validation_failures
    checks = {"three_independent_seeds_complete": len(complete_seeds) >= 3, "matrix_complete": len(rows) == len(seeds) * len(MATRIX_CASES), "typed_positive_per_case": typed_positive == len(rows) and len(rows) > 0, "matched_negative_control_per_case": negative_controls == len(rows) and len(rows) > 0, "evidence_hash_per_case": evidence_hashes == len(rows) and len(rows) > 0, "fresh_disposable_target_per_case": fresh_reset, "get_post_covered": {"GET", "POST"}.issubset({str(row["method"]).upper() for row in rows}), "trace_episodes_accepted": trace_accepted, "frozen_known_confirm_recall": float(frozen_metrics.get("confirm_recall", 0.0)) >= 0.80, "frozen_known_false_accept_zero": int(frozen_metrics.get("false_accept_count", 1)) == 0, "no_raw_persistence": all(not bool(row.get("raw_payload_stored")) and not bool(row.get("raw_response_body_stored")) for row in rows)}
    hard_status = "passed" if all(checks.values()) else "blocked"
    report = {"protocol_id": PROTOCOL_ID, "schema_version": SCHEMA_VERSION, "status": "completed_evaluation" if rows else "blocked_no_rows", "source": {"pinned_image": IMAGE, "seeds_requested": list(seeds), "seeds_complete": sorted(complete_seeds), "matrix_case_count": len(MATRIX_CASES), "docker_errors": errors, "independent_implementation_count": 1}, "scope": {"case_count": len(rows), "methods": sorted({str(row["method"]).upper() for row in rows}), "families": sorted({str(row["family"]) for row in rows}), "loopback_only": True, "external_network": False, "raw_payloads_stored": False, "raw_response_bodies_stored": False}, "metrics": {"typed_positive_count": typed_positive, "negative_control_pass_count": negative_controls, "evidence_hash_valid_count": evidence_hashes, "unique_candidate_target_instance_count": len(set(candidate_instances)), "fresh_reset_per_action": fresh_reset, "get_post_covered": {"GET": sum(int(row["method"] == "GET") for row in rows), "POST": sum(int(row["method"] == "POST") for row in rows)}, "trace_episode_count": trace["episode_count"], "trace_accepted_episode_count": trace["accepted_episode_count"], "frozen_head": frozen_metrics}, "frozen_head_details": frozen_details, "hard_gate": {"status": hard_status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"status": "blocked_evaluation_only" if hard_status != "passed" else "hard_gate_passed_evaluation_only_no_promotion", "training_allowed": False, "memory_promotion_allowed": False, "training_catalog_generated": False, "reason": "PG-72 is a frozen-head cross-seed replay; no data or weights are promoted"}, "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT)) if rows else None}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "sift-pg72-independent-seed-fresh-docker-matrix-protocol-v1", "pre_registered_matrix": {"seeds": list(seeds), "cases_per_seed": len(MATRIX_CASES), "families": sorted({str(case["family"]) for case in MATRIX_CASES}), "methods": ["GET", "POST"], "fresh_container_per_pair": True}, "input_contract": {"frozen_pg71_head_only": True, "retrain_forbidden": True, "raw_probe_and_response_persistence_forbidden": True, "typed_oracle_after_action_only": True}, "required_gates": {"three_independent_seeds_complete": True, "matrix_complete": True, "typed_positive_per_case": True, "matched_negative_control_per_case": True, "evidence_hash_per_case": True, "fresh_disposable_target_per_case": True, "get_post_covered": True, "trace_episodes_accepted": True, "frozen_known_confirm_recall_min": 0.80, "frozen_known_false_accept_zero": True, "no_raw_persistence": True}, "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG73 add an independently implemented unknown family plus a larger accepted known matrix before any candidate training"}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-72 independent seed + fresh Docker matrix\n\n" + f"cases={len(rows)}/{len(seeds) * len(MATRIX_CASES)}；seeds={sorted(complete_seeds)}；frozen recall={frozen_metrics.get('confirm_recall', 0.0)}；false accept={frozen_metrics.get('false_accept_count', 0)}。\n\n硬门：`{hard_status}`；training_allowed=`false`；memory_promotion_allowed=`false`。\n\n阻塞项：" + (", ".join(report["hard_gate"]["blocking_reasons"]) if report["hard_gate"]["blocking_reasons"] else "无") + "。\n", encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--seeds", default=",".join(str(item) for item in SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(item.strip()) for item in str(args.seeds).split(",") if item.strip())
    report = run(skip_docker=bool(args.skip_docker), seeds=seeds)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["hard_gate"]["status"], "case_count": report["scope"]["case_count"], "seeds_complete": report["source"]["seeds_complete"], "frozen_confirm_recall": report["metrics"]["frozen_head"].get("confirm_recall", 0.0), "training_allowed": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
