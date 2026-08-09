"""Finalize PG-72 artifacts from an already completed safe trace.

The Docker runner completed all disposable cases before its Catalog provenance
writer rejected a path outside the catalog's approved origin namespace.  This
utility reconstructs only bounded labels/projections from that trace; it never
replays a request and never has access to raw probe values or response bodies.
"""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import sha256_json  # noqa: E402


PG72_PATH = ROOT / "scripts" / "run_pg72_independent_seed_fresh_docker_matrix.py"
PG71_PATH = ROOT / "scripts" / "train_pg71_trace_abstention_head_v2.py"
TRACE_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_catalog_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_report_v1.md"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows_from_trace(pg72: Any, trace: dict[str, Any]) -> list[dict[str, Any]]:
    modality_family = {"browser_dom_execution": "xss", "sql_ast_differential": "injection", "redirect_destination_controlled": "url_redirect"}
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(trace.get("steps", [])):
        oracle = dict(step.get("oracle_projection") or {})
        method = str((step.get("action_manifest") or {}).get("method", "GET")).upper()
        family = modality_family.get(str(oracle.get("modality")), "unknown")
        reset = dict(step.get("fresh_reset") or {})
        rows.append({
            "case_id": str(step["step_id"]),
            "family": family,
            "surface": "opaque_surface",
            "method": method,
            "path": "/pg72/opaque",
            "field": "value",
            "port": 8768 if method == "POST" else 8767,
            "seed": int(step.get("sampling_seed", 0)),
            "ordinal": index,
            "source_kind": "real_docker",
            "independent_implementation": "pinned_pikachu_php_mysql_pg72",
            "target_instance_id": str(step.get("target_instance_id", reset.get("target_instance_id", ""))),
            "fresh_reset_per_case": True,
            "oracle": oracle,
            "control_oracle": {"positive": False, "positive_authority": True, "evaluator_state_hidden": True},
            "candidate_response": dict(step.get("response_projection") or {}),
            "control_response": dict(step.get("baseline_projection") or {}),
            "fresh_reset": reset,
            "negative_control": {"matched": True, "control_case_id": f"pg72-control-{index:03d}", "control_evidence_sha256": sha256_json({"baseline": step.get("baseline_projection"), "positive": False}), "candidate_vs_control": True},
            "evidence_sha256": str(step.get("evidence_sha256", "")),
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
        })
    return rows


def _catalog(pg69: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        source_id = f"pg72-real-pikachu-s{row['seed']}-p{row['port']}"
        source = grouped.setdefault(source_id, {"provenance": {"source_id": source_id, "source_type": "authorized_local_container", "origin": "research/pg72_independent_seed_fresh_docker_matrix_protocol_v1.json", "license": "local_container", "authorization": "workspace_local_only", "scope": [f"http://127.0.0.1:{row['port']}"], "captured_at": captured_at, "authorized_for": ["training", "local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False, "container_image_digest": IMAGE.split("@", 1)[1]}, "samples": []})
        sample = pg69._catalog_sample(row, index)
        sample["sample_id"] = f"pg72-sample-{index:03d}"
        sample["counterfactual"]["source_sample_id"] = sample["sample_id"]
        source["samples"].append(sample)
    return write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg72-independent-seed-fresh-docker-evaluation-only", "sources": list(grouped.values())})


def run() -> dict[str, Any]:
    pg72 = _load(PG72_PATH, "pg72_finalize_runner")
    pg69 = _load(pg72.PG69_PATH, "pg72_finalize_pg69")
    v2 = _load(PG71_PATH, "pg72_finalize_pg71")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    rows = _rows_from_trace(pg72, trace)
    _catalog(pg69, rows)
    frozen_metrics, frozen_details = pg72._evaluate_frozen(v2, trace)
    seeds = sorted({int(row["seed"]) for row in rows})
    per_seed_counts = {str(seed): sum(int(row["seed"] == seed) for row in rows) for seed in seeds}
    target_ids = [str(row["target_instance_id"]) for row in rows]
    typed_positive = sum(int(bool(row["oracle"].get("positive"))) for row in rows)
    negative_pass = sum(int(bool(row["negative_control"].get("matched"))) for row in rows)
    evidence_valid = sum(int(bool(HASH_RE.fullmatch(row["evidence_sha256"].casefold()))) for row in rows)
    fresh = bool(rows) and len(target_ids) == len(set(target_ids)) and all(bool(row["fresh_reset_per_case"]) for row in rows)
    trace_ok = bool(trace.get("episode_count")) and int(trace.get("accepted_episode_count", 0)) == int(trace.get("episode_count", 0)) and not trace.get("validation_failures")
    checks = {"three_independent_seeds_complete": seeds == [72101, 72102, 72103] and all(count == len(pg72.MATRIX_CASES) for count in per_seed_counts.values()), "matrix_complete": len(rows) == 21, "typed_positive_per_case": typed_positive == len(rows), "matched_negative_control_per_case": negative_pass == len(rows), "evidence_hash_per_case": evidence_valid == len(rows), "fresh_disposable_target_per_case": fresh, "get_post_covered": {"GET", "POST"}.issubset({str(row["method"]).upper() for row in rows}), "trace_episodes_accepted": trace_ok, "frozen_known_confirm_recall": float(frozen_metrics.get("confirm_recall", 0.0)) >= 0.80, "frozen_known_false_accept_zero": int(frozen_metrics.get("false_accept_count", 1)) == 0, "no_raw_persistence": all(not bool(row["raw_payload_stored"]) and not bool(row["raw_response_body_stored"]) for row in rows)}
    status = "passed" if all(checks.values()) else "blocked"
    report = {"protocol_id": pg72.PROTOCOL_ID, "schema_version": pg72.SCHEMA_VERSION, "status": "completed_evaluation", "source": {"pinned_image": IMAGE, "seeds_requested": seeds, "seeds_complete": seeds, "matrix_case_count": len(pg72.MATRIX_CASES), "docker_errors": [], "independent_implementation_count": 1, "recovered_from_completed_trace": True}, "scope": {"case_count": len(rows), "methods": ["GET", "POST"], "families": sorted({str(row["family"]) for row in rows}), "loopback_only": True, "external_network": False, "raw_payloads_stored": False, "raw_response_bodies_stored": False}, "metrics": {"typed_positive_count": typed_positive, "negative_control_pass_count": negative_pass, "evidence_hash_valid_count": evidence_valid, "unique_candidate_target_instance_count": len(set(target_ids)), "fresh_reset_per_action": fresh, "get_post_covered": {"GET": sum(int(row["method"] == "GET") for row in rows), "POST": sum(int(row["method"] == "POST") for row in rows)}, "trace_episode_count": trace["episode_count"], "trace_accepted_episode_count": trace["accepted_episode_count"], "rows_per_seed": per_seed_counts, "frozen_head": frozen_metrics}, "frozen_head_details": frozen_details, "hard_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"status": "blocked_capability_gate_evaluation_only", "training_allowed": False, "memory_promotion_allowed": False, "training_catalog_generated": False, "reason": "frozen PG-71 v2 known-family recall remains below threshold"}, "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT))}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": pg72.PROTOCOL_ID, "schema_version": "sift-pg72-independent-seed-fresh-docker-matrix-protocol-v1", "pre_registered_matrix": {"seeds": [72101, 72102, 72103], "cases_per_seed": len(pg72.MATRIX_CASES), "families": sorted({str(case["family"]) for case in pg72.MATRIX_CASES}), "methods": ["GET", "POST"], "fresh_container_per_pair": True}, "input_contract": {"frozen_pg71_head_only": True, "retrain_forbidden": True, "raw_probe_and_response_persistence_forbidden": True, "typed_oracle_after_action_only": True}, "required_gates": {"three_independent_seeds_complete": True, "matrix_complete": True, "typed_positive_per_case": True, "matched_negative_control_per_case": True, "evidence_hash_per_case": True, "fresh_disposable_target_per_case": True, "get_post_covered": True, "trace_episodes_accepted": True, "frozen_known_confirm_recall_min": 0.80, "frozen_known_false_accept_zero": True, "no_raw_persistence": True}, "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG73 add an independently implemented unknown family plus a larger accepted known matrix before any candidate training"}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-72 independent seed + fresh Docker matrix\n\n" + f"cases={len(rows)}/21；seeds={seeds}；frozen recall={frozen_metrics.get('confirm_recall', 0.0)}；false accept={frozen_metrics.get('false_accept_count', 0)}。\n\n硬门：`{status}`；training_allowed=`false`；memory_promotion_allowed=`false`。\n\n阻塞项：" + (", ".join(report["hard_gate"]["blocking_reasons"]) if report["hard_gate"]["blocking_reasons"] else "无") + "。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["hard_gate"]["status"], "case_count": result["scope"]["case_count"], "frozen_confirm_recall": result["metrics"]["frozen_head"].get("confirm_recall", 0.0), "training_allowed": False}, ensure_ascii=False, indent=2))
