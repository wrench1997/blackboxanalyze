"""Run PG-146 against three pinned, loopback-only Docker labs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg146_public_lab_replay import SCHEMA_VERSION, TARGETS, collect_target  # noqa: E402


RESEARCH = ROOT / "research"
PROTOCOL = RESEARCH / "pg146_public_lab_replay_protocol_v1.json"
PROPOSAL = RESEARCH / "pg146_public_lab_replay_proposal_v1.json"
CATALOG = RESEARCH / "pg146_public_lab_replay_catalog_v1.json"
MODEL_DATASET = RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json"
TRACE = RESEARCH / "pg146_public_lab_replay_trace_v1.json"
REPORT = RESEARCH / "pg146_public_lab_replay_report_v1.json"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    target_reports: list[dict[str, Any]] = []
    for target in TARGETS:
        collected = collect_target(target)
        rows.extend(collected)
        target_reports.append(
            {
                "target_id": target.target_id,
                "lab": target.lab,
                "container": target.container,
                "image_digest": target.image_digest,
                "base_url": target.base_url,
                "row_count": len(collected),
                "get_count": sum(row["method"] == "GET" for row in collected),
                "post_count": sum(row["method"] == "POST" for row in collected),
                "fresh_reset_count": sum(bool(row["fresh_reset"]) for row in collected),
                "ready_count": sum(bool(row["readiness"].get("ready")) for row in collected),
                "unknown_oracle_count": sum(row["oracle"]["availability"] == "unknown_oracle" for row in collected),
            }
        )

    get_count = sum(row["method"] == "GET" for row in rows)
    post_count = sum(row["method"] == "POST" for row in rows)
    body_hashes = [row["response"]["projection"]["body_sha256"] for row in rows]
    evidence_hashes = [row["evidence_hash"] for row in rows]
    all_local = all(str(target.base_url).startswith("http://127.0.0.1:") for target in TARGETS)
    all_unknown = all(row["oracle"]["availability"] == "unknown_oracle" for row in rows)
    no_raw = all(
        not row["model_projection"]["raw_request_body_in_model"]
        and not row["model_projection"]["raw_response_body_in_model"]
        for row in rows
    )
    hard_checks = {
        "three_pinned_targets": len(TARGETS) == 3,
        "six_rows": len(rows) == 6,
        "get_post_balanced": get_count == post_count == 3,
        # A transport error is an environment failure, not a neutral model
        # observation.  Do not report a green replay gate when one target
        # surface never became ready.
        "all_target_surfaces_ready": all(bool(row["readiness"].get("ready")) for row in rows),
        "all_loopback": all_local,
        "all_fresh_reset": all(bool(row["fresh_reset"]) for row in rows),
        "all_evidence_hashes_present": len(evidence_hashes) == len(rows) and all(len(value) == 64 for value in evidence_hashes),
        "response_hashes_present": len(body_hashes) == len(rows) and all(len(value) == 64 for value in body_hashes),
        "html_or_response_projection_present": all("projection" in row["response"] for row in rows),
        "all_unknown_without_typed_oracle": all_unknown,
        "raw_bodies_excluded_from_model": no_raw,
    }
    hard_gates_passed = all(hard_checks.values())
    ready_count = sum(bool(row["readiness"].get("ready")) for row in rows)

    model_rows = [
        {
            "row_id": row["row_id"],
            "split": "evaluation_only",
            "tokens": row["model_projection"]["tokens"],
            "token_count": len(row["model_projection"]["tokens"]),
            "label": "unknown_oracle",
            "raw_request_body_in_model": False,
            "raw_response_body_in_model": False,
            "target_identity_in_model": False,
            "oracle_authority_in_model": False,
        }
        for row in rows
    ]
    catalog = {
        "schema_version": f"{SCHEMA_VERSION}-catalog",
        "protocol_id": "pg-pk-146-public-lab-replay-v1",
        "evaluation_only": True,
        "targets": target_reports,
        "rows": rows,
        "raw_request_bodies_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = _sha256_json({key: value for key, value in catalog.items() if key != "catalog_sha256"})
    model_dataset = {
        "schema_version": f"{SCHEMA_VERSION}-model-dataset",
        "evaluation_only": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "rows": model_rows,
        "labels_are_unknown_oracle": True,
        "raw_payloads_in_model": False,
        "dataset_sha256": "",
    }
    model_dataset["dataset_sha256"] = _sha256_json({key: value for key, value in model_dataset.items() if key != "dataset_sha256"})
    trace = {
        "schema_version": f"{SCHEMA_VERSION}-trace",
        "protocol_id": "pg-pk-146-public-lab-replay-v1",
        "status": "completed_pg146_public_lab_replay",
        "target_count": len(TARGETS),
        "step_count": len(rows),
        "get_step_count": get_count,
        "post_step_count": post_count,
        "fresh_reset_per_step": True,
        "failure_signature_present": all(bool(row["failure_signature"]) for row in rows),
        "typed_oracle_step_count": 0,
        "unknown_oracle_step_count": len(rows),
        "raw_request_response_saved": False,
        "long_term_memory_write": False,
        "trace_sha256": "",
    }
    trace["trace_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_sha256"})
    report = {
        "protocol_id": "pg-pk-146-public-lab-replay-v1",
        "schema_version": "pg146-public-lab-replay-report-v1",
        "status": "completed_pg146_public_lab_replay",
        "scope": {
            "labs": [target.lab for target in TARGETS],
            "pinned_images": True,
            "loopback_only": True,
            "safe_probe_scope": "baseline GET plus empty/invalid-form POST only",
        },
        "counts": {
            "target_count": len(TARGETS),
            "row_count": len(rows),
            "get_count": get_count,
            "post_count": post_count,
            "fresh_reset_count": sum(bool(row["fresh_reset"]) for row in rows),
            "ready_count": sum(bool(row["readiness"].get("ready")) for row in rows),
            "unknown_oracle_count": sum(row["oracle"]["availability"] == "unknown_oracle" for row in rows),
            "typed_oracle_count": 0,
        },
        "hard_checks": hard_checks,
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": False,
        "training_artifact_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "model_input_contract": {
            "raw_request_body_in_model": False,
            "raw_response_body_in_model": False,
            "target_identity_in_model": False,
            "oracle_authority_in_model": False,
            "external_payloads": False,
        },
        "diagnosis": {
            "experiment_vs_engineering": "real_loopback_surface_collection",
            "typed_oracle_available": False,
            "environment_failure": not hard_checks["all_target_surfaces_ready"],
            "ready_count": ready_count,
            "next_required_step": "lab-specific evaluator-only oracle and matched negative/positive replay before capability training",
        },
        "artifacts": {
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "catalog": str(CATALOG.relative_to(ROOT)),
            "model_dataset": str(MODEL_DATASET.relative_to(ROOT)),
            "trace": str(TRACE.relative_to(ROOT)),
        },
        "report_sha256": "",
    }
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    protocol = {
        "protocol_id": "pg-pk-146-public-lab-replay-v1",
        "schema_version": "pg146-public-lab-replay-protocol-v1",
        "objective": "在固定公开漏洞靶场的 loopback 实例上采集真实 GET/POST 因果表面基线。",
        "targets": [
            {
                "target_id": target.target_id,
                "lab": target.lab,
                "container": target.container,
                "image_digest": target.image_digest,
                "base_url": target.base_url,
            }
            for target in TARGETS
        ],
        "collection_contract": {
            "fresh_reset_before_each_step": True,
            "get_post_balanced": True,
            "response_projection_in_memory": True,
            "raw_request_response_persistence": False,
            "typed_oracle_required_for_positive": True,
            "unknown_oracle_cannot_train": True,
        },
        "source_repositories": {
            "juice_shop": "https://github.com/juice-shop/juice-shop",
            "webgoat": "https://github.com/WebGoat/WebGoat",
            "dvwa": "https://github.com/digininja/DVWA",
        },
        "run_report": str(REPORT.relative_to(ROOT)),
    }
    proposal = {
        "protocol_id": "pg-pk-146-public-lab-replay-v1",
        "proposal_id": "pg146-public-lab-replay-proposal-v1",
        "prediction": {
            "real_response_projection_present": True,
            "get_post_balanced": True,
            "typed_oracle_count": 0,
            "training_eligible": False,
        },
        "failure_rule": "若容器可达但无 typed evaluator，保留真实 response projection 作为 evaluation-only，禁止把 unknown 变成 negative/positive。",
        "next_experiment": "PG-147 为至少一个漏洞族接入 lab-side typed oracle、matched negative 和 fresh positive replay。",
    }
    _write(PROTOCOL, protocol)
    _write(PROPOSAL, proposal)
    _write(CATALOG, catalog)
    _write(MODEL_DATASET, model_dataset)
    _write(TRACE, trace)
    _write(REPORT, report)
    print(json.dumps({"status": report["status"], "target_count": len(TARGETS), "row_count": len(rows), "get_count": get_count, "post_count": post_count, "ready_count": report["counts"]["ready_count"], "typed_oracle_count": 0, "hard_gates_passed": hard_gates_passed, "training_eligible": False, "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
